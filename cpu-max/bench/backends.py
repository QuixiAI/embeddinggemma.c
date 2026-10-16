"""Benchmark backends for EmbeddingGemma 2 on CPU.

Every backend consumes the same manifest items (bench/fixtures/manifest.json) and
returns (embedding[768], timings_ms) where timings_ms has at least
  total      end-to-end wall time of one request
  preprocess media/text preprocessing (processor, or server-side front ends)
  inference  model execution
Backends:
  st      official Sentence Transformers + PyTorch FP32 (the reference)
  onnx    onnx-community/embeddinggemma-2-ONNX "quantized" (int8) via onnxruntime CPU EP
  native  an embeddinggemma.c-compatible HTTP server (upstream QuixiAI build or our fork)
"""
from __future__ import annotations

import base64
import glob
import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
import http.client
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "bench" / "fixtures"
HF = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
SR = 16000


def snapshot(repo_dir: str) -> Path:
    snaps = sorted(glob.glob(str(HF / repo_dir / "snapshots" / "*")))
    if not snaps:
        raise FileNotFoundError(repo_dir)
    return Path(snaps[-1])


def load_manifest() -> dict:
    return json.loads((FIX / "manifest.json").read_text())


def load_wav(rel: str) -> np.ndarray:
    import soundfile as sf

    a, sr = sf.read(FIX / rel, dtype="float32")
    assert sr == SR and a.ndim == 1
    return a


def load_img(rel: str) -> Image.Image:
    return Image.open(FIX / rel).convert("RGB")


def parts_of(item: dict, variant: int = 0) -> list[dict]:
    """Normalize any manifest item (or pool variant) into an ordered list of parts."""
    src = item["pool"][variant] if variant else item
    mod = item["modality"]
    if mod == "text":
        return [dict(type="text", text=src["text"])]
    if mod == "image":
        return [dict(type="image", image=src["image"], vision_tokens=item["vision_tokens"])]
    if mod == "audio":
        return [dict(type="audio", audio=src["audio"])]
    if mod == "video":
        return [dict(type="video", frames=src["frames"], vision_tokens=item["vision_tokens"], fps=item["fps"])]
    return src["parts"]


def _now() -> float:
    return time.perf_counter() * 1000.0


# ----------------------------------------------------------------------------- HF input building
PLACEHOLDER = {"image": "<|image|>", "audio": "<|audio|>", "video": "<|video|>"}


def hf_inputs(parts: list[dict]):
    """Text with explicit placeholders + media lists, in the HF processor layout."""
    text, images, audios, videos, budgets = "", [], [], [], set()
    for p in parts:
        if p["type"] == "text":
            text += p["text"]
        else:
            text += PLACEHOLDER[p["type"]]
            if p["type"] == "image":
                images.append(load_img(p["image"]))
                budgets.add(("image", p.get("vision_tokens", 280)))
            elif p["type"] == "audio":
                audios.append(load_wav(p["audio"]))
            else:
                videos.append([load_img(f) for f in p["frames"]])
                budgets.add(("video", p.get("vision_tokens", 140)))
    return text, images, audios, videos, budgets


def processing_kwargs(budgets) -> dict:
    kw: dict = {"video": {"do_sample_frames": False}}
    for kind, b in budgets:
        if kind == "image":
            kw.setdefault("image", {})["max_soft_tokens"] = b
        else:
            kw["video"]["max_soft_tokens"] = b
    return kw


# ----------------------------------------------------------------------------- Sentence Transformers FP32
class STBackend:
    name = "st_fp32"

    def __init__(self, threads: int):
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        from sentence_transformers import SentenceTransformer

        t = _now()
        self.model = SentenceTransformer("google/embeddinggemma-2", device="cpu",
                                         model_kwargs={"dtype": torch.float32})
        self.model.eval()
        self.load_ms = _now() - t
        self.torch = torch
        self.versions = dict(torch=torch.__version__, sentence_transformers=__import__("sentence_transformers").__version__,
                             transformers=__import__("transformers").__version__,
                             mkldnn=torch.backends.mkldnn.is_available(), cpu_capability=torch.backends.cpu.get_cpu_capability())

    def _input(self, parts):
        """Text-only -> str; one media part -> that modality alone; otherwise a
        multimodal dict whose text carries explicit placeholders (fixed order)."""
        text, images, audios, videos, budgets = hf_inputs(parts)
        kw = processing_kwargs(budgets)
        if len(parts) == 1:
            kind = parts[0]["type"]
            if kind == "text":
                return text, {}
            if kind == "image":
                return images[0], kw
            if kind == "audio":
                return {"audio": {"array": audios[0], "sampling_rate": SR}}, kw
            return {"video": videos[0]}, kw
        x: dict = {"text": text}
        if images:
            x["image"] = images if len(images) > 1 else images[0]
        if audios:
            x["audio"] = ([{"array": a, "sampling_rate": SR} for a in audios] if len(audios) > 1
                          else {"array": audios[0], "sampling_rate": SR})
        if videos:
            x["video"] = videos[0] if len(videos) == 1 else videos
        return x, kw

    def prepare(self, parts):
        return self._input(parts)

    def run(self, prepared, batch: int = 1):
        x, kw = prepared
        torch = self.torch
        t0 = _now()
        feats = self.model.preprocess([x] * batch, processing_kwargs=kw) if kw else self.model.preprocess([x] * batch)
        t1 = _now()
        with torch.inference_mode():
            out = self.model.forward(feats)
        t2 = _now()
        emb = out["sentence_embedding"].float().numpy()
        n_tokens = int(feats["attention_mask"][0].sum())
        return emb, dict(total=t2 - t0, preprocess=t1 - t0, inference=t2 - t1, tokens=n_tokens)

    def close(self):
        pass


# ----------------------------------------------------------------------------- ONNX Runtime int8
class ONNXBackend:
    name = "onnx_q8"

    def __init__(self, threads: int, variant: str = "quantized"):
        import onnxruntime as ort
        from transformers import AutoProcessor

        snap = snapshot("models--onnx-community--embeddinggemma-2-ONNX")
        t = _now()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        prov = ["CPUExecutionProvider"]
        self.text = ort.InferenceSession(str(snap / "onnx" / f"model_{variant}.onnx"), so, providers=prov)
        self.vision = ort.InferenceSession(str(snap / "onnx" / f"vision_encoder_{variant}.onnx"), so, providers=prov)
        self.audio = ort.InferenceSession(str(snap / "onnx" / f"audio_encoder_{variant}.onnx"), so, providers=prov)
        self.processor = AutoProcessor.from_pretrained(str(snap))
        self.load_ms = _now() - t
        self.versions = dict(onnxruntime=ort.__version__, providers=self.text.get_providers(), variant=variant,
                             repo_revision=snap.name)

    def prepare(self, parts):
        return hf_inputs(parts)

    def run(self, prepared, batch: int = 1):
        text, images, audios, videos, budgets = prepared
        kw = processing_kwargs(budgets)
        t0 = _now()
        call = dict(text=[text] * batch, return_tensors="np", padding=True)
        if images:
            call["images"] = [images] * batch
            call["images_kwargs"] = kw.get("image", {})
        if audios:
            call["audio"] = audios * batch
        if videos:
            call["videos"] = videos * batch
            call["videos_kwargs"] = kw["video"]
        f = self.processor(**call)
        t1 = _now()
        empty = np.zeros((0, 512), np.float32)
        img = vid = aud = empty
        if "pixel_values" in f:
            img = self.vision.run(None, {"pixel_values": f["pixel_values"].astype(np.float32),
                                         "pixel_position_ids": f["image_position_ids"].astype(np.int64)})[0]
        if "pixel_values_videos" in f:
            vid = self.vision.run(None, {"pixel_values": f["pixel_values_videos"].astype(np.float32),
                                         "pixel_position_ids": f["video_position_ids"].astype(np.int64)})[0]
        if "input_features" in f:
            aud = self.audio.run(None, {"input_features": f["input_features"].astype(np.float32),
                                        "input_features_mask": f["input_features_mask"].astype(bool)})[0]
        t2 = _now()
        out = self.text.run(["sentence_embedding"], {"input_ids": f["input_ids"].astype(np.int64),
                                                     "attention_mask": f["attention_mask"].astype(np.int64),
                                                     "image_features": img, "video_features": vid,
                                                     "audio_features": aud})[0]
        t3 = _now()
        out = out / np.linalg.norm(out, axis=-1, keepdims=True)
        return out, dict(total=t3 - t0, preprocess=t1 - t0, inference=t3 - t1, towers=t2 - t1, text=t3 - t2,
                         tokens=int(f["attention_mask"][0].sum()))

    def close(self):
        pass


# ----------------------------------------------------------------------------- native HTTP server
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def b64file(rel: str) -> str:
    return base64.b64encode((FIX / rel).read_bytes()).decode()


STAGE = re.compile(r"^media (\S+)\s+([0-9.]+) ms")


class NativeBackend:
    """Spawns the server (all caches off) and posts content objects."""

    def __init__(self, binary: str, threads: int, cpus: str | None, name: str, video_as_images: bool = False,
                 extra_args: list[str] | None = None, env: dict | None = None, timing: bool = True):
        self.name = name
        self.video_as_images = video_as_images
        self.port = _free_port()
        gg = ROOT / "models" / "gguf"
        cmd = [binary, "--bind", "127.0.0.1", "--port", str(self.port), "--threads", str(threads),
               "--model", str(gg / "embeddinggemma-2-Q8_0.gguf"), "--mmproj", str(gg / "mmproj-embeddinggemma-2-Q8_0.gguf"),
               "--cache-entries", "0", "--response-cache-mb", "0", "--media-cache-mb", "0",
               "--keepalive-timeout-ms", "60000"] + (extra_args or [])
        if cpus:
            cmd = ["taskset", "-c", cpus] + cmd
        e = dict(os.environ, CUDA_VISIBLE_DEVICES="", **(env or {}))
        if timing:
            e["EI_MEDIA_TIMING"] = "1"
        self.stderr_lines: list[str] = []
        t = _now()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=e, text=True)
        import atexit

        atexit.register(self.close)  # never leave a server running if the client dies
        self._reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._reader.start()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited: {''.join(self.stderr_lines[-20:])}")
            try:
                c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
                c.request("GET", "/healthz")
                if c.getresponse().status == 200:
                    break
            except OSError:
                time.sleep(0.05)
        self.load_ms = _now() - t
        self.conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
        self.versions = dict(binary=binary, cmd=cmd, git=_git_rev(Path(binary).resolve().parent.parent))
        self.pid = self.proc.pid

    def _read_stderr(self):
        for line in self.proc.stderr:
            self.stderr_lines.append(line)

    def body(self, parts) -> bytes:
        content = []
        for p in parts:
            if p["type"] == "text":
                content.append({"type": "text", "text": p["text"]})
            elif p["type"] == "image":
                content.append({"type": "image", "data": b64file(p["image"]), "vision_tokens": p.get("vision_tokens", 280)})
            elif p["type"] == "audio":
                content.append({"type": "audio", "data": b64file(p["audio"])})
            elif self.video_as_images:
                content += [{"type": "image", "data": b64file(f), "vision_tokens": p.get("vision_tokens", 140)}
                            for f in p["frames"]]
            else:
                content.append({"type": "video", "frames": [b64file(f) for f in p["frames"]], "fps": p.get("fps", 1),
                                "vision_tokens": p.get("vision_tokens", 140)})
        if len(parts) == 1 and parts[0]["type"] == "text":
            payload = {"model": "embeddinggemma-2", "input": [parts[0]["text"]]}
        else:
            payload = {"model": "embeddinggemma-2", "input": {"content": content}}
        return json.dumps(payload).encode()

    def prepare(self, parts):
        t = _now()
        b = self.body(parts)
        return b, _now() - t

    def post(self, body: bytes, conn=None):
        conn = conn or self.conn
        try:
            conn.request("POST", "/v1/embeddings", body=body, headers={"Content-Type": "application/json"})
            r = conn.getresponse()
        except (BrokenPipeError, ConnectionResetError, http.client.RemoteDisconnected):
            # The server closes idle keep-alive connections beyond its slot limit;
            # reconnect once (the request was not processed).
            conn.close()
            conn.connect()
            conn.request("POST", "/v1/embeddings", body=body, headers={"Content-Type": "application/json"})
            r = conn.getresponse()
        data = r.read()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {data[:300]!r}")
        return json.loads(data)

    def run(self, prepared, batch: int = 1):
        body, encode_ms = prepared
        n0 = len(self.stderr_lines)
        t0 = _now()
        resp = self.post(body)
        t1 = _now()
        time.sleep(0.002)  # let the stderr reader catch the stage lines
        stages: dict[str, float] = {}
        for line in self.stderr_lines[n0:]:
            m = STAGE.match(line)
            if m:
                stages[m.group(1)] = stages.get(m.group(1), 0.0) + float(m.group(2))
        emb = np.array(resp["data"][0]["embedding"], dtype=np.float32)[None]
        pre = sum(v for k, v in stages.items() if k in ("base64", "decode", "resize", "wav", "logmel"))
        tower = sum(v for k, v in stages.items() if k in ("vision", "audio"))
        timings = dict(total=t1 - t0, client_encode=encode_ms, preprocess=pre, towers=tower,
                       text=stages.get("finish", float("nan")), tokens=resp["usage"]["prompt_tokens"],
                       stages=stages)
        timings["inference"] = tower + stages.get("finish", 0.0) if stages else float("nan")
        return emb, timings

    def rss_peak_mb(self) -> float:
        for line in Path(f"/proc/{self.pid}/status").read_text().splitlines():
            if line.startswith("VmHWM"):
                return int(line.split()[1]) / 1024
        return float("nan")

    def cpu_seconds(self) -> float:
        f = Path(f"/proc/{self.pid}/stat").read_text().rsplit(")", 1)[1].split()
        tick = os.sysconf("SC_CLK_TCK")
        return (int(f[11]) + int(f[12])) / tick

    def close(self):
        if self.proc.poll() is not None:
            return
        try:
            self.conn.close()
        except Exception:
            pass
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _git_rev(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip() + (
            "+dirty" if subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                                       capture_output=True, text=True).stdout.strip() else "")
    except Exception:
        return None
