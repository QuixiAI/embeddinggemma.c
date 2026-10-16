#!/usr/bin/env python3
"""Detect CPU/memory/toolchain capabilities and write reports/hardware.json.

Every capability is read from the running system (/proc, /sys, compiler
probes); nothing is assumed. Run: python3 scripts/hw_detect.py [out.json]
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sh(cmd: list[str] | str, timeout: float = 20) -> str:
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"<error: {exc}>"


def read(path: str | Path, default: str | None = None) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def parse_cpulist(text: str | None) -> list[int]:
    out: list[int] = []
    if not text:
        return out
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def cpuinfo() -> dict:
    flags: set[str] = set()
    model = vendor = None
    family = model_id = stepping = microcode = None
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if ":" not in line:
            continue
        k, v = (s.strip() for s in line.split(":", 1))
        if k == "flags":
            flags |= set(v.split())
        elif k == "model name" and model is None:
            model = v
        elif k == "vendor_id" and vendor is None:
            vendor = v
        elif k == "cpu family" and family is None:
            family = int(v)
        elif k == "model" and model_id is None:
            model_id = int(v)
        elif k == "stepping" and stepping is None:
            stepping = v
        elif k == "microcode" and microcode is None:
            microcode = v
    return dict(flags=flags, model=model, vendor=vendor, family=family, model_id=model_id,
                stepping=stepping, microcode=microcode)


def caches() -> list[dict]:
    seen = {}
    base = Path("/sys/devices/system/cpu")
    for cpu_dir in sorted(base.glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        for idx in sorted(cpu_dir.glob("cache/index*")):
            shared = read(idx / "shared_cpu_list")
            key = (read(idx / "level"), read(idx / "type"), shared)
            if key in seen:
                continue
            seen[key] = dict(level=int(read(idx / "level", "0")), type=read(idx / "type"),
                             size=read(idx / "size"), shared_cpu_list=shared,
                             line_size=int(read(idx / "coherency_line_size", "0") or 0))
    # Summarize: per (level, type, size) count of instances.
    summary: dict[tuple, dict] = {}
    for v in seen.values():
        k = (v["level"], v["type"], v["size"])
        s = summary.setdefault(k, dict(level=v["level"], type=v["type"], size=v["size"],
                                       line_size=v["line_size"], instances=0, shared_cpu_lists=[]))
        s["instances"] += 1
        s["shared_cpu_lists"].append(v["shared_cpu_list"])
    return sorted(summary.values(), key=lambda s: (s["level"], s["type"], s["size"]))


def topology() -> dict:
    base = Path("/sys/devices/system/cpu")
    cpus = parse_cpulist(read(base / "online"))
    cores: dict[tuple, list[int]] = {}
    per_cpu = []
    for c in cpus:
        d = base / f"cpu{c}"
        pkg = int(read(d / "topology/physical_package_id", "0"))
        core = int(read(d / "topology/core_id", "0"))
        cores.setdefault((pkg, core), []).append(c)
        per_cpu.append(dict(cpu=c, package=pkg, core_id=core,
                            max_khz=int(read(d / "cpufreq/cpuinfo_max_freq", "0") or 0),
                            min_khz=int(read(d / "cpufreq/cpuinfo_min_freq", "0") or 0),
                            governor=read(d / "cpufreq/scaling_governor"),
                            epp=read(d / "cpufreq/energy_performance_preference")))
    hybrid = {}
    for kind in ("cpu_core", "cpu_atom"):
        lst = read(f"/sys/devices/{kind}/cpus")
        if lst is not None:
            hybrid[kind] = parse_cpulist(lst)
    return dict(logical_cpus=len(cpus), physical_cores=len(cores),
                packages=len({p for p, _ in cores}), smt_threads_per_core_max=max(len(v) for v in cores.values()),
                core_groups=[dict(package=p, core_id=c, cpus=v) for (p, c), v in sorted(cores.items())],
                hybrid=hybrid, per_cpu=per_cpu)


def numa() -> list[dict]:
    nodes = []
    for n in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        mem = read(n / "meminfo", "")
        total = re.search(r"MemTotal:\s+(\d+) kB", mem or "")
        nodes.append(dict(node=int(n.name[4:]), cpus=parse_cpulist(read(n / "cpulist")),
                          mem_total_kib=int(total.group(1)) if total else None))
    return nodes


def meminfo() -> dict:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        if k in ("MemTotal", "MemAvailable", "MemFree", "SwapTotal", "SwapFree", "Hugepagesize",
                 "HugePages_Total"):
            out[k] = v.strip()
    out["transparent_hugepage"] = read("/sys/kernel/mm/transparent_hugepage/enabled")
    return out


def compiler_probe(cc: str) -> dict:
    """Ask the compiler what -march=native enables and whether ISA intrinsics compile."""
    path = shutil.which(cc)
    if not path:
        return dict(available=False)
    info = dict(available=True, path=path, version=sh([cc, "--version"]).splitlines()[0])
    macros = sh(f"echo | {cc} -march=native -dM -E -")
    info["native_isa_macros"] = sorted(m for m in re.findall(r"#define (__(?:AVX|FMA|F16C|SSE|AMX|AVXVNNI|BMI|VNNI|GFNI|VAES|SHA)[A-Z0-9_]*__)", macros))
    info["native_march"] = sh(f"{cc} -march=native -Q --help=target 2>/dev/null | grep -E '^\\s+-march=' | head -1").strip()
    tests = {
        "avx2_fma": ("avx2,fma", "#include <immintrin.h>\nint main(){__m256 a=_mm256_set1_ps(1);a=_mm256_fmadd_ps(a,a,a);return (int)_mm256_cvtss_f32(a);}"),
        "avx_vnni": ("avx2,avxvnni", "#include <immintrin.h>\nint main(){__m256i a=_mm256_set1_epi8(1);a=_mm256_dpbusd_avx_epi32(a,a,a);return _mm256_extract_epi32(a,0);}"),
        "avx512_vnni": ("avx512f,avx512vnni", "#include <immintrin.h>\nint main(){__m512i a=_mm512_set1_epi8(1);a=_mm512_dpbusd_epi32(a,a,a);return _mm512_reduce_add_epi32(a);}"),
        "amx_int8": ("amx-tile,amx-int8", "#include <immintrin.h>\nint main(){_tile_zero(0);return 0;}"),
    }
    compiled = {}
    tmp = ROOT / "build" / "hwprobe"
    tmp.mkdir(parents=True, exist_ok=True)
    for name, (target, src) in tests.items():
        f = tmp / f"{name}.c"
        f.write_text(src)
        r = subprocess.run([cc, "-O2", f"-m{target.replace(',', ' -m')}".split()[0]] + [f"-m{t}" for t in target.split(",")[1:]] + [str(f), "-o", str(tmp / name)], capture_output=True, text=True)
        compiled[name] = r.returncode == 0
    info["compiler_can_target_isa (compile-only, NOT runtime support)"] = compiled
    return info


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports" / "hardware.json"
    ci = cpuinfo()
    flags = ci.pop("flags")
    isa_keys = ["sse4_2", "avx", "avx2", "fma", "f16c", "bmi2", "avx_vnni", "avx512f", "avx512bw", "avx512vl",
                "avx512dq", "avx512_vnni", "avx512_bf16", "avx512_fp16", "avx_vnni_int8", "avx_ne_convert",
                "amx_tile", "amx_int8", "amx_bf16", "amx_fp16", "gfni", "vaes", "sha_ni"]
    isa = {k: (k in flags) for k in isa_keys}
    # ARM features are reported for completeness on aarch64 hosts.
    if platform.machine() in ("aarch64", "arm64"):
        for k in ("asimd", "asimddp", "i8mm", "bf16", "sve", "sve2", "sme"):
            isa[k] = k in flags
    gpus = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader") if shutil.which("nvidia-smi") else ""
    report = dict(
        generated_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        hostname="redacted",  # not recorded
        os=dict(kernel=platform.release(), system=platform.system(), machine=platform.machine(),
                distro=(read("/etc/os-release", "") or "").split("\n")[0].replace("PRETTY_NAME=", "").strip('"'),
                glibc=platform.libc_ver()[1]),
        cpu=dict(**ci, isa=isa, all_flags=sorted(flags)),
        topology=topology(),
        caches=caches(),
        numa=numa(),
        memory=meminfo(),
        power=dict(
            ac_online=read("/sys/class/power_supply/ADP1/online") or read("/sys/class/power_supply/AC/online"),
            intel_pstate_status=read("/sys/devices/system/cpu/intel_pstate/status"),
            no_turbo=read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
            governor_cpu0=read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            epp_cpu0=read("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"),
        ),
        profiling=dict(
            perf_path=shutil.which("perf"),
            perf_version=sh(["perf", "--version"]) if shutil.which("perf") else None,
            perf_event_paranoid=read("/proc/sys/kernel/perf_event_paranoid"),
            hardware_counters_accessible=("Disallow" not in sh(["perf", "stat", "-e", "cycles", "true"])) if shutil.which("perf") else False,
            valgrind=shutil.which("valgrind"),
        ),
        toolchain=dict(gcc=compiler_probe("gcc"), clang=compiler_probe("clang"),
                       make=sh(["make", "--version"]).splitlines()[0] if shutil.which("make") else None,
                       cmake=sh(["cmake", "--version"]).splitlines()[0] if shutil.which("cmake") else None,
                       python=sys.version.split()[0]),
        accelerators=dict(present=gpus or None,
                          policy="Present accelerators are NOT used: all measured runs are CPU-only "
                                 "(CPU torch wheel, onnxruntime CPUExecutionProvider, native C without GPU code, "
                                 "CUDA_VISIBLE_DEVICES='')."),
        load_at_detection=dict(loadavg=read("/proc/loadavg")),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    t = report["topology"]
    print(f"{ci['model']}: {t['physical_cores']} cores / {t['logical_cpus']} threads, hybrid={ {k: len(v) for k, v in t['hybrid'].items()} }")
    print("ISA:", ", ".join(k for k, v in isa.items() if v))
    print("absent:", ", ".join(k for k, v in isa.items() if not v))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
