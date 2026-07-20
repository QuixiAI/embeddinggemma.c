#define _POSIX_C_SOURCE 200809L

#include "engine.h"

#include <errno.h>
#include <math.h>
#include <time.h>

#define PERF_SCHEMA 1
#define MAX_TOKEN_SHAPES 16

static uint64_t now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        ei_die("clock_gettime failed: %s", strerror(errno));
    }
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int compare_double(const void *a, const void *b) {
    const double x = *(const double *)a;
    const double y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median_sorted(const double *values, int count) {
    return (count & 1) ? values[count / 2]
                       : 0.5 * (values[count / 2 - 1] + values[count / 2]);
}

static double cosine(const float *a, const float *b) {
    double dot = 0.0;
    double aa = 0.0;
    double bb = 0.0;
    for (int i = 0; i < EI_N_EMBD; i++) {
        dot += (double)a[i] * (double)b[i];
        aa += (double)a[i] * (double)a[i];
        bb += (double)b[i] * (double)b[i];
    }
    return dot / sqrt(aa * bb);
}

static int parse_token_shapes(const char *value, int32_t out[MAX_TOKEN_SHAPES]) {
    char *copy = strdup(value);
    if (!copy) ei_die("out of memory");
    int count = 0;
    char *save = NULL;
    for (char *item = strtok_r(copy, ",", &save); item; item = strtok_r(NULL, ",", &save)) {
        char *end = NULL;
        long tokens = strtol(item, &end, 10);
        if (*end != '\0' || tokens < 1 || tokens > EI_N_CTX || count == MAX_TOKEN_SHAPES) {
            free(copy);
            ei_die("--tokens must contain 1-%d comma-separated values", EI_N_CTX);
        }
        out[count++] = (int32_t)tokens;
    }
    free(copy);
    if (count == 0) ei_die("--tokens cannot be empty");
    return count;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s --model model.gguf [--backend auto|cpu|metal|cuda|rocm|xpu] "
            "[--tokens 1,8,32,128] [--warmup n] [--iters n] "
            "[--ab-gelu-quant] [--ab-metal-fp16-kv] "
            "[--ab-rocm-command-graph] [--ab-rocm-fp16-gemm] "
            "[--ab-rocm-q8-latency] [--ab-rocm-direct-qkv] "
            "[--ab-rocm-q8-two-row] [--ab-rocm-direct-context] "
            "[--ab-rocm-tensor-attention] [--ab-rocm-mfma-attention] "
            "[--ab-rocm-batched-attention] "
            "[--ab-rocm-fp16-attention-scores] "
            "[--ab-rocm-softmax-128] "
            "[--ab-rocm-v-only] "
            "[--ab-rocm-direct-rms] "
            "[--ab-rocm-direct-q4-pair] "
            "[--ab-rocm-direct-q4-quad] "
            "[--ab-rocm-pinned-io] "
            "[--ab-rocm-fused-embedding-rms] "
            "[--ab-rocm-singleton-metadata] "
            "[--ab-rocm-final-singleton-pool] "
            "[--ab-rocm-native-q4] [--ab-rocm-native-fused] "
            "[--ab-rocm-native-fused-activation] "
            "[--ab-rocm-native-wide] "
            "[--ab-rocm-native-direct-qkv] "
            "[--ab-rocm-rms-register-cache] "
            "[--ab-xpu-fp16-attention] [--ab-xpu-swa-banded] "
            "[--ab-xpu-v-only] [--ab-xpu-cooperative-rms] "
            "[--ab-xpu-cooperative-pool] [--ab-xpu-xe2-flash] "
            "[--ab-xpu-flash-event] "
            "[--ab-xpu-command-graph] "
            "[--ab-xpu-xe2-w4] "
            "[--ab-xpu-rms-register-cache] "
            "[--ab-xpu-onednn-w4] [--ab-xpu-onednn-f16] "
            "[--ab-xpu-q4-m-tiled]\n",
            argv0);
}

static double timed_embed(ei_engine *engine, const int32_t *ids, int32_t tokens,
                          float output[EI_N_EMBD]) {
    char err[256];
    uint64_t start = now_ns();
    if (!ei_engine_embed_tokens(engine, ids, (size_t)tokens,
                                output, err, sizeof err)) {
        ei_die("benchmark failed: %s", err);
    }
    return (double)(now_ns() - start) / 1000000.0;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *backend = "auto";
    const char *token_shapes = "1,8,32,128";
    int warmup = 1;
    int iters = 5;
    bool ab_gelu_quant = false;
    bool ab_metal_fp16_kv = false;
    bool ab_rocm_command_graph = false;
    bool ab_rocm_fp16_gemm = false;
    bool ab_rocm_q8_latency = false;
    bool ab_rocm_q8_two_row = false;
    bool ab_rocm_direct_qkv = false;
    bool ab_rocm_direct_context = false;
    bool ab_rocm_tensor_attention = false;
    bool ab_rocm_mfma_attention = false;
    bool ab_rocm_batched_attention = false;
    bool ab_rocm_fp16_attention_scores = false;
    bool ab_rocm_softmax_128 = false;
    bool ab_rocm_v_only = false;
    bool ab_rocm_direct_rms = false;
    bool ab_rocm_direct_q4_pair = false;
    bool ab_rocm_direct_q4_quad = false;
    bool ab_rocm_pinned_io = false;
    bool ab_rocm_fused_embedding_rms = false;
    bool ab_rocm_singleton_metadata = false;
    bool ab_rocm_final_singleton_pool = false;
    bool ab_rocm_native_q4 = false;
    bool ab_rocm_native_fused = false;
    bool ab_rocm_native_fused_activation = false;
    bool ab_rocm_native_wide = false;
    bool ab_rocm_native_direct_qkv = false;
    bool ab_rocm_rms_register_cache = false;
    bool ab_xpu_fp16_attention = false;
    bool ab_xpu_swa_banded = false;
    bool ab_xpu_v_only = false;
    bool ab_xpu_cooperative_rms = false;
    bool ab_xpu_cooperative_pool = false;
    bool ab_xpu_xe2_flash = false;
    bool ab_xpu_flash_event = false;
    bool ab_xpu_command_graph = false;
    bool ab_xpu_xe2_w4 = false;
    bool ab_xpu_rms_register_cache = false;
    bool ab_xpu_onednn_w4 = false;
    bool ab_xpu_onednn_f16 = false;
    bool ab_xpu_q4_m_tiled = false;
    const char *xpu_swa_tile = "1024";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend = argv[++i];
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            token_shapes = argv[++i];
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ab-gelu-quant") == 0) {
            ab_gelu_quant = true;
        } else if (strcmp(argv[i], "--ab-metal-fp16-kv") == 0) {
            ab_metal_fp16_kv = true;
        } else if (strcmp(argv[i], "--ab-rocm-command-graph") == 0) {
            ab_rocm_command_graph = true;
        } else if (strcmp(argv[i], "--ab-rocm-fp16-gemm") == 0) {
            ab_rocm_fp16_gemm = true;
        } else if (strcmp(argv[i], "--ab-rocm-q8-latency") == 0) {
            ab_rocm_q8_latency = true;
        } else if (strcmp(argv[i], "--ab-rocm-q8-two-row") == 0) {
            ab_rocm_q8_two_row = true;
        } else if (strcmp(argv[i], "--ab-rocm-direct-qkv") == 0) {
            ab_rocm_direct_qkv = true;
        } else if (strcmp(argv[i], "--ab-rocm-direct-context") == 0) {
            ab_rocm_direct_context = true;
        } else if (strcmp(argv[i], "--ab-rocm-tensor-attention") == 0) {
            ab_rocm_tensor_attention = true;
        } else if (strcmp(argv[i], "--ab-rocm-mfma-attention") == 0) {
            ab_rocm_mfma_attention = true;
        } else if (strcmp(argv[i], "--ab-rocm-batched-attention") == 0) {
            ab_rocm_batched_attention = true;
        } else if (strcmp(argv[i], "--ab-rocm-fp16-attention-scores") == 0) {
            ab_rocm_fp16_attention_scores = true;
        } else if (strcmp(argv[i], "--ab-rocm-softmax-128") == 0) {
            ab_rocm_softmax_128 = true;
        } else if (strcmp(argv[i], "--ab-rocm-v-only") == 0) {
            ab_rocm_v_only = true;
        } else if (strcmp(argv[i], "--ab-rocm-direct-rms") == 0) {
            ab_rocm_direct_rms = true;
        } else if (strcmp(argv[i], "--ab-rocm-direct-q4-pair") == 0) {
            ab_rocm_direct_q4_pair = true;
        } else if (strcmp(argv[i], "--ab-rocm-direct-q4-quad") == 0) {
            ab_rocm_direct_q4_quad = true;
        } else if (strcmp(argv[i], "--ab-rocm-pinned-io") == 0) {
            ab_rocm_pinned_io = true;
        } else if (strcmp(argv[i], "--ab-rocm-fused-embedding-rms") == 0) {
            ab_rocm_fused_embedding_rms = true;
        } else if (strcmp(argv[i], "--ab-rocm-singleton-metadata") == 0) {
            ab_rocm_singleton_metadata = true;
        } else if (strcmp(argv[i], "--ab-rocm-final-singleton-pool") == 0) {
            ab_rocm_final_singleton_pool = true;
        } else if (strcmp(argv[i], "--ab-rocm-native-q4") == 0) {
            ab_rocm_native_q4 = true;
        } else if (strcmp(argv[i], "--ab-rocm-native-fused") == 0) {
            ab_rocm_native_fused = true;
        } else if (strcmp(argv[i], "--ab-rocm-native-fused-activation") == 0) {
            ab_rocm_native_fused_activation = true;
        } else if (strcmp(argv[i], "--ab-rocm-native-wide") == 0) {
            ab_rocm_native_wide = true;
        } else if (strcmp(argv[i], "--ab-rocm-native-direct-qkv") == 0) {
            ab_rocm_native_direct_qkv = true;
        } else if (strcmp(argv[i], "--ab-rocm-rms-register-cache") == 0) {
            ab_rocm_rms_register_cache = true;
        } else if (strcmp(argv[i], "--ab-xpu-fp16-attention") == 0) {
            ab_xpu_fp16_attention = true;
        } else if (strcmp(argv[i], "--ab-xpu-swa-banded") == 0) {
            ab_xpu_swa_banded = true;
        } else if (strcmp(argv[i], "--xpu-swa-tile") == 0 && i + 1 < argc) {
            xpu_swa_tile = argv[++i];
        } else if (strcmp(argv[i], "--ab-xpu-v-only") == 0) {
            ab_xpu_v_only = true;
        } else if (strcmp(argv[i], "--ab-xpu-cooperative-rms") == 0) {
            ab_xpu_cooperative_rms = true;
        } else if (strcmp(argv[i], "--ab-xpu-cooperative-pool") == 0) {
            ab_xpu_cooperative_pool = true;
        } else if (strcmp(argv[i], "--ab-xpu-xe2-flash") == 0) {
            ab_xpu_xe2_flash = true;
        } else if (strcmp(argv[i], "--ab-xpu-flash-event") == 0) {
            ab_xpu_flash_event = true;
        } else if (strcmp(argv[i], "--ab-xpu-command-graph") == 0) {
            ab_xpu_command_graph = true;
        } else if (strcmp(argv[i], "--ab-xpu-xe2-w4") == 0) {
            ab_xpu_xe2_w4 = true;
        } else if (strcmp(argv[i], "--ab-xpu-rms-register-cache") == 0) {
            ab_xpu_rms_register_cache = true;
        } else if (strcmp(argv[i], "--ab-xpu-onednn-w4") == 0) {
            ab_xpu_onednn_w4 = true;
        } else if (strcmp(argv[i], "--ab-xpu-onednn-f16") == 0) {
            ab_xpu_onednn_f16 = true;
        } else if (strcmp(argv[i], "--ab-xpu-q4-m-tiled") == 0) {
            ab_xpu_q4_m_tiled = true;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!model_path || warmup < 0 || iters < 1) {
        usage(argv[0]);
        return 2;
    }

    int32_t shapes[MAX_TOKEN_SHAPES];
    int shape_count = parse_token_shapes(token_shapes, shapes);
    const bool ab_rocm = ab_rocm_command_graph || ab_rocm_fp16_gemm ||
        ab_rocm_q8_latency || ab_rocm_q8_two_row || ab_rocm_direct_qkv ||
        ab_rocm_direct_context || ab_rocm_tensor_attention ||
        ab_rocm_mfma_attention || ab_rocm_batched_attention ||
        ab_rocm_fp16_attention_scores ||
        ab_rocm_softmax_128 ||
        ab_rocm_v_only ||
        ab_rocm_direct_rms ||
        ab_rocm_direct_q4_pair ||
        ab_rocm_direct_q4_quad ||
        ab_rocm_pinned_io ||
        ab_rocm_fused_embedding_rms ||
        ab_rocm_singleton_metadata ||
        ab_rocm_final_singleton_pool ||
        ab_rocm_native_q4 ||
        ab_rocm_native_fused || ab_rocm_native_fused_activation ||
        ab_rocm_native_wide ||
        ab_rocm_native_direct_qkv ||
        ab_rocm_rms_register_cache;
    if (ab_metal_fp16_kv || ab_rocm ||
        ab_xpu_fp16_attention || ab_xpu_swa_banded ||
        ab_xpu_v_only || ab_xpu_cooperative_rms ||
        ab_xpu_cooperative_pool || ab_xpu_xe2_flash || ab_xpu_flash_event ||
        ab_xpu_command_graph || ab_xpu_xe2_w4 ||
        ab_xpu_rms_register_cache || ab_xpu_onednn_w4 ||
        ab_xpu_onednn_f16 || ab_xpu_q4_m_tiled) {
        const char *expected_backend =
            ab_metal_fp16_kv ? "metal"
            : ab_rocm ? "rocm" : "xpu";
        const char *environment = ab_metal_fp16_kv
            ? "EI_METAL_FP16_KV_MIN_TOKENS"
            : ab_rocm_command_graph ? "EI_ROCM_COMMAND_GRAPH"
            : ab_rocm_fp16_gemm ? "EI_ROCM_GEMM_MIN_TOKENS"
            : ab_rocm_q8_latency ? "EI_ROCM_Q8_LATENCY"
            : ab_rocm_q8_two_row ? "EI_ROCM_Q8_TWO_ROW"
            : ab_rocm_direct_qkv ? "EI_ROCM_DIRECT_FP16_QKV"
            : ab_rocm_direct_context ? "EI_ROCM_DIRECT_FP16_CONTEXT"
            : ab_rocm_tensor_attention ? "EI_ROCM_TENSOR_ATTENTION_MIN_TOKENS"
            : ab_rocm_mfma_attention ? "EI_ROCM_MFMA_ATTENTION"
            : ab_rocm_batched_attention ? "EI_ROCM_BATCHED_TENSOR_ATTENTION"
            : ab_rocm_fp16_attention_scores ? "EI_ROCM_FP16_ATTENTION_SCORES"
            : ab_rocm_softmax_128 ? "EI_ROCM_TENSOR_SOFTMAX_THREADS"
            : ab_rocm_v_only ? "EI_ROCM_SINGLE_TOKEN_V_ONLY"
            : ab_rocm_direct_rms ? "EI_ROCM_DIRECT_RMS_FUSION"
            : ab_rocm_direct_q4_pair ? "EI_ROCM_DIRECT_Q4_PAIR"
            : ab_rocm_direct_q4_quad ? "EI_ROCM_DIRECT_Q4_QUAD"
            : ab_rocm_pinned_io ? "EI_ROCM_PINNED_IO_STAGING"
            : ab_rocm_fused_embedding_rms ? "EI_ROCM_FUSED_EMBEDDING_RMS"
            : ab_rocm_singleton_metadata ? "EI_ROCM_SINGLETON_METADATA_ELISION"
            : ab_rocm_final_singleton_pool ? "EI_ROCM_FINAL_SINGLETON_POOL"
            : ab_rocm_native_q4 ? "EI_ROCM_NATIVE_Q4_GEMM"
            : ab_rocm_native_fused ? "EI_ROCM_NATIVE_Q4_FUSED"
            : ab_rocm_native_fused_activation ? "EI_ROCM_NATIVE_Q4_FUSED_ACTIVATION"
            : ab_rocm_native_wide ? "EI_ROCM_NATIVE_Q4_WIDE"
            : ab_rocm_native_direct_qkv ? "EI_ROCM_NATIVE_Q4_DIRECT_FP16_QKV"
            : ab_rocm_rms_register_cache ? "EI_ROCM_RMS_REGISTER_CACHE"
            : ab_xpu_swa_banded ? "EI_XPU_SWA_TENSOR_TILE_TOKENS"
            : ab_xpu_v_only ? "EI_XPU_SINGLE_TOKEN_V_ONLY"
            : ab_xpu_cooperative_rms ? "EI_XPU_COOPERATIVE_RMS_MAX_ROWS"
            : ab_xpu_cooperative_pool ? "EI_XPU_COOPERATIVE_POOL"
            : ab_xpu_xe2_flash ? "EI_XPU_XE2_FLASH"
            : ab_xpu_flash_event ? "EI_XPU_XE2_FLASH_HOST_FENCE"
            : ab_xpu_command_graph ? "EI_XPU_COMMAND_GRAPH"
            : ab_xpu_xe2_w4 ? "EI_XPU_XE2_W4"
            : ab_xpu_rms_register_cache ? "EI_XPU_RMS_REGISTER_CACHE"
            : ab_xpu_onednn_w4 ? "EI_XPU_ONEDNN_W4"
            : ab_xpu_onednn_f16 ? "EI_XPU_ONEDNN_F16"
            : ab_xpu_q4_m_tiled ? "EI_XPU_Q4_M_TILED"
                                : "EI_XPU_FP16_ATTENTION";
        const char *baseline_value = ab_rocm_softmax_128 ? "256"
            : (ab_metal_fp16_kv || ab_rocm_fp16_gemm ||
               ab_rocm_tensor_attention) ? "65536"
            : ab_xpu_flash_event ? "1" : "0";
        const char *candidate_value = ab_metal_fp16_kv
            ? "1" : ab_rocm_softmax_128 ? "128" : ab_rocm ? "1"
            : ab_xpu_swa_banded ? xpu_swa_tile
            : ab_xpu_v_only ? "1"
            : ab_xpu_cooperative_rms ? "4"
            : ab_xpu_cooperative_pool ? "1"
            : ab_xpu_xe2_flash ? "1"
            : ab_xpu_flash_event ? "0"
            : ab_xpu_command_graph ? "1"
            : ab_xpu_xe2_w4 ? "1"
            : ab_xpu_rms_register_cache ? "1"
            : ab_xpu_onednn_w4 ? "1"
            : ab_xpu_onednn_f16 ? "1"
            : ab_xpu_q4_m_tiled ? "1" : "auto";
        const char *variant = ab_metal_fp16_kv ? "fp16_kv_ab"
            : ab_rocm_command_graph ? "command_graph_ab"
            : ab_rocm_fp16_gemm ? "fp16_gemm_ab"
            : ab_rocm_q8_latency ? "q8_latency_ab"
            : ab_rocm_q8_two_row ? "q8_two_row_ab"
            : ab_rocm_direct_qkv ? "direct_qkv_ab"
            : ab_rocm_direct_context ? "direct_context_ab"
            : ab_rocm_tensor_attention ? "tensor_attention_ab"
            : ab_rocm_mfma_attention ? "mfma_attention_ab"
            : ab_rocm_batched_attention ? "batched_attention_ab"
            : ab_rocm_fp16_attention_scores ? "fp16_attention_scores_ab"
            : ab_rocm_softmax_128 ? "softmax_128_ab"
            : ab_rocm_v_only ? "v_only_ab"
            : ab_rocm_direct_rms ? "direct_rms_ab"
            : ab_rocm_direct_q4_pair ? "direct_q4_pair_ab"
            : ab_rocm_direct_q4_quad ? "direct_q4_quad_ab"
            : ab_rocm_pinned_io ? "pinned_io_ab"
            : ab_rocm_fused_embedding_rms ? "fused_embedding_rms_ab"
            : ab_rocm_singleton_metadata ? "singleton_metadata_ab"
            : ab_rocm_final_singleton_pool ? "final_singleton_pool_ab"
            : ab_rocm_native_q4 ? "native_q4_ab"
            : ab_rocm_native_fused ? "native_fused_ab"
            : ab_rocm_native_fused_activation ? "native_fused_activation_ab"
            : ab_rocm_native_wide ? "native_wide_ab"
            : ab_rocm_native_direct_qkv ? "native_direct_qkv_ab"
            : ab_rocm_rms_register_cache ? "rms_register_cache_ab"
            : ab_xpu_swa_banded ? "swa_banded_ab"
            : ab_xpu_v_only ? "single_token_v_only_ab"
            : ab_xpu_cooperative_rms ? "cooperative_rms_ab"
            : ab_xpu_cooperative_pool ? "cooperative_pool_ab"
            : ab_xpu_xe2_flash ? "xe2_flash_ab"
            : ab_xpu_flash_event ? "flash_event_ab"
            : ab_xpu_command_graph ? "command_graph_ab"
            : ab_xpu_xe2_w4 ? "xe2_w4_ab"
            : ab_xpu_rms_register_cache ? "rms_register_cache_ab"
            : ab_xpu_onednn_w4 ? "onednn_w4_ab"
            : ab_xpu_onednn_f16 ? "onednn_f16_ab"
            : ab_xpu_q4_m_tiled ? "q4_m_tiled_ab"
                            : "fp16_dense_auto_ab";
        const char *candidate_field =
            ab_rocm_command_graph ? "graph_ms"
            : ab_rocm_fp16_gemm ? "fp16_ms"
            : ab_rocm_q8_latency ? "q8_ms"
            : ab_rocm_q8_two_row ? "two_row_ms"
            : ab_rocm_direct_qkv ? "direct_ms"
            : ab_rocm_direct_context ? "direct_ms"
            : ab_rocm_tensor_attention ? "tensor_ms"
            : ab_rocm_mfma_attention ? "mfma_ms"
            : ab_rocm_batched_attention ? "batched_ms"
            : ab_rocm_fp16_attention_scores ? "fp16_scores_ms"
            : ab_rocm_softmax_128 ? "softmax_128_ms"
            : ab_rocm_v_only ? "v_only_ms"
            : ab_rocm_direct_rms ? "fused_ms"
            : ab_rocm_direct_q4_pair ? "pair_ms"
            : ab_rocm_direct_q4_quad ? "quad_ms"
            : ab_rocm_pinned_io ? "pinned_ms"
            : ab_rocm_fused_embedding_rms ? "fused_ms"
            : ab_rocm_singleton_metadata ? "elided_ms"
            : ab_rocm_final_singleton_pool ? "fused_ms"
            : ab_rocm_native_q4 ? "native_ms"
            : ab_rocm_native_fused ? "fused_ms"
            : ab_rocm_native_fused_activation ? "activation_ms"
            : ab_rocm_native_wide ? "wide_ms"
            : ab_rocm_native_direct_qkv ? "direct_ms"
            : ab_rocm_rms_register_cache ? "register_ms"
            : ab_xpu_swa_banded ? "banded_ms"
            : ab_xpu_v_only ? "v_only_ms"
            : ab_xpu_cooperative_rms ? "cooperative_ms"
            : ab_xpu_cooperative_pool ? "cooperative_ms"
            : ab_xpu_xe2_flash ? "flash_ms"
            : ab_xpu_flash_event ? "event_ms"
            : ab_xpu_command_graph ? "graph_ms"
            : ab_xpu_xe2_w4 ? "w4_ms"
            : ab_xpu_rms_register_cache ? "register_ms"
            : ab_xpu_onednn_w4 ? "w4_ms"
            : ab_xpu_onednn_f16 ? "onednn_ms"
            : ab_xpu_q4_m_tiled ? "m_tiled_ms" : "fp16_ms";
        if (strcmp(backend, expected_backend) != 0) {
            ei_die("A/B mode requires --backend %s", expected_backend);
        }
        ei_engine baseline;
        ei_engine fp16;
        if (ab_xpu_q4_m_tiled) {
            setenv("EI_XPU_GEMM_MIN_TOKENS", "65536", 1);
        }
        if (ab_xpu_swa_banded) {
            setenv("EI_XPU_SWA_TENSOR_MIN_TOKENS", "1", 1);
        }
        if (ab_rocm_fp16_gemm) {
            setenv("EI_ROCM_Q8_LATENCY", "0", 1);
        }
        if (ab_rocm_q8_two_row) {
            setenv("EI_ROCM_GEMM_MIN_TOKENS", "65536", 1);
            setenv("EI_ROCM_Q8_LATENCY", "1", 1);
        }
        if (ab_rocm_native_q4 || ab_rocm_native_wide) {
            setenv("EI_ROCM_NATIVE_Q4_MAX_TOKENS", "65536", 1);
        }
        setenv(environment, baseline_value, 1);
        ei_engine_load_backend(&baseline, model_path, expected_backend);
        setenv(environment, candidate_value, 1);
        ei_engine_load_backend(&fp16, model_path, expected_backend);
        unsetenv(environment);
        if (ab_xpu_q4_m_tiled) {
            unsetenv("EI_XPU_GEMM_MIN_TOKENS");
        }
        if (ab_xpu_swa_banded) {
            unsetenv("EI_XPU_SWA_TENSOR_MIN_TOKENS");
        }
        if (ab_rocm_fp16_gemm) {
            unsetenv("EI_ROCM_Q8_LATENCY");
        }
        if (ab_rocm_q8_two_row) {
            unsetenv("EI_ROCM_GEMM_MIN_TOKENS");
            unsetenv("EI_ROCM_Q8_LATENCY");
        }
        if (ab_rocm_native_q4 || ab_rocm_native_wide) {
            unsetenv("EI_ROCM_NATIVE_Q4_MAX_TOKENS");
        }

        for (int shape = 0; shape < shape_count; shape++) {
            int32_t tokens = shapes[shape];
            int32_t *ids = ei_xmalloc(sizeof(*ids) * (size_t)tokens);
            for (int32_t i = 0; i < tokens; i++) ids[i] = 1000 + (i * 7919) % 240000;
            float baseline_output[EI_N_EMBD];
            float fp16_output[EI_N_EMBD];
            for (int i = 0; i < warmup; i++) {
                (void)timed_embed(&baseline, ids, tokens, baseline_output);
                (void)timed_embed(&fp16, ids, tokens, fp16_output);
            }
            double *baseline_samples = ei_xmalloc(sizeof(*baseline_samples) * (size_t)iters);
            double *fp16_samples = ei_xmalloc(sizeof(*fp16_samples) * (size_t)iters);
            for (int i = 0; i < iters; i++) {
                if ((i & 1) == 0) {
                    baseline_samples[i] = timed_embed(&baseline, ids, tokens, baseline_output);
                    fp16_samples[i] = timed_embed(&fp16, ids, tokens, fp16_output);
                } else {
                    fp16_samples[i] = timed_embed(&fp16, ids, tokens, fp16_output);
                    baseline_samples[i] = timed_embed(&baseline, ids, tokens, baseline_output);
                }
            }
            double similarity = cosine(baseline_output, fp16_output);
            const double minimum_cosine =
                (ab_rocm_fp16_gemm || ab_rocm_q8_latency) ? 0.99 : 0.998;
            if (!isfinite(similarity) || similarity < minimum_cosine) {
                ei_die("A/B parity failed: %.9g", similarity);
            }
            qsort(baseline_samples, (size_t)iters, sizeof(*baseline_samples), compare_double);
            qsort(fp16_samples, (size_t)iters, sizeof(*fp16_samples), compare_double);
            double baseline_median = median_sorted(baseline_samples, iters);
            double fp16_median = median_sorted(fp16_samples, iters);
            printf("{\"schema\":%d,\"backend\":\"%s\","
                   "\"kernel\":\"engine_embed_tokens\","
                   "\"variant\":\"%s\",\"shape\":{\"tokens\":%d},"
                   "\"threads\":1,\"%s\":%.9g,\"baseline_ms\":%.9g,"
                   "\"throughput_speedup\":%.9g,\"cosine\":%.9g}\n",
                   PERF_SCHEMA, expected_backend, variant, tokens,
                   candidate_field,
                   fp16_median, baseline_median,
                   baseline_median / fp16_median, similarity);
            free(fp16_samples);
            free(baseline_samples);
            free(ids);
        }
        ei_engine_free(&fp16);
        ei_engine_free(&baseline);
        return 0;
    }

    ei_engine engine;
    ei_engine_load_backend(&engine, model_path, backend);
    if (ab_gelu_quant && strcmp(ei_engine_backend(&engine), "cpu") != 0) {
        ei_die("--ab-gelu-quant requires the CPU backend");
    }

    for (int shape = 0; shape < shape_count; shape++) {
        int32_t tokens = shapes[shape];
        int32_t *ids = ei_xmalloc(sizeof(*ids) * (size_t)tokens);
        for (int32_t i = 0; i < tokens; i++) ids[i] = 1000 + (i * 7919) % 240000;

        float output[EI_N_EMBD];
        char err[256];
        if (ab_gelu_quant) {
            for (int i = 0; i < warmup; i++) {
                engine.fused_gelu_quant = true;
                (void)timed_embed(&engine, ids, tokens, output);
                engine.fused_gelu_quant = false;
                (void)timed_embed(&engine, ids, tokens, output);
            }
            double *fused = ei_xmalloc(sizeof(*fused) * (size_t)iters);
            double *baseline = ei_xmalloc(sizeof(*baseline) * (size_t)iters);
            double checksum_fused = 0.0;
            double checksum_baseline = 0.0;
            for (int i = 0; i < iters; i++) {
                if ((i & 1) == 0) {
                    engine.fused_gelu_quant = true;
                    fused[i] = timed_embed(&engine, ids, tokens, output);
                    checksum_fused += output[(i * 97) % EI_N_EMBD];
                    engine.fused_gelu_quant = false;
                    baseline[i] = timed_embed(&engine, ids, tokens, output);
                    checksum_baseline += output[(i * 97) % EI_N_EMBD];
                } else {
                    engine.fused_gelu_quant = false;
                    baseline[i] = timed_embed(&engine, ids, tokens, output);
                    checksum_baseline += output[(i * 97) % EI_N_EMBD];
                    engine.fused_gelu_quant = true;
                    fused[i] = timed_embed(&engine, ids, tokens, output);
                    checksum_fused += output[(i * 97) % EI_N_EMBD];
                }
            }
            qsort(fused, (size_t)iters, sizeof(*fused), compare_double);
            qsort(baseline, (size_t)iters, sizeof(*baseline), compare_double);
            double fused_median = fused[iters / 2];
            double baseline_median = baseline[iters / 2];
            printf("{\"schema\":%d,\"backend\":\"cpu\","
                   "\"kernel\":\"engine_embed_tokens\","
                   "\"variant\":\"fused_gelu_quant_ab\","
                   "\"shape\":{\"tokens\":%d},\"threads\":%d,"
                   "\"fused_ms\":%.9g,\"baseline_ms\":%.9g,"
                   "\"throughput_speedup\":%.9g,"
                   "\"checksum_delta\":%.9g}\n",
                   PERF_SCHEMA, tokens, ei_engine_threads(&engine),
                   fused_median, baseline_median,
                   baseline_median / fused_median,
                   checksum_fused - checksum_baseline);
            free(baseline);
            free(fused);
            free(ids);
            continue;
        }

        for (int i = 0; i < warmup; i++) {
            if (!ei_engine_embed_tokens(&engine, ids, (size_t)tokens, output, err, sizeof err)) {
                ei_die("warmup failed: %s", err);
            }
        }

        double *samples = ei_xmalloc(sizeof(*samples) * (size_t)iters);
        double checksum = 0.0;
        for (int i = 0; i < iters; i++) {
            uint64_t start = now_ns();
            if (!ei_engine_embed_tokens(&engine, ids, (size_t)tokens, output, err, sizeof err)) {
                ei_die("benchmark failed: %s", err);
            }
            uint64_t end = now_ns();
            samples[i] = (double)(end - start) / 1000000.0;
            checksum += output[(i * 97) % EI_N_EMBD];
        }
        qsort(samples, (size_t)iters, sizeof(*samples), compare_double);
        double median = median_sorted(samples, iters);
        double p20 = samples[(int)(0.20 * (double)(iters - 1))];
        double p80 = samples[(int)(0.80 * (double)(iters - 1))];
        printf("{\"schema\":%d,\"backend\":\"%s\",\"kernel\":\"engine_embed_tokens\","
               "\"variant\":\"full_graph\",\"shape\":{\"tokens\":%d},\"threads\":%d,"
               "\"target_ms\":%.9g,\"target_p20_ms\":%.9g,\"target_p80_ms\":%.9g,"
               "\"tokens_per_second\":%.9g,\"checksum\":%.9g}\n",
               PERF_SCHEMA, ei_engine_backend(&engine), tokens, ei_engine_threads(&engine),
               median, p20, p80, 1000.0 * (double)tokens / median, checksum);
        free(samples);
        free(ids);
    }

    ei_engine_free(&engine);
    return 0;
}
