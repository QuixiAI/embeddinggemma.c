#include "engine_xpu.h"
#include "engine_xpu_flash.h"
#include "engine_xpu_onednn.h"
#include "engine_xpu_w4.h"

#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t kSubgroup = 16;
constexpr size_t kSubgroupsPerWorkgroup = 8;
constexpr size_t kWorkgroup = kSubgroup * kSubgroupsPerWorkgroup;
constexpr size_t kQK = 32;
constexpr size_t kQ4BlockBytes = 18;
constexpr size_t kQ8BlockBytes = 34;

static bool profiling_requested() {
    const char *value = std::getenv("EI_XPU_PROFILE");
    return value && *value && std::strcmp(value, "0") != 0;
}

static sycl::property_list queue_properties(bool profiling) {
    if (profiling) {
        return sycl::property_list{
            sycl::property::queue::in_order{},
            sycl::property::queue::enable_profiling{},
            sycl::ext::intel::property::queue::immediate_command_list{}};
    }
    return sycl::property_list{
        sycl::property::queue::in_order{},
        sycl::ext::intel::property::queue::immediate_command_list{}};
}

static bool is_xe2_device(sycl::device device) {
#ifdef EI_XPU_XE2_FLASH
    using architecture = sycl::ext::oneapi::experimental::architecture;
    try {
        return device.ext_oneapi_architecture_is(
                   architecture::intel_gpu_bmg_g21) ||
               device.ext_oneapi_architecture_is(
                   architecture::intel_gpu_bmg_g31);
    } catch (const sycl::exception &) {
        return false;
    }
#else
    (void)device;
    return false;
#endif
}

struct xpu_profile_sample {
    const char *label;
    sycl::event event;
};

struct xpu_layer_weights {
    sycl::half *qkv = nullptr;
    sycl::half *attn_output = nullptr;
    sycl::half *up_gate = nullptr;
    sycl::half *ffn_down = nullptr;
    ei_xpu_onednn_weight qkv_q4;
    ei_xpu_onednn_weight attn_output_q4;
    ei_xpu_onednn_weight up_gate_q4;
    ei_xpu_onednn_weight ffn_down_q4;
    ei_xpu_w4_weight up_gate_w4;
};

namespace graph_exp = sycl::ext::oneapi::experimental;
using xpu_executable_graph =
    graph_exp::command_graph<graph_exp::graph_state::executable>;

struct xpu_graph_entry {
    std::vector<uint32_t> sequence_lengths;
    std::shared_ptr<xpu_executable_graph> graph;
    uint64_t last_used = 0;
};

struct xpu_engine {
    explicit xpu_engine(const sycl::device &selected)
        : device(selected),
          xe2_device(is_xe2_device(selected)),
          profiling(profiling_requested()),
          queue(selected,
                [](sycl::exception_list errors) {
                    for (const std::exception_ptr &error : errors) {
                        std::rethrow_exception(error);
                    }
                },
                queue_properties(profiling)) {}

    const ei_model *model = nullptr;
    sycl::device device;
    bool xe2_device;
    bool profiling;
    sycl::queue queue;
    std::vector<xpu_profile_sample> profile_samples;
    uint8_t *model_data = nullptr;
    float *rope_full = nullptr;
    float *rope_swa = nullptr;
    std::array<xpu_layer_weights, EI_N_LAYER> layers;
    uint32_t gemm_min_tokens = 1;
    uint32_t tensor_attention_min_tokens = 128;
    bool tensor_attention_threshold_forced = false;
    uint32_t swa_tensor_tile_tokens = 0;
    uint32_t swa_tensor_banded_min_tokens = 1536;
    bool single_token_v_only = true;
    uint32_t cooperative_rms_max_rows = 4;
    bool rms_register_cache = false;
    int cooperative_pool_mode = 0;
#ifdef EI_XPU_XE2_FLASH
    int xe2_flash_mode = 0;
#else
    int xe2_flash_mode = -1;
#endif
    uint32_t xe2_flash_min_tokens = 256;
    uint32_t xe2_flash_batch_min_tokens = 128;
    bool xe2_flash_host_fence = false;
    int fp16_attention_mode = 0;
    bool fp16_attention = false;
    bool onednn_w4 = false;
    bool onednn_f16 = false;
    void *onednn = nullptr;
    bool q4_m_tiled = false;
    bool xe2_w4 = false;
    bool fuse_attn_output_half = false;
    int command_graph_mode = 0;
    uint64_t graph_clock = 0;
    std::vector<xpu_graph_entry> graph_cache;

    size_t capacity = 0;
    size_t batch_capacity = 0;
    int32_t *ids = nullptr;
    uint32_t *positions = nullptr;
    uint32_t *sequence_ids = nullptr;
    uint32_t *offsets = nullptr;
    float *x = nullptr;
    float *norm = nullptr;
    float *q = nullptr;
    float *k = nullptr;
    float *v = nullptr;
    sycl::half *q_half = nullptr;
    sycl::half *k_half = nullptr;
    sycl::half *v_half = nullptr;
    float *ctx = nullptr;
    float *up = nullptr;
    float *gate = nullptr;
    float *tmp = nullptr;
    sycl::half *half_input = nullptr;
    float *qkv_combined = nullptr;
    float *up_gate_combined = nullptr;
    float *attention_scores = nullptr;
    sycl::half *attention_probabilities = nullptr;
    sycl::half *w4_output = nullptr;
    float *result = nullptr;
};

static void record_profile(xpu_engine *engine, const char *label,
                           const sycl::event &event) {
    if (engine->profiling) {
        engine->profile_samples.push_back({label, event});
    }
}

static void report_profile(xpu_engine *engine, size_t tokens,
                           size_t batch_size, double submit_us,
                           double total_us) {
    struct aggregate {
        uint64_t nanoseconds = 0;
        size_t calls = 0;
    };
    std::map<std::string, aggregate> totals;
    for (const xpu_profile_sample &sample : engine->profile_samples) {
        try {
            const uint64_t start = sample.event.get_profiling_info<
                sycl::info::event_profiling::command_start>();
            const uint64_t end = sample.event.get_profiling_info<
                sycl::info::event_profiling::command_end>();
            aggregate &entry = totals[sample.label];
            entry.nanoseconds += end - start;
            entry.calls++;
        } catch (const sycl::exception &) {
        }
    }
    std::fprintf(stderr,
                 "XPU profile tokens=%zu batch=%zu submit_us=%.3f total_us=%.3f\n",
                 tokens, batch_size, submit_us, total_us);
    for (const auto &item : totals) {
        const double total = static_cast<double>(item.second.nanoseconds) /
                             1000.0;
        std::fprintf(stderr,
                     "  %-28s calls=%zu device_us=%.3f average_us=%.3f\n",
                     item.first.c_str(), item.second.calls, total,
                     total / static_cast<double>(item.second.calls));
    }
}

static void set_error(char *err, size_t err_len, const char *message) {
    if (err && err_len) std::snprintf(err, err_len, "%s", message);
}

static void set_exception(char *err, size_t err_len, const char *operation,
                          const std::exception &exception) {
    if (err && err_len) {
        std::snprintf(err, err_len, "%s: %s", operation, exception.what());
    }
}

static sycl::device select_device() {
    std::vector<sycl::device> devices;
    for (const sycl::platform &platform : sycl::platform::get_platforms()) {
        const std::string name =
            platform.get_info<sycl::info::platform::name>();
        if (name.find("Level-Zero") == std::string::npos &&
            name.find("Level Zero") == std::string::npos) {
            continue;
        }
        std::vector<sycl::device> platform_devices =
            platform.get_devices(sycl::info::device_type::gpu);
        devices.insert(devices.end(), platform_devices.begin(),
                       platform_devices.end());
    }
    if (devices.empty()) {
        devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    }
    if (devices.empty()) throw std::runtime_error("no SYCL GPU devices are available");

    size_t index = 0;
    const char *device_env = std::getenv("EI_XPU_DEVICE");
    if (device_env && *device_env) {
        char *end = nullptr;
        const unsigned long parsed = std::strtoul(device_env, &end, 10);
        if (*end != '\0' || parsed >= devices.size()) {
            throw std::runtime_error(
                "EI_XPU_DEVICE is outside the available device range");
        }
        index = static_cast<size_t>(parsed);
    }
    return devices[index];
}

template <typename T>
static T *device_allocate(xpu_engine *engine, size_t count) {
    T *allocation = sycl::malloc_device<T>(count, engine->queue);
    if (!allocation) throw std::bad_alloc();
    return allocation;
}

template <typename T>
static void device_release(xpu_engine *engine, T *&pointer) {
    if (pointer) sycl::free(pointer, engine->queue);
    pointer = nullptr;
}

static void release_workspace(xpu_engine *engine) {
    engine->graph_cache.clear();
    device_release(engine, engine->ids);
    device_release(engine, engine->positions);
    device_release(engine, engine->sequence_ids);
    device_release(engine, engine->offsets);
    device_release(engine, engine->x);
    device_release(engine, engine->norm);
    device_release(engine, engine->q);
    device_release(engine, engine->k);
    device_release(engine, engine->v);
    device_release(engine, engine->q_half);
    device_release(engine, engine->k_half);
    device_release(engine, engine->v_half);
    device_release(engine, engine->ctx);
    device_release(engine, engine->up);
    device_release(engine, engine->gate);
    device_release(engine, engine->tmp);
    device_release(engine, engine->half_input);
    device_release(engine, engine->qkv_combined);
    device_release(engine, engine->up_gate_combined);
    device_release(engine, engine->attention_scores);
    device_release(engine, engine->attention_probabilities);
    device_release(engine, engine->w4_output);
    device_release(engine, engine->result);
    engine->capacity = 0;
    engine->batch_capacity = 0;
}

static size_t grow_capacity(size_t current, size_t required) {
    size_t capacity = current ? current : 16;
    while (capacity < required) {
        if (capacity > std::numeric_limits<size_t>::max() / 2) {
            throw std::overflow_error("XPU workspace size overflow");
        }
        capacity *= 2;
    }
    return capacity;
}

static void ensure_capacity(xpu_engine *engine, size_t tokens,
                            size_t batch_size) {
    if (tokens <= engine->capacity && batch_size <= engine->batch_capacity) {
        return;
    }
    const size_t capacity = grow_capacity(engine->capacity, tokens);
    const size_t batch_capacity =
        grow_capacity(engine->batch_capacity, batch_size);

    release_workspace(engine);
    try {
        engine->ids = device_allocate<int32_t>(engine, capacity);
        engine->positions = device_allocate<uint32_t>(engine, capacity);
        engine->sequence_ids = device_allocate<uint32_t>(engine, capacity);
        engine->offsets = device_allocate<uint32_t>(engine, batch_capacity + 1);
        engine->x = device_allocate<float>(engine, capacity * EI_N_EMBD);
        engine->norm = device_allocate<float>(engine, capacity * EI_N_EMBD);
        engine->q = device_allocate<float>(engine, capacity * EI_N_EMBD);
        engine->k = device_allocate<float>(engine, capacity * EI_HEAD_DIM);
        engine->v = device_allocate<float>(engine, capacity * EI_HEAD_DIM);
        engine->q_half =
            device_allocate<sycl::half>(engine, capacity * EI_N_EMBD);
        engine->k_half =
            device_allocate<sycl::half>(engine, capacity * EI_HEAD_DIM);
        engine->v_half =
            device_allocate<sycl::half>(engine, capacity * EI_HEAD_DIM);
        engine->ctx = device_allocate<float>(engine, capacity * EI_N_EMBD);
        engine->up = device_allocate<float>(engine, capacity * EI_N_FF);
        engine->gate = device_allocate<float>(engine, capacity * EI_N_FF);
        engine->tmp = device_allocate<float>(engine, capacity * EI_N_EMBD);
        engine->half_input =
            device_allocate<sycl::half>(engine, capacity * EI_N_FF);
        engine->qkv_combined = device_allocate<float>(
            engine, capacity * (EI_N_EMBD + 2 * EI_HEAD_DIM));
        engine->up_gate_combined =
            device_allocate<float>(engine, capacity * 2 * EI_N_FF);
        const size_t score_tokens = std::min(capacity,
            static_cast<size_t>(EI_N_CTX));
        engine->attention_scores = device_allocate<float>(
            engine, EI_N_HEAD * score_tokens * score_tokens);
        engine->attention_probabilities = device_allocate<sycl::half>(
            engine, EI_N_HEAD * score_tokens * score_tokens);
#ifdef EI_XPU_XE2_W4
        engine->w4_output = device_allocate<sycl::half>(
            engine, capacity * 2 * EI_N_FF);
#endif
        engine->result =
            device_allocate<float>(engine, batch_capacity * EI_N_EMBD);
        engine->capacity = capacity;
        engine->batch_capacity = batch_capacity;
    } catch (...) {
        release_workspace(engine);
        throw;
    }
}

static inline float load_half(const uint8_t *bytes) {
    const uint16_t bits = static_cast<uint16_t>(bytes[0]) |
                          (static_cast<uint16_t>(bytes[1]) << 8);
    return static_cast<float>(sycl::bit_cast<sycl::half>(bits));
}

static const uint8_t *tensor_bytes(const xpu_engine *engine,
                                   const ei_tensor *tensor) {
    return engine->model_data + tensor->offset;
}

static const float *float_weights(const xpu_engine *engine,
                                  const float *host_weights) {
    const uint8_t *host_base = engine->model->gguf.map +
                               engine->model->gguf.data_off;
    const size_t offset = reinterpret_cast<const uint8_t *>(host_weights) -
                          host_base;
    return reinterpret_cast<const float *>(engine->model_data + offset);
}

static void dequantize_q4_rows(const ei_tensor *tensor,
                               std::vector<sycl::half> &output,
                               size_t output_row) {
    const size_t cols = static_cast<size_t>(tensor->ne[0]);
    const size_t rows = static_cast<size_t>(tensor->ne[1]);
    const size_t blocks_per_row = cols / kQK;
    const uint8_t *weights = static_cast<const uint8_t *>(tensor->data);
    for (size_t row = 0; row < rows; row++) {
        for (size_t block = 0; block < blocks_per_row; block++) {
            const uint8_t *packed = weights +
                (row * blocks_per_row + block) * kQ4BlockBytes;
            const uint16_t scale_bits = static_cast<uint16_t>(packed[0]) |
                (static_cast<uint16_t>(packed[1]) << 8);
            const float scale = ei_fp16_to_fp32(scale_bits);
            const size_t base = (output_row + row) * cols + block * kQK;
            for (size_t quant = 0; quant < 16; quant++) {
                const uint8_t code = packed[2 + quant];
                output[base + quant] = static_cast<sycl::half>(
                    scale * static_cast<float>((code & 0x0f) - 8));
                output[base + 16 + quant] = static_cast<sycl::half>(
                    scale * static_cast<float>((code >> 4) - 8));
            }
        }
    }
}

static void initialize_expanded_weights(xpu_engine *engine) {
    for (int layer_index = 0; layer_index < EI_N_LAYER; layer_index++) {
        const ei_layer &model_layer = engine->model->layers[layer_index];
        xpu_layer_weights &layer = engine->layers[layer_index];
        const size_t qkv_rows = EI_N_EMBD + 2 * EI_HEAD_DIM;
        const size_t up_gate_rows = 2 * EI_N_FF;
        layer.qkv = device_allocate<sycl::half>(
            engine, qkv_rows * EI_N_EMBD);
        layer.attn_output = device_allocate<sycl::half>(
            engine, static_cast<size_t>(EI_N_EMBD) * EI_N_EMBD);
        layer.up_gate = device_allocate<sycl::half>(
            engine, up_gate_rows * EI_N_EMBD);
        layer.ffn_down = device_allocate<sycl::half>(
            engine, static_cast<size_t>(EI_N_EMBD) * EI_N_FF);

        std::vector<sycl::half> qkv(qkv_rows * EI_N_EMBD);
        dequantize_q4_rows(model_layer.attn_q, qkv, 0);
        dequantize_q4_rows(model_layer.attn_k, qkv, EI_N_EMBD);
        dequantize_q4_rows(model_layer.attn_v, qkv,
                           EI_N_EMBD + EI_HEAD_DIM);
        std::vector<sycl::half> attention(
            static_cast<size_t>(EI_N_EMBD) * EI_N_EMBD);
        dequantize_q4_rows(model_layer.attn_output, attention, 0);
        std::vector<sycl::half> up_gate(up_gate_rows * EI_N_EMBD);
        dequantize_q4_rows(model_layer.ffn_up, up_gate, 0);
        dequantize_q4_rows(model_layer.ffn_gate, up_gate, EI_N_FF);
        std::vector<sycl::half> down(
            static_cast<size_t>(EI_N_EMBD) * EI_N_FF);
        dequantize_q4_rows(model_layer.ffn_down, down, 0);

        engine->queue.memcpy(layer.qkv, qkv.data(),
                             qkv.size() * sizeof(sycl::half));
        engine->queue.memcpy(layer.attn_output, attention.data(),
                             attention.size() * sizeof(sycl::half));
        engine->queue.memcpy(layer.up_gate, up_gate.data(),
                             up_gate.size() * sizeof(sycl::half));
        engine->queue.memcpy(layer.ffn_down, down.data(),
                             down.size() * sizeof(sycl::half));
        engine->queue.wait_and_throw();
    }
}

static void release_expanded_weights(xpu_engine *engine) {
    for (xpu_layer_weights &layer : engine->layers) {
        device_release(engine, layer.qkv);
        device_release(engine, layer.attn_output);
        device_release(engine, layer.up_gate);
        device_release(engine, layer.ffn_down);
    }
}

#ifdef EI_XPU_XE2_W4
static void initialize_xe2_w4_weights(xpu_engine *engine) {
    for (int layer_index = 0; layer_index < EI_N_LAYER; layer_index++) {
        const ei_layer &model_layer = engine->model->layers[layer_index];
        xpu_layer_weights &layer = engine->layers[layer_index];
        const ei_tensor *up_gate[] = {
            model_layer.ffn_up, model_layer.ffn_gate};
        ei_xpu_w4_prepare_weight(
            engine->queue, layer.up_gate_w4, up_gate, 2);
    }
    engine->queue.wait_and_throw();
}

static void release_xe2_w4_weights(xpu_engine *engine) {
    for (xpu_layer_weights &layer : engine->layers) {
        ei_xpu_w4_release_weight(engine->queue, layer.up_gate_w4);
    }
}
#else
static void release_xe2_w4_weights(xpu_engine *) {}
#endif

#ifdef EI_XPU_ONEDNN
static void initialize_onednn_weights(xpu_engine *engine) {
    if (!engine->onednn) engine->onednn = ei_xpu_onednn_create(engine->queue);
    for (int layer_index = 0; layer_index < EI_N_LAYER; layer_index++) {
        const ei_layer &model_layer = engine->model->layers[layer_index];
        xpu_layer_weights &layer = engine->layers[layer_index];
        const ei_tensor *qkv[] = {
            model_layer.attn_q, model_layer.attn_k, model_layer.attn_v};
        const ei_tensor *attention[] = {model_layer.attn_output};
        const ei_tensor *up_gate[] = {
            model_layer.ffn_up, model_layer.ffn_gate};
        const ei_tensor *down[] = {model_layer.ffn_down};
        ei_xpu_onednn_prepare_weight(
            engine->queue, layer.qkv_q4, qkv, 3);
        ei_xpu_onednn_prepare_weight(
            engine->queue, layer.attn_output_q4, attention, 1);
        ei_xpu_onednn_prepare_weight(
            engine->queue, layer.up_gate_q4, up_gate, 2);
        ei_xpu_onednn_prepare_weight(
            engine->queue, layer.ffn_down_q4, down, 1);
    }
}

static void release_onednn_weights(xpu_engine *engine) {
    for (xpu_layer_weights &layer : engine->layers) {
        ei_xpu_onednn_release_weight(engine->queue, layer.qkv_q4);
        ei_xpu_onednn_release_weight(engine->queue, layer.attn_output_q4);
        ei_xpu_onednn_release_weight(engine->queue, layer.up_gate_q4);
        ei_xpu_onednn_release_weight(engine->queue, layer.ffn_down_q4);
    }
    ei_xpu_onednn_destroy(engine->onednn);
    engine->onednn = nullptr;
}
#else
static void release_onednn_weights(xpu_engine *) {}
#endif

static void launch_embedding(xpu_engine *engine, size_t tokens) {
    const uint8_t *table = tensor_bytes(engine, engine->model->token_embd);
    const int32_t *ids = engine->ids;
    float *output = engine->x;
    const float scale = std::sqrt(static_cast<float>(EI_N_EMBD));
    const size_t count = tokens * EI_N_EMBD;
    const sycl::event event = engine->queue.parallel_for(
        sycl::range<1>(count), [=](sycl::id<1> index) {
        const size_t linear = index[0];
        const size_t token = linear / EI_N_EMBD;
        const size_t dim = linear - token * EI_N_EMBD;
        const size_t block = dim / kQK;
        const size_t item = dim - block * kQK;
        const size_t row_bytes = (EI_N_EMBD / kQK) * kQ8BlockBytes;
        const uint8_t *packed = table +
            static_cast<size_t>(ids[token]) * row_bytes + block * kQ8BlockBytes;
        const float d = load_half(packed);
        const int8_t quant = static_cast<int8_t>(packed[2 + item]);
        output[linear] = static_cast<float>(quant) * d * scale;
        });
    record_profile(engine, "embedding_q8", event);
}

static void launch_rms_norm(xpu_engine *engine, const float *input,
                            const float *weight, float *output, size_t rows,
                            size_t cols, float eps) {
    const size_t groups = (rows + kSubgroupsPerWorkgroup - 1) /
                          kSubgroupsPerWorkgroup;
    const sycl::event event = engine->queue.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(groups * kWorkgroup),
                          sycl::range<1>(kWorkgroup)),
        [=](sycl::nd_item<1> item)
            [[sycl::reqd_sub_group_size(kSubgroup)]] {
            const sycl::sub_group subgroup = item.get_sub_group();
            const size_t subgroup_id = item.get_local_id(0) / kSubgroup;
            const size_t lane = subgroup.get_local_linear_id();
            const size_t row = item.get_group(0) * kSubgroupsPerWorkgroup +
                               subgroup_id;
            if (row >= rows) return;
            const size_t base = row * cols;
            float sum = 0.0f;
            for (size_t col = lane; col < cols; col += kSubgroup) {
                const float value = input[base + col];
                sum = sycl::fma(value, value, sum);
            }
            sum = sycl::reduce_over_group(subgroup, sum, sycl::plus<float>());
            const float inv = sycl::rsqrt(sum / static_cast<float>(cols) + eps);
            for (size_t col = lane; col < cols; col += kSubgroup) {
                output[base + col] = input[base + col] * weight[col] * inv;
            }
        });
    record_profile(engine, "rms_norm_f32", event);
}

static void launch_rms_norm_half(xpu_engine *engine, const float *input,
                                 const float *weight, sycl::half *output,
                                 size_t rows, size_t cols, float eps) {
    if (rows <= engine->cooperative_rms_max_rows) {
        const sycl::event event = engine->queue.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(rows * kWorkgroup),
                              sycl::range<1>(kWorkgroup)),
            [=](sycl::nd_item<1> item) {
                const size_t row = item.get_group(0);
                const size_t lane = item.get_local_id(0);
                const size_t base = row * cols;
                float sum = 0.0f;
                for (size_t col = lane; col < cols; col += kWorkgroup) {
                    const float value = input[base + col];
                    sum = sycl::fma(value, value, sum);
                }
                sum = sycl::reduce_over_group(
                    item.get_group(), sum, sycl::plus<float>());
                const float inv =
                    sycl::rsqrt(sum / static_cast<float>(cols) + eps);
                for (size_t col = lane; col < cols; col += kWorkgroup) {
                    output[base + col] = static_cast<sycl::half>(
                        input[base + col] * weight[col] * inv);
                }
            });
        record_profile(engine, "rms_norm_f16_cooperative", event);
        return;
    }
    const size_t groups = (rows + kSubgroupsPerWorkgroup - 1) /
                          kSubgroupsPerWorkgroup;
    const sycl::event event = engine->queue.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(groups * kWorkgroup),
                          sycl::range<1>(kWorkgroup)),
        [=](sycl::nd_item<1> item)
            [[sycl::reqd_sub_group_size(kSubgroup)]] {
            const sycl::sub_group subgroup = item.get_sub_group();
            const size_t subgroup_id = item.get_local_id(0) / kSubgroup;
            const size_t lane = subgroup.get_local_linear_id();
            const size_t row = item.get_group(0) * kSubgroupsPerWorkgroup +
                               subgroup_id;
            if (row >= rows) return;
            const size_t base = row * cols;
            float sum = 0.0f;
            for (size_t col = lane; col < cols; col += kSubgroup) {
                const float value = input[base + col];
                sum = sycl::fma(value, value, sum);
            }
            sum = sycl::reduce_over_group(subgroup, sum, sycl::plus<float>());
            const float inv = sycl::rsqrt(sum / static_cast<float>(cols) + eps);
            for (size_t col = lane; col < cols; col += kSubgroup) {
                output[base + col] = static_cast<sycl::half>(
                    input[base + col] * weight[col] * inv);
            }
        });
    record_profile(engine, "rms_norm_f16", event);
}

static void launch_float_to_half(xpu_engine *engine, const float *input,
                                 sycl::half *output, size_t count) {
    const sycl::event event = engine->queue.parallel_for(
        sycl::range<1>(count), [=](sycl::id<1> index) {
            output[index[0]] = static_cast<sycl::half>(input[index[0]]);
        });
    record_profile(engine, "float_to_half", event);
}

static void launch_repeat_value_half(xpu_engine *engine,
                                     const float *value, size_t tokens) {
    sycl::half *output = engine->half_input;
    const size_t count = tokens * EI_N_EMBD;
    const sycl::event event = engine->queue.parallel_for(
        sycl::range<1>(count), [=](sycl::id<1> index) {
            const size_t linear = index[0];
            const size_t token = linear / EI_N_EMBD;
            const size_t dim = linear - token * EI_N_EMBD;
            output[linear] = static_cast<sycl::half>(
                value[token * EI_HEAD_DIM + dim % EI_HEAD_DIM]);
        });
    record_profile(engine, "repeat_value_f16", event);
}

static void launch_gemm(xpu_engine *engine, const sycl::half *input,
                        const sycl::half *weights, float *output,
                        size_t tokens, size_t rows, size_t cols,
                        const char *profile_label,
                        const ei_xpu_onednn_weight *q4_weight = nullptr) {
#ifdef EI_XPU_ONEDNN
    if (engine->onednn_f16) {
        const sycl::event event = ei_xpu_onednn_dense_matmul(
            engine->onednn, input, weights, output, tokens, rows, cols);
        record_profile(engine, profile_label, event);
        return;
    }
    if (engine->onednn_w4 && q4_weight && q4_weight->values) {
        const sycl::event event = ei_xpu_onednn_matmul(
            engine->onednn, input, *q4_weight, output, tokens);
        record_profile(engine, profile_label, event);
        return;
    }
#else
    (void)q4_weight;
#endif
    const sycl::event event = oneapi::mkl::blas::row_major::gemm(
        engine->queue, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, static_cast<int64_t>(tokens),
        static_cast<int64_t>(rows), static_cast<int64_t>(cols), 1.0f,
        input, static_cast<int64_t>(cols), weights,
        static_cast<int64_t>(cols), 0.0f, output,
        static_cast<int64_t>(rows));
    record_profile(engine, profile_label, event);
}

static void launch_rms_residual(xpu_engine *engine, const float *projection,
                                const float *weight, float *residual,
                                size_t rows, size_t cols, float eps) {
    const size_t groups = (rows + kSubgroupsPerWorkgroup - 1) /
                          kSubgroupsPerWorkgroup;
    const sycl::event event = engine->queue.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(groups * kWorkgroup),
                          sycl::range<1>(kWorkgroup)),
        [=](sycl::nd_item<1> item)
            [[sycl::reqd_sub_group_size(kSubgroup)]] {
            const sycl::sub_group subgroup = item.get_sub_group();
            const size_t subgroup_id = item.get_local_id(0) / kSubgroup;
            const size_t lane = subgroup.get_local_linear_id();
            const size_t row = item.get_group(0) * kSubgroupsPerWorkgroup +
                               subgroup_id;
            if (row >= rows) return;
            const size_t base = row * cols;
            float sum = 0.0f;
            for (size_t col = lane; col < cols; col += kSubgroup) {
                const float value = projection[base + col];
                sum = sycl::fma(value, value, sum);
            }
            sum = sycl::reduce_over_group(subgroup, sum, sycl::plus<float>());
            const float inv = sycl::rsqrt(sum / static_cast<float>(cols) + eps);
            for (size_t col = lane; col < cols; col += kSubgroup) {
                residual[base + col] +=
                    projection[base + col] * weight[col] * inv;
            }
        });
    record_profile(engine, "rms_residual", event);
}

static void launch_rms_residual_next_half(
    xpu_engine *engine, const float *projection, const float *post_weight,
    float *residual, const float *next_weight, sycl::half *next_output,
    size_t rows, size_t cols, float eps) {
    if (rows > 1 && rows <= engine->cooperative_rms_max_rows &&
        engine->rms_register_cache && cols == EI_N_EMBD) {
        const sycl::event event = engine->queue.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(rows * kWorkgroup),
                              sycl::range<1>(kWorkgroup)),
            [=](sycl::nd_item<1> item) {
                const size_t row = item.get_group(0);
                const size_t lane = item.get_local_id(0);
                const size_t base = row * EI_N_EMBD;
                constexpr size_t values_per_lane = EI_N_EMBD / kWorkgroup;
                float updated[values_per_lane];
                float projected_sum = 0.0f;
                for (size_t slot = 0; slot < values_per_lane; slot++) {
                    const size_t col = lane + slot * kWorkgroup;
                    const float value = projection[base + col];
                    projected_sum = sycl::fma(value, value, projected_sum);
                }
                projected_sum = sycl::reduce_over_group(
                    item.get_group(), projected_sum, sycl::plus<float>());
                const float projected_inv = sycl::rsqrt(
                    projected_sum / static_cast<float>(EI_N_EMBD) + eps);
                float residual_sum = 0.0f;
                for (size_t slot = 0; slot < values_per_lane; slot++) {
                    const size_t col = lane + slot * kWorkgroup;
                    const float value = residual[base + col] +
                        projection[base + col] * post_weight[col] *
                            projected_inv;
                    updated[slot] = value;
                    residual[base + col] = value;
                    residual_sum = sycl::fma(value, value, residual_sum);
                }
                residual_sum = sycl::reduce_over_group(
                    item.get_group(), residual_sum, sycl::plus<float>());
                const float residual_inv = sycl::rsqrt(
                    residual_sum / static_cast<float>(EI_N_EMBD) + eps);
                for (size_t slot = 0; slot < values_per_lane; slot++) {
                    const size_t col = lane + slot * kWorkgroup;
                    next_output[base + col] = static_cast<sycl::half>(
                        updated[slot] * next_weight[col] * residual_inv);
                }
            });
        record_profile(
            engine, "rms_residual_next_f16_cooperative_register", event);
        return;
    }
    if (rows <= engine->cooperative_rms_max_rows) {
        const sycl::event event = engine->queue.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(rows * kWorkgroup),
                              sycl::range<1>(kWorkgroup)),
            [=](sycl::nd_item<1> item) {
                const size_t row = item.get_group(0);
                const size_t lane = item.get_local_id(0);
                const size_t base = row * cols;
                float projected_sum = 0.0f;
                for (size_t col = lane; col < cols; col += kWorkgroup) {
                    const float value = projection[base + col];
                    projected_sum =
                        sycl::fma(value, value, projected_sum);
                }
                projected_sum = sycl::reduce_over_group(
                    item.get_group(), projected_sum, sycl::plus<float>());
                const float projected_inv = sycl::rsqrt(
                    projected_sum / static_cast<float>(cols) + eps);
                float residual_sum = 0.0f;
                for (size_t col = lane; col < cols; col += kWorkgroup) {
                    const float value = residual[base + col] +
                        projection[base + col] * post_weight[col] *
                            projected_inv;
                    residual[base + col] = value;
                    residual_sum = sycl::fma(value, value, residual_sum);
                }
                residual_sum = sycl::reduce_over_group(
                    item.get_group(), residual_sum, sycl::plus<float>());
                const float residual_inv = sycl::rsqrt(
                    residual_sum / static_cast<float>(cols) + eps);
                for (size_t col = lane; col < cols; col += kWorkgroup) {
                    next_output[base + col] = static_cast<sycl::half>(
                        residual[base + col] * next_weight[col] *
                            residual_inv);
                }
            });
        record_profile(engine, "rms_residual_next_f16_cooperative", event);
        return;
    }
    if (engine->rms_register_cache && cols == EI_N_EMBD) {
        const size_t groups = (rows + kSubgroupsPerWorkgroup - 1) /
                              kSubgroupsPerWorkgroup;
        const sycl::event event = engine->queue.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(groups * kWorkgroup),
                              sycl::range<1>(kWorkgroup)),
            [=](sycl::nd_item<1> item)
                [[sycl::reqd_sub_group_size(kSubgroup)]] {
                const sycl::sub_group subgroup = item.get_sub_group();
                const size_t subgroup_id = item.get_local_id(0) / kSubgroup;
                const size_t lane = subgroup.get_local_linear_id();
                const size_t row = item.get_group(0) * kSubgroupsPerWorkgroup +
                                   subgroup_id;
                if (row >= rows) return;
                const size_t base = row * EI_N_EMBD;
                constexpr size_t values_per_lane = EI_N_EMBD / kSubgroup;
                float updated[values_per_lane];

                float projected_sum = 0.0f;
                for (size_t slot = 0; slot < values_per_lane; slot++) {
                    const size_t col = lane + slot * kSubgroup;
                    const float value = projection[base + col];
                    projected_sum = sycl::fma(value, value, projected_sum);
                }
                projected_sum = sycl::reduce_over_group(
                    subgroup, projected_sum, sycl::plus<float>());
                const float projected_inv = sycl::rsqrt(
                    projected_sum / static_cast<float>(EI_N_EMBD) + eps);

                float residual_sum = 0.0f;
                for (size_t slot = 0; slot < values_per_lane; slot++) {
                    const size_t col = lane + slot * kSubgroup;
                    const float value = residual[base + col] +
                        projection[base + col] * post_weight[col] *
                            projected_inv;
                    updated[slot] = value;
                    residual[base + col] = value;
                    residual_sum = sycl::fma(value, value, residual_sum);
                }
                residual_sum = sycl::reduce_over_group(
                    subgroup, residual_sum, sycl::plus<float>());
                const float residual_inv = sycl::rsqrt(
                    residual_sum / static_cast<float>(EI_N_EMBD) + eps);
                for (size_t slot = 0; slot < values_per_lane; slot++) {
                    const size_t col = lane + slot * kSubgroup;
                    next_output[base + col] = static_cast<sycl::half>(
                        updated[slot] * next_weight[col] * residual_inv);
                }
            });
        record_profile(engine, "rms_residual_next_f16_register", event);
        return;
    }
    const size_t groups = (rows + kSubgroupsPerWorkgroup - 1) /
                          kSubgroupsPerWorkgroup;
    const sycl::event event = engine->queue.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(groups * kWorkgroup),
                          sycl::range<1>(kWorkgroup)),
        [=](sycl::nd_item<1> item)
            [[sycl::reqd_sub_group_size(kSubgroup)]] {
            const sycl::sub_group subgroup = item.get_sub_group();
            const size_t subgroup_id = item.get_local_id(0) / kSubgroup;
            const size_t lane = subgroup.get_local_linear_id();
            const size_t row = item.get_group(0) * kSubgroupsPerWorkgroup +
                               subgroup_id;
            if (row >= rows) return;
            const size_t base = row * cols;
            float projected_sum = 0.0f;
            for (size_t col = lane; col < cols; col += kSubgroup) {
                const float value = projection[base + col];
                projected_sum = sycl::fma(value, value, projected_sum);
            }
            projected_sum = sycl::reduce_over_group(
                subgroup, projected_sum, sycl::plus<float>());
            const float projected_inv = sycl::rsqrt(
                projected_sum / static_cast<float>(cols) + eps);

            float residual_sum = 0.0f;
            for (size_t col = lane; col < cols; col += kSubgroup) {
                const float value = residual[base + col] +
                    projection[base + col] * post_weight[col] * projected_inv;
                residual[base + col] = value;
                residual_sum = sycl::fma(value, value, residual_sum);
            }
            residual_sum = sycl::reduce_over_group(
                subgroup, residual_sum, sycl::plus<float>());
            const float residual_inv = sycl::rsqrt(
                residual_sum / static_cast<float>(cols) + eps);
            for (size_t col = lane; col < cols; col += kSubgroup) {
                next_output[base + col] = static_cast<sycl::half>(
                    residual[base + col] * next_weight[col] * residual_inv);
            }
        });
    record_profile(engine, "rms_residual_next_f16", event);
}

static void launch_q4_projection(xpu_engine *engine, const ei_tensor *tensor,
                                 const float *input, float *output,
                                 size_t tokens) {
    const uint8_t *weights = tensor_bytes(engine, tensor);
    const size_t cols = static_cast<size_t>(tensor->ne[0]);
    const size_t rows = static_cast<size_t>(tensor->ne[1]);
    const size_t blocks_per_row = cols / kQK;
    const size_t row_bytes = blocks_per_row * kQ4BlockBytes;
    if (engine->q4_m_tiled && tokens <= 8) {
        const size_t groups =
            (rows + kSubgroupsPerWorkgroup - 1) / kSubgroupsPerWorkgroup;
        const sycl::event event = engine->queue.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(groups * kWorkgroup),
                              sycl::range<1>(kWorkgroup)),
            [=](sycl::nd_item<1> item)
                [[sycl::reqd_sub_group_size(kSubgroup)]] {
                const sycl::sub_group subgroup = item.get_sub_group();
                const size_t subgroup_id =
                    item.get_local_id(0) / kSubgroup;
                const size_t lane = subgroup.get_local_linear_id();
                const size_t row =
                    item.get_group(0) * kSubgroupsPerWorkgroup + subgroup_id;
                if (row >= rows) return;
                const uint8_t *weight_row = weights + row * row_bytes;
                float sums[8] = {};
                for (size_t block = lane; block < blocks_per_row;
                     block += kSubgroup) {
                    const uint8_t *packed =
                        weight_row + block * kQ4BlockBytes;
                    const float d = load_half(packed);
                    const uint8_t *quants = packed + 2;
                    const size_t input_base = block * kQK;
#pragma unroll
                    for (size_t quant = 0; quant < 16; quant++) {
                        const uint8_t code = quants[quant];
                        const float low = d *
                            static_cast<float>((code & 0x0f) - 8);
                        const float high = d *
                            static_cast<float>((code >> 4) - 8);
                        for (size_t token = 0; token < tokens; token++) {
                            const float *input_row = input + token * cols;
                            sums[token] = sycl::fma(
                                low, input_row[input_base + quant],
                                sums[token]);
                            sums[token] = sycl::fma(
                                high, input_row[input_base + 16 + quant],
                                sums[token]);
                        }
                    }
                }
                for (size_t token = 0; token < tokens; token++) {
                    const float sum = sycl::reduce_over_group(
                        subgroup, sums[token], sycl::plus<float>());
                    if (lane == 0) output[token * rows + row] = sum;
                }
            });
        record_profile(engine, "projection_q4_m_tiled", event);
        return;
    }
    const size_t tasks = tokens * rows;
    const size_t groups = (tasks + kSubgroupsPerWorkgroup - 1) /
                          kSubgroupsPerWorkgroup;
    const sycl::event event = engine->queue.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(groups * kWorkgroup),
                          sycl::range<1>(kWorkgroup)),
        [=](sycl::nd_item<1> item)
            [[sycl::reqd_sub_group_size(kSubgroup)]] {
            const sycl::sub_group subgroup = item.get_sub_group();
            const size_t subgroup_id = item.get_local_id(0) / kSubgroup;
            const size_t lane = subgroup.get_local_linear_id();
            const size_t task = item.get_group(0) * kSubgroupsPerWorkgroup +
                                subgroup_id;
            if (task >= tasks) return;
            const size_t token = task / rows;
            const size_t row = task - token * rows;
            const uint8_t *weight_row = weights + row * row_bytes;
            const float *input_row = input + token * cols;
            float sum = 0.0f;
            for (size_t block = lane; block < blocks_per_row;
                 block += kSubgroup) {
                const uint8_t *packed =
                    weight_row + block * kQ4BlockBytes;
                const float d = load_half(packed);
                const uint8_t *quants = packed + 2;
                const size_t input_base = block * kQK;
#pragma unroll
                for (size_t quant = 0; quant < 16; quant++) {
                    const uint8_t code = quants[quant];
                    sum = sycl::fma(
                        d * static_cast<float>((code & 0x0f) - 8),
                        input_row[input_base + quant], sum);
                    sum = sycl::fma(
                        d * static_cast<float>((code >> 4) - 8),
                        input_row[input_base + 16 + quant], sum);
                }
            }
            sum = sycl::reduce_over_group(subgroup, sum, sycl::plus<float>());
            if (lane == 0) output[token * rows + row] = sum;
        });
    record_profile(engine, "projection_q4", event);
}

static void launch_qk_norm_rope(xpu_engine *engine, const ei_layer *layer,
                                const float *rope, size_t tokens) {
    constexpr size_t combined_heads = EI_N_HEAD + EI_N_HEAD_KV;
    const size_t tasks = tokens * combined_heads;
    const size_t groups = (tasks + kSubgroupsPerWorkgroup - 1) /
                          kSubgroupsPerWorkgroup;
    float *query = engine->q;
    float *key = engine->k;
    const float *q_weight = float_weights(engine, layer->attn_q_norm);
    const float *k_weight = float_weights(engine, layer->attn_k_norm);
    const uint32_t *positions = engine->positions;
    const float eps = engine->model->rms_eps;
    const sycl::event event = engine->queue.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(groups * kWorkgroup),
                          sycl::range<1>(kWorkgroup)),
        [=](sycl::nd_item<1> item)
            [[sycl::reqd_sub_group_size(kSubgroup)]] {
            const sycl::sub_group subgroup = item.get_sub_group();
            const size_t subgroup_id = item.get_local_id(0) / kSubgroup;
            const size_t lane = subgroup.get_local_linear_id();
            const size_t task = item.get_group(0) * kSubgroupsPerWorkgroup +
                                subgroup_id;
            if (task >= tasks) return;
            const size_t token = task / combined_heads;
            const size_t combined_head = task - token * combined_heads;
            const bool is_key = combined_head == EI_N_HEAD;
            const size_t head = is_key ? 0 : combined_head;
            const size_t stride = is_key ? EI_HEAD_DIM : EI_N_EMBD;
            float *values = is_key ? key : query;
            const float *weight = is_key ? k_weight : q_weight;
            const size_t base = token * stride + head * EI_HEAD_DIM;
            float sum = 0.0f;
            for (size_t dim = lane; dim < EI_HEAD_DIM; dim += kSubgroup) {
                const float value = values[base + dim];
                sum = sycl::fma(value, value, sum);
            }
            sum = sycl::reduce_over_group(subgroup, sum, sycl::plus<float>());
            const float query_scale = is_key ? 1.0f : 0.0625f;
            const float inv = sycl::rsqrt(
                sum / static_cast<float>(EI_HEAD_DIM) + eps) * query_scale;
            const size_t rope_base =
                static_cast<size_t>(positions[token]) * EI_HEAD_DIM;
            for (size_t dim = lane; dim < EI_HEAD_DIM / 2;
                 dim += kSubgroup) {
                const float cosine = rope[rope_base + 2 * dim];
                const float sine = rope[rope_base + 2 * dim + 1];
                const float x0 = values[base + dim] * weight[dim] * inv;
                const float x1 = values[base + dim + EI_HEAD_DIM / 2] *
                                 weight[dim + EI_HEAD_DIM / 2] * inv;
                values[base + dim] = sycl::fma(-x1, sine, x0 * cosine);
                values[base + dim + EI_HEAD_DIM / 2] =
                    sycl::fma(x0, sine, x1 * cosine);
            }
        });
    record_profile(engine, "qk_norm_rope", event);
}

static void launch_qkv_epilogue(xpu_engine *engine, const ei_layer *layer,
                                const float *rope, size_t tokens) {
    constexpr size_t tasks_per_token = EI_N_HEAD + EI_N_HEAD_KV + 1;
    constexpr size_t combined_stride = EI_N_EMBD + 2 * EI_HEAD_DIM;
    const size_t tasks = tokens * tasks_per_token;
    const size_t groups = (tasks + kSubgroupsPerWorkgroup - 1) /
                          kSubgroupsPerWorkgroup;
    const float *combined = engine->qkv_combined;
    float *query = engine->q;
    float *key = engine->k;
    float *value = engine->v;
    sycl::half *query_half = engine->q_half;
    sycl::half *key_half = engine->k_half;
    sycl::half *value_half = engine->v_half;
    const bool use_half = engine->fp16_attention;
    const float *q_weight = float_weights(engine, layer->attn_q_norm);
    const float *k_weight = float_weights(engine, layer->attn_k_norm);
    const uint32_t *positions = engine->positions;
    const float eps = engine->model->rms_eps;
    const sycl::event event = engine->queue.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(groups * kWorkgroup),
                          sycl::range<1>(kWorkgroup)),
        [=](sycl::nd_item<1> item)
            [[sycl::reqd_sub_group_size(kSubgroup)]] {
            const sycl::sub_group subgroup = item.get_sub_group();
            const size_t subgroup_id = item.get_local_id(0) / kSubgroup;
            const size_t lane = subgroup.get_local_linear_id();
            const size_t task = item.get_group(0) * kSubgroupsPerWorkgroup +
                                subgroup_id;
            if (task >= tasks) return;
            const size_t token = task / tasks_per_token;
            const size_t kind = task - token * tasks_per_token;
            const size_t input_base = token * combined_stride;
            if (kind == tasks_per_token - 1) {
                const size_t source = input_base + EI_N_EMBD + EI_HEAD_DIM;
                const size_t destination = token * EI_HEAD_DIM;
                for (size_t dim = lane; dim < EI_HEAD_DIM;
                     dim += kSubgroup) {
                    if (use_half) {
                        value_half[destination + dim] =
                            static_cast<sycl::half>(combined[source + dim]);
                    } else {
                        value[destination + dim] = combined[source + dim];
                    }
                }
                return;
            }

            const bool is_key = kind == EI_N_HEAD;
            const size_t head = is_key ? 0 : kind;
            const size_t source = input_base +
                (is_key ? EI_N_EMBD : head * EI_HEAD_DIM);
            const size_t output_stride = is_key ? EI_HEAD_DIM : EI_N_EMBD;
            const size_t destination = token * output_stride +
                                       head * EI_HEAD_DIM;
            float *output = is_key ? key : query;
            sycl::half *output_half = is_key ? key_half : query_half;
            const float *weight = is_key ? k_weight : q_weight;
            float sum = 0.0f;
            for (size_t dim = lane; dim < EI_HEAD_DIM; dim += kSubgroup) {
                const float item_value = combined[source + dim];
                sum = sycl::fma(item_value, item_value, sum);
            }
            sum = sycl::reduce_over_group(subgroup, sum, sycl::plus<float>());
            const float query_scale = is_key ? 1.0f : 0.0625f;
            const float inv = sycl::rsqrt(
                sum / static_cast<float>(EI_HEAD_DIM) + eps) * query_scale;
            const size_t rope_base =
                static_cast<size_t>(positions[token]) * EI_HEAD_DIM;
            for (size_t dim = lane; dim < EI_HEAD_DIM / 2;
                 dim += kSubgroup) {
                const float cosine = rope[rope_base + 2 * dim];
                const float sine = rope[rope_base + 2 * dim + 1];
                const float x0 = combined[source + dim] * weight[dim] * inv;
                const float x1 = combined[source + dim + EI_HEAD_DIM / 2] *
                                 weight[dim + EI_HEAD_DIM / 2] * inv;
                const float rotated0 = sycl::fma(-x1, sine, x0 * cosine);
                const float rotated1 = sycl::fma(x0, sine, x1 * cosine);
                if (use_half) {
                    output_half[destination + dim] =
                        static_cast<sycl::half>(rotated0);
                    output_half[destination + dim + EI_HEAD_DIM / 2] =
                        static_cast<sycl::half>(rotated1);
                } else {
                    output[destination + dim] = rotated0;
                    output[destination + dim + EI_HEAD_DIM / 2] = rotated1;
                }
            }
        });
    record_profile(engine, "qkv_epilogue", event);
}

static void launch_attention(xpu_engine *engine, size_t tokens,
                             uint32_t window, uint32_t tensor_min_tokens,
                             sycl::half *output_half = nullptr) {
    constexpr size_t attention_subgroups = 4;
    constexpr size_t attention_workgroup = kSubgroup * attention_subgroups;
    const size_t tasks = tokens * EI_N_HEAD;
    const size_t groups = (tasks + attention_subgroups - 1) /
                          attention_subgroups;
    const float *query = engine->q;
    const float *key = engine->k;
    const float *value = engine->v;
    const sycl::half *query_half = engine->q_half;
    const sycl::half *key_half = engine->k_half;
    const sycl::half *value_half = engine->v_half;
    const bool use_half = engine->fp16_attention;
    float *output = engine->ctx;
    sycl::half *output_half_buffer = output_half;
    const uint32_t *offsets = engine->offsets;
    const uint32_t *sequence_ids = engine->sequence_ids;
    const sycl::event event = engine->queue.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(groups * attention_workgroup),
                          sycl::range<1>(attention_workgroup)),
        [=](sycl::nd_item<1> item)
            [[sycl::reqd_sub_group_size(kSubgroup)]] {
            const sycl::sub_group subgroup = item.get_sub_group();
            const size_t subgroup_id = item.get_local_id(0) / kSubgroup;
            const size_t lane = subgroup.get_local_linear_id();
            const size_t task = item.get_group(0) * attention_subgroups +
                                subgroup_id;
            if (task >= tasks) return;
            const size_t query_token = task / EI_N_HEAD;
            const size_t head = task - query_token * EI_N_HEAD;

            float q_values[EI_HEAD_DIM / kSubgroup];
            float out_values[EI_HEAD_DIM / kSubgroup];
#pragma unroll
            for (size_t slot = 0; slot < EI_HEAD_DIM / kSubgroup; slot++) {
                const size_t dim = lane + slot * kSubgroup;
                const size_t query_index = query_token * EI_N_EMBD +
                                           head * EI_HEAD_DIM + dim;
                q_values[slot] = use_half
                    ? static_cast<float>(query_half[query_index])
                    : query[query_index];
                out_values[slot] = 0.0f;
            }

            const uint32_t sequence = sequence_ids[query_token];
            const uint32_t sequence_start = offsets[sequence];
            const uint32_t sequence_stop = offsets[sequence + 1];
            if (sequence_stop - sequence_start >= tensor_min_tokens) return;
            uint32_t first = sequence_start;
            uint32_t last = sequence_stop;
            if (window != 0) {
                const uint32_t half_window = window / 2;
                first = query_token > sequence_start + half_window
                    ? static_cast<uint32_t>(query_token) - half_window
                    : sequence_start;
                const uint32_t candidate =
                    static_cast<uint32_t>(query_token) + half_window + 1;
                last = candidate < sequence_stop ? candidate : sequence_stop;
            }

            float max_score = -3.402823466e+38F;
            float denominator = 0.0f;
            for (uint32_t key_token = first; key_token < last; key_token++) {
                float score = 0.0f;
#pragma unroll
                for (size_t slot = 0; slot < EI_HEAD_DIM / kSubgroup; slot++) {
                    const size_t dim = lane + slot * kSubgroup;
                    const size_t key_index =
                        static_cast<size_t>(key_token) * EI_HEAD_DIM + dim;
                    const float key_value = use_half
                        ? static_cast<float>(key_half[key_index])
                        : key[key_index];
                    score = sycl::fma(q_values[slot], key_value, score);
                }
                score = sycl::reduce_over_group(
                    subgroup, score, sycl::plus<float>());
                const float next_max = sycl::fmax(max_score, score);
                const float alpha = sycl::exp(max_score - next_max);
                const float beta = sycl::exp(score - next_max);
                denominator = denominator * alpha + beta;
#pragma unroll
                for (size_t slot = 0; slot < EI_HEAD_DIM / kSubgroup; slot++) {
                    const size_t dim = lane + slot * kSubgroup;
                    const size_t value_index =
                        static_cast<size_t>(key_token) * EI_HEAD_DIM + dim;
                    const float value_item = use_half
                        ? static_cast<float>(value_half[value_index])
                        : value[value_index];
                    out_values[slot] = sycl::fma(
                        beta, value_item, out_values[slot] * alpha);
                }
                max_score = next_max;
            }
            const float inv_denominator = 1.0f / denominator;
#pragma unroll
            for (size_t slot = 0; slot < EI_HEAD_DIM / kSubgroup; slot++) {
                const size_t dim = lane + slot * kSubgroup;
                const size_t index =
                    query_token * EI_N_EMBD + head * EI_HEAD_DIM + dim;
                const float result = out_values[slot] * inv_denominator;
                output[index] = result;
                // Fuse the ctx->half convert here so the attention-output GEMM
                // reads half_input directly, skipping a separate submission.
                if (output_half_buffer) {
                    output_half_buffer[index] =
                        static_cast<sycl::half>(result);
                }
            }
        });
    record_profile(engine, "attention_online", event);
}

static void launch_attention_softmax(xpu_engine *engine, size_t tokens,
                                     uint32_t window) {
    constexpr size_t local_size = 256;
    const size_t rows = EI_N_HEAD * tokens;
    float *scores = engine->attention_scores;
    sycl::half *probabilities = engine->attention_probabilities;
    const bool use_half = engine->fp16_attention;
    const sycl::event event = engine->queue.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(rows * local_size),
                          sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) {
            const size_t row = item.get_group(0);
            const size_t lane = item.get_local_id(0);
            const size_t query = row % tokens;
            size_t first = 0;
            size_t last = tokens;
            if (window != 0) {
                const size_t half_window = window / 2;
                first = query > half_window ? query - half_window : 0;
                last = std::min(tokens, query + half_window + 1);
            }
            float maximum = -3.402823466e+38F;
            const size_t base = row * tokens;
            for (size_t key = first + lane; key < last; key += local_size) {
                maximum = sycl::fmax(maximum, scores[base + key]);
            }
            maximum = sycl::reduce_over_group(
                item.get_group(), maximum, sycl::maximum<float>());
            float sum = 0.0f;
            for (size_t key = first + lane; key < last; key += local_size) {
                const float probability = sycl::exp(scores[base + key] - maximum);
                scores[base + key] = probability;
                sum += probability;
            }
            sum = sycl::reduce_over_group(
                item.get_group(), sum, sycl::plus<float>());
            const float inv = 1.0f / sum;
            for (size_t key = lane; key < tokens; key += local_size) {
                const float probability = key >= first && key < last
                    ? scores[base + key] * inv : 0.0f;
                if (use_half) {
                    probabilities[base + key] =
                        static_cast<sycl::half>(probability);
                } else {
                    scores[base + key] = probability;
                }
            }
        });
    record_profile(engine, "attention_softmax", event);
}

static void launch_attention_softmax_banded(
    xpu_engine *engine, uint32_t query_start, uint32_t key_start,
    uint32_t query_count, uint32_t key_count, uint32_t sequence_tokens,
    uint32_t window) {
    constexpr size_t local_size = 256;
    const size_t rows = EI_N_HEAD * static_cast<size_t>(query_count);
    float *scores = engine->attention_scores;
    sycl::half *probabilities = engine->attention_probabilities;
    const bool use_half = engine->fp16_attention;
    const sycl::event event = engine->queue.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(rows * local_size),
                          sycl::range<1>(local_size)),
        [=](sycl::nd_item<1> item) {
            const size_t row = item.get_group(0);
            const size_t lane = item.get_local_id(0);
            const size_t head = row / query_count;
            const uint32_t local_query =
                static_cast<uint32_t>(row - head * query_count);
            const uint32_t query = query_start + local_query;
            const uint32_t half_window = window / 2;
            const uint32_t first = query > half_window
                ? query - half_window : 0;
            const uint32_t last = sycl::min(
                sequence_tokens, query + half_window + 1);
            const uint32_t local_first = first - key_start;
            const uint32_t local_last = last - key_start;
            const size_t score_stride =
                static_cast<size_t>(query_count) * key_count;
            const size_t base = head * score_stride +
                static_cast<size_t>(local_query) * key_count;
            float maximum = -3.402823466e+38F;
            for (uint32_t key = local_first + static_cast<uint32_t>(lane);
                 key < local_last; key += local_size) {
                maximum = sycl::fmax(maximum, scores[base + key]);
            }
            maximum = sycl::reduce_over_group(
                item.get_group(), maximum, sycl::maximum<float>());
            float sum = 0.0f;
            for (uint32_t key = local_first + static_cast<uint32_t>(lane);
                 key < local_last; key += local_size) {
                const float probability =
                    sycl::exp(scores[base + key] - maximum);
                scores[base + key] = probability;
                sum += probability;
            }
            sum = sycl::reduce_over_group(
                item.get_group(), sum, sycl::plus<float>());
            const float inv = 1.0f / sum;
            for (uint32_t key = static_cast<uint32_t>(lane);
                 key < key_count; key += local_size) {
                const float probability =
                    key >= local_first && key < local_last
                    ? scores[base + key] * inv : 0.0f;
                if (use_half) {
                    probabilities[base + key] =
                        static_cast<sycl::half>(probability);
                } else {
                    scores[base + key] = probability;
                }
            }
        });
    record_profile(engine, "attention_softmax_banded", event);
}

static void launch_banded_tensor_attention(xpu_engine *engine,
                                           size_t start, size_t tokens,
                                           uint32_t window) {
    const uint32_t token_count = static_cast<uint32_t>(tokens);
    const uint32_t query_tile = engine->swa_tensor_tile_tokens;
    for (uint32_t query_start = 0; query_start < token_count;
         query_start += query_tile) {
        const uint32_t query_count =
            std::min(query_tile, token_count - query_start);
        const uint32_t half_window = window / 2;
        const uint32_t key_start = query_start > half_window
            ? query_start - half_window : 0;
        const uint32_t key_stop = std::min(
            token_count, query_start + query_count + half_window);
        const uint32_t key_count = key_stop - key_start;
        const int64_t score_stride =
            static_cast<int64_t>(query_count) * key_count;
        for (int head = 0; head < EI_N_HEAD; head++) {
            const sycl::event event = engine->fp16_attention
                ? oneapi::mkl::blas::row_major::gemm(
                    engine->queue, oneapi::mkl::transpose::nontrans,
                    oneapi::mkl::transpose::trans, query_count, key_count,
                    EI_HEAD_DIM, 1.0f,
                    engine->q_half +
                        (start + query_start) * EI_N_EMBD +
                        head * EI_HEAD_DIM,
                    EI_N_EMBD,
                    engine->k_half + (start + key_start) * EI_HEAD_DIM,
                    EI_HEAD_DIM, 0.0f,
                    engine->attention_scores + head * score_stride,
                    key_count)
                : oneapi::mkl::blas::row_major::gemm(
                    engine->queue, oneapi::mkl::transpose::nontrans,
                    oneapi::mkl::transpose::trans, query_count, key_count,
                    EI_HEAD_DIM, 1.0f,
                    engine->q + (start + query_start) * EI_N_EMBD +
                        head * EI_HEAD_DIM,
                    EI_N_EMBD,
                    engine->k + (start + key_start) * EI_HEAD_DIM,
                    EI_HEAD_DIM, 0.0f,
                    engine->attention_scores + head * score_stride,
                    key_count);
            record_profile(engine, "attention_qk_banded_gemm", event);
        }
        launch_attention_softmax_banded(
            engine, query_start, key_start, query_count, key_count,
            token_count, window);
        for (int head = 0; head < EI_N_HEAD; head++) {
            float *output = engine->ctx +
                (start + query_start) * EI_N_EMBD + head * EI_HEAD_DIM;
            const sycl::event event = engine->fp16_attention
                ? oneapi::mkl::blas::row_major::gemm(
                    engine->queue, oneapi::mkl::transpose::nontrans,
                    oneapi::mkl::transpose::nontrans, query_count,
                    EI_HEAD_DIM, key_count, 1.0f,
                    engine->attention_probabilities + head * score_stride,
                    key_count,
                    engine->v_half + (start + key_start) * EI_HEAD_DIM,
                    EI_HEAD_DIM, 0.0f, output, EI_N_EMBD)
                : oneapi::mkl::blas::row_major::gemm(
                    engine->queue, oneapi::mkl::transpose::nontrans,
                    oneapi::mkl::transpose::nontrans, query_count,
                    EI_HEAD_DIM, key_count, 1.0f,
                    engine->attention_scores + head * score_stride,
                    key_count,
                    engine->v + (start + key_start) * EI_HEAD_DIM,
                    EI_HEAD_DIM, 0.0f, output, EI_N_EMBD);
            record_profile(engine, "attention_pv_banded_gemm", event);
        }
    }
}

static void launch_tensor_attention(xpu_engine *engine,
                                    const std::vector<uint32_t> &offsets,
                                    uint32_t window,
                                    uint32_t tensor_min_tokens) {
    for (size_t sequence = 0; sequence + 1 < offsets.size(); sequence++) {
        const size_t start = offsets[sequence];
        const size_t tokens = offsets[sequence + 1] - start;
        if (tokens < tensor_min_tokens) continue;
        if (window != 0 && engine->swa_tensor_tile_tokens != 0 &&
            tokens >= engine->swa_tensor_banded_min_tokens) {
            launch_banded_tensor_attention(engine, start, tokens, window);
            continue;
        }
        const int64_t token_count = static_cast<int64_t>(tokens);
        const int64_t score_stride = token_count * token_count;
        const float *query = engine->q + start * EI_N_EMBD;
        const float *key = engine->k + start * EI_HEAD_DIM;
        const float *value = engine->v + start * EI_HEAD_DIM;
        const sycl::half *query_half =
            engine->q_half + start * EI_N_EMBD;
        const sycl::half *key_half =
            engine->k_half + start * EI_HEAD_DIM;
        const sycl::half *value_half =
            engine->v_half + start * EI_HEAD_DIM;
        float *output = engine->ctx + start * EI_N_EMBD;
        for (int head = 0; head < EI_N_HEAD; head++) {
            const sycl::event event = engine->fp16_attention
                ? oneapi::mkl::blas::row_major::gemm(
                    engine->queue, oneapi::mkl::transpose::nontrans,
                    oneapi::mkl::transpose::trans, token_count, token_count,
                    EI_HEAD_DIM, 1.0f,
                    query_half + head * EI_HEAD_DIM, EI_N_EMBD,
                    key_half, EI_HEAD_DIM, 0.0f,
                    engine->attention_scores + head * score_stride,
                    token_count)
                : oneapi::mkl::blas::row_major::gemm(
                    engine->queue, oneapi::mkl::transpose::nontrans,
                    oneapi::mkl::transpose::trans, token_count, token_count,
                    EI_HEAD_DIM, 1.0f, query + head * EI_HEAD_DIM, EI_N_EMBD,
                    key, EI_HEAD_DIM, 0.0f,
                    engine->attention_scores + head * score_stride,
                    token_count);
            record_profile(engine, "attention_qk_gemm", event);
        }
        launch_attention_softmax(engine, tokens, window);
        for (int head = 0; head < EI_N_HEAD; head++) {
            const sycl::event event = engine->fp16_attention
                ? oneapi::mkl::blas::row_major::gemm(
                    engine->queue, oneapi::mkl::transpose::nontrans,
                    oneapi::mkl::transpose::nontrans, token_count,
                    EI_HEAD_DIM, token_count, 1.0f,
                    engine->attention_probabilities + head * score_stride,
                    token_count, value_half, EI_HEAD_DIM, 0.0f,
                    output + head * EI_HEAD_DIM, EI_N_EMBD)
                : oneapi::mkl::blas::row_major::gemm(
                    engine->queue, oneapi::mkl::transpose::nontrans,
                    oneapi::mkl::transpose::nontrans, token_count,
                    EI_HEAD_DIM, token_count, 1.0f,
                    engine->attention_scores + head * score_stride,
                    token_count, value, EI_HEAD_DIM, 0.0f,
                    output + head * EI_HEAD_DIM, EI_N_EMBD);
            record_profile(engine, "attention_pv_gemm", event);
        }
    }
}

static void launch_gelu_mul(xpu_engine *engine, size_t tokens) {
    float *gate = engine->gate;
    const float *up = engine->up;
    const size_t count = tokens * EI_N_FF;
    const sycl::event event = engine->queue.parallel_for(
        sycl::range<1>(count), [=](sycl::id<1> index) {
        const size_t i = index[0];
        const float value = gate[i];
        constexpr float coefficient = 0.044715f;
        constexpr float scale = 0.79788456080286535587989211986876f;
        const float activated = 0.5f * value *
            (1.0f + sycl::tanh(scale * value *
                               (1.0f + coefficient * value * value)));
        gate[i] = activated * up[i];
        });
    record_profile(engine, "gelu_mul", event);
}

static void launch_up_gate_gelu_half(xpu_engine *engine, size_t tokens) {
    const float *combined = engine->up_gate_combined;
    sycl::half *output = engine->half_input;
    const size_t count = tokens * EI_N_FF;
    const sycl::event event = engine->queue.parallel_for(
        sycl::range<1>(count), [=](sycl::id<1> index) {
        const size_t linear = index[0];
        const size_t token = linear / EI_N_FF;
        const size_t row = linear - token * EI_N_FF;
        const size_t base = token * (2 * EI_N_FF);
        const float up = combined[base + row];
        const float gate = combined[base + EI_N_FF + row];
        constexpr float coefficient = 0.044715f;
        constexpr float scale = 0.79788456080286535587989211986876f;
        const float activated = 0.5f * gate *
            (1.0f + sycl::tanh(scale * gate *
                               (1.0f + coefficient * gate * gate)));
        output[linear] = static_cast<sycl::half>(activated * up);
        });
    record_profile(engine, "up_gate_gelu_f16", event);
}

#ifdef EI_XPU_XE2_W4
static void launch_xe2_w4_up_gate(xpu_engine *engine,
                                  const ei_xpu_w4_weight &weight,
                                  size_t tokens) {
    const sycl::event gemm = ei_xpu_w4_matmul(
        engine->queue, engine->half_input, weight,
        engine->w4_output, tokens);
    record_profile(engine, "projection_up_gate_gemm", gemm);

    const sycl::half *combined = engine->w4_output;
    sycl::half *output = engine->half_input;
    const size_t count = tokens * EI_N_FF;
    const sycl::event gelu = engine->queue.parallel_for(
        sycl::range<1>(count), [=](sycl::id<1> index) {
            const size_t linear = index[0];
            const size_t token = linear / EI_N_FF;
            const size_t row = linear - token * EI_N_FF;
            const size_t base = token * (2 * EI_N_FF);
            const float up = static_cast<float>(combined[base + row]);
            const float gate = static_cast<float>(
                combined[base + EI_N_FF + row]);
            constexpr float coefficient = 0.044715f;
            constexpr float scale = 0.79788456080286535587989211986876f;
            const float activated = 0.5f * gate *
                (1.0f + sycl::tanh(scale * gate *
                                   (1.0f + coefficient * gate * gate)));
            output[linear] = static_cast<sycl::half>(activated * up);
        });
    record_profile(engine, "up_gate_gelu_f16", gelu);
}
#endif

static void launch_pool(xpu_engine *engine, size_t batch_size,
                        bool cooperative_pool) {
    const float *input = engine->x;
    const float *weight = float_weights(engine, engine->model->output_norm);
    float *output = engine->result;
    const uint32_t *offsets = engine->offsets;
    const float eps = engine->model->rms_eps;
    if (cooperative_pool) {
        const sycl::event event = engine->queue.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(batch_size * kWorkgroup),
                              sycl::range<1>(kWorkgroup)),
            [=](sycl::nd_item<1> item) {
                const size_t sequence = item.get_group(0);
                const size_t lane = item.get_local_id(0);
                const uint32_t start = offsets[sequence];
                const uint32_t stop = offsets[sequence + 1];
                float pooled[EI_N_EMBD / kWorkgroup];
#pragma unroll
                for (size_t slot = 0; slot < EI_N_EMBD / kWorkgroup;
                     slot++) {
                    pooled[slot] = 0.0f;
                }
                for (uint32_t token = start; token < stop; token++) {
                    float sum = 0.0f;
#pragma unroll
                    for (size_t slot = 0; slot < EI_N_EMBD / kWorkgroup;
                         slot++) {
                        const size_t dim = lane + slot * kWorkgroup;
                        const float value =
                            input[static_cast<size_t>(token) * EI_N_EMBD + dim];
                        sum = sycl::fma(value, value, sum);
                    }
                    sum = sycl::reduce_over_group(
                        item.get_group(), sum, sycl::plus<float>());
                    const float inv = sycl::rsqrt(
                        sum / static_cast<float>(EI_N_EMBD) + eps);
#pragma unroll
                    for (size_t slot = 0; slot < EI_N_EMBD / kWorkgroup;
                         slot++) {
                        const size_t dim = lane + slot * kWorkgroup;
                        pooled[slot] +=
                            input[static_cast<size_t>(token) * EI_N_EMBD + dim] *
                            weight[dim] * inv;
                    }
                }
                const float inv_tokens =
                    1.0f / static_cast<float>(stop - start);
                float sum = 0.0f;
#pragma unroll
                for (size_t slot = 0; slot < EI_N_EMBD / kWorkgroup;
                     slot++) {
                    pooled[slot] *= inv_tokens;
                    sum = sycl::fma(pooled[slot], pooled[slot], sum);
                }
                sum = sycl::reduce_over_group(
                    item.get_group(), sum, sycl::plus<float>());
                const float inv_l2 = sum == 0.0f ? 1.0f : sycl::rsqrt(sum);
#pragma unroll
                for (size_t slot = 0; slot < EI_N_EMBD / kWorkgroup;
                     slot++) {
                    const size_t dim = lane + slot * kWorkgroup;
                    output[sequence * EI_N_EMBD + dim] =
                        pooled[slot] * inv_l2;
                }
            });
        record_profile(engine, "final_norm_pool_cooperative", event);
        return;
    }
    const size_t groups = batch_size;
    const sycl::event event = engine->queue.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(groups * kSubgroup),
                          sycl::range<1>(kSubgroup)),
        [=](sycl::nd_item<1> item)
            [[sycl::reqd_sub_group_size(kSubgroup)]] {
            const sycl::sub_group subgroup = item.get_sub_group();
            const size_t sequence = item.get_group(0);
            const size_t lane = subgroup.get_local_linear_id();
            const uint32_t start = offsets[sequence];
            const uint32_t stop = offsets[sequence + 1];
            float pooled[EI_N_EMBD / kSubgroup];
#pragma unroll
            for (size_t slot = 0; slot < EI_N_EMBD / kSubgroup; slot++) {
                pooled[slot] = 0.0f;
            }
            for (uint32_t token = start; token < stop; token++) {
                float sum = 0.0f;
#pragma unroll
                for (size_t slot = 0; slot < EI_N_EMBD / kSubgroup; slot++) {
                    const size_t dim = lane + slot * kSubgroup;
                    const float value = input[static_cast<size_t>(token) *
                                              EI_N_EMBD + dim];
                    sum = sycl::fma(value, value, sum);
                }
                sum = sycl::reduce_over_group(
                    subgroup, sum, sycl::plus<float>());
                const float inv = sycl::rsqrt(
                    sum / static_cast<float>(EI_N_EMBD) + eps);
#pragma unroll
                for (size_t slot = 0; slot < EI_N_EMBD / kSubgroup; slot++) {
                    const size_t dim = lane + slot * kSubgroup;
                    pooled[slot] += input[static_cast<size_t>(token) *
                                          EI_N_EMBD + dim] * weight[dim] * inv;
                }
            }
            const float inv_tokens = 1.0f / static_cast<float>(stop - start);
            float sum = 0.0f;
#pragma unroll
            for (size_t slot = 0; slot < EI_N_EMBD / kSubgroup; slot++) {
                pooled[slot] *= inv_tokens;
                sum = sycl::fma(pooled[slot], pooled[slot], sum);
            }
            sum = sycl::reduce_over_group(
                subgroup, sum, sycl::plus<float>());
            const float inv_l2 = sum == 0.0f ? 1.0f : sycl::rsqrt(sum);
#pragma unroll
            for (size_t slot = 0; slot < EI_N_EMBD / kSubgroup; slot++) {
                const size_t dim = lane + slot * kSubgroup;
                output[sequence * EI_N_EMBD + dim] =
                    pooled[slot] * inv_l2;
            }
        });
    record_profile(engine, "final_norm_pool", event);
}

static bool execute_batch(xpu_engine *engine, const int32_t *ids,
                          const size_t *offsets, size_t batch_size, float *out,
                          char *err, size_t err_len) {
    const size_t tokens = offsets[batch_size];
    try {
        using clock = std::chrono::steady_clock;
        const clock::time_point total_begin = clock::now();
        engine->profile_samples.clear();
        ensure_capacity(engine, tokens, batch_size);
        std::vector<uint32_t> host_offsets(batch_size + 1);
        std::vector<uint32_t> host_positions(tokens);
        std::vector<uint32_t> host_sequence_ids(tokens);
        for (size_t sequence = 0; sequence < batch_size; sequence++) {
            host_offsets[sequence] = static_cast<uint32_t>(offsets[sequence]);
            for (size_t token = offsets[sequence];
                 token < offsets[sequence + 1]; token++) {
                host_positions[token] =
                    static_cast<uint32_t>(token - offsets[sequence]);
                host_sequence_ids[token] = static_cast<uint32_t>(sequence);
            }
        }
        host_offsets[batch_size] = static_cast<uint32_t>(tokens);
        uint32_t tensor_attention_min_tokens =
            engine->tensor_attention_min_tokens;
        if (!engine->tensor_attention_threshold_forced) {
            if (batch_size == 1) {
                tensor_attention_min_tokens = 80;
            } else if (batch_size <= 4) {
                tensor_attention_min_tokens = 192;
            } else if (batch_size <= 8) {
                tensor_attention_min_tokens = 384;
            } else {
                tensor_attention_min_tokens = 512;
            }
        }
        bool all_sequences_use_tensor_attention = true;
        bool all_sequences_use_online_attention = true;
        bool all_sequences_single_token = true;
        bool all_sequences_use_cooperative_pool = true;
        size_t max_sequence_tokens = 0;
        size_t min_sequence_tokens = EI_N_CTX;
        for (size_t sequence = 0; sequence < batch_size; sequence++) {
            const size_t sequence_tokens =
                offsets[sequence + 1] - offsets[sequence];
            max_sequence_tokens =
                std::max(max_sequence_tokens, sequence_tokens);
            min_sequence_tokens =
                std::min(min_sequence_tokens, sequence_tokens);
            if (sequence_tokens != 1) {
                all_sequences_single_token = false;
            }
            if (sequence_tokens != 1 && sequence_tokens < 128) {
                all_sequences_use_cooperative_pool = false;
            }
            if (sequence_tokens <
                tensor_attention_min_tokens) {
                all_sequences_use_tensor_attention = false;
            } else {
                all_sequences_use_online_attention = false;
            }
        }
        bool use_xe2_flash = false;
#ifdef EI_XPU_XE2_FLASH
        const size_t xe2_flash_auto_min_tokens = batch_size == 1
            ? engine->xe2_flash_min_tokens
            : engine->xe2_flash_batch_min_tokens;
        use_xe2_flash = tokens >= engine->gemm_min_tokens &&
            !all_sequences_single_token &&
            (engine->xe2_flash_mode > 0 ||
             (engine->xe2_flash_mode == 0 &&
              min_sequence_tokens >= xe2_flash_auto_min_tokens));
#endif
        engine->fp16_attention = engine->fp16_attention_mode > 0 ||
            (engine->fp16_attention_mode == 0 &&
             (all_sequences_use_tensor_attention || use_xe2_flash));

        record_profile(engine, "copy_ids", engine->queue.memcpy(
            engine->ids, ids, tokens * sizeof(*ids)));
        record_profile(engine, "copy_positions", engine->queue.memcpy(
            engine->positions, host_positions.data(),
            tokens * sizeof(uint32_t)));
        record_profile(engine, "copy_sequence_ids", engine->queue.memcpy(
            engine->sequence_ids, host_sequence_ids.data(),
            tokens * sizeof(uint32_t)));
        record_profile(engine, "copy_offsets", engine->queue.memcpy(
            engine->offsets, host_offsets.data(),
            (batch_size + 1) * sizeof(uint32_t)));
        const clock::time_point submit_begin = clock::now();
        auto enqueue_forward = [&]() {
            launch_embedding(engine, tokens);
            const bool use_gemm = tokens >= engine->gemm_min_tokens;
            const bool use_v_only = use_gemm &&
                engine->single_token_v_only && all_sequences_single_token;

            for (int layer_index = 0; layer_index < EI_N_LAYER; layer_index++) {
                const ei_layer *layer = &engine->model->layers[layer_index];
                const bool is_swa = ei_layer_is_swa(engine->model, layer_index);
                const float *rope =
                    is_swa ? engine->rope_swa : engine->rope_full;
                if (use_gemm) {
                    if (layer_index == 0) {
                        launch_rms_norm_half(
                            engine, engine->x,
                            float_weights(engine, layer->attn_norm),
                            engine->half_input, tokens, EI_N_EMBD,
                            engine->model->rms_eps);
                    }
                    if (use_v_only) {
                        launch_gemm(
                            engine, engine->half_input,
                            engine->layers[layer_index].qkv +
                                static_cast<size_t>(EI_N_EMBD + EI_HEAD_DIM) *
                                    EI_N_EMBD,
                            engine->qkv_combined, tokens, EI_HEAD_DIM,
                            EI_N_EMBD, "projection_v_gemm");
                        launch_repeat_value_half(
                            engine, engine->qkv_combined, tokens);
                    } else {
                        launch_gemm(engine, engine->half_input,
                                    engine->layers[layer_index].qkv,
                                    engine->qkv_combined, tokens,
                                    EI_N_EMBD + 2 * EI_HEAD_DIM, EI_N_EMBD,
                                    "projection_qkv_gemm",
                                    &engine->layers[layer_index].qkv_q4);
                        launch_qkv_epilogue(engine, layer, rope, tokens);
                    }
                } else {
                    launch_rms_norm(engine, engine->x,
                                    float_weights(engine, layer->attn_norm),
                                    engine->norm, tokens, EI_N_EMBD,
                                    engine->model->rms_eps);
                    launch_q4_projection(engine, layer->attn_q, engine->norm,
                                         engine->q, tokens);
                    launch_q4_projection(engine, layer->attn_k, engine->norm,
                                         engine->k, tokens);
                    launch_q4_projection(engine, layer->attn_v, engine->norm,
                                         engine->v, tokens);
                    launch_qk_norm_rope(engine, layer, rope, tokens);
                    if (engine->fp16_attention) {
                        launch_float_to_half(engine, engine->q,
                                             engine->q_half,
                                             tokens * EI_N_EMBD);
                        launch_float_to_half(engine, engine->k,
                                             engine->k_half,
                                             tokens * EI_HEAD_DIM);
                        launch_float_to_half(engine, engine->v,
                                             engine->v_half,
                                             tokens * EI_HEAD_DIM);
                    }
                }
                const uint32_t window =
                    is_swa ? engine->model->swa_window : 0;
                // When every sequence uses online attention, the attention
                // kernel can write the half context inline, skipping the
                // standalone float_to_half(ctx) submission before the GEMM.
                const bool fuse_ctx_half = engine->fuse_attn_output_half &&
                    use_gemm && !use_v_only && !use_xe2_flash &&
                    all_sequences_use_online_attention;
                if (use_xe2_flash) {
#ifdef EI_XPU_XE2_FLASH
                    const sycl::event flash_event =
                        ei_xpu_xe2_flash_attention(
                        engine->queue, engine->q_half, engine->k_half,
                        engine->v_half, engine->half_input, engine->offsets,
                        tokens, batch_size,
                        static_cast<uint32_t>(max_sequence_tokens), window);
                    record_profile(engine, "attention_xe2_flash", flash_event);
                    if (engine->xe2_flash_host_fence) {
                        engine->queue.wait_and_throw();
                    } else {
                        record_profile(
                            engine, "attention_xe2_flash_barrier",
                            engine->queue.ext_oneapi_submit_barrier(
                                {flash_event}));
                    }
#endif
                } else if (!use_v_only) {
                    launch_attention(engine, tokens, window,
                                     tensor_attention_min_tokens,
                                     fuse_ctx_half ? engine->half_input
                                                   : nullptr);
                    launch_tensor_attention(engine, host_offsets, window,
                                            tensor_attention_min_tokens);
                }
                if (use_gemm) {
                    if (!use_v_only && !use_xe2_flash && !fuse_ctx_half) {
                        launch_float_to_half(engine, engine->ctx,
                                             engine->half_input,
                                             tokens * EI_N_EMBD);
                    }
                    launch_gemm(engine, engine->half_input,
                                engine->layers[layer_index].attn_output,
                                engine->tmp, tokens, EI_N_EMBD, EI_N_EMBD,
                                "projection_attention_gemm",
                                &engine->layers[layer_index].attn_output_q4);
                } else {
                    launch_q4_projection(engine, layer->attn_output,
                                         engine->ctx, engine->tmp, tokens);
                }
                if (use_gemm) {
                    launch_rms_residual_next_half(
                        engine, engine->tmp,
                        float_weights(engine, layer->post_attention_norm),
                        engine->x, float_weights(engine, layer->ffn_norm),
                        engine->half_input, tokens, EI_N_EMBD,
                        engine->model->rms_eps);
                } else {
                    launch_rms_residual(
                        engine, engine->tmp,
                        float_weights(engine, layer->post_attention_norm),
                        engine->x, tokens, EI_N_EMBD,
                        engine->model->rms_eps);
                }

                if (use_gemm) {
#ifdef EI_XPU_XE2_W4
                    if (engine->xe2_w4 && tokens == 2 && batch_size == 1) {
                        launch_xe2_w4_up_gate(
                            engine, engine->layers[layer_index].up_gate_w4,
                            tokens);
                    } else
#endif
                    {
                        launch_gemm(engine, engine->half_input,
                                    engine->layers[layer_index].up_gate,
                                    engine->up_gate_combined, tokens,
                                    2 * EI_N_FF, EI_N_EMBD,
                                    "projection_up_gate_gemm",
                                    &engine->layers[layer_index].up_gate_q4);
                        launch_up_gate_gelu_half(engine, tokens);
                    }
                    launch_gemm(engine, engine->half_input,
                                engine->layers[layer_index].ffn_down,
                                engine->tmp, tokens, EI_N_EMBD, EI_N_FF,
                                "projection_down_gemm",
                                &engine->layers[layer_index].ffn_down_q4);
                } else {
                    launch_rms_norm(engine, engine->x,
                                    float_weights(engine, layer->ffn_norm),
                                    engine->norm, tokens, EI_N_EMBD,
                                    engine->model->rms_eps);
                    launch_q4_projection(engine, layer->ffn_up, engine->norm,
                                         engine->up, tokens);
                    launch_q4_projection(engine, layer->ffn_gate, engine->norm,
                                         engine->gate, tokens);
                    launch_gelu_mul(engine, tokens);
                    launch_q4_projection(engine, layer->ffn_down, engine->gate,
                                         engine->tmp, tokens);
                }
                if (use_gemm && layer_index + 1 < EI_N_LAYER) {
                    launch_rms_residual_next_half(
                        engine, engine->tmp,
                        float_weights(engine, layer->post_ffw_norm),
                        engine->x,
                        float_weights(
                            engine,
                            engine->model->layers[layer_index + 1].attn_norm),
                        engine->half_input, tokens, EI_N_EMBD,
                        engine->model->rms_eps);
                } else {
                    launch_rms_residual(
                        engine, engine->tmp,
                        float_weights(engine, layer->post_ffw_norm),
                        engine->x, tokens, EI_N_EMBD,
                        engine->model->rms_eps);
                }
            }
            const bool cooperative_pool = engine->cooperative_pool_mode > 0 ||
                (engine->cooperative_pool_mode == 0 &&
                 all_sequences_use_cooperative_pool);
            launch_pool(engine, batch_size, cooperative_pool);
        };

        bool replayed_graph = false;
        const bool use_command_graph = engine->command_graph_mode > 0 ||
            (engine->command_graph_mode == 0 && all_sequences_single_token &&
             batch_size <= 4);
        if (use_command_graph && !engine->profiling) {
            std::vector<uint32_t> sequence_lengths(batch_size);
            for (size_t sequence = 0; sequence < batch_size; sequence++) {
                sequence_lengths[sequence] =
                    host_offsets[sequence + 1] - host_offsets[sequence];
            }
            xpu_graph_entry *entry = nullptr;
            for (xpu_graph_entry &candidate : engine->graph_cache) {
                if (candidate.sequence_lengths == sequence_lengths) {
                    entry = &candidate;
                    break;
                }
            }
            if (!entry) {
                graph_exp::command_graph graph(engine->queue);
                graph.begin_recording(engine->queue);
                enqueue_forward();
                graph.end_recording(engine->queue);
                auto executable = std::make_shared<xpu_executable_graph>(
                    graph.finalize());
                if (engine->graph_cache.size() == 8) {
                    auto oldest = std::min_element(
                        engine->graph_cache.begin(), engine->graph_cache.end(),
                        [](const xpu_graph_entry &a,
                           const xpu_graph_entry &b) {
                            return a.last_used < b.last_used;
                        });
                    engine->graph_cache.erase(oldest);
                }
                engine->graph_cache.push_back({
                    std::move(sequence_lengths), std::move(executable),
                    ++engine->graph_clock});
                entry = &engine->graph_cache.back();
            } else {
                entry->last_used = ++engine->graph_clock;
            }
            record_profile(engine, "command_graph",
                           engine->queue.ext_oneapi_graph(*entry->graph));
            replayed_graph = true;
        }
        if (!replayed_graph) enqueue_forward();
        record_profile(engine, "copy_result", engine->queue.memcpy(
            out, engine->result, batch_size * EI_N_EMBD * sizeof(float)));
        const clock::time_point submit_end = clock::now();
        engine->queue.wait_and_throw();
        const clock::time_point total_end = clock::now();
        if (engine->profiling) {
            const double submit_us = std::chrono::duration<double, std::micro>(
                submit_end - submit_begin).count();
            const double total_us = std::chrono::duration<double, std::micro>(
                total_end - total_begin).count();
            report_profile(engine, tokens, batch_size, submit_us, total_us);
        }
        if (err && err_len) err[0] = '\0';
        return true;
    } catch (const std::exception &exception) {
        set_exception(err, err_len, "execute XPU forward pass", exception);
        return false;
    }
}

static std::vector<float> make_rope_table(float base) {
    std::vector<float> table(static_cast<size_t>(EI_N_CTX) * EI_HEAD_DIM);
    for (size_t position = 0; position < EI_N_CTX; position++) {
        for (size_t dim = 0; dim < EI_HEAD_DIM / 2; dim++) {
            const float theta = static_cast<float>(position) *
                std::pow(base, -2.0f * static_cast<float>(dim) /
                                static_cast<float>(EI_HEAD_DIM));
            const size_t index = position * EI_HEAD_DIM + 2 * dim;
            table[index] = std::cos(theta);
            table[index + 1] = std::sin(theta);
        }
    }
    return table;
}

}  // namespace

extern "C" void *ei_xpu_engine_create(const ei_model *model,
                                       char *err, size_t err_len) {
    xpu_engine *engine = nullptr;
    try {
        engine = new xpu_engine(select_device());
        engine->model = model;
#ifdef EI_XPU_XE2_FLASH
        engine->xe2_flash_mode = engine->xe2_device ? 0 : -1;
        engine->xe2_w4 = engine->xe2_device;
        engine->rms_register_cache = engine->xe2_device;
#endif
        const char *gemm_min_env = std::getenv("EI_XPU_GEMM_MIN_TOKENS");
        if (gemm_min_env && *gemm_min_env) {
            char *end = nullptr;
            const unsigned long parsed = std::strtoul(gemm_min_env, &end, 10);
            if (*end != '\0' || parsed < 1 || parsed > 65536) {
                throw std::runtime_error(
                    "EI_XPU_GEMM_MIN_TOKENS must be an integer from 1 to 65536");
            }
            engine->gemm_min_tokens = static_cast<uint32_t>(parsed);
        }
        const char *q4_m_tiled_env = std::getenv("EI_XPU_Q4_M_TILED");
        if (q4_m_tiled_env && *q4_m_tiled_env) {
            if (std::strcmp(q4_m_tiled_env, "0") == 0) {
                engine->q4_m_tiled = false;
            } else if (std::strcmp(q4_m_tiled_env, "1") == 0) {
                engine->q4_m_tiled = true;
            } else {
                throw std::runtime_error(
                    "EI_XPU_Q4_M_TILED must be 0 or 1");
            }
        }
#ifdef EI_XPU_XE2_W4
        const char *xe2_w4_env = std::getenv("EI_XPU_XE2_W4");
        if (xe2_w4_env && *xe2_w4_env) {
            if (std::strcmp(xe2_w4_env, "0") == 0) {
                engine->xe2_w4 = false;
            } else if (std::strcmp(xe2_w4_env, "1") == 0) {
                if (!engine->xe2_device) {
                    throw std::runtime_error(
                        "EI_XPU_XE2_W4=1 requires an Intel Xe2/BMG GPU");
                }
                engine->xe2_w4 = true;
            } else {
                throw std::runtime_error("EI_XPU_XE2_W4 must be 0 or 1");
            }
        }
#endif
#ifdef EI_XPU_ONEDNN
        const char *onednn_w4_env = std::getenv("EI_XPU_ONEDNN_W4");
        if (onednn_w4_env && *onednn_w4_env) {
            if (std::strcmp(onednn_w4_env, "0") == 0) {
                engine->onednn_w4 = false;
            } else if (std::strcmp(onednn_w4_env, "1") == 0) {
                engine->onednn_w4 = true;
            } else {
                throw std::runtime_error(
                    "EI_XPU_ONEDNN_W4 must be 0 or 1");
            }
        }
        const char *onednn_f16_env = std::getenv("EI_XPU_ONEDNN_F16");
        if (onednn_f16_env && *onednn_f16_env) {
            if (std::strcmp(onednn_f16_env, "0") == 0) {
                engine->onednn_f16 = false;
            } else if (std::strcmp(onednn_f16_env, "1") == 0) {
                engine->onednn_f16 = true;
            } else {
                throw std::runtime_error(
                    "EI_XPU_ONEDNN_F16 must be 0 or 1");
            }
        }
        if (engine->onednn_w4 && engine->onednn_f16) {
            throw std::runtime_error(
                "EI_XPU_ONEDNN_W4 and EI_XPU_ONEDNN_F16 are mutually exclusive");
        }
#endif
        const char *tensor_min_env =
            std::getenv("EI_XPU_TENSOR_ATTENTION_MIN_TOKENS");
        if (tensor_min_env && *tensor_min_env) {
            char *end = nullptr;
            const unsigned long parsed =
                std::strtoul(tensor_min_env, &end, 10);
            if (*end != '\0' || parsed < 1 || parsed > 65536) {
                throw std::runtime_error(
                    "EI_XPU_TENSOR_ATTENTION_MIN_TOKENS must be an integer from 1 to 65536");
            }
            engine->tensor_attention_min_tokens =
                static_cast<uint32_t>(parsed);
            engine->tensor_attention_threshold_forced = true;
        }
        const char *swa_tile_env =
            std::getenv("EI_XPU_SWA_TENSOR_TILE_TOKENS");
        if (swa_tile_env && *swa_tile_env) {
            char *end = nullptr;
            const unsigned long parsed = std::strtoul(swa_tile_env, &end, 10);
            if (*end != '\0' || (parsed != 0 && parsed != 128 &&
                                 parsed != 256 && parsed != 512 &&
                                 parsed != 1024)) {
                throw std::runtime_error(
                    "EI_XPU_SWA_TENSOR_TILE_TOKENS must be 0, 128, 256, 512, or 1024");
            }
            engine->swa_tensor_tile_tokens = static_cast<uint32_t>(parsed);
        }
        const char *swa_min_env =
            std::getenv("EI_XPU_SWA_TENSOR_MIN_TOKENS");
        if (swa_min_env && *swa_min_env) {
            char *end = nullptr;
            const unsigned long parsed = std::strtoul(swa_min_env, &end, 10);
            if (*end != '\0' || parsed < 1 || parsed > 65536) {
                throw std::runtime_error(
                    "EI_XPU_SWA_TENSOR_MIN_TOKENS must be an integer from 1 to 65536");
            }
            engine->swa_tensor_banded_min_tokens =
                static_cast<uint32_t>(parsed);
        }
        const char *v_only_env = std::getenv("EI_XPU_SINGLE_TOKEN_V_ONLY");
        if (v_only_env && *v_only_env) {
            if (std::strcmp(v_only_env, "0") == 0) {
                engine->single_token_v_only = false;
            } else if (std::strcmp(v_only_env, "1") == 0) {
                engine->single_token_v_only = true;
            } else {
                throw std::runtime_error(
                    "EI_XPU_SINGLE_TOKEN_V_ONLY must be 0 or 1");
            }
        }
        const char *cooperative_rms_env =
            std::getenv("EI_XPU_COOPERATIVE_RMS_MAX_ROWS");
        if (cooperative_rms_env && *cooperative_rms_env) {
            char *end = nullptr;
            const unsigned long parsed =
                std::strtoul(cooperative_rms_env, &end, 10);
            if (*end != '\0' || parsed > 128) {
                throw std::runtime_error(
                    "EI_XPU_COOPERATIVE_RMS_MAX_ROWS must be an integer from 0 to 128");
            }
            engine->cooperative_rms_max_rows =
                static_cast<uint32_t>(parsed);
        }
        const char *rms_register_env =
            std::getenv("EI_XPU_RMS_REGISTER_CACHE");
        if (rms_register_env && *rms_register_env) {
            if (std::strcmp(rms_register_env, "0") == 0) {
                engine->rms_register_cache = false;
            } else if (std::strcmp(rms_register_env, "1") == 0) {
                engine->rms_register_cache = true;
            } else {
                throw std::runtime_error(
                    "EI_XPU_RMS_REGISTER_CACHE must be 0 or 1");
            }
        }
        const char *cooperative_pool_env =
            std::getenv("EI_XPU_COOPERATIVE_POOL");
        if (cooperative_pool_env && *cooperative_pool_env) {
            if (std::strcmp(cooperative_pool_env, "0") == 0) {
                engine->cooperative_pool_mode = -1;
            } else if (std::strcmp(cooperative_pool_env, "1") == 0) {
                engine->cooperative_pool_mode = 1;
            } else if (std::strcmp(cooperative_pool_env, "auto") == 0) {
                engine->cooperative_pool_mode = 0;
            } else {
                throw std::runtime_error(
                    "EI_XPU_COOPERATIVE_POOL must be 0, 1, or auto");
            }
        }
        const char *xe2_flash_env = std::getenv("EI_XPU_XE2_FLASH");
        if (xe2_flash_env && *xe2_flash_env) {
            if (std::strcmp(xe2_flash_env, "0") == 0) {
                engine->xe2_flash_mode = -1;
            } else if (std::strcmp(xe2_flash_env, "1") == 0) {
                if (!engine->xe2_device) {
                    throw std::runtime_error(
                        "EI_XPU_XE2_FLASH=1 requires an Intel Xe2/BMG GPU");
                }
                engine->xe2_flash_mode = 1;
            } else if (std::strcmp(xe2_flash_env, "auto") == 0) {
                engine->xe2_flash_mode = engine->xe2_device ? 0 : -1;
            } else {
                throw std::runtime_error(
                    "EI_XPU_XE2_FLASH must be 0, 1, or auto");
            }
        }
        const char *xe2_flash_min_env =
            std::getenv("EI_XPU_XE2_FLASH_MIN_TOKENS");
        if (xe2_flash_min_env && *xe2_flash_min_env) {
            char *end = nullptr;
            const unsigned long parsed =
                std::strtoul(xe2_flash_min_env, &end, 10);
            if (*end != '\0' || parsed < 1 || parsed > 65536) {
                throw std::runtime_error(
                    "EI_XPU_XE2_FLASH_MIN_TOKENS must be an integer from 1 to 65536");
            }
            engine->xe2_flash_min_tokens = static_cast<uint32_t>(parsed);
        }
        const char *xe2_flash_batch_min_env =
            std::getenv("EI_XPU_XE2_FLASH_BATCH_MIN_TOKENS");
        if (xe2_flash_batch_min_env && *xe2_flash_batch_min_env) {
            char *end = nullptr;
            const unsigned long parsed =
                std::strtoul(xe2_flash_batch_min_env, &end, 10);
            if (*end != '\0' || parsed < 1 || parsed > 65536) {
                throw std::runtime_error(
                    "EI_XPU_XE2_FLASH_BATCH_MIN_TOKENS must be an integer from 1 to 65536");
            }
            engine->xe2_flash_batch_min_tokens =
                static_cast<uint32_t>(parsed);
        }
        const char *xe2_flash_fence_env =
            std::getenv("EI_XPU_XE2_FLASH_HOST_FENCE");
        if (xe2_flash_fence_env && *xe2_flash_fence_env) {
            if (std::strcmp(xe2_flash_fence_env, "0") == 0) {
                engine->xe2_flash_host_fence = false;
            } else if (std::strcmp(xe2_flash_fence_env, "1") == 0) {
                engine->xe2_flash_host_fence = true;
            } else {
                throw std::runtime_error(
                    "EI_XPU_XE2_FLASH_HOST_FENCE must be 0 or 1");
            }
        }
        const char *fp16_attention_env =
            std::getenv("EI_XPU_FP16_ATTENTION");
        if (fp16_attention_env && *fp16_attention_env) {
            if (std::strcmp(fp16_attention_env, "0") == 0) {
                engine->fp16_attention_mode = -1;
            } else if (std::strcmp(fp16_attention_env, "1") == 0) {
                engine->fp16_attention_mode = 1;
            } else if (std::strcmp(fp16_attention_env, "auto") == 0) {
                engine->fp16_attention_mode = 0;
            } else {
                throw std::runtime_error(
                    "EI_XPU_FP16_ATTENTION must be 0, 1, or auto");
            }
        }
        const char *command_graph_env =
            std::getenv("EI_XPU_COMMAND_GRAPH");
        if (command_graph_env && *command_graph_env) {
            if (std::strcmp(command_graph_env, "0") == 0) {
                engine->command_graph_mode = -1;
            } else if (std::strcmp(command_graph_env, "1") == 0) {
                engine->command_graph_mode = 1;
            } else if (std::strcmp(command_graph_env, "auto") == 0) {
                engine->command_graph_mode = 0;
            } else {
                throw std::runtime_error(
                    "EI_XPU_COMMAND_GRAPH must be 0, 1, or auto");
            }
        }
        if (!engine->device.has(sycl::aspect::ext_oneapi_graph)) {
            if (engine->command_graph_mode > 0) {
                throw std::runtime_error(
                    "EI_XPU_COMMAND_GRAPH=1 requires ext_oneapi_graph support");
            }
            engine->command_graph_mode = -1;
        }
        const char *fuse_attn_output_env =
            std::getenv("EI_XPU_FUSE_ATTN_OUTPUT_HALF");
        if (fuse_attn_output_env && *fuse_attn_output_env) {
            if (std::strcmp(fuse_attn_output_env, "0") == 0) {
                engine->fuse_attn_output_half = false;
            } else if (std::strcmp(fuse_attn_output_env, "1") == 0) {
                engine->fuse_attn_output_half = true;
            } else {
                throw std::runtime_error(
                    "EI_XPU_FUSE_ATTN_OUTPUT_HALF must be 0 or 1");
            }
        }
        const uint8_t *model_bytes = model->gguf.map + model->gguf.data_off;
        const size_t model_length = model->gguf.map_len - model->gguf.data_off;
        engine->model_data = device_allocate<uint8_t>(engine, model_length);
        engine->rope_full = device_allocate<float>(
            engine, static_cast<size_t>(EI_N_CTX) * EI_HEAD_DIM);
        engine->rope_swa = device_allocate<float>(
            engine, static_cast<size_t>(EI_N_CTX) * EI_HEAD_DIM);
        const std::vector<float> rope_full =
            make_rope_table(model->rope_base_full);
        const std::vector<float> rope_swa =
            make_rope_table(model->rope_base_swa);
        engine->queue.memcpy(engine->model_data, model_bytes, model_length);
        engine->queue.memcpy(engine->rope_full, rope_full.data(),
                             rope_full.size() * sizeof(float));
        engine->queue.memcpy(engine->rope_swa, rope_swa.data(),
                             rope_swa.size() * sizeof(float));
        initialize_expanded_weights(engine);
#ifdef EI_XPU_XE2_W4
        if (engine->xe2_w4) initialize_xe2_w4_weights(engine);
#endif
#ifdef EI_XPU_ONEDNN
        if (engine->onednn_w4) {
            initialize_onednn_weights(engine);
        } else if (engine->onednn_f16) {
            engine->onednn = ei_xpu_onednn_create(engine->queue);
        }
#endif
        engine->queue.wait_and_throw();
        if (err && err_len) err[0] = '\0';
        return engine;
    } catch (const std::exception &exception) {
        set_exception(err, err_len, "initialize XPU backend", exception);
        if (engine) {
            release_workspace(engine);
            release_xe2_w4_weights(engine);
            release_onednn_weights(engine);
            release_expanded_weights(engine);
            device_release(engine, engine->model_data);
            device_release(engine, engine->rope_full);
            device_release(engine, engine->rope_swa);
            delete engine;
        }
        return nullptr;
    }
}

extern "C" void ei_xpu_engine_free(void *handle) {
    xpu_engine *engine = static_cast<xpu_engine *>(handle);
    if (!engine) return;
    try {
        engine->queue.wait_and_throw();
    } catch (...) {
    }
    release_workspace(engine);
    release_xe2_w4_weights(engine);
    release_onednn_weights(engine);
    release_expanded_weights(engine);
    device_release(engine, engine->model_data);
    device_release(engine, engine->rope_full);
    device_release(engine, engine->rope_swa);
    delete engine;
}

extern "C" bool ei_xpu_engine_reserve(void *handle, size_t total_tokens,
                                       size_t batch_size, char *err,
                                       size_t err_len) {
    xpu_engine *engine = static_cast<xpu_engine *>(handle);
    if (!engine || total_tokens == 0 || total_tokens > 65536 ||
        batch_size == 0 || batch_size > 256) {
        set_error(err, err_len,
                  "XPU workspace reservation is outside supported limits");
        return false;
    }
    try {
        ensure_capacity(engine, total_tokens, batch_size);
        if (err && err_len) err[0] = '\0';
        return true;
    } catch (const std::exception &exception) {
        set_exception(err, err_len, "reserve XPU workspace", exception);
        return false;
    }
}

extern "C" bool ei_xpu_engine_embed_tokens(void *handle, const int32_t *ids,
                                            size_t n_tokens,
                                            float out[EI_N_EMBD], char *err,
                                            size_t err_len) {
    xpu_engine *engine = static_cast<xpu_engine *>(handle);
    if (!engine || !ids || !out || n_tokens == 0 || n_tokens > EI_N_CTX) {
        set_error(err, err_len, "invalid XPU token request");
        return false;
    }
    const size_t offsets[2] = {0, n_tokens};
    return execute_batch(engine, ids, offsets, 1, out, err, err_len);
}

extern "C" bool ei_xpu_engine_embed_tokens_batch(
    void *handle, const int32_t *ids, const size_t *offsets,
    size_t batch_size, float *out, char *err, size_t err_len) {
    xpu_engine *engine = static_cast<xpu_engine *>(handle);
    if (!engine || !ids || !offsets || !out || batch_size == 0 ||
        batch_size > 256 || offsets[0] != 0 || offsets[batch_size] == 0 ||
        offsets[batch_size] > 65536) {
        set_error(err, err_len, "invalid XPU batch request");
        return false;
    }
    return execute_batch(engine, ids, offsets, batch_size, out, err, err_len);
}
