#include "engine_xpu_w4.h"
#include "model.h"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static float fp16(const uint8_t *bytes) {
    const uint16_t bits = static_cast<uint16_t>(bytes[0]) |
        (static_cast<uint16_t>(bytes[1]) << 8);
    return static_cast<float>(sycl::bit_cast<sycl::half>(bits));
}

static float q4_value(const ei_tensor *tensor, size_t row, size_t col) {
    const size_t cols = static_cast<size_t>(tensor->ne[0]);
    const size_t block = col / 32;
    const size_t item = col % 32;
    const uint8_t *q4 = static_cast<const uint8_t *>(tensor->data) +
        (row * (cols / 32) + block) * 18;
    const uint8_t packed = q4[2 + item % 16];
    const int code = item < 16 ? packed & 0x0f : packed >> 4;
    return fp16(q4) * static_cast<float>(code - 8);
}

static bool test_projection(
    sycl::queue &queue, const char *name,
    const ei_tensor *const *tensors, size_t tensor_count, size_t tokens) {
    const size_t cols = static_cast<size_t>(tensors[0]->ne[0]);
    size_t rows = 0;
    for (size_t tensor_index = 0; tensor_index < tensor_count; tensor_index++) {
        rows += static_cast<size_t>(tensors[tensor_index]->ne[1]);
    }

    std::vector<sycl::half> input(tokens * cols);
    for (size_t token = 0; token < tokens; token++) {
        for (size_t col = 0; col < cols; col++) {
            input[token * cols + col] = static_cast<sycl::half>(std::sin(
                static_cast<float>((token + 1) * (col + 1)) * 0.013f));
        }
    }
    std::vector<float> reference(tokens * rows, 0.0f);
    size_t output_row = 0;
    for (size_t tensor_index = 0; tensor_index < tensor_count; tensor_index++) {
        const ei_tensor *tensor = tensors[tensor_index];
        const size_t tensor_rows = static_cast<size_t>(tensor->ne[1]);
        for (size_t token = 0; token < tokens; token++) {
            for (size_t row = 0; row < tensor_rows; row++) {
                float sum = 0.0f;
                for (size_t col = 0; col < cols; col++) {
                    sum += static_cast<float>(input[token * cols + col]) *
                        q4_value(tensor, row, col);
                }
                reference[token * rows + output_row + row] = sum;
            }
        }
        output_row += tensor_rows;
    }

    ei_xpu_w4_weight weight;
    ei_xpu_w4_prepare_weight(queue, weight, tensors, tensor_count);
    sycl::half *device_input =
        sycl::malloc_device<sycl::half>(input.size(), queue);
    sycl::half *device_output =
        sycl::malloc_device<sycl::half>(tokens * rows, queue);
    if (!device_input || !device_output) return false;
    queue.memcpy(device_input, input.data(), input.size() * sizeof(sycl::half));
    ei_xpu_w4_matmul(queue, device_input, weight, device_output, tokens);
    std::vector<sycl::half> output(tokens * rows);
    queue.memcpy(output.data(), device_output,
                 output.size() * sizeof(sycl::half));
    queue.wait_and_throw();

    double dot = 0.0;
    double aa = 0.0;
    double bb = 0.0;
    size_t nonfinite = 0;
    for (size_t index = 0; index < output.size(); index++) {
        const float value = static_cast<float>(output[index]);
        if (!std::isfinite(value)) nonfinite++;
        dot += static_cast<double>(value) * reference[index];
        aa += static_cast<double>(value) * value;
        bb += static_cast<double>(reference[index]) * reference[index];
    }
    const double similarity = dot / std::sqrt(aa * bb);
    std::printf(
        "Xe2 W4 %-11s M=%zu N=%zu K=%zu cosine=%.9f nonfinite=%zu\n",
        name, tokens, rows, cols, similarity, nonfinite);

    sycl::free(device_output, queue);
    sycl::free(device_input, queue);
    ei_xpu_w4_release_weight(queue, weight);
    return nonfinite == 0 && std::isfinite(similarity) && similarity >= 0.999;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s model.gguf\n", argv[0]);
        return 2;
    }
    ei_model model;
    ei_model_load(&model, argv[1]);
    sycl::queue queue(
        sycl::gpu_selector_v,
        sycl::property_list{
            sycl::property::queue::in_order{},
            sycl::ext::intel::property::queue::immediate_command_list{}});

    const ei_layer &layer = model.layers[0];
    const ei_tensor *qkv[] = {layer.attn_q, layer.attn_k, layer.attn_v};
    const ei_tensor *value[] = {layer.attn_v};
    const ei_tensor *attention[] = {layer.attn_output};
    const ei_tensor *up_gate[] = {layer.ffn_up, layer.ffn_gate};
    const ei_tensor *down[] = {layer.ffn_down};
    bool ok = true;
    for (size_t tokens : {size_t{1}, size_t{7}, size_t{32}}) {
        ok &= test_projection(queue, "qkv", qkv, 3, tokens);
        ok &= test_projection(queue, "value", value, 1, tokens);
        ok &= test_projection(queue, "attn_output", attention, 1, tokens);
        ok &= test_projection(queue, "up_gate", up_gate, 2, tokens);
        ok &= test_projection(queue, "ffn_down", down, 1, tokens);
    }
    ei_model_free(&model);
    return ok ? 0 : 1;
}
