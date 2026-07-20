#include "engine_xpu_onednn.h"

#ifdef EI_XPU_ONEDNN

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

constexpr size_t kQK = 32;
constexpr size_t kQ4BlockBytes = 18;

struct primitive_entry {
    dnnl::memory::desc src;
    dnnl::memory::desc weights;
    dnnl::memory::desc scales;
    dnnl::memory::desc dst;
    dnnl::matmul primitive;

    primitive_entry(const dnnl::engine &engine, size_t m, size_t n, size_t k)
        : src({static_cast<dnnl::memory::dim>(m),
               static_cast<dnnl::memory::dim>(k)},
              dnnl::memory::data_type::f16, dnnl::memory::format_tag::ab),
          weights({static_cast<dnnl::memory::dim>(k),
                   static_cast<dnnl::memory::dim>(n)},
                  dnnl::memory::data_type::s4, dnnl::memory::format_tag::ab),
          scales({static_cast<dnnl::memory::dim>(k / kQK),
                  static_cast<dnnl::memory::dim>(n)},
                 dnnl::memory::data_type::f16, dnnl::memory::format_tag::ab),
          dst({static_cast<dnnl::memory::dim>(m),
               static_cast<dnnl::memory::dim>(n)},
              dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab),
          primitive(make_primitive(engine)) {}

private:
    dnnl::matmul make_primitive(const dnnl::engine &engine) {
        dnnl::primitive_attr attributes;
        attributes.set_scales(
            DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1),
            {static_cast<dnnl::memory::dim>(kQK), 1},
            dnnl::memory::data_type::f16);
        attributes.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
        return dnnl::matmul(dnnl::matmul::primitive_desc(
            engine, src, weights, dst, attributes));
    }
};

struct dense_primitive_entry {
    dnnl::memory::desc src;
    dnnl::memory::desc weights;
    dnnl::memory::desc dst;
    dnnl::matmul primitive;

    dense_primitive_entry(const dnnl::engine &engine,
                          size_t m, size_t n, size_t k)
        : src({static_cast<dnnl::memory::dim>(m),
               static_cast<dnnl::memory::dim>(k)},
              dnnl::memory::data_type::f16, dnnl::memory::format_tag::ab),
          weights({static_cast<dnnl::memory::dim>(k),
                   static_cast<dnnl::memory::dim>(n)},
                  dnnl::memory::data_type::f16,
                  {1, static_cast<dnnl::memory::dim>(k)}),
          dst({static_cast<dnnl::memory::dim>(m),
               static_cast<dnnl::memory::dim>(n)},
              dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab),
          primitive(dnnl::matmul::primitive_desc(
              engine, src, weights, dst)) {}
};

struct onednn_engine {
    explicit onednn_engine(sycl::queue &queue)
        : engine(dnnl::sycl_interop::make_engine(
              queue.get_device(), queue.get_context())),
          stream(dnnl::sycl_interop::make_stream(engine, queue)) {}

    dnnl::engine engine;
    dnnl::stream stream;
    std::map<std::tuple<size_t, size_t, size_t>,
             std::unique_ptr<primitive_entry>> primitives;
    std::map<std::tuple<size_t, size_t, size_t>,
             std::unique_ptr<dense_primitive_entry>> dense_primitives;
};

static primitive_entry &get_primitive(onednn_engine &engine,
                                      size_t m, size_t n, size_t k) {
    const auto key = std::make_tuple(m, n, k);
    auto found = engine.primitives.find(key);
    if (found == engine.primitives.end()) {
        found = engine.primitives.emplace(
            key, std::make_unique<primitive_entry>(engine.engine, m, n, k)).first;
    }
    return *found->second;
}

static dense_primitive_entry &get_dense_primitive(
    onednn_engine &engine, size_t m, size_t n, size_t k) {
    const auto key = std::make_tuple(m, n, k);
    auto found = engine.dense_primitives.find(key);
    if (found == engine.dense_primitives.end()) {
        found = engine.dense_primitives.emplace(
            key, std::make_unique<dense_primitive_entry>(
                     engine.engine, m, n, k)).first;
    }
    return *found->second;
}

static void set_nibble(std::vector<uint8_t> &packed, size_t index, int value) {
    const uint8_t nibble = static_cast<uint8_t>(value) & 0x0f;
    uint8_t &byte = packed[index / 2];
    if ((index & 1) == 0) {
        byte = static_cast<uint8_t>((byte & 0xf0) | nibble);
    } else {
        byte = static_cast<uint8_t>((byte & 0x0f) | (nibble << 4));
    }
}

}  // namespace

void *ei_xpu_onednn_create(sycl::queue &queue) {
    return new onednn_engine(queue);
}

void ei_xpu_onednn_destroy(void *handle) {
    delete static_cast<onednn_engine *>(handle);
}

void ei_xpu_onednn_prepare_weight(
    sycl::queue &queue, ei_xpu_onednn_weight &weight,
    const ei_tensor *const *tensors, size_t tensor_count) {
    if (!tensors || tensor_count == 0) {
        throw std::invalid_argument("oneDNN Q4 weight list is empty");
    }
    const size_t cols = static_cast<size_t>(tensors[0]->ne[0]);
    size_t rows = 0;
    for (size_t tensor_index = 0; tensor_index < tensor_count; tensor_index++) {
        if (static_cast<size_t>(tensors[tensor_index]->ne[0]) != cols ||
            cols % kQK != 0) {
            throw std::invalid_argument("oneDNN Q4 weight shapes are incompatible");
        }
        rows += static_cast<size_t>(tensors[tensor_index]->ne[1]);
    }

    std::vector<uint8_t> values((cols * rows + 1) / 2, 0);
    std::vector<sycl::half> scales((cols / kQK) * rows);
    size_t output_row = 0;
    for (size_t tensor_index = 0; tensor_index < tensor_count; tensor_index++) {
        const ei_tensor *tensor = tensors[tensor_index];
        const size_t tensor_rows = static_cast<size_t>(tensor->ne[1]);
        const uint8_t *source = static_cast<const uint8_t *>(tensor->data);
        for (size_t row = 0; row < tensor_rows; row++) {
            for (size_t block = 0; block < cols / kQK; block++) {
                const uint8_t *q4 = source +
                    (row * (cols / kQK) + block) * kQ4BlockBytes;
                const uint16_t scale_bits = static_cast<uint16_t>(q4[0]) |
                    (static_cast<uint16_t>(q4[1]) << 8);
                scales[block * rows + output_row + row] =
                    sycl::bit_cast<sycl::half>(scale_bits);
                for (size_t quant = 0; quant < 16; quant++) {
                    const size_t low_k = block * kQK + quant;
                    const size_t high_k = low_k + 16;
                    set_nibble(values, low_k * rows + output_row + row,
                               static_cast<int>(q4[2 + quant] & 0x0f) - 8);
                    set_nibble(values, high_k * rows + output_row + row,
                               static_cast<int>(q4[2 + quant] >> 4) - 8);
                }
            }
        }
        output_row += tensor_rows;
    }

    weight.values = sycl::malloc_device<uint8_t>(values.size(), queue);
    weight.scales = sycl::malloc_device<sycl::half>(scales.size(), queue);
    if (!weight.values || !weight.scales) {
        ei_xpu_onednn_release_weight(queue, weight);
        throw std::bad_alloc();
    }
    weight.rows = rows;
    weight.cols = cols;
    queue.memcpy(weight.values, values.data(), values.size());
    queue.memcpy(weight.scales, scales.data(), scales.size() * sizeof(sycl::half));
    queue.wait_and_throw();
}

void ei_xpu_onednn_release_weight(
    sycl::queue &queue, ei_xpu_onednn_weight &weight) {
    if (weight.values) sycl::free(weight.values, queue);
    if (weight.scales) sycl::free(weight.scales, queue);
    weight = {};
}

sycl::event ei_xpu_onednn_matmul(
    void *handle, const sycl::half *input,
    const ei_xpu_onednn_weight &weight, float *output, size_t tokens) {
    onednn_engine &engine = *static_cast<onednn_engine *>(handle);
    primitive_entry &entry = get_primitive(
        engine, tokens, weight.rows, weight.cols);
    const auto memory = [&](const dnnl::memory::desc &desc, void *pointer) {
        return dnnl::sycl_interop::make_memory(
            desc, engine.engine, dnnl::sycl_interop::memory_kind::usm, pointer);
    };
    std::unordered_map<int, dnnl::memory> arguments = {
        {DNNL_ARG_SRC, memory(entry.src, const_cast<sycl::half *>(input))},
        {DNNL_ARG_WEIGHTS, memory(entry.weights, weight.values)},
        {DNNL_ARG_DST, memory(entry.dst, output)},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
         memory(entry.scales, weight.scales)},
    };
    return dnnl::sycl_interop::execute(
        entry.primitive, engine.stream, arguments);
}

sycl::event ei_xpu_onednn_dense_matmul(
    void *handle, const sycl::half *input, const sycl::half *weights,
    float *output, size_t tokens, size_t rows, size_t cols) {
    onednn_engine &engine = *static_cast<onednn_engine *>(handle);
    dense_primitive_entry &entry = get_dense_primitive(
        engine, tokens, rows, cols);
    const auto memory = [&](const dnnl::memory::desc &desc, void *pointer) {
        return dnnl::sycl_interop::make_memory(
            desc, engine.engine, dnnl::sycl_interop::memory_kind::usm, pointer);
    };
    std::unordered_map<int, dnnl::memory> arguments = {
        {DNNL_ARG_SRC, memory(entry.src, const_cast<sycl::half *>(input))},
        {DNNL_ARG_WEIGHTS,
         memory(entry.weights, const_cast<sycl::half *>(weights))},
        {DNNL_ARG_DST, memory(entry.dst, output)},
    };
    return dnnl::sycl_interop::execute(
        entry.primitive, engine.stream, arguments);
}

#endif
