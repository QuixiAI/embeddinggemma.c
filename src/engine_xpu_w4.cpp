#include "engine_xpu_w4.h"

#ifdef EI_XPU_XE2_W4

#include "csrc/xpu/grouped_gemm/xe_2/gemm_xe2_policy.hpp"
#include "csrc/xpu/grouped_gemm/xe_2/grouped_gemm_xe2.hpp"

#include <stdexcept>
#include <vector>

namespace {

constexpr size_t kQK = 32;
constexpr size_t kQ4BlockBytes = 18;

static void set_nibble(std::vector<uint8_t> &packed,
                       size_t index, int value) {
    const uint8_t nibble = static_cast<uint8_t>(value) & 0x0f;
    uint8_t &byte = packed[index / 2];
    if ((index & 1) == 0) {
        byte = static_cast<uint8_t>((byte & 0xf0) | nibble);
    } else {
        byte = static_cast<uint8_t>((byte & 0x0f) | (nibble << 4));
    }
}

template <typename Policy>
class w4_gemm_kernel;

template <typename Policy>
sycl::event launch_w4(
    sycl::queue &queue, const sycl::half *input,
    const ei_xpu_w4_weight &weight, sycl::half *output, size_t tokens) {
    using namespace cute;
    using namespace MoE;
    using ElementA = half_t;
    using ElementB = uint8_t;
    using ElementS = half_t;
    using ElementD = half_t;
    using WGTile = typename Policy::WGTile;
    using SGLayout = typename Policy::SGLayout;
    using MMA = typename TiledMMAHelper<
        MMA_Atom<XE_DPAS_TT<8, float, ElementA>>,
        Layout<WGTile>, SGLayout>::TiledMMA;
    using GmemTiledCopyA = typename Policy::GmemTiledCopyA;
    using GmemTiledCopyB = typename Policy::GmemTiledCopyB;
    using GmemTiledCopyD = typename Policy::GmemTiledCopyD;

    const int m = static_cast<int>(tokens);
    const int n = static_cast<int>(weight.rows);
    const int k = static_cast<int>(weight.cols);
    const int tile_m = get<0>(WGTile{});
    const int tile_n = get<1>(WGTile{});
    const int groups_m = (m + tile_m - 1) / tile_m;
    const int groups_n = (n + tile_n - 1) / tile_n;
    const MMA mma{};
    const size_t local_threads = size(mma);

    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;
    syclex::properties kernel_properties{
        syclex::sub_group_size<16>, intelex::grf_size<256>};
    return queue.submit([&](sycl::handler &handler) {
        handler.parallel_for<w4_gemm_kernel<Policy>>(
            sycl::nd_range<3>{
                sycl::range<3>(1, static_cast<size_t>(groups_m),
                               static_cast<size_t>(groups_n) * local_threads),
                sycl::range<3>(1, 1, local_threads)},
            kernel_properties, [=](sycl::nd_item<3> item) {
                auto activations = make_moe_tensor<ElementA, 'R'>(
                    const_cast<ElementA *>(
                        reinterpret_cast<const ElementA *>(input)), m, k);
                auto weights = make_moe_tensor<int4_t, 'R'>(
                    reinterpret_cast<int4_t *>(weight.values), n, k);
                auto outputs = make_moe_tensor<ElementD, 'R'>(
                    reinterpret_cast<ElementD *>(output), m, n);
                xe_gemm_4bits<
                    GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD, kQK>(
                    activations, weights,
                    reinterpret_cast<const ElementS *>(weight.scales),
                    static_cast<const ElementA *>(nullptr), outputs,
                    make_coord(static_cast<int>(item.get_group(1)),
                               static_cast<int>(item.get_group(2)), _, 0),
                    mma);
            });
    });
}

} // namespace

void ei_xpu_w4_prepare_weight(
    sycl::queue &queue, ei_xpu_w4_weight &weight,
    const ei_tensor *const *tensors, size_t tensor_count) {
    if (!tensors || tensor_count == 0) {
        throw std::invalid_argument("Xe2 W4 weight list is empty");
    }
    const size_t cols = static_cast<size_t>(tensors[0]->ne[0]);
    size_t rows = 0;
    for (size_t tensor_index = 0; tensor_index < tensor_count;
         tensor_index++) {
        if (static_cast<size_t>(tensors[tensor_index]->ne[0]) != cols ||
            cols % kQK != 0) {
            throw std::invalid_argument("Xe2 W4 weight shapes are incompatible");
        }
        rows += static_cast<size_t>(tensors[tensor_index]->ne[1]);
    }

    std::vector<uint8_t> values((rows * cols + 1) / 2, 0);
    std::vector<sycl::half> scales(rows * (cols / kQK));
    size_t output_row = 0;
    for (size_t tensor_index = 0; tensor_index < tensor_count;
         tensor_index++) {
        const ei_tensor *tensor = tensors[tensor_index];
        const size_t tensor_rows = static_cast<size_t>(tensor->ne[1]);
        const uint8_t *source = static_cast<const uint8_t *>(tensor->data);
        for (size_t row = 0; row < tensor_rows; row++) {
            const size_t destination_row = output_row + row;
            for (size_t block = 0; block < cols / kQK; block++) {
                const uint8_t *q4 = source +
                    (row * (cols / kQK) + block) * kQ4BlockBytes;
                const uint16_t scale_bits = static_cast<uint16_t>(q4[0]) |
                    (static_cast<uint16_t>(q4[1]) << 8);
                scales[destination_row * (cols / kQK) + block] =
                    sycl::bit_cast<sycl::half>(scale_bits);
                for (size_t quant = 0; quant < 16; quant++) {
                    const size_t low = block * kQK + quant;
                    const size_t high = low + 16;
                    set_nibble(values, destination_row * cols + low,
                               static_cast<int>(q4[2 + quant] & 0x0f) - 8);
                    set_nibble(values, destination_row * cols + high,
                               static_cast<int>(q4[2 + quant] >> 4) - 8);
                }
            }
        }
        output_row += tensor_rows;
    }

    weight.values = sycl::malloc_device<uint8_t>(values.size(), queue);
    weight.scales = sycl::malloc_device<sycl::half>(scales.size(), queue);
    if (!weight.values || !weight.scales) {
        ei_xpu_w4_release_weight(queue, weight);
        throw std::bad_alloc();
    }
    weight.rows = rows;
    weight.cols = cols;
    queue.memcpy(weight.values, values.data(), values.size());
    queue.memcpy(weight.scales, scales.data(),
                 scales.size() * sizeof(sycl::half)).wait_and_throw();
}

void ei_xpu_w4_release_weight(
    sycl::queue &queue, ei_xpu_w4_weight &weight) {
    if (weight.values) sycl::free(weight.values, queue);
    if (weight.scales) sycl::free(weight.scales, queue);
    weight = {};
}

sycl::event ei_xpu_w4_matmul(
    sycl::queue &queue, const sycl::half *input,
    const ei_xpu_w4_weight &weight, sycl::half *output, size_t tokens) {
    if (tokens <= 4) {
        return launch_w4<MoE::w4a16_policy_m_8>(
            queue, input, weight, output, tokens);
    }
    if (tokens <= 8) {
        return launch_w4<MoE::w4a16_policy_m_16>(
            queue, input, weight, output, tokens);
    }
    return launch_w4<MoE::w4a16_policy_m_32>(
        queue, input, weight, output, tokens);
}

#endif
