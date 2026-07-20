#include "engine_xpu_flash.h"

#ifdef EI_XPU_XE2_FLASH

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include <cute/tensor.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "csrc/xpu/attn/xe_2/collective/chunk_prefill_epilogue.hpp"
#include "csrc/xpu/attn/xe_2/collective/chunk_prefill_scheduler.hpp"
#include "csrc/xpu/attn/xe_2/kernel/chunk_prefill_kernel.hpp"

#include <stdexcept>

namespace {

using namespace cute;

struct flash_args {
    void *query = nullptr;
    void *key = nullptr;
    void *value = nullptr;
    void *out = nullptr;
    void *block_table = nullptr;
    void *cu_seqlens_q = nullptr;
    void *cu_seqlens_k = nullptr;
    int max_queries = 0;
    int max_keys = 0;
    int total_seqlen_q = 0;
    int total_seqlen_k = 0;
    void *k_scale = nullptr;
    void *v_scale = nullptr;
    float sm_scale = 1.0f;
    void *sm_sink = nullptr;
    int batch_size = 0;
    int num_heads_q = 0;
    int num_heads_k = 0;
    int head_size = 0;
    int max_blocks_per_seq = 0;
    int block_size = 0;
    int window_size_left = -1;
    int window_size_right = -1;
    float *softmax_lse = nullptr;
    int lse_stride = 0;
    int q_stride_seq = 0;
    int q_stride_heads = 0;
    int q_stride_batch = 0;
    int k_stride_seq = 0;
    int k_stride_heads = 0;
    int k_stride_batch = 0;
    int v_stride_seq = 0;
    int v_stride_heads = 0;
    int v_stride_batch = 0;
    int o_stride_seq = 0;
    int o_stride_heads = 0;
    int o_stride_batch = 0;
    void *is_prefill = nullptr;
    int page_stride_elements = 0;
};

struct flash_head256_policy {
    using ShapeQK = Shape<_256, _32, _32>;
    using ShapePV = Shape<_256, _32, _32>;
    using ShapeOut = Shape<_256, _256>;
    using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
};

template <
    typename TileShapeQK,
    typename TileShapePV,
    typename TileShapeOutput,
    typename SubgroupLayoutQK,
    typename SubgroupLayoutPV,
    int PipelineStages,
    bool Local>
struct event_fmha_config {
    static constexpr int kSubgroupTileQ =
        get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
    using MMAOperation = XE_DPAS_TT<
        cute::gcd(kSubgroupTileQ, 8), float, half_t>;

    template <class Scheduler>
    static sycl::event run(sycl::queue &queue,
                           const flash_args &args) {
        constexpr bool kVarLen = true;
        cutlass::KernelHardwareInfo hardware_info;
        using ProblemShape =
            cutlass::fmha::kernel::FMHAProblemShape<kVarLen>;
        using TiledMMAQK = typename TiledMMAHelper<
            MMA_Atom<MMAOperation>, Layout<TileShapeQK>,
            SubgroupLayoutQK>::TiledMMA;
        using TiledMMAPV = typename TiledMMAHelper<
            MMA_Atom<MMAOperation>, Layout<TileShapePV>,
            SubgroupLayoutPV>::TiledMMA;
        static_assert(get<0>(TileShapeOutput{}) == get<0>(TileShapePV{}));
        constexpr int kValueTiles =
            get<1>(TileShapeOutput{}) / get<1>(TileShapePV{});

        using StrideQ = Stride<int, _1, int, int>;
        using StrideK = Stride<int, _1, int, int>;
        using StrideV = Stride<_1, int, int, int>;
        using StrideO = Stride<int, _1, int, int>;
        auto make_dummy_tensor = [&](auto value, auto stride) {
            return make_tensor(
                make_gmem_ptr(&value),
                make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
        };
        using TensorQ = decltype(make_dummy_tensor(half_t{}, StrideQ{}));
        using TensorK = decltype(make_dummy_tensor(half_t{}, StrideK{}));
        using TensorV = decltype(make_dummy_tensor(half_t{}, StrideV{}));
        using TensorO = decltype(make_dummy_tensor(half_t{}, StrideO{}));
        using Mainloop = cutlass::fmha::collective::FMHAFwdMainloop<
            cutlass::fmha::XeDefault<PipelineStages>, false, Local, false,
            TiledMMAQK, TiledMMAPV, kValueTiles, TensorQ, TensorK, TensorV,
            void, void, void>;
        using Epilogue = cutlass::fmha::collective::FMHAFwdEpilogue<
            false, Mainloop, TileShapeOutput, TensorO, void>;
        using Kernel = cutlass::fmha::kernel::XeFMHAFwdKernel<
            ProblemShape, Mainloop, Epilogue, Scheduler, false>;

        ProblemShape problem;
        problem.batch = args.batch_size;
        problem.num_heads_q = args.num_heads_q;
        problem.num_heads_kv = args.num_heads_k;
        problem.head_size_qk = args.head_size;
        problem.head_size_vo = args.head_size;
        problem.seq_len_qo =
            cutlass::fmha::collective::VariableLength{args.max_queries};
        problem.seq_len_qo.cumulative_length =
            reinterpret_cast<int *>(args.cu_seqlens_q);
        problem.seq_len_kv =
            cutlass::fmha::collective::VariableLength{args.max_keys};
        problem.seq_len_kv.cumulative_length =
            reinterpret_cast<int *>(args.cu_seqlens_k);

        StrideQ stride_q{};
        get<0>(stride_q) = args.q_stride_seq;
        get<2>(stride_q) = args.q_stride_heads;
        get<3>(stride_q) = args.q_stride_batch;
        StrideK stride_k{};
        get<0>(stride_k) = args.k_stride_seq;
        get<2>(stride_k) = args.k_stride_heads;
        get<3>(stride_k) = args.k_stride_batch;
        StrideV stride_v{};
        get<1>(stride_v) = args.v_stride_seq;
        get<2>(stride_v) = args.v_stride_heads;
        get<3>(stride_v) = args.v_stride_batch;
        StrideO stride_o{};
        get<0>(stride_o) = args.o_stride_seq;
        get<2>(stride_o) = args.o_stride_heads;
        get<3>(stride_o) = args.o_stride_batch;
        typename Kernel::Arguments arguments{
            {problem,
             reinterpret_cast<half_t *>(args.query), stride_q,
             reinterpret_cast<half_t *>(args.key), stride_k,
             reinterpret_cast<half_t *>(args.value), stride_v,
             reinterpret_cast<half_t *>(args.out), stride_o,
             reinterpret_cast<half_t *>(args.sm_sink), args.softmax_lse,
             args.lse_stride, static_cast<const bool *>(args.is_prefill)},
            {args.sm_scale, args.k_scale, args.v_scale,
             static_cast<int *>(args.block_table), args.block_size,
             args.max_blocks_per_seq, args.total_seqlen_k,
             args.window_size_left, args.window_size_right,
             args.page_stride_elements},
            {}, hardware_info};
        if (Kernel::get_workspace_size(arguments) != 0) {
            throw std::runtime_error(
                "Xe2 Flash Attention unexpectedly requires workspace");
        }
        Kernel::initialize_workspace(arguments, nullptr);
        auto params = Kernel::to_underlying_arguments(arguments, nullptr);

        namespace syclex = sycl::ext::oneapi::experimental;
        namespace intelex = sycl::ext::intel::experimental;
        const dim3 block = Kernel::get_block_shape();
        const dim3 grid = Kernel::get_grid_shape(params);
        compat::experimental::launch_properties launch_properties{
            syclex::work_group_scratch_size(Kernel::SharedStorageSize)};
        compat::experimental::kernel_properties kernel_properties{
            syclex::sub_group_size<cute::intel::sg_size>,
            intelex::grf_size<256>};
        compat::experimental::launch_policy policy{
            compat::dim3(grid.x, grid.y, grid.z),
            compat::dim3(block.x, block.y, block.z),
            launch_properties, kernel_properties};
        return compat::experimental::launch<
            cutlass::device_kernel<Kernel>>(policy, queue, params);
    }
};

template <bool Local>
sycl::event launch_head256(sycl::queue &queue,
                           const flash_args &args) {
    using config = event_fmha_config<
        flash_head256_policy::ShapeQK,
        flash_head256_policy::ShapePV,
        flash_head256_policy::ShapeOut,
        flash_head256_policy::SubgroupLayoutQK,
        decltype(cutlass::fmha::collective::get_sg_layout_pv(
            flash_head256_policy::SubgroupLayoutQK{})),
        2, Local>;
    return config::template run<
        cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(queue, args);
}

} // namespace

sycl::event ei_xpu_xe2_flash_attention(
    sycl::queue &queue, const sycl::half *query, const sycl::half *key,
    const sycl::half *value, sycl::half *output,
    const uint32_t *cumulative_lengths, size_t total_tokens,
    size_t batch_size, uint32_t max_sequence_tokens, uint32_t window) {
    flash_args args{};
    args.query = const_cast<sycl::half *>(query);
    args.key = const_cast<sycl::half *>(key);
    args.value = const_cast<sycl::half *>(value);
    args.out = output;
    args.cu_seqlens_q = const_cast<uint32_t *>(cumulative_lengths);
    args.cu_seqlens_k = const_cast<uint32_t *>(cumulative_lengths);
    args.max_queries = static_cast<int>(max_sequence_tokens);
    args.max_keys = static_cast<int>(max_sequence_tokens);
    args.total_seqlen_q = static_cast<int>(total_tokens);
    args.total_seqlen_k = static_cast<int>(total_tokens);
    args.sm_scale = 1.0f;
    args.batch_size = static_cast<int>(batch_size);
    args.num_heads_q = 3;
    args.num_heads_k = 1;
    args.head_size = 256;
    args.window_size_left = window == 0 ? -1 : static_cast<int>(window / 2);
    args.window_size_right = window == 0 ? -1 : static_cast<int>(window / 2);
    args.q_stride_seq = 768;
    args.q_stride_heads = 256;
    args.k_stride_seq = 256;
    args.k_stride_heads = 256;
    args.v_stride_seq = 256;
    args.v_stride_heads = 256;
    args.o_stride_seq = 768;
    args.o_stride_heads = 256;

    return window == 0
        ? launch_head256<false>(queue, args)
        : launch_head256<true>(queue, args);
}

#endif
