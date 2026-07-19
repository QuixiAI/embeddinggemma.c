#include "embeddinggemma.metal"

// Gather Q8_0 token rows directly from the GGUF tensor and apply sqrt(n_embd).
kernel void ei_embedding_q8_0_f32(
    device const uchar *table [[buffer(0)]],
    device const int *token_ids [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &n_tokens [[buffer(3)]],
    constant float &scale [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    const uint total = n_tokens * EI_METAL_N_EMBD;
    if (tid >= total) return;

    const uint token = tid / EI_METAL_N_EMBD;
    const uint col = tid - token * EI_METAL_N_EMBD;
    const uint blocks_per_row = EI_METAL_N_EMBD / EI_METAL_QK;
    const uint block_index = col / EI_METAL_QK;
    const uint block_col = col - block_index * EI_METAL_QK;
    device const ei_metal_block_q8_0 *row =
        (device const ei_metal_block_q8_0 *)table + uint(token_ids[token]) * blocks_per_row;
    output[tid] = scale * float(row[block_index].d) * float(row[block_index].qs[block_col]);
}
