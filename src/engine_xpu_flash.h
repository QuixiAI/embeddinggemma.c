#ifndef EI_ENGINE_XPU_FLASH_H
#define EI_ENGINE_XPU_FLASH_H

#ifdef EI_XPU_XE2_FLASH

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

sycl::event ei_xpu_xe2_flash_attention(
    sycl::queue &queue, const sycl::half *query, const sycl::half *key,
    const sycl::half *value, sycl::half *output,
    const uint32_t *cumulative_lengths, size_t total_tokens,
    size_t batch_size, uint32_t max_sequence_tokens, uint32_t window);

#endif

#endif
