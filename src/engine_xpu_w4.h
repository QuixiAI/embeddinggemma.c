#ifndef EI_ENGINE_XPU_W4_H
#define EI_ENGINE_XPU_W4_H

#include "model.h"

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

struct ei_xpu_w4_weight {
    uint8_t *values = nullptr;
    sycl::half *scales = nullptr;
    size_t rows = 0;
    size_t cols = 0;
};

#ifdef EI_XPU_XE2_W4

void ei_xpu_w4_prepare_weight(
    sycl::queue &queue, ei_xpu_w4_weight &weight,
    const ei_tensor *const *tensors, size_t tensor_count);
void ei_xpu_w4_release_weight(
    sycl::queue &queue, ei_xpu_w4_weight &weight);
sycl::event ei_xpu_w4_matmul(
    sycl::queue &queue, const sycl::half *input,
    const ei_xpu_w4_weight &weight, sycl::half *output, size_t tokens);

#endif

#endif
