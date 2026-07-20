#ifndef EI_ENGINE_XPU_ONEDNN_H
#define EI_ENGINE_XPU_ONEDNN_H

#include "model.h"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

struct ei_xpu_onednn_weight {
    uint8_t *values = nullptr;
    sycl::half *scales = nullptr;
    size_t rows = 0;
    size_t cols = 0;
};

#ifdef EI_XPU_ONEDNN

void *ei_xpu_onednn_create(sycl::queue &queue);
void ei_xpu_onednn_destroy(void *handle);
void ei_xpu_onednn_prepare_weight(
    sycl::queue &queue, ei_xpu_onednn_weight &weight,
    const ei_tensor *const *tensors, size_t tensor_count);
void ei_xpu_onednn_release_weight(
    sycl::queue &queue, ei_xpu_onednn_weight &weight);
sycl::event ei_xpu_onednn_matmul(
    void *handle, const sycl::half *input,
    const ei_xpu_onednn_weight &weight, float *output, size_t tokens);
sycl::event ei_xpu_onednn_dense_matmul(
    void *handle, const sycl::half *input, const sycl::half *weights,
    float *output, size_t tokens, size_t rows, size_t cols);

#endif

#endif
