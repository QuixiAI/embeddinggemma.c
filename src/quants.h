#ifndef EI_QUANTS_H
#define EI_QUANTS_H

#include "gguf.h"

#define EI_QK 32

typedef struct {
    ei_fp16 d;
    uint8_t qs[16];
} ei_block_q4_0;

typedef struct {
    ei_fp16 d;
    int8_t qs[32];
} ei_block_q8_0;

_Static_assert(sizeof(ei_block_q4_0) == 18, "q4_0 block must be 18 bytes");
_Static_assert(sizeof(ei_block_q8_0) == 34, "q8_0 block must be 34 bytes");

ei_fp16 ei_fp32_to_fp16(float f);

void ei_dequantize_row_q8_0(const ei_tensor *t, int32_t row, float *out);
void ei_dequantize_row_q8_0_scaled(const ei_tensor *t, int32_t row, float scale, float *out);
void ei_quantize_row_q8_0(const float *x, ei_block_q8_0 *out, int32_t n);
float ei_vec_dot_q4_0_q8_0(const ei_block_q4_0 *x, const ei_block_q8_0 *y, int32_t n);
void ei_vec_dot_q4_0_q8_0_dual(const ei_block_q4_0 *x0, const ei_block_q4_0 *x1,
                               const ei_block_q8_0 *y, int32_t n,
                               float *out0, float *out1);
void ei_vec_dot_q4_0_q8_0_triple(const ei_block_q4_0 *x0, const ei_block_q4_0 *x1,
                                 const ei_block_q4_0 *x2, const ei_block_q8_0 *y,
                                 int32_t n, float *out0, float *out1, float *out2);
void ei_matmul_q4_0_q8_0_rows(const ei_tensor *w, const ei_block_q8_0 *xq, float *out,
                              int32_t row_begin, int32_t row_end);
void ei_matmul_q4_0_q8_0_rows3(const ei_tensor *w, const ei_block_q8_0 *xq, float *out,
                               int32_t row_begin, int32_t row_end);
void ei_matmul_q4_0_q8_0_batch_rows(const ei_tensor *w,
                                    const ei_block_q8_0 *inputs,
                                    int32_t n_inputs, float *out,
                                    int32_t row_begin, int32_t row_end);
void ei_matmul_q4_0_q8_0(const ei_tensor *w, const ei_block_q8_0 *xq, float *out);
void ei_matmul_q4_0_q8_0_dual_rows(const ei_tensor *w0, const ei_tensor *w1,
                                   const ei_block_q8_0 *xq, float *out0, float *out1,
                                   int32_t row_begin, int32_t row_end);
void ei_matmul_q4_0_q8_0_triple_rows(const ei_tensor *w0, const ei_tensor *w1,
                                     const ei_tensor *w2, const ei_block_q8_0 *xq,
                                     float *out0, float *out1, float *out2,
                                     int32_t row_begin, int32_t row_end);
void ei_matmul_q4_0_f32(const ei_tensor *w, const float *x, float *out, ei_block_q8_0 *tmp);
const char *ei_quants_kernel_variant(void);

#endif /* EI_QUANTS_H */
