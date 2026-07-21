#ifndef EI_FLOAT_FORMAT_H
#define EI_FLOAT_FORMAT_H

#include <stddef.h>

#define EI_F32_TO_CHARS_CAPACITY 32u

/* Returns zero for non-finite values. The result is not NUL-terminated. */
size_t ei_f32_to_chars(char dst[static EI_F32_TO_CHARS_CAPACITY], float value);

#endif
