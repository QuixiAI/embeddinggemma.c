#include "float_format.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t bits_of(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    return bits;
}

static int check_bits(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof value);
    if (!isfinite(value)) return 0;

    char buffer[EI_F32_TO_CHARS_CAPACITY + 1u];
    const size_t len = ei_f32_to_chars(buffer, value);
    if (len == 0 || len > EI_F32_TO_CHARS_CAPACITY) {
        fprintf(stderr, "invalid length %zu for 0x%08x\n", len, bits);
        return 1;
    }
    buffer[len] = '\0';
    char *end = NULL;
    const float parsed = strtof(buffer, &end);
    if (end != buffer + len || bits_of(parsed) != bits) {
        fprintf(stderr, "round-trip mismatch 0x%08x -> %s -> 0x%08x\n",
                bits, buffer, bits_of(parsed));
        return 1;
    }
    return 0;
}

int main(void) {
    static const uint32_t cases[] = {
        0x00000000u, 0x80000000u, 0x00000001u, 0x007fffffu,
        0x00800000u, 0x3dccccddu, 0x3f000000u, 0x3f800000u,
        0x41200000u, 0x7f7fffffu, 0x80800000u, 0xff7fffffu,
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        if (check_bits(cases[i])) return 1;
    }

    uint32_t state = 0x6d2b79f5u;
    for (size_t i = 0; i < 1000000u; i++) {
        state ^= state << 13u;
        state ^= state >> 17u;
        state ^= state << 5u;
        if (check_bits(state)) return 1;
    }

    char buffer[EI_F32_TO_CHARS_CAPACITY];
    if (ei_f32_to_chars(buffer, INFINITY) != 0 ||
        ei_f32_to_chars(buffer, NAN) != 0) {
        fprintf(stderr, "non-finite values must be rejected\n");
        return 1;
    }
    puts("float formatter: 1,000,012 finite values round-trip exactly");
    return 0;
}
