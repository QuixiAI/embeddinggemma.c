/*
 * Shortest round-trip float32 to chars.
 *
 * Digit generation is a float-only C11 port of the Ryu algorithm.
 * SPDX-FileCopyrightText: 2018 Ulf Adams
 * SPDX-License-Identifier: Apache-2.0 OR BSL-1.0
 *
 * Output presentation (decimal point and exponent layout) follows
 * nlohmann/json so responses keep the same textual format; Ryu may pick a
 * shorter digit string than Grisu2 for the rare values where Grisu2 is
 * suboptimal. Bitwise round-trip exactness is verified by test_float_format.
 * SPDX-FileCopyrightText: 2009 Florian Loitsch <https://florian.loitsch.com/>
 * SPDX-FileCopyrightText: 2013-2025 Niels Lohmann <https://nlohmann.me>
 * SPDX-License-Identifier: MIT
 *
 * The pow5 tables are generated exactly from big-integer arithmetic; see
 * https://github.com/ulfjack/ryu for the reference implementation.
 */

#include "float_format.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define EI_RYU_POW5_INV_BITCOUNT 59
#define EI_RYU_POW5_BITCOUNT 61

static const uint64_t EI_RYU_POW5_INV_SPLIT[31] = {
    576460752303423489ull, 461168601842738791ull, 368934881474191033ull,
    295147905179352826ull, 472236648286964522ull, 377789318629571618ull,
    302231454903657294ull, 483570327845851670ull, 386856262276681336ull,
    309485009821345069ull, 495176015714152110ull, 396140812571321688ull,
    316912650057057351ull, 507060240091291761ull, 405648192073033409ull,
    324518553658426727ull, 519229685853482763ull, 415383748682786211ull,
    332306998946228969ull, 531691198313966350ull, 425352958651173080ull,
    340282366920938464ull, 544451787073501542ull, 435561429658801234ull,
    348449143727040987ull, 557518629963265579ull, 446014903970612463ull,
    356811923176489971ull, 570899077082383953ull, 456719261665907162ull,
    365375409332725730ull,
};

static const uint64_t EI_RYU_POW5_SPLIT[47] = {
    1152921504606846976ull, 1441151880758558720ull, 1801439850948198400ull,
    2251799813685248000ull, 1407374883553280000ull, 1759218604441600000ull,
    2199023255552000000ull, 1374389534720000000ull, 1717986918400000000ull,
    2147483648000000000ull, 1342177280000000000ull, 1677721600000000000ull,
    2097152000000000000ull, 1310720000000000000ull, 1638400000000000000ull,
    2048000000000000000ull, 1280000000000000000ull, 1600000000000000000ull,
    2000000000000000000ull, 1250000000000000000ull, 1562500000000000000ull,
    1953125000000000000ull, 1220703125000000000ull, 1525878906250000000ull,
    1907348632812500000ull, 1192092895507812500ull, 1490116119384765625ull,
    1862645149230957031ull, 1164153218269348144ull, 1455191522836685180ull,
    1818989403545856475ull, 2273736754432320594ull, 1421085471520200371ull,
    1776356839400250464ull, 2220446049250313080ull, 1387778780781445675ull,
    1734723475976807094ull, 2168404344971008868ull, 1355252715606880542ull,
    1694065894508600678ull, 2117582368135750847ull, 1323488980084844279ull,
    1654361225106055349ull, 2067951531382569187ull, 1292469707114105741ull,
    1615587133892632177ull, 2019483917365790221ull,
};

static inline uint32_t ryu_pow5bits(int32_t e) {
    return (uint32_t)(((e * 1217359) >> 19) + 1);
}

static inline uint32_t ryu_log10_pow2(int32_t e) {
    return (uint32_t)((e * 78913) >> 18);
}

static inline uint32_t ryu_log10_pow5(int32_t e) {
    return (uint32_t)((e * 732923) >> 20);
}

static inline uint32_t ryu_pow5_factor(uint32_t value) {
    uint32_t count = 0;
    for (;;) {
        if (value == 0) return 0;
        if (value % 5 != 0) return count;
        value /= 5;
        count++;
    }
}

static inline bool ryu_multiple_of_pow5(uint32_t value, uint32_t p) {
    return ryu_pow5_factor(value) >= p;
}

static inline bool ryu_multiple_of_pow2(uint32_t value, uint32_t p) {
    return (value & ((1u << p) - 1u)) == 0;
}

static inline uint32_t ryu_mul_shift(uint32_t m, uint64_t factor,
                                     int32_t shift) {
    const uint64_t factor_lo = (uint32_t)factor;
    const uint64_t factor_hi = factor >> 32;
    const uint64_t bits0 = (uint64_t)m * factor_lo;
    const uint64_t bits1 = (uint64_t)m * factor_hi;
    return (uint32_t)(((bits0 >> 32) + bits1) >> (shift - 32));
}

typedef struct {
    uint32_t digits;
    int32_t exponent;
} ryu_decimal;

static ryu_decimal ryu_f2d(uint32_t ieee_mantissa, uint32_t ieee_exponent) {
    enum { mantissa_bits = 23, bias = 127 };
    int32_t e2;
    uint32_t m2;
    if (ieee_exponent == 0) {
        e2 = 1 - bias - mantissa_bits - 2;
        m2 = ieee_mantissa;
    } else {
        e2 = (int32_t)ieee_exponent - bias - mantissa_bits - 2;
        m2 = (1u << mantissa_bits) | ieee_mantissa;
    }
    const bool even = (m2 & 1) == 0;
    const bool accept_bounds = even;

    const uint32_t mv = 4 * m2;
    const uint32_t mp = 4 * m2 + 2;
    const uint32_t mm_shift = (ieee_mantissa != 0 || ieee_exponent <= 1) ? 1 : 0;
    const uint32_t mm = 4 * m2 - 1 - mm_shift;

    uint32_t vr, vp, vm;
    int32_t e10;
    bool vm_trailing_zeros = false;
    bool vr_trailing_zeros = false;
    uint32_t last_removed_digit = 0;
    if (e2 >= 0) {
        const uint32_t q = ryu_log10_pow2(e2);
        e10 = (int32_t)q;
        const int32_t k =
            EI_RYU_POW5_INV_BITCOUNT + (int32_t)ryu_pow5bits((int32_t)q) - 1;
        const int32_t i = -e2 + (int32_t)q + k;
        vr = ryu_mul_shift(mv, EI_RYU_POW5_INV_SPLIT[q], i);
        vp = ryu_mul_shift(mp, EI_RYU_POW5_INV_SPLIT[q], i);
        vm = ryu_mul_shift(mm, EI_RYU_POW5_INV_SPLIT[q], i);
        if (q != 0 && (vp - 1) / 10 <= vm / 10) {
            const int32_t l = EI_RYU_POW5_INV_BITCOUNT +
                (int32_t)ryu_pow5bits((int32_t)q - 1) - 1;
            last_removed_digit = ryu_mul_shift(
                mv, EI_RYU_POW5_INV_SPLIT[q - 1],
                -e2 + (int32_t)q - 1 + l) % 10;
        }
        if (q <= 9) {
            if (mv % 5 == 0) {
                vr_trailing_zeros = ryu_multiple_of_pow5(mv, q);
            } else if (accept_bounds) {
                vm_trailing_zeros = ryu_multiple_of_pow5(mm, q);
            } else {
                vp -= ryu_multiple_of_pow5(mp, q);
            }
        }
    } else {
        const uint32_t q = ryu_log10_pow5(-e2);
        e10 = (int32_t)q + e2;
        const int32_t i = -e2 - (int32_t)q;
        const int32_t k = (int32_t)ryu_pow5bits(i) - EI_RYU_POW5_BITCOUNT;
        int32_t j = (int32_t)q - k;
        vr = ryu_mul_shift(mv, EI_RYU_POW5_SPLIT[i], j);
        vp = ryu_mul_shift(mp, EI_RYU_POW5_SPLIT[i], j);
        vm = ryu_mul_shift(mm, EI_RYU_POW5_SPLIT[i], j);
        if (q != 0 && (vp - 1) / 10 <= vm / 10) {
            j = (int32_t)q - 1 -
                ((int32_t)ryu_pow5bits(i + 1) - EI_RYU_POW5_BITCOUNT);
            last_removed_digit =
                ryu_mul_shift(mv, EI_RYU_POW5_SPLIT[i + 1], j) % 10;
        }
        if (q <= 1) {
            vr_trailing_zeros = true;
            if (accept_bounds) {
                vm_trailing_zeros = mm_shift == 1;
            } else {
                --vp;
            }
        } else if (q < 31) {
            vr_trailing_zeros = ryu_multiple_of_pow2(mv, q - 1);
        }
    }

    int32_t removed = 0;
    uint32_t output;
    if (vm_trailing_zeros || vr_trailing_zeros) {
        while (vp / 10 > vm / 10) {
            vm_trailing_zeros &= vm % 10 == 0;
            vr_trailing_zeros &= last_removed_digit == 0;
            last_removed_digit = vr % 10;
            vr /= 10;
            vp /= 10;
            vm /= 10;
            ++removed;
        }
        if (vm_trailing_zeros) {
            while (vm % 10 == 0) {
                vr_trailing_zeros &= last_removed_digit == 0;
                last_removed_digit = vr % 10;
                vr /= 10;
                vp /= 10;
                vm /= 10;
                ++removed;
            }
        }
        if (vr_trailing_zeros && last_removed_digit == 5 && vr % 2 == 0) {
            last_removed_digit = 4;
        }
        output = vr +
            ((vr == vm && (!accept_bounds || !vm_trailing_zeros)) ||
             last_removed_digit >= 5);
    } else {
        while (vp / 10 > vm / 10) {
            last_removed_digit = vr % 10;
            vr /= 10;
            vp /= 10;
            vm /= 10;
            ++removed;
        }
        output = vr + (vr == vm || last_removed_digit >= 5);
    }
    return (ryu_decimal){output, e10 + removed};
}

static char *append_exponent(char *buf, int exponent) {
    if (exponent < 0) {
        exponent = -exponent;
        *buf++ = '-';
    } else {
        *buf++ = '+';
    }
    uint32_t value = (uint32_t)exponent;
    if (value < 10u) {
        *buf++ = '0';
        *buf++ = (char)('0' + value);
    } else if (value < 100u) {
        *buf++ = (char)('0' + value / 10u);
        *buf++ = (char)('0' + value % 10u);
    } else {
        *buf++ = (char)('0' + value / 100u);
        value %= 100u;
        *buf++ = (char)('0' + value / 10u);
        *buf++ = (char)('0' + value % 10u);
    }
    return buf;
}

static char *format_buffer(char *buf, int len, int decimal_exponent) {
    enum { min_exp = -4, max_exp = 6 };
    const int k = len;
    const int n = len + decimal_exponent;
    if (k <= n && n <= max_exp) {
        memset(buf + k, '0', (size_t)(n - k));
        buf[n] = '.';
        buf[n + 1] = '0';
        return buf + n + 2;
    }
    if (0 < n && n <= max_exp) {
        memmove(buf + n + 1, buf + n, (size_t)(k - n));
        buf[n] = '.';
        return buf + k + 1;
    }
    if (min_exp < n && n <= 0) {
        memmove(buf + 2 - n, buf, (size_t)k);
        buf[0] = '0';
        buf[1] = '.';
        memset(buf + 2, '0', (size_t)-n);
        return buf + 2 - n + k;
    }
    if (k == 1) {
        buf++;
    } else {
        memmove(buf + 2, buf + 1, (size_t)(k - 1));
        buf[1] = '.';
        buf += k + 1;
    }
    *buf++ = 'e';
    return append_exponent(buf, n - 1);
}

size_t ei_f32_to_chars(char dst[static EI_F32_TO_CHARS_CAPACITY], float value) {
    if (!isfinite(value)) return 0;
    char *first = dst;
    if (signbit(value)) {
        value = -value;
        *first++ = '-';
    }
    if (value == 0.0f) {
        *first++ = '0';
        *first++ = '.';
        *first++ = '0';
        return (size_t)(first - dst);
    }
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    const ryu_decimal decimal =
        ryu_f2d(bits & 0x7FFFFFu, bits >> 23);
    char reversed[10];
    int len = 0;
    uint32_t digits = decimal.digits;
    while (digits != 0) {
        reversed[len++] = (char)('0' + digits % 10);
        digits /= 10;
    }
    for (int i = 0; i < len; i++) {
        first[i] = reversed[len - 1 - i];
    }
    return (size_t)(format_buffer(first, len, decimal.exponent) - dst);
}
