/* Bounded adapter for the existing local Gaussian128-v1 payload.
 * rANS primitives: Fabian Giesen's public-domain rans_byte.h (see README).
 * No neural arithmetic, scale indexing, CDF quantization or stream ID changes.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "rans_byte.h"
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif
#define MAX_COUNT 8388608u
#define TABLES 128u
#define TOTAL 65536u
#define PRECISION 16u

API int glc_rans_abi(void) { return 1; }

static uint32_t be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}
static void put32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8); p[3] = (uint8_t)v;
}

/* All arrays are owned, sized and immutable in the Python wrapper. Validate
 * CDF bounds at every ABI call as well; a corrupt table must never reach rANS. */
static int valid_tables(const uint32_t *cdf, size_t cdf_n, const uint32_t *offset,
                        const int32_t *radius) {
    if (!cdf || !offset || !radius || offset[0] != 0 || offset[TABLES] != cdf_n) return 1;
    for (size_t t = 0; t < TABLES; ++t) {
        if (radius[t] < 1 || radius[t] > 1536) return 1;
        size_t a = offset[t], b = offset[t + 1];
        if (b <= a || b > cdf_n || b - a != (size_t)(2 * radius[t] + 3)) return 1;
        if (cdf[a] != 0 || cdf[b - 1] != TOTAL) return 1;
        for (size_t j = a + 1; j < b; ++j)
            if (cdf[j] <= cdf[j - 1] || cdf[j] > TOTAL) return 1;
    }
    return 0;
}

/* Maximum two renormalization bytes per symbol (16-bit precision, freq>=1),
 * four state bytes, and five bytes for each int32 escape. */
API int glc_rans_encode(const int32_t *values, const uint16_t *indexes, size_t count,
    const uint32_t *cdf, size_t cdf_n, const uint32_t *offset, const int32_t *radius,
    uint8_t *out, size_t capacity, size_t *written) {
    if (!values || !indexes || !out || !written || count > MAX_COUNT) return 1;
    if (valid_tables(cdf, cdf_n, offset, radius)) return 1;
    if (capacity < 16 + 7 * count) return 2;
    size_t rcap = 4 + 2 * count;
    uint8_t *escape_start = out + 12 + rcap, *escape = escape_start;
    uint32_t escapes = 0;
    for (size_t i = 0; i < count; ++i) {
        uint16_t t = indexes[i];
        if (t >= TABLES) return 1;
        int32_t v = values[i], r = radius[t];
        if (v < -r || v > r) {
            uint32_t z = (uint32_t)(v < 0 ? -(int64_t)v * 2 - 1 : (int64_t)v * 2);
            while (z >= 128) { *escape++ = (uint8_t)((z & 127) | 128); z >>= 7; }
            *escape++ = (uint8_t)z;
            ++escapes;
        }
    }
    RansState state;
    RansEncInit(&state);
    uint8_t *end = out + 12 + rcap, *ptr = end;
    for (size_t i = count; i-- > 0;) {
        uint16_t t = indexes[i];
        int32_t v = values[i], r = radius[t];
        uint32_t symbol = (v < -r || v > r) ? (uint32_t)(2 * r + 1) : (uint32_t)(v + r);
        const uint32_t *table = cdf + offset[t];
        RansEncPut(&state, &ptr, table[symbol], table[symbol + 1] - table[symbol], PRECISION);
    }
    RansEncFlush(&state, &ptr);
    size_t rlen = (size_t)(end - ptr), elen = (size_t)(escape - escape_start);
    memmove(out + 12, ptr, rlen);
    memmove(out + 12 + rlen, escape_start, elen);
    put32(out, (uint32_t)count); put32(out + 4, (uint32_t)rlen); put32(out + 8, escapes);
    *written = 12 + rlen + elen;
    return 0;
}

API int glc_rans_decode(const uint8_t *data, size_t length, const uint16_t *indexes, size_t count,
    const uint32_t *cdf, size_t cdf_n, const uint32_t *offset, const int32_t *radius,
    int32_t *out, size_t output_count) {
    if (!data || !indexes || !out || count > MAX_COUNT || output_count < count) return 1;
    if (valid_tables(cdf, cdf_n, offset, radius)) return 1;
    if (length < 16) return 3;
    uint32_t encoded_count = be32(data), rlen = be32(data + 4), escape_count = be32(data + 8);
    if (encoded_count != count || rlen < 4 || (size_t)rlen > length - 12 || escape_count > count) return 3;
    const uint8_t *ptr = data + 16, *rend = data + 12 + rlen, *eptr = rend, *end = data + length;
    /* Cast before shifts: avoid the signed-shift UB of a high top byte. */
    RansState state = (uint32_t)data[12] | ((uint32_t)data[13] << 8) |
                      ((uint32_t)data[14] << 16) | ((uint32_t)data[15] << 24);
    if (state < RANS_BYTE_L || state >= RANS_BYTE_L * 256u) return 4;
    uint32_t used = 0;
    for (size_t i = 0; i < count; ++i) {
        uint16_t t = indexes[i];
        if (t >= TABLES) return 1;
        const uint32_t *table = cdf + offset[t];
        uint32_t slot = RansDecGet(&state, PRECISION);
        size_t low = 0, high = offset[t + 1] - offset[t] - 1;
        while (low + 1 < high) {
            size_t mid = low + (high - low) / 2;
            if (table[mid] <= slot) low = mid; else high = mid;
        }
        uint32_t symbol = (uint32_t)low;
        RansDecAdvanceStep(&state, table[symbol], table[symbol + 1] - table[symbol], PRECISION);
        /* Upstream's unbounded decoder renormalization is deliberately not used. */
        while (state < RANS_BYTE_L) {
            if (ptr == rend) return 5;
            state = (state << 8) | *ptr++;
        }
        int32_t r = radius[t], value;
        if (symbol == (uint32_t)(2 * r + 1)) {
            uint64_t z = 0;
            unsigned shift;
            for (shift = 0; shift <= 28; shift += 7) {
                if (eptr == end) return 6;
                uint8_t byte = *eptr++;
                z |= (uint64_t)(byte & 127) << shift;
                if (!(byte & 128)) {
                    if (z > UINT32_MAX || (shift && byte == 0)) return 6;
                    break;
                }
            }
            if (shift > 28) return 6;
            int64_t signed_value = (z & 1) ? -(int64_t)(z >> 1) - 1 : (int64_t)(z >> 1);
            value = (int32_t)signed_value;
            if (value >= -r && value <= r) return 6;
            ++used;
        } else value = (int32_t)symbol - r;
        out[i] = value;
    }
    if (state != RANS_BYTE_L || ptr != rend || eptr != end || used != escape_count) return 7;
    return 0;
}
