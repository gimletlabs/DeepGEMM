#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cutlass/cutlass.h>

namespace deep_gemm::fp4 {

// Decode 8 packed e2m1 values (low nibble is the even element) with their UE8M0 scale exponent `e` into 8 BF16
// values `e2m1 * 2 ^ (e - 127)`, as 4 BF16x2 words in element order.
// NOTES: the hardware converts e2m1 pairs to exact F16; shifting an e2m1-valued F16 right by 3 gives BF16 bits of
// `value * 2 ^ -112` (F16 mantissas of e2m1 values only use their top bit, and every exponent stays normal), so one
// power-of-two BF16 multiply applies both `2 ^ 112` and the block scale. Exact for `e` in [0, 142], i.e. scales up to
// 2 ^ 15, which covers MXFP4 weights.
CUTLASS_DEVICE uint4 e2m1x8_to_bf16x8_scaled(const uint32_t& packed, const uint32_t& e) {
    uint32_t f16x2[4];
    asm("{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
        "}\n"
        : "=r"(f16x2[0]), "=r"(f16x2[1]), "=r"(f16x2[2]), "=r"(f16x2[3]) : "r"(packed));
    const uint32_t scale_bits = ((e + 112u) << 7) | ((e + 112u) << 23);
    const auto scale = *reinterpret_cast<const __nv_bfloat162*>(&scale_bits);
    uint4 result;
    auto out = reinterpret_cast<__nv_bfloat162*>(&result);
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++ i) {
        const uint32_t bits = ((f16x2[i] >> 3) & 0x0fff0fffu) | (f16x2[i] & 0x80008000u);
        out[i] = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&bits), scale);
    }
    return result;
}

} // namespace deep_gemm::fp4
