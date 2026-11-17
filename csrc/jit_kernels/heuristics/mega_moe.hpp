#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_jit/utils/env.hpp>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "sm100.hpp"

#include <deep_gemm/heuristics/mega_moe.cuh>

namespace deep_gemm {

// We need to distribute work across locality domains, so we must use the whole GPU
static int get_num_sms_for_mega_moe() {
    const auto num_sms = runtime->get_num_sms();
    DG_HOST_ASSERT(num_sms == jit->device.get_num_sms());
    return num_sms;
}

static MmaKind parse_mma_kind(const std::string& mma_type_str) {
    if (mma_type_str == "bf16xbf16")
        return MmaKind::BF16;
    if (mma_type_str == "fp8xfp4" or mma_type_str == "fp8xfp8")
        return MmaKind::MXFP8FP4;
    if (mma_type_str == "mxf4xmxf4")
        return MmaKind::MXFP4;
    DG_HOST_ASSERT(mma_type_str == "nvfp4xnvfp4");
    return MmaKind::NVFP4;
}

static std::string to_mma_kind_name(const MmaKind& mma_kind) {
    switch (mma_kind) {
        case MmaKind::BF16:     return "MmaKind::BF16";
        case MmaKind::MXFP8FP4: return "MmaKind::MXFP8FP4";
        case MmaKind::MXFP4:    return "MmaKind::MXFP4";
        case MmaKind::NVFP4:    return "MmaKind::NVFP4";
    }
    DG_HOST_UNREACHABLE("Unknown MMA kind");
}

static MegaMoEConfig get_mega_moe_config(
    const int& num_ranks, const int& num_experts, const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank, const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_ring_tokens,
    const int& num_sf_ring_tokens,
    const MmaKind& mma_kind) {

    const auto config = compute_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank, num_max_tokens_per_rank,
        num_tokens, num_topk, hidden, intermediate_hidden, num_ring_tokens,
        num_sf_ring_tokens, mma_kind);

    // Print configs for the first time
    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS")) {
        const auto key = std::format(
            "MegaMoEConfig(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={})",
            num_ranks, num_experts, hidden, intermediate_hidden, num_max_tokens_per_rank, num_tokens,
            num_topk);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": " << config << std::endl;
            printed.insert(key);
        }
    }
    return config;
}

} // namespace deep_gemm
