#pragma once

#include <format>
#include <torch/torch.h>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../heuristics/mega_moe.hpp"
#include "mega_moe_runtime_utils.hpp"

namespace deep_gemm {

static void sm100_bf16_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l2_acts,
    const torch::Tensor& shared_l1_acts, const torch::Tensor& shared_l2_acts,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& shared_l1_weights, const torch::Tensor& shared_l2_weights,
    const torch::Tensor& sm_locality_domains,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_shared_experts,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math,
    const float* post_expert_norm_weight,
    const bool& post_expert_norm,
    const bool& post_expert_norm_weight_first,
    const float& post_expert_norm_eps,
    const bool& use_trtllm_weights = false
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));

    // Heuristics
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, 0, MmaKind::BF16);

    if (use_trtllm_weights) {
        DG_HOST_ASSERT(num_shared_experts == 0);
        DG_HOST_ASSERT(hidden % 64 == 0 and intermediate_hidden % 64 == 0);
        DG_HOST_ASSERT(config.block_k % 64 == 0 and config.block_n == 128);
    }
    DG_HOST_ASSERT(config.store_block_m_l1 == config.store_block_m_l2);

    const auto maps = mega_moe_tensor_map_builder().make_bf16_mega_moe_maps(
        config,
        mega_moe_tensor_view(l1_acts),
        mega_moe_tensor_view(l1_weights),
        mega_moe_tensor_view(l2_acts),
        mega_moe_tensor_view(l2_weights),
        num_shared_experts > 0 ? mega_moe_tensor_view(shared_l1_acts) : host::TensorView{},
        num_shared_experts > 0 ? mega_moe_tensor_view(shared_l1_weights) : host::TensorView{},
        num_shared_experts > 0 ? mega_moe_tensor_view(shared_l2_acts) : host::TensorView{},
        num_shared_experts > 0 ? mega_moe_tensor_view(shared_l2_weights) : host::TensorView{},
        hidden,
        intermediate_hidden,
        num_experts_per_rank,
        num_shared_experts,
        num_max_tokens_per_rank,
        use_trtllm_weights);

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    const auto num_sms = get_num_sms_for_mega_moe();

    // Compile
    const auto kernel = jit->compile("sm100_bf16_mega_moe", std::format(R"(
#include <deep_gemm/impls/sm100_bf16_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {}, {}, {},
        {},
        {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {},
        {}, {}, {}, {}, {}
    >);
}};
)", num_max_tokens_per_rank,
        hidden, intermediate_hidden,
        num_experts, num_shared_experts,
        num_topk,
        config.block_m, config.block_n, config.block_k,
        config.store_block_m_l1,
        config.num_ring_tokens,
        config.num_stages,
        config.num_bytes_per_pull,
        config.num_dispatch_threads, config.num_non_epilogue_threads, config.num_epilogue_threads,
        num_sms, num_ranks,
        to_string(activation_clamp),
        fast_math ? "true" : "false",
        post_expert_norm ? "true" : "false",
        post_expert_norm_weight_first ? "true" : "false",
        post_expert_norm_weight != nullptr ? "true" : "false",
        to_string(post_expert_norm_eps),
        use_trtllm_weights ? "true" : "false"));

    host::with_bf16_mega_moe_args(
        [&](const auto&... args) {
            jit->launch(
                kernel, {
                    .num_smem_bytes = config.smem_size,
                    .grid_dim = dim3(num_sms, 1, 1),
                    .block_dim = dim3(config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads, 1, 1),
                    .cluster_dim = dim3(2, 1, 1),
                },
                args...);
        }, maps, layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        y.data_ptr(), cumulative_local_expert_recv_stats_ptr, num_tokens,
        sm_locality_domains.data_ptr<uint8_t>(), post_expert_norm_weight);
}

// BF16 activations with MXFP4 weights (W4A16), decoded to BF16 in shared memory
static void sm100_bf16_fp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l2_acts,
    const torch::Tensor& l1_weights, const torch::Tensor& l1_weights_sf,
    const torch::Tensor& l2_weights, const torch::Tensor& l2_weights_sf,
    const torch::Tensor& sm_locality_domains,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math,
    const MegaMoEActivation& activation
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));

    // Heuristics
    const auto config = compute_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, 0, MmaKind::BF16, /*with_fp4_weights=*/true);
    if (deep_jit::get_env<int>("DG_PRINT_CONFIGS"))
        std::cout << "W4A16 " << config << std::endl;
    DG_HOST_ASSERT(config.store_block_m_l1 == config.store_block_m_l2);

    const auto maps = mega_moe_tensor_map_builder().make_bf16_fp4_mega_moe_maps(
        config,
        mega_moe_tensor_view(l1_acts),
        mega_moe_tensor_view(l1_weights), mega_moe_tensor_view(l1_weights_sf),
        mega_moe_tensor_view(l2_acts),
        mega_moe_tensor_view(l2_weights), mega_moe_tensor_view(l2_weights_sf),
        hidden, intermediate_hidden, num_experts_per_rank);

    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    const auto num_sms = get_num_sms_for_mega_moe();

    // Compile
    const auto kernel = jit->compile("sm100_bf16_fp4_mega_moe", std::format(R"(
#include <deep_gemm/impls/sm100_bf16_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_mega_moe_impl<
        {},
        {}, {},
        {}, 0,
        {}, {}, {}, {},
        {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {},
        false, false, false, 1e-6f, false,
        true, {}
    >);
}};
)", num_max_tokens_per_rank,
        hidden, intermediate_hidden,
        num_experts,
        num_topk,
        config.block_m, config.block_n, config.block_k,
        config.store_block_m_l1,
        config.num_ring_tokens,
        config.num_stages,
        config.num_bytes_per_pull,
        config.num_dispatch_threads, config.num_non_epilogue_threads, config.num_epilogue_threads,
        num_sms, num_ranks,
        to_string(activation_clamp),
        fast_math ? "true" : "false",
        activation == MegaMoEActivation::SiTU ? "MegaMoEActivation::SiTU" : "MegaMoEActivation::SwiGLU"));

    host::with_bf16_mega_moe_args(
        [&](const auto&... args) {
            jit->launch(
                kernel, {
                    .num_smem_bytes = config.smem_size,
                    .grid_dim = dim3(num_sms, 1, 1),
                    .block_dim = dim3(config.num_dispatch_threads + config.num_non_epilogue_threads +
                                      config.num_epilogue_threads + config.num_convert_threads, 1, 1),
                    .cluster_dim = dim3(2, 1, 1),
                },
                args...);
        }, maps, layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        y.data_ptr(), cumulative_local_expert_recv_stats_ptr, num_tokens,
        sm_locality_domains.data_ptr<uint8_t>(), static_cast<const float*>(nullptr));
}

} // namespace deep_gemm
