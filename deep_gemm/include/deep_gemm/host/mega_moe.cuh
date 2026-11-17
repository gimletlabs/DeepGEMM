#pragma once

#include <array>
#include <cstdint>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <deep_gemm/heuristics/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <stdexcept>
#include <string>
#include <utility>

namespace deep_gemm::host {

struct TensorView {
    void *data = nullptr;
    CUtensorMapDataType dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    int bytes = 1;
    int rank = 0;
    std::array<int64_t, 4> sizes{}, strides{};
    bool packed_fp4 = false;
    bool contiguous = true;

    void *data_ptr() const { return data; }
    int element_size() const { return bytes; }
    int dim() const { return rank; }
    int64_t size(int i) const { return sizes[i < 0 ? i + rank : i]; }
    int64_t stride(int i) const { return strides[i < 0 ? i + rank : i]; }
    bool is_contiguous() const { return contiguous; }
    TensorView as_bytes() const {
        DG_HOST_ASSERT(bytes == 1);
        auto view = *this;
        view.dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
        view.packed_fp4 = false;
        return view;
    }
};

struct MegaMoETensorMaps {
    CUtensorMap tensor_map_l1_acts{};
    CUtensorMap tensor_map_l1_acts_sf{};
    CUtensorMap tensor_map_l1_weights{};
    CUtensorMap tensor_map_l1_weights_sf{};
    CUtensorMap tensor_map_l1_output{};
    CUtensorMap tensor_map_l2_acts{};
    CUtensorMap tensor_map_l2_acts_sf{};
    CUtensorMap tensor_map_l2_weights{};
    CUtensorMap tensor_map_l2_weights_sf{};
    CUtensorMap tensor_map_shared_l1_acts{};
    CUtensorMap tensor_map_shared_l1_acts_sf{};
    CUtensorMap tensor_map_shared_l1_weights{};
    CUtensorMap tensor_map_shared_l1_weights_sf{};
    CUtensorMap tensor_map_shared_l1_output{};
    CUtensorMap tensor_map_shared_l2_acts{};
    CUtensorMap tensor_map_shared_l2_acts_sf{};
    CUtensorMap tensor_map_shared_l2_weights{};
    CUtensorMap tensor_map_shared_l2_weights_sf{};
};

class TensorMapBuilder {
    PFN_cuTensorMapEncodeTiled_v12000 encode_;
    int num_domains_;
    void check(CUresult result) const {
        if (result != CUDA_SUCCESS)
            throw std::runtime_error("MegaMoE tensor map encoding failed (CUDA error " + std::to_string(result) + ")");
    }
    static int ceil_div(int a, int b) { return (a + b - 1) / b; }
    static int get_tma_aligned_size(int x, int bytes) {
        DG_HOST_ASSERT(16 % bytes == 0);
        return ceil_div(x, 16 / bytes) * (16 / bytes);
    }
    static CUtensorMapDataType dtype(const TensorView &t, bool allow_tf32, bool unpacked) {
        if (t.packed_fp4)
            return unpacked ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B : CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
        return allow_tf32 && t.dtype == CU_TENSOR_MAP_DATA_TYPE_FLOAT32 ? CU_TENSOR_MAP_DATA_TYPE_TFLOAT32 : t.dtype;
    }

public:
    explicit TensorMapBuilder(PFN_cuTensorMapEncodeTiled_v12000 encode, int num_domains = 2)
        : encode_(encode), num_domains_(num_domains) {}
    static CUtensorMapSwizzle mode_into_tensor_map_swizzle(const int &mode, const int &base) {
        if (base != 0) {
            DG_HOST_ASSERT(base == 32 and mode == 128);
            return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B;
        }

        DG_HOST_ASSERT(base == 0);
        switch (mode) {
        case 0:
        case 16:
            return CU_TENSOR_MAP_SWIZZLE_NONE;
        case 32:
            return CU_TENSOR_MAP_SWIZZLE_32B;
        case 64:
            return CU_TENSOR_MAP_SWIZZLE_64B;
        case 128:
            return CU_TENSOR_MAP_SWIZZLE_128B;
        default:
            throw std::runtime_error("Unsupported swizzling mode");
        }
    }

    CUtensorMap make_tma_2d_desc(const TensorView &t, int gmem_inner_dim, int gmem_outer_dim, int smem_inner_dim,
                                 int smem_outer_dim, const int &gmem_outer_stride, const int &swizzle_mode,
                                 const int &swizzle_base = 0, const bool &allow_tf32 = false,
                                 const bool &fp4_unpacked_smem = true) {
        const auto elem_size = static_cast<int>(t.element_size());
        const auto num_gmem_outer_stride_bytes =
            static_cast<uint32_t>(gmem_outer_stride) * static_cast<uint32_t>(elem_size);
        DG_HOST_ASSERT(reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 16u == 0u);
        DG_HOST_ASSERT(num_gmem_outer_stride_bytes % 16u == 0u);
        if (swizzle_mode != 0)
            smem_inner_dim = swizzle_mode / elem_size;

        if (t.packed_fp4) {
            // Inner dim must be a multiple of 64B for .b4x16_p64
            DG_HOST_ASSERT(not fp4_unpacked_smem or gmem_inner_dim % 128 == 0);

            // Fix FP4 packed smem
            if (not fp4_unpacked_smem and swizzle_mode != 0)
                smem_inner_dim = swizzle_mode * 2;
        }

        CUtensorMap tensor_map;
        const cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(gmem_inner_dim),
                                         static_cast<cuuint64_t>(gmem_outer_dim)};
        const cuuint32_t smem_dims[2] = {static_cast<cuuint32_t>(smem_inner_dim),
                                         static_cast<cuuint32_t>(smem_outer_dim)};
        const cuuint64_t gmem_strides[1] = {
            static_cast<cuuint64_t>(gmem_outer_stride * elem_size),
        };
        const cuuint32_t elem_strides[2] = {1, 1};
        check(encode_(&tensor_map, dtype(t, allow_tf32, fp4_unpacked_smem), 2, t.data_ptr(), gmem_dims, gmem_strides,
                      smem_dims, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                      mode_into_tensor_map_swizzle(swizzle_mode, swizzle_base), CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        return tensor_map;
    }

    CUtensorMap make_tma_4d_desc(const TensorView &t, int gmem_dim_0, int gmem_dim_1, int gmem_dim_2, int gmem_dim_3,
                                 int smem_dim_0, int smem_dim_1, int smem_dim_2, int smem_dim_3,
                                 const int64_t &gmem_stride_0, const int64_t &gmem_stride_1,
                                 const int64_t &gmem_stride_2, const int &swizzle_mode, const int &swizzle_base = 0,
                                 const bool &allow_tf32 = false, const bool &fp4_unpacked_smem = true) {
        const auto elem_size = static_cast<int>(t.element_size());
        const auto num_gmem_stride_0_bytes = static_cast<cuuint64_t>(gmem_stride_0) * elem_size;
        const auto num_gmem_stride_1_bytes = static_cast<cuuint64_t>(gmem_stride_1) * elem_size;
        const auto num_gmem_stride_2_bytes = static_cast<cuuint64_t>(gmem_stride_2) * elem_size;
        DG_HOST_ASSERT(reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 16u == 0u);
        DG_HOST_ASSERT(num_gmem_stride_0_bytes % 16u == 0u);
        DG_HOST_ASSERT(num_gmem_stride_1_bytes % 16u == 0u);
        DG_HOST_ASSERT(num_gmem_stride_2_bytes % 16u == 0u);
        if (swizzle_mode != 0)
            smem_dim_0 = swizzle_mode / elem_size;

        if (t.packed_fp4) {
            // Inner dim must be a multiple of 64B for .b4x16_p64
            DG_HOST_ASSERT(not fp4_unpacked_smem or gmem_dim_0 % 128 == 0);

            // Fix fp4 packed smem
            if (not fp4_unpacked_smem and swizzle_mode != 0)
                smem_dim_0 = swizzle_mode * 2;
        }

        CUtensorMap tensor_map;
        const cuuint64_t gmem_dims[4] = {static_cast<cuuint64_t>(gmem_dim_0), static_cast<cuuint64_t>(gmem_dim_1),
                                         static_cast<cuuint64_t>(gmem_dim_2), static_cast<cuuint64_t>(gmem_dim_3)};
        const cuuint32_t smem_dims[4] = {static_cast<cuuint32_t>(smem_dim_0), static_cast<cuuint32_t>(smem_dim_1),
                                         static_cast<cuuint32_t>(smem_dim_2), static_cast<cuuint32_t>(smem_dim_3)};
        const cuuint64_t gmem_strides[3] = {num_gmem_stride_0_bytes, num_gmem_stride_1_bytes, num_gmem_stride_2_bytes};
        const cuuint32_t elem_strides[4] = {1, 1, 1, 1};
        check(encode_(&tensor_map, dtype(t, allow_tf32, fp4_unpacked_smem), 4, t.data_ptr(), gmem_dims, gmem_strides,
                      smem_dims, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                      mode_into_tensor_map_swizzle(swizzle_mode, swizzle_base), CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        return tensor_map;
    }

    CUtensorMap make_tma_weights_2d_desc(const TensorView &t, const int &block_k, const int &load_block_n,
                                         const int &swizzle_mode) {
        const auto num_domains = num_domains_;
        const bool localized = t.dim() == 3;
        const auto n_per_domain = static_cast<int>(localized ? t.size(1) : t.size(0) / num_domains);
        const auto k = static_cast<int>(t.size(-1)) * (t.packed_fp4 ? 2 : 1);
        const int64_t slice_stride = localized ? t.stride(0) : n_per_domain * t.stride(-2);
        return make_tma_4d_desc(t, k, n_per_domain, num_domains, 1, block_k, load_block_n, 1, 1, t.stride(-2),
                                slice_stride, slice_stride * num_domains, swizzle_mode);
    }

    CUtensorMap make_tma_weights_3d_desc(const TensorView &t, const int &block_k, const int &load_block_n,
                                         const int &swizzle_mode) {
        const auto num_domains = num_domains_;
        const bool localized = t.dim() == 4;
        const auto num_batches = static_cast<int>(t.size(localized ? 1 : 0));
        const auto n_per_domain = static_cast<int>(localized ? t.size(2) : t.size(1) / num_domains);
        const auto k = static_cast<int>(t.size(-1)) * (t.packed_fp4 ? 2 : 1);
        const int64_t slice_stride = localized ? t.stride(0) : n_per_domain * t.stride(-2);
        return make_tma_4d_desc(t, k, n_per_domain, num_domains, num_batches, block_k, load_block_n, 1, 1, t.stride(-2),
                                slice_stride, t.stride(localized ? 1 : 0), swizzle_mode);
    }

    CUtensorMap make_tma_sf_desc(const cute::UMMA::Major &major, const TensorView &t, int shape_mn, int shape_k,
                                 const int &block_mn, const int &gran_k, const int &num_groups, const int &swizzle_mode,
                                 const int &swizzle_base = 0, const bool &allow_tf32 = false,
                                 const int &smem_outer_dim = 1, const int &sf_k_stride = 0) {
        DG_HOST_ASSERT(major == cute::UMMA::Major::MN);

        // TODO: maybe swizzle SF as well
        DG_HOST_ASSERT(swizzle_mode == 0);

        shape_mn = get_tma_aligned_size(shape_mn, static_cast<int>(t.element_size()));
        // A zero K stride denotes the compact, TMA-aligned layout.
        const auto gmem_outer_stride = sf_k_stride == 0 ? shape_mn : sf_k_stride;
        DG_HOST_ASSERT(gmem_outer_stride >= shape_mn);
        return make_tma_2d_desc(
            t, shape_mn, ceil_div(shape_k, gran_k * (t.dtype == CU_TENSOR_MAP_DATA_TYPE_FLOAT32 ? 1 : 4)) * num_groups,
            block_mn, smem_outer_dim, gmem_outer_stride, swizzle_mode, swizzle_base, allow_tf32);
    }

    CUtensorMap make_trtllm_bf16_weight_tma_desc(const TensorView &weights, const int rows, const bool is_l1) {
        DG_HOST_ASSERT(weights.is_contiguous() and rows % 128 == 0);
        CUtensorMap result;
        const cuuint64_t dims[5] = {64, is_l1 ? 2u : 4u, is_l1 ? 4u : 8u,
                                    is_l1 ? 2u : static_cast<cuuint64_t>(rows / 32), static_cast<cuuint64_t>(rows / 4)};
        const cuuint64_t strides[4] = {is_l1 ? 16u * 128u : 8u * 128u, 128, is_l1 ? 8u * 128u : 32u * 128u, 4u * 128u};
        const cuuint32_t box[5] = {64, is_l1 ? 2u : 4u, is_l1 ? 4u : 8u, is_l1 ? 2u : 4u, 2};
        const cuuint32_t element_strides[5] = {1, 1, 1, 1, 1};
        check(encode_(&result, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, is_l1 ? 5 : 4, weights.data_ptr(), dims, strides, box,
                      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        return result;
    }

    CUtensorMap make_trtllm_fp4_l2_tma_desc(const TensorView &weights, const int inner_bytes, const int rows) {
        DG_HOST_ASSERT(weights.is_contiguous() and rows % 128 == 0);
        DG_HOST_ASSERT(inner_bytes % 128 == 0);
        CUtensorMap result;
        const cuuint64_t dims[4] = {static_cast<cuuint64_t>(inner_bytes), 4, 8, static_cast<cuuint64_t>(rows / 32)};
        const cuuint64_t strides[3] = {8ull * inner_bytes, static_cast<cuuint64_t>(inner_bytes), 32ull * inner_bytes};
        const cuuint32_t box[4] = {128, 4, 8, 4};
        const cuuint32_t element_strides[4] = {1, 1, 1, 1};
        check(encode_(&result, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, weights.data_ptr(), dims, strides, box,
                      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        return result;
    }

    CUtensorMap make_trtllm_fp4_sf_tma_desc(const TensorView &sf, const bool &is_l1, const int &num_row_blocks,
                                            const int &num_k_chunks, const int &box_k_chunks,
                                            const int &box_row_blocks) {
        DG_HOST_ASSERT(num_row_blocks > 0 and num_k_chunks > 0);
        DG_HOST_ASSERT(box_k_chunks <= num_k_chunks and box_row_blocks <= num_row_blocks);
        CUtensorMap result;
        const auto row_block_stride = static_cast<cuuint64_t>(num_k_chunks) * 512ull;
        const cuuint32_t element_strides[5] = {1, 1, 1, 1, 1};
        if (is_l1) {
            const cuuint64_t dims[3] = {128, static_cast<cuuint64_t>(num_k_chunks),
                                        static_cast<cuuint64_t>(num_row_blocks)};
            const cuuint64_t strides[2] = {512ull, row_block_stride};
            const cuuint32_t box[3] = {128, static_cast<cuuint32_t>(box_k_chunks),
                                       static_cast<cuuint32_t>(box_row_blocks)};
            check(encode_(&result, CU_TENSOR_MAP_DATA_TYPE_UINT32, 3, sf.data_ptr(), dims, strides, box,
                          element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                          CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        } else {
            const cuuint64_t dims[5] = {4, 4, 8, static_cast<cuuint64_t>(num_k_chunks),
                                        static_cast<cuuint64_t>(num_row_blocks)};
            const cuuint64_t strides[4] = {128ull, 16ull, 512ull, row_block_stride};
            const cuuint32_t box[5] = {4, 4, 8, static_cast<cuuint32_t>(box_k_chunks),
                                       static_cast<cuuint32_t>(box_row_blocks)};
            check(encode_(&result, CU_TENSOR_MAP_DATA_TYPE_UINT32, 5, sf.data_ptr(), dims, strides, box,
                          element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                          CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        }
        return result;
    }

    MegaMoETensorMaps make_bf16_mega_moe_maps(const MegaMoEConfig &config, const TensorView &l1_acts,
                                              const TensorView &l1_weights, const TensorView &l2_acts,
                                              const TensorView &l2_weights, const TensorView &shared_l1_acts,
                                              const TensorView &shared_l1_weights, const TensorView &shared_l2_acts,
                                              const TensorView &shared_l2_weights, int hidden, int intermediate_hidden,
                                              int num_experts_per_rank, int num_shared_experts,
                                              int num_max_tokens_per_rank, bool use_trtllm_weights) {
        const int shared_intermediate_hidden = intermediate_hidden * num_shared_experts;
        MegaMoETensorMaps maps;
        // Make tensormap
        maps.tensor_map_l1_acts =
            make_tma_2d_desc(l1_acts, hidden, config.num_ring_tokens, config.block_k, config.load_block_m,
                             static_cast<int>(l1_acts.stride(-2)), config.swizzle_acts_mode);
        // BlockMajorK storage: [expert, K/64, row, 64].
        maps.tensor_map_l1_weights =
            use_trtllm_weights ? make_trtllm_bf16_weight_tma_desc(
                                     l1_weights, num_experts_per_rank * (hidden / 64) * intermediate_hidden * 2, true)
                               : make_tma_weights_3d_desc(l1_weights, config.block_k, config.load_block_n,
                                                          config.swizzle_weights_mode);
        maps.tensor_map_l1_output =
            make_tma_2d_desc(l2_acts, intermediate_hidden, config.num_ring_tokens, config.block_n / 2,
                             config.store_block_m_l1, static_cast<int>(l2_acts.stride(-2)), config.swizzle_acts_mode);
        maps.tensor_map_l2_acts =
            make_tma_2d_desc(l2_acts, intermediate_hidden, config.num_ring_tokens, config.block_k, config.load_block_m,
                             static_cast<int>(l2_acts.stride(-2)), config.swizzle_acts_mode);
        maps.tensor_map_l2_weights =
            use_trtllm_weights ? make_trtllm_bf16_weight_tma_desc(
                                     l2_weights, num_experts_per_rank * (intermediate_hidden / 64) * hidden, false)
                               : make_tma_weights_3d_desc(l2_weights, config.block_k, config.load_block_n,
                                                          config.swizzle_weights_mode);

        maps.tensor_map_shared_l1_acts =
            num_shared_experts > 0
                ? make_tma_2d_desc(shared_l1_acts, hidden, num_max_tokens_per_rank, config.block_k, config.load_block_m,
                                   static_cast<int>(shared_l1_acts.stride(-2)), config.swizzle_acts_mode)
                : maps.tensor_map_l1_acts;
        maps.tensor_map_shared_l1_weights =
            num_shared_experts > 0 ? make_tma_weights_2d_desc(shared_l1_weights, config.block_k, config.load_block_n,
                                                              config.swizzle_weights_mode)
                                   : maps.tensor_map_l1_weights;
        maps.tensor_map_shared_l1_output =
            num_shared_experts > 0
                ? make_tma_2d_desc(shared_l2_acts, shared_intermediate_hidden, num_max_tokens_per_rank,
                                   config.block_n / 2, config.store_block_m_l1,
                                   static_cast<int>(shared_l2_acts.stride(-2)), config.swizzle_acts_mode)
                : maps.tensor_map_l1_output;
        maps.tensor_map_shared_l2_acts =
            num_shared_experts > 0
                ? make_tma_2d_desc(shared_l2_acts, shared_intermediate_hidden, num_max_tokens_per_rank, config.block_k,
                                   config.load_block_m, static_cast<int>(shared_l2_acts.stride(-2)),
                                   config.swizzle_acts_mode)
                : maps.tensor_map_l2_acts;
        maps.tensor_map_shared_l2_weights =
            num_shared_experts > 0 ? make_tma_weights_2d_desc(shared_l2_weights, config.block_k, config.load_block_n,
                                                              config.swizzle_weights_mode)
                                   : maps.tensor_map_l2_weights;

        return maps;
    }

    MegaMoETensorMaps make_fp4_mega_moe_maps(const MegaMoEConfig &config, const TensorView &l1_acts,
                                             const TensorView &l1_acts_sf, const TensorView &l1_weights,
                                             const TensorView &l1_weights_sf, const TensorView &l2_acts,
                                             const TensorView &l2_acts_sf, const TensorView &l2_weights,
                                             const TensorView &l2_weights_sf, const TensorView &shared_l1_acts,
                                             const TensorView &shared_l1_acts_sf, const TensorView &shared_l1_weights,
                                             const TensorView &shared_l1_weights_sf, const TensorView &shared_l2_acts,
                                             const TensorView &shared_l2_acts_sf, const TensorView &shared_l2_weights,
                                             const TensorView &shared_l2_weights_sf, int hidden,
                                             int intermediate_hidden, int num_experts_per_rank, int num_shared_experts,
                                             int num_max_tokens_per_rank, bool use_trtllm_weights, MmaKind mma_kind) {
        const int shared_intermediate_hidden = intermediate_hidden * num_shared_experts;
        MegaMoETensorMaps maps;
        // Make tensormap
        const bool is_packed_fp4 = mma_kind == MmaKind::NVFP4 or mma_kind == MmaKind::MXFP4;
        const int kGranK = config.gran_k;
        const int elem_bits = get_element_bits(mma_kind);
        const auto to_inner = [=](const int &num_elems) { return num_elems * elem_bits / 8; };
        const auto weight_tensor = [=](const TensorView &t) { return is_packed_fp4 ? t.as_bytes() : t; };
        const auto weight_inner = [=](const int &num_elems) { return is_packed_fp4 ? num_elems / 2 : num_elems; };
        const int block_k_inner = to_inner(config.block_k);
        const int sf_smem_outer_dim = config.block_k / (kGranK * 4);
        maps.tensor_map_l1_acts =
            make_tma_2d_desc(l1_acts, to_inner(hidden), config.num_ring_tokens, block_k_inner, config.load_block_m,
                             static_cast<int>(l1_acts.stride(-2)), config.swizzle_acts_mode);
        maps.tensor_map_l1_acts_sf =
            make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf, config.num_sf_ring_tokens, hidden, config.sf_block_m,
                             kGranK, 1, 0, 0, false, sf_smem_outer_dim, static_cast<int>(l1_acts_sf.stride(-1)));
        maps.tensor_map_l1_weights =
            not is_packed_fp4
                ? make_tma_weights_3d_desc(l1_weights, config.block_k, config.load_block_n, config.swizzle_weights_mode)
                : make_tma_2d_desc(weight_tensor(l1_weights), weight_inner(hidden),
                                   num_experts_per_rank * intermediate_hidden * 2, block_k_inner, config.load_block_n,
                                   static_cast<int>(l1_weights.stride(-2)), config.swizzle_weights_mode, 0, false,
                                   not is_packed_fp4);
        maps.tensor_map_l1_weights_sf =
            use_trtllm_weights
                ? make_trtllm_fp4_sf_tma_desc(l1_weights_sf, true, num_experts_per_rank * intermediate_hidden * 2 / 128,
                                              hidden / (kGranK * 4), sf_smem_outer_dim, config.block_n / 128)
                : make_tma_sf_desc(cute::UMMA::Major::MN, l1_weights_sf, intermediate_hidden * 2, hidden,
                                   config.block_n, kGranK, num_experts_per_rank, 0, 0, false, sf_smem_outer_dim,
                                   static_cast<int>(l1_weights_sf.stride(-1)));
        // NOTES: L1 output and L2 activations are essentially the same tensor.
        // Post-SwiGLU output has half the N width (`BLOCK_N / 2` per input tile),
        // so the swizzle mode is also halved (128 -> 64).
        const int l1_out_block_n_bytes = to_inner(config.block_n / 2);
        maps.tensor_map_l1_output =
            make_tma_2d_desc(l2_acts, to_inner(intermediate_hidden), config.num_ring_tokens, l1_out_block_n_bytes,
                             config.store_block_m_l1, static_cast<int>(l2_acts.stride(-2)), l1_out_block_n_bytes);
        maps.tensor_map_l2_acts =
            make_tma_2d_desc(l2_acts, to_inner(intermediate_hidden), config.num_ring_tokens, block_k_inner,
                             config.load_block_m, static_cast<int>(l2_acts.stride(-2)), config.swizzle_acts_mode);
        maps.tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf, config.num_sf_ring_tokens,
                                                      intermediate_hidden, config.sf_block_m, kGranK, 1, 0, 0, false,
                                                      sf_smem_outer_dim, static_cast<int>(l2_acts_sf.stride(-1)));
        maps.tensor_map_l2_weights =
            not is_packed_fp4
                ? make_tma_weights_3d_desc(l2_weights, config.block_k, config.load_block_n, config.swizzle_weights_mode)
            : use_trtllm_weights ? make_trtllm_fp4_l2_tma_desc(l2_weights, weight_inner(intermediate_hidden),
                                                               num_experts_per_rank * hidden)
                                 : make_tma_2d_desc(weight_tensor(l2_weights), weight_inner(intermediate_hidden),
                                                    num_experts_per_rank * hidden, block_k_inner, config.load_block_n,
                                                    static_cast<int>(l2_weights.stride(-2)),
                                                    config.swizzle_weights_mode, 0, false, not is_packed_fp4);
        maps.tensor_map_l2_weights_sf =
            use_trtllm_weights ? make_trtllm_fp4_sf_tma_desc(l2_weights_sf, false, num_experts_per_rank * hidden / 128,
                                                             intermediate_hidden / (kGranK * 4), sf_smem_outer_dim,
                                                             config.block_n / 128)
                               : make_tma_sf_desc(cute::UMMA::Major::MN, l2_weights_sf, hidden, intermediate_hidden,
                                                  config.block_n, kGranK, num_experts_per_rank, 0, 0, false,
                                                  sf_smem_outer_dim, static_cast<int>(l2_weights_sf.stride(-1)));

        maps.tensor_map_shared_l1_acts =
            num_shared_experts > 0
                ? make_tma_2d_desc(shared_l1_acts, to_inner(hidden), num_max_tokens_per_rank, block_k_inner,
                                   config.load_block_m, static_cast<int>(shared_l1_acts.stride(-2)),
                                   config.swizzle_acts_mode)
                : maps.tensor_map_l1_acts;
        maps.tensor_map_shared_l1_acts_sf =
            num_shared_experts > 0
                ? make_tma_sf_desc(cute::UMMA::Major::MN, shared_l1_acts_sf,
                                   static_cast<int>(shared_l1_acts_sf.size(0)), hidden, config.sf_block_m, kGranK, 1, 0,
                                   0, false, sf_smem_outer_dim, static_cast<int>(shared_l1_acts_sf.stride(-1)))
                : maps.tensor_map_l1_acts_sf;
        maps.tensor_map_shared_l1_weights =
            num_shared_experts > 0 ? make_tma_weights_2d_desc(shared_l1_weights, config.block_k, config.load_block_n,
                                                              config.swizzle_weights_mode)
                                   : maps.tensor_map_l1_weights;
        maps.tensor_map_shared_l1_weights_sf =
            num_shared_experts > 0
                ? make_tma_sf_desc(cute::UMMA::Major::MN, shared_l1_weights_sf, shared_intermediate_hidden * 2, hidden,
                                   config.block_n, kGranK, 1, 0, 0, false, sf_smem_outer_dim,
                                   static_cast<int>(shared_l1_weights_sf.stride(-1)))
                : maps.tensor_map_l1_weights_sf;
        maps.tensor_map_shared_l1_output =
            num_shared_experts > 0
                ? make_tma_2d_desc(shared_l2_acts, to_inner(shared_intermediate_hidden), num_max_tokens_per_rank,
                                   l1_out_block_n_bytes, config.store_block_m_l1,
                                   static_cast<int>(shared_l2_acts.stride(-2)), l1_out_block_n_bytes)
                : maps.tensor_map_l1_output;
        maps.tensor_map_shared_l2_acts =
            num_shared_experts > 0
                ? make_tma_2d_desc(shared_l2_acts, to_inner(shared_intermediate_hidden), num_max_tokens_per_rank,
                                   block_k_inner, config.load_block_m, static_cast<int>(shared_l2_acts.stride(-2)),
                                   config.swizzle_acts_mode)
                : maps.tensor_map_l2_acts;
        maps.tensor_map_shared_l2_acts_sf =
            num_shared_experts > 0
                ? make_tma_sf_desc(cute::UMMA::Major::MN, shared_l2_acts_sf,
                                   static_cast<int>(shared_l2_acts_sf.size(0)), shared_intermediate_hidden,
                                   config.sf_block_m, kGranK, 1, 0, 0, false, sf_smem_outer_dim,
                                   static_cast<int>(shared_l2_acts_sf.stride(-1)))
                : maps.tensor_map_l2_acts_sf;
        maps.tensor_map_shared_l2_weights =
            num_shared_experts > 0 ? make_tma_weights_2d_desc(shared_l2_weights, config.block_k, config.load_block_n,
                                                              config.swizzle_weights_mode)
                                   : maps.tensor_map_l2_weights;
        maps.tensor_map_shared_l2_weights_sf =
            num_shared_experts > 0
                ? make_tma_sf_desc(cute::UMMA::Major::MN, shared_l2_weights_sf, hidden, shared_intermediate_hidden,
                                   config.block_n, kGranK, 1, 0, 0, false, sf_smem_outer_dim,
                                   static_cast<int>(shared_l2_weights_sf.stride(-1)))
                : maps.tensor_map_l2_weights_sf;

        return maps;
    }
};

template <typename Launch, typename Peers>
decltype(auto) with_bf16_mega_moe_args(Launch &&launch, const MegaMoETensorMaps &maps, const Peers &peers, void *y,
                                       int *stats, int num_tokens, uint8_t *locality,
                                       const float *post_expert_norm_weight) {
    return std::forward<Launch>(launch)(y, post_expert_norm_weight, stats, num_tokens, peers, maps.tensor_map_l1_acts,
                                        maps.tensor_map_l1_weights, maps.tensor_map_l1_output, maps.tensor_map_l2_acts,
                                        maps.tensor_map_l2_weights, maps.tensor_map_shared_l1_acts,
                                        maps.tensor_map_shared_l1_weights, maps.tensor_map_shared_l1_output,
                                        maps.tensor_map_shared_l2_acts, maps.tensor_map_shared_l2_weights, locality);
}

template <typename Launch, typename Peers>
decltype(auto) with_fp4_mega_moe_args(Launch &&launch, const MegaMoETensorMaps &maps, const Peers &peers, void *y,
                                      int *stats, int num_tokens, uint8_t *locality,
                                      const float *post_expert_norm_weight, const float *l1_alphas,
                                      const float *l2_alphas, const float *l2_act_scales) {
    return std::forward<Launch>(launch)(
        y, stats, l1_alphas, l2_alphas, l2_act_scales, post_expert_norm_weight, num_tokens, peers,
        maps.tensor_map_l1_acts, maps.tensor_map_l1_acts_sf, maps.tensor_map_l1_weights, maps.tensor_map_l1_weights_sf,
        maps.tensor_map_l1_output, maps.tensor_map_l2_acts, maps.tensor_map_l2_acts_sf, maps.tensor_map_l2_weights,
        maps.tensor_map_l2_weights_sf, maps.tensor_map_shared_l1_acts, maps.tensor_map_shared_l1_acts_sf,
        maps.tensor_map_shared_l1_weights, maps.tensor_map_shared_l1_weights_sf, maps.tensor_map_shared_l1_output,
        maps.tensor_map_shared_l2_acts, maps.tensor_map_shared_l2_acts_sf, maps.tensor_map_shared_l2_weights,
        maps.tensor_map_shared_l2_weights_sf, locality);
}

struct MegaMoEPreDispatchArgs {
    // Runtime args (passed to the kernel via the params struct).
    const void *x;
    const void *topk_idx;
    const void *topk_weights;
    const void *expert_scales;
    void *buf_x;
    void *buf_x_sf;
    void *buf_x_scales;
    void *buf_topk_idx;
    void *buf_topk_weights;
    uint32_t num_tokens;
    uint32_t padded_max;
    uint32_t hidden;
    uint32_t num_groups;
    uint32_t top_k;
};

template <typename Launch>
decltype(auto) with_mega_moe_pre_dispatch_args(Launch &&launch, const MegaMoEPreDispatchArgs &args) {
    return std::forward<Launch>(launch)(args.x, args.topk_idx, args.topk_weights, args.expert_scales, args.buf_x,
                                        args.buf_x_sf, args.buf_x_scales, args.buf_topk_idx, args.buf_topk_weights,
                                        args.num_tokens, args.padded_max, args.hidden, args.num_groups, args.top_k);
}

} // namespace deep_gemm::host
