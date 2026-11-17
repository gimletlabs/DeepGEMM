#pragma once

#include "runtime_utils.hpp"

#include <deep_gemm/host/mega_moe.cuh>

namespace deep_gemm {

static host::TensorView mega_moe_tensor_view(const torch::Tensor& t) {
    host::TensorView view;
    view.data = t.data_ptr();
    view.dtype = aten_dtype_to_tensor_map_dtype(t.scalar_type(), false, true);
    view.bytes = static_cast<int>(t.element_size());
    view.rank = t.dim();
    DG_HOST_ASSERT(view.rank <= 4);
    view.packed_fp4 = t.scalar_type() == kPackedFP4;
    view.contiguous = t.is_contiguous();
    for (int i = 0; i < view.rank; ++i) {
        view.sizes[i] = t.size(i);
        view.strides[i] = t.stride(i);
    }
    return view;
}
static host::TensorMapBuilder mega_moe_tensor_map_builder() {
    return host::TensorMapBuilder(+[](CUtensorMap* map, CUtensorMapDataType dtype, unsigned rank,
        void* ptr, const cuuint64_t* dims, const cuuint64_t* strides,
        const cuuint32_t* box, const cuuint32_t* elements, CUtensorMapInterleave interleave,
        CUtensorMapSwizzle swizzle, CUtensorMapL2promotion promotion, CUtensorMapFloatOOBfill fill) {
            return deep_jit::cuda::driver::lazy_cuTensorMapEncodeTiled(
                map, dtype, rank, ptr, dims, strides, box, elements,
                interleave, swizzle, promotion, fill);
        },
                                  LocalityDomainAllocator::get_num_locality_domains());
}

} // namespace deep_gemm
