// Compile and run through test_mega_moe_host.py; no Python headers, GPU, or DeepJIT required.
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <tuple>

#define DG_HOST_ASSERT(condition) do { if (!(condition)) throw std::runtime_error(#condition); } while (0)
#include <deep_gemm/host/mega_moe.cuh>

namespace {
struct Descriptor {
    CUtensorMapDataType dtype;
    unsigned rank;
    uint64_t dims[5]{};
    uint64_t strides[4]{};
    uint32_t box[5]{};
} captured;

CUresult capture(CUtensorMap* map, CUtensorMapDataType dtype, unsigned rank, void*,
                 const cuuint64_t* dims, const cuuint64_t* strides, const cuuint32_t* box,
                 const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                 CUtensorMapL2promotion, CUtensorMapFloatOOBfill) {
    captured = {.dtype = dtype, .rank = rank};
    std::copy_n(dims, rank, captured.dims);
    std::copy_n(strides, rank - 1, captured.strides);
    std::copy_n(box, rank, captured.box);
    std::memset(map, 0, sizeof(*map));
    return CUDA_SUCCESS;
}
}  // namespace

int main() {
    using namespace deep_gemm;
    host::TensorMapBuilder builder(capture);
    host::TensorView weights;
    weights.dtype = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    weights.bytes = 2;
    weights.rank = 3;
    weights.sizes = {4, 512, 256};
    weights.strides = {512 * 256, 256, 1};
    builder.make_tma_weights_3d_desc(weights, 128, 128, 128);
    const auto ordinary = captured;
    assert(ordinary.rank == 4 && ordinary.dims[0] == 256 && ordinary.dims[1] == 256);
    assert(ordinary.dims[2] == 2 && ordinary.dims[3] == 4);
    assert(ordinary.strides[0] == 512 && ordinary.strides[1] == 131072);
    assert(ordinary.strides[2] == 262144);

    // Localized weights group each locality domain's rows across experts.
    weights.rank = 4;
    weights.sizes = {2, 4, 256, 256};
    weights.strides = {4 * 256 * 256, 256 * 256, 256, 1};
    builder.make_tma_weights_3d_desc(weights, 128, 128, 128);
    assert(std::equal(captured.dims, captured.dims + 4, ordinary.dims));
    assert(captured.strides[0] == 512 && captured.strides[1] == 524288);
    assert(captured.strides[2] == 131072);

    // Packed FP4 doubles the logical K dimension without changing the byte stride.
    weights.dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    weights.bytes = 1;
    weights.packed_fp4 = true;
    builder.make_tma_weights_3d_desc(weights, 256, 128, 128);
    assert(captured.dtype == CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B);
    assert(captured.dims[0] == 512 && captured.strides[0] == 256);

    // FP8 scale rows can have padding in their K stride.
    host::TensorView scales;
    scales.dtype = CU_TENSOR_MAP_DATA_TYPE_INT32;
    scales.bytes = 4;
    builder.make_tma_sf_desc(cute::UMMA::Major::MN, scales, 17, 256, 32, 16, 1, 0,
                             0, false, 1, 32);
    assert(captured.dims[0] == 20 && captured.dims[1] == 4);
    assert(captured.strides[0] == 128);

    host::MegaMoETensorMaps maps;
    layout::SymBuffer<4> peers;
    host::with_bf16_mega_moe_args([&](const auto&... values) {
        static_assert(sizeof...(values) == 16);
        auto args = std::tie(values...);
        assert(std::get<3>(args) == 1536);
        assert(&std::get<4>(args) == &peers);
        assert(&std::get<5>(args) == &maps.tensor_map_l1_acts);
        assert(&std::get<14>(args) == &maps.tensor_map_shared_l2_weights);
    }, maps, peers, nullptr, nullptr, 1536, nullptr, nullptr);
    host::with_fp4_mega_moe_args([&](const auto&... values) {
        static_assert(sizeof...(values) == 27);
        auto args = std::tie(values...);
        assert(std::get<6>(args) == 4096);
        assert(&std::get<7>(args) == &peers);
        assert(&std::get<25>(args) == &maps.tensor_map_shared_l2_weights_sf);
    }, maps, peers, nullptr, nullptr, 4096, nullptr, nullptr, nullptr, nullptr, nullptr);
    host::MegaMoEPreDispatchArgs inputs{};
    inputs.num_tokens = 1536;
    inputs.padded_max = 5760;
    host::with_mega_moe_pre_dispatch_args([](const auto&... values) {
        static_assert(sizeof...(values) == 14);
        auto args = std::tie(values...);
        assert(std::get<9>(args) == 1536 && std::get<10>(args) == 5760);
    }, inputs);
}
