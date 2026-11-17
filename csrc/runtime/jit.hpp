#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <cutlass/version.h>
#include <deep_jit/backend/cuda/backend.hpp>
#include <deep_jit/utils/env.hpp>

namespace deep_gemm {

inline deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>> jit(nullptr);

inline void init_jit(const std::string& library_root_path,
                     const std::vector<std::filesystem::path>& dependency_includes = {}) {
    const auto library_root = std::filesystem::absolute(library_root_path);
    const auto include_dir = library_root / "include";
    auto include_dirs = dependency_includes;
    include_dirs.push_back(include_dir);
    const auto config = deep_jit::Config(
        library_root,
        "DG",
        "cutlass-" + std::to_string(CUTLASS_VERSION),
        include_dirs,
        {"deep_gemm/"});

    jit = deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>>([config] {
        auto runtime = std::make_shared<deep_jit::Runtime<deep_jit::CUDA>>(config);
        runtime->default_compiler_options.nvcc_flags->
            emplace_back("--diag-suppress=39,161,174,177,186,940");
        runtime->default_compiler_options.nvcc_flags->
            emplace_back("--compiler-options=-Wno-deprecated-declarations,-Wno-abi");
        // Compiler flags enter the JIT cache key, so each timeout gets its own kernels.
        if (const auto timeout = deep_jit::get_env<int>("DG_JIT_BARRIER_TIMEOUT_SECONDS", 0); timeout > 0)
            runtime->default_compiler_options.nvcc_flags->
                emplace_back("-DDG_BARRIER_TIMEOUT_SECONDS=" + std::to_string(timeout));
        return runtime;
    });
}

}  // namespace deep_gemm
