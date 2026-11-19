"""Exercise the shared launch interface without importing Torch or DeepJIT."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class TestMegaMoEHost(unittest.TestCase):
    def test_descriptors_and_launch_arguments(self):
        root = Path(__file__).resolve().parents[1]
        cuda = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))
        cutlass = Path(os.environ.get('CUTLASS_PATH', root / 'third-party/cutlass'))
        compiler = shutil.which(os.environ.get('CXX', 'c++'))
        if compiler is None or not (cuda / 'include/cuda.h').exists():
            self.skipTest('requires a C++ compiler and CUDA headers')
        if not (cutlass / 'include/cute/tensor.hpp').exists():
            self.skipTest('requires the CUTLASS submodule or CUTLASS_PATH')
        with tempfile.TemporaryDirectory(prefix='mega-moe-host-') as directory:
            binary = Path(directory) / 'test'
            subprocess.run([
                compiler, '-std=c++20', '-Wno-deprecated-declarations',
                '-I' + str(root / 'deep_gemm/include'),
                '-I' + str(cutlass / 'include'), '-I' + str(cuda / 'include'),
                '-I' + str(cuda / 'include/cccl'),
                str(Path(__file__).with_suffix('.cpp')), '-o', str(binary),
            ], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
