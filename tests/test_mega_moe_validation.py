"""Reject unsupported FP4-activation layouts before constructing TMA descriptors."""

import unittest

import torch
import deep_gemm


class TestMegaMoEValidation(unittest.TestCase):
    def call_mega_moe(self, mma_type, weight_shape, shared=False):
        weights = torch.empty(weight_shape, dtype=torch.int8)
        scales = torch.empty(0, dtype=torch.int32)
        pair = (weights, scales)
        # These unsupported configurations must fail before accessing CUDA or the buffer.
        deep_gemm._C.fp8_fp4_mega_moe(
            torch.empty((1, 256), dtype=torch.bfloat16), pair, pair,
            pair if shared else None, pair if shared else None, None,
            torch.empty(0, dtype=torch.int8), [0], 0, 1, 4, 1,
            (1, 1, 16 if mma_type == 'nvfp4xnvfp4' else 32), mma_type,
            'swiglu', None, True, False, None, None, None,
            False, None, 1e-6, False, False,
        )

    def test_shared_experts_with_fp4_activations(self):
        for mma_type in ('nvfp4xnvfp4', 'mxf4xmxf4'):
            with self.subTest(mma_type=mma_type):
                with self.assertRaisesRegex(RuntimeError, 'does not support shared experts'):
                    self.call_mega_moe(mma_type, (4, 256, 128), shared=True)

    def test_localized_weights_with_fp4_activations(self):
        for mma_type in ('nvfp4xnvfp4', 'mxf4xmxf4'):
            with self.subTest(mma_type=mma_type):
                with self.assertRaisesRegex(RuntimeError, 'does not support localized weights'):
                    self.call_mega_moe(mma_type, (2, 4, 128, 128))

    @unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA for weight layout validation')
    def test_fp8_shared_experts_accept_fp4_and_fp8_routed_weights(self):
        def weights(shape, dtype):
            packed_shape = (*shape[:-1], shape[-1] // 2) if dtype == torch.int8 else shape
            w = torch.empty(packed_shape, device='cuda', dtype=dtype)
            sf = torch.empty((*shape[:-2], shape[-1] // 128, shape[-2]),
                             device='cuda', dtype=torch.int32).transpose(-1, -2)
            return w, sf

        shared_l1 = weights((512, 256), torch.float8_e4m3fn)
        shared_l2 = weights((256, 256), torch.float8_e4m3fn)
        for mma_type, dtype in (('fp8xfp4', torch.int8),
                                ('fp8xfp8', torch.float8_e4m3fn)):
            with self.subTest(mma_type=mma_type):
                # A deliberately empty buffer stops before launch, after all weight checks.
                with self.assertRaisesRegex(RuntimeError, 'symmetric buffer is 0 bytes'):
                    deep_gemm._C.fp8_fp4_mega_moe(
                        torch.empty((1, 256), device='cuda', dtype=torch.bfloat16),
                        weights((4, 512, 256), dtype), weights((4, 256, 256), dtype),
                        shared_l1, shared_l2, None,
                        torch.empty(0, device='cuda', dtype=torch.int8), [0], 0, 1, 4, 1,
                        (1, 1, 32), mma_type, 'swiglu', None, True, False, None, None, None,
                        False, None, 1e-6, False, False,
                    )


if __name__ == '__main__':
    unittest.main()
