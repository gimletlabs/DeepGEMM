"""Reject unsupported FP4-activation layouts before constructing TMA descriptors."""

import unittest

import torch
import deep_gemm


class TestMegaMoEValidation(unittest.TestCase):
    def call_mega_moe(self, mma_type, weight_shape, shared=False, activation='swiglu', clamp=None):
        weights = torch.empty(weight_shape, dtype=torch.int8)
        scales = torch.empty(0, dtype=torch.int32)
        pair = (weights, scales)
        # These unsupported configurations must fail before accessing CUDA or the buffer.
        deep_gemm._C.fp8_fp4_mega_moe(
            torch.empty((1, 256), dtype=torch.bfloat16), pair, pair,
            pair if shared else None, pair if shared else None, None,
            torch.empty(0, dtype=torch.int8), [0], 0, 1, 4, 1,
            (1, 1, 16 if mma_type == 'nvfp4xnvfp4' else 32), mma_type,
            activation, clamp, True, False, None, None, None,
            False, None, 1e-6, False, False,
        )

    @unittest.skipUnless(torch.cuda.is_available(), 'buffer sizing queries the device SM count')
    def test_buffer_size_is_independent_of_activation(self):
        size = deep_gemm._C.get_symm_buffer_size_for_mega_moe
        for mma_type, shared in [('nvfp4xnvfp4', 0), ('fp8xfp4', 1)]:
            with self.subTest(mma_type=mma_type, shared=shared):
                args = (8, 896, 1024, 16, 3584, 3072, mma_type)
                self.assertEqual(size(*args, 'situ', shared)[0], size(*args, 'swiglu', shared)[0])

    def test_situ_rejects_finite_clamp(self):
        for clamp in (0.0, 7.0):
            with self.subTest(clamp=clamp), self.assertRaisesRegex(RuntimeError, 'not use_situ or activation_clamp'):
                self.call_mega_moe('fp8xfp4', (4, 256, 128), activation='situ', clamp=clamp)

    def test_unknown_activation(self):
        with self.assertRaisesRegex(RuntimeError, 'activation == "swiglu" or use_situ'):
            self.call_mega_moe('fp8xfp4', (4, 256, 128), activation='unknown')

    def test_shared_experts_with_fp4_activations(self):
        for mma_type in ('nvfp4xnvfp4', 'mxf4xmxf4'):
            with self.subTest(mma_type=mma_type):
                with self.assertRaisesRegex(RuntimeError, 'does not support shared experts'):
                    self.call_mega_moe(mma_type, (4, 256, 128), shared=True)

    def test_localized_weights_with_fp4_activations(self):
        for mma_type in ('nvfp4xnvfp4', 'mxf4xmxf4'):
            for activation in ('swiglu', 'situ'):
                with self.subTest(mma_type=mma_type, activation=activation):
                    with self.assertRaisesRegex(RuntimeError, 'does not support localized weights'):
                        self.call_mega_moe(mma_type, (2, 4, 128, 128), activation=activation)

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
            for activation, clamp in [('swiglu', 7.0), ('situ', None), ('situ', float('inf'))]:
                with self.subTest(mma_type=mma_type, activation=activation, clamp=clamp):
                    # A deliberately empty buffer stops before launch, after all weight checks.
                    with self.assertRaisesRegex(RuntimeError, 'symmetric buffer is 0 bytes'):
                        deep_gemm._C.fp8_fp4_mega_moe(
                            torch.empty((1, 256), device='cuda', dtype=torch.bfloat16),
                            weights((4, 512, 256), dtype), weights((4, 256, 256), dtype),
                            shared_l1, shared_l2, None,
                            torch.empty(0, device='cuda', dtype=torch.int8), [0], 0, 1, 4, 1,
                            (1, 1, 32), mma_type, activation, clamp, True, False, None, None, None,
                            False, None, 1e-6, False, False,
                        )


if __name__ == '__main__':
    unittest.main()
