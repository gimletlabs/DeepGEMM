"""Run with torchrun --standalone --nproc-per-node=8 tests/test_kimi_mega_moe.py on B200."""

import os

import torch
import torch.distributed as dist

import deep_gemm


def projection(experts, rows, cols, *, gate_up=False, expert_offset=0):
    # One nonzero coefficient per row; down projections alternate sign by expert.
    device = torch.device('cuda', torch.cuda.current_device())
    weights = torch.zeros((experts, rows, cols // 2), dtype=torch.uint8, device=device)
    row = torch.arange(rows, device=device)
    if gate_up:
        col = row % (rows // 2)
        code = torch.where(row < rows // 2, 2, 4).to(torch.uint8)  # gate=x, up=2*x
    else:
        col = row % cols
        expert = torch.arange(expert_offset, expert_offset + experts, device=device)
        code = torch.where(expert % 3 == 0, 10, 2).to(torch.uint8)[:, None]  # -1 or +1
    weights[:, row, col // 2] = code << (4 * (col % 2)).to(torch.uint8)
    # UE8M0 exponent 127 represents a unit scale.
    scales = torch.full((experts, cols // 128, rows), 0x7f7f7f7f,
                        dtype=torch.int32, device=device).transpose(1, 2)
    return weights.view(torch.int8), scales


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('cpu:gloo,cuda:nccl')
    try:
        assert dist.get_world_size() == 8
        assert torch.cuda.get_device_name() == 'NVIDIA B200'
        hidden, intermediate, experts, topk, tokens = 3584, 3072, 896, 16, 1024
        local = experts // dist.get_world_size()
        w1, w2 = deep_gemm.transform_weights_for_mega_moe(
            projection(local, intermediate * 2, hidden, gate_up=True),
            projection(local, hidden, intermediate, expert_offset=rank * local),
            activation='situ')
        buf = deep_gemm.get_symm_buffer_for_mega_moe(
            dist.group.WORLD, experts, tokens, topk, hidden, intermediate, activation='situ')
        t = torch.arange(tokens, device='cuda')
        x = torch.tensor([-8, -1, 1, 4, 16], device='cuda', dtype=torch.bfloat16)[(t + rank) % 5]
        activations = x[:, None].expand(tokens, hidden).contiguous()
        ids = ((t[:, None] + rank * 17 + torch.arange(topk, device='cuda') * 53) %
               experts).int()
        scores = torch.full((tokens, topk), 1 / topk, device='cuda')
        output = torch.empty_like(activations)

        def launch():
            deep_gemm._C.mega_moe_pre_dispatch(activations, ids, scores, buf.x, buf.x_sf,
                                               buf.topk_idx, buf.topk_weights, tokens)
            deep_gemm.fp8_fp4_mega_moe(output, w1, w2, buf, activation='situ', fast_math=False)

        launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        graph.replay()
        # Round the weighted activation to MXFP8 before FC2. Every 32-channel
        # group in this fixture has the same absolute maximum.
        x = x.float()
        value = 4 * torch.tanh(x / 4) * torch.sigmoid(x) * 25 * torch.tanh(2 * x / 25) / topk
        scale = torch.exp2(torch.ceil(torch.log2(value.abs() / 448)))
        value = (value / scale).to(torch.float8_e4m3fn).float() * scale
        sign_sum = torch.where(ids % 3 == 0, -1, 1).sum(dim=1)
        expected = (value * sign_sum).bfloat16()[:, None].expand_as(output)
        torch.testing.assert_close(output, expected, rtol=0.01, atol=1e-4)
        dist.barrier()
        del graph
        buf.destroy()
        if rank == 0:
            print('Kimi EP8 SiTU CUDA graph replay passed', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
