"""Correctness test for the W4A16 (BF16 activations x MXFP4 weights) Mega MoE kernel.

Synthetic mode quantizes random weights to MXFP4. Kimi mode loads one real Kimi K3.1 MoE layer
(MXFP4 experts, SiTU) and its saved inputs. Both compare against an FP32 reference and against a
W4A16 emulation (BF16 operands, FP32 accumulation, BF16 FC1 handoff).
"""
import argparse
import json
import os
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F

import deep_gemm
from deep_gemm.utils.dist import dist_print, init_dist

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    packed = packed.view(torch.uint8)
    lo, hi = packed & 0xF, packed >> 4
    nib = torch.stack([lo, hi], dim=-1).flatten(-2).long()
    return E2M1.to(packed.device)[nib & 7] * torch.where(nib >= 8, -1.0, 1.0)


def mx_dequant(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return unpack_e2m1(packed) * torch.pow(2.0, scale.view(torch.uint8).float() - 127.0).repeat_interleave(32, -1)


def mx_quant(w: torch.Tensor):
    """Round-to-nearest MXFP4 quantization: e2m1 values with one UE8M0 power-of-two scale per 32."""
    *lead, k = w.shape
    g = w.float().reshape(*lead, k // 32, 32)
    amax = g.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    exp = torch.ceil(torch.log2(amax / 6.0)).clamp(-126, 127)
    q = g / torch.pow(2.0, exp)
    grid = E2M1.to(w.device)
    mag = (q.abs().unsqueeze(-1) - grid).abs().argmin(-1)
    nib = (mag | torch.where(q < 0, 8, 0)).to(torch.uint8).reshape(*lead, k)
    packed = (nib[..., 0::2] | (nib[..., 1::2] << 4)).contiguous()
    scale = (exp + 127).to(torch.uint8).reshape(*lead, k // 32).contiguous()
    return packed, scale


def activation_fn(h: torch.Tensor, inter: int, activation: str) -> torch.Tensor:
    gate, up = h[:, :inter], h[:, inter:]
    if activation == 'situ':
        return 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate) * 25.0 * torch.tanh(up / 25.0)
    return F.silu(gate) * up


def reference(x, idx, w, w13, w2, inter, activation, mode):
    out = torch.zeros(x.shape[0], w2.shape[1], dtype=torch.float32, device=x.device)
    for e in torch.unique(idx).tolist():
        if e < 0:
            continue
        tok, slot = torch.where(idx == e)
        if mode == 'fp32':
            y = activation_fn(x[tok].float() @ w13[e].float().t(), inter, activation) @ w2[e].float().t()
        elif mode == 'w4a16':
            h = F.linear(x[tok].bfloat16(), w13[e].bfloat16()).float()
            y = F.linear(activation_fn(h, inter, activation).bfloat16(), w2[e].bfloat16()).float()
        else:
            # The kernel's rounding points: FP32 FC1 accumulator into SiTU (SwiGLU rounds gate/up to BF16 first),
            # routing weight folded into the BF16 FC1 handoff, BF16 FC2 output per top-k slot, FP32 top-k sum,
            # BF16 result
            h = x[tok].bfloat16().float() @ w13[e].bfloat16().float().t()
            if activation != 'situ':
                h = h.bfloat16().float()
            a = (activation_fn(h, inter, activation) * w[tok, slot, None]).bfloat16()
            out.index_add_(0, tok, F.linear(a, w2[e].bfloat16()).float())
            continue
        out.index_add_(0, tok, y * w[tok, slot, None])
    return out.bfloat16().float() if mode == 'kernel' else out


def load_kimi_layer(checkpoint: str, layer: int, device: str):
    from safetensors import safe_open
    with open(os.path.join(checkpoint, 'model.safetensors.index.json')) as f:
        where = json.load(f)['weight_map']
    prefix = f'language_model.model.layers.{layer}.block_sparse_moe.experts'
    experts = sorted({int(k.split('.')[6]) for k in where if k.startswith(prefix + '.')})
    handles = {}

    def get(name):
        file = where[name]
        if file not in handles:
            handles[file] = safe_open(os.path.join(checkpoint, file), 'pt')
        return handles[file].get_tensor(name)

    def stack(wi, part):
        return torch.stack([get(f'{prefix}.{e}.{wi}.weight_{part}').view(torch.uint8) for e in experts]).to(device)

    w1p, w1s, w3p, w3s, w2p, w2s = (stack(w, q) for w in ('w1', 'w3', 'w2') for q in ('packed', 'scale'))
    # Canonical w13 is [gate; up] = [w1; w3]
    return torch.cat([w1p, w3p], 1), torch.cat([w1s, w3s], 1), w2p, w2s


def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.kimi_checkpoint:
        w13p, w13s, w2p, w2s = load_kimi_layer(args.kimi_checkpoint, args.kimi_layer, 'cuda')
        d = torch.load(args.kimi_inputs)
        x_all, idx_all, w_all = d['x'].cuda().bfloat16(), d['idx'].cuda().long(), d['w'].cuda().float()
        if args.kimi_tokens:
            x_all, idx_all, w_all = x_all[:args.kimi_tokens], idx_all[:args.kimi_tokens], w_all[:args.kimi_tokens]
        num_all = x_all.shape[0]
        per_rank = (num_all + num_ranks - 1) // num_ranks
        sl = slice(rank_idx * per_rank, min(num_all, (rank_idx + 1) * per_rank))
        x, topk_idx, topk_weights = x_all[sl].contiguous(), idx_all[sl].contiguous(), w_all[sl].contiguous()
        num_experts, num_topk = w13p.shape[0], idx_all.shape[1]
        hidden, inter = w13p.shape[2] * 2, w13p.shape[1] // 2
        activation = args.kimi_activation
        num_max_tokens_per_rank = per_rank
    else:
        num_experts, num_topk = args.num_experts, args.num_topk
        hidden, inter = args.hidden, args.intermediate_hidden
        activation = args.activation
        num_max_tokens_per_rank = args.num_max_tokens_per_rank
        # Every rank builds every expert from the same seed, then keeps its own slice
        w13 = torch.randn((num_experts, inter * 2, hidden), device='cuda') * hidden ** -0.5
        w2 = torch.randn((num_experts, hidden, inter), device='cuda') * inter ** -0.5
        (w13p, w13s), (w2p, w2s) = mx_quant(w13), mx_quant(w2)
        torch.manual_seed(args.seed + 1 + rank_idx)
        x = torch.randn((args.num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        scores = torch.randn((args.num_tokens, num_experts), device='cuda')
        topk_weights, topk_idx = torch.topk(scores.softmax(-1), num_topk, dim=-1, sorted=False)
        if args.masked_ratio > 0:
            mask = torch.rand_like(topk_weights) < args.masked_ratio
            topk_idx = topk_idx.masked_fill(mask, -1)
            topk_weights = topk_weights.masked_fill(mask, 0)
    num_tokens = x.shape[0]
    num_experts_per_rank = num_experts // num_ranks
    local = slice(rank_idx * num_experts_per_rank, (rank_idx + 1) * num_experts_per_rank)

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts, num_max_tokens_per_rank, num_topk, hidden, inter, mma_type='bf16xbf16')
    l1, l2 = deep_gemm.transform_fp4_weights_for_bf16_mega_moe(
        (w13p[local], w13s[local]), (w2p[local], w2s[local]))

    def run():
        buffer.x[:num_tokens].copy_(x)
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)
        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        deep_gemm.bf16_fp4_mega_moe(y, l1, l2, buffer, activation=activation, fast_math=bool(args.fast_math))
        return y

    y = run().float()
    torch.cuda.synchronize()
    dist.barrier()

    # References over the dequantized MXFP4 weights (identical values to what the kernel decodes)
    w13_deq, w2_deq = mx_dequant(w13p, w13s), mx_dequant(w2p, w2s)
    ref = reference(x, topk_idx, topk_weights, w13_deq, w2_deq, inter, activation, 'fp32')
    emu = reference(x, topk_idx, topk_weights, w13_deq, w2_deq, inter, activation, 'w4a16')
    kemu = reference(x, topk_idx, topk_weights, w13_deq, w2_deq, inter, activation, 'kernel')
    rel = lambda a, b: float((a - b).norm() / b.norm())
    res = {
        'rank': rank_idx, 'ranks': num_ranks, 'tokens': num_tokens, 'experts': num_experts, 'topk': num_topk,
        'hidden': hidden, 'inter': inter, 'activation': activation,
        'kernel_vs_fp32': rel(y, ref), 'w4a16_emulation_vs_fp32': rel(emu, ref), 'kernel_vs_emulation': rel(y, emu),
        'kernel_rounding_emulation_vs_fp32': rel(kemu, ref), 'kernel_vs_kernel_rounding_emulation': rel(y, kemu),
        'w4a16_emulation_bf16_output_vs_fp32': rel(emu.bfloat16().float(), ref),
        'kernel_max_abs_err': float((y - ref).abs().max()), 'ref_max_abs': float(ref.abs().max()),
        'nan': bool(torch.isnan(y).any()),
    }

    # Repeat to check run-to-run determinism of the reduction and the workspace cleanup
    for _ in range(args.num_repeats):
        y2 = run().float()
        res.setdefault('repeat_max_abs_diff', 0.0)
        res['repeat_max_abs_diff'] = max(res['repeat_max_abs_diff'], float((y2 - y).abs().max()))
    torch.cuda.synchronize()

    def bench(fn):
        for _ in range(3):
            fn()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.bench):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000 / args.bench

    if args.bench:
        res['avg_us_incl_copy'] = bench(run)
        if args.compare_bf16:
            # The BF16 x BF16 kernel on the same (dequantized) weights, SwiGLU only
            assert activation == 'swiglu'
            l1_bf16, l2_bf16 = deep_gemm.transform_weights_for_mega_moe(
                w13_deq[local].bfloat16(), w2_deq[local].bfloat16())

            def run_bf16():
                buffer.x[:num_tokens].copy_(x)
                buffer.topk_idx[:num_tokens].copy_(topk_idx)
                buffer.topk_weights[:num_tokens].copy_(topk_weights)
                y_ = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                deep_gemm.bf16_mega_moe(y_, l1_bf16, l2_bf16, buffer, fast_math=bool(args.fast_math))
                return y_
            res['bf16_kernel_vs_fp32'] = rel(run_bf16().float(), ref)
            res['bf16_avg_us_incl_copy'] = bench(run_bf16)

    dist_print(json.dumps(res))
    if args.out and rank_idx == 0:
        with open(args.out, 'w') as f:
            json.dump(res, f)
    dist.barrier()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-processes', type=int, default=1)
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=256)
    parser.add_argument('--num-tokens', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--intermediate-hidden', type=int, default=256)
    parser.add_argument('--num-experts', type=int, default=16)
    parser.add_argument('--num-topk', type=int, default=4)
    parser.add_argument('--masked-ratio', type=float, default=0.0)
    parser.add_argument('--activation', type=str, default='situ', choices=('swiglu', 'situ'))
    parser.add_argument('--fast-math', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-repeats', type=int, default=2)
    parser.add_argument('--bench', type=int, default=0)
    parser.add_argument('--compare-bf16', action='store_true')
    parser.add_argument('--kimi-activation', type=str, default='situ', choices=('swiglu', 'situ'))
    parser.add_argument('--kimi-tokens', type=int, default=0, help='Use only the first N saved tokens (0: all)')
    parser.add_argument('--kimi-checkpoint', type=str, default='')
    parser.add_argument('--kimi-layer', type=int, default=3)
    parser.add_argument('--kimi-inputs', type=str, default='')
    parser.add_argument('--out', type=str, default='')
    args = parser.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
