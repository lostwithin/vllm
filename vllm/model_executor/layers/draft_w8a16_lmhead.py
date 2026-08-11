# SPDX-License-Identifier: Apache-2.0
"""(实验)投机解码 drafter 的 lm_head W8A16 原地量化 — T4/Turing 专用优化。

背景:batch=1 解码时 drafter 的 lm_head 是 decode 周期里最大的单块权重流量
(vocab 分片 124160x4096 fp16 = ~1GB/卡,每周期按 M=1 读 K 次;而 target 验证
只按 M=K+1 读 1 次)。sm75 上 Marlin 走非优化路径,实测比 fp16 慢;本模块用一个
带宽导向的分组反量化 GEMV Triton kernel(实测 2.16x)原地替换 drafter lm_head
的 quant_method,并释放 fp16 权重(净省 ~0.5GB/卡)。

正确性依据:量化只作用于 draft 路径;rejection sampling 保证最终输出仍严格等于
target 模型的采样结果,draft 精度只影响接受率(速度),不影响输出。

启用:环境变量 VLLM_DRAFT_LMHEAD_W8A16=1(默认关闭,不影响任何现有路径)。
只在模型的 load_weights 末尾由 MTP 模型显式调用 maybe_quantize_draft_lmhead()。
"""

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

GROUP_SIZE = 128
ENV_FLAG = "VLLM_DRAFT_LMHEAD_W8A16"


@triton.jit
def _w8a16_gemv_kernel(
    X, W, S, Y,
    K, stride_wn, stride_sn,
    GROUP_SIZE: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M=1 快路径:向量累加器、循环外一次归约(归约放循环内会延迟受限,实测慢 2-4x)。
    n = tl.program_id(0)
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for kb in range(0, K, BLOCK_K):
        k_off = kb + tl.arange(0, BLOCK_K)
        x = tl.load(X + k_off).to(tl.float32)
        w = tl.load(W + n * stride_wn + k_off).to(tl.float32)
        g_idx = kb // GROUP_SIZE + tl.arange(0, BLOCK_K) // GROUP_SIZE
        s = tl.load(S + n * stride_sn + g_idx).to(tl.float32)
        acc += x * w * s
    tl.store(Y + n, tl.sum(acc, axis=0).to(Y.dtype.element_ty))


@triton.jit
def _w8a16_gemm_kernel(
    X, W, S, Y, M, K,
    stride_xm, stride_wn, stride_sn, stride_ym,
    GROUP_SIZE: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
):
    # 2<=M<=8 保险丝路径(drafter 的 logits 只对采样位置算,batch=1 下永远 M=1,
    # 这条只为 batch>1 兜底;正确性够、速度非目标)。
    n = tl.program_id(0)
    m_offs = tl.arange(0, BLOCK_M)
    m_mask = m_offs < M
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    for kb in range(0, K, BLOCK_K):
        k_off = kb + tl.arange(0, BLOCK_K)
        w = tl.load(W + n * stride_wn + k_off).to(tl.float32)
        g_idx = kb // GROUP_SIZE + tl.arange(0, BLOCK_K) // GROUP_SIZE
        s = tl.load(S + n * stride_sn + g_idx).to(tl.float32)
        ws = w * s
        x = tl.load(X + m_offs[:, None] * stride_xm + k_off[None, :],
                    mask=m_mask[:, None], other=0.0).to(tl.float32)
        acc += x * ws[None, :]
    y_val = tl.sum(acc, axis=1)
    tl.store(Y + m_offs * stride_ym + n, y_val.to(Y.dtype.element_ty), mask=m_mask)


def w8a16_gemm(x: torch.Tensor, w_int8: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    M, K = x2d.shape
    N = w_int8.shape[0]
    assert K % 512 == 0, f"K={K} 需被 BLOCK_K 整除(kernel 无 K 掩码)"
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    for m0 in range(0, M, 8):
        m_len = min(8, M - m0)
        if m_len == 1:
            _w8a16_gemv_kernel[(N,)](
                x2d[m0], w_int8, scales, y[m0],
                K, w_int8.stride(0), scales.stride(0),
                GROUP_SIZE=GROUP_SIZE, BLOCK_K=512, num_warps=4,
            )
        else:
            block_m = triton.next_power_of_2(m_len)
            block_k = max(128, 1024 // block_m)
            _w8a16_gemm_kernel[(N,)](
                x2d[m0:], w_int8, scales, y[m0:], m_len, K,
                x2d.stride(0), w_int8.stride(0), scales.stride(0), y.stride(0),
                GROUP_SIZE=GROUP_SIZE, BLOCK_K=block_k, BLOCK_M=block_m,
                num_warps=4,
            )
    return y.reshape(*orig_shape[:-1], N)


def _quantize_g128(w: torch.Tensor, chunk_rows: int = 8192):
    # 按行分块:整表 w.float() 会造出 ~1.9GB 的 fp32 临时副本,模型加载中途的
    # 显存余量(~1.2GB)装不下(实测 OOM)。分块后峰值临时占用 ~130MB。
    n, k = w.shape
    ng = k // GROUP_SIZE
    w_int8 = torch.empty(n, k, dtype=torch.int8, device=w.device)
    scales = torch.empty(n, ng, dtype=torch.float16, device=w.device)
    for r0 in range(0, n, chunk_rows):
        r1 = min(n, r0 + chunk_rows)
        wg = w[r0:r1].float().view(r1 - r0, ng, GROUP_SIZE)
        amax = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        sc = amax / 127.0
        w_int8[r0:r1] = ((wg / sc).round().clamp(-127, 127)
                         .to(torch.int8).view(r1 - r0, k))
        scales[r0:r1] = sc.view(r1 - r0, ng).to(torch.float16)
    return w_int8, scales


class DraftW8A16LMHeadMethod:
    """替换 drafter lm_head 的 quant_method:apply() 走分组反量化 GEMV。

    LogitsProcessor 的两条路径(_get_logits 与 get_top_tokens)都经由
    lm_head.quant_method.apply(lm_head, hidden_states, bias=...),换这一个点全覆盖。
    """

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        y = w8a16_gemm(x, layer.draft_w8a16_qweight, layer.draft_w8a16_scales)
        if bias is not None:
            y = y + bias
        return y


def maybe_quantize_draft_lmhead(lm_head: torch.nn.Module) -> None:
    """env 门控入口:量化 drafter lm_head 并释放 fp16 权重。默认无操作。"""
    if os.environ.get(ENV_FLAG, "0") != "1":
        return
    weight = getattr(lm_head, "weight", None)
    if weight is None or weight.dtype not in (torch.float16, torch.bfloat16) \
            or not weight.is_cuda or weight.dim() != 2:
        logger.warning("%s=1 但 lm_head 权重不满足量化条件(dtype=%s),跳过",
                       ENV_FLAG, None if weight is None else weight.dtype)
        return
    n, k = weight.shape
    if k % 512 != 0 or k % GROUP_SIZE != 0:
        logger.warning("%s=1 但 K=%d 不满足对齐要求,跳过", ENV_FLAG, k)
        return

    w_int8, scales = _quantize_g128(weight.data.to(torch.float16))
    lm_head.register_buffer("draft_w8a16_qweight", w_int8, persistent=False)
    lm_head.register_buffer("draft_w8a16_scales", scales, persistent=False)
    freed = weight.numel() * weight.element_size()
    # 置空而非删除:保留 Parameter 语义;若有旁路代码仍直接用 .weight 会立刻报错(要响,不要静默)。
    weight.data = weight.data.new_empty((0,))
    lm_head.quant_method = DraftW8A16LMHeadMethod()
    torch.cuda.empty_cache()
    logger.info_once(
        "Draft lm_head W8A16 已启用: shape=(%d,%d) group=%d, 释放 fp16 权重 %.2f GB/卡",
        n, k, GROUP_SIZE, freed / 1e9,
    )
