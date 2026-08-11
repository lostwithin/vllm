# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton W8A16 GEMV kernel for pre-Ampere CUDA GPUs (Turing/sm75, e.g. T4).

Decode-time W8A16 matmul (M <= 8) is a bandwidth-bound reduction: int8
tensor cores (IMMA/DP4A) cannot be used because activations are fp16, so
the only win is reading half the weight bytes from HBM. Marlin's
Ampere-oriented cp.async pipeline is counterproductive on sm75 (measured
~6% slower than fp16 on T4), while this straightforward
dequantize-and-FMA kernel reaches the bandwidth limit (measured 2.14x /
1.91x / 1.13x vs cuBLAS fp16 at M=1/2/4 on Qwen3.5-9B shapes).

M > 8 (prefill / large batch) falls back to one-shot dequantization +
cuBLAS fp16 GEMM.

Weight layout (post-process_weights_after_loading):
  weight_packed: [N, K]      int8  (unpacked from compressed-tensors int32)
  weight_scale:  [N, K // G] fp16

Opt-in via VLLM_W8A16_TRITON_GEMV=1; intended for sm75 where Marlin is
slow, disabled by default so kernel selection elsewhere is unchanged.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

logger = init_logger(__name__)

TRITON_W8A16_GEMV_MAX_M = 8
TRITON_W8A16_GEMV_GROUP_SIZES = [128]
TRITON_W8A16_GEMV_QUANT_TYPES = [scalar_types.uint8b128]


@triton.jit
def _w8a16_gemv_kernel(
    x_ptr,  # [BLOCK_M, K] fp16 activations (pre-padded, no masking)
    w_ptr,  # [N, K] int8 weights
    s_ptr,  # [N, K // GROUP_SIZE] fp16 scales
    y_ptr,  # [BLOCK_M, N] fp16 output (pre-padded)
    K,
    stride_xm,
    stride_wn,
    stride_sn,
    stride_ym,
    GROUP_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # One program per output feature; the int8 weight row is read once and
    # shared across all BLOCK_M activation rows. X/Y are padded to BLOCK_M
    # rows by the caller so all loads/stores are mask-free (masked loads
    # block vectorization on sm75).
    n = tl.program_id(0)
    m_offs = tl.arange(0, BLOCK_M)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    w_row = w_ptr + n * stride_wn
    for kb in range(0, K, BLOCK_K):
        k_off = kb + tl.arange(0, BLOCK_K)
        k_off = tl.max_contiguous(tl.multiple_of(k_off, BLOCK_K), BLOCK_K)
        w = tl.load(w_row + k_off).to(tl.float32)
        g_idx = kb // GROUP_SIZE + tl.arange(0, BLOCK_K) // GROUP_SIZE
        s = tl.load(s_ptr + n * stride_sn + g_idx).to(tl.float32)
        ws = w * s
        x = tl.load(x_ptr + m_offs[:, None] * stride_xm + k_off[None, :]).to(
            tl.float32
        )
        acc += x * ws[None, :]

    y = tl.sum(acc, axis=1)
    tl.store(y_ptr + m_offs * stride_ym + n, y.to(y_ptr.dtype.element_ty))


def _pick_block_k(k: int) -> int:
    for block_k in (1024, 512, 256, 128):
        if k % block_k == 0:
            return block_k
    return 0


def triton_w8a16_gemv(
    x: torch.Tensor,  # [M, K] fp16, M <= TRITON_W8A16_GEMV_MAX_M
    w_q: torch.Tensor,  # [N, K] int8
    w_s: torch.Tensor,  # [N, K // group_size] fp16
    group_size: int,
) -> torch.Tensor:
    M, K = x.shape
    N = w_q.shape[0]

    block_m = triton.next_power_of_2(M)
    if block_m != M or not x.is_contiguous():
        x_pad = torch.zeros((block_m, K), device=x.device, dtype=x.dtype)
        x_pad[:M] = x
    else:
        x_pad = x
    y_pad = torch.empty((block_m, N), device=x.device, dtype=x.dtype)

    _w8a16_gemv_kernel[(N,)](
        x_pad,
        w_q,
        w_s,
        y_pad,
        K,
        x_pad.stride(0),
        w_q.stride(0),
        w_s.stride(0),
        y_pad.stride(0),
        GROUP_SIZE=group_size,
        BLOCK_K=_pick_block_k(K),
        BLOCK_M=block_m,
        num_warps=4,
    )
    return y_pad[:M]


def _w8a16_gemv_apply(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    if x.shape[0] <= TRITON_W8A16_GEMV_MAX_M:
        return triton_w8a16_gemv(x, w_q, w_s, group_size)
    # Prefill / large batch: dequantize once, use cuBLAS fp16 GEMM
    # (Triton tl.dot degrades to scalar FMA on sm75, so a fused GEMM
    # kernel cannot win there).
    n, k = w_q.shape
    groups = k // group_size
    w_deq = (w_q.view(n, groups, group_size).to(x.dtype) * w_s.view(n, groups, 1)).view(
        n, k
    )
    return x @ w_deq.t()


def _w8a16_gemv_apply_fake(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], w_q.shape[0]))


# Register as an opaque custom op: inductor must not trace into the
# Triton launch / dequant fallback (combo-kernel benchmarking of the
# decomposed dequant hangs workers for minutes on sm75).
direct_register_custom_op(
    op_name="w8a16_triton_gemv",
    op_func=_w8a16_gemv_apply,
    fake_impl=_w8a16_gemv_apply_fake,
)


class TritonW8A16GemvLinearKernel(MPLinearKernel):
    """Bandwidth-oriented W8A16 kernel for sm75 decode (opt-in via env)."""

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if os.environ.get("VLLM_W8A16_TRITON_GEMV", "0") != "1":
            return False, "VLLM_W8A16_TRITON_GEMV is not set"

        if not current_platform.is_cuda():
            return False, "TritonW8A16GemvLinearKernel requires CUDA"

        if c.weight_type not in TRITON_W8A16_GEMV_QUANT_TYPES:
            return (
                False,
                f"Quant type {c.weight_type} not supported; "
                f"supported: {TRITON_W8A16_GEMV_QUANT_TYPES}",
            )

        if c.act_type != torch.float16:
            return False, "Only float16 activations are supported (sm75 target)"

        if c.zero_points:
            return False, "Zero points are not supported"

        if c.has_g_idx:
            return False, "Activation reordering (g_idx) is not supported"

        if c.group_size not in TRITON_W8A16_GEMV_GROUP_SIZES:
            return (
                False,
                f"Group size {c.group_size} not supported; "
                f"supported: {TRITON_W8A16_GEMV_GROUP_SIZES}",
            )

        K = c.partition_weight_shape[0]
        if K % c.group_size != 0:
            return False, f"Input features {K} not divisible by group size"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Unpack compressed-tensors int32 weights to [N, K] int8.

        Checkpoint (from compressed_tensors_wNa16 create_weights):
          weight_packed: [N, K//4] int32, 4 sequential K-values per int32
                         (byte i of int32 j = K index 4*j+i), stored as
                         unsigned (signed value + weight_type.bias).
          weight_scale:  [N, K//G] fp16
        """
        bias = self.config.weight_type.bias  # 128 for uint8b128

        def unpack_w_q(x: BasevLLMParameter) -> BasevLLMParameter:
            permute_param_layout_(x, input_dim=1, output_dim=0, packed_dim=1)
            packed = x.data  # [N, K//4] int32
            n, k4 = packed.shape
            k = k4 * 4
            shifts = torch.arange(0, 32, 8, device=packed.device, dtype=torch.int32)
            w_q = torch.empty((n, k), device=packed.device, dtype=torch.int8)
            # Chunk rows: the int32 unpack intermediate is 4x the int8 result
            for r0 in range(0, n, 8192):
                chunk = packed[r0 : r0 + 8192]
                u = (chunk.unsqueeze(-1) >> shifts) & 0xFF
                w_q[r0 : r0 + 8192] = (u - bias).to(torch.int8).view(-1, k)
            x.data = w_q
            return x

        def make_contiguous(x: BasevLLMParameter) -> BasevLLMParameter:
            permute_param_layout_(x, input_dim=1, output_dim=0)
            x.data = x.data.contiguous()
            return x

        self._transform_param(layer, self.w_q_name, unpack_w_q)
        self._transform_param(layer, self.w_s_name, make_contiguous)
        logger.info_once("TritonW8A16GemvLinearKernel active (sm75 W8A16 GEMV)")

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, _, _ = self._get_weight_params(layer)

        x_2d = x.reshape(-1, x.shape[-1])
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        output = torch.ops.vllm.w8a16_triton_gemv(x_2d, w_q, w_s, c.group_size)

        if bias is not None:
            output.add_(bias)

        return output.reshape(out_shape)
