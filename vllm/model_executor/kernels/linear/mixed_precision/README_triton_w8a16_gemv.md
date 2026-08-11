# TritonW8A16GemvLinearKernel — sm75 (Turing/T4) W8A16 解码 GEMV

源码:[`triton_w8a16_gemv.py`](./triton_w8a16_gemv.py)

一个**默认关闭、需显式 opt-in** 的混合精度 linear kernel:在 Turing (sm75, 如 T4)
上用带宽导向的 "反量化 + FMA" Triton GEMV 替换 Marlin,加速 W8A16 (int8 权重 /
fp16 激活) 模型的 decode 阶段。

---

## 1. 动机

decode (M ≤ 8) 时的 W8A16 matmul 是**带宽瓶颈的归约**,不是算力瓶颈:

- 激活是 fp16,所以 int8 tensor core (IMMA/DP4A) 用不上 —— 权重必须先反量化成
  fp16 再算。唯一的收益来源是**从 HBM 少读一半权重字节**。
- Marlin 的 Ampere 导向 cp.async 流水在 sm75 上是反效果的(sm75 没有
  `cp.async`,走非优化回退路径),实测在 T4 上比直接跑 fp16 还慢 ~6%。
- Triton 的 `tl.dot` 在 sm75 上会退化成标量 FMA,所以"融合 GEMM"这条路在
  decode 尺寸下也赢不了。

结论:与其模拟 GEMM,不如老老实实写一个把 HBM 带宽吃满的 GEMV。

## 2. 实测性能

以下数据来自源码 docstring 中记录的实测结果(T4,Qwen3.5-9B 的 linear 形状,
group_size=128),基线是 cuBLAS fp16:

| M (batch tokens) | 相对 cuBLAS fp16 加速比 |
| ---------------- | ----------------------- |
| 1                | 2.14x                   |
| 2                | 1.91x                   |
| 4                | 1.13x                   |

M > 8 无加速(见下方 fallback 说明)。作为对照,同硬件上 Marlin 约为 fp16 的
0.94x。

## 3. 启用方式

```bash
VLLM_W8A16_TRITON_GEMV=1 vllm serve <w8a16-compressed-tensors-model> --dtype float16
```

生效后日志里会出现一次:

```
TritonW8A16GemvLinearKernel active (sm75 W8A16 GEMV)
```

注意事项:

- 该环境变量**没有注册到 `vllm/envs.py`**,是在 `can_implement()` 里用
  `os.environ` 直接读的。它在 **layer 构造期**被读取,因此必须在引擎启动前设置
  (多进程 TP 下由子进程继承,无需额外处理)。
- 关闭方式:不设该变量即可。也可以用 vLLM 通用机制强制排除:
  `VLLM_DISABLED_KERNELS=TritonW8A16GemvLinearKernel`。

### ⚠️ 优先级与适用硬件

本 kernel 被插在 `_POSSIBLE_KERNELS[PlatformEnum.CUDA]` 列表的**最前面**
(`vllm/model_executor/kernels/linear/__init__.py`),而 `get_min_capability()`
返回 75。也就是说:**一旦设置了该环境变量,在任何 sm75+ 的 CUDA 卡上(包括
A100/H100)只要 config 匹配就会抢占 Machete / Marlin**。

Ampere 及以上请**不要**开这个开关 —— 那些卡上 Marlin/Machete 明显更快。这个
kernel 只为 sm75 而写。

## 4. 生效条件(`can_implement`)

全部满足才会被选中,否则回落到列表里的下一个 kernel:

| 条件         | 要求                                          |
| ------------ | --------------------------------------------- |
| 环境变量     | `VLLM_W8A16_TRITON_GEMV=1`                    |
| 平台         | CUDA                                          |
| 算力         | ≥ sm75                                        |
| 量化类型     | `scalar_types.uint8b128`(对称 int8,bias=128) |
| 激活 dtype   | `torch.float16`(bf16 不支持,sm75 无原生支持) |
| zero points  | 不支持                                        |
| `g_idx` 重排 | 不支持(act-order GPTQ 走不了这条路)          |
| group_size   | 仅 `128`                                      |
| K 对齐       | `K % group_size == 0`(TP 切分后的分片 K)     |

## 5. 实现要点

### 5.1 权重布局转换(`process_weights_after_loading`)

compressed-tensors checkpoint 给到的是打包格式,这里一次性解包成裸 int8:

```
checkpoint:  weight_packed [N, K//4] int32   (int32 的第 i 个字节 = K 索引 4*j+i,
                                              以无符号存储 = 有符号值 + 128)
             weight_scale  [N, K//G] fp16
    ↓ unpack
kernel 侧:   weight_packed [N, K]    int8    (行主序,K 连续)
             weight_scale  [N, K//G] fp16    (contiguous)
```

解包按 **8192 行一块**分批做:`(chunk >> shifts) & 0xFF` 的 int32 中间张量是最终
int8 结果的 4 倍大,整表一次性展开会在模型加载阶段爆显存。

### 5.2 Triton kernel(`_w8a16_gemv_kernel`)

- **一个 program 负责一个输出通道 N**:`grid = (N,)`。int8 权重行只从 HBM 读一
  次,在寄存器里被 `BLOCK_M` 行激活复用。
- **X / Y 由 host 侧 pad 到 `BLOCK_M = next_power_of_2(M)` 行**,kernel 内部完全
  无 mask。sm75 上带 mask 的 load 会阻断向量化,是实测的主要性能杀手。
- **归约放在 K 循环之外**:循环内累加到 `[BLOCK_M, BLOCK_K]` 的 fp32 累加器,循环
  结束后才 `tl.sum(axis=1)`。循环内做归约会变成延迟受限,慢 2–4x。
- **`BLOCK_K` 在 host 侧选**:`_pick_block_k()` 从 1024/512/256/128 里挑第一个能
  整除 K 的值(kernel 内无 K 方向 mask)。
- 累加与反量化全程 fp32,`w * s` 只算一次并广播到 M 行。
- `tl.multiple_of` / `tl.max_contiguous` 提示编译器生成向量化访存。

### 5.3 M > 8 的 fallback

prefill / 大 batch 走 `w_deq = w_q * w_s` 一次性反量化 + `x @ w_deq.t()`(cuBLAS
fp16)。理由见 §1:sm75 上没有能赢过 cuBLAS 的融合路径。代价是每次调用产生一个
[N, K] 的 fp16 临时张量。

### 5.4 注册为 opaque custom op

通过 `direct_register_custom_op("w8a16_triton_gemv", ...)` 注册,并提供了
`fake_impl`。这样做是**必须的**:如果让 inductor trace 进去,它会对分解后的
dequant 表达式做 combo-kernel benchmark,在 sm75 上会把编译 worker 卡住好几分钟。

## 6. 数值行为

- 权重反量化是精确的(int8 → fp32 无损),误差只来自 fp32 累加后写回 fp16。
- 与 "先反量化成 fp16 再 cuBLAS" 的参考实现相比,本 kernel 的累加精度更高
  (fp32 累加 vs cuBLAS fp16 输入路径),不会引入额外精度损失。
- 注意 M ≤ 8 与 M > 8 走的是两条不同路径,二者结果在 fp16 舍入量级上可能有微小
  差异 —— 这是 batch 相关的正常抖动,不是 bug。

## 7. 快速验证

数值对拍(直接调底层函数,不需要起服务):

```python
import torch
from vllm.model_executor.kernels.linear.mixed_precision.triton_w8a16_gemv import (
    triton_w8a16_gemv,
)

torch.manual_seed(0)
M, N, K, G = 4, 4096, 4096, 128
x = torch.randn(M, K, device="cuda", dtype=torch.float16)
w_q = torch.randint(-127, 128, (N, K), device="cuda", dtype=torch.int8)
w_s = torch.rand(N, K // G, device="cuda", dtype=torch.float16) * 0.01 + 1e-3

w_deq = (w_q.view(N, K // G, G).float() * w_s.view(N, K // G, 1).float()).view(N, K)
ref = x.float() @ w_deq.t()
out = triton_w8a16_gemv(x, w_q, w_s, G)
print("max abs err:", (out.float() - ref).abs().max().item())
print("rel err:", ((out.float() - ref).norm() / ref.norm()).item())
```

端到端:开 `VLLM_W8A16_TRITON_GEMV=1` 起服务,确认日志里出现
`TritonW8A16GemvLinearKernel active`,再跑一次相同 prompt 与关闭时对比输出。

## 8. 已知限制

- 只覆盖 `uint8b128` + `group_size=128` + fp16 激活。其它组合直接 `can_implement`
  返回 False。
- `_pick_block_k()` 在 `K % 128 != 0` 时返回 0(会导致 kernel 启动失败)。走
  `can_implement` 的正常路径不会命中,但**直接调用 `triton_w8a16_gemv()` 时需要
  自己保证 K 对齐**。
- M 不是 2 的幂时会额外拷贝一份 padded 激活。M ≤ 8 时开销可忽略。
- `grid = (N,)` 且每 program 只用 4 个 warp,N 较小的层无法填满 SM。Qwen3.5-9B
  这类形状(N ≥ 4096)不受影响。
- 没有 autotune;`BLOCK_K` 是启发式选的,`num_warps` 固定为 4。
- 没有配套的单元测试进 `tests/`。

## 9. 相关文件

| 文件                                                | 关系                                                                                                                                                 |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kernels/linear/__init__.py`                        | 把本 kernel 插入 `_POSSIBLE_KERNELS[CUDA]` 首位                                                                                                      |
| `kernels/linear/mixed_precision/__init__.py`        | 导出 `TritonW8A16GemvLinearKernel`                                                                                                                   |
| `layers/draft_w8a16_lmhead.py`                      | 同一思路的**独立**实现:对投机解码 drafter 的 lm_head 做**运行时**原地 W8A16 量化(`VLLM_DRAFT_LMHEAD_W8A16=1`),走的是 quant_method 替换而非 kernel 选择 |
| `models/qwen3_5_mtp.py`                             | 在 `load_weights()` 末尾调用 `maybe_quantize_draft_lmhead()`                                                                                          |

两者的区别:本 kernel 处理的是**checkpoint 里已经量化好**的权重,接入 vLLM 的
MPLinearKernel 选择体系;`draft_w8a16_lmhead.py` 处理的是**原本是 fp16**、在加载
后被临时量化掉的 drafter lm_head,靠 rejection sampling 保证输出正确性。
