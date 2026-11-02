# FusedRMSNormPosQKVQKNorm 需求分析

## 1. 算子概述

融合算子，将四个阶段合并为一次计算图：

```
h   = rmsnorm(x, gamma, eps) + pos_emb      # 逐行 RMSNorm（约 D 维）+ 位置编码加法
qkv = h @ Wqkv                              # QKV 线性投影 [BS, D] x [D, 3H]
q, k, v = split(qkv, 3, dim=-1)             # 各 [BS, H]
q   = rmsnorm(q, gamma_q, eps)              # 逐行 RMSNorm（约 H 维）
k   = rmsnorm(k, gamma_k, eps)
输出 = (q, k, v)
```

其中 `rmsnorm(x, g, eps) = x / sqrt(mean(x^2, dim=-1) + eps) * g`。

关键观察：

- **两次 RMSNorm 的归约维度不同**：第一次约 D（gamma 形状 [D]），第二次约 H（gamma_q/gamma_k 形状 [H]）。归约均按整行进行，**不按 head 分组**——`num_heads` 仅是元数据，不参与任何数学计算（golden 中 q 的 mean 是对整个 H 维）。
- 数据依赖是严格串行链：norm1 → add → GEMM → split → norm2。GEMM 是主要计算量（最大 case 约 3.2 GFLOP），RMSNorm/加法是 Vector 操作，属典型的 Vector+Cube 混合算子。
- golden 全程 fp32 计算（含 matmul），仅在最终 return 处转回原 dtype。精度容差按 MERE/MARE 阈值判定（见 §3），允许中间用输入 dtype 做 Cube dot（fp32 累加）。

## 2. 目标芯片

| 项目 | 值 |
|---|---|
| 芯片型号 | Ascend910B3 |
| NPU 架构 | dav-c220 |
| AI Core 数量 | 20（Cube 20 + Vector 40，device_info.json 口径） |
| UB | 192 KB / 核 |
| L1 | 512 KB |
| L0A / L0B / L0C | 64 / 64 / 128 KB |
| L2 Cache | 192 MB |
| 后端 | triton 3.2.0 + triton-ascend 3.2.1，target npu/Ascend910B3，SPMD block program |

### 针对该芯片的 tiling 建议

- **多核切分**：以行为主维度切 program。大 case（BS=2048）下 BLOCK_M=32~64 可得 32~64 个 program，超过 20 核，负载充分；小 case（BS=64~256，如 case 1/7/14/15/18/21）1D 行切分只有 1~4 个 program，**建议 K1 采用 2D grid（M 块 × N 块）**，N 方向按 BLOCK_N=128 切（3H 最多 1536 → 12 块），以 Wqkv 列块重读（L2 命中，代价小）和 h 重复计算（每 program 一次 D 维归约，开销小）换核间并行度。
- **tile_size 上限估算（UB 192KB 预算内）**：
  - D=512：BLOCK_M=32~64，BLOCK_N=64~128，K 内循环 chunk=128。驻留量：h fp16 [64,512]=64KB、W tile fp16 [128,128]=32KB、acc fp32 [64,128]=32KB，合计约 128KB，留余量给 x/pos 分段（fp16 载入后 fp32 计算，避免整块 [64,512] fp32=128KB 峰值）。需编译+profiler 验证实际驻留。
  - D=128/256：W tile 小得多；D≤256 时可让 BLOCK_N 覆盖整个 H（q/k tile 一次到位，单 kernel 内完成第二次归一化，见 §6 路由策略）。
- **Cube 使用**：QKV 投影必须走 `tl.dot`（Cube，fp32 累加）。K=D≤512，一次 K 循环（1~4 个 chunk）即可；Wqkv 最大 1.5MB，20 个 program 共享 L2，权重重复读不构成瓶颈。
- **L0 约束**：L0A/L0B 各 64KB，限制单次 dot 的 K_CHUNK×BLOCK_N ≤ 64KB（fp16），即 128×256 组合已到边界，D=512 时选 K_CHUNK=128、BLOCK_N≤128 更稳妥；由编译器分块实现，最终以编译结果为准。

## 3. 接口分析

- **算子注册名：`fused_rmsnorm_pos_qkv_qknorm`，即 `cann_bench.fused_rmsnorm_pos_qkv_qknorm`**。依据：proto.yaml `schema` 字段与 desc.md 使用示例一致，已是 snake_case，不做额外大小写变换。实现后需用同一解释器核对 `cann_bench` 包实际导出名与 `cann_bench.__file__`，防止旧构建副本。
- 调用签名（proto.yaml schema，位置顺序固定）：

```python
cann_bench.fused_rmsnorm_pos_qkv_qknorm(
    x,          # Tensor [B, S, D]      fp16/bf16
    gamma,      # Tensor [D]            fp16/bf16
    pos_emb,    # Tensor [1, S, D]      fp16/bf16，对 B 广播
    Wqkv,       # Tensor [D, 3*H]       fp16/bf16
    gamma_q,    # Tensor [H]            fp16/bf16
    gamma_k,    # Tensor [H]            fp16/bf16
    num_heads,  # int，2~16，数学上无作用
    eps,        # float，1e-6（默认）/1e-5
) -> (q, k, v)  # 各 [B, S, H]，dtype 同输入，连续布局
```

- H 由 `Wqkv.shape[1] // 3` 推导；全部 case 满足 H = D，head_dim 恒为 64，但实现按通用 H 处理。
- **eps 是运行时属性**（case 11/17 为 1e-5），必须作为 kernel 参数传入，不得 constexpr 固化。
- 精度要求：fp16 阈值 T=2^-10≈0.00098（MERE < T 且 MARE < 10T）；bf16 阈值 T=2^-7≈0.0078（同条件）。
- 输出为三个独立连续 tensor [B,S,H]，由 host 分配。

## 4. 用例覆盖分析

共 22 个 case，全部为同 dtype 六输入；fp16 × 14，bf16 × 8。

| 维度 | 取值 | 备注 |
|---|---|---|
| B | 1, 2, 4, 8 | — |
| S | 64, 128, 256, 384, 512, 1024, 2048 | 384/2048 非 2 的幂，但均被 64 整除 |
| D (=H) | 128, 256, 512 | 3H 最大 1536 |
| BS 行数 | 64 ~ 2048 | 最大行数 case 12/19 |
| num_heads | 2, 4, 8 | 不参与计算 |
| eps | 1e-6（20 个）、1e-5（case 11/17，均 bf16） | 运行时传入 |

边界/困难 case：

- **case 19**（B2S1024D512，bf16）：最大计算量 2×2048×512×1536 ≈ 3.22 GFLOP，且 D=512 时 3H=1536 最宽，UB 压力最大，是性能与 tiling 的主战场。
- **case 12**（B1S2048D256，fp16）：S 最大，行数 2048。
- **case 7/18**（D=128，B1S64/B8S64）：最小规模，1D grid 核利用率低，考察 2D grid / 小 BLOCK_M 策略。
- **case 15**（B2S64D256，fp16）：x 值域 [-2,2]、W [-0.1,0.1] 为全部 case 中最大幅度，fp16 中间量上限约 √256×2×0.1×1.5≈4.8，无溢出风险。
- **case 11/17**：eps=1e-5，验证 eps 未被固化。
- 全部维度均被 64 整除，正常 BLOCK 选择下尾块 mask 极少触发，但 mask 仍须保留（防御 B*S 非 BLOCK_M 倍数，如 BS=384/96 等组合）。
- 值域有界（x∈[-2,2]，W∈[-0.1,0.1]，pos∈[-0.2,0.2]），qkv 幅值 O(1)~O(10)，fp16/bf16 中间量化安全。

## 5. 实现难点

1. **跨 N 块的第二次 RMSNorm 依赖**：q 列（N∈[0,H)）的均方和横跨多个 N 块。Triton SPMD 无跨 program 屏障，也无寄存器数组的循环追加能力——要么单 N 块覆盖整 H（D≤256 可行），要么引入 workspace / 重复计算（D=512 必须二选一）。
2. **UB/L0B 容量与 D=512 的矛盾**：Wqkv 的 q 列块 [512,512] fp16 = 512KB，远超 UB(192KB) 与 L0B(64KB)，D=512 时 q/k 必然多 N 块。
3. **fp32 matmul 的 Cube 支持未证实**：golden 用 fp32 matmul；triton-ascend 3.2.1 的 `tl.dot` 需按 fp16/bf16 输入 + fp32 累加设计（h 先 cast 回输入 dtype）。由此引入的相对误差约 2^-11（fp16）/2^-8（bf16），低于阈值 2^-10/2^-7；且值域有界，即使 bf16 dot 不可用而降级 fp16 dot，数值上仍安全（需编译验证）。
4. **pos_emb 广播寻址**：pos_emb 为 [1,S,D]，行偏移 = (row % S) * D，需按行向量做模运算寻址，不能按 x 的连续行号直接偏移。
5. **精度链路**：两次归约（mean over D、mean over H）都必须 fp32 累加；h cast 到输入 dtype 的时机在 pos 加法之后；第二次归一化直接用 dot 的 fp32 累加结果（若用 fp16 workspace 会额外引入 ~2^-11 误差，优先 fp32 workspace）。
6. **小 case 核利用率**：BS=64 时 1D grid 仅 1~4 program，需要 2D N 切分或小 BLOCK_M 才能铺满 20 核。
7. **反作弊合规**：全部核心计算（两次 RMSNorm、加法、matmul、split、cast）必须在 `@triton.jit` 内完成；host 仅做输出/workspace 分配与 kernel 启动；不得用 torch/aclnn 代算，不得缓存结果。

## 6. Triton Ascend 实现建议

### 总体结构（两 kernel 基线 + 形状路由优化）

**Kernel 1（fused norm+add+GEMM，Vector+Cube 混合）**：grid 按 M 块（小 BS 加 N 块维）。

1. 载入 x [BLOCK_M, D]、pos [BLOCK_M, D]（pos 行号 = row % S），fp32 计算 `h = x/sqrt(mean(x²)+eps)*gamma + pos`，cast 回输入 dtype 得 h_tile [BLOCK_M, D]（D=512 时 64KB）。
2. N 维循环（覆盖 3H，BLOCK_N 步进）：K 维 chunk 循环载入 Wqkv [K_CHUNK, BLOCK_N]，`tl.dot(h_tile, w_tile, acc fp32)`；每个 N 块结果以 fp32 存入 qkv workspace [BS, 3H]（最大 2048×1536×4 = 12.6MB；若验证精度允许可降为 fp16 减半流量）。
3. h 只在片上流转，不落 HBM。

**Kernel 2（split+二次归一化，纯 Vector）**：grid 按 M 块（BLOCK_M2=16~32）。

1. 载入 workspace 的 q 段 [BLOCK_M2, H] fp32 → fp32 归约 mean(q²)+eps → `q/rms*gamma_q` → cast 输出 dtype 存 q。
2. 同法处理 k 段；v 段直接 cast 存出。
3. workspace 只被本 kernel 读，由 Kernel 1 全量先写，满足"读前初始化"要求。

### 形状路由优化（Phase 4 方向）

- **D≤256：单 kernel 方案**。BLOCK_N 覆盖整 H，q/k tile 一次 dot 到位后在寄存器内完成均方和与归一化，免 workspace、免 kernel2（v 可同 kernel 分块写出）。D=256 时 Wq 块 [256,256] fp16=128KB + h tile 8~16KB + acc 16~32KB，UB 约 176KB 峰值，需 BLOCK_M=16 并经编译验证；D=128 时余量充足。
- **D=512：二选一**——(a) 上述两 kernel；(b) 单 kernel 内对 q/k 各做两遍 dot（第一遍累计 sum-of-squares 不存储，第二遍重算并归一化），dot FLOPs 变为 5/3 倍（+2.1 GFLOP ≈ Cube 上十几 µs），换取省去 25MB fp32 workspace 往返与一次 launch，需 benchmark 对比。
- fp16 workspace（流量减半）作为 D=512 两 kernel 方案的次级优化，先验证 MERE/MARE。

### 精度与健壮性要点

- 归约：`tl.sum(x*x, axis=1) / D`（fp32），eps 以 fp32 标量参数参与。
- 所有 load/store 带 mask（行维 `rows < BS`、列维按 D/H 尾块），stride 按实际张量传入；host 侧对非连续输入先 `.contiguous()`（属数据准备，不违反复合规）。
- 输出 dtype cast 只在最终 store 处做一次；q/k/v 三个输出独立分配、独立写。
- 自测覆盖：全部 22 case、同 shape 换数据重复调用、eps=1e-5 组、bf16 全链路、首个 JIT 编译真实通过（import 成功 ≠ 编译通过）。
