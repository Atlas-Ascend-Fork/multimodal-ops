# FusedMulAddLayernormMulAdd 需求分析（ANALYSIS.md）

> 来源：`算子3/fused_mul_add_layernorm_mul_add_20261003_132221` 工作目录（develop 4 轮 + eval 4 轮），最终实现为 operator_iter/iter4。

---

## 1. 算子概述

**FusedMulAddLayernormMulAdd**：DiT（Diffusion Transformer）注意力残差更新与 MLP 自适应 LayerNorm 的融合算子。

```
residual = x + gate * attn_out
y        = LayerNorm(residual) * (1 + scale) + shift
```

- 输入 `x [B, N, C]`（残差输入）、`attn_out [B, N, C]`（注意力输出含输出投影）、`gate [B, C]`（残差门控，按 N 广播）、`scale [B, C]`（LayerNorm 调制，按 `(1 + scale)` 应用）、`shift [B, C]`（LayerNorm 平移，按 N 广播）
- 属性 `eps`（方差正则项，DiT 用 1e-6）
- 输出 `residual [B, N, C]` 与 `y [B, N, C]`；LayerNorm 无固定仿射参数
- 类别 **Fusion**、难度 **L3**——逐 token 的 LayerNorm 统计 + 逐元素链式计算，是典型的访存密集型 Vector 融合场景

融合价值：独立实现需要 3 次完整 HBM 往返（mul-add 写 residual、LayerNorm 读 residual 写中间量、调制加 shift 写输出）；本实现单 kernel 内完成 mul-add → LayerNorm 统计 → 调制输出，residual tile 在片上驻留（LayerNorm 消费零 L2/HBM 重读），HBM 往返压缩为一次读（5 个输入）+ 一次写（2 个输出）。

---

## 2. 计算语义（与 golden 逐位对齐）

| 步骤 | 计算 | 舍入点 |
|---|---|---|
| residual | `r = x + gate * attn_out` | native dtype 逐操作舍入（`ga`、`r` 两处） |
| LayerNorm 统计 | fp32 两遍方案：先 mean，再中心化二阶矩，`rstd = 1/sqrt(var + eps)` | 统计全程 fp32 |
| 归一化 | `norm = (r - mean) * rstd`，落回 native dtype（`nn32` 舍入点） | native dtype |
| 输出 | `y = norm * (1 + scale) + shift`（`t = 1+s`、`y1`、`y` 三处 native 舍入点） | native dtype |

与 golden 的逐位一致性经 36 case 精度自测确认（residual 全部位级一致；y 在 2T 门限内，仅 case 10/15 各 1 个双舍入边界元素触线，fp64-margin 下 impl==golden）。

---

## 3. 实现结构（Triton-Ascend，4 个 kernel + host 三级路由）

| kernel | 适用 | 说明 |
|---|---|---|
| `_fused_mul_add_layernorm_mul_add_kernel` | C ≤ 1024 | 一个 program 服务一个 (b, n) token 行，BLOCK_C = next_pow2(C) |
| `_fused_mul_add_layernorm_mul_add_wide_kernel` | 1024 < C ≤ 2048 且行数 ≤ 512 | 一行多 program 分块 |
| `_fused_mul_add_layernorm_mul_add_wide_rows_kernel` | 1024 < C ≤ 2048 且行数 > 512 | 多行 wide kernel + 二级路由：`P = min(seq, max(1, 40//batch))`（40 = Vector core 数），参数一次加载行间复用 |
| `_fused_mul_add_layernorm_mul_add_blocked_kernel` | C > 2048 | 行内分块 + 两遍统计跨块累计 |

host 路由为纯元数据谓词（只用 channels/batch/seq，无 case 编号、无 benchmark 身份判断），任意合法输入恰落一条路径。

LayerNorm 统计采用两遍（mean → 中心化二阶矩）fp32 方案，近常数行保持数值稳定；逐元素链保留 golden 的 native-dtype 舍入点。

---

## 4. 评测结论（eval/iter4，36 case 全过）

| 指标 | 值 |
|---|---|
| 通过率 | **100%**（36/36） |
| 综合得分 | **90.32** |
| 平均加速比 | **3.58x**（min 1.31x / median 3.48x / max 8.4x） |
| kernel 耗时 | min 1.58us / median 12.48us / max 43.44us |

迭代轨迹：iter1 88.07（3.34x）→ iter2 87.05（3.18x，融合方案回退修正）→ iter3 88.27（3.38x）→ iter4 **90.32（3.58x）**（多行 wide kernel + 二级路由落地）。num_stages=2 探针无收益，按验收标准回退为默认 `tl.range`。

详细设计推演见 [DESIGN_RATIONALE.md](DESIGN_RATIONALE.md)。
