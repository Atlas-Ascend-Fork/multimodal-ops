# ANALYSIS — fused_groupnorm_mul_add_silu

> 本文档由 pypto-pro-workflow 运行产物（develop/iter0~iter3、eval/iter1~iter3）的事实数据整理而成；融合方案决策详见 [DESIGN_RATIONALE.md](DESIGN_RATIONALE.md)，case 14 未达标（0.97×）的专项分析见工作流记录。

## 算子概述

`y = SiLU(GroupNorm(x, weight, bias, groups=32, eps) * (1 + scale) + shift)`（NCHW，DiT 风格 adaptive GroupNorm + SiLU），Triton-Ascend 实现：

- 输入：`x [B, C, H, W]`（fp16/bf16/fp32）、`weight/bias [C]`（GroupNorm 仿射）、`scale/shift [B, C]`（条件调制）
- 输出：`y [B, C, H, W]`，dtype 与输入一致
- 融合方式：GroupNorm 统计/归一化、affine、条件调制与 SiLU 单次融合，避免中间张量往返

## 三路径分发（F2 vertical fusion + F5.schedule_routing）

1. **小组合单 tile kernel**：整组以 [block_cpg, block_hw] 常驻 tile 一次载入，两Pass 方差 + 向量化 per-channel 参数，epilogue 超 UB 预算时按 HW 分块重读
2. **stats+apply 拆分流水线**（大 fp32 组，N ≥ 49152）：stats kernel 算原始矩 partial 和写入 GM workspace，apply kernel 合并 partial 得 mean/var/rstd 并执行逐通道 epilogue
3. **streaming kernel**（其余）：合并单遍原始矩统计（s1/s2，少读一遍 x），双Pass 兜底由 constexpr 开关保留

## 数值语义（对齐 GroupNorm32 混合精度约定）

- fp32：以 double-double 系数对（Dekker/Knuth error-free 链）评估 `z = v*A + B'`，合并单系数 + 精确残差形式（~5 flops/元素）跟踪 double-double 精度
- bf16：fp32 链 + 输出端整数 RNE 单次舍入；fp16：逐 op 舍入
- SiLU 在 fp32 计算后单次舍入，对齐 `torch.nn.functional.silu`

## 开发与评测摘要

| 项目 | 值 |
|------|-----|
| 开发框架 | Triton-Ascend（`@triton.jit`） |
| 最终交付 | iter3 |
| 精度 | 36/36 case 全部通过 |
| 平均加速比 | 2.85×（整体） |
| 未达标项 | case 14（fp32，[1,192,256,256]）speedup 0.97×（GroupNorm 全 fp32 计算下三 dtype kernel 耗时几乎相同，fp32 baseline 反而更快，详见工作流 case14 分析） |
| 评测总分 | 87.05 |
| 实测设备 | Ascend910B3（dav-c220），AI Core 20（Cube 20 + Vector 40），UB 192KB |
| 软件版本 | triton 3.2.0 + triton-ascend 3.2.1，torch_npu 2.7.1.post8，CANN 9.1.0 |
| 评测协议 | cann-bench `kernel_details`（kernel 执行耗时口径） |

## 迭代历程（3 轮）

- iter1：三路径框架建立（1.81×，36/36 通过）
- iter2：常驻 tile 与 streaming 分派调优（2.71×）
- iter3：stats+apply 拆分流水线引入（2.85×），工作流终止
