# ANALYSIS — fused_patch_concat_layernorm

> 本文档由 pypto-pro-workflow 运行产物（develop/iter0~iter11、eval/iter0~iter11）的事实数据整理而成；融合方案决策详见 [DESIGN_RATIONALE.md](DESIGN_RATIONALE.md)。

## 算子概述

Swin Transformer v1 patch merging 阶段的「patch 拼接 + LayerNorm」融合算子（线性降维前级），Triton-Ascend 实现，单 kernel 完成全部数值计算：

- 输入：`x [B, H*W, C]`（spatial row-major tokens）、`weight/bias [4C]`（LayerNorm 参数）、`H`/`W`（偶数空间尺寸）、`eps`
- 输出：`[B, (H/2)*(W/2), 4C]`，dtype 与输入一致
- 融合方式：空间 2×2 邻域拼接（C→4C）与 LayerNorm 归一化在片上一次完成，避免中间 token 展开/回写

## 开发与评测摘要

| 项目 | 值 |
|------|-----|
| 开发框架 | Triton-Ascend（`@triton.jit`） |
| 最佳快照 | iter10 交付代码（sha256 `2c278e28…`，iter11 逐字节未变），最终以 `eval/iter12` 复测为最终成绩 |
| 精度 | 30/30 case 全部通过（含连续调用与变输入检查） |
| 平均加速比 | 1.966×（同一交付代码的最终复测，30/30 精度通过，全部 case speedup ≥ 1.0） |
| 实测设备 | Ascend910B3（dav-c220），AI Core 20（Cube 20 + Vector 40） |
| 软件版本 | triton 3.2.0 + triton-ascend 3.2.1，torch_npu 2.7.1.post8，CANN 9.1.0 |
| 评测协议 | cann-bench `kernel_details`（kernel 执行耗时口径） |

## 迭代历程（11 轮有效评测）

- iter0~iter3：建立 single/paired（2×2 patch 聚合）双路径框架，完成精度收敛
- iter4~iter9：针对 per-iteration DMA 与 compute 分解做块级调优（BLOCK_C/BLOCK_2C 扫描、corr 消除实验）
- iter10：SKIP_CORR constexpr 条件性消除 + 块参数定型，实测 1.926×
- iter11：三项 Stage9 任务中 T2（BLOCK_C=256/BLOCK_2C=512 扩展）经 A/B 一致退步被否决，交付代码与 iter10 相同
- iter12：对同一交付代码复测，最终成绩 1.966×（overall 83.29），工作流终止
