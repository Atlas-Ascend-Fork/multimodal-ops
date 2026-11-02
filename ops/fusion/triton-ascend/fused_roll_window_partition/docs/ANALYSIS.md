# ANALYSIS — fused_roll_window_partition

> 本文档由 pypto-pro-workflow 运行产物（develop/iter0~iter8、eval/iter1~iter9）的事实数据整理而成；融合方案决策详见 [DESIGN_RATIONALE.md](DESIGN_RATIONALE.md)，case 22-24 基线高耗时的根因分析见工作流记录（CANN `aclnnRoll` 在 C=65 非对齐通道上触发低效 kernel）。

## 算子概述

Swin Transformer 的「循环移位（roll）+ 窗口划分」融合算子（NHWC），Triton-Ascend 实现。两个阶段都是纯数据搬运，整个算子是一次位精确拷贝：

    windows[n, u, v, c] = x[b, (i*M + u + s) % H, (j*M + v + s) % W, c]

- 输入：`x [B, H, W, C]`（NHWC，fp16/bf16/fp32），`window_size = M`，`shift_size = s`
- 输出：`[B*(H//M)*(W//M), M, M, C]`，位精确（bit-exact）
- 融合方式：roll 与 partition 在一次拷贝中完成，省去中间 roll 张量的整体读写

## 双路径路由（F5.fusion_boundary_routing）

- **行路径**（默认）：每输出行按 W 回绕点拆成两段严格连续的段拷贝（段 B 在 s==0 时自然零趟），grid 固定为 Vector 核数、grid-stride 步进
- **窗口 2D tile 路径**（多窗口 + 宽行 + UB 预算内）：每窗口一个 grid-stride 迭代，四个回绕象限以 TU×TC 二维 tile 拷贝（TU 行共享一次事务对）；`s == 0` 时编译 SHIFT_ZERO 专用变体，B/C/D 象限循环被 constexpr 消除

## 开发与评测摘要

| 项目 | 值 |
|------|-----|
| 开发框架 | Triton-Ascend（`@triton.jit`） |
| 最终交付 | iter9（与 iter8 交付逐字节一致，sha256 `5e853c64…`） |
| 精度 | 30/30 case 全部通过（位精确拷贝） |
| 平均加速比 | 33.08×（最差 case 3.8×，全部 case speedup ≥ 1.0） |
| 评测总分 | 100.17（iter9 复测；iter8 100.20） |
| 实测设备 | Ascend910B3（dav-c220），AI Core 20（Cube 20 + Vector 40），UB 192KB |
| 软件版本 | triton 3.2.0 + triton-ascend 3.2.1，torch_npu 2.7.1.post8，CANN 9.1.0 |
| 评测协议 | cann-bench `kernel_details`（kernel 执行耗时口径） |

## 迭代历程（9 轮）

- iter1：行路径基线（4.94×，30/30 通过）
- iter2：窗口 tile 路径引入（24.5×）
- iter3~iter5：tile 准入谓词按实测校准（窗口数门槛 40→32→12，行宽门槛 1536→1，per-chunk 二级准入）
- iter6~iter8：TU/TC 几何与路由细化（32.9×，overall 稳定 100+）
- iter9：最终复测（33.08×），工作流终止
