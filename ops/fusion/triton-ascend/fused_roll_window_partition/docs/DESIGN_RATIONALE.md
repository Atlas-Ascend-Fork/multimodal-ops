# 设计思路详解 — iter8 开发轮（输出目录 develop/iter8）

> 交付代码：`impl/cann_bench/fused_roll_window_partition.py`（sha256 `5e853c64151050156d52d16b8fb2256b54474700c4ec0495c725415492fc1eca`，与 selection/records/iter8-*/impl 逐字节一致）。
> 本轮按 Stage9 v2 任务单执行：T1（P1 inspect）+ T2（P2 modify）+ T3（P2 inspect）。**T2 原型经受控实测为性能回退，按其验收门槛撤销；交付代码回到 eval/iter8 状态，本轮净源码改动为零**，全部产出为机制证据与负结果。
> 原始日志：`develop/iter8/self_test.log`；机器可读明细：`self_test_detail.json`；T1 数据：`t1_route_trace.json`、`t1_forced_tile_result.json`；T2 数据：`t2_ab_result.json`、`t2_mw_msprof_comparison.json`、`t2_mw{1,2}_op_summary.csv`；T3 数据：`t3_case14_aic_summary.json`、`t3_case{14,13}_op_summary.csv`、`t3_tc2048_result.json`。临时脚本均位于 /tmp/opencode。

## 0. 重要更正：Stage9 分析中 case19/20/21 的形状参数有误

任务单及其依据（`profile/iter8/bottleneck_analysis.md`）将 case19/20/21 记为 "[4,24,24,512] **M=8** s=4, nwin=18, row_elems=4096"。**实际 `task/cases.yaml` 中该组为 window_size=12、shift_size=6**：

| 量 | 任务单假设 | 实际值 |
|---|---|---|
| window_size M | 8 | **12** |
| num_windows | 18 或 36 | **16**（B=4 × (24/12)² = 4×2×2） |
| row_elems | 4096 | **6144**（12×512） |

由此，任务单 T1 的核心论据"primary UB = 8×4096×2 = 65.5KB ≤ 96KB 应通过"不成立。第 1 节按实际值重新 trace。

## 1. 按 task_id 的完成情况

### T1（P1, inspect）：case19/21 路由核实 + 强制 tile 计时 — 已完成（未修改任何源文件）

**路由 trace（host 谓词逐条复算，安装包实际函数 `impl._use_tile_kernel`，证据 `t1_route_trace.json`）**：

对 [4,24,24,512]，M=12，s=6：

| case | dtype/elem_bytes | num_windows | row_elems | primary UB: M×row_elems×elem_bytes ≤ 98304？ | secondary: row_elems ≤ 3072？（TU×TC×eb=16384B(fp16)/32768B(fp32)） | `_use_tile_kernel` 返回 | 实际执行 kernel（profiler 抓取） |
|---|---|---|---|---|---|---|---|
| 19 | fp16 / 2B | 16 | 6144 | 12×6144×2=**147456B > 98304 → 拒** | 6144 > 3072 → 拒 | **False** | `_roll_window_partition_kernel`（row）✓ |
| 20 | fp32 / 4B | 16 | 6144 | 294912B > 98304 → 拒 | 6144 > 3072 → 拒 | **False** | row ✓ |
| 21 | bf16 / 2B | 16 | 6144 | 147456B > 98304 → 拒 | 6144 > 3072 → 拒 | **False** | row ✓ |

**结论（对照验收要点）**：

1. case19/21 在 row 路径是**设计内正确行为（intentional override），不是 bug**：primary UB 检查（147456B）与 secondary cap（6144>3072）**双双拒绝**，不存在任务单猜测的"primary 通过却被 secondary 误拦"的 order-of-checks 问题——primary 本身就未通过。与 history insight 11 的既有记录（"Cases 19/20/21 (row_elems=6144) correctly excluded by secondary cap"）一致。
2. **强制 tile 冷测（case19 fp16，n=8 中位数 × 6 输入，证据 `t1_forced_tile_result.json`）**：row=6.670us vs tile=6.520us，**tile 仅 -2.25%**（逐位一致）。row_elems=6144 每象限行需 6 个 TC=1024 列迭代，事务数缩减被列迭代稀释——与 develop/iter5 case21 bf16 -2.0%（噪声）的历史结果一致。**即使放宽路由也无实质收益，row 路径维持**；本轮不改 `_use_tile_kernel`。
3. case19 的 8.38us（eval 口径）处于 row kernel 结构地板（iter7 T2 msprof：scalar 2.15us/44% + MTE 61%），本轮数据不改变该结论。

### T2（P2, modify）：MULTI_WIN 多窗口/程序原型 — 已实现并实测，**按验收门槛撤销（verified-negative）**

**实现（曾真实部署到 NPU 实测，后撤销）**：按任务单 method，`_roll_window_partition_tile_kernel` 增加 `MULTI_WIN: tl.constexpr`，grid-stride 循环改为 `for n0 in range(pid*MULTI_WIN, NWIN, num_core*MULTI_WIN)` + `tl.static_range(MULTI_WIN)` 内层展开；dispatch 按 `shift_size==0 and nwin>=16 and nwin%2==0` 传 `MULTI_WIN=2`（grid 16→8）。窗口→数据映射不变，逐位正确。

**受控 A/B 实测（case29 形状，8 输入 × n=30 kernel Duration 中位数，证据 `t2_ab_result.json`）**：

| case | MW1 (grid=16) | MW2 (grid=8) | Δ | 逐位 |
|---|---|---|---|---|
| 28 fp16 | 2.32us | 2.48us | **+6.9%** | 两者均逐位 ✓ |
| 29 fp32 | 2.34us | 2.48us | **+5.98%** | ✓ |
| 30 bf16 | 2.36us | 2.48us | **+5.08%** | ✓ |

24 组输入级中位数对比**全部为负**（MW2 每组都更慢），非噪声。

**机制诊断（msprof PipeUtilization，case29 fp32，n=26 中位数，证据 `t2_mw_msprof_comparison.json`）**：

| 指标 | MW1 (Block=16) | MW2 (Block=8) | Δ |
|---|---|---|---|
| Task Duration | 2.68us | 2.78us | **+0.10us (+3.7%)** |
| aiv_scalar | 1.004us | 0.878us | **-0.126us**（框架摊销确实兑现） |
| aiv_mte2+mte3 | 0.450us | 0.904us | **+0.454us（翻倍）** |

任务单的收益模型（"每程序框架成本被 2 窗口摊销，预计省 ~0.7us"）被实测证伪，原因：

- **摊销收益确实存在但很小**：scalar 下降 0.126us（arange/循环控制跨 2 窗口复用）；
- **但 MTE 关键路径翻倍**：MW1 下 16 个程序把 16 个窗口的 MTE2/MTE3 传输**跨核并行**；MW2 下只有 8 个程序，每程序串行处理 2 个窗口，第二个窗口的 tile 拷贝必须排在第一个之后（程序内顺序依赖），单程序 MTE 忙时从 0.45us 涨到 0.90us。核间并行度的损失（16→8）大于标量摊销的收益；
- 净效应：aiv 总忙时 +0.30us，Duration +3.7%（msprof 口径）/ +5.1~6.9%（profiler 中位数口径）。

**结论**：cases 28-30 的 ~2.6us 地板中，MTE 传输虽只占 ~0.45us，但它**已经完全跨核并行**——把窗口串行进更少的程序只会拉长关键路径。**"多窗口/程序摊销框架成本"方向对该 16 窗几何证伪关闭**；标量主体（~0.88-1.0us）仍为每程序固定框架价，与 insight 12 的结论互相印证。

**代码去留**：按验收门槛（"kernel time should be <2.0us"、"No regression"）**撤销该改动**，dispatch 恢复 `MULTI_WIN=1` 等价的原实现（即删除 MULTI_WIN 机制），交付代码与 eval/iter8 逐字节一致。原型的全部精度与性能证据已归档。这与 iter7 对 SHIFT_ZERO 的处理（保留）不同：SHIFT_ZERO 实测方向为正（-0.4%~-1.5%），而 MULTI_WIN 实测方向为负，无保留价值。

### T3（P2, inspect）：case14 fp32 管线分解 + TC=2048 实验 — 已完成（未修改任何源文件）

**msprof PipeUtilization（n=26 事件、Block=40 中位数，证据 `t3_case14_aic_summary.json` 与归档 CSV）**：

| 指标 | case14 fp32 | case13 fp16（同形状对照） |
|---|---|---|
| Task Duration | 7.65us | 6.72us |
| aiv_scalar | 2.601us（**47.5%**） | 2.611us（55.1%） |
| aiv_mte2 / mte3 | 1.370 / 1.381us（合计 **2.751us，51.0%**） | 1.060 / 1.111us（合计 47.2%） |
| aiv_vec | 0.419us（7.7%） | 0.235us（5.1%） |

**对照验收判据**：

1. **MTE2+MTE3 = 51.0% < 85% → 非带宽饱和**。"9.4MB 流量 @1TB/s ≈ 9.4us 即纯带宽地板"的推断不成立：msprof 口径下 MTE 忙时仅 ~2.75us，fp16→fp32 数据翻倍时 MTE 只涨 1.27 倍（2.171→2.751us，事务数相同、固定开销主导），说明该 kernel 处于 **scalar/MTE 混合开销态，仍有理论空间**。
2. **scalar = 47.5% > 30% → 触发 TC=2048 / 象限合并调查**。实施 TC=2048 强制实验（直调 kernel，row_elems=1536 单列块、每象限行 2→1 个列迭代，工作集 8×2048×4B=64KB ≤ 96KB 预算，逐位一致）：**case14 +7.9%（7.47→8.06us）、case13 +2.36%——回退**。原因：TC=2048 虽把列迭代减半，但 64KB 工作集逼近 multibuffer 预算、且 1536/2048 的 mask 浪费与 iter4"TC=2048 输给 lane 浪费"的结论一致（本轮把该结论从窄行推广到 SwinB-384 宽行组）。**TU=8 几何下 TC=2048 与象限合并方向关闭**；scalar 的 2.6us 主体是 40 程序 × 每窗 4 象限 × 2 列块 × 2 行块的循环控制与地址计算，属于事务数固有成本，除非改变并行几何无法回收。
3. 结论：case14 维持 TC=1024 现状；9.6us（eval 口径）是当前架构下该组的实际运行点，"带宽地板"判断不成立但"无可低风险行动项"成立。

## 2. 改动前后 Tiling/数据流/多核方案对比

**交付代码（= eval/iter8）**：

- **Tiling**：row kernel（40 核 grid-stride，BLOCK 1024/2048/4096 按行宽路由，每行 1-2 个两段连续拷贝）；tile kernel（每程序 1 窗口，TU=8×TC=1024 2D 块，四象限循环，grid=min(nwin,40)；s=0 组走 SHIFT_ZERO=True 单象限特化）。
- **数据流**：单 kernel 1R1W——`windows[n,u,v,c] = x[b,(i*M+u+s)%H,(j*M+v+s)%W,c]` 由 kernel 地址映射直接实现；片上仅一个 TU×TC chunk 在途（fp32 32KB / 2B 16KB ≤ 96KB 预算）；零中间张量、零 GM 往返、零 H2D。
- **多核**：row=40 程序均分 144-768 行；tile=每窗口一程序（12-40 程序），s=0 组 16 程序。

**T2 原型（已撤销）对比**：唯一差异是 tile kernel 的 s=0 组从 16 程序×1 窗变为 8 程序×2 窗（串行）。逐位语义相同；多核方案上把"跨核窗口并行"换成"程序内窗口串行"，实测标量 -0.126us、MTE 关键路径 +0.454us、Duration +3.7% —— **跨核 MTE 并行的价值大于框架摊销**。该对比表本身就是本轮的机制产出。

**T1/T3 检查**不改变任何 tiling/数据流。

## 3. 预期效果（对正式评测的预测，不作断言）

- 交付代码与 eval/iter8 **逐字节一致**：30 个 case 的正式评测预期与 eval/iter8（avg_speedup 32.97、performance_score 50.20）同分布，仅 run-to-run 噪声（极端加速 case 22-24/28 的 0.2-0.5us 内核波动主导）。
- 本轮**不主张任何性能收益**：唯一候选改动（T2）实测为负并被撤销。本轮价值 = 三组封顶证据：
  1. cases 28-30 多窗口摊销方向关闭（机制：标量 -0.126us vs MTE 关键路径 +0.454us）；
  2. cases 19/21 row 路由确认设计内正确，强制 tile 仅 -2.25%（噪声级），路由谓词无需放宽；
  3. case14 非带宽饱和（MTE 51%）但 TC=2048 实测 +7.9%，TU=8 下 TC/象限方向关闭。
- 自测抽测（n=30 中位数）：28/29/30 = 2.32/2.37/2.31us、19 = 6.67us、13/14/17 = 6.42/7.43/6.65us，全部处于 iter5-iter8 已知噪声带内，无回退迹象。

## 4. 最慢 case 逐个说明（worst-6 by perf_score：28/29/30 + 19/20/21）

| case | 路径 | 本轮动作 | 结果 |
|---|---|---|---|
| 28 (fp16 s=0) | tile SHIFT_ZERO=True | T2 原型 A/B 实测后撤销 | MW2 2.48 vs MW1 2.32us（**+6.9% 回退**，24 组对比全负）；交付保持 MW1；逐位 ✓ |
| 29 (fp32 s=0) | tile SHIFT_ZERO=True | T2 原型 + msprof 机制诊断 | MW2 +5.98%；msprof：scalar -0.126us / MTE +0.454us——多窗口方向证伪关闭 |
| 30 (bf16 s=0) | tile SHIFT_ZERO=True | T2 原型 A/B 实测后撤销 | MW2 +5.08%；同上，交付保持 MW1 |
| 19 (fp16 s=6) | row | T1 路由 trace + 强制 tile 计时 | 路由为设计内正确（primary UB 147456B>98304 与 secondary 6144>3072 双拒，任务单前提有误）；强制 tile 仅 -2.25%（噪声级）→ row 维持，未修改 |
| 20 (fp32 s=6) | row | T1 trace 覆盖（同组 fp32） | 294912B UB / 6144>3072 双拒，row 为唯一合法路径；未修改（任务单未授权） |
| 21 (bf16 s=6) | row | T1 trace 覆盖（目标 case 之一） | 147456B / 6144>3072 双拒；与 case19 同形状，强制 tile 历史证据 -2.0%（噪声）；row 维持，未修改 |

另：case14（第 4 差，tile）由 T3 完成管线分解与 TC=2048 反证（+7.9%），维持现状。

## 5. 融合算子方案

**当前融合方式**：维持 F5.fusion_boundary_routing 双 kernel 融合结构——cyclic shift + window partition 在**单个 kernel 内**完成（1R1W 纯数据重排）。host 按标量元数据路由：27 case → `_roll_window_partition_tile_kernel`（2D 每窗口 tile，SHIFT_ZERO True/False 两编译特化），3 case（19/20/21）→ `_roll_window_partition_kernel`（逐行两段拷贝）。本轮 T2 原型在 tile kernel 内尝试增加 MULTI_WIN 编译维度（每程序 2 窗），实测回退后撤销——**交付融合边界与 eval/iter8 完全一致**。

**数据流向**：窗口重排由 kernel 内地址映射直接实现，load（MTE2，GM→UB tile chunk/行段）与 store（MTE3，UB→GM）一一配对；片上在途数据 ≤ 一个 TU×TC chunk（fp32 32KB / 2B 16KB，预算 96KB）或一个行段 BLOCK（≤4096 elem）；无跨 kernel 工作区、无 GM/HBM 中间读写、无 host tensor。证据：self_test 11 shape 组 memcpy 审计 0 事件；每 case trace 中设备 kernel 仅两个交付 kernel 之一。T2 原型数据流唯一变化是同程序内第二个窗口的 tile 拷贝串行排在第一个之后（仍在同一 kernel、同一 1R1W 结构内），msprof MTE 忙时 0.450→0.904us 为其直接观测。

**本轮修改对融合的影响**：净改动为零——原型撤销后融合程度、kernel 数量、HBM 往返均不变。原型的负结果（跨核 MTE 并行 > 程序内摊销）封顶了"在不改变融合边界的前提下进一步压缩 cases 28-30"的一条路径。

**与 history 中 ❌/负结果方向的关系**：本轮主动规避 iter3 的 host 地址表方向（H2D 违反反作弊——本轮所有谓词均为 host Python 标量，memcpy 审计 0）；同时本轮自身产出两个新负结果（MULTI_WIN 摊销、TC=2048），已如实归档供 tech_lead 更新 insights，避免后续轮次重复押注。

Measured implementation selection (Jev probabilities are advisory; measured evidence takes priority).
Underperforming stagnation requests review, never mandatory fusion replacement.
