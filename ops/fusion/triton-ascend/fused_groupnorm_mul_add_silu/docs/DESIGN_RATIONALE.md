# 融合方案选择决策依据 — FusedGroupnormMulAddSilu（develop/iter3 性能优化轮）

目标芯片：Ascend910B3（dav-c220，20 AI Core / 40 Vector，UB 192 KB，L2 192 MB）。
后端：triton-ascend 3.2.1（triton 3.2.0），Triton block program (SPMD)。
本轮任务：perf_optimize（iter3 基线 avg_speedup 2.849，perf_score 37.05/50，仅 case 14 未达标 speedup=0.972）。
本轮 Stage9 任务单：`knowledge/stage9/iter3/3342144a19ee4f3f9ffc30ce430275c4/decision.json`（T1 + T2，均为 P1，可改文件仅 `impl/cann_bench/fused_groupnorm_mul_add_silu.py`）。

## 1. 选了什么（最终交付状态）

**保持 F2 垂直融合（rank 1，Jev 0.74）为主融合族 + F5.schedule_routing（rank 3，Jev 0.67）三路 shape 分发 + F1（rank 2，Jev 0.70）受限变体 F1.compute_epilogue_split 仅用于 fp32 大分组；本轮唯一行为改动是 T2——SPLIT 阶梯上探 8**：

| 路由谓词（host，纯 shape/dtype 元数据） | kernel | grid | case |
|---|---|---|---|
| `next_pow2(CPG)×next_pow2(HW) ≤ 8192` | tile kernel | `(B×32,)` | 7–12、22–36（21 个，不变） |
| `fp32 且 group_elems ≥ 49152`，SPLIT=8（hw≥65536）/ 4 / 2 | stats + apply 两 kernel | `(B×32×SPLIT,)` | **14（SPLIT 4→8，grid 128→256）**、17、20（不变） |
| 其余（含全部 fp16/bf16 大分组） | 流式单 kernel | `(B×32,)` | 1–6、13、15、16、18、19、21（不变） |

- **T2 保留**：`split = 8 if hw >= 65536 else (4 if hw >= 32768 else 2)`。case 14 grid 256，每 program slab 98304→49152 元素，workspace 仍 ≤4KB。
- **T1 按指令原文实施后回退**：先按任务单逐字实施 `if group_elems >= _SPLIT_MIN_ELEMS:`（去掉 `is_fp32 and`），实测全部受影响 fp16/bf16 case 退步（见 §3），按 FIX_DIRECTIVE 门槛"若 case 13/15 在 split 路径退步……回退 T1 恢复 `is_fp32 and` 前缀。两处回退互相独立"执行单行回退。最终路由与 iter3 完全一致，仅 SPLIT 阶梯多一档。
- 初始 Jev 概率（F2=0.74 / F1=0.70 / F5=0.67）在 `fusion_library.json` candidates 中原样保留；attempts 新增 4 条记录 probability=null（不自行评分）。

## 2. 参考了哪些概率和已测证据

- **Jev 概率（仅初始方向参考）**：F1=0.70 的 graph_conditions（"Stages prefer different schedules... compute and epilogue need different tiling or parallelism"）正是 split 路径的适用条件——但本轮实测证明该条件在 fp16/bf16 上**不成立**（epilogue 是逐元素计算受限，不是带宽/并行度受限），因此 F1 的适用域被实测收窄到 fp32；F2/F5 维持不变。
- **已测证据（本轮新增，全部可复核）**：
  - `develop/iter3/ab_stream_vs_split.log`：同输入（seed 42 randn）同环境、2 次 profiled launch 取均值的严格 A/B；
  - `develop/iter3/self_test_kernel_evidence.log`：最终代码状态的 kernel_details（kernel 名、时长、block_num）；
  - `develop/iter3/self_test.log`：36/36 给定 case + 18/18 连续调用 + DD 回退 6/6（kernel_eval 生产比较器 compare_tensors 判定）。
- **iter3 正式口径（cann-bench）**：case 14 split4 = 157.84us（stats 88.82 + apply 69.02，block_num=128）；case 13/15 stream = 150.9/159.7us；case 16/18/19/21 = 40.82/44.16/51.64/58.0us；case 17/20 = 46.88/47.46us。

## 3. 针对哪些 case、实际改了什么、为什么

### T1（dtype 无关 split 路由）——按指令实施、实测退步、按门槛回退

- **指令方法**：`if is_fp32 and group_elems >= _SPLIT_MIN_ELEMS:` → `if group_elems >= _SPLIT_MIN_ELEMS:`，预期 case 13/15（fp16/bf16，[1,192,256,256]）进入 split、grid 32→128、目标 ≤95/≤85us。
- **实施**：逐字实施。同时发现任务单的一处口径偏差：任务单与 FIX_DIRECTIVE 认为 case 16/18/19/21 的 group_elems<49152（"24576"），实际 [1,192,128,128] 的 group_elems=**196608**、[2,384,64,64] 的 group_elems=**49152**（=12×4096，恰好等于阈值），逐字改动会使 6 个 fp16/bf16 case（13/15/16/18/19/21）全部进入 split 路径。
- **实测（同输入 A/B，stream vs split2；13/15 另有 split8 数据）**：

| case | stream（A/B / iter3 正式） | split 后（A/B） | 变化 |
|---|---|---|---|
| 13 fp16 | 141.79 / 150.9us | 175.46us（split2）；158.96us（split8，stats 61.92+apply 97.04） | **+23.7% / +12%** |
| 15 bf16 | 149.19 / 159.7us | 185.18us（split2）；174.14us（split8） | **+24.1% / +9%** |
| 16 fp16 | 38.31 / 40.82us | 50.91us（20.33+30.58） | **+32.9%** |
| 18 bf16 | 41.07 / 44.16us | 55.33us（20.70+34.63） | **+34.7%** |
| 19 fp16 | 45.93 / 51.64us | 53.75us（19.15+34.60） | **+17.0%** |
| 21 bf16 | 52.70 / 58.0us | 65.25us（19.09+46.16） | **+23.8%** |

- **根因**：fp16/bf16 逐元素链（多次 fp16↔fp32 转换，bf16 还有整数 RNE）是**计算受限**，加大 grid 不改变每元素计算量；split 只净增一次完整 stats 相位（19–69us），而流式 kernel 把 stats 读与 epilogue 融合在一遍里（第二次读命中 L2）。fp32 的 stats kernel 是纯带宽相位（838 GB/s 可达），拆分才能解耦两相位——F1 变体的适用条件在 fp16/bf16 上实测不成立。
- **精度不是回退原因**：split 模式下 36/36 全过，case 13 mare 7.463e-03 与同 seed 流式值逐位一致（fp16/bf16 epilogue 链未动）。
- **回退执行**：恢复 `is_fp32 and` 前缀（单行），路由注释改为记录实测结论（"fp16/bf16 epilogue chains are per-element compute-bound; split adds the stats pass without any parallelism win, +17..35% slower"），防止后续轮次重蹈。回退后终态 kernel 证据：13/15/16/18/19/21 = 141.98/149.54/38.40/41.20/45.92/53.28us（randn），全部 ≤ iter3 正式值，确认无残留影响。
- 此处置与 Stage7 case 19 追踪中的预判一致："If T1 dtype-agnostic routing is applied and case 19's group_elems turns out to be >=49152, verify precision and performance before/after"——已按此验证，结论为退步回退。

### T2（SPLIT=8 档）——保留

- **指令方法**：`split = 4 if hw >= 32768 else 2` → `split = 8 if hw >= 65536 else (4 if hw >= 32768 else 2)`；case 14（hw=65536）grid 128→256；case 17（hw=16384）/20（hw=4096）不受影响。
- **实测**：
  - 同输入 A/B：case 14 split4=134.08us（stats 65.19 + apply 68.89）→ split8=**126.66us**（stats 60.31 + apply 66.35），**-5.5%**；stats 有效带宽 50.3MB/60.31us ≈ **834 GB/s**（SPLIT=4 正式口径 566 GB/s，接近小 shape 峰值 >750 GB/s，流水线填充补满）。
  - 终态 kernel 证据：stats 60.58 + apply 64.80 = **125.38us**，两 kernel block_num=**256**（SPLIT=8 生效实证）。
  - 整除性核验：393216%8=0、chunk 49152%4096=0；65536%8=0、slice 8192%4096=0，EVEN 快路径成立。
  - 精度：case 14 mare 1.535e-05 << 阈值 1.22e-4；36/36 + 18/18 全过。
- **对 case 17/20 的影响**：hw<65536，仍走 SPLIT=2，终态实测 42.64/42.58us（randn）与 iter2/iter3 自测一致，无回归。

## 4. 预期收益 vs 已测收益（明确区分）

| case | iter3 正式 us | 本轮自测 us（randn） | 对象 | 验收目标 | 状态 |
|---|---|---|---|---|---|
| 14（split8：60.58+64.80） | 157.84 | **125.38** | T2 | ≤149us | ✅（自测口径） |
| 17（split2：20.70+21.94） | 46.88 | 42.64 | 对照 | 不受影响 | ✅ |
| 20（split2：18.82+23.76） | 47.46 | 42.58 | 对照 | 不受影响 | ✅ |
| 13（stream 回归） | 150.9 | 141.98 | T1 回退 | 不劣于正式值 | ✅ |
| 15（stream 回归） | 159.7 | 149.54 | T1 回退 | 不劣于正式值 | ✅ |
| 16/18/19/21（stream） | 40.82/44.16/51.64/58.0 | 38.40/41.20/45.92/53.28 | T1 回退 | 不劣于正式值 | ✅ |
| 8/23/11/12（tile） | 19.34/19.94/22.46/22.26 | 17.92/18.94/23.48/22.00 | 回归对照 | 不受影响 | ✅ |
| 1/2/4/5/6（stream） | 15.5/13.26/30.42/27.44/35.38 | 13.36/11.74/26.52/23.38/31.26 | 回归对照 | 不受影响 | ✅ |

- **已实测**：上表自测列均为 torch_npu.profiler kernel_details.csv 真实值；A/B 同输入对比（split8 vs split4 = -5.5%）是本轮最硬的证据。
- **待正式评测确认**：case 14 的正式时间与 speedup（自测 split4 曾比正式低 17%，若同比例放大，split8 ≈ 146us，恰在基线 153.42 之下、目标 ≤149 边缘；**正式涨跌以 cann-bench 同口径报告为准，本文件不预写**）。T1 目标（case 13 ≤95us / 15 ≤85us）经实测否决——split 路径无法对 fp16/bf16 达成该目标，回退后两 case 维持 iter3 正式值（150.9/159.7us）。
- **预期未兑现部分**：T1 全部预期（13/15 提速）未兑现且被实测否决；已按指令门槛回退并归档证据。

## 5. 对融合程度的影响

- **融合程度不降**：最终代码的融合拓扑与 iter3 完全一致——三条路径的逐元素链仍是"归一化→调制→SiLU 单趟、无逐元素 HBM 中间物化"；split 路径唯一跨 kernel 物化仍是 ≤4KB 的 fp32 标量对 workspace。T2 只改 grid 规模（128→256），不增减 kernel 数、不改数据流。
- **HBM 中间读写**：tile/stream 路径为零；split 路径仅 ≤4KB workspace；apply 对 x 的重读命中 L2（50.3MB < 192MB）。总流量 2R+1W 不变。
- **T1 试错对融合的影响**：dtype 无关路由曾把 6 个 fp16/bf16 case 移入 split（增加一次 kernel 间边界），实测证明该边界对计算受限链是纯开销——回退即恢复零边界融合。该结论已写入代码注释与 attempts，防止重复试错。
- **history ❌ 方向规避**：shifted 单遍统计、tl.atomic_add/DSM/共享内存/num_warps/持久 grid 等历史否决方向全部未触碰；本轮新否决"fp16/bf16 split"仅涉路由谓词，不涉融合拓扑。
- **概率使用纪律**：初始库 3 个候选 method 定义与 Jev 概率原样保留；本轮 attempts 新增 4 条 probability=null；所有选择以实测证据为准（"Measured implementation selection; Jev probabilities are advisory"）。

## 6. 研究线索（本轮只测量、不落地，供 Stage7/8 下一轮评估）

case 14 同输入 A/B 中，**合并 epilogue 后的流式 kernel（直接启动，grid=32）仅 104.92us**，低于 split4（134.08）与 split8（126.66）：合并系数把 epilogue 降到 ~5 flops/元素后，每 program 1.57MB slab 的二次读稳定命中 L2，"stats+epilogue 融合单 kernel"可能已不再需要拆分。未采纳原因：(a) 将 case 14 移出 split 超出本轮任务单授权（T2 明确以 split 路径为前提）；(b) 仅自测口径证据，与正式口径存在已知差距（iter2 split4 自测 134.86 vs 正式 157.84，+17%，Stage7 归因于 DataGenerator 输入分布与缓存行为），存在噪声过拟合风险。建议下一轮以 cann-bench 同口径验证后再议路由。

## 7. 回退开关

1. `_SPLIT_MIN_ELEMS` 调大（或恢复 `split = 4 if hw >= 32768 else 2` 单行）：case 14 在 SPLIT=8 正式退步时一步回到 iter3 状态（本轮自测无退步，未触发）。
2. `_DD_EPILOGUE=False → True`（模块常量，一行）：fp32 epilogue 整体回退 double-double 链（TC6 实测 6/6 通过、case 26 mare 精确回到 7.711e-04，真回退）。
3. T1 回退本身即本轮交付的一部分（`is_fp32 and` 前缀已恢复，注释留档实测依据）。

Measured implementation selection (Jev probabilities are advisory; measured evidence takes priority).
Underperforming stagnation requests review, never mandatory fusion replacement.
