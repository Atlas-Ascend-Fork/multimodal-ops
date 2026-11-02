# design_rationale — fused_patch_concat_layernorm（develop/iter11，第 12 轮开发）

> 本轮任务类型 perf_pass_optimize（性能已达标后继续优化），输入为 eval/iter11 结果（30/30 精度通过，avg_speedup=1.890，最佳快照仍为 iter10 的 1.926）。Stage9 任务单三项：T1（P1，检查：PAIRED 路径 per-iteration DMA vs compute 分解）、T2（P2，修改实验：BLOCK_C=256/BLOCK_2C=512 for PAIRED C≤96 的 A/B）、T3（P2，检查：SKIP_CORR 编译二进制对照）。**结论先行：T1/T3 检查完成并产出修正性发现；T2 经 3 会话交错 A/B 一致退步，按验收线否决，交付代码与 develop/iter10 交付逐字节相同（sha256=2c278e28…）。**

## 一、按 task_id 的执行与验收对照

### T1（P1，inspect）— 完成，含一项对历史记录的修正

**验收线逐条对照**：

1. *Per-iteration DMA vs compute ratio documented for each of the 6 cases* — 完成。数据源为 eval/iter11 正式评测 profiler `kernel_details.csv`（150 个目标 kernel 行 = 30 case × 5 step，每 step 恰 1 个目标 kernel、Block Num=40、无其他 static kernel），逐 case 稳态中位数与迭代数交叉计算（脚本 `/tmp/opencode/fpcl_dev_iter11/t1_analysis.py`，输出 `self_test.log` SECTION 6）：

| case | dtype | C | BLOCK_2C | EXACT_2C | tokens | iters/prog | 稳态中位(us) | us/iter | us/GM事务 | useful KB/token | mask 浪费 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | fp16 | 96 | 256 | False | 784 | 20 | 32.72 | 1.636 | 0.273 | 1.75 | 25.0% |
| 3 | bf16 | 96 | 256 | False | 784 | 20 | 32.50 | 1.625 | 0.271 | 1.75 | 25.0% |
| 19 | fp16 | 80 | 256 | False | **690** | **18** | 31.57 | 1.754 | 0.292 | 1.62 | 37.5% |
| 21 | bf16 | 80 | 256 | False | **690** | **18** | 30.86 | 1.714 | 0.286 | 1.62 | 37.5% |
| 4 | fp16 | 192 | 512 | False | 784 | 20 | 32.65 | 1.633 | 0.272 | 3.50 | 25.0% |
| 10 | fp16 | 128 | 256 | True | 2304 | 58 | 42.77 | 0.737 | 0.123 | 2.00 | 0% |

   **修正（重要）**：历史记录与任务单均按 tokens=1380 给 case 19/21 计 35 iters/program、0.89us/iter。实际 tokens = B·(H/2)·(W/2) = 2×15×23 = **690**（1380 是 B·H·W=length），grid=40 下最慢 program 处理 **18** 个 token，per-iteration 成本实为 **1.71~1.75us**。修正后"case 19 比 case 10 只慢 0.15us/iter、说明 DMA 固定延迟主导"的旧推断不再成立；正确的图景是：**全部非 EXACT PAIRED case（C=80/96/192，BLOCK_2C=256/512）的 per-iter 成本聚在 1.62~1.75us，EXACT_2C（C=128，BLOCK_2C=256）为 0.74us**。

2. *EXACT_2C (case 10) vs non-EXACT (cases 1/19) per-iteration overhead quantified* — 完成。同宽度（BLOCK_2C=256）下 EXACT_2C 比 non-EXACT 低约 0.9us/iter（跨 case 差值）；同 case 单变量的干净测量来自 iter7 的 EXACT_2C A/B（case 10/11/12 −8.8%~−11.4%，即 ~0.15us/iter）。两者的差由 C 相关的固有成本（store/cast 宽度、有效数据量）解释，不应全部记在 mask/where 头上。另一个量化发现：**fp16/bf16 相对 fp32 的 cast 代价（~0.44~0.47us/iter）只出现在非 EXACT 路径**（case 1 vs 2：1.636 vs 1.161；case 19 vs 20：1.754 vs 1.311），EXACT 路径上 cast 几乎免费（case 10 vs 11：0.737 vs 0.714；13 vs 14：0.977 vs 0.967；16 vs 17：1.300 vs 1.262）——提示 masked-load+where 路径阻碍了 cast 与 DMA 的融合/流水。

3. *Conclusion on whether BLOCK_C tuning can reduce per-iteration cost for C=80/96* — **不能**。机理证据：per-iter 成本对 BLOCK_2C 宽度不敏感——case 1（BLOCK_2C=256，2C=192）1.636us/iter ≈ case 4（BLOCK_2C=512，2C=384）1.633us/iter，而 case 7/8/9（BLOCK_2C=1024，2C=768）反而升至 1.65~2.11us/iter。per-token 固定开销（6 次 GM 事务 + token 解码 + 统计）不随宽度摊薄，加宽只增加 mask 浪费。带宽维度亦不构成瓶颈：40 program × ~1.7KB/1.64us ≈ 42GB/s 聚合，远低于 HBM/L2 带宽；0.27us/事务 × 6 事务 ≈ 全部 per-iter 成本，属**延迟主导**（事务内实际数据传输 <10%）。该结论随后被 T2 实验直接证实（见下）。

### T2（P2，modify 实验）— 完成，**否决回退，交付代码零改动**

**验收线逐条对照**：

1. *A/B comparison: cases 1/3 kernel time vs iter11 baseline (32.7/32.6us target)* — 未达成。变体仅按任务单授权改 host 侧 constexpr：`tokens>64 and channels<=96` 时 BLOCK_C=256 / BLOCK_2C=512（C=96：mask 浪费 25%→62.5%；门控同时波及 C=80 的 19/20/21：37.5%→68.75%）。3 会话 × 20 rounds × {base, t2, dup} 交错（s2 反序、s3 轮换，dup 为 base 逐字节副本做噪声包络，cache_clean + 每 step 单变体调用，profiler kernel Duration）：

| case | 门控 | t2−base 逐会话 | 中位数 | dup 包络 |
|---|---|---|---|---|
| 1 | C=96 | −9.7% / +5.5% / +7.0% | **+5.5%** | −7.0%~+19.4% |
| 2 | C=96 (fp32) | +11.6% / +19.2% / +9.6% | **+11.6%** | −6.8%~+8.5% |
| 3 | C=96 | +10.1% / +6.5% / +24.4% | **+10.1%** | −2.1%~+2.5% |
| 19 | C=80 | +6.3% / +11.7% / −2.7% | **+6.3%** | −10.3%~+1.7% |
| 20 | C=80 (fp32) | +13.4% / +12.4% / +8.0% | **+12.4%** | −5.5%~+5.5% |
| 21 | C=80 | +10.4% / +33.5% / +11.5% | **+11.5%** | +2.3%~+17.4% |
| 4 | 对照（不受门控） | −4.1% / −0.1% / +15.8% | −0.1% | −0.3%~+14.1% |
| 10 | 对照（不受门控） | −1.0% / +1.7% / −1.3% | −1.0% | −1.5%~+1.0% |

   s1 中 case 1 的 −9.7% 在 s2/s3 双双反转，且 s1 的 dup 包络宽达 ±13% 以上，判为噪声会话；其余 5 个受影响 case 三会话方向一致退步。**任务单预设的"20→10 次迭代减半"前提不成立**：kernel 每 iteration 仍只处理 1 个 token，BLOCK 宽度不改变迭代数、也不改变 per-token 的 6 次 GM 事务数——这与 T1 的"per-iter 对宽度不敏感"互为印证。

2. *No precision regression on cases 1/3* — A/B 会话内 sentinel 通过（BLOCK 宽度不改变数值：masked 零填充 lane 参与的求和/where 结果与基线逐位一致，本轮 TC6 亦覆盖）。
3. *No regression >3% on other PAIRED cases* — 被违反（19/20/21 +6.3%~+12.4%）。**两项均不满足 → 按验收线否决**。变体仅存在于 /tmp/opencode/fpcl_dev_iter11/variants/t2.py，交付文件全程未动（sha256 前后一致）。

### T3（P2，inspect）— 完成

**验收线逐条对照**：

1. *Binary comparison result documented* — 完成（脚本 `/tmp/opencode/fpcl_dev_iter11/t3_npubin.py`，`self_test.log` SECTION 5）：

| 特化 | iter10 快照 npubin | 交付 npubin | 结论 |
|---|---|---|---|
| PAIRED fp16（对照） | 10744 B | 10744 B，**逐字节相等** | PAIRED 路径零代码差异 |
| single fp32 | 7736 B | 7736 B，等长、TTIR op 数相等（231 vs 231），858 字节散布漂移（163 段） | 多出的 constexpr 实参只造成编译布局/调度漂移；数值路径相同（TC6 23/26/29 逐位相等佐证） |
| single fp16 | 9040 B | 8384 B（**−656 B，−7.3%**），TTIR 248→232 ops | **SKIP_CORR 是真实机器码削减** |
| single bf16 | 9040 B | 8384 B，同上 | 同上 |

2. *If different: instruction count delta quantified* — TTIR op 差恰为 corr 链的移除：`tt.reduce −2、arith.subf −5、arith.addf −4、arith.divf −1、arith.maxnumf −1、arith.mulf −1、tt.splat −1、arith.constant −1`（合计 −16），与设计意图逐项对应（1 次 BLOCK_C 宽归约 + 2 次 BLOCK_C 宽向量减法 + 标量除法 + 标量乘减 + tl.maximum）。
3. *Recommendation* — 二进制证据表明 iter11 正式评测中 single 路径 ±12~18% 的波动是 1~4us 短 kernel 的评测噪声，而自测 A/B 的 −8%~−12% 方向是真实的。**建议后续对 single 路径 case 采用多次评测取中位/均值的方式确认收益，而不是再改代码**；insight 12/13 的"待验证(性能)"可升级为"已验证（编译产物级）"。

## 二、改动前后的 Tiling / 数据流 / 多核方案对比

本轮交付代码零改动，此处给出"交付版 vs 被否决的 T2 变体"的对比，作为否决依据的存档：

| 维度 | 交付版（= iter10 交付） | T2 变体（已否决） |
|---|---|---|
| Tiling | PAIRED：每 iteration 处理 1 个 token，BLOCK_2C=next_pow2(2C)（C=96→256 / C=80→256）；single：BLOCK_C=next_pow2(C) | 仅对 tokens>64 且 C≤96 把 BLOCK_2C 提到 512（C=96 浪费 62.5%、C=80 浪费 68.75%） |
| 迭代数 | tokens/40（case 1/3：20，case 19/21：18，case 10：58） | **不变**（任务单预设的"迭代减半"不存在） |
| per-token GM 事务 | 2 load + 4 half-masked store = 6 | **不变**（每事务搬运字节更多、有效占比更低） |
| 多核 | grid=min(tokens,40)，40 Vector Core 一波 | 不变 |
| 数据流 | Slice×4 strided gather → 片上配对 concat → fp32 统计 → 4 半掩码 store，无中间 HBM 落盘 | 同左 |
| 实测 | case 1：32.72us（正式）/ A/B base 中位 32.42us | case 1：A/B 中位 34.68us（+5.5%）；19/20/21 +6.3%~+12.4% |

否决机理：per-iteration 成本 = 6×GM 事务固定延迟（~0.27us/事务）+ 非 EXACT 路径的 mask/where/cast 暴露成本，两项均不随 BLOCK_2C 加宽而下降；加宽只按比例增加零填充 lane 的向量管线浪费（case 19 的 37.5% 浪费已是 6 个最慢 case 中最高，其 1.75us/iter 亦最高，方向一致）。

## 三、预期效果（哪些 case 会变好、为什么）

- **无任何 case 因本轮交付变化**——交付代码逐字节不变，正式评测预期复现 eval/iter11 分布（30/30 精度；avg_speedup 1.890 ± 运行间噪声；最佳快照保持 iter10 1.926）。PAIRED case 1-21 与 fp32 single 的 npubin 已被证明与快照等价/逐字节相等，波动只来自测量。
- 本轮"收益"是知识与方向性的：
  1. **关闭 BLOCK_C 调优方向**（数据支撑：宽度不敏感 + T2 一致退步），避免后续轮次重复试错；
  2. **修正 case 19/21 的迭代数记录**（690/18 iters，非 1380/35），使最慢 case 的结构性下限论证建立在正确账目上；
  3. **SKIP_CORR 编译产物级验证**，为后续 short-kernel case 的评测方法（多次取中位）提供依据。
- 6 个最慢 case 的剩余空间判断：非 EXACT 路径的 cast 暴露（~0.45us/iter）与 EXACT 路径的显著差异是唯一尚未解释干净的现象（推测：masked-load 阻碍 cast 与 DMA 融合），但它由后端编译行为决定，host 侧 constexpr 无法触及；EXACT_2C 已覆盖 C 为 2 幂的 case（10-18），C=80/96/192 无法享受。

## 四、针对最慢 6 case 的逐个说明

- **case 1（C=96 fp16，speedup 1.065）**：T1 主对象之一——per-iter 1.636us（20 iters，0.273us/事务，有效 1.75KB/token，浪费 25%）；T2 主对象——BLOCK_2C=512 实测中位 +5.5% 退步，回退。本轮代码零改动。
- **case 3（C=96 bf16）**：T2 主对象——+10.1%（三会话一致退步），回退；T1 数字与 case 1 一致（1.625us/iter），bf16 与 fp16 cast 等价。
- **case 2（C=96 fp32）**：任务单 1/3 之外但同被门控波及，主动纳入 A/B——+11.6%（三会话一致），进一步证伪 T2；fp32 per-iter 1.161us/iter 提供 cast 代价对照。
- **case 19（C=80 fp16，speedup 1.072）**：T1 修正对象——tokens=690、18 iters、1.754us/iter（此前记录 0.89us/iter 系迭代数错账）；37.5% 浪费 + 最高 per-iter，确认其结构性下限地位；T2 门控波及对象——+6.3% 退步。
- **case 21（C=80 bf16）**：同 case 19（1.714us/iter）；T2 门控波及对象——+11.5% 退步。
- **case 4（C=192 fp16）**：T1 对照——BLOCK_2C=512 与 case 1 的 256 同为 1.63us/iter，是"宽度不敏感"结论的关键证据；T2 不受门控影响，A/B 中位 −0.1%（噪声内），验证 harness 无系统偏差。
- **case 10（C=128 EXACT fp16）**：T1 对照——0.737us/iter 为全部 PAIRED case 最低，EXACT_2C 有效性的量化锚点；T2 不受门控影响（−1.0% 噪声内）。已处不可约串行循环下限，本轮未做修改。

## 融合算子方案

- **当前融合方式**（F2 垂直融合，十一轮评测验证，本轮保持不变）：单 kernel `_fused_patch_concat_layernorm_kernel` 内完成全部计算——x 的 4 个 patch 段以 2 条 2C 宽 strided affine load（PAIRED 路径，利用同行相邻列内存连续）或 4 条 C 宽 masked load（single 路径）取入 UB，在片上完成 concat 通道重排（配对 load + 4 条半掩码 affine store 的结构性转置修正）、fp32 统计（mean/var/rstd，fp16/bf16 single 路径按 dtype 条件跳过 corr 残差）、仿射输出。**没有任何计算步骤拆分为多个 kernel，无中间张量落盘 HBM。**
- **数据流向**：x/weight/bias GM →（DMA，PAIRED 每 token 2 事务 load + 4 事务 store）→ UB 内 fp32 向量寄存 → 统计与 epilogue 全在片上 → y GM。中间量（seg_at/seg_bb、diff、sq_sum）均为 program 内 UB/寄存器存活，无跨 kernel 工作区。TC3 profiler 证据：每次调用恰 1 个设备 kernel（AI_VECTOR_CORE，static），无辅助 kernel。
- **本轮修改对融合的影响**：零（交付逐字节不变）。被否决的 T2 也只是 constexpr 加宽，不改变融合边界；其失败恰好说明在"已全融合、GM 事务延迟主导"的结构下，片上 tiling 参数已无杠杆——融合程度本身（0 次 HBM 中间读写）不是瓶颈，事务延迟才是。
- **history 中 ❌ 方向的规避**：融合策略史上的失败项本轮全部未触碰——2D tile（iter3 灾难，insight 6）未使用；store 合并（iter8 T1 无收益）未重试；grid=20（iter11 T3 否决）未重试；single 分支无条件去 corr（iter9 精度失败）保持 dtype 条件守卫。本轮唯一修改尝试（T2）按验收线否决回退，未引入新方向。
- **与初始候选库的关系**：F2（rank 1，probability 0.79）继续作为唯一实现形态；F5（0.58）的按 shape 路由在现有 constexpr 族（PAIRED_SEG/MULTI_TOKEN/EXACT_2C/SKIP_CORR）中已以特化形式存在；F8（0.44）的计算/访存流水对 3~43us 的 kernel 无收益空间（T1 数据：事务延迟主导、带宽占用 <25%）。概率仅作先验，本轮结论完全由实测支撑。

## 五、反作弊与合规

- 核心计算全部在 `@triton.jit` kernel 内（forward 仅元数据校验 + torch.empty + kernel[grid]）；TC3 证明真实 NPU 执行（AI_VECTOR_CORE，每调用恰 1 kernel）。
- 临时脚本/变体全部位于 /tmp/opencode/fpcl_dev_iter11；task/ 与全部只读文件零改动；交付文件 sha256 全程 2c278e28…。
- 性能实验数字均为 profiler 实测（3 会话交错 + dup 噪声包络 + 反序对照），不冒充正式评测结果；正式口径以 cann-bench 下轮 eval 为准。
