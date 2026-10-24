# 设计思路 — fused_mul_add_layernorm_mul_add（develop/iter3，对应正式评测 eval/iter4）

- 问题类型：perf_optimize（第 3 轮开发退修，账本 iter3 评测后 direction 的落地轮）
- 唯一修改文件：`impl/cann_bench/fused_mul_add_layernorm_mul_add.py`
  - 改前 sha256：`02e9b5f27f6f15b756693583acc0fc12f2dfe83966f0701321ab47c6486ab3b6`（= iter3 正式评测快照）
  - 改后 sha256：`91c98a9e438d7b226203bcbab00e9a60f5db7bfd02ee8b3db36f8f8404b42f9c`
- 依据：Stage9 v2 任务单（`knowledge/stage9/iter3/f24d14eacda043f2aeda05bb2f278b6b/decision.json` T1/T2）、Stage7 瓶颈分析（`profile/iter3/bottleneck_analysis.md`）、Stage8 指令（`search/iter3/FIX_DIRECTIVE.md`）
- ⚠️ **本轮未做正式性能评测**；所有性能数字中，探针/自测的 kernel 时长为 torch_npu profiler op_summary 实测（Task Duration），speedup 相关结论一律标注"待正式评测"。

## 一、按 task_id 执行情况

### T1（P1，修改）— 多行 wide kernel + 二级路由：已完成

**change 1（新增 kernel，位置：wide kernel 之后）**：新增 `@triton.jit _fused_mul_add_layernorm_mul_add_wide_rows_kernel`，签名在 wide kernel 基础上增加运行时形参 `P`。逐条对照任务单方法落实：

- `pid = tl.program_id(0)`；`b = pid // P`（一个 program 只服务一个 batch 元素）；`p_local = pid % P`；
- **参数一次加载、行间复用**：`g32/s32/h32` 在行循环前各加载一次，`t = (1.0 + s32).to(dt)` 一并外提——合法仅因同 program 全部行共享同一 `b`（`b = pid // P` 保证），与 FIX_DIRECTIVE 第 3 条一致；
- **行循环**：`for k in tl.range(p_local, N, P)`（默认 tl.range，见 T2），行基址 `base = (b * N + k) * C`；
- **行体与 wide kernel 逐语句相同**（AST 比对确认）：六处 `.to(dt)` 舍入点位置一致（`ga`、`r`、`nn32`、`t`、`y1`、`y`），两遍 fp32 统计（sum → 中心化二阶矩），mask 逻辑一致，residual tile 在其 store 与 LayerNorm 消费之间驻留片上（零 L2 重读）。

**change 2（host 二级路由，位置：`elif channels <= 2048` 分支）**：纯元数据谓词（只用 channels/batch/seq）：

```python
total_rows = batch * seq
if total_rows > 512:
    P = min(seq, max(1, 40 // batch))     # 40 = device_info.json vector_core_num
    grid = (batch * P,)
    _fused_mul_add_layernorm_mul_add_wide_rows_kernel[grid](..., channels, seq, P, eps, BLOCK_C=channels)
else:
    grid = (total_rows,)
    _fused_mul_add_layernorm_mul_add_wide_kernel[grid](...)   # 原调用逐字节不动（case 19-21 走此路径）
```

谓词覆盖完整性：`channels<=1024` one-shot（含非整除尾块 mask）/ `1024<channels<=2048` 按 `total_rows>512` 二分 / `channels>2048` blocked 兜底——任意合法输入恰落一条路径；`P = min(seq, max(1, 40//batch)) ≥ 1`（batch>40 时退化为每 program 处理本 batch 全部 N 行，仍正确）。无 case 编号、无 benchmark 身份判断。

**验收要点对照**：

| 验收项 | 状态 | 证据 |
|---|---|---|
| 6 个目标 case（10-15）正式评测 speedup ≥ 1.0 | **待正式评测** | 探针 kernel 中位时长 23.0–26.7us（B=1,N=1024,C=1152 三 dtype），对应基线 65.16–79.49us（eval/iter3），结构上预期 ≈2.4–3.0x；正式验收以 cann-bench 评测为准 |
| 36 case 精度自测全过 | 达成 | 36/36 PASS，residual 全部位级一致，y 全部在 2T 门限内（self_test.log TC4；严格门限仅 case 10/15 各 1 个双舍入边界元素，fp64-margin impl==golden，见自测报告说明） |
| case 19-21 不退步、仍在原 wide 路径 | 达成（待正式评测确认数值） | TC4b profiler 实证 `..._wide_kernel` 仍被调用（median 10.90us，与 iter3 case 19 的 10.84us 同量级）；grid_check.log 实证其 Block Num=128（原 grid=(B*N,) 未变）；wide kernel 函数体与调用参数逐字节未动（t1_structural.log） |
| 27 个 C≤1024 case 不退步（±2% 噪声内） | 达成（待正式评测确认数值） | one-shot kernel 与 `channels<=1024` 分支逐字节未动（AST source-segment 比对，t1_structural.log）；TC4 27 case 全过、fp64-margin 与 golden 一致 |
| 新路径连续调用（10 次、同 shape 换数据）通过 | 达成 | TC5 共 14 次连续调用（覆盖全部四条路由），每次换全部输入/权重/偏置并逐次对照 golden，全 PASS；输入未被修改、输出独立存储（self_test.log TC5） |

### T2（P2，修改→探针后回退）— num_stages=2 探针：已执行、按验收标准不采纳

按任务单以独立探针执行（`probe_ns2.log`，fp32 case-14 形态 + fp16 case-13 形态，`num_stages=2` 与默认 `tl.range` 各 3 轮交替实测、每轮 30 次 kernel 取中位，双缓冲约 72KB < 192KB UB，编译无溢出报错）：

| 形态 | 默认 tl.range（3 轮中位） | num_stages=2（3 轮中位） | 结论 |
|---|---|---|---|
| fp32 B=1,N=1024,C=1152（case 14） | 23.03 / 22.91 / 23.07 us | 23.04 / 23.04 / 23.16 us | **无一轮更快** |
| fp16 B=1,N=1024,C=1152（case 13） | 26.41 / 26.34 / 26.32 us | 26.25 / 26.19 / 26.28 us | 快 ~0.1us（边缘） |

验收标准要求"两个目标点均更快且精度通过才保留"：case 14 形态无收益 → **按 acceptance_checks 回退为默认 `tl.range` 并记录**（impl 现状即默认版）。fp16 形态的 ~0.1us 差异在运行间噪声量级，不足以单独采纳。未使用 `flatten`/`disable_licm`/`disallow_acc_multi_buffer`/`warp_specialize`（FIX_DIRECTIVE 明令禁用项）。

### 前置门槛（FIX_DIRECTIVE 第 0 条）：已执行、通过

实现前先以独立探针脚本（`probe_gate.py`，不入 impl/）验证：合成输入 B=1,N=1024,C=1152 × fp32/fp16/bf16：

1. **编译通过**：非 2 的幂 BLOCK_C=1152 + 行循环在 triton-ascend 3.2.1 正常编译执行（JIT 缓存实证）；
2. **精度**：residual 与 golden 位级一致、`wide_rows` 输出与 `wide` kernel **位级相同**（y 亦 bit-exact）、严格门限失败元素 0（两种子 × 三 dtype）；
3. **kernel 时长**（torch_npu profiler op_summary Task Duration，各 50 次 × 2 轮交替取中位）：

| dtype | wide（grid=1024） | wide_rows（grid=40） | 降幅 |
|---|---|---|---|
| fp32 | 77.29 / 77.02 us | 23.52 / 23.16 us | −69.6% / −69.9% |
| fp16 | 77.92 / 77.87 us | 26.20 / 26.24 us | −66.4% / −66.3% |
| bf16 | 76.99 / 77.06 us | 26.74 / 26.69 us | −65.3% / −65.4% |

门槛全过（rows ≤ wide 且编译/精度通过）→ 执行 T1。

## 二、改动前后 Tiling / 数据流 / 多核方案对比

### 改动前（iter3 评测版，wide kernel）

- **Tiling**：BLOCK_C=1152（精确行宽，单趟），grid=(B*N,)，**一行一 program**。
- **多核**：1024 个 program 派发到 40 个 Vector Core，每核串行 ~26 行；每行都重新付出：program 启动 + 7 次 DMA 启动（x/attn_out/gate/scale/shift 读 + residual/y 写）。
- **参数访问**：gate/scale/shift（3×1152 向量）被每个 program 独立读取 → 同一 batch 的参数被重复读取 N 次（1024 行时全卡共读 1024×3 次），L2 命中不稳定造成 65–90us 方差（Stage7 §一）。
- **数据流**（单 kernel 内，零 GM 中间转存）：x/attn_out/gate →(UB)→ residual 计算 → store residual → 同一 tile 驻 UB → fp32 均值/方差 → 归一化 → scale/shift epilogue → store y。

### 改动后（本轮，B*N>512 的 1024<C≤2048 组）

- **Tiling**：行宽不变（BLOCK_C=channels=1152，仍单趟无子块循环）；行维组织从"一行一 program"改为"**一 program 服务同一 batch 的 P 步进行序列**"（P=min(seq, 40//batch)）。
- **多核**：grid=(batch*P,)=40 个 program，恰好贴合 40 个 Vector Core（case 10-12：B=4,P=10,grid=40；case 13-15：B=1,P=40,grid=40），每 program 串行 ~26 行。
- **参数访问**：3 个参数向量在循环前加载一次、跨 ~26 行复用（每 program 3 次 DMA 替代 26×3 次）；全卡参数读从 1024×3 次降为 40×3 次——同时缓解 Stage7 指出的 L2 状态不稳定（1024 个并发读者 → 40 个）。
- **DMA 计数**（每 program）：改前每行 7 次 × 26 行/核 = 182 次；改后 3（参数）+ 26×4（x/attn_out 读 + residual/y 写）≈ 107 次/核，且消除了 per-program 启动开销的 26 次重复。
- **数据流**：与改前完全同构（仍单 kernel、零 GM/HBM 中间转存、residual tile 生产者-消费者双角色驻 UB）——**融合度不变，本轮只改并行/访存组织**。
- **数值链**：逐语句不变（探针显示与 wide kernel 输出位级相同），精度零漂移。
- **不引入已否决模式**：无多 pass（iter2 ❌）、无子块循环（iter2 ❌）、无整行 L2 重读（iter2 ❌）、非 persistent grid-stride 全量遍历（iter2 ❌ 的 ~700 次循环/program）；本轮行循环每 program 仅 ~26 次且各 program 行集互斥、整卡每行恰处理一次。

### 未改动路径

one-shot kernel（C≤1024，27 case）、wide kernel（B*N≤512 的 C=1152 组，case 19-21）、blocked kernel（C>2048 兜底）三者函数体与调用参数逐字节未动（`develop/iter3/t1_structural.log` AST source-segment 比对）。

## 三、最慢 case 逐个说明（worst_6 = case 10–15）

所有 6 个 case 本轮均由**同一处路由变更**覆盖（B*N=1024>512 → wide_rows kernel，grid=(40,)，参数复用），均已通过精度自测；逐 case 的 iter3 基线与探针预期：

| case | dtype | iter3 kernel | iter3 speedup | 基线 | 本轮修改 | 探针 kernel 中位（同 dtype） | 预期 |
|---|---|---|---|---|---|---|---|
| 14 | fp32 | 89.08us | 0.775 | 69.08us | 路由→wide_rows | 23.2–23.5us | ≈69.08/23.2 ≈ 2.9x（待正式评测） |
| 10 | fp16 | 77.30us | 0.892 | 68.97us | 路由→wide_rows | 26.2us | ≈2.6x（待正式评测） |
| 11 | fp32 | 78.42us | 0.929 | 72.86us | 路由→wide_rows | 23.2–23.5us | ≈3.1x（待正式评测） |
| 13 | fp16 | 68.24us | 0.955 | 65.16us | 路由→wide_rows | 26.2us | ≈2.5x（待正式评测） |
| 15 | bf16 | 74.82us | 0.982 | 73.44us | 路由→wide_rows | 26.7us | ≈2.7x（待正式评测） |
| 12 | bf16 | 79.36us | 1.002 | 79.49us | 路由→wide_rows | 26.7us | ≈3.0x（待正式评测） |

说明：上表"探针 kernel 中位"是 B=1,N=1024,C=1152 合成输入的实测（B=4,N=256 的自测路由实证 median 25.94us 同量级）；case 10-12 的 grid 亦为 40（P=10×4 batch），行循环形态与 B=1 情形一致，Stage7 已证实 B 维不影响 per-row 性能（行内连续、无 B 间依赖）。预期 speedup 由"基线/探针中位"估算，**仅供方向判断；≥1.0 的正式验收以 cann-bench 评测为准**。

## 四、预期效果与风险

- **预期收益**：6 个目标 case 的 kernel 时长从 68–89us 降至 ~23–27us（−66% ~ −70%，探针实测），全部越过 1.0 并留出 >2x 余量（待正式评测）；31 个已达标 case 走未改动路径，预期 ±2% 噪声内持平（待正式评测）。
- **风险与回退**：wide kernel 原路径保留在代码中（回退成本为零）：若正式评测显示 wide_rows 反而劣化，可将 `total_rows>512` 分支改回原 wide 调用，一行谓词即回滚。
- **精度风险**：无——输出与被评测的 wide kernel 位级相同（探针实证），且 36 case 自测、14 次连续换参调用、eps 敏感性、别名安全全部通过。

## 五、融合算子方案

### 当前融合方式

- **单 kernel 全融合（F2 垂直融合 + F4 多消费者）**：全部四条路由（one-shot / wide / wide_rows / blocked）都在一个 `@triton.jit` kernel 内完成五步链：`ga = gate*attn_out`（逐元素乘）→ `residual = x + ga` → residual store → fp32 两遍统计（mean、中心化方差）→ 归一化 → `(1+scale)`/`shift` epilogue → y store。无任何步骤拆成第二个 kernel。
- **哪些分成了多个 kernel**：无。每条路由每次调用仅一个 device kernel（TC4b profiler 实证：每次调用仅捕获一个 `_fused_*` kernel 事件）。

### 数据流向

- **同一 program 内的中间张量**：`ga`（乘积）、`r`（residual）、`rf`（fp32 视图）、`mean/var/rstd/norm`（标量与广播量）、`y1/y` 全部为寄存器/UB 内中间量，不落 GM。**residual tile 是生产者-消费者双角色**：store 到 `residual_ptr`（契约输出 1）的同时，同一 tile 继续驻留 UB 供 LayerNorm 统计与归一化消费——无重读（声称的片上驻留由代码结构直接保证：`r` 在 store 后以 `rf = tl.where(mask, r.to(tl.float32), 0.0)` 复用，iter3 Stage7 瓶颈报告 §二第 6 条亦以 profiler 证实单 kernel 调用、无中间存储事件）。
- **跨 kernel 工作区**：无任何工作区 tensor；GM/HBM 读写仅 5 个输入 load + 2 个契约输出 store。
- **本轮修改对融合的影响**：融合程度**不变**（各路径仍单 kernel、零中间 GM 读写）。本轮改变的是行到 program 的映射与参数访存复用，属于 F5.schedule_routing 的调度维度调整，不触碰融合边界。wide_rows 路径下 gate/scale/shift 的 UB 驻留时间跨 ~26 行（每行复用同一份 fp32 参数 ~13.8KB×3），进一步降低了参数的 HBM/L2 流量。

### 与历史经验的一致性

- 融合家族 F2+F4 为 verified-effective（insight #1：31/36 case 达标 1.4–8.8x），本轮不改变融合边界。
- 不重复任何 ❌ 方向：3-pass blocked（insight #3）、persistent grid ~700 次循环（insight #4）均未采用；本轮行循环 ~26 次/program 且参数外提，与被否决模式有数量级差异（FIX_DIRECTIVE 注意事项明确区分）。
- 落地 insight #8（"grid=B*N per-row DMA overhead → 多行 kernel 是下一步"，状态 to be verified in iter4）：探针已给出 −66%~−70% 的 kernel 时长实测；正式评测后该 insight 应转 verified 或否决。
- T2 num_stages=2 探针为否决性证据存档：默认 tl.range 保留，避免后续重复尝试。

## 六、精度说明（perf 轮的精度保持性验证）

本轮为性能修改，无精度错误需要修复；精度目标是不漂移：

1. **数值链逐语句不变**：六处 `.to(dt)` 舍入点与 wide kernel 位置一致（AST 比对）；
2. **参数外提不改数值**：`g32/s32/h32/t` 对同一 `b` 的全部行是循环不变量，加载一次与逐行加载产生完全相同的位级输入；
3. **实测**：探针中 wide_rows 与 wide 输出**位级相同**（三 dtype × 两种子）；36 个给定 case residual 全部位级一致、y 全部在 2T 校准门限内，严格门限下仅 case 10/15 各 1 个 y 边界元素（fp64-margin impl==golden，属 iter2 已校准披露的双舍入抖动类，非实现缺陷）。
