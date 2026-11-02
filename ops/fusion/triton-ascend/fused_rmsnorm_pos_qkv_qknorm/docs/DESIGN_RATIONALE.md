# 融合方案选择决策依据 — fused_rmsnorm_pos_qkv_qknorm（iter6 开发轮：C kernel tile 精化 + case22 分离扩展轮）

> 本文档说明本轮融合方案的选择、依据与影响面；与自测报告分开。
> 结论：**维持 F1（算子拆分/半融合）四-kernel 框架与全部分离边界不变**，在 F1 的
> partition 语义内落地 tech_lead 三项任务：T1（检查 case19 C kernel 的 Cube 占比，
> 只读）、T2（C kernel BLOCK_M 64→128 @ n_rows>=1024，仅 split 路径）、T3（split
> 门控放宽为 D>=512 only，使 case22 走分离路径）。初始库候选、方法定义与原 Jev
> 概率在 `develop/iter6/fusion_library.json` 的 `candidates` 中原样保留，本轮未新增
> 方法；T2/T3 均先经 A/B 实测验证后再交付。

## 1. 选择了什么

- **method_ids: ["F1"]**（初始 Jev 库 rank-1，probability=0.76；rank-2 F5=0.76、
  rank-3 F7=0.75 未选）。
- 融合边界与 kernel 结构**零变化**：K0（bf16 且 D>=512 时设备端 W 预转 fp16）、
  P（`_h_producer_kernel`，纯 Vector）、C（`_gemm_consumer_kernel`，Cube+Vector）、
  K2（`_qk_norm_kernel`，纯 Vector）；fused 路径 K1（`_norm_pos_qkv_kernel`）保留。
  host 门控从 `D>=512 and n_rows>=512` 放宽为 `D>=512`（T3）。
- 本轮两处调度差异全部内嵌于同一实现的 host 端 metadata 谓词：
  - **T2**：`bm_c = _BLOCK_M_C_BIG(128) if n_rows >= 1024 else bm`，仅作用于 split
    路径 C kernel launch（grid M 维 + constexpr BLOCK_M）。P kernel、fused K1、K2
    零改动。BM=128×BK=128 的 UB 峰值 ~160KB < 192KB（h_hi/h_lo/w 各 [128,128]
    fp16 = 32KB×3 + acc [128,128] fp32 = 64KB），编译成功。
  - **T3**：case22（bf16 B1S384D512，n_rows=384）新走 K0→P（bm=32，grid=12
    programs）→C（bm_c=32，grid=(12,12)=144 programs）→K2；h_hi/h_lo [384,512]
    fp16 = 384KB×2，每次调用新分配。
- 依赖：全部同 stream 顺序，无同步原语。

## 2. 参考了哪些概率与已测证据

- **初始概率（只读参考）**：F1/F5/F7 = 0.76/0.76/0.75。实测优先，本轮未偏离 F1。
- **T1 检查证据（先于 T2 完成，来自 eval/iter6 正式评测 profiler）**：
  `kernel_details.csv` 中 case19 的 `_gemm_consumer_kernel` 行：Block Num=384 ✓、
  Duration 160.10~174.22us（5 次中位 166.84，正式报告取 167.66）✓、Accelerator
  Core = MIX_AIC（Cube+Vector 混合）。Cube 利用率 FLOP 推算：
  - hi/lo 双 dot 全量 MAC = 2（hi+lo）× M×N×K = 2×2048×1536×512 = 3.221e9 MAC
    = 6.442 GFLOP（按 2 FLOP/MAC）；
  - 6.442 GFLOP / 167.66us = **38.4 TFLOPS**；Ascend910B3 Cube fp16 峰值
    ~375 TFLOPS → **Cube busy ≈ 10.2%**；
  - tech_lead 任务单的 2.6% 口径是把 MAC 数（1.611e9）当作 FLOP 数除以 375 TFLOPS，
    系统性低估 4×；无论采用哪个口径（2.6%~10.2%），**均远低于 30% 饱和线**；
  - 结论：分离后 C 仍由 per-tile 固定标量序列（2×h load + W load + 2×dot 发射）
    与 load 延迟主导，而非 Cube 饱和 → **T2 tile 减半有空间**（验收条款"若
    Cube 已饱和则不推荐"不触发）。
- **T2 A/B 实测（old=iter6 评测版 vs new，同会话各 3 次中位，op_summary 口径）**：
  - case19：C 154.32→**106.84us（-30.8%）**，远超验收线 ≤140us；K1 等效
    （K0+P+C）178.92→132.40（-26.0%）；总 kernel 时 196.92→151.84（-22.9%）。
  - case13：C 75.74→53.84（**-28.9%**）；case5：C 81.42→52.96（**-35.0%**）。
  - case2/16（n_rows=512<1024，谓词不触发）：C 43.40→43.74 / 43.74→43.38
    （持平，±0.8%），总 kernel 时 ±0.6% —— 行为不变无退步 ✓。
  - 收益机制：tiles 减半（384→192 / 192→96）使固定标量序列与 W 重复读
    （每 M-tile 一遍全量 W）同比例减半；单 tile 时间 0.40→0.56us（wall/tile，
    数据量翻倍所致）但总时间大降，进一步证实标量摊薄主导。
- **T3 A/B 实测**：
  - case22：fused K1 86.46us → split P+C = 7.54+56.52 = 64.06us（**-25.9%**）；
    K0 4.68→5.06（噪声级）；K2 4.98→5.28（噪声级）；总 kernel 时
    96.12→74.40（**-22.6%**），≤ 验收线 85us ✓。
  - tech_lead 担心的 P grid 过小（6 programs <20 核）被实测否定：放宽后 P 以
    bm=32 分块得 12 programs，P 仅 7.54us，launch+buffer 开销已被 GEMM 收益
    覆盖且有余。
  - C per-tile 56.52/144 tiles = 0.392us，与 tech_lead 预测的 ~0.39us 一致。
- **精度与残留证据**：TC4 22/22 PASS（mismatch=0、sv=0，含 BM=128 触发的
  case19/13/5 与新 split 的 case22）；连续调用 21/21 PASS（case22 换参 3 次
  验证 h buffer 无跨调用残留）；TC3 五种 kernel 事件齐全，case22 走分离链。

## 3. 针对哪些 case

- 全部 22 case 共用单入口 + host 端 metadata 谓词，无 case 编号路由。
- **T2 BM=128 命中 3 个 case**（worst_6 中的 19/13/5）：
  - **case19**（bf16 B2S1024D512，最慢 case）：C tiles 384→192，C -30.8%；
  - **case13**（fp16 B4S256D512，n_rows=1024 临界触发）：C tiles 192→96，-28.9%；
  - **case5**（fp16 B1S1024D512，n_rows=1024）：C tiles 192→96，-35.0%。
- **T3 命中 1 个 case**：**case22**（bf16 B1S384D512，worst_6 之一）——fused→split，
  K1 等效 -25.9%，总 kernel 时 -22.6%。
- **仅检查、未修改**：**case16**（n_rows=512<1024，C BM 保持 64；A/B C 持平
  43.38us，K2 5.58→5.66us 噪声）；**case12**（D=256，fused 路径逐字节不变，
  K1 117.08→120.12us +2.6% 属历史 ±7.4% 噪声带；D=256 K-loop 仅 4 次 BK=64，
  per-tile 已是固有计算量，无 tiling 空间，维持 iter6 结论）。
- 其余 16 个 case：代码路径行为不变（fused 路径逐字节不变；case7 A/B -1.8%
  噪声）。受益面：4 个 case（19/13/5/22）K1 等效 -24.5%~-30.4%（含 K0 口径）。

## 4. 实际改了什么、为什么预计有效

1. **修改点（全部在 `impl/cann_bench/fused_rmsnorm_pos_qkv_qknorm.py`）**：
   - 常量区新增 `_BLOCK_M_C_BIG = 128`；
   - split 门控 `if D >= 512 and n_rows >= 512:` → `if D >= 512:`（T3）；
   - split 路径内新增 `bm_c` 选择，C launch 的 grid M 维与 BLOCK_M constexpr
     改用 `bm_c`（T2）；
   - 模块 docstring 与门控注释同步更新；五个 kernel 主体与 fused 路径零改动。
2. **机制（T2）**：C per-tile 成本由固定标量序列主导（T1 证实 Cube busy 仅
   ~10%）。BM 64→128 后 tiles 减半 → 标量序列与 W 重复读（每 M-tile 一遍全量
   W：48MB→24MB L2 读，case19 口径）近似减半，而 dot 工作量不变、Cube 远未
   饱和可吸收单 tile 变大的指令粒度。实测 C -28.9%~-35.0% 与"标量减半"模型
   定量吻合。
3. **机制（T3）**：case22 的 fused K1 per-tile 0.604us 中含 h 重建标量序列
   （12 个 N 块程序重复执行）；分离后 P 一次性产出 h 对（7.54us），C 只剩
   load+双 dot（per-tile 0.392us）。h round-trip 384KB×2 全落 L2（工作集远小于
   192MB），launch 数 +1 的代价被 per-tile 收益覆盖。
4. **精度保障**：T2 只改 C 的 M 维 tile 划分（行块边界变化，列/K 维与累加顺序
   不变，hi/lo 双 dot 算术逐位保持）；T3 复用与 case19/13/5 完全相同的
   P/C kernel 与谓词语义（case22 走 bm=32/bk=128，与 K1 同分块）。TC4 实测
   22/22 mismatch=0、sv=0。

## 5. 为什么保留原方案而未选更高/同概率方案

- **未选 F5（混合路由）**：本轮的两处调度差异（C BM=128、split 门控放宽）都是
  单一 metadata 谓词（n_rows、D）驱动的 host 端分支，行为等价
  "F5.schedule_routing + fallback"；引入多 variant dispatcher 只会增加编译
  成本与边界脆弱性，无额外收益。
- **未选 F7（拆分-融合协同搜索）**：C tile 高度与分离边界均由 tech_lead 依据
  正式评测数据直接裁定，且本轮 A/B 逐一证实；不存在需要搜索的多边界自由度，
  直接实测比搜索更省更准。
- **T2 未取更激进的 BM=256**：acc [256,128] fp32 = 128KB 加 h/w 对 >192KB UB
  预算，不可编译；BM=128 已达成标量减半的主要收益。
- **T3 未回退**：验收条款"若 elapsed 增加则回退门控"未触发（A/B -22.6%）。

## 6. 本轮融合程度的影响

- **融合边界零变化**：本轮没有新增或拆除任何 HBM 中转——split 路径仍是
  P（rmsnorm+pos）→ h_hi/h_lo [n_rows,D] fp16（唯一 HBM 中间流量）→
  C（GEMM，q/k 写 fp32 ws、v 直写）→ K2（split+二次 RMSNorm）；fused 路径
  K1→K2 不变。T2 只改变 C 内部的 tile 划分，不改变任何跨 kernel 数据流。
- **case22 的融合度有意下降**（T3）：由 K1 深融合（3 kernel）改为 P/C 分离
  （4 kernel），新增 384KB×2 h round-trip 与 1 次 launch——与 iter6 的
  case19 同一权衡，实测净赚 -25.9% K1 等效，方向与 F1 的 dataflow 描述一致
  （"cache hits do not guarantee an HBM-free boundary"——本轮收益论证全部
  基于实测 A/B，不依赖缓存命中假设）。
- **其余 case 融合程度零变化**；x 不再被 N 块重复读的收益（iter6 已获得）
  保持。

## 7. 本轮新增的失败/排除记录（防止重复尝试）

| 尝试 | 结果 | 证据 |
|---|---|---|
| C kernel BM=128 @ n_rows<1024（假想） | 未实施：case2/16 n_rows=512 时 M-tiles 仅 8，BM=128 会把 grid 压到 (4,12)=48 programs 欠填充 20 核，且单 tile 数据量翻倍的收益消失；本轮谓词 n_rows>=1024 保证 M-tiles>=8×12=96 | probe_ab.py case2/16 对照组 |
| split 门控放宽到 D>=512 之外的更小 D（假想） | 未实施：D<=256 K-loop 仅 4 次（BK=64），P/C 的 launch+buffer 开销无法回本（iter6 已证），case12 A/B 路径不变 | knowledge/history.json insights #11 |
| BM=256（假想） | 未实施：acc [256,128] fp32=128KB + h/w 对超 192KB UB | UB 核算 |
| persistent grid、num_stages、fp32 dot、单 dot consumer、K2 BM2=64、BLOCK_K=256、host 侧 W 预转 | 历史已否决，本轮未触碰 | knowledge/history.json insights |

## 8. 下一轮建议的候选方向（供裁定，非本轮执行）

1. **正式评测确认**：本轮投影（case19 speedup 6.34→约8.0、case13 12.04→约14.9、
   case5 12.59→约16.9、case22 13.59→约17.4，avg_speedup 31.80→约32.3~32.5，
   按正式/探针口径偏差 ~9% 保守约 32.2）待 cann-bench 同口径评测确认；重点
   核对 BM=128 触发的 3 个 case 与新 split 的 case22 的 kernel 明细是否复现
   A/B 比例。
2. **case19 C kernel 的进一步压缩**（若评测后 case19 仍为最慢）：C 的 hi/lo
   双 dot 中 h_lo dot 理论上可仅对 q/k 段保留、v 段单 dot（v 精度阈值较宽）——
   需精度实测，属 kernel 结构变更，未在本轮范围内尝试。
3. **K0 频次优化**（低优先）：case19/22 的 K0（4.6~7.8us）每次调用重转全量 W；
   若正式评测确认其占比上升，可评估按 W 元素数分段的更小 cast 粒度，但每调用
   新分配的合规约束（无跨调用缓存）限制了空间。

Measured implementation selection (Jev probabilities are advisory; measured evidence takes priority).
Underperforming stagnation requests review, never mandatory fusion replacement.
