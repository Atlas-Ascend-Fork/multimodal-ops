# 算子评测报告：fused_conv_tanh

生成时间：2026-09-06
总迭代次数：1（1轮有效评测）
终止原因：target_met_fusion_verified

## 最终结果（iter1）

| case_id | dtype | baseline(us) | 实际耗时(us) | speedup | 是否达标(≥1.0) |
|---------|-------|-------------|-------------|---------|--------------|
| fused_conv_tanh_1 | fp16 | 25.83 | 3.80 | 6.797 | ✅ |
| fused_conv_tanh_2 | fp16 | 27.89 | 6.99 | 3.990 | ✅ |
| fused_conv_tanh_3 | fp32 | 67.82 | 15.67 | 4.328 | ✅ |
| fused_conv_tanh_4 | bf16 | 30.63 | 9.92 | 3.088 | ✅ |
| fused_conv_tanh_5 | fp16 | 26.62 | 3.35 | 7.946 | ✅ |
| fused_conv_tanh_6 | fp16 | 26.87 | 6.94 | 3.872 | ✅ |
| fused_conv_tanh_7 | fp32 | 38.35 | 6.06 | 6.328 | ✅ |
| fused_conv_tanh_8 | bf16 | 34.56 | 15.74 | 2.196 | ✅ |
| fused_conv_tanh_9 | fp16 | 26.62 | 6.89 | 3.864 | ✅ |
| fused_conv_tanh_10 | fp32 | 36.11 | 8.41 | 4.294 | ✅ |
| fused_conv_tanh_11 | fp16 | 33.40 | 14.82 | 2.254 | ✅ |
| fused_conv_tanh_12 | fp32 | 26.04 | 5.95 | 4.376 | ✅ |
| fused_conv_tanh_13 | bf16 | 25.75 | 10.43 | 2.469 | ✅ |
| fused_conv_tanh_14 | fp16 | 59.65 | 20.90 | 2.854 | ✅ |
| fused_conv_tanh_15 | fp32 | 26.04 | 8.35 | 3.119 | ✅ |
| fused_conv_tanh_16 | bf16 | 29.22 | 10.83 | 2.698 | ✅ |
| fused_conv_tanh_17 | fp16 | 26.62 | 5.20 | 5.119 | ✅ |
| fused_conv_tanh_18 | fp32 | 33.97 | 29.83 | 1.139 | ✅ |
| fused_conv_tanh_19 | fp16 | 26.03 | 5.10 | 5.104 | ✅ |
| fused_conv_tanh_20 | bf16 | 37.64 | 31.25 | 1.204 | ✅ |
| fused_conv_tanh_21 | fp16 | 30.25 | 16.44 | 1.840 | ✅ |
| fused_conv_tanh_22 | fp32 | 37.65 | 5.28 | 7.131 | ✅ |
| fused_conv_tanh_23 | fp16 | 27.68 | 5.30 | 5.223 | ✅ |
| fused_conv_tanh_24 | bf16 | 28.34 | 9.96 | 2.845 | ✅ |

- overall_score: 90.82
- performance_score: 40.82
- compilation_score: 20.00
- function_score: 30.00
- avg_speedup: 3.92x
- 达标 case 数：24/24（100%）
- 精度：24/24 全部通过

## 迭代历史

| 迭代 | precision | overall_score | perf_score | avg_speedup | 优化方向 | 效果 |
|------|----------|--------------|------------|-------------|---------|------|
| iter1 | 24/24 通过 | 90.82 | 40.82 | 3.92x | 基线迭代, 半融合双kernel架构(K1 Cube conv + K2 Vector tanh) | baseline |

## 最佳迭代

- 最佳轮次：iter1（唯一有效评测轮次）
- 该轮指标：overall_score=90.82 / performance_score=40.82 / avg_speedup=3.92x
- 首轮即达标退出，无退化情况

## 优化经验总结（从 history.json 提取）

| 方向 | 轮次 | 效果 | 状态 | 说明 |
|------|------|------|------|------|
| 半融合架构 | iter0 | half-fusion 4.56x > true-fusion 2.87x（参考数据） | ✅已验证 | Ascend 950上Cube→Vector数据路径必经HBM, 半融合是架构最优解 |
| fp32三段分解 | iter0 | MERE≤1.5e-6(阈值1.22e-4, 余量80-1000x) | ✅已验证 | 精度极优但fp32 cases K1耗时6x, 是fp32性能瓶颈根因 |
| 统一Tile TP=TC=KB=64 | iter0 | 通用参数, 24 cases全用64x64 | 🔄待继续 | 部分case可能受益于更宽tile或shape-specific分支 |
| profiler清洁度 | iter0 | 24个csv均为CacheClean+K1+K2, 零aclnn | ✅已验证 | 代码架构合规, 无需改动 |
| fp32 case性能弱 | iter0 | case18 speedup=1.14, 6x K-chain使K1 duration高 | 🔄待继续 | 可考虑减少K-chain到4段或HF32模式 |
| 小channel/小spatial | iter0 | case21 Cube利用率低(75%空MAC), case13 tile数少 | 🔄待继续 | 优化空间有限, 受Cube利用率和launch overhead限制 |

**关键结论：**
- ✅已验证：半融合双kernel架构是Ascend 950最优融合方案（参考fused_conv_sigmoid: half-fusion 4.56x > true-fusion 2.87x）
- ✅已验证：fp32 三段分解精度极优（MERE余量80-1000x），但6x K-chain是fp32性能瓶颈的结构性根因
- ✅已验证：profiler清洁度合规，24个case均为自定义kernel，零aclnn辅助算子
- 🔄待继续：fp32路径K-chain简化（6x→4x）预计可将case18从1.14x提升到~1.4-1.6x
- 🔄待继续：shape-specific tile调优（统一64x64对非power-of-2 shape不最优）
- 🔄待继续：HF32 Cube模式替代3-way split，可能一步消除fp32路径6x开销

## 性能瓶颈分析

### 1. 核心瓶颈

最弱6个case中, case18(fp32,speedup=1.14)和case20(bf16,speedup=1.20)是性能短板。case18根因: fp32 6x K-chain使K1 duration达到fp16等效的6倍; case20根因: batch=8产生512个tiles, 单tile粒度小, K1+K2总开销接近baseline。其余cases(21/8/11/13)受限于小channel Cube利用率低或小spatial tile数少。整体avg_speedup=3.92x已接近半融合架构天花板(参考fused_conv_sigmoid 4.56x)。

### 2. 框架/硬件限制

本算子全部24个case均达标(speedup≥1.0)，不存在不可达标case。但以下结构性限制制约了进一步提升：

- **fp32 6x K-chain开销**：fp32精度要求使用normalized 3-way fp16 split，K迭代数为fp16的6倍。case18(fp32,[2,64,48,48])的K1占比>90%，是最接近不达标的case(1.14x)。此为精度-性能权衡的结构性代价。
- **Cube→Vector数据路径必经HBM**：Ascend 950硬件上L0C无法直通Vector UB，中间结果wsT必须经HBM一次写+一次读。这是半融合（而非真融合）的硬件根因。
- **小channel Cube利用率低**：case21(C_in=16)实际有效计算仅占Ci_pad=64的25%，75%为空MAC。

### 3. worst cases 追踪

| case | 描述 | speedup | baseline(us) | elapsed(us) | 根因 |
|------|------|---------|-------------|-------------|------|
| case_18 | fp32,[2,64,48,48],64→64ch,3x3k | 1.14 | 33.97 | 29.83 | fp32 6x K-chain使K1 duration极高 |
| case_20 | bf16,[8,64,32,32],64→128ch,3x3k | 1.20 | 37.64 | 31.25 | batch=8产生大量tiles, 总开销接近baseline |
| case_21 | fp16,[1,16,128,128],16→32ch,3x3k | 1.84 | 30.25 | 16.44 | C_in=16极小, 75%空MAC |
| case_8 | bf16,[1,128,64,64],128→256ch,3x3k | 2.20 | 34.56 | 15.74 | 大spatial+大channel, 性能合理 |
| case_11 | fp16,[2,128,32,32],dilation=2,3x3k | 2.25 | 33.40 | 14.82 | dilation=2使等效kernel 5x5 |
| case_13 | bf16,[16,32,8,8],32→32ch,3x3k | 2.47 | 25.75 | 10.43 | 8x8 tiny spatial, launch overhead限制 |

## 融合算子方案演变（从 history.json 提取）

| 迭代 | 融合方案 | 证据 | 状态 |
|------|---------|------|------|
| iter0 | 半融合双kernel: K1(Cube,28核) implicit-GEMM conv → wsT(HBM) → K2(Vector,56子块) tanh全链 | profiler: 24 case均为CacheClean+K1+K2, 零aclnn; 参考: half-fusion 4.56x > true-fusion 2.87x | ✅有效 |
| iter1 | exit验证: 半融合是Ascend 950架构最优解, Cube→Vector必经HBM, 单kernel反而丧失56 Vector子块并行度 | 24/24 cases kernel_details均为2个自定义kernel+CacheClean, avg_speedup=3.92x接近天花板4.56x | ✅有效 |

**重点分析：**

1. **最终采用的融合方式**：半融合双kernel架构。K1(Cube,28核)执行implicit-GEMM卷积，L0C fp32累加→fixpipe store→wsT(HBM, fp32 co-major)；K2(Vector,56子块)从HBM读wsT→UB，在UB内完成tanh全链(exp→neg→exp→sub→add→div, 7个独立stage buffer)→cast→store NCHW。

2. **融合程度评价**：**半融合**——K1 Cube输出的中间结果wsT经HBM一次写+一次读传递给K2 Vector，但K2在UB内完成tanh全链零额外HBM访问。相比baseline的conv+tanh两个独立aclnn算子，减少了一次完整的输出tensor HBM写+读（baseline需要conv输出→HBM→tanh读入→HBM→最终输出，共4次HBM访问；半融合仅需3次）。

3. **演变过程**：设计阶段即选定半融合方案，参考fused_conv_sigmoid的实测数据（half-fusion 4.56x vs true-fusion 2.87x）。Ascend 950硬件上L0C（Cube累加缓冲）无法直通Vector UB，数据路径必经HBM，因此真融合（零HBM中间读写）在当前硬件上不可行。半融合方案充分利用了28 Cube核+56 Vector子块的异构并行度，首轮即达标。

4. **为何无法做到真融合**：Ascend 950的Cube单元和Vector单元在硬件层面共享AI Core但使用不同的片上存储层级（L0C vs UB），L0C到UB没有直通数据路径。强行将conv和tanh放入单个kernel会丧失Vector 56子块并行度（Cube核数28限制），反而性能更差（参考数据: true-fusion仅2.87x）。

## 建议的后续优化方向

| 优先级 | 方向 | 预期收益 |
|--------|------|---------|
| P1 | fp32路径K-chain从6x简化为4x（去掉hi*lo和lo*hi交叉项） | case18从1.14x→~1.4-1.6x |
| P1 | case20负载均衡：检查tile decode是否导致核间不均 | case20从1.20x提升 |
| P2 | HF32 Cube模式替代3-way split | fp32路径可回归1x K-chain |

## 最终实现文件
- `impl/` 目录
- 最佳轮次即最后一轮（iter1），代码为最终版本

## 详细日志
- `WORK_RECORD.md`
- `log/workflow.log`
- `knowledge/history.json`（完整的跨轮经验数据）
