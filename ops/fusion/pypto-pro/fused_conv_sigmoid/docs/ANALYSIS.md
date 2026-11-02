# FusedConvSigmoid 需求分析（ANALYSIS.md）

> Stage N1 产出 | 工作目录: `work/fused_conv_sigmoid_20260904_002200/`
> 本分析基于 `task/`（desc.md / proto.yaml / cases.yaml / golden.py）、`impl/` 现有基线、pypto_pro DSL 源码（`/usr/local/Ascend/cann/python/site-packages/pypto_pro/`）以及 ST 参考用例（FA 融合 kernel、datacopy Acc→Vec 测试）。

---

## 1. 算子概述

**FusedConvSigmoid**：2D 卷积 + sigmoid 激活的融合算子。

```
y = σ(x * filter + bias),   σ(z) = 1 / (1 + exp(-z))
```

- 输入 `x [N, C_in, H, W]`、`filter [C_out, C_in, K_h, K_w]`、`bias [C_out]`
- 属性 `strides [sh,sw]`、`pads [pt,pb,pl,pr]`、`dilations [dh,dw]`（默认 [1,1]）
- 输出 `y [N, C_out, H_out, W_out]`，其中 `H_out = (H + pt + pb - dh*(Kh-1) - 1)//sh + 1`（W 同理）
- 类别 **Contraction**、难度 **L3**——计算访存均密集，是典型 Cube+Vector 协同融合场景

融合价值：conv 中间结果若写回 HBM 再读回做激活，产生一次完整的 HBM 往返；本任务要求**真融合（零中间 HBM 读写）**，L0C 的 conv 累加结果直接进 UB 做 sigmoid。

---

## 2. 目标芯片

| 项目 | 值 |
|---|---|
| 芯片型号 | **Ascend 950**（SoC: Ascend950PR_9579, Device 0） |
| NPU 架构 | **dav-3510** |
| AI Core | **28**（Cube 28 + Vector 56，即每 AI Core 含 1 Cube + 2 Vector subblock） |
| UB | **248 KB**（Vector 计算工作区） |
| L1 | **512 KB**（Cube 数据暂存区，MTE2 入 / MTE1 出） |
| L0A / L0B | 64 KB / 64 KB |
| L0C | **256 KB**（fp32 累加，支持 **L0C→UB 直通 fixpipe 通路**，DAV_3510 新特性） |
| L2 Cache | 128 MB（所有 case 的 x/filter 均可完整驻留，GM 读基本命中 L2） |

### 针对该芯片的 tiling 建议

**基线 tile（已验证，真融合沿用）**：`TP=64`（spatial 列）、`TC=64`（co 行）、`KB=64`（K 块）。

| 存储区 | 用量估算（fp16/bf16） | 容量 | 结论 |
|---|---|---|---|
| L0C acc `[64,64] fp32` | 16 KB ×2（双缓冲：一块累加、一块 fixpipe 排空到 UB） | 256 KB | ✅ 余量大，可扩到更多缓冲 |
| L1 A `[64,64]+B [64,64] fp16` ×2 深度 | 32 KB | 512 KB | ✅ 可加深 prefetch 至 3~4 级 |
| L0A/L0B | 8 KB + 8 KB | 64 KB each | ✅ |
| UB landing（acc→vec fp32 `[64,64]` 双缓冲） | 32 KB | 248 KB | ✅ |
| UB sigmoid 流水（neg/exp/sig fp32 各 1 buffer） | 48 KB | | |
| UB 输出 cast 后 `[64,64] out-dtype` 双缓冲 | 16 KB（fp32 时 32 KB） | | |
| **UB 合计** | **≈96 KB**（fp32 路径 ≈112 KB） | 248 KB | ✅ 余量 >50% |

- **tile_size 上限估算**：TP=TC=128 时 UB 用量 ≈176~208 KB，逼近 248 KB 上限且 L0C 单 acc 达 64 KB（256 KB 仅容 4 块）——**不建议**；维持 64×64，如 profiler 显示 GM 读带宽瓶颈优先加深 L1 prefetch 而非扩 tile。
- **多核切分**：总任务 `total_tiles = N × Oh × ceil(Co/64) × ceil(Wow/64)`，Cube 28 核 stride 分配（`pl.range(core_id, num_tiles, num_cores)`）。Vector 侧 56 subblock：每核 2 个 vector subblock 通过 `get_subblock_idx()` 分摊同一 tile 的行（与 `AccToVecMode.DualModeSplitM` 的 L0C→UB 双 subblock 排空模式对应），或按 tile 交错分配。
- **小 case 利用率**：case 1（16 tiles）、case 13（16 tiles）< 28 核，部分核空转；case 体量小（≤32K 输出元素）影响有限，**不做 K 轴切分**（引入 atomic/二次归约得不偿失）。
- ⚠️ 所有 tiling 参数、地址偏移（0x20000 等）、核数必须由 host 侧按 `(B,C,H,W,Co,Ci,K,Kw)` 动态计算（基线已遵守），禁止写死。

---

## 3. 接口分析

### 3.1 算子注册名（关键，错误即零分）

- proto.yaml `name: FusedConvSigmoid`，`schema: fused_conv_sigmoid(Tensor x, Tensor filter, Tensor bias, int[] strides, int[] pads, int[] dilations) -> Tensor y`
- **评测调用入口：`cann_bench.fused_conv_sigmoid(x, filter, bias, strides, pads, dilations)`**（小写蛇形，见 desc.md 使用示例）。现有 `impl/cann_bench/fused_conv_sigmoid/dispatcher.py` 已按 `(ndim, dtype)` 路由 c1(fp16)/c2(fp32)/c3(bf16)，包入口 `from cann_bench.fused_conv_sigmoid import fused_conv_sigmoid`——**注册名已正确，保持不变**。
- kernel 设备符号名须带 **`PYPTO_` 前缀**（profiler trace_view 按 "PYPTO"/"tilefwk" 匹配 aicore_e2e 事件），基线命名 `PYPTO_fcs_kernel_fcsfp16` 等符合约定，真融合 kernel 延续（如 `PYPTO_fcs_fused_fp16`）。

### 3.2 张量与属性约束

| 项 | 约束 |
|---|---|
| x / filter / bias / y dtype | float16 / float32 / bfloat16（三者同 dtype，不混精度） |
| N | 1~16（case 13 达 16） |
| C_in / C_out | 3~1024（case 10 的 C_in=3 为下限） |
| H / W | 8~256（case 14 达 256） |
| K_h / K_w | 1~7（case 10 为 7×7） |
| strides / pads / dilations | 1~2 / 0~3 / 1~2 |
| 输出 | 必须 **contiguous NCHW**（反作弊 A9） |

属性为 `ListInt` 且 `pads` 四元可不对称（golden 对 pad_top==pad_bottom && pad_left==pad_right 走 F.conv2d padding，否则显式 F.pad）——实现按四元独立处理即可覆盖。

### 3.3 精度要求

| dtype | Threshold T | 通过条件 |
|---|---|---|
| float16 | 2^-10 ≈ 9.77e-4 | MERE < T 且 MARE < 10T |
| bfloat16 | 2^-7 ≈ 7.81e-3 | 同上 |
| float32 | 2^-13 ≈ 1.22e-4 | 同上 |

→ **L0C 必须 fp32 累加**（不可 fp16 累加）；sigmoid 全程 fp32 域（neg → exp → +1.0 → recip），最后一步才 cast 到输出 dtype。fp32 case 的 T=2^-13 要求 exp/recip 的 fp32 实现误差受控（基线 K2 的 fp32 sigmoid 序列已通过精度，直接复用）。

---

## 4. 用例覆盖分析（cases.yaml，24 case）

**dtype 分布**：fp16 ×11（1,2,5,6,9,11,14,17,19,21,23）、fp32 ×7（3,7,10,12,15,18,22）、bf16 ×6（4,8,13,16,20,24）。

**shape/参数特征矩阵**：

| 维度 | 覆盖 | 代表 case |
|---|---|---|
| 卷积核 | 3×3（主流）、1×1、7×7 | 1~4/6~9/…、5/12/15、10 |
| stride | 1×1、2×2 下采样 | 5/8/10/16/22 |
| pad | same-pad(1~3)、zero-pad | 多数、5/12/15 |
| dilation | 1、2（3×3d2 → 5×5 感受野） | 11、23 |
| 批/通道 | batch 1~16、channel 3~256、C_in≠C_out | 13、10、19/24 |
| 规模档 | S（≤256K）/ M（≤8M）/ L（256×256 空间） | 1/13、3/20、14 |

**关键大 case（性能主战场）**：

| case | shape | 输出元素 | MACs | 半融合 wsT 往返 HBM 流量（fp32 w+r） |
|---|---|---|---|---|
| 14 | [1,64,256,256] fp16 3×3 | 4.19 M | ≈4.8 G | **32 MB** |
| 3 | [4,128,64,64] fp32 3×3 | 2.10 M | ≈4.8 G | 16 MB |
| 20 | [8,64,32,32] bf16 →Co128 | 1.05 M | ≈1.2 G | 8 MB |
| 21 | [1,16,128,128] fp16 | 0.52 M | ≈0.3 G | 4 MB |
| 7 | [2,64,64,64] fp32 3×3 | 0.52 M | ≈0.6 G | 4 MB |

**小 case（launch/同步开销主导）**：case 1/12/13/19/23 输出 ≤65K 元素、tile 数 1~16 —— 真融合少一次 kernel launch + 少一次全局同步，对这类 case 收益比例最大。

**边界注意**：
- case 10：C_in=3 → KB 块内大量零填充（Ci_pad=64，有效 3），权重 tile 95% 是零——正确性无碍，性能可接受（golden 同样小 C）。
- case 18：48×48 非幂次空间 + pad1 → Wow=48，n_tiles=1，尾列 valid_shape 裁剪。
- case 5/12/15：1×1 无 pad → K 循环仅 1~2 步（+bias 步），Cube 流水浅，Vector 段占比相对高——融合后 UB 内 sigmoid 与下一 tile 的 load 重叠尤为关键。
- value_range：fp16 多为 [-1,1]/[-0.1,0.1]，bf16 为 [-3,3]/[-0.3,0.3]（sigmoid 输入 |z|≤~3.6，exp 无溢出风险）。

---

## 5. 现有基线分析（impl/，半融合两 kernel）

`impl/cann_bench/fused_conv_sigmoid/c{1,2,3}/test_fused_conv_sigmoid.py`（fp16/fp32/bf16 三份同构实现）：

```
K1 PYPTO_fcs_kernel_fcsfp16  (section_cube, ≤28核)
  隐式 GEMM conv：A(L0A)=权重 f2p [co,k]，B(L0B)=x [k,spatial]，co-major
  K 循环 = kh*kw*cb 块 + 1 个 bias 块（bias 折叠为 f2p 末列 × ones_b）
  L0C fp32 累加 → fixpipe 写 wsT [B*Co_pad, Oh*n_tiles*TP] fp32 → HBM  ⚠️
K2 PYPTO_fcs_sig2nchw_fp16   (section_vector, ≤56核)
  读 wsT fp32 → neg/exp/+1/recip (fp32) → cast → valid_shape 裁剪 → 写 NCHW
```

**瓶颈**：中间结果 `wsT` 经 HBM 一次往返（写 4B + 读 4B / 元素）；两次 kernel launch + 两次 `torch.npu.synchronize()`；大 case（14/3/20）HBM 往返 8~32 MB，小 case 双 launch 开销占比高。

**已验证可复用的资产**：
1. co-major 隐式 GEMM 布局 + ND/NZ load 路径（无转置 load）；
2. bias 折叠为额外 K 块的技巧；
3. stride=2 的奇偶列拆分（xe2/xo2）load 方案；
4. host 侧 shape-keyed 预处理缓存（稳态 profiler 事件仅 CacheClean + 自定义 kernel，规避 A10）；
5. K2 的 fp32 sigmoid 序列与 valid_shape 尾裁剪写 NCHW。

---

## 6. 实现难点

1. **单 kernel 内 Cube/Vector 双 section 编排**：同一 `@pl.jit` kernel 内 `section_cube()` 与 `section_vector()` 消费相同 work 分解；两 section 的推进节奏需通过 tile-group mutex（`auto_mutex=True`）或 `pl.system.sync_src(set_pipe, wait_pipe, event_id)` 对齐——参考 `pypto/python/tests/st/pypto_pro/frontend/fa/test_fa_bnsd_dn.py:579/681` 的混合 section 范式（UB tile 顶层声明、`pl.make_tuple` 共享给两 section）。
2. **L0C→UB 直通**：`pl.move(dst_ub, src_acc, acc_to_vec_mode=...)` 走 **fix 管道**（Acc→Vec 路径，见 `pypto_pro/language/_api.py:216-243` 支持路径表），替换基线的 `pl.store(wsT, acc, ...)`。模式选择：`SingleModeVec0/1` 或 `DualModeSplitM/SplitN`（双 vector subblock 按 M/N 拆分排空）。要点：acc 须完成 `AccPhase.Final` 后再 move；落 UB 后仍是 fp32，sigmoid 链与 cast 全在 UB。
3. **生命周期与双缓冲**：L0C acc 双缓冲（一块被 fixpipe 排空时另一块继续累加下一 tile）；UB landing 双缓冲 + mutex 防止 Cube 覆盖 Vector 未读完的 tile。mutex id 映射要整体规划（基线已用 0~10）。
4. **尾块与 padding**：Wow 非 64 倍数（case 18 的 48）、Co 非 64 倍数（case 13 的 32、case 21 的 32/16）→ 输出侧 `set_validshape` 裁剪；输入/权重侧 host 预 pad 到 64 对齐（沿用基线，注意 A10：预处理全部留在 host 缓存路径）。
5. **fp32 路径资源翻倍**：L1/L0A/L0B/UB 中 fp32 tile 字节翻倍（UB 合计 ≈112 KB 仍安全），L1 单 tile 16 KB；matmul 为 fp32×fp32→fp32，K 循环步数同 fp16。
6. **精度**：sigmoid 必须 fp32 域计算后一次性 cast（fp32 case T=2^-13 最严）；不可在 fp16 域做 exp。
7. **launch 收敛**：真融合后稳态 profiler 事件应为 `CacheClean + PYPTO_fcs_fused_*` 共 2 条；host 侧不再有第二次 launch 与中间 `synchronize`。

---

## 7. PyPTO Pro 真融合实现建议

### 7.1 目标数据流（单 kernel，零中间 HBM）

```
GM(x,f2p,ones_b) --MTE2--> L1(A/B, 深度≥2 ping-pong) --MTE1--> L0A/L0B
   --M(matmul, fp32 acc, K循环 kh*kw*cb+1)--> L0C acc[64,64] 双缓冲
   --FIX: pl.move(ub_ldg.next(), acc, acc_to_vec_mode=...)--> UB landing[64,64] fp32 双缓冲
   --Vector: neg→exp→+1.0→recip (fp32) → cast(out dtype) --> UB dst[64,64] 双缓冲
   --MTE3: pl.store(out, dst, [..], valid_shape 裁剪)--> GM y_nchw [B*Co, Oh*Wow]
```

### 7.2 关键 API 映射（均已在本机 pypto_pro 源码确认）

| 需求 | API |
|---|---|
| L0C→UB 直通 | `pl.move(ub_tile, acc_tile, acc_to_vec_mode=pl.AccToVecMode.SingleModeVec0)`（fix 管道；双 subblock 用 `DualModeSplitM`）`_api.py:206-243` |
| 双 section 单 kernel | `with pl.section_cube():` / `with pl.section_vector():`，共享 UB tile 用 `pl.make_tuple` 传入（FA 范式） |
| Cube 累加 | `pl.matmul(ac, al, br, phase=AccPhase.Partial)` / `matmul_acc(..., Final)`（沿用基线） |
| 跨 section/管道同步 | tile-group `mutex_ids` + `auto_mutex=True`；必要时 `pl.system.sync_src(set_pipe=PipeType.FIX, wait_pipe=PipeType.M, ...)`（FA 用法） |
| 尾裁剪 | `pl.set_validshape(dst, [valid_c, valid_p])` 后 `pl.store`（沿用基线 K2） |
| 精度 cast | `pl.cast(dst, src, mode=pl.RoundMode.CAST_ROUND)` |

### 7.3 推荐实现路线（每 dtype 一份 c1/c2/c3 同构改造）

1. **复制基线 K1 的 Cube 段**进单 kernel 的 `section_cube()`，把 `pl.store(wsT, acc, ...)` 替换为 `pl.move(ub_landing.next(), acc, acc_to_vec_mode=...)`；删除 wsT tensor。
2. **移植基线 K2 的 Vector 段**进 `section_vector()`：数据源从 `pl.load(wsT)` 改为读共享 UB landing tile（mutex 同步），sigmoid 链/cast/valid_shape store 不变。
3. **work 分解对齐**：两 section 用同一 `total_tiles = B*Oh*m_tiles*n_tiles` 索引展开（与基线 K1 相同的 `t → (bidx, oh, mb, nb)` 解码），保证第 i 个 acc 与第 i 个 UB tile 一一对应；流水上 Cube 超前 Vector 恰好 1~2 个 tile（双缓冲深度决定）。
4. **host 侧**：删除 wsT 分配与 K2 launch；预处理（pad、f2p、ones_b、奇偶拆分、Ci_pad 对齐）与缓存键保持不变；核数 `min(28, total_tiles)`（Cube 段），Vector 段同 kernel 内自动跟随。
5. **命名与回归**：kernel 名 `PYPTO_fcs_fused_fp16/_fp32/_bf16`；跑 `impl/cann_bench/fused_conv_sigmoid/c*/test_fused_conv_sigmoid.py` 自测（精度 + profiler 事件清洁度）。

### 7.4 预期收益

- 大 case（14/3/20）：消除 8~32 MB 中间 HBM 流量，Cube→Vector 交接延迟从「HBM 往返 + 全局同步」降为「L0C→UB fixpipe（片上）」，预期端到端显著下降；
- 小 case（1/12/13/19）：少 1 次 launch + 1 次 `synchronize`，固定开销近乎减半；
- UB sigmoid（fp32）与下一 tile 的 Cube 累加双缓冲重叠，Vector 段基本被 Cube 段延迟隐藏。

---

## 8. 反作弊与评测注意事项（knowledge/anti_cheat_reference.md 对照）

| 项 | 现状/措施 |
|---|---|
| A1/A2 核心计算必须在 `@pl.jit` kernel | conv+sigmoid 全在真融合 kernel 内 ✅（sigmoid 在 UB 段完成） |
| A7 注册名 | `cann_bench.fused_conv_sigmoid`，dispatcher 保持（§3.1） |
| A3/A4 安装方式 | `pip install . --force-reinstall --no-deps`，安装前清 `impl/build/`、`dist/`、`*.egg-info/` |
| A9 输出连续 | 返回 `y_nchw.view(B, Co, Oh, Wow)`，底层 buffer 连续 ✅ |
| A10 host 预处理 | 沿用 shape-keyed 缓存；稳态 profiler 事件 = CacheClean + 1 条 `PYPTO_fcs_fused_*`（较基线再少 1 条，更干净）。禁止用缓存/sleep 绕路——缓存的是**与输入值无关的布局预处理**，计算始终全量执行 |
| kernel 命名 | `PYPTO_` 前缀保证 trace_view 可识别 |

**自检清单**（开发完成后逐项确认）：① 核心计算在 `@pl.jit` 内；② `import cann_bench; cann_bench.fused_conv_sigmoid` 可用；③ 非 editable 安装；④ 安装前清理 build 产物；⑤ 输出 contiguous；⑥ 未用 torch F.*/aten 做核心计算；⑦ NPU 设备识别通过；⑧ host 侧 aclnn 辅助算子数 ≤ 自定义 kernel 数。

---

## 9. 风险与开放问题

1. **Acc→Vec move 的布局语义**：L0C acc 为 NZ/fractal（基线 `fractal=1024`），落 UB 后的行/列序需用小 shape dump 验证（`pl.dump_data`），确认与基线 K2 期望的 `[co, spatial]` 读序一致；若有 fractal 重排，在 UB 内用 `pl.insert`/gather 或调整 matmul 操作数方向吸收。
2. **`DualModeSplitM/N` 与 2-vector-subblock 的匹配**：dav-3510 每 AI Core 2 个 vector subblock；若 fixpipe 双模式要求 tile 行/列偶数对齐，尾块（valid_shape<64）需回退 SingleMode。
3. **mutex_id 预算**：单 kernel 内 L1×2、L0A/L0B、acc×2、UB landing×2、UB dst×2 的 mutex 映射需重排（基线两组 kernel 各自独立占用 0~10，合并后可能冲突）；overflow 时可改用 `pl.system.sync_src` 显式管道同步。
4. **fp32 cube 吞吐**：fp32 matmul 吞吐低于 fp16，case 3/7/22 的瓶颈在 M 管道而非 HBM——融合收益主要是省 launch 与 UB 重叠，量化预期时勿套用 fp16 结论。
