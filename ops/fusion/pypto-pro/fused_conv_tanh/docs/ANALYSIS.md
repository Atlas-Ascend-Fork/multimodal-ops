# FusedConvTanh 需求分析（ANALYSIS.md）

> Stage N1 产出 | 工作目录: `work/fused_conv_tanh_20260905_175720/`
> 本分析基于 `task/`（desc.md / proto.yaml / cases.yaml / golden.py）、cann-bench 反作弊参考（`knowledge/anti_cheat_reference.md`）、PyPTO-Pro 样例（`example/`）以及同类算子 **fused_conv_sigmoid** 的已验证实现与评测经验（半融合 avg_speedup≈4.56x，真融合≈2.87x）。

---

## 1. 算子概述

**FusedConvTanh**：2D 卷积 + tanh 激活的融合算子（类别 **Contraction**，难度 **L3**）。

```
y = tanh(x * filter + bias)
tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
```

- 输入 `x [N, C_in, H, W]`、`filter [C_out, C_in, K_h, K_w]`、`bias [C_out]`
- 属性 `strides [sh,sw]`、`pads [pt,pb,pl,pr]`、`dilations [dh,dw]`（默认 [1,1]）
- 输出 `y [N, C_out, H_out, W_out]`，其中 `H_out = (H + pt + pb - dh*(Kh-1) - 1)//sh + 1`（W 同理）
- 计算密集（Cube matmul）+ 访存密集（特征图 HBM 搬运）+ Vector 激活，是典型 **Cube+Vector 协同融合**场景

与同类算子 FusedConvSigmoid 的唯一差异在激活函数：tanh 的 VF 计算链比 sigmoid（neg→exp→+1→recip，4 步）多一步，且涉及**两次 exp**（或等价单 exp 形式，见 §7.2）。conv 侧的隐式 GEMM 数据流、tiling、预处理资产可整体复用。

**融合方向（用户指定，本文遵循）**：**优先实现两个独立 kernel 的半融合方案**——K1（Cube）做 conv 写 HBM workspace，K2（Vector）读 HBM 做 tanh；确保性能达标后，再评估是否升级为单 kernel 真融合（L0C→UB 片上直传）。依据：fused_conv_sigmoid 实测中半融合 avg_speedup=4.56x 远优于真融合的 2.87x——大 shape 时 per-tile Cube↔Vector 同步开销 > 省掉的 HBM 搬运。**不要强制真融合，性能数据说了算**。

---

## 2. 目标芯片

| 项目 | 值 |
|---|---|
| 芯片型号 | **Ascend 950**（SoC: Ascend950PR_9579, Device 0） |
| NPU 架构 | **dav-3510** |
| AI Core | **28**（Cube 28 + Vector 56，即每 AI Core 含 1 Cube + 2 Vector subblock） |
| UB | **248 KB**（Vector 计算工作区） |
| L1 | **512 KB**（Cube 数据暂存区，MTE2 入 / MTE1 出） |
| L0A / L0B | 64 KB / 64 KB（Cube 左/右操作数缓冲） |
| L0C | **256 KB**（fp32 累加，支持 **L0C→UB fixpipe 直通**，真融合升级路径用） |
| L2 Cache | **128 MB**（所有 case 的 x/filter/bias 均可完整驻留，GM 读基本命中 L2） |
| 检测方式 | tbe_device_query（orchestrator 从设备直读，所有 stage 共享） |

### 针对该芯片的 tiling 建议

**推荐基线 tile（与已验证的 fused_conv_sigmoid 资产一致）**：`TP=64`（spatial 列）、`TC=64`（co 行）、`KB=64`（K 块）。

**K1（Cube conv）资源估算**（fp16/bf16）：

| 存储区 | 用量估算 | 容量 | 结论 |
|---|---|---|---|
| L0C acc `[64,64] fp32` | 16 KB ×2（双缓冲：一块累加、一块 fixpipe 排空写 wsT） | 256 KB | ✅ 余量大 |
| L1 A(权重 f2p) + B(x) `[64,64] fp16` ×2~3 深度 | 32~48 KB | 512 KB | ✅ 可加深 prefetch 至 3~4 级 |
| L0A / L0B | 8 KB + 8 KB | 64 KB each | ✅ |

**K2（Vector tanh）UB 估算**（fp16 输出，tile `[64,64]`）：

| UB 区 | 用量 |
|---|---|
| landing fp32（wsT 读入）×2 双缓冲 | 32 KB |
| tanh 中间 fp32（e_pos/e_neg/den 或 t）×2~3 | 32~48 KB |
| dst 输出 dtype ×2 双缓冲（fp16 8KB / fp32 16KB each） | 16~32 KB |
| **合计** | **≈80~112 KB**（fp32 路径取上界） |

- **tile_size 上限估算**：TP/TC 扩到 128 时 K2 UB 用量 ≈176~208 KB，逼近 248 KB 上限且 K1 L0C 单 acc 达 64 KB（256 KB 仅容 4 块）——**首轮不建议**；维持 64×64，若 profiler 显示 K1 GM 读带宽瓶颈优先加深 L1 prefetch（L1 512 KB 余量充足）而非扩 tile。
- **多核切分**：总任务 `total_tiles = N × Oh × ceil(Co/64) × ceil(Wow/TP)`。K1（Cube 段）28 核 stride 分配（`pl.range(core_id, num_tiles, num_cores)`）；K2（Vector 段）可用满 **56 subblock**（28 核 × 2，`get_subblock_idx` 分摊或 tile 交错）——K2 是纯 Vector，tanh 链 2 次 exp 较重，56 subblock 正好摊薄。
- **小 case 利用率**：case 1/19（16 tiles）、case 12/13/23（tiles 少）< 28 核，部分核空转；case 体量小（≤64K 输出元素），launch/同步开销主导，K 轴切分（atomic/二次归约）得不偿失，**不做**。
- ⚠️ 所有 tiling 参数（TP/TC/KB）、L1 地址偏移、UB/L0 分配、核数必须由 host 侧按 `(N,C_in,H,W,C_out,K_h,K_w,strides,pads,dilations)` 动态计算（参考 sigmoid 基线的 shape-keyed 计算），**禁止写死**。

---

## 3. 接口分析

### 3.1 算子注册名（关键，错误即零分）

- proto.yaml：`name: FusedConvTanh`，`schema: fused_conv_tanh(Tensor x, Tensor filter, Tensor bias, int[] strides, int[] pads, int[] dilations) -> Tensor y`
- **评测调用入口：`cann_bench.fused_conv_tanh(x, filter, bias, strides, pads, dilations)`**（proto `name` 的小写蛇形形式，见 desc.md 使用示例）。实现时包结构必须保证 `import cann_bench; cann_bench.fused_conv_tanh` 可被 `getattr` 找到（顶层 `__init__.py` thin forwarder → `cann_bench/fused_conv_tanh/` 子包 → dispatcher → c{N}/test_*.py，复用 example 的目录约定）。
- kernel 设备符号名须带 **`PYPTO_` 前缀**（profiler trace_view 按 "PYPTO"/"tilefwk" 匹配 aicore 事件，反作弊 A1/A2 依赖此识别）。建议命名：K1 `PYPTO_fct_kernel_fctfp16/_fp32/_bf16`，K2 `PYPTO_fct_tanh2nchw_fp16/_fp32/_bf16`（镜像已验证的 fcs 命名体系）。

### 3.2 张量与属性约束

| 项 | 约束 |
|---|---|
| x / filter / bias / y dtype | float16 / float32 / bfloat16（三者同 dtype，不混精度） |
| N | 1~16（case 13 达 16） |
| C_in / C_out | 3~1024（case 10 的 C_in=3 为下限） |
| H / W | 8~256（case 14 达 256） |
| K_h / K_w | 1~7（case 10 为 7×7） |
| strides | 1~2（case 5/8/10/16/22 为 stride=2） |
| pads | 0~3，四元可不对称（golden 对 `pt==pb && pl==pr` 走 F.conv2d padding，否则显式 F.pad——实现按四元独立处理即可覆盖） |
| dilations | 1~2（case 11/23 为 dilation=2） |
| 输出 | 必须 **contiguous NCHW**（反作弊 A9，返回前 `.contiguous()` 或直接写连续 buffer） |

### 3.3 精度要求

| dtype | Threshold T | 通过条件 |
|---|---|---|
| float16 | 2^-10 ≈ 9.77e-4 | MERE < T 且 MARE < 10T |
| bfloat16 | 2^-7 ≈ 7.81e-3 | 同上 |
| float32 | 2^-13 ≈ 1.22e-4 | 同上 |

→ 推论：
- **L0C 必须 fp32 累加**（不可 fp16 累加）；
- **tanh 全程 fp32 域计算**（exp 链 + 除法均在 fp32），最后一步才 cast 到输出 dtype（`pl.cast(mode=CAST_ROUND)`）。fp32 case 的 T=2^-13 最严，不可在 fp16 域做 exp；
- 半融合方案中 wsT 保持 **fp32**（与 sigmoid 基线一致的已验证做法：conv 结果不提前 round 到 fp16，tanh 在 fp32 wsT 上计算，误差传播经 tanh 的 1-Lipschitz 特性有界，fp16 T=2^-10 实测余量充足）；
- fp32 case 可评估 **HF32 Cube 模式**（sibling 项目 `PYPTO_fcs_fused_fp32_hf32_*` 变体已精度通过），但 tanh 侧需重新验证 T=2^-13。

---

## 4. 用例覆盖分析（cases.yaml，24 case）

**dtype 分布**：fp16 ×11（1,2,5,6,9,11,14,17,19,21,23）、fp32 ×7（3,7,10,12,15,18,22）、bf16 ×6（4,8,13,16,20,24）。

**shape/参数特征矩阵**：

| 维度 | 覆盖 | 代表 case |
|---|---|---|
| 卷积核 | 3×3（主流）、1×1、7×7 | 1~4/6~9/…、5/12/15、10 |
| stride | 1×1、2×2 下采样 | 5/8/10/16/22 |
| pad | same-pad(1~3)、zero-pad、非对称可能 | 多数、5/12/15 |
| dilation | 1、2（3×3d2 → 5×5 感受野） | 11、23 |
| 批/通道 | batch 1~16、channel 3~256、C_in≠C_out | 13、10、19/24 |
| 规模档 | S（≤256K）/ M（≤8M）/ L（256×256 空间） | 1/13、3/20、14 |

**关键大 case（性能主战场，半融合 wsT HBM 往返 = 输出元素 × 8B）**：

| case | shape（out） | 输出元素 | MACs | wsT 往返（fp32 w+r） |
|---|---|---|---|---|
| 14 | [1,64,256,256] fp16 3×3 | 4.19 M | ≈2.4 G | **33.5 MB** |
| 3 | [4,128,64,64] fp32 3×3 | 2.10 M | ≈2.4 G | 16.8 MB |
| 20 | [8,128,32,32] bf16 3×3 | 1.05 M | ≈0.6 G | 8.4 MB |
| 10 | [1,64,112,112] fp32 7×7 s2 | 0.80 M | ≈0.12 G | 6.4 MB |
| 7 | [2,64,64,64] fp32 3×3 | 0.52 M | ≈0.3 G | 4.2 MB |
| 21 | [1,32,128,128] fp16 3×3 | 0.52 M | ≈0.08 G | 4.2 MB |

**小 case（launch/同步开销主导）**：case 1/12/13/19/23 输出 ≤65K 元素、tile 数 1~16 < 28 核——半融合的两次 launch + 一次中间 synchronize 对这类 case 占比最高；真融合升级的主要收益点（若最终需要）。

**边界注意**：
- **case 10**：C_in=3 → KB 块内大量零填充（Ci_pad=64，有效 3），权重 tile ~95% 是零——正确性无碍，性能可接受（golden 同样小 C）；7×7×s2 使 K 循环步数=49/KB×cb，流水最深的 case。
- **case 18**：48×48 非幂次空间 + pad1 → Wow=48，spatial 尾块 valid_shape 裁剪（n_tiles=1）。
- **case 13**：batch16 × 8×8 小空间 → tile 数靠 N×Co 维度撑起（128 tiles），多核切分按 `t → (n, oh, mb, nb)` 解码即可均匀铺满。
- **case 5/12/15**：1×1 无 pad → K 循环仅 1~2 步（+bias 步），Cube 流水浅，K1 快 K2 相对重——K2 的 56 subblock 并行与 tanh 链精简（见 §7.2）对这些 case 尤为关键。
- **dilation=2（11/23）**：im2col 等效 K 展开窗口 5×5，load 地址计算需含 dilation 项（sibling 基线 xe2/xo2 奇偶拆分方案按 stride 处理，dilation 只影响 L1 内取数步长）。
- **value_range 与 tanh 定义域**：fp16 cases |z|≤~1.0，fp32 ≤~5.0，bf16 ≤~8.4 → `exp(±z)` 与 `exp(2z)≈e^16.8` 均远离 fp32 溢出（~88.7），**单 exp 优化形式数值安全**（见 §7.2）；tanh 输出天然有界 [-1,1]，无 saturate 风险。

---

## 5. 融合方向可行性分析（半融合优先）

### 5.1 半融合 vs 真融合的量化对比

| 维度 | 半融合（K1+K2） | 真融合（单 kernel L0C→UB） |
|---|---|---|
| 中间流量 | wsT fp32 HBM 往返 8B/elem（case 14: 33.5 MB） | 零（fixpipe 片上） |
| launch/同步 | 2 次 launch + 1 次 synchronize | 1 次 launch |
| Cube↔Vector 耦合 | 无（HBM 解耦，K1/K2 各自满流水） | per-tile M↔FIX mutex/事件同步 |
| 实测（fused_conv_sigmoid 同硬件） | **avg_speedup ≈ 4.56x** | avg_speedup ≈ 2.87x（per-tile 同步拖累大 case） |
| 失败风险 | 低（K1/K2 均有已验证同构代码可移植） | 中（双 section 编排、mutex 预算、acc 布局语义需逐项验证） |

**结论**：按用户指定路径——**首轮实现半融合**。理由：(1) sigmoid 同类实测数据明确支持；(2) K2 只比 sigmoid 的 K2 多 1~2 个 VF 步，风险集中在已被验证的 K1 conv 资产复用上；(3) 半融合达标后再评估真融合，届时仅优化 case 1/12/13/19 等小 case 的 launch 开销，目标增量明确。

### 5.2 半融合的片上数据流（推荐实现）

```
K1  PYPTO_fct_kernel_fct<dtype>   (section_cube, ≤28 Cube 核)
  GM(x, f2p(filter), ones_b) --MTE2--> L1(A/B, 2~3-deep ping-pong) --MTE1--> L0A/L0B
     --M: matmul K循环(kh*kw*cb 块 + 1 个 bias 块, fp32 acc)--> L0C acc[64,64] 双缓冲
     --FIX: pl.store(wsT, acc)--> HBM wsT [B*Co_pad, Oh*ntiles*TP] fp32（co-major）

K2  PYPTO_fct_tanh2nchw_<dtype>   (section_vector, ≤56 subblock)
  HBM wsT fp32 --MTE2: pl.load--> UB landing[64,64] fp32 双缓冲
     --Vector: tanh 链(fp32 域)--> UB y fp32
     --Vector: pl.cast(out dtype, CAST_ROUND)--> UB dst 双缓冲
     --MTE3: pl.store(out, valid_shape 尾裁剪)--> GM y_nchw [B*Co, Oh*Wow]
```

要点：
- K1 与 sigmoid 基线 K1 **逐行同构**（隐式 GEMM co-major、bias 折叠为 f2p 末列 × ones_b、stride=2 奇偶列拆分、Ci_pad 对齐），仅把 `pl.store(wsT, acc)` 的目标名/形状改前缀；
- K2 与 sigmoid 基线 K2（`PYPTO_fcs_sig2nchw_*`）同构，把 sigmoid 链（neg→exp→+1→recip，4 步）替换为 tanh 链（§7.2，6 步/单 exp 5 步）；
- wsT 布局沿用 co-major `[B*Co_pad, Oh*ntiles*TP]`：K2 按 `(b, co_block, oh, p)` 解码索引与 NCHW 写回地址——该"读 wsT → 裁剪 → 写 NCHW"路径已在 sigmoid 全部 24 case 精度通过，**零新逻辑**。

### 5.3 真融合升级路径（预留，性能达标后评估）

仅当半融合达标但小 case（1/12/13/19）拖累整体分时考虑：把 K2 移入 K1 的 `section_vector()`，`pl.store(wsT, acc)` 替换为 `pl.move(ub_landing.next(), acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)`（L0C→UB fixpipe 直通，sibling 真融合 kernel 已验证该 API 与 DualModeSplitM 双 subblock 排空模式），wsT tensor 删除。升级判据：per-tile 同步开销（mutex + M→FIX 事件）< 省下的 wsT HBM 往返时间。**不要预先实现**。

---

## 6. 实现难点

1. **tanh 的 VF 链设计**：2 次 exp 是 K2 主要开销（exp 为多周期 VF 指令）。默认按 prompt 给定链 `(e^z - e^-z)/(e^z + e^-z)`（6 VF 步）；可用代数等价 `tanh(z) = (e^{2z} - 1)/(e^{2z} + 1) = 1 - 2/(e^{2z} + 1)` 压到 **1 次 exp + 5 步**（§7.2），case 5/12/15 这类 K1 浅流水的 case 收益最明显。数值安全域已验证（§4）。
2. **fp32 case 精度（T=2^-13）**：exp/recip 的 fp32 实现误差需受控；不可在 fp16 域做任何中间计算；bf16 输入需先升精度 fp32（`pl.cast CAST_NONE`）再进 conv/tanh 链。
3. **尾块与 padding**：Wow 非 64 倍数（case 18 的 48）、Co 非 64 倍数（case 13 的 32、case 21 的 32/16）→ K2 输出侧 `set_validshape` 裁剪；输入/权重侧 host 预 pad 到 64 对齐（沿用 sigmoid 缓存方案，注意 A10：预处理不得在稳态产生额外 device kernel）。
4. **case 10 的 C_in=3**：KB 块 95% 零填充，K 循环 49 步 × cb 块——K1 流水最深 case，L1 3-deep prefetch 的价值最大。
5. **双 kernel 的核数/索引解耦**：K1 用 28 核、K2 用 56 subblock，两 kernel 的 tile 分解公式必须同一套 `t → (n, oh, mb, nb)` 解码，保证 wsT 写读地址一致（错位即精度全崩）。
6. **fp32 路径资源翻倍**：L1/L0/UB 中 fp32 tile 字节翻倍（K2 UB ≈112 KB 仍 < 248 KB 安全）；fp32 matmul 吞吐低于 fp16，case 3/7/22 瓶颈在 M 管道，量化收益预期时勿套用 fp16 结论。
7. **launch 收敛与 A10**：稳态 profiler 事件应为 `CacheClean + PYPTO_fct_kernel_* + PYPTO_fct_tanh2nchw_*` 共 3 条；host 侧不得出现 F.pad/F.conv2d/torch.tanh 等 torch 算子（A2 明确把 `Conv2D`、`PadV3` 列为判死项）。

---

## 7. PyPTO Pro 实现建议

### 7.1 包结构与分发（反作弊 A7 前置）

```
cann_bench/
├── __init__.py                    # thin forwarder: from .fused_conv_tanh import fused_conv_tanh
└── fused_conv_tanh/
    ├── __init__.py                # from .dispatcher import fused_conv_tanh
    ├── dispatcher.py              # 按 (ndim, dtype) 路由 c1(fp16)/c2(fp32)/c3(bf16)
    └── c1/c2/c3/test_fused_conv_tanh.py   # K1 + K2 @pl.jit kernel + host wrapper
```

入口函数名查找顺序 `<op>` → `<op>_wrapper`，wrapper 签名 `fused_conv_tanh(x, filter, bias, strides, pads, dilations=[1,1])`（与 golden 一致，`dilations` 有默认值）。

### 7.2 K2 tanh VF 链（fp32 域，tile `[64,64]`）

**默认链（prompt 给定，两 exp）**：
```
neg = vf.subs(z, 0.0)... 或 vf.neg(z)     # -z
epos = vf.exp(z)                           # e^z
eneg = vf.exp(neg)                         # e^-z
num  = vf.sub(epos, eneg)
den  = vf.add(epos, eneg)
y    = vf.div(num, den)                    # 或 recip(den) 后 mul
```
UB 预算：landing×2 + epos/eneg/den + dst ≈ 96 KB（fp16）/112 KB（fp32）✅。

**优化链（单 exp，推荐 A/B 对比）**：`tanh(z) = 1 - 2/(exp(2z)+1)`
```
z2   = vf.mul(z, 2.0)
t    = vf.exp(z2)
t1   = vf.adds(t, 1.0)
r    = vf.recip(t1)
r2   = vf.mul(r, 2.0)
y    = vf.rsub(r2, 1.0)                    # 1.0 - r2
```
5~6 步、**1 次 exp**；溢出安全域 |z|≤8.4 → exp(16.8)≈2e7 ≪ fp32 上限；大 |z| 时 `1-ε` 无灾难性消去。实现时两条链做成同 kernel 内的编译期开关，精度/性能实测后择优。

### 7.3 K1 实现要点（复用 sigmoid 已验证资产）

1. 隐式 GEMM co-major：A(L0A)=权重 f2p `[co,k]`，B(L0B)=x `[k,spatial]`，输出天然 `[co, spatial]` 行主序与 wsT/NCHW 写回对齐（无转置 load）；
2. bias 折叠为额外 K 块（f2p 末列 × ones_b），K 循环 = `ceil(kh*kw*Ci_pad/KB) + 1`；
3. L0C fp32 双缓冲 + fixpipe 写 wsT（`pl.store`）；
4. stride=2 的奇偶列拆分（xe2/xo2）load 方案照搬；
5. host 侧 shape-keyed 预处理缓存（f2p/ones_b/pad/拆分），稳态 profiler 仅 CacheClean + 2 条自定义 kernel。

### 7.4 推荐实现路线

1. 从 `work/fused_conv_sigmoid_20260904_002200/impl/cann_bench/fused_conv_sigmoid/c{1,2,3}/` **复制三份同构实现**为骨架（c1 fp16 / c2 fp32 / c3 bf16）；
2. K1 改名 `PYPTO_fct_kernel_fct<dtype>`，逻辑不动；
3. K2 改名 `PYPTO_fct_tanh2nchw_<dtype>`，把 sigmoid 链替换为 §7.2 tanh 链（先默认两 exp 链保正确，再 A/B 单 exp 链）；
4. dispatcher/forwarder/包名全部换 `fused_conv_tanh`（A7：`getattr(cann_bench, "fused_conv_tanh")` 必须命中）；
5. host wrapper：`y = torch.empty(..., device)`（连续 buffer）→ launch K1 → launch K2 → `torch.npu.synchronize()` → 返回 `y.view(N, C_out, H_out, W_out)`（contiguous，A9）；
6. 自测：24 case 精度（官方判定器 MERE/MARE）+ kernel_details.csv 事件清洁度。

### 7.5 预期收益

- 大 case（14/3/20/10/7/21）：K1 与 sigmoid 基线等速，K2 比 sigmoid 多 ~1 次 exp —— K2 大 case 是 wsT 带宽受限而非 VF 受限，**预期 avg_speedup 与 sigmoid 半融合的 4.56x 同量级**；
- 小 case（1/12/13/19）：双 launch 固定开销占比高，若最终总分被拖累，再走 §5.3 真融合升级路径定向优化。

---

## 8. 反作弊与评测注意事项（knowledge/anti_cheat_reference.md 对照）

| 项 | 措施 |
|---|---|
| A1/A2 核心计算必须在 `@pl.jit` kernel | conv 在 K1、tanh 在 K2，全部自定义 kernel ✅；**严禁 aclnnConv2D/PadV3/torch.tanh 拼接**（A2 点名列举） |
| A7 注册名 | `cann_bench.fused_conv_tanh`（proto `name: FusedConvTanh` 小写蛇形），forwarder→dispatcher→c{N} 链路完整 |
| A3/A4 安装方式 | `pip install . --force-reinstall --no-deps`（非 editable）；安装前清 `impl/build/`、`dist/`、`*.egg-info/` |
| A9 输出连续 | 返回 `y.view(N, C_out, H_out, W_out)`，底层 buffer 由 host `torch.empty` 连续分配 ✅ |
| A10 host 预处理 | 沿用 shape-keyed 缓存；稳态 profiler 事件 = CacheClean + `PYPTO_fct_kernel_*` + `PYPTO_fct_tanh2nchw_*`；禁止用缓存/sleep 绕路——缓存的是布局预处理，计算始终全量执行 |
| kernel 命名 | `PYPTO_` 前缀保证 trace_view 可识别 |

**自检清单**（开发完成后逐项确认）：① 核心计算在 `@pl.jit` 内；② `import cann_bench; cann_bench.fused_conv_tanh` 可用；③ 非 editable 安装；④ 安装前清理 build 产物；⑤ 输出 contiguous；⑥ 未用 torch F.*/aten 做核心计算；⑦ NPU 设备识别通过；⑧ host 侧 aclnn 辅助算子数 ≤ 自定义 kernel 数。

---

## 9. 风险与开放问题

1. **tanh 单 exp 链的 fp32 精度余量**：`1 - 2/(e^{2z}+1)` 在 z≈0 附近与 golden `torch.tanh` 的相对误差需实测（理论 ~2 ulp fp32，远小于 T=2^-13，但以官方判定器为准）；不达标则回退两 exp 链。
2. **fp32 case 的 HF32 选项**：sibling 的 hf32 变体已精度通过，但 tanh 不改变 conv 误差量级——若 fp32 case 性能不足再启用，启用前必须重跑精度。
3. **wsT fp32 vs 输出 dtype**：wsT 存 fp32（已验证）流量 8B/elem；改存 out-dtype（K1 内先 cast）可省一半流量但引入"conv 结果先 round 再 tanh"的二次量化误差——fp16 T=2^-10 理论可过，仅当 case 14 带宽实测为瓶颈时作为备选优化。
4. **case 13（batch16）的 tile 解码**：N 维进入 tile 分解首位，确保 `(n, oh, mb, nb)` 解码在 K1/K2 两侧一致，避免 batch 维错位。
5. **真融合升级的 mutex 预算**：若走 §5.3，单 kernel 内 L1×2、L0A/L0B、acc×2、UB landing×2、dst×2 的 mutex_id 需重排（sibling 真融合已解决，可整体移植其映射方案）。
