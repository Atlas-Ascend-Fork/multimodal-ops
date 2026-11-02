# WORK_RECORD — fused_mul_add_layernorm_mul_add

> 工作目录：`算子3/fused_mul_add_layernorm_mul_add_20261003_132221`（develop 5 轮设计 + eval 4 轮正式评测）。
> 本记录为提交到 multimodal-ops 的收敛摘要，逐轮细节见各 develop/iterN/design_rationale.md 与 self_test_report.md，正式数字见 eval/iterN/perf_reports/。

## 交付物

| 路径 | 内容 |
|---|---|
| `impl/cann_bench/fused_mul_add_layernorm_mul_add/fused_mul_add_layernorm_mul_add.py` | Triton-Ascend 融合实现（4 个 kernel + host 三级路由），取自 operator_iter/iter4（最终版） |
| `impl/build.sh` / `impl/setup.py` | 构建脚本（wheel 打包 + import 自检），按仓库规范对齐 |
| `docs/ANALYSIS.md` | 需求分析（算子概述、计算语义、实现结构、评测结论） |
| `docs/DESIGN_RATIONALE.md` | 最终轮设计推演（多行 wide kernel + 二级路由 + num_stages 探针回退，来自 develop/iter3） |
| `docs/WORK_RECORD.md` | 本文件 |

## 迭代轨迹

| 轮次 | 主题 | 精度自测 | 正式评测 |
|---|---|---|---|
| develop/iter0 | 首版：4 kernel + 三级路由的基线设计 | 36/36 PASS | — |
| develop/iter1 → eval/iter1 | perf_optimize 第 1 轮 | 36/36 PASS | 综合 88.07，加速比 3.34x |
| develop/iter2 → eval/iter2 | perf_optimize 第 2 轮（融合方案回退修正） | 36/36 PASS | 综合 87.05，加速比 3.18x |
| develop/iter3 → eval/iter3 | perf_optimize 第 3 轮 | 36/36 PASS | 综合 88.27，加速比 3.38x |
| develop/iter4（develop/iter3 rationale 对应实现轮）→ eval/iter4 | 多行 wide kernel + 二级路由落地（T1）+ num_stages=2 探针按验收标准回退（T2） | 14/14 PASS（路由覆盖全路径） | **综合 90.32，加速比 3.58x，36/36 通过** |

## 最终实现要点

- **4 个 Triton kernel + host 三级路由**（C≤1024 one-shot / 1024<C≤2048 按行数二分 wide vs wide-rows / C>2048 blocked），路由为纯元数据谓词，无 case 编号、无 benchmark 身份判断
- **多行 wide-rows kernel**（iter4 新增）：`P = min(seq, max(1, 40//batch))`（40 = Vector core 数），每 program 服务一个 batch 元素的多行，gate/scale/shift 参数一次加载行间复用
- **num_stages=2 探针**：双形态 3 轮交替实测无一致收益，按验收标准回退默认 `tl.range`
- **精度**：与 golden 逐位对齐（6 处 native 舍入点一致），36 case 精度 100% 通过；residual 位级一致
