# multimodal-ops

> CANN Multimodal-AI SIG — 算子层仓库

多模态 AI 算子库，面向昇腾 NPU 提供高性能多模态融合算子实现。当前已实现融合算子族，后续将按社区共建扩展编码、空间几何、扩散生成等算子族。

## 源码下载

```bash
git clone https://gitcode.com/cann/multimodal-ops.git
cd multimodal-ops
```

## 算子族

| 算子族 | 目录 | 典型算子 | 状态 | 说明 |
|--------|------|----------|------|------|
| **融合算子** | `ops/fusion/` | Fused Conv+Sigmoid、Fused Conv+Tanh、Fused MulAdd+LayerNorm+MulAdd | ✅ 已实现 | 卷积+激活、归一化类融合算子，实现跨模态信息对齐与聚合；按开发框架分 `pypto-pro/` 与 `triton-ascend/` 两个子目录 |
| 编码算子 | `ops/encoders/` | Patch Embedding、Voxel Encoding、STFT | 待共建 | 将不同模态的原始输入编码为统一的 token 表示 |
| 空间几何算子 | `ops/geometry/` | Camera Projection、BEV Transform | 待共建 | 多传感器空间坐标变换与对齐 |
| 扩散生成算子 | `ops/diffusion/` | Noise Scheduler、Denoising Step | 待共建 | 扩散模型核心计算 |

## 开发框架

本仓算子不限开发框架。PyPTO Pro 是面向昇腾达芬奇架构的 SPMD 编程模型，通过 `@pl.jit` 装饰器将 Python 代码编译为 NPU kernel，支持 Cube（矩阵运算）和 Vector（向量运算）两种计算单元的协同编程；Triton-Ascend（`@triton.jit`）亦可作为实现框架。

算子的开发和迭代优化使用 **pypto-pro-workflow** 多智能体自动开发框架驱动（见 [multimodal-benchmark 仓](https://gitcode.com/cann/multimodal-benchmark) 的 `ops_tools/workflow/`），3 个 AI Agent 协作完成需求分析→代码编写→编译部署→精度/性能评测→迭代优化的全流程，评测工具为 cann-bench。

## 已有算子

| 算子 | 目录 | 公式 | 融合方式 | 评测得分 | 平均加速比 |
|------|------|------|---------|---------|-----------|
| Fused Conv+Sigmoid | `ops/fusion/pypto-pro/fused_conv_sigmoid/` | `y = σ(conv2d(x, w, b))` | Cube-Vector 片上直传融合（L0C→UB） | 87.50 | 2.87× |
| Fused Conv+Tanh | `ops/fusion/pypto-pro/fused_conv_tanh/` | `y = tanh(conv2d(x, w, b))` | 双 Kernel 流水融合（K1 Cube + K2 Vector） | 90.82 | 3.92× |
| Fused MulAdd+LayerNorm+MulAdd | `ops/fusion/triton-ascend/fused_mul_add_layernorm_mul_add/` | `y = LN(x + gate·attn_out) · (1+scale) + shift` | Triton-Ascend 单 kernel 融合（residual 片上驻留，零 L2 重读） | 90.32 | 3.58× |

## 目标硬件

当前算子均基于 **Ascend 950**（28 AI Core）开发和验证，当前提供 fp16 实现。详细硬件参数见各算子 `docs/ANALYSIS*.md`。

## 版本配套

| 项目 | 版本 |
|------|------|
| 产品型号 | Ascend950PR |
| NPU 芯片 | Ascend 950 |
| CANN 版本 | 9.1.0 |
| 驱动版本 | 25.7.rc1 |
| 操作系统 | Ubuntu 24.04.3 LTS (x86_64) |
| Python | >= 3.10（验证版本 3.12.3） |
| PyTorch | >= 2.1（验证版本 2.7.1） |
| torch_npu | 与 PyTorch 版本匹配（验证版本 2.7.1.post8） |
| PyPTO | 0.2.1（CANN 内置） |

## 快速使用

以 fused_conv_sigmoid 为例：

### 构建安装

```bash
cd ops/fusion/pypto-pro/fused_conv_sigmoid/impl
pip install . --force-reinstall --no-deps
```

### Python 调用

```python
from cann_bench import fused_conv_sigmoid

output = fused_conv_sigmoid(x, conv_filter, bias, strides=[1, 1], pads=[0, 0, 0, 0])
```

算子接收 fp16 输入张量，自动分发到对应的 NPU kernel。详细设计文档见各算子 `docs/` 目录。

## 目录结构

```
multimodal-ops/
├── ops/                                # 算子实现（按族分目录）
│   └── fusion/                         #   融合算子（按开发框架分子目录）
│       ├── pypto-pro/                  #     PyPTO Pro 实现
│       │   ├── fused_conv_sigmoid/    #       Conv+Sigmoid 融合算子
│       │   │   ├── impl/              #         算子实现（融合 kernel）
│       │   │   └── docs/              #         设计文档与评测报告
│       │   └── fused_conv_tanh/       #       Conv+Tanh 融合算子
│       │       ├── impl/              #         算子实现
│       │       └── docs/              #         设计文档与评测报告
│       └── triton-ascend/             #     Triton-Ascend 实现
│           └── fused_mul_add_layernorm_mul_add/  # MulAdd+LayerNorm+MulAdd 融合算子
├── common/                             # 公共工具代码（待共建）
├── examples/                           # 算子调用示例（待共建）
└── docs/                               # 算子设计文档与 API 参考（待共建）
```

## 贡献指南

欢迎参与贡献，请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解贡献流程与规范。

## 安全声明

请参阅 [SECURITY.md](SECURITY.md) 了解安全相关信息。

## 许可证

本项目采用 [CANN Open Software License Agreement Version 2.0](LICENSE) 许可证。

## 所属 SIG

本仓库隶属于 CANN 社区 [multimodal-ai SIG](https://gitcode.com/cann/community/blob/master/CANN/sigs/multimodal-ai/README.md)。

## 问题反馈

如有问题或建议，请通过 [Issue](../../issues) 提交反馈。
