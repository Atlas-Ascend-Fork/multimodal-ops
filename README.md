# multimodal-ops

# multimodal-ops

> CANN Multimodal-AI SIG — 算子层仓库

多模态 AI 算子库，面向昇腾 NPU 提供高性能多模态融合算子实现。按算子功能族组织，覆盖从编码、融合、空间几何到扩散生成的完整算子栈。

## 算子族

| 算子族 | 目录 | 典型算子 | 说明 |
|--------|------|----------|------|
| **编码算子** | `ops/encoders/` | Patch Embedding、Voxel Encoding、STFT | 将不同模态的原始输入（图像 patch、点云 voxel、音频时频）编码为统一的 token 表示 |
| **融合算子** | `ops/fusion/` | Cross-Attention、Query Fusion、Feature Fusion | 多模态特征交互与融合，实现跨模态信息对齐与聚合 |
| **空间几何算子** | `ops/geometry/` | Camera Projection、BEV Transform、Spatial Alignment | 多传感器空间坐标变换与对齐，支持 2D→3D 投影、BEV 视角转换等几何运算 |
| **扩散生成算子** | `ops/diffusion/` | Noise Scheduler、Conditional Injection、Denoising Step | 扩散模型核心计算，包含噪声调度、条件控制注入和去噪迭代步 |

## 目录结构

```
multimodal-ops/
├── ops/                    # 算子实现（按族分目录）
│   ├── encoders/           #   编码算子（Patch · Voxel · STFT）
│   ├── fusion/             #   融合算子（Attention · Query · Feature）
│   ├── geometry/           #   空间几何算子（投影 · 空间对齐）
│   └── diffusion/          #   扩散生成算子（调度 · 条件 · 生成）
├── common/                 # 公共工具代码（ut