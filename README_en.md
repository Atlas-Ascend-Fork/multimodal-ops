# multimodal-ops

> CANN Multimodal-AI SIG — Operator Repository

A multimodal AI operator library providing high-performance fused operator implementations for Ascend NPUs. Currently includes fusion operators, with encoding, spatial geometry, and diffusion operator families planned for community co-development.

## Source Code

```bash
git clone https://gitcode.com/cann/multimodal-ops.git
cd multimodal-ops
```

## Operator Families

| Family | Directory | Typical Operators | Status | Description |
|--------|-----------|-------------------|--------|-------------|
| **Fusion** | `ops/fusion/` | Fused Conv+Sigmoid, Fused Conv+Tanh | Implemented | Conv+activation fused operators for cross-modal alignment |
| Encoding | `ops/encoders/` | Patch Embedding, Voxel Encoding, STFT | Planned | Encode multi-modal raw inputs into unified token representations |
| Spatial Geometry | `ops/geometry/` | Camera Projection, BEV Transform | Planned | Multi-sensor spatial coordinate transforms |
| Diffusion | `ops/diffusion/` | Noise Scheduler, Denoising Step | Planned | Core diffusion model computations |

## Development Framework

Operators are developed using the **PyPTO Pro** programming framework — an SPMD programming model for Ascend DaVinci architecture. It compiles Python code into NPU kernels via the `@pl.jit` decorator, supporting both Cube (matrix) and Vector (vector) compute units.

Operator development and iterative optimization are driven by the **pypto-pro-workflow** multi-agent framework (see `ops_tools/workflow/` in the [multimodal-benchmark repo](https://gitcode.com/cann/multimodal-benchmark)).

## Available Operators

| Operator | Directory | Formula | Fusion Method | Score | Avg Speedup |
|----------|-----------|---------|---------------|-------|-------------|
| Fused Conv+Sigmoid | `ops/fusion/fused_conv_sigmoid/` | `y = σ(conv2d(x, w, b))` | Cube-Vector on-chip direct transfer (L0C→UB) | 87.50 | 2.87× |
| Fused Conv+Tanh | `ops/fusion/fused_conv_tanh/` | `y = tanh(conv2d(x, w, b))` | Dual-kernel pipeline (K1 Cube + K2 Vector) | 90.82 | 3.92× |

## Target Hardware

All operators are developed and validated on **Ascend 950** (28 AI Cores), currently providing fp16 implementations. See `docs/ANALYSIS*.md` in each operator directory for detailed hardware parameters.
## Version Compatibility

| Item | Version |
|------|---------|
| Product | Ascend950PR |
| NPU Chip | Ascend 950 |
| CANN | 9.1.0 |
| Driver | 25.7.rc1 |
| OS | Ubuntu 24.04.3 LTS (x86_64) |
| Python | >= 3.10 (validated with 3.12.3) |
| PyTorch | >= 2.1 (validated with 2.7.1) |
| torch_npu | Matching PyTorch version (validated with 2.7.1.post8) |
| PyPTO | 0.2.1 (bundled with CANN) |

## Quick Start

Example with fused_conv_sigmoid:

### Build & Install

```bash
cd ops/fusion/fused_conv_sigmoid/impl
pip install . --force-reinstall --no-deps
```

### Python Usage

```python
from cann_bench import fused_conv_sigmoid

output = fused_conv_sigmoid(x, conv_filter, bias, strides=[1, 1], pads=[0, 0, 0, 0])
```

The operator accepts fp16 input tensors and dispatches to the appropriate NPU kernel.

## Directory Structure

```
multimodal-ops/
├── ops/                                # Operator implementations (by family)
│   └── fusion/                         #   Fusion operators
│       ├── fused_conv_sigmoid/         #     Conv+Sigmoid
│       │   ├── impl/                  #       Implementation (fused kernel)
│       │   └── docs/                  #       Design docs & evaluation reports
│       └── fused_conv_tanh/           #     Conv+Tanh
│           ├── impl/                  #       Implementation
│           └── docs/                  #       Design docs & evaluation reports
├── common/                             # Shared utility code (planned)
├── examples/                           # Operator usage examples (planned)
└── docs/                               # Design docs & API reference (planned)
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

See [SECURITY.md](SECURITY.md) for security information.

## License

Licensed under [CANN Open Software License Agreement Version 2.0](LICENSE).

## SIG

This repository belongs to the CANN community [multimodal-ai SIG](https://gitcode.com/cann/community/blob/master/CANN/sigs/multimodal-ai/README.md).

## Feedback

For questions or suggestions, please submit an [Issue](../../issues).
