# Qwen3.8-Flash-Next GB10 Optimization & Hugging Face Deployment

## Project Overview
This project creates an NVFP4-quantized, PLE-packed version of Qwen3.8-Flash-Next optimized for NVIDIA GB10 (DGX Spark) with 128GB unified memory.

## Model Architecture
- **125B** parameter main model with Mixture-of-Experts (MoE)
- **51B** N-gram embedding table (PLE)
- **6B** parameters activated per token (10 routed experts + 1 shared)
- **48** decoder layers with hybrid GDN + Qwen Sparse Attention (QSA)
- **1** MTP (Multi-Token Prediction) layer
- **27** layer vision encoder (multimodal)
- **262,144** token context length natively

## Target Hardware
- **NVIDIA GB10** (DGX Spark): 128GB unified LPDDR5X memory, 273 GB/s bandwidth
- **Compute Capability**: sm_121 (Blackwell)
- **Tensor Cores**: 5th generation with native FP4 support

## Licensing
- Model: Qwen Community License 1.0 (see LICENSE)
- Commercial use as "AI Work Assistant" requires separate Qwen license
- See `licenses/` directory for details

## Quick Start

```bash
# 1. Prepare environment (see ENV_SETUP.md)
bash scripts/prepare_env.sh

# 2. Download original model
bash scripts/download_model.sh

# 3. Quantize to NVFP4 with PLE packing
bash scripts/quantize.sh

# 4. Upload to Hugging Face
bash scripts/upload_hf.sh
```

## Directory Structure
```
Qwen3.8-Flash-Next/
├── containers/        # Docker definitions for vLLM/SGLang
├── configs/           # Quantization and serve configs
├── tools/             # Helper scripts (doctor, download, etc.)
├── quantization/      # ModelOpt recipes and export
├── runtime/           # GB10-specific kernel adapters
├── kernels/           # Custom CUDA kernels (NVFP4, GDN, PLE)
├── benchmarks/        # Performance/evaluation suites
├── evals/             # Quality benchmarks (coding, vision, etc.)
├── deploy/            # Deployment manifests
└── licenses/          # License texts
```

## Prerequisites
- Ubuntu 24.04 / DGX OS with aarch64
- NVIDIA DGX Spark (GB10) or compatible Blackwell GPU
- 1TB+ NVMe storage for model files
- Docker with GPU support
- NVIDIA Container Toolkit
- Hugging Face account with write access
- Python 3.10+ with CUDA 13.x

## Key Dependencies
- `nvidia-modelopt[hf]`: NVFP4 quantization pipeline
- `vllm`: Serving engine
- `sglang`: Alternative serving engine
- `compressed-tensors`: Weight format support
- `torch==2.5.0+`: With CUDA 13 PTX 12.1