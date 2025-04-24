# Qwen25 JAX Model with Tensor Parallelism

This directory contains a JAX implementation of the Qwen25 model with tensor parallelism support, enabling efficient inference on multi-GPU/TPU systems.

## Features

- JAX-based implementation for XLA compilation
- Tensor parallelism support for multi-device inference
- Memory-efficient attention mechanism
- Streaming text generation
- Support for standard sampling techniques (temperature, top-k, top-p)

## Getting Started

### Prerequisites

- JAX and Flax
- Transformers library (for tokenizer)
- Safetensors (for loading model weights)

### Installation

Make sure you have the required dependencies:

```bash
pip install jax jaxlib flax transformers safetensors
```

For GPU support, install the appropriate JAX version:

```bash
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Downloading Qwen25 Model Weights

The Qwen25 weights are already available locally at `qwen25-weights` directory.

If you need to download fresh weights:

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-7B
```

## Usage

### Running Inference

There are two ways to run inference with the Qwen25 model:

#### 1. Memory-Efficient Inference

This script focuses on memory optimization and runs a simple forward pass with the model:

```bash
# Basic usage - run the memory-efficient version
python run_memory_efficient.py --weights_dir qwen25-weights --dtype bfloat16

# With additional options
python run_memory_efficient.py --weights_dir qwen25-weights --dtype bfloat16 --generate --profile
```

#### 2. Text Generation

For interactive text generation with custom prompts:

```bash
# Generate text from a prompt
python run_inference.py --model_path qwen25-weights \
    --prompt "Write a short story about a robot learning to paint:" \
    --max_tokens 200

# With custom generation parameters
python run_inference.py --model_path qwen25-weights \
    --prompt "Explain how quantum computing works:" \
    --max_tokens 300 \
    --temperature 0.8 \
    --top_p 0.92 \
    --top_k 40 \
    --dtype bfloat16
```

### Advanced Options

- `--dtype`: Choose between float32, float16, or bfloat16 (default is bfloat16)
- `--mesh_shape`: Configure device mesh for tensor parallelism (e.g., '1,8')
- `--debug`: Enable detailed debug logging
- `--no_stream`: Disable streaming output during generation