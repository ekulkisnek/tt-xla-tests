# Tensor-Parallel Qwen2.5-7B JAX Implementation

A JAX implementation of Qwen2.5-7B with tensor parallelism support for multi-device inference.

## Overview

This implementation provides a memory-efficient, tensor-parallelized version of the Qwen2.5-7B model running in JAX. It supports various mesh shapes for tensor parallelism and is designed to run efficiently on multiple devices.

Key features:
- Tensor parallelism (not data parallelism)
- Memory-efficient parameter loading
- Support for various mesh shapes (1x8, 2x4, etc.)
- JAX JIT compilation for fast inference
- Incremental decoding with KV-cache

## Model Architecture

The Qwen2.5-7B model is a decoder-only transformer with the following components:

- **Token Embeddings**: Maps input token IDs to dense vectors
- **Transformer Layers** (28 layers):
  - **Self-Attention**: Multi-head attention with rotary position embeddings
  - **Feed-Forward Network**: SwiGLU activation with gate and up/down projections
- **Layer Normalization**: Applied before attention and FFN
- **Language Modeling Head**: Maps hidden states to vocabulary logits

### Tensor Parallelism Implementation

The model uses tensor parallelism to shard computation across multiple devices:

1. **Attention Computation**:
   - Query, Key, Value projections are sharded across the "model" dimension
   - Each device processes a subset of attention heads
   - Output projection is sharded in the opposite direction

2. **Feed-Forward Network**:
   - Gate and Up projections are sharded across the "model" dimension
   - Down projection is sharded in the opposite direction

3. **Language Modeling Head**:
   - Sharded across the "model" dimension to distribute the large vocabulary matrix

## Usage

### Requirements

- JAX >= 0.4.10
- Flax >= 0.7.2
- safetensors
- transformers

### Running the Model

Basic usage:

```bash
python qwen25_tp.py --model_path /path/to/qwen25-weights --prompt "Your prompt here" --mesh_shape 1,8
```

Options:
- `--model_path`: Path to the Qwen2.5-7B model weights (safetensors format with config.json)
- `--prompt`: Text prompt for generation
- `--max_tokens`: Maximum number of tokens to generate (default: 100)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top_k`: Top-k sampling parameter (default: 50)
- `--top_p`: Nucleus sampling threshold (default: 0.9)
- `--mesh_shape`: Device mesh shape for tensor parallelism (default: "1,8")
- `--dtype`: Data type for parameters (default: "bfloat16")
- `--simulate_tp`: Simulate tensor parallelism on a single device

### Supported Mesh Shapes

The implementation supports various mesh shapes for tensor parallelism:
- `1,8`: 1 data-parallel shard, 8 model-parallel shards
- `2,4`: 2 data-parallel shards, 4 model-parallel shards
- `1,32`: 1 data-parallel shard, 32 model-parallel shards
- `8,4`: 8 data-parallel shards, 4 model-parallel shards

## Memory Efficiency

The implementation includes several optimizations for memory efficiency:
1. **Streaming Parameter Loading**: Processes one safetensors file at a time
2. **Garbage Collection**: Actively frees memory during parameter loading
3. **KV-Cache Management**: Efficient handling of cached key-value states
4. **JAX JIT Compilation**: Optimizes computation through XLA

## Single-Device Usage

For systems with limited hardware, you can:
1. Use `--simulate_tp` to simulate tensor parallelism on a single device
2. The model automatically adjusts the mesh shape based on available devices

## Implementation Details

### Key Components

- `TensorParallelDense`: Dense layer with tensor parallelism support
- `QwenAttention`: Multi-head attention with tensor parallelism
- `QwenMLP`: Feed-forward network with tensor parallelism
- `Qwen25ForCausalLM`: Top-level model class

### Parameter Loading

The model loads parameters in a memory-efficient manner by:
1. Processing one safetensors file at a time
2. Converting parameters to JAX arrays with the correct dtype
3. Transposing weight matrices as needed
4. Merging parameters into the parameter dictionary
5. Garbage collecting after processing each file

## License

This implementation follows the license of the original Qwen2.5 model.

## Citation

If you use this implementation in your work, please cite both this repository and the original Qwen2.5 model.