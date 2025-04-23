# Qwen2.5-7B Tensor Parallel Implementation in JAX/Flax

This directory contains a tensor parallel implementation of the Qwen2.5-7B model using JAX and Flax. The implementation supports various mesh shapes for tensor parallelism, making it suitable for running on multiple devices efficiently.

## Features

- Tensor parallelism (not data parallelism) following the JAX GSPMD approach
- Memory-efficient weight loading from HuggingFace safetensors files
- Support for Grouped Query Attention (GQA) with proper tensor sharding
- Support for various mesh shapes (1x8, 2x4, 1x32, 8x4)
- Text generation with customizable parameters
- BFloat16/Float16/Float32 precision support

## Requirements

- JAX
- Flax
- Safetensors
- Transformers (for tokenizer)

Install dependencies with:

```bash
pip install jax jaxlib flax transformers safetensors numpy
```

## Model Architecture

The Qwen2.5-7B model is a decoder-only transformer with the following architecture:

- 28 transformer layers
- 3584 hidden dimensions
- 28 attention heads
- 4 key/value heads (Grouped Query Attention)
- SwiGLU activation function
- RMSNorm for normalization
- Rotary positional embeddings

## Implementation Details

The implementation focuses on tensor parallelism, sharding key model parameters across devices:

- Attention query, key, value projections are sharded across the heads dimension
- MLP intermediate projections are sharded across the intermediate dimension
- Embedding and LM head are sharded appropriately

The parameter loading pipeline includes:

1. Loading weights incrementally from safetensors files to reduce memory usage
2. Mapping PyTorch parameter names to Flax parameter structure
3. Handling dimension mismatches and transpositions
4. Special handling for GQA to ensure proper head distribution

## Usage

### Running the Model

```bash
python qwen25_tp_model.py --weights_dir /path/to/qwen25-weights --mesh_shape 1,8 --prompt "Once upon a time"
```

### Arguments

- `--weights_dir`: Path to the directory containing Qwen2.5 weights
- `--mesh_shape`: Device mesh shape for tensor parallelism (e.g., "1,8", "2,4")
- `--prompt`: Text prompt for generation
- `--max_length`: Maximum number of tokens to generate
- `--dtype`: Data type (float32, float16, bfloat16)
- `--temperature`: Sampling temperature
- `--top_p`: Nucleus sampling probability
- `--deterministic`: Use deterministic decoding (no sampling)
- `--test`: Run a basic test without generation

### Code Example

```python
from tt_xla.tests.jax.models.qwen25.qwen25_tp_model import (
    TPQwenForCausalLM, 
    load_qwen_model, 
    create_mesh_from_string,
    generate_text
)
from transformers import AutoTokenizer

# Create a device mesh
mesh_shape = "1,8"
mesh = create_mesh_from_string(mesh_shape)

# Load model and tokenizer
with mesh:
    model, params = load_qwen_model(
        weights_dir="/path/to/qwen25-weights",
        mesh_shape=mesh_shape,
        dtype=jnp.bfloat16,
    )
tokenizer = AutoTokenizer.from_pretrained("/path/to/qwen25-weights")

# Generate text
prompt = "Artificial intelligence is"
with mesh:
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        params=params,
        prompt=prompt,
        max_length=100,
    )
print(output)
```

## Tensor Parallel Design

This implementation uses JAX's GSPMD for tensor parallelism:

1. **Sharding Specifications**: Uses `PartitionSpec` to annotate how tensors should be partitioned.
2. **Explicit Constraints**: Uses `with_sharding_constraint` to enforce sharding decisions.
3. **Parameter Placement**: Explicitly places parameters on the right devices during loading.
4. **Mesh Context**: All operations are performed within the mesh context.

### Attention Sharding

The attention mechanism is sharded as follows:

- Query heads are sharded across model dimension (more heads)
- Key/Value heads are sharded efficiently for GQA (fewer heads)
- Rotary embeddings are applied with sharding awareness

### MLP Sharding

The MLP (SwiGLU) is sharded as follows:

- Gate and up projections are sharded on the output dimension
- Down projection is sharded on the input dimension

## Performance Considerations

- Using bfloat16 precision is recommended for optimal performance
- Memory usage scales with model size and sequence length
- Generation speed depends on the hardware and mesh configuration

## License

This implementation follows the license of the original Qwen2.5 model. 