"""Qwen-2.5-7B model implementation with tensor parallelism support.

Memory-optimized version for reduced RAM usage.
"""

import time
import os
import math
import logging
import sys
import traceback
from functools import partial, wraps
from typing import Any, Dict, Optional, Tuple, Union, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.core
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.lax import with_sharding_constraint
import numpy as np

# Define local function to replace the import
def load_params_from_checkpoint(checkpoint_dir, params_shape=None):
    """Load parameters from a checkpoint directory."""
    import os
    import pickle
    import json
    
    # Check if the directory exists
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")
    
    # Look for parameter files in the directory
    param_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pickle') or f.endswith('.pkl')]
    
    # Check for safetensors format if no pickle files found
    if not param_files:
        logger.info("No pickle files found, checking for safetensors files...")
        safetensor_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.safetensors')])
        index_file = os.path.join(checkpoint_dir, "model.safetensors.index.json")
        
        if safetensor_files and os.path.exists(index_file):
            try:
                # Import safetensors 
                from safetensors import safe_open
                
                logger.info(f"Loading model from safetensors files: {len(safetensor_files)} files found")
                
                # Load index to get the structure
                with open(index_file, 'r') as f:
                    index = json.load(f)
                
                # Create a dictionary to store parameters
                params = {"params": {}}
                
                # Mapping from PyTorch parameter names to JAX parameter structure
                def get_param_path(name):
                    """Map a PyTorch parameter name to its JAX/Flax path."""
                    # Direct mappings
                    direct_mapping = {
                        "model.embed_tokens.weight": ("embed_tokens", "embedding"),
                        "model.norm.weight": ("norm", "scale"),
                        "model.norm.bias": ("norm", "bias"),
                        "lm_head.weight": ("lm_head", "kernel"),
                        "lm_head.bias": ("lm_head", "bias"),
                    }
                    
                    if name in direct_mapping:
                        return direct_mapping[name]
                    
                    # Handle layer parameters
                    import re
                    
                    # Patterns for layer parameters
                    layer_norm_pattern = r"model\.layers\.(\d+)\.(input|post_attention)_layernorm\.weight"
                    layer_norm_bias_pattern = r"model\.layers\.(\d+)\.(input|post_attention)_layernorm\.bias"
                    attention_pattern = r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.(weight|bias)"
                    mlp_pattern = r"model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.(weight|bias)"
                    
                    # Handle layer norms
                    layer_norm_match = re.match(layer_norm_pattern, name)
                    if layer_norm_match:
                        layer_idx = int(layer_norm_match.group(1))
                        norm_type = layer_norm_match.group(2)
                        layer_name = f"layers_{layer_idx}"
                        norm_name = "input_layernorm" if norm_type == "input" else "post_attention_layernorm"
                        return (layer_name, norm_name, "scale")
                    
                    layer_norm_bias_match = re.match(layer_norm_bias_pattern, name)
                    if layer_norm_bias_match:
                        layer_idx = int(layer_norm_bias_match.group(1))
                        norm_type = layer_norm_bias_match.group(2)
                        layer_name = f"layers_{layer_idx}"
                        norm_name = "input_layernorm" if norm_type == "input" else "post_attention_layernorm"
                        return (layer_name, norm_name, "bias")
                    
                    # Handle attention parameters
                    attn_match = re.match(attention_pattern, name)
                    if attn_match:
                        layer_idx = int(attn_match.group(1))
                        proj_type = attn_match.group(2)
                        param_type = attn_match.group(3)
                        layer_name = f"layers_{layer_idx}"
                        proj_name = f"{proj_type}_proj"
                        param_name = "kernel" if param_type == "weight" else "bias"
                        return (layer_name, "self_attn", proj_name, param_name)
                    
                    # Handle MLP parameters
                    mlp_match = re.match(mlp_pattern, name)
                    if mlp_match:
                        layer_idx = int(mlp_match.group(1))
                        proj_type = mlp_match.group(2)
                        param_type = mlp_match.group(3)
                        layer_name = f"layers_{layer_idx}"
                        proj_name = f"{proj_type}_proj"
                        param_name = "kernel" if param_type == "weight" else "bias"
                        return (layer_name, "mlp", proj_name, param_name)
                    
                    # Log any unhandled parameter patterns
                    logger.warning(f"Unknown parameter pattern: {name}")
                    return None
                
                def transpose_if_needed(name, param):
                    """Transpose weight matrices if needed based on the parameter name."""
                    # Special case for embedding weights
                    if "embed_tokens.weight" in name:
                        # Do not transpose embedding weights
                        return param
                    
                    # Other attention and MLP weights need to be transposed
                    if "weight" in name and ("proj" in name or "lm_head" in name):
                        # For attention and MLP weight matrices
                        return jnp.transpose(param)
                    
                    return param
                
                # Process each safetensor file
                file_count = 0
                for safetensor_file in safetensor_files:
                    file_count += 1
                    file_path = os.path.join(checkpoint_dir, safetensor_file)
                    logger.info(f"Loading weights from {safetensor_file} ({file_count}/{len(safetensor_files)})")
                    
                    with safe_open(file_path, framework="numpy") as f:
                        for key in f.keys():
                            # Get parameter path
                            param_path = get_param_path(key)
                            if param_path is None:
                                continue
                            
                            # Get parameter and convert to JAX array
                            param = f.get_tensor(key)
                            param = jnp.array(param)
                            
                            # Transpose if needed
                            param = transpose_if_needed(key, param)
                            
                            # Add to the parameter dictionary with the correct nested structure
                            current_dict = params["params"]
                            for path_part in param_path[:-1]:
                                if path_part not in current_dict:
                                    current_dict[path_part] = {}
                                current_dict = current_dict[path_part]
                            
                            current_dict[param_path[-1]] = param
                
                logger.info(f"Successfully loaded parameters from {len(safetensor_files)} safetensors files")
                return params
                
            except (ImportError, Exception) as e:
                logger.error(f"Error loading safetensors: {e}")
                raise ValueError(f"Could not load safetensors files: {e}")
        else:
            raise ValueError(f"No parameter files found in {checkpoint_dir}")
    
    # Sort files for consistent loading
    param_files.sort()
    
    # Load parameters from the first file
    params_path = os.path.join(checkpoint_dir, param_files[0])
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    
    return params

# Import tensor parallel components
try:
    from tests.jax.models.qwen25.tensor_parallel import TensorParallelDense
except ImportError:
    try:
        from tt_xla.tests.jax.models.qwen25.tensor_parallel import TensorParallelDense
    except ImportError:
        # Fall back to a compatible implementation
        class TensorParallelDense(nn.Module):
            """Compatibility implementation of TensorParallelDense."""
            features: int
            use_bias: bool = True
            dtype: jnp.dtype = jnp.float32
            param_dtype: jnp.dtype = jnp.float32
            kernel_init: Any = nn.initializers.lecun_normal()
            bias_init: Any = nn.initializers.zeros
            precision: Optional[Union[str, jax.lax.Precision]] = None
            name: Optional[str] = None
            
            @nn.compact
            def __call__(self, inputs, precision=None):
                """Apply the dense layer."""
                precision = precision or self.precision
                kernel = self.param(
                    'kernel',
                    self.kernel_init,
                    (inputs.shape[-1], self.features),
                    self.param_dtype
                ).astype(self.dtype)
                
                y = jnp.matmul(inputs, kernel, precision=precision)
                
                if self.use_bias:
                    bias = self.param(
                        'bias',
                        self.bias_init,
                        (self.features,),
                        self.param_dtype
                    ).astype(self.dtype)
                    y = y + bias
                
                return y

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set up logging for shape debugging
logger = logging.getLogger("qwen25_model")

# Global config for debug mode
_DEBUG_ENABLED = False

# Function to enable/disable detailed shape logging
def set_debug_logging(enabled=True):
    """Enable or disable detailed tensor shape logging."""
    global _DEBUG_ENABLED
    _DEBUG_ENABLED = enabled
    if enabled:
        logger.setLevel(logging.DEBUG)
        logger.info("Detailed tensor shape logging enabled")
    else:
        logger.setLevel(logging.INFO)
        logger.info("Detailed tensor shape logging disabled")

# Error-handling wrapper for model methods
def error_handler(func):
    """Decorator to handle errors during model execution and enable debugging."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            global _DEBUG_ENABLED
            # Enable debug mode on error if not already enabled
            if not _DEBUG_ENABLED:
                logger.error("Error detected! Enabling detailed logging for diagnostics.")
                set_debug_logging(True)
            
            # Get class/method information
            if hasattr(args[0], '__class__'):
                cls_name = args[0].__class__.__name__
                func_name = func.__name__
                logger.error(f"Error in {cls_name}.{func_name}: {str(e)}")
            else:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                
            # Log additional diagnostic information
            logger.error("Function arguments:")
            for i, arg in enumerate(args[1:], 1):  # Skip self
                if hasattr(arg, 'shape'):
                    logger.error(f"  arg[{i}] (shape={arg.shape}, dtype={arg.dtype})")
                else:
                    logger.error(f"  arg[{i}] ({type(arg)})")
            
            for k, v in kwargs.items():
                if hasattr(v, 'shape'):
                    logger.error(f"  {k}={v.shape}")
                else:
                    logger.error(f"  {k}={type(v)}")
            
            # Re-raise with original traceback
            logger.error("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            raise
    return wrapper

# Shape debugging functions
def log_tensor_shape(tensor, name, level="DEBUG"):
    """Log the shape of a tensor with appropriate detail level."""
    if tensor is None:
        logger.debug(f"{name}: None")
        return
    
    shape_str = f"{name}: shape={tensor.shape}, dtype={tensor.dtype}"
    
    # Only try to access concrete values if not in tracing mode
    if hasattr(tensor, 'size') and tensor.size < 10:  # For very small tensors, show actual values
        # Skip .tolist() for JAX tracers which aren't compatible with it
        if not isinstance(tensor, jax.core.Tracer):
            try:
                shape_str += f", values={tensor.flatten().tolist()}"
            except:
                # If any error occurs, just skip showing values
                pass
    
    logger.debug(shape_str)

def validate_attention_shapes(query, key, value, config):
    """Validate shapes consistency for attention operation."""
    batch_size, num_q_heads, q_len, head_dim = query.shape
    k_batch, num_kv_heads, kv_len, k_head_dim = key.shape
    v_batch, v_num_kv_heads, v_kv_len, v_head_dim = value.shape
    
    expected_q_heads = config["num_attention_heads"]
    expected_kv_heads = config.get("num_key_value_heads", expected_q_heads)
    
    error_messages = []
    if num_q_heads != expected_q_heads:
        error_messages.append(f"Query heads ({num_q_heads}) don't match expected ({expected_q_heads})")
    if num_kv_heads != expected_kv_heads:
        error_messages.append(f"Key heads ({num_kv_heads}) don't match expected ({expected_kv_heads})")
    if head_dim != k_head_dim or head_dim != v_head_dim:
        error_messages.append(f"Head dimensions inconsistent: q={head_dim}, k={k_head_dim}, v={v_head_dim}")
    
    if error_messages:
        raise ValueError("Attention shape validation failed:\n" + "\n".join(error_messages))

def validate_qwen_configuration(config):
    """Validate the complete model configuration for consistency and compatibility."""
    # Check essential parameters
    required_fields = ["hidden_size", "num_attention_heads", "num_hidden_layers"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required configuration field '{field}' is missing")
    
    # Validate attention configuration
    hidden_size = config["hidden_size"]
    num_heads = config["num_attention_heads"]
    
    # Check divisibility for head dimension
    if hidden_size % num_heads != 0:
        raise ValueError(
            f"Hidden size ({hidden_size}) must be divisible by number of attention heads ({num_heads})"
        )
    
    # Validate grouped query attention if enabled
    if "num_key_value_heads" in config:
        num_kv_heads = config["num_key_value_heads"]
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"Number of attention heads ({num_heads}) must be divisible by "
                f"number of key/value heads ({num_kv_heads})"
            )
    
    # Check other parameters for consistency
    head_dim = hidden_size // num_heads
    if config.get("head_dim", head_dim) != head_dim:
        raise ValueError(
            f"Explicit head_dim ({config['head_dim']}) doesn't match calculated "
            f"from hidden_size/num_heads ({head_dim})"
        )
        
    # Return validated and normalized configuration
    normalized_config = config.copy()
    if "head_dim" not in normalized_config:
        normalized_config["head_dim"] = head_dim
    if "num_key_value_heads" not in normalized_config:
        normalized_config["num_key_value_heads"] = num_heads
        
    return normalized_config

# Rotation helpers for applying rotary position embeddings
@error_handler
def apply_rotary_emb(
    query_states: jnp.ndarray,
    key_states: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply rotary positional embeddings to the query and key tensors.
    
    Args:
        query_states: Query state tensor.
        key_states: Key state tensor.
        cos: Cosine of positions.
        sin: Sine of positions.
        
    Returns:
        query_states and key_states with rotary embeddings applied
    """
    # Log input shapes for debugging
    log_tensor_shape(query_states, "apply_rotary_emb input query_states")
    log_tensor_shape(key_states, "apply_rotary_emb input key_states")
    log_tensor_shape(cos, "apply_rotary_emb input cos")
    log_tensor_shape(sin, "apply_rotary_emb input sin")
    
    # Get appropriate dimensions
    q_dim = query_states.shape[-1] // 2
    k_dim = key_states.shape[-1] // 2
    
    # Split into real and imaginary parts
    query_states_r = query_states.reshape(*query_states.shape[:-1], -1, 2)
    query_real, query_imag = query_states_r[..., 0], query_states_r[..., 1]
    
    # Calculate the actual dimension for the reshape
    dim_half = query_real.shape[-1]
    
    # Make sure the cos/sin are properly shaped for the operation
    # The key issue: We need to ensure cos/sin have correct size before reshaping
    # instead of hardcoding the dimensions
    cos_seq_len = cos.shape[0]
    cos_feature_dim = cos.shape[-1] if len(cos.shape) > 1 else dim_half
    
    # If cos/sin last dimension != dim_half, we need to slice it
    if cos_feature_dim > dim_half:
        logger.info(f"Slicing cos/sin feature dim from {cos_feature_dim} to {dim_half}")
        cos = cos[..., :dim_half]
        sin = sin[..., :dim_half]
        cos_feature_dim = dim_half
    
    # Now reshape using the actual dimensions, preserving the original shape structure
    # Instead of hardcoding the shape, we'll adapt to the input shape
    if len(cos.shape) == 2:
        # If cos shape is (seq_len, dim)
        cos_for_q = cos.reshape(1, cos_seq_len, 1, cos_feature_dim)
        sin_for_q = sin.reshape(1, cos_seq_len, 1, cos_feature_dim)
    elif len(cos.shape) == 3:
        # If cos shape is (batch, seq_len, dim) or similar
        # Preserve the first dimension, adjust the remaining ones
        cos_for_q = cos.reshape(cos.shape[0], cos.shape[1], 1, cos.shape[2])
        sin_for_q = sin.reshape(sin.shape[0], sin.shape[1], 1, sin.shape[2])
    else:
        # For other shapes, log the shape and attempt a reasonable reshape
        logger.warning(f"Unusual cos shape: {cos.shape}, attempting flexible reshape")
        # Just add a dimension for head at position -2
        shape_list = list(cos.shape)
        shape_list.insert(len(shape_list)-1, 1)
        cos_for_q = cos.reshape(shape_list)
        sin_for_q = sin.reshape(shape_list)
    
    # Log shapes for debugging
    logger.debug(f"query_real shape={query_real.shape}, cos_for_q shape={cos_for_q.shape}")
    
    # Apply rotation to query
    query_out_real = query_real * cos_for_q - query_imag * sin_for_q
    query_out_imag = query_real * sin_for_q + query_imag * cos_for_q
    
    # Combine and reshape back
    query_out = jnp.stack([query_out_real, query_out_imag], axis=-1)
    query_out = query_out.reshape(*query_states.shape)
    
    # Similarly process key states
    # Make sure cos/sin dimensions match key states
    if cos.shape[-1] > k_dim:
        # Slice again for key dimension if needed
        logger.info(f"Slicing cos/sin from shape {cos.shape} to match key dimension {k_dim}")
        cos_for_key = cos[..., :k_dim]
        sin_for_key = sin[..., :k_dim]
    else:
        cos_for_key = cos
        sin_for_key = sin
    
    # Split key into real and imaginary parts
    key_states_r = key_states.reshape(*key_states.shape[:-1], -1, 2)
    key_real, key_imag = key_states_r[..., 0], key_states_r[..., 1]
    
    # Calculate actual dimensions for key reshape
    key_dim_half = key_real.shape[-1]
    key_cos_feature_dim = cos_for_key.shape[-1] if len(cos_for_key.shape) > 1 else key_dim_half
    
    # If cos/sin last dimension != key_dim_half, we need to slice it
    if key_cos_feature_dim > key_dim_half:
        logger.info(f"Slicing cos/sin key feature dim from {key_cos_feature_dim} to {key_dim_half}")
        cos_for_key = cos_for_key[..., :key_dim_half]
        sin_for_key = sin_for_key[..., :key_dim_half]
        key_cos_feature_dim = key_dim_half
    
    # Reshape cos/sin for broadcasting with key - use same approach as for query
    if len(cos_for_key.shape) == 2:
        # If cos shape is (seq_len, dim)
        cos_for_k = cos_for_key.reshape(1, cos_for_key.shape[0], 1, cos_for_key.shape[1])
        sin_for_k = sin_for_key.reshape(1, sin_for_key.shape[0], 1, sin_for_key.shape[1])
    elif len(cos_for_key.shape) == 3:
        # If cos shape is (batch, seq_len, dim) or similar
        cos_for_k = cos_for_key.reshape(cos_for_key.shape[0], cos_for_key.shape[1], 1, cos_for_key.shape[2])
        sin_for_k = sin_for_key.reshape(sin_for_key.shape[0], sin_for_key.shape[1], 1, sin_for_key.shape[2])
    else:
        # For other shapes, log the shape and attempt a reasonable reshape
        logger.warning(f"Unusual cos_for_key shape: {cos_for_key.shape}, attempting flexible reshape")
        # Just add a dimension for head at position -2
        shape_list = list(cos_for_key.shape)
        shape_list.insert(len(shape_list)-1, 1)
        cos_for_k = cos_for_key.reshape(shape_list)
        sin_for_k = sin_for_key.reshape(shape_list)
    
    # Log shapes for debugging
    logger.debug(f"key_real shape={key_real.shape}, cos_for_k shape={cos_for_k.shape}")
    
    # Apply rotation to key
    key_out_real = key_real * cos_for_k - key_imag * sin_for_k
    key_out_imag = key_real * sin_for_k + key_imag * cos_for_k
    
    # Combine and reshape back
    key_out = jnp.stack([key_out_real, key_out_imag], axis=-1)
    key_out = key_out.reshape(*key_states.shape)
    
    logger.debug(f"apply_rotary_emb: output query_out shape={query_out.shape}")
    logger.debug(f"apply_rotary_emb: output key_out shape={key_out.shape}")
    
    return query_out, key_out

def generalized_attention(
    query, key, value, mask=None, precision=None, mesh=None, 
    use_memory_efficient=False, config=None
):
    """Unified attention implementation that handles both standard and memory-efficient modes.
    
    Args:
        query: Query tensor [batch, num_q_heads, q_len, head_dim]
        key: Key tensor [batch, num_kv_heads, kv_len, head_dim]
        value: Value tensor [batch, num_kv_heads, kv_len, head_dim]
        mask: Optional attention mask
        precision: Optional precision configuration for matrix operations
        mesh: Optional mesh for tensor parallelism
        use_memory_efficient: Whether to use memory-efficient implementation
        config: Model configuration for validation
        
    Returns:
        Attention output tensor [batch, num_q_heads, q_len, head_dim]
    """
    batch_size, num_q_heads, q_len, head_dim = query.shape
    _, num_kv_heads, kv_len, _ = key.shape
    
    log_tensor_shape(query, "generalized_attention input query")
    log_tensor_shape(key, "generalized_attention input key")
    log_tensor_shape(value, "generalized_attention input value")
    
    # Handle case where num_q_heads != num_kv_heads (Grouped Query Attention)
    if num_q_heads != num_kv_heads:
        # Calculate repeat factor
        repeats = num_q_heads // num_kv_heads
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"Number of query heads ({num_q_heads}) must be divisible by "
                f"number of key/value heads ({num_kv_heads})"
            )
        
        # Repeat key and value tensors along the head dimension
        key = jnp.repeat(key, repeats=repeats, axis=1)
        value = jnp.repeat(value, repeats=repeats, axis=1)
        
        log_tensor_shape(key, "generalized_attention after repeat key")
        log_tensor_shape(value, "generalized_attention after repeat value")
    
    # Choose between memory-efficient or standard implementation
    if use_memory_efficient:
        # Flash attention prefers F32 accumulation
        dtype = query.dtype
        acc_dtype = jnp.float32
        
        # Scale query by 1/sqrt(head_dim)
        scale = jnp.sqrt(head_dim).astype(acc_dtype)
        query = query / scale
        
        # Pre-process the attention mask if provided
        if mask is not None:
            # Extract mask shape for dynamic slicing later
            mask_b, mask_h, mask_q, mask_k = mask.shape
            
            # Check that mask is compatible with our attention shape
            if mask_h != 1 and mask_h != num_q_heads:
                logger.warning(f"Mask head dim ({mask_h}) does not match query heads ({num_q_heads}). "
                             f"Assuming broadcasting is intended.")
            
            if mask_q != 1 and mask_q != q_len:
                logger.warning(f"Mask query length ({mask_q}) does not match query length ({q_len}). "
                             f"This may cause unexpected behavior.")
            
            logger.debug(f"Using attention mask with shape {mask.shape}")
        
        # Batched matrix multiplication for Q·K^T using scan
        def attention_step(_, q_idx):
            # Get current query chunk
            q = jax.lax.dynamic_slice(
                query, (0, 0, q_idx, 0), 
                (batch_size, num_q_heads, 1, head_dim)
            )
            q = jnp.squeeze(q, axis=2)  # Remove seq_len dimension (now 1)
            
            # Compute attention scores for this query with all keys
            attn_weights = jnp.einsum("bhd,bhkd->bhk", q, key, precision=precision)
            
            # Apply mask if provided
            if mask is not None:
                # Use dynamic_slice for the mask instead of NumPy indexing
                # Get the slice of the mask for this query position
                if mask_q > 1:
                    # If mask has separate entries for each query position
                    # We need to slice out the current position
                    start_indices = jnp.array([0, 0, q_idx, 0])
                    slice_sizes = jnp.array([mask_b, mask_h, 1, mask_k])
                    mask_slice = jax.lax.dynamic_slice(
                        mask, start_indices, slice_sizes
                    )
                    # Reshape to remove the singleton dimension
                    mask_slice = jnp.squeeze(mask_slice, axis=2)
                else:
                    # If mask is the same for all query positions (mask_q == 1)
                    # We can just use the entire mask
                    mask_slice = mask[:, :, 0, :]
                
                # Add the mask to attention weights
                attn_weights = attn_weights + mask_slice
            
            # Normalize with softmax
            attn_weights = attn_weights.astype(acc_dtype)  # Convert to accumulation dtype
            attn_weights = jax.nn.softmax(attn_weights, axis=-1)
            
            # Apply attention weights to values
            attn_output = jnp.einsum("bhk,bhkd->bhd", attn_weights, value, precision=precision)
            return None, attn_output
        
        # Scan over the sequence length dimension to save memory
        _, attn_outputs = jax.lax.scan(
            attention_step, None, jnp.arange(q_len)
        )
        
        # Reshape result back to [batch, heads, seq_len, head_dim]
        attn_outputs = jnp.transpose(attn_outputs, (1, 0, 2, 3))
        
        # Convert back to original dtype
        attn_output = attn_outputs.astype(dtype)
    else:
        # Standard implementation
        # Calculate scaled dot-product attention
        # QK^T / sqrt(head_dim) -> [batch, num_heads, seq_len, seq_len]
        scale = math.sqrt(head_dim)
        attention_scores = jnp.matmul(query, jnp.swapaxes(key, -1, -2)) / scale
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores + mask
            
        # Apply softmax to get attention weights
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        
        # Apply attention weights to values
        # [batch, num_heads, seq_len, seq_len] @ [batch, num_heads, seq_len, head_dim]
        # -> [batch, num_heads, seq_len, head_dim]
        attn_output = jnp.matmul(attention_weights, value)
    
    log_tensor_shape(attn_output, "generalized_attention output")
    return attn_output

def memory_efficient_attention(query, key, value, mask=None, precision=None, mesh=None):
    """Memory-efficient multi-head attention implementation (legacy wrapper)."""
    return generalized_attention(
        query, key, value, mask=mask, precision=precision, mesh=mesh,
        use_memory_efficient=True
    )

class QwenAttention(nn.Module):
    """Memory-optimized multi-head attention for Qwen2.5 model."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        config = self.config
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = config["head_dim"] if "head_dim" in config else self.hidden_size // self.num_heads
        
        # Note: Qwen2.5 uses KV grouping - fewer key/value heads than query heads
        self.num_kv_heads = config.get("num_key_value_heads", self.num_heads)
        self.kv_dim = self.num_kv_heads * self.head_dim
        
        # Log setup information
        logger.info(f"Setting up QwenAttention with: hidden_size={self.hidden_size}, "
                   f"num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, "
                   f"head_dim={self.head_dim}")
        
        # Attention projections without partitioning
        self.q_proj = nn.Dense(
            self.hidden_size,
            kernel_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="q_proj",
        )
        
        self.k_proj = nn.Dense(
            self.kv_dim,
            kernel_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="k_proj",
        )
        
        self.v_proj = nn.Dense(
            self.kv_dim,
            kernel_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            name="v_proj",
        )
        
        self.o_proj = nn.Dense(
            self.hidden_size,
            kernel_init=nn.initializers.normal(stddev=0.02),
            use_bias=False,
            dtype=self.dtype,
            name="o_proj",
        )
        
        # Rotary embeddings
        self.rope_theta = config.get("rope_theta", 10000.0)
        self.max_position_embeddings = config.get("max_position_embeddings", 4096)
        
        # Use memory-efficient attention if specified
        self.use_memory_efficient_attention = config.get("use_memory_efficient_attention", False)
        
        # Set attention function to use generalized implementation
        self.attention_fn = partial(
            generalized_attention,
            use_memory_efficient=self.use_memory_efficient_attention,
            config=self.config
        )
        
        logger.info(f"QwenAttention initialized with memory_efficient_attention={self.use_memory_efficient_attention}")
    
    @error_handler
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        cos: jnp.ndarray = None,
        sin: jnp.ndarray = None,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        output_attentions: bool = False,
        mesh: Optional[Mesh] = None,
        **kwargs  # Handle any additional arguments
    ) -> jnp.ndarray:
        # Log input shapes for debugging
        log_tensor_shape(hidden_states, "QwenAttention input hidden_states")
        log_tensor_shape(attention_mask, "QwenAttention input attention_mask")
        log_tensor_shape(cos, "QwenAttention input cos")
        log_tensor_shape(sin, "QwenAttention input sin")
        
        bsz, seqlen, _ = hidden_states.shape
        
        # Project to query, key, value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        log_tensor_shape(query_states, "QwenAttention after projection query")
        log_tensor_shape(key_states, "QwenAttention after projection key")
        log_tensor_shape(value_states, "QwenAttention after projection value")
        
        # Reshape and apply rope
        query_states = query_states.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        key_states = key_states.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        value_states = value_states.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        
        log_tensor_shape(query_states, "QwenAttention after reshape query")
        log_tensor_shape(key_states, "QwenAttention after reshape key")
        log_tensor_shape(value_states, "QwenAttention after reshape value")
        
        # Apply rotary embeddings if sin and cos are provided
        if cos is not None and sin is not None:
            query_states, key_states = apply_rotary_emb(query_states, key_states, cos, sin)
            log_tensor_shape(query_states, "QwenAttention after rotary emb query")
            log_tensor_shape(key_states, "QwenAttention after rotary emb key")
        
        # Transpose for attention computation
        # [batch, seq, heads, dim] -> [batch, heads, seq, dim]
        query_states = jnp.transpose(query_states, (0, 2, 1, 3))
        key_states = jnp.transpose(key_states, (0, 2, 1, 3))
        value_states = jnp.transpose(value_states, (0, 2, 1, 3))
        
        log_tensor_shape(query_states, "QwenAttention after transpose query")
        log_tensor_shape(key_states, "QwenAttention after transpose key")
        log_tensor_shape(value_states, "QwenAttention after transpose value")
        
        # Validate attention shapes before proceeding
        try:
            validate_attention_shapes(
                query_states, key_states, value_states, self.config
            )
        except ValueError as e:
            logger.error(f"Attention shape validation error: {e}")
            logger.error(f"This often indicates a mismatch in the number of attention heads or their dimensions.")
            raise
        
        # Process attention mask if provided
        attn_mask = None
        if attention_mask is not None:
            # Make sure attention mask has the right shape for the attention function
            # Expected shape: [batch, heads, q_length, kv_length]
            if len(attention_mask.shape) == 4:
                # Already in the right shape
                attn_mask = attention_mask
            elif len(attention_mask.shape) == 2:
                # Expand from [batch, length] to [batch, 1, 1, length]
                attn_mask = attention_mask[:, None, None, :]
                # Convert from additive (0/1) to large negative for softmax
                attn_mask = (1.0 - attn_mask) * jnp.finfo(self.dtype).min
            else:
                # Interpret mask shape
                attn_mask = attention_mask
            
            log_tensor_shape(attn_mask, "QwenAttention processed mask")
        
        # Compute attention
        attn_output = self.attention_fn(
            query_states,
            key_states,
            value_states,
            mask=attn_mask,
            mesh=mesh,
        )
        
        log_tensor_shape(attn_output, "QwenAttention after attention computation")
        
        # Reshape to [batch, seq, hidden_size]
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(bsz, seqlen, self.hidden_size)
        
        log_tensor_shape(attn_output, "QwenAttention after reshape attention output")
        
        # Apply output projection
        attn_output = self.o_proj(attn_output)
        
        log_tensor_shape(attn_output, "QwenAttention final output")
        
        outputs = (attn_output,)
        
        if output_attentions:
            # We don't store attention weights in the current implementation
            # Just return a dummy tensor
            attention_weights = jnp.zeros((bsz, self.num_heads, seqlen, seqlen), dtype=self.dtype)
            outputs = outputs + (attention_weights,)
        
        return outputs[0] if len(outputs) == 1 else outputs


class QwenMLP(nn.Module):
    """Memory-optimized MLP module for Qwen2.5 model."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        config = self.config
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config.get("intermediate_size", 4 * self.hidden_size)
        
        # MLP projections without partitioning
        self.gate_proj = nn.Dense(
            self.intermediate_size,
            kernel_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            use_bias=False,
            name="gate_proj",
        )
        
        self.up_proj = nn.Dense(
            self.intermediate_size,
            kernel_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            use_bias=False,
            name="up_proj",
        )
        
        self.down_proj = nn.Dense(
            self.hidden_size,
            kernel_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            use_bias=False,
            name="down_proj",
        )
    
    def __call__(self, x: jnp.ndarray, mesh: Optional[Mesh] = None) -> jnp.ndarray:
        # Apply projections with precision control
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # Apply SwiGLU activation
        hidden_states = gate * jax.nn.silu(up)
        
        # Apply down projection
        down = self.down_proj(hidden_states)
        
        return down


class QwenDecoderLayer(nn.Module):
    """Memory-optimized Transformer decoder layer for Qwen2.5 model."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        config = self.config
        self.hidden_size = config["hidden_size"]
        
        # Layer normalization
        self.input_layernorm = nn.LayerNorm(
            epsilon=config.get("layer_norm_epsilon", 1e-5),
            dtype=self.dtype,
            use_bias=False,
            name="input_layernorm",
        )
        self.post_attention_layernorm = nn.LayerNorm(
            epsilon=config.get("layer_norm_epsilon", 1e-5),
            dtype=self.dtype,
            use_bias=False,
            name="post_attention_layernorm",
        )
        
        # Self-attention and MLP modules
        self.self_attn = QwenAttention(config, dtype=self.dtype)
        self.mlp = QwenMLP(config, dtype=self.dtype)
        
    @error_handler
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        output_attentions: bool = False,
        mesh: Optional[Mesh] = None,
    ) -> Union[Tuple[jnp.ndarray, ...], jnp.ndarray]:
        # Log input shapes for debugging
        log_tensor_shape(hidden_states, "QwenDecoderLayer input hidden_states")
        log_tensor_shape(attention_mask, "QwenDecoderLayer input attention_mask")
        log_tensor_shape(position_ids, "QwenDecoderLayer input position_ids")
        
        # Self-attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Prepare position embeddings for rotary attention if needed
        cos = None
        sin = None
        
        try:
            if position_ids is not None:
                # Generate sinusoidal position embeddings
                logger.debug(f"Generating rotary embeddings with position_ids shape={position_ids.shape}")
                
                # Use a fixed max position instead of dynamically computing it
                # This avoids JAX concretization errors during JIT compilation
                MAX_SEQUENCE_LENGTH = 4096  # Large enough for most use cases
                
                # Get dimensions from attention module
                head_dim = self.self_attn.head_dim
                rope_theta = self.self_attn.rope_theta
                logger.debug(f"Using head_dim={head_dim}, rope_theta={rope_theta}")
                
                # Generate frequencies exactly matching the head dimension
                feature_dim = head_dim
                freqs = 1.0 / (rope_theta ** (jnp.arange(0, feature_dim // 2, dtype=jnp.float32) / (feature_dim // 2)))
                
                # Use fixed positions instead of dynamic max_pos
                t = jnp.arange(MAX_SEQUENCE_LENGTH, dtype=jnp.float32)
                
                # Matrix of [positions, freqs]
                freqs = jnp.outer(t, freqs)
                
                # Create embeddings that match the expected format in apply_rotary_emb
                emb = jnp.repeat(freqs[..., None], 2, axis=-1)
                emb = emb.reshape(freqs.shape[0], -1)
                
                # Log the intended shape for debugging
                logger.debug(f"Embedding dimension: {emb.shape}")
                
                # Generate cos and sin components
                cos = jnp.cos(emb)
                sin = jnp.sin(emb)
                
                # Log the shapes to make sure they're correct
                logger.debug(f"Generated cos with shape={cos.shape}, sin with shape={sin.shape}")
                
                # Always slice embeddings to get only what we need for the current positions
                # This ensures we don't waste memory or computation on unused positions
                cos = jnp.take(cos, position_ids, axis=0)
                sin = jnp.take(sin, position_ids, axis=0)
                logger.debug(f"Sliced cos to shape={cos.shape}, sin to shape={sin.shape}")
                
                log_tensor_shape(cos, "QwenDecoderLayer generated cos")
                log_tensor_shape(sin, "QwenDecoderLayer generated sin")
            else:
                # No position_ids provided, use sequence length for positions
                logger.warning("No position_ids provided to QwenDecoderLayer, using default positions")
                seq_length = hidden_states.shape[1]
                batch_size = hidden_states.shape[0]
                position_ids = jnp.arange(seq_length, dtype=jnp.int32)[None, :].repeat(batch_size, axis=0)
                
                # Get dimensions from attention module
                head_dim = self.self_attn.head_dim
                rope_theta = self.self_attn.rope_theta
                logger.debug(f"Using head_dim={head_dim}, rope_theta={rope_theta} for default positions")
                
                # Generate frequencies that match exactly the head dimension
                # Using the same fixed max position approach as above
                MAX_SEQUENCE_LENGTH = 4096
                feature_dim = head_dim
                freqs = 1.0 / (rope_theta ** (jnp.arange(0, feature_dim // 2, dtype=jnp.float32) / (feature_dim // 2)))
                
                # Use fixed-size positional indices
                t = jnp.arange(MAX_SEQUENCE_LENGTH, dtype=jnp.float32)
                freqs = jnp.outer(t, freqs)
                
                # Create embeddings that match the expected format
                emb = jnp.repeat(freqs[..., None], 2, axis=-1)
                emb = emb.reshape(freqs.shape[0], -1)
                
                # Log the intended shape for debugging
                logger.debug(f"Default embedding dimension: {emb.shape}")
                
                # Generate sin and cos components
                cos = jnp.cos(emb)
                sin = jnp.sin(emb)
                
                # Slice to get only what we need based on the position_ids
                cos = jnp.take(cos, position_ids, axis=0)
                sin = jnp.take(sin, position_ids, axis=0)
                
                logger.debug(f"Generated default cos with shape={cos.shape}, sin with shape={sin.shape}")
                
                log_tensor_shape(cos, "QwenDecoderLayer generated default cos")
                log_tensor_shape(sin, "QwenDecoderLayer generated default sin")
        except Exception as e:
            logger.error(f"Error generating rotary embeddings: {e}")
            logger.error(f"Attempted with head_dim={getattr(self.self_attn, 'head_dim', 'unknown')}")
            logger.error(f"Position IDs shape: {None if position_ids is None else position_ids.shape}")
            logger.exception("Stack trace:")
            raise
        
        # Apply attention
        try:
            attn_outputs = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                cos=cos,
                sin=sin,
                mesh=mesh,
            )
            
            # Apply residual connection
            hidden_states = attn_outputs[0] if isinstance(attn_outputs, tuple) else attn_outputs
            hidden_states = residual + hidden_states
            
            # MLP block
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states, mesh=mesh)
            hidden_states = residual + hidden_states
            
            outputs = (hidden_states,)
            
            if output_attentions and isinstance(attn_outputs, tuple) and len(attn_outputs) > 1:
                outputs = outputs + (attn_outputs[1],)
            
            log_tensor_shape(hidden_states, "QwenDecoderLayer output hidden_states")
            return outputs
        except Exception as e:
            logger.error(f"Error in QwenDecoderLayer attention or MLP: {e}")
            logger.error(f"hidden_states shape: {hidden_states.shape}")
            logger.error(f"attention_mask shape: {None if attention_mask is None else attention_mask.shape}")
            logger.error(f"cos shape: {None if cos is None else cos.shape}")
            logger.error(f"sin shape: {None if sin is None else sin.shape}")
            logger.exception("Stack trace:")
            raise


class Qwen25ForCausalLM(nn.Module):
    """Memory-optimized Qwen2.5 model for causal language modeling with tensor parallelism support."""
    config: Dict[str, Any]
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        config = self.config
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_hidden_layers"]
        
        # Use standard embedding without partitioning for now
        self.embed_tokens = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            dtype=self.dtype,
            embedding_init=nn.initializers.normal(stddev=0.02),
            name="embed_tokens",
        )
        
        # Create decoder layers
        self.layers = [
            QwenDecoderLayer(config, dtype=self.dtype, name=f"layers_{i}")
            for i in range(self.num_layers)
        ]
        
        # Final layer norm
        self.norm = nn.LayerNorm(
            epsilon=config.get("layer_norm_epsilon", 1e-5),
            dtype=self.dtype,
            use_bias=False,
            name="norm",
        )
        
        # Use TensorParallelDense to support precision parameter
        self.lm_head = TensorParallelDense(
            features=self.vocab_size,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            precision=jax.lax.Precision.DEFAULT,
            name="lm_head",
        )
    
    @error_handler
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: Optional[Tuple[Tuple[jnp.ndarray, jnp.ndarray], ...]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        mesh: Optional[Mesh] = None,
    ) -> Dict[str, jnp.ndarray]:
        # Log input shapes for debugging
        log_tensor_shape(input_ids, "Qwen25ForCausalLM input input_ids")
        log_tensor_shape(attention_mask, "Qwen25ForCausalLM input attention_mask")
        log_tensor_shape(position_ids, "Qwen25ForCausalLM input position_ids")
        
        batch_size, seq_length = input_ids.shape
        
        # Get token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        if mesh is not None:
            # Get the mesh axis names
            mesh_axis_names = mesh.axis_names
            
            # Choose partition spec based on available axes
            if all(axis in mesh_axis_names for axis in ['batch', 'length', 'model']):
                # Use detailed partition spec with batch, length, model
                hidden_sharding = NamedSharding(mesh, P("batch", "length", "model"))
            elif 'data' in mesh_axis_names and len(mesh_axis_names) == 1:
                # For simple 1D mesh with 'data' axis
                hidden_sharding = NamedSharding(mesh, P("data"))
            elif 'data' in mesh_axis_names and 'model' in mesh_axis_names:
                # For 2D mesh with data and model axes
                hidden_sharding = NamedSharding(mesh, P("data", None, "model"))
            else:
                # Fallback if no recognized axis names
                logger.warning(f"Mesh axes {mesh_axis_names} not recognized for hidden states, skipping sharding constraint")
                hidden_sharding = None
                
            if hidden_sharding is not None:
                try:
                    hidden_states = with_sharding_constraint(hidden_states, hidden_sharding)
                except Exception as e:
                    logger.warning(f"Error applying sharding constraint to hidden states: {e}")
        
        # Prepare attention masks for causal attention if not provided
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
        
        # Extend attention mask for attention computation
        extended_attention_mask = jnp.expand_dims(jnp.expand_dims(attention_mask, axis=1), axis=1)
        extended_attention_mask = (1.0 - extended_attention_mask) * jnp.finfo(self.dtype).min
        
        # Generate position ids if not provided
        if position_ids is None:
            # Standard position ids: 0, 1, 2, ... for each sequence
            position_ids = jnp.arange(seq_length, dtype=jnp.int32)[None, :].repeat(batch_size, axis=0)
            
            # If attention_mask is provided, adjust position_ids to account for padding tokens
            if attention_mask is not None:
                # Find the position of the first padding token in each sequence
                # and set all positions after that to zero
                position_ids = position_ids * attention_mask
                
                # Count valid tokens (non-padding) for each sequence
                valid_tokens = jnp.sum(attention_mask, axis=1, keepdims=True)
                
                # Create cumulative sequence for each batch item: [0, 1, 2, ...] but starting from each valid token count
                # This handles continued generation with no padding
                position_offset = jnp.maximum(0, jnp.arange(seq_length, dtype=jnp.int32) - (valid_tokens - 1))
                position_ids = jnp.where(
                    position_offset[None, :] == 0,
                    position_ids,
                    valid_tokens + position_offset[None, :] - 1
                )
            
            logger.debug(f"Generated position_ids with shape: {position_ids.shape}")
        
        log_tensor_shape(position_ids, "Qwen25ForCausalLM processed position_ids")
        log_tensor_shape(attention_mask, "Qwen25ForCausalLM processed attention_mask")
        log_tensor_shape(extended_attention_mask, "Qwen25ForCausalLM extended_attention_mask")
        
        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Process through decoder layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            # Apply decoder layer
            try:
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    attention_mask=extended_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    mesh=mesh,
                )
                
                hidden_states = layer_outputs[0]
                
                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)
            except Exception as e:
                logger.error(f"Error in layer {i}: {e}")
                logger.error(f"Input shapes - hidden_states: {hidden_states.shape}, "
                           f"position_ids: {position_ids.shape}, "
                           f"attention_mask: {extended_attention_mask.shape}")
                raise
        
        # Apply final layer norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        if mesh is not None:
            # Get the mesh axis names
            mesh_axis_names = mesh.axis_names
            
            # Choose partition spec based on available axes
            if all(axis in mesh_axis_names for axis in ['batch', 'length', 'vocab']):
                # Use detailed partition spec with batch, length, vocab
                logits_sharding = NamedSharding(mesh, P("batch", "length", "vocab"))
            elif 'data' in mesh_axis_names and len(mesh_axis_names) == 1:
                # For simple 1D mesh with 'data' axis
                logits_sharding = NamedSharding(mesh, P("data"))
            elif 'data' in mesh_axis_names and 'model' in mesh_axis_names:
                # For 2D mesh with data and model axes - vocab is similar to model dimension
                logits_sharding = NamedSharding(mesh, P("data", None, "model"))
            else:
                # Fallback if no recognized axis names
                logger.warning(f"Mesh axes {mesh_axis_names} not recognized for logits, skipping sharding constraint")
                logits_sharding = None
                
            if logits_sharding is not None:
                try:
                    logits = with_sharding_constraint(logits, logits_sharding)
                except Exception as e:
                    logger.warning(f"Error applying sharding constraint to logits: {e}")
        
        log_tensor_shape(logits, "Qwen25ForCausalLM output logits")
        
        if not return_dict:
            return (logits, all_hidden_states, all_attentions)
        
        return {
            "logits": logits,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
        }


def create_qwen25_model(
    config: Dict[str, Any],
    dtype: jnp.dtype = jnp.float32,
) -> Qwen25ForCausalLM:
    """Create a Qwen2.5 model instance."""
    # Add memory optimization flags if not present
    if "use_memory_efficient_attention" not in config:
        config["use_memory_efficient_attention"] = True
    
    # Validate and normalize configuration
    validated_config = validate_qwen_configuration(config)
    
    logger.info(f"Creating Qwen25ForCausalLM with configuration: "
               f"hidden_size={validated_config['hidden_size']}, "
               f"num_heads={validated_config['num_attention_heads']}, "
               f"num_kv_heads={validated_config.get('num_key_value_heads', validated_config['num_attention_heads'])}, "
               f"num_layers={validated_config['num_hidden_layers']}, "
               f"head_dim={validated_config.get('head_dim', validated_config['hidden_size'] // validated_config['num_attention_heads'])}")
        
    return Qwen25ForCausalLM(config=validated_config, dtype=dtype)


def stream_load_qwen25_model(
    model: Qwen25ForCausalLM,
    weights_dir: str,
    dtype: jnp.dtype = jnp.bfloat16,
) -> Dict[str, Any]:
    """
    Memory-efficient function to load Qwen2.5 model parameters.
    This function is designed to be used with the streaming parameter loading
    implementation in run_model.py.
    
    Args:
        model: The model instance to load parameters into
        weights_dir: Directory containing the model weights
        dtype: Data type for model parameters
        
    Returns:
        config: Model configuration
    """
    import json
    import os
    import re
    
    # Load configuration
    config_path = os.path.join(weights_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Set additional configuration for memory efficiency
    config["use_memory_efficient_attention"] = True
    
    # Validate configuration
    validated_config = validate_qwen_configuration(config)
    
    logger.info(f"Loaded and validated Qwen2.5 configuration from {config_path}")
    
    return validated_config


def load_qwen25_model(
    weights_dir: str,
    mesh_shape: Tuple[int, ...],
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[Qwen25ForCausalLM, Dict[str, Any]]:
    """
    Load a Qwen2.5 model with tensor parallelism from checkpoint.
    
    Args:
        weights_dir: Directory containing the model weights
        mesh_shape: Shape of the device mesh for tensor parallelism
        dtype: Data type for model parameters
        
    Returns:
        model: Initialized model instance
        config: Model configuration
    """
    start_time = time.time()
    
    # Load model configuration
    import json
    import os
    
    config_path = os.path.join(weights_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Set additional configuration for memory efficiency
    config["use_memory_efficient_attention"] = True
    
    # Validate and normalize configuration
    validated_config = validate_qwen_configuration(config)
    
    # Create device mesh for tensor parallelism
    devices = jax.devices()
    
    # Handle different mesh shapes
    if len(mesh_shape) == 1:
        # For 1D mesh, use a single axis name
        mesh = Mesh(np.array(devices), ("data",))
    else:
        # For 2D mesh, reshape devices to match the mesh shape
        devices_reshaped = np.array(devices).reshape(mesh_shape)
        mesh = Mesh(devices_reshaped, ("data", "model"))
    
    # Create model instance with validated config
    model = create_qwen25_model(validated_config, dtype=dtype)
    
    # Load model parameters
    params = load_params_from_checkpoint(weights_dir)
    
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f} seconds")
    
    return model, validated_config, params, mesh

# Initialize debug mode from environment variables or command line arguments
if __name__ != "__main__":
    # Check for environment variable
    if os.environ.get("QWEN_DEBUG", "").lower() in ("1", "true", "yes", "on"):
        set_debug_logging(True)
        logger.info("Debug mode enabled via environment variable")
    
    # Check if any command line argument contains debug flag
    for arg in sys.argv:
        if arg.lower() in ("--debug", "--qwen-debug", "--enable-debug"):
            set_debug_logging(True)
            logger.info("Debug mode enabled via command line argument")
            break 