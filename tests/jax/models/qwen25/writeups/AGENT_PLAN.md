# Qwen2.5-7B Tensor-Parallel Implementation Plan

## Required Context Files and Directories

- `/Users/lu/Documents/hf-transformers/src/transformers/models/qwen2/modeling_qwen2.py` - Qwen2 PyTorch implementation
- `/Users/lu/Documents/hf-transformers/src/transformers/models/qwen2/configuration_qwen2.py` - Qwen2 configuration
- `/Users/lu/Documents/hf-transformers/src/transformers/models/llama/modeling_flax_llama.py` - Base Flax model reference
- `/Users/lu/Documents/hf-transformers/src/transformers/modeling_flax_utils.py` - Flax model utilities
- `/Users/lu/Documents/hf-transformers/src/transformers/modeling_flax_pytorch_utils.py` - PyTorch to Flax conversion
- `/Users/lu/Documents/hf-transformers/examples/flax/language-modeling/run_clm_flax.py` - JAX parallelism examples
- `/Users/lu/Documents/tt-bounty-1/qwen2.5-7b/` - Pre-downloaded model files
- Flax GSPMD Guide: https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html
- JAX SPMD Guide: https://jax.readthedocs.io/en/latest/spmd.html

## Project Structure

Create the following files in `tt-xla/tests/jax/models/qwen2_5/`:

1. `configuration_qwen2_5.py` - Model configuration
2. `modeling_flax_qwen2_5.py` - Core model implementation
3. `tensor_parallel.py` - Device mesh and sharding utilities
4. `weight_loading.py` - Weight loading and conversion
5. `gsm8k_eval.py` - GSM8K evaluation
6. `__init__.py` - Package initialization

## Phase 1: Model Implementation

### 1.1 Configuration (configuration_qwen2_5.py)

```python
# coding=utf-8
# Copyright 2024 Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen2.5 model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

class Qwen25Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen25Model`] or a [`FlaxQwen25Model`]. 
    It is used to instantiate a Qwen2.5 model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 152064):
            Vocabulary size of the Qwen2.5 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen25Model`] or [`FlaxQwen25Model`].
        hidden_size (`int`, *optional*, defaults to 3584):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 18944):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 28):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key value heads for each attention layer in the Transformer encoder.
            For standard attention, `num_key_value_heads` = `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with. Qwen models use rotary position embeddings.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the beginning-of-sequence token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the end-of-sequence token.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        attention_bias (`bool`, defaults to `True`):
            Whether to use bias in the query, key, value and output projection layers.
        attention_dropout (`float`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to None):
            The sliding window size for attention. If unset but use_sliding_window is True, will default to max position embeddings.
        max_window_layers (`int`, *optional*):
            Number of layers that use sliding window attention. Starts from the first layer.
        _attn_implementation (`str`, *optional*):
            The attention implementation to use. Can be "eager", "sdpa", "flash_attention_2".

    Example:

    ```python
    >>> from transformers import FlaxQwen25Model, Qwen25Config

    >>> # Initializing a Qwen2.5 style configuration
    >>> configuration = Qwen25Config()

    >>> # Initializing a model from the configuration
    >>> model = FlaxQwen25Model(configuration)
    ```
    """
    model_type = "qwen2_5"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=1000000.0,
        attention_bias=True,
        attention_dropout=0.0,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=None,
        _attn_implementation=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self._attn_implementation = _attn_implementation
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
```

### 1.2 Core Model (modeling_flax_qwen2_5.py)

#### 1.2.1 Attention Implementation

```python
# coding=utf-8
# Copyright 2024 Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Flax Qwen2.5 model."""

from functools import partial
from typing import Callable, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .configuration_qwen2_5 import Qwen25Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Qwen25Config"
_CHECKPOINT_FOR_DOC = "qwen/Qwen2-5-7B"


def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2:], tensor[..., :tensor.shape[-1] // 2]), axis=-1
    )
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)


def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2) / dim))
    freqs = jnp.einsum("i , j -> i j", jnp.arange(num_pos), inv_freq)

    emb = jnp.concatenate((freqs, freqs), axis=-1)
    sin_pos = jnp.sin(emb)
    cos_pos = jnp.cos(emb)
    
    return sin_pos, cos_pos


class FlaxQwen25RMSNorm(nn.Module):
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        self.epsilon = self.config.rms_norm_eps
        self.weight = self.param(
            "weight", 
            jax.nn.initializers.ones, 
            (self.config.hidden_size,)
        )
        
    def __call__(self, hidden_states):
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)
        
        return self.weight * jnp.asarray(hidden_states, dtype=self.dtype)


class FlaxQwen25RotaryEmbedding(nn.Module):
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.sin_pos, self.cos_pos = create_sinusoidal_positions(self.config.max_position_embeddings, head_dim)
    
    def __call__(self, key, query, position_ids):
        # Extract the correct sinusoidal embeddings based on position_ids
        sin_pos = self.sin_pos[position_ids]
        cos_pos = self.cos_pos[position_ids]
        
        # Apply rotary embeddings
        key = apply_rotary_pos_emb(key, sin_pos, cos_pos)
        query = apply_rotary_pos_emb(query, sin_pos, cos_pos)
        
        # Convert to desired dtype
        key = jnp.asarray(key, dtype=self.dtype)
        query = jnp.asarray(query, dtype=self.dtype)
        
        return key, query


class FlaxQwen25Attention(nn.Module):
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32
    causal: bool = True
    
    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        
        # Initialize projections with partitioning annotations
        self.q_proj = nn.Dense(
            self.num_heads * self.head_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            param_axes={"kernel": nn.AxisMetadata(names=("embed", "heads"))},
        )
        
        self.k_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            param_axes={"kernel": nn.AxisMetadata(names=("embed", "heads"))},
        )
        
        self.v_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            param_axes={"kernel": nn.AxisMetadata(names=("embed", "heads"))},
        )
        
        self.o_proj = nn.Dense(
            self.embed_dim,
            use_bias=config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            param_axes={"kernel": nn.AxisMetadata(names=("heads", "embed"))},
        )
        
        self.rotary_emb = FlaxQwen25RotaryEmbedding(config, dtype=self.dtype)
        self.causal_mask = make_causal_mask(jnp.ones((1, config.max_position_embeddings), dtype="bool"), dtype="bool")

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))
    
    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps.
        """
        # detect if we're initializing by absence of existing cache data.
        is_initialized = self.has_variable("cache", "cached_key")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask
    
    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        deterministic=True,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Sharding constraint for tensor parallelism
        query = jax.lax.with_sharding_constraint(query, jax.sharding.PartitionSpec(None, None, "model"))
        key = jax.lax.with_sharding_constraint(key, jax.sharding.PartitionSpec(None, None, "model"))
        value = jax.lax.with_sharding_constraint(value, jax.sharding.PartitionSpec(None, None, "model"))

        query = self._split_heads(query, self.num_heads)
        key = self._split_heads(key, self.num_key_value_heads)
        value = self._split_heads(value, self.num_key_value_heads)

        key, query = self.rotary_emb(key, query, position_ids)

        query_length, key_length = query.shape[1], key.shape[1]

        if self.has_variable("cache", "cached_key"):
            mask_shift = self.variables["cache"]["cache_index"]
            max_decoder_length = self.variables["cache"]["cached_key"].shape[1]
            causal_mask = lax.dynamic_slice(
                self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length)
            )
        else:
            causal_mask = self.causal_mask[:, :, :query_length, :key_length]

        batch_size = hidden_states.shape[0]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])

        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
        attention_mask = combine_masks(attention_mask, causal_mask)

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if past_key_value is not None:
            # past_key has shape (batch_size, 1, num_heads, seq_len, head_dim)
            if isinstance(past_key_value, tuple) and len(past_key_value) == 2:
                # This branch handles backward compatibility with the previous implementation
                past_key = past_key_value[0]
                past_value = past_key_value[1]
                key = jnp.concatenate([past_key, key], axis=2)
                value = jnp.concatenate([past_value, value], axis=2)
            else:
                # Normal path for current implementation
                key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)

        # Repeat k/v heads if n_kv_heads < n_heads
        key = jnp.repeat(key, self.num_key_value_groups, axis=2)
        value = jnp.repeat(value, self.num_key_value_groups, axis=2)
        
        # Apply tensor parallelism constraints
        key = jax.lax.with_sharding_constraint(key, jax.sharding.PartitionSpec(None, None, "model", None))
        value = jax.lax.with_sharding_constraint(value, jax.sharding.PartitionSpec(None, None, "model", None))
        query = jax.lax.with_sharding_constraint(query, jax.sharding.PartitionSpec(None, None, "model", None))

        # transform boolean mask into float mask
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )

        # for numerical stability, compute attention in float32
        attention_dtype = jnp.float32
        query = query.astype(attention_dtype)
        key = key.astype(attention_dtype)
        
        # usual dot product attention
        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=None,
            dropout_rate=0.0,
            deterministic=deterministic,
            dtype=attention_dtype,
        )

        attn_weights = attn_weights.astype(self.dtype)
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        
        # Apply tensor parallelism constraints
        attn_output = jax.lax.with_sharding_constraint(
            attn_output, jax.sharding.PartitionSpec(None, None, "model", None)
        )
        
        attn_output = self._merge_heads(attn_output)
        attn_output = self.o_proj(attn_output)
        
        # Apply tensor parallelism constraints
        attn_output = jax.lax.with_sharding_constraint(
            attn_output, jax.sharding.PartitionSpec(None, None, None)
        )

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        
        if use_cache:
            outputs = outputs + (key, value)
            
        return outputs
```

#### 1.2.2 MLP Implementation

```python
class FlaxQwen25MLP(nn.Module):
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        config = self.config
        self.gate_proj = nn.Dense(
            config.intermediate_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            param_axes={"kernel": nn.AxisMetadata(names=("embed", "mlp"))},
        )
        self.up_proj = nn.Dense(
            config.intermediate_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            param_axes={"kernel": nn.AxisMetadata(names=("embed", "mlp"))},
        )
        self.down_proj = nn.Dense(
            config.hidden_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            param_axes={"kernel": nn.AxisMetadata(names=("mlp", "embed"))},
        )
        self.act_fn = ACT2FN[config.hidden_act]
        
    def __call__(self, x, deterministic: bool = True):
        # Apply gate projection with sharding constraint
        gate_proj = self.gate_proj(x)
        gate_proj = jax.lax.with_sharding_constraint(
            gate_proj, jax.sharding.PartitionSpec(None, None, "model")
        )
        
        # Apply up projection with sharding constraint
        up_proj = self.up_proj(x)
        up_proj = jax.lax.with_sharding_constraint(
            up_proj, jax.sharding.PartitionSpec(None, None, "model")
        )
        
        # Apply SwiGLU activation
        hidden_states = self.act_fn(gate_proj) * up_proj
        
        # Apply down projection with sharding constraint
        down_proj = self.down_proj(hidden_states)
        down_proj = jax.lax.with_sharding_constraint(
            down_proj, jax.sharding.PartitionSpec(None, None, None)
        )
        
        return down_proj
```

#### 1.2.3 Decoder Layer Implementation

```python
class FlaxQwen25DecoderLayer(nn.Module):
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        self.input_layernorm = FlaxQwen25RMSNorm(self.config, dtype=self.dtype)
        self.self_attn = FlaxQwen25Attention(self.config, dtype=self.dtype)
        self.post_attention_layernorm = FlaxQwen25RMSNorm(self.config, dtype=self.dtype)
        self.mlp = FlaxQwen25MLP(self.config, dtype=self.dtype)
    
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        deterministic=True,
    ):
        residual = hidden_states
        
        # Apply tensor parallelism constraint for sequence dimension
        hidden_states = jax.lax.with_sharding_constraint(
            hidden_states, jax.sharding.PartitionSpec(None, None, None)
        )
        
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            deterministic=deterministic,
        )
        
        hidden_states = outputs[0]
        
        # Apply tensor parallelism constraint
        hidden_states = jax.lax.with_sharding_constraint(
            hidden_states, jax.sharding.PartitionSpec(None, None, None)
        )
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, deterministic=deterministic)
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        # Apply tensor parallelism constraint
        hidden_states = jax.lax.with_sharding_constraint(
            hidden_states, jax.sharding.PartitionSpec(None, None, None)
        )
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (outputs[1],)
            
        if use_cache:
            outputs += (outputs[2:4] if output_attentions else outputs[1:3],)
            
        return outputs  # hidden_states, (attn_weights), (past_key_value)


class FlaxQwen25PreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Qwen25Config
    base_model_prefix = "model"
    module_class: nn.Module = None
    
    def __init__(
        self,
        config: Qwen25Config,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
    
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        
        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                position_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                return_dict=False,
            )
        else:
            module_init_outputs = self.module.init(
                rngs, input_ids, attention_mask, position_ids, return_dict=False
            )
            
        random_params = module_init_outputs["params"]
        
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in set(random_params) - set(params):
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set(random_params) - set(params)
            return freeze(unflatten_dict(params))
        else:
            return random_params
    
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length), dtype="i4")
        attention_mask = jnp.ones_like(input_ids, dtype="i4")
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)
        
        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return init_variables["cache"]


class FlaxQwen25LayerCollection(nn.Module):
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        self.layers = [
            FlaxQwen25DecoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]
        
    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
            
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            layer_outputs = layer(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value=past_key_values[i],
                output_attentions=output_attentions,
                use_cache=True,
                deterministic=deterministic,
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                
            # Cache handling for auto-regressive generation
            if init_cache:
                past_key_values[i] = layer_outputs[-1]
                
        # Add final layer norm
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        outputs = (hidden_states, past_key_values)
        
        if not return_dict:
            return outputs
            
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            past_key_values=past_key_values,
        )


class FlaxQwen25Module(nn.Module):
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            param_axes={"embedding": nn.AxisMetadata(names=("vocab", "embed"))},
        )
        self.layers = FlaxQwen25LayerCollection(self.config, dtype=self.dtype)
        self.norm = FlaxQwen25RMSNorm(self.config, dtype=self.dtype)
        
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Create embedding
        inputs_embeds = self.embed_tokens(input_ids.astype("i4"))
        
        # Apply tensor parallelism constraint
        inputs_embeds = jax.lax.with_sharding_constraint(
            inputs_embeds, jax.sharding.PartitionSpec(None, None, "model")
        )
        
        # By default, attention_mask in Flax is an additive mask (as opposed to a multiplicative mask in PyTorch).
        # This is why we expand the `attention_mask` to 4 dimensions, making the second dimension the "head"
        # dimension. This matches the hidden shape, which is (batch, heads, seq, features).
        
        # Prepare padding attention mask
        if attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(1, 2))
            # 1 => not masked, 0 => masked
            # Convert 0-padding to attention bias that is added to attention logits.
            # (1, 1, 1, seq_len) * (batch_size, 1, 1, 1) = (batch_size, 1, 1, seq_len)
            # => then broadcast to (batch_size, 1, seq_len, seq_len)
            attention_mask = jnp.broadcast_to(
                attention_mask, (attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
            )
            
        # Handle any PRNG if needed
        rngs = {}
        
        # Pass through the model
        outputs = self.layers(
            inputs_embeds,
            attention_mask,
            position_ids,
            past_key_values=past_key_values,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        
        if self.config.norm_elementwise_affine:
            hidden_states = self.norm(hidden_states)
        
        if not return_dict:
            return (hidden_states,) + outputs[1:]
        
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
        )


@add_start_docstrings(
    "The bare Qwen2.5 Model transformer outputting raw hidden-states without any specific head on top.",
    "QWEN2_5_START_DOCSTRING",
)
class FlaxQwen25Model(FlaxQwen25PreTrainedModel):
    module_class = FlaxQwen25Module


class FlaxQwen25ForCausalLMModule(nn.Module):
    config: Qwen25Config
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        self.model = FlaxQwen25Module(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            param_axes={"kernel": nn.AxisMetadata(names=("embed", "vocab"))},
        )
    
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        outputs = self.model(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values=past_key_values,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        
        # Apply tensor parallelism constraint
        hidden_states = jax.lax.with_sharding_constraint(
            hidden_states, jax.sharding.PartitionSpec(None, None, "model")
        )
        
        # Apply language modeling head
        logits = self.lm_head(hidden_states)
        
        # Apply tensor parallelism constraint
        logits = jax.lax.with_sharding_constraint(
            logits, jax.sharding.PartitionSpec(None, None, None)
        )
        
        if not return_dict:
            return (logits,) + outputs[1:]
        
        return FlaxCausalLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
        )


@add_start_docstrings(
    """
    The Qwen2.5 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    "QWEN2_5_START_DOCSTRING",
)
class FlaxQwen25ForCausalLM(FlaxQwen25PreTrainedModel):
    module_class = FlaxQwen25ForCausalLMModule
    
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # initializing the cache
        batch_size, seq_length = input_ids.shape
        
        past_key_values = self.init_cache(batch_size, max_length)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since the decoder uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))
            
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }
    
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
```

## Phase 2: Tensor Parallelism Implementation

### 2.1 Device Mesh and Sharding Utilities (tensor_parallel.py)

```python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tensor Parallelism utilities for Flax models."""

import os
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict


def create_device_mesh(mesh_shape):
    """
    Create a device mesh with the specified shape for tensor parallelism.
    
    Args:
        mesh_shape: Tuple of (batch_dim, model_dim) for the mesh
        
    Returns:
        Mesh object with named axes ('batch', 'model')
    """
    devices = jax.devices()
    total_devices = mesh_shape[0] * mesh_shape[1]
    
    if len(devices) < total_devices:
        raise ValueError(f"Requested mesh shape {mesh_shape} requires {total_devices} devices, "
                         f"but only {len(devices)} are available")
    
    # Create the mesh
    device_mesh = jnp.array(devices[:total_devices]).reshape(mesh_shape)
    return Mesh(device_mesh, ('batch', 'model'))


def get_partition_rules():
    """
    Get appropriate partition rules for model parameters that match the Transformers naming conventions.
    
    Returns:
        List of tuples mapping parameter patterns to partition specs
    """
    return [
        # Embeddings - shard on vocab dimension
        ('model/embed_tokens/embedding', P(None, 'model')),
        
        # Layer norms - replicated
        ('model/norm/.*', P(None)),
        ('model/layers/\\d+/input_layernorm/.*', P(None)),
        ('model/layers/\\d+/post_attention_layernorm/.*', P(None)),
        
        # Attention QKV projections - shard on output dimension
        ('model/layers/\\d+/self_attn/q_proj/kernel', P('model', None)),
        ('model/layers/\\d+/self_attn/k_proj/kernel', P('model', None)),
        ('model/layers/\\d+/self_attn/v_proj/kernel', P('model', None)),
        ('model/layers/\\d+/self_attn/[qkv]_proj/bias', P('model')),
        
        # Attention output projection - shard on input dimension
        ('model/layers/\\d+/self_attn/o_proj/kernel', P(None, 'model')),
        
        # Feed-forward projections - shard on output (gate, up) or input (down) dimension
        ('model/layers/\\d+/mlp/gate_proj/kernel', P('model', None)),
        ('model/layers/\\d+/mlp/up_proj/kernel', P('model', None)),
        ('model/layers/\\d+/mlp/down_proj/kernel', P(None, 'model')),
        
        # LM head - shard on output dimension (vocab)
        ('lm_head/kernel', P(None, 'model')),
        
        # Default rule
        ('.*', P(None)),
    ]


def get_params_by_path(nested_dict, path_parts):
    """Helper to navigate nested dictionary by path_parts"""
    if not path_parts:
        return nested_dict
    if path_parts[0] in nested_dict:
        return get_params_by_path(nested_dict[path_parts[0]], path_parts[1:])
    return None


def apply_partition_rules(params, rules, params_path=''):
    """
    Apply partition rules to parameters. Used during weight loading.
    
    Args:
        params: Parameter dictionary (or value)
        rules: List of (pattern, spec) tuples
        params_path: Path of the current parameter in the hierarchy
        
    Returns:
        Parameter dictionary with appropriate partition specs
    """
    import re
    
    if not isinstance(params, dict):
        # Base case: we've reached a leaf parameter
        for pattern, spec in rules:
            if re.match(pattern, params_path):
                return spec
        return P(None)  # Default: replicate
    
    # Recursive case: navigate the dictionary
    return {k: apply_partition_rules(v, rules, f"{params_path}/{k}" if params_path else k) for k, v in params.items()}


def with_sharding_constraint(x, partition_spec):
    """
    Apply sharding constraint if JAX version supports it.
    
    Args:
        x: Array to constrain
        partition_spec: PartitionSpec to apply
        
    Returns:
        Constrained array (or original if pjit not active)
    """
    if jax.process_count() > 1:
        return jax.lax.with_sharding_constraint(x, partition_spec)
    else:
        return x


def shard_params(params, mesh, partition_rules):
    """
    Shard parameters according to the partition rules.
    
    Args:
        params: Parameter dictionary or FrozenDict
        mesh: Device mesh
        partition_rules: List of (pattern, spec) tuples for partitioning
        
    Returns:
        Sharded parameters
    """
    flat_params = flatten_dict(unfreeze(params) if isinstance(params, FrozenDict) else params)
    flat_partition_specs = {
        path: apply_partition_rules(param, partition_rules, "/".join(path))
        for path, param in flat_params.items()
    }
    
    # Function to apply appropriate sharding to each parameter
    def shard_param(param, spec):
        # Skip parameters that are already on the right device
        if hasattr(param, 'sharding') and param.sharding.spec == spec:
            return param
        return jax.device_put(param, jax.sharding.NamedSharding(mesh, spec))
    
    # Apply sharding to each parameter
    sharded_flat_params = {
        path: shard_param(param, flat_partition_specs[path])
        for path, param in flat_params.items()
    }
    
    # Reconstruct the original nested structure
    return freeze(unflatten_dict(sharded_flat_params)) if isinstance(params, FrozenDict) else unflatten_dict(sharded_flat_params)
```

### 2.2 Cross-Device Communication Primitives

```python
def apply_rotary_pos_emb_with_tp(q, k, cos, sin, position_ids, mesh):
    """
    Apply rotary position embeddings with tensor parallelism awareness.
    
    Args:
        q: Query tensor, sharded across model dimension
        k: Key tensor, sharded across model dimension
        cos: Cosine part of rotary embeddings
        sin: Sine part of rotary embeddings
        position_ids: Position ids tensor
        mesh: Device mesh
        
    Returns:
        Rotated query and key tensors with same sharding as inputs
    """
    # Apply normal rotary embeddings
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    # Make sure the sharding constraint is properly maintained
    q_embed = with_sharding_constraint(q_embed, P('batch', None, 'model', None))
    k_embed = with_sharding_constraint(k_embed, P('batch', None, 'model', None))
    
    # Convert to desired dtype
    q_embed = jnp.asarray(q_embed, dtype=q.dtype)
    k_embed = jnp.asarray(k_embed, dtype=k.dtype)
    
    return q_embed, k_embed


def tensor_parallel_attention(query, key, value, attention_mask, mesh, num_heads, num_kv_heads, num_kv_groups, scaling):
    """
    Attention implementation with tensor parallelism constraints.
    
    Args:
        query: Query tensor [batch, seq_len, num_heads, head_dim]
        key: Key tensor [batch, seq_len, num_kv_heads, head_dim]
        value: Value tensor [batch, seq_len, num_kv_heads, head_dim]
        attention_mask: Attention mask
        mesh: Device mesh
        num_heads: Total number of attention heads
        num_kv_heads: Number of key/value heads
        num_kv_groups: Number of query heads per key/value head
        scaling: Scaling factor for attention scores
        
    Returns:
        Output tensor after attention
    """
    # Repeat k/v heads if needed (for MHA with GQA)
    if num_kv_groups > 1:
        # Apply sharding constraints
        key = with_sharding_constraint(key, P('batch', None, 'model', None))
        value = with_sharding_constraint(value, P('batch', None, 'model', None))
        
        # Repeat across model dimension
        key = jnp.repeat(key, num_kv_groups, axis=2)
        value = jnp.repeat(value, num_kv_groups, axis=2)
        
        # Apply sharding constraints again after repeat
        key = with_sharding_constraint(key, P('batch', None, 'model', None))
        value = with_sharding_constraint(value, P('batch', None, 'model', None))
    
    # Apply appropriate sharding constraints
    query = with_sharding_constraint(query, P('batch', None, 'model', None))
    key = with_sharding_constraint(key, P('batch', None, 'model', None))
    value = with_sharding_constraint(value, P('batch', None, 'model', None))
    
    # Compute attention scores with appropriate scaling
    attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key) * scaling
    
    # Apply mask
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    
    # Softmax with appropriate precision
    attn_weights = jax.nn.softmax(attn_weights, axis=-1, dtype=jnp.float32)
    attn_weights = attn_weights.astype(query.dtype)
    
    # Apply attention
    attn_output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)
    
    # Apply sharding constraint to output
    attn_output = with_sharding_constraint(attn_output, P('batch', None, 'model', None))
    
    return attn_output
```

## Phase 3: Weight Loading Implementation

### 3.1 Parameter Conversion (weight_loading.py)

```python
from flax.traverse_util import flatten_dict, unflatten_dict

def apply_parameter_partitioning(params, param_specs, mesh):
    """
    Apply partitioning specifications to parameters with pattern matching.
    
    Args:
        params: Parameter dictionary
        param_specs: Dictionary mapping parameter patterns to partition specs
        mesh: Device mesh
        
    Returns:
        Parameters with appropriate partitioning
    """
    import re
    
    # Flatten parameters for easier matching
    flat_params = flatten_dict(params)
    
    # Apply partitioning based on patterns
    sharded_params = {}
    for param_path, param in flat_params.items():
        # Convert path tuple to string for pattern matching
        path_str = '.'.join(param_path)
        
        # Find matching pattern
        matching_spec = None
        for pattern, spec in param_specs.items():
            # Convert * to regex pattern
            pattern_regex = pattern.replace('.', '\\.').replace('*', '.*')
            if re.match(pattern_regex, path_str):
                matching_spec = spec
                break
        
        # Apply partitioning if spec found
        if matching_spec is not None:
            # Apply partitioning using jax.device_put
            param = jax.device_put(param, jax.sharding.NamedSharding(mesh, matching_spec))
        else:
            # Default to replication if no spec found
            param = jax.device_put(param, jax.sharding.NamedSharding(mesh, P(None)))
        
        sharded_params[param_path] = param
    
    # Unflatten parameters back to original structure
    return unflatten_dict(sharded_params)
```

### 3.2 Model Loading Utility

```python
def create_and_load_tensor_parallel_model(
    model_path,
    mesh_shape=(1, 8),
    dtype=jnp.bfloat16,
    from_pt=True
):
    """
    Create and load a tensor-parallel Qwen2.5 model.
    
    Args:
        model_path: Path to the model weights
        mesh_shape: Shape of the device mesh (rows, cols)
        dtype: Data type for model weights
        from_pt: Whether to convert from PyTorch format
        
    Returns:
        Loaded model with tensor parallelism
    """
    # 1. Create device mesh
    mesh = create_device_mesh(mesh_shape)
    
    # 2. Define parameter partitioning
    param_partition_specs = get_partition_rules()
    
    # 3. Create and load model with tensor parallelism
    with mesh:
        # Load configuration
        config = FlaxQwen2Config.from_pretrained(model_path)
        
        # Initialize model without weights
        model = FlaxQwen2ForCausalLM(
            config,
            dtype=dtype,
            _do_init=False,
        )
        
        # Load and shard weights
        if from_pt:
            # Load from PyTorch weights
            params = FlaxQwen2ForCausalLM.from_pretrained(
                model_path,
                from_pt=True,
                _do_init=False,
                dtype=dtype,
            ).params
            
            # Apply sharding to parameters
            params = apply_parameter_partitioning(params, param_partition_specs, mesh)
            model.params = params
        else:
            # Directly load Flax weights with sharding
            model = FlaxQwen2ForCausalLM.from_pretrained(
                model_path,
                _do_init=False,
                dtype=dtype,
                param_partition_specs=param_partition_specs,
            )
    
    return model, mesh
```

## Phase 4: GSM8K Evaluation

### 4.1 Evaluation Function (gsm8k_eval.py)

```python
def evaluate_gsm8k_with_tensor_parallelism(
    model,
    tokenizer,
    dataset,
    mesh,
    max_new_tokens=512,
    batch_size=1
):
    """
    Evaluate model on GSM8K with tensor parallelism.
    
    Args:
        model: Tensor-parallel model
        tokenizer: Tokenizer for the model
        dataset: GSM8K dataset
        mesh: Device mesh
        max_new_tokens: Maximum number of tokens to generate
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with evaluation results
    """
    # Prepare batched dataset
    batched_dataset = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        batched_dataset.append(batch)
    
    # Run evaluation with tensor parallelism
    results = []
    
    def generate_with_tensor_parallelism(input_ids, attention_mask):
        # Apply sharding constraints
        input_ids = jax.lax.with_sharding_constraint(input_ids, P('batch', None))
        attention_mask = jax.lax.with_sharding_constraint(attention_mask, P('batch', None))
        
        # Generate text
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.0,  # Use greedy decoding
            do_sample=False,
        )
        
        return output.sequences
    
    # Run evaluation
    with mesh:
        for batch in batched_dataset:
            # Tokenize inputs
            inputs = tokenizer(batch["question"], padding=True, truncation=True, return_tensors="np")
            
            # Generate answers
            output_sequences = generate_with_tensor_parallelism(
                inputs["input_ids"],
                inputs["attention_mask"]
            )
            
            # Decode outputs
            generated_texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            
            # Extract answers and calculate metrics
            for text, reference in zip(generated_texts, batch["answer"]):
                extracted_answer = extract_answer(text)
                is_correct = check_answer(extracted_answer, reference)
                results.append({
                    "generated": text,
                    "answer": extracted_answer,
                    "reference": reference,
                    "correct": is_correct
                })
    
    # Calculate final metrics
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    
    return {
        "results": results,
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }
```

### 4.2 Answer Extraction Utilities

```python
def extract_answer(text):
    """
    Extract the final answer from generated text.
    
    Args:
        text: Generated text from the model
        
    Returns:
        Extracted final answer
    """
    # Look for patterns like "The answer is X" or just the final number
    import re
    
    answer_patterns = [
        r"The answer is\s*(\d+\.?\d*)",
        r"The final answer is\s*(\d+\.?\d*)",
        r"The result is\s*(\d+\.?\d*)",
        r"Therefore, the answer is\s*(\d+\.?\d*)",
        r"(\d+\.?\d*)$",  # Just a number at the end
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    
    # If no match found, try to find any number in the last sentence
    sentences = text.split('.')
    last_sentence = sentences[-1]
    numbers = re.findall(r"(\d+\.?\d*)", last_sentence)
    if numbers:
        return float(numbers[-1])
    
    return None

def check_answer(predicted, reference):
    """
    Check if the predicted answer matches the reference.
    
    Args:
        predicted: Extracted answer from model output
        reference: Reference answer from the dataset
        
    Returns:
        Boolean indicating whether the answer is correct
    """
    if predicted is None:
        return False
    
    try:
        # Convert reference to float for numerical comparison
        ref_value = float(reference)
        return abs(predicted - ref_value) < 1e-6
    except (ValueError, TypeError):
        # If reference is not a number, do string comparison
        return str(predicted) == str(reference)
```

## Phase 5: Complete Example Integration

### 5.1 End-to-End Example

```python
def run_end_to_end_tensor_parallel_inference():
    """
    Complete end-to-end example of tensor-parallel inference with Qwen2.5-7B.
    """
    # 1. Define mesh configurations
    mesh_shapes = {
        '2x4': (2, 4),
        '1x8': (1, 8),
        '1x32': (1, 32),
        '8x4': (8, 4),
    }
    
    # 2. Choose configuration based on available devices
    available_devices = len(jax.devices())
    if available_devices >= 32:
        mesh_shape = mesh_shapes['1x32']
    elif available_devices >= 8:
        mesh_shape = mesh_shapes['1x8']
    else:
        raise ValueError(f"Not enough devices available: {available_devices}")
    
    # 3. Create mesh
    mesh = create_device_mesh(mesh_shape)
    
    # 4. Load model with tensor parallelism
    model_path = "/Users/lu/Documents/tt-bounty-1/qwen2.5-7b"
    model, _ = create_and_load_tensor_parallel_model(
        model_path, 
        mesh_shape=mesh_shape,
        dtype=jnp.bfloat16,
        from_pt=True
    )
    
    # 5. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 6. Define tensor-parallel inference function
    def generate_with_tensor_parallelism(prompt, max_new_tokens=100):
        inputs = tokenizer(prompt, return_tensors="np")
        
        with mesh:
            # Apply sharding constraints
            input_ids = jax.lax.with_sharding_constraint(
                inputs["input_ids"], P('batch', None))
            
            attention_mask = None
            if "attention_mask" in inputs:
                attention_mask = jax.lax.with_sharding_constraint(
                    inputs["attention_mask"], P('batch', None))
            
            # Generate with tensor parallelism
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )
            
            # Get generated text
            generated_ids = outputs.sequences
            
            # Apply final sharding constraint
            generated_ids = jax.lax.with_sharding_constraint(
                generated_ids, P('batch', None))
        
        # Decode generated text
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # 7. Run inference
    prompt = "Solve the following math problem: If John has 5 apples and buys 3 more, how many does he have?"
    result = generate_with_tensor_parallelism(prompt)
    
    return result
```

## Implementation Timeline and Milestones

1. **Days 1-2: Setup and Core Model Implementation**
   - Set up development environment with JAX/Flax
   - Implement config.py and base model structure
   - Adapt LLaMA Flax code to Qwen2.5 architecture

2. **Days 3-4: Tensor Parallelism Implementation**
   - Implement device mesh utilities
   - Add parameter partitioning specifications
   - Integrate sharding constraints in forward pass

3. **Days 5-6: Weight Loading and Testing**
   - Implement weight conversion utilities
   - Test with pre-downloaded weights
   - Verify tensor shapes and parameter compatibility

4. **Days 7-8: GSM8K Evaluation**
   - Implement GSM8K evaluation logic
   - Test across different mesh configurations
   - Optimize performance and memory usage

5. **Days 9-10: Documentation and Final Testing**
   - Create comprehensive README
   - Document tensor parallelism approach
   - Finalize code and submit

## Key Implementation Checklist

- [ ] Core model structure implementation (config.py, modeling_flax_qwen2_5.py)
- [ ] Device mesh implementation (tensor_parallel.py)
- [ ] Weight loading utilities (weight_loading.py)
- [ ] Sharding constraints in attention and MLP
- [ ] Cross-device communication primitives
- [ ] GSM8K evaluation (gsm8k_eval.py)
- [ ] Support for all mesh configurations (2x4, 1x8, 1x32, 8x4)
- [ ] Documentation and examples

## Key Technical Insights

- Use Flax's `param_axes` for partitioning information in module definition
- Apply `jax.lax.with_sharding_constraint` to intermediate tensors during computation
- Leverage JAX's SPMD programming model for efficient parallelism
- Use pattern matching for parameter partitioning
- Apply one-time weight conversion from PyTorch to Flax for faster development
- Use appropriate partitioning strategy for different parameter types:
  - Input projections: partitioned on output dimension 
  - Output projections: partitioned on input dimension
  - Embeddings: partitioned on embedding dimension
  - Layer norms: replicated across devices 