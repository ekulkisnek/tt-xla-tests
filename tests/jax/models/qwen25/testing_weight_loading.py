#!/usr/bin/env python3
# Copyright 2023 The JAX Authors.
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

"""
Testing Weight Loading for Qwen 2.5 in Flax

This script tests loading the Qwen 2.5 weights from safetensors files
into a compatible Flax model structure without implementing the full model.

based on context @modeling_flax_utils.py @modeling_flax_pytorch_utils.py @hub.py @modeling_flax_llama.py @modeling_flax_mistral.py 


"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
import flax.linen as nn

# Import from transformers for weight loading utilities
try:
    from transformers import AutoConfig, FlaxPreTrainedModel
    from transformers.modeling_flax_pytorch_utils import load_pytorch_checkpoint_in_flax_state_dict
    from transformers.utils.hub import get_checkpoint_shard_files, cached_file
    from safetensors import safe_open
    from safetensors.flax import load_file as safe_load_file
except ImportError:
    raise ImportError(
        "This script requires transformers and safetensors. "
        "Please install with: pip install transformers safetensors"
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to weights directory
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen25-weights")


class DummyConfig:
    """
    Dummy configuration class to mimic HuggingFace's config for testing
    
    Based on transformers.PretrainedConfig pattern
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Default values
        self.hidden_size = 4096
        self.intermediate_size = 14336
        self.num_hidden_layers = 32
        self.num_attention_heads = 32
        self.num_key_value_heads = 32
        self.hidden_act = "silu"
        self.max_position_embeddings = 4096
        self.initializer_range = 0.02
        self.rms_norm_eps = 1e-5
        self.tie_word_embeddings = False
        self.rope_theta = 10000.0
        self.attention_bias = True
        self.vocab_size = 151936

    @classmethod
    def from_json_file(cls, json_file):
        """Load config from a json file"""
        with open(json_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = {}
        for key, value in self.__dict__.items():
            output[key] = value
        return output


class DummyFlaxQwen25Module(nn.Module):
    """
    Minimal dummy module to test weight loading
    
    Based on transformers.models.mistral.modeling_flax_mistral.FlaxMistralModule pattern
    """
    config: Any
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            dtype=self.dtype,
        )
        
    def __call__(self, input_ids):
        # This function does nothing, just for interface compatibility
        return input_ids


class DummyFlaxQwen25Model(FlaxPreTrainedModel):
    """
    Dummy model class to test weight loading functionality
    
    Based on transformers.models.mistral.modeling_flax_mistral.FlaxMistralPreTrainedModel pattern
    """
    config_class = DummyConfig
    base_model_prefix = "model"
    module_class = DummyFlaxQwen25Module

    def __init__(self, config, input_shape=(1, 1), seed=0, dtype=jnp.float32, _do_init=False):
        module = self.module_class(config=config, dtype=dtype)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)


def load_config_from_weights_dir():
    """
    Load configuration from the weights directory
    
    Based on pattern in transformers.models.auto.configuration_auto.AutoConfig.from_pretrained
    """
    logger.info(f"Loading config from {WEIGHTS_DIR}")
    
    config_path = os.path.join(WEIGHTS_DIR, "config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found at {config_path}")
    
    return DummyConfig.from_json_file(config_path)


def load_safetensors_weights(model_path: str) -> Dict:
    """
    Load weights directly from safetensors files
    
    Inspired by weight_loading.py in the qwen25 directory and 
    transformers.modeling_flax_utils.load_flax_sharded_weights
    """
    logger.info(f"Loading safetensors weights from {model_path}")
    
    # Check for safetensors index file
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    safetensors_files = []
    
    if os.path.exists(index_file):
        logger.info(f"Found safetensors index file: {index_file}")
        with open(index_file, "r") as f:
            index = json.load(f)
            if "weight_map" in index:
                weight_map = index["weight_map"]
                files = sorted(list(set(weight_map.values())))
                safetensors_files = [os.path.join(model_path, f) for f in files]
    
    # If no index file, check for safetensors files directly
    if not safetensors_files:
        import glob
        safetensors_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {model_path}")
    
    logger.info(f"Found {len(safetensors_files)} safetensors files: {safetensors_files}")
    
    # Load all tensors into a flat dictionary
    all_params = {}
    for file_path in safetensors_files:
        logger.info(f"Loading weights from {file_path}")
        with safe_open(file_path, framework="flax") as f:
            for key in f.keys():
                all_params[key] = f.get_tensor(key)
    
    # Convert flat dictionary to nested dictionary
    nested_params = unflatten_dict(all_params, sep=".")
    
    logger.info(f"Loaded {len(all_params)} parameters from safetensors files")
    return nested_params


def load_weights_with_transformers_utils(model_path: str) -> Dict:
    """
    Load weights using transformers utilities
    
    Based on transformers.modeling_flax_utils.FlaxPreTrainedModel.from_pretrained 
    and transformers.modeling_flax_pytorch_utils.load_pytorch_checkpoint_in_flax_state_dict
    """
    logger.info(f"Loading weights with transformers utilities from {model_path}")
    
    # Load config
    config = load_config_from_weights_dir()
    
    # Initialize model
    dummy_model = DummyFlaxQwen25Model(config, _do_init=True)
    
    # Check for sharded model
    try:
        # Check if model is sharded - will use index file
        if os.path.exists(os.path.join(model_path, "model.safetensors.index.json")):
            logger.info("Found sharded model, loading with get_checkpoint_shard_files")
            shard_files = get_checkpoint_shard_files(
                model_path, 
                "model.safetensors.index.json",
            )
            # Will use load_pytorch_checkpoint_in_flax_state_dict which handles safetensors and sharded models
            flax_state_dict = load_pytorch_checkpoint_in_flax_state_dict(
                dummy_model, shard_files, is_sharded=True
            )
            return flax_state_dict
        else:
            # For single-file models
            flax_state_dict = load_pytorch_checkpoint_in_flax_state_dict(
                dummy_model, os.path.join(model_path, "model.safetensors"), is_sharded=False
            )
            return flax_state_dict
    except Exception as e:
        logger.error(f"Error loading with transformers utils: {e}")
        # If transformers utility fails, try direct loading
        return load_safetensors_weights(model_path)


def summarize_weights(weights_dict: Dict) -> None:
    """
    Print a summary of the weights
    """
    # Flatten the weights dictionary for easier analysis
    flattened = flatten_dict(weights_dict)
    
    total_params = 0
    param_shapes = {}
    param_types = {}
    
    for key, tensor in flattened.items():
        # Count parameters
        num_params = np.prod(tensor.shape)
        total_params += num_params
        
        # Group by shape
        shape_str = str(tensor.shape)
        if shape_str not in param_shapes:
            param_shapes[shape_str] = 0
        param_shapes[shape_str] += 1
        
        # Group by tensor type
        dtype_str = str(tensor.dtype)
        if dtype_str not in param_types:
            param_types[dtype_str] = 0
        param_types[dtype_str] += 1
    
    logger.info(f"Total number of parameters: {total_params:,}")
    logger.info(f"Parameter shapes summary: {param_shapes}")
    logger.info(f"Parameter types summary: {param_types}")
    logger.info(f"Number of weight tensors: {len(flattened)}")
    
    # Print a sample of tensor names
    sample_keys = list(flattened.keys())[:10]
    logger.info(f"Sample of tensor names: {sample_keys}")


def main():
    """
    Main function to test weight loading
    """
    logger.info(f"Testing weight loading for Qwen 2.5")
    logger.info(f"JAX devices: {jax.devices()}")
    
    start_time = time.time()
    
    # Test direct loading from safetensors
    try:
        logger.info("=== Testing direct loading from safetensors ===")
        weights = load_safetensors_weights(WEIGHTS_DIR)
        logger.info("Successfully loaded weights directly from safetensors")
        summarize_weights(weights)
    except Exception as e:
        logger.error(f"Error loading directly from safetensors: {e}")
    
    # Test loading with transformers utilities
    try:
        logger.info("\n=== Testing loading with transformers utilities ===")
        weights = load_weights_with_transformers_utils(WEIGHTS_DIR)
        logger.info("Successfully loaded weights with transformers utilities")
        summarize_weights(weights)
    except Exception as e:
        logger.error(f"Error loading with transformers utilities: {e}")
    
    end_time = time.time()
    logger.info(f"Weight loading test completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main() 