# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen2.5 model implementation for tensor parallelism in JAX/FLAX.
"""

import collections
import logging
from typing import Dict, Optional, Tuple, Type, Union

from .model_implementation import (
    RMSNorm,
    precompute_freqs_cis,
    apply_rotary_emb,
    QwenAttention,
    QwenMLP,
    QwenTransformerBlock,
    Qwen2Model,
    Qwen2ForCausalLM,
)

from .tensor_parallel import (
    TensorParallelQwenAttention,
    TensorParallelQwenMLP,
    TensorParallelQwenTransformerBlock,
    TensorParallelQwen2ForCausalLM,
    TensorParallelQwen2Model,
    create_device_mesh,
    get_partition_specs,
)

from .config import (
    load_qwen_config,
    get_mesh_config,
    create_partition_specs,
    supported_mesh_configs,
    get_qwen2_7b_config,
    get_small_config,
    create_device_mesh as config_create_device_mesh,
)

from .weight_loading import (
    load_safetensors_index,
    convert_weight_name_to_flax,
    load_qwen_weights,
    init_model_from_weights,
    load_safetensors_into_params,
)

from .integration import (
    MODEL_NAME,
    get_supported_mesh_configs,
    get_tensor_parallel_test_configs,
    get_model_test_runner,
    run_tensor_parallel_tests,
    load_and_run_inference,
)

from .register import (
    get_model_metadata,
    register_model_factory,
    get_model_example,
)

from .tester import (
    Qwen25Tester,
    Qwen25SmallTester,
)

# Set up logging
logger = logging.getLogger(__name__)

# Function to control debug logging
def set_debug_logging(enabled=False):
    """
    Enable or disable detailed shape debugging logs.
    
    Args:
        enabled: Whether to enable detailed shape logs (default: False)
    """
    try:
        # Set flags in model_implementation
        from .model_implementation import DEBUG_MODE
        import sys
        sys.modules['tt-xla.tests.jax.models.qwen25.model_implementation'].DEBUG_MODE = enabled
        
        # Set flags in tensor_parallel
        from .tensor_parallel import DEBUG_SHAPES
        sys.modules['tt-xla.tests.jax.models.qwen25.tensor_parallel'].DEBUG_SHAPES = enabled
        
        # Set logging levels appropriately
        if enabled:
            logging.getLogger("tensor_parallel").setLevel(logging.DEBUG)
            logging.getLogger("model_implementation").setLevel(logging.DEBUG)
        else:
            logging.getLogger("tensor_parallel").setLevel(logging.INFO)
            logging.getLogger("model_implementation").setLevel(logging.WARNING)
        
        logger.info(f"Shape debug logging {'enabled' if enabled else 'disabled'}")
    except Exception as e:
        logger.warning(f"Could not set debug logging: {e}")

# Create ordered dictionaries for model mappings
MODEL_MAPPING_NAMES = collections.OrderedDict([
    ("qwen2_5", "Qwen2ForCausalLM"),
])

MODEL_TENSOR_PARALLEL_MAPPING_NAMES = collections.OrderedDict([
    ("qwen2_5", "TensorParallelQwen2ForCausalLM"),
])

# Create actual model mappings
MODEL_MAPPING = {
    "qwen2_5": Qwen2ForCausalLM,
}

MODEL_TENSOR_PARALLEL_MAPPING = {
    "qwen2_5": TensorParallelQwen2ForCausalLM,
}

# Base class for auto models
class _BaseAutoQwenClass:
    """Base class for auto classes."""
    _model_mapping = None

    @classmethod
    def from_config(cls, config, **kwargs):
        """Instantiate a model from config."""
        model_type = config.get("model_type", "qwen2_5")
        if model_type not in cls._model_mapping:
            raise ValueError(
                f"Unrecognized model type: {model_type}. "
                f"Model type should be one of {', '.join(cls._model_mapping.keys())}."
            )
        return cls._model_mapping[model_type](config, **kwargs)

    @classmethod
    def from_pretrained(cls, model_path, *model_args, **kwargs):
        """Load a pretrained model."""
        config = load_qwen_config(model_path)
        model_type = config.get("model_type", "qwen2_5")
        if model_type not in cls._model_mapping:
            raise ValueError(
                f"Unrecognized model type: {model_type}. "
                f"Model type should be one of {', '.join(cls._model_mapping.keys())}."
            )
        
        model_class = cls._model_mapping[model_type]
        model = model_class(config, **kwargs)
        
        # Initialize with pretrained weights
        params = load_qwen_weights(model_path, model)
        model_initialized = init_model_from_weights(model, params)
        
        return model_initialized


# Auto Classes
class AutoQwenModel(_BaseAutoQwenClass):
    """Auto class for Qwen2.5 base models."""
    _model_mapping = MODEL_MAPPING


class AutoQwenModelTensorParallel(_BaseAutoQwenClass):
    """Auto class for tensor-parallel Qwen2.5 models."""
    _model_mapping = MODEL_TENSOR_PARALLEL_MAPPING
    
    @classmethod
    def from_pretrained(cls, model_path, mesh_shape=(1, 8), *model_args, **kwargs):
        """Load a pretrained tensor-parallel model."""
        config = load_qwen_config(model_path)
        model_type = config.get("model_type", "qwen2_5")
        if model_type not in cls._model_mapping:
            raise ValueError(
                f"Unrecognized model type: {model_type}. "
                f"Model type should be one of {', '.join(cls._model_mapping.keys())}."
            )
        
        # Create device mesh for tensor parallelism
        mesh = create_device_mesh(mesh_shape)
        
        # Include mesh in kwargs
        kwargs["mesh"] = mesh
        
        model_class = cls._model_mapping[model_type]
        model = model_class(config, **kwargs)
        
        # Initialize with pretrained weights, adapting to tensor parallelism
        with mesh:
            params = load_qwen_weights(model_path, model)
            model_initialized = init_model_from_weights(model, params)
        
        return model_initialized


# Simple helper function to get a model
def get_model(
    model_type: str = "qwen2_5",
    use_tensor_parallel: bool = False,
    mesh_shape: Tuple[int, int] = (1, 8),
    config: Optional[Dict] = None,
    dtype = None,
    param_dtype = None,
):
    """
    Get a Qwen2.5 model instance.
    
    Args:
        model_type: Type of model to load ("qwen2_5" is the default and only option currently)
        use_tensor_parallel: Whether to use tensor parallelism
        mesh_shape: Shape of the device mesh for tensor parallelism
        config: Optional model configuration (if None, uses default config)
        dtype: Optional data type for model parameters
        param_dtype: Optional parameter storage data type
        
    Returns:
        A Qwen2.5 model instance
    """
    if config is None:
        config = get_qwen2_7b_config()
    
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if param_dtype is not None:
        kwargs["param_dtype"] = param_dtype
    
    if use_tensor_parallel:
        mesh = create_device_mesh(mesh_shape)
        kwargs["mesh"] = mesh
        model_class = MODEL_TENSOR_PARALLEL_MAPPING[model_type]
        return model_class(config, **kwargs)
    else:
        model_class = MODEL_MAPPING[model_type]
        return model_class(config, **kwargs)


# Version
__version__ = "0.1.0"

__all__ = [
    # Model implementation
    "RMSNorm",
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "QwenAttention",
    "QwenMLP",
    "QwenTransformerBlock",
    "Qwen2Model",
    "Qwen2ForCausalLM",
    
    # Tensor parallel implementation
    "TensorParallelQwenAttention",
    "TensorParallelQwenMLP",
    "TensorParallelQwenTransformerBlock",
    "TensorParallelQwen2Model",
    "TensorParallelQwen2ForCausalLM",
    
    # Configuration
    "load_qwen_config",
    "get_mesh_config",
    "create_partition_specs",
    "create_device_mesh",
    "supported_mesh_configs",
    "get_qwen2_7b_config",
    "get_small_config",
    
    # Weight loading
    "load_safetensors_index",
    "convert_weight_name_to_flax",
    "load_qwen_weights",
    "init_model_from_weights",
    "load_safetensors_into_params",
    
    # Auto classes
    "AutoQwenModel",
    "AutoQwenModelTensorParallel",
    "get_model",
    
    # Model mappings
    "MODEL_MAPPING",
    "MODEL_TENSOR_PARALLEL_MAPPING",
    
    # Integration
    "MODEL_NAME",
    "get_supported_mesh_configs",
    "get_tensor_parallel_test_configs",
    "get_model_test_runner",
    "run_tensor_parallel_tests",
    "load_and_run_inference",
    
    # Registration
    "get_model_metadata",
    "register_model_factory",
    "get_model_example",
    
    # Tester
    "Qwen25Tester",
    "Qwen25SmallTester",
]
