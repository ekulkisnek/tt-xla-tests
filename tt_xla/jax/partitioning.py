"""Utilities for partitioning computations across devices."""

from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

# For JAX 0.6.0, the function is in a different location
try:
    # Try newer JAX versions
    from jax.sharding import with_sharding_constraint as _with_sharding_constraint
except ImportError:
    try:
        # Try JAX 0.6.0 location
        from jax.experimental.pjit import with_sharding_constraint as _with_sharding_constraint
    except ImportError:
        # Fallback to a dummy implementation 
        def _with_sharding_constraint(x, partition_spec):
            print("WARNING: with_sharding_constraint not available, using dummy implementation")
            return x

def with_sharding_constraint(x, partition_spec, mesh):
    """Apply a sharding constraint to a value.
    
    Args:
        x: The array to apply the constraint to.
        partition_spec: The PartitionSpec to use.
        mesh: The device mesh (ignored in current implementation).
    
    Returns:
        The array with the sharding constraint applied.
    """
    return _with_sharding_constraint(x, partition_spec) 