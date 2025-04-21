import logging
import os
import json
from typing import Dict, Any
from safetensors.flax import load_file
import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Load the safetensors file
    weights_path = '/Users/lu/Documents/tt-bounty-1/qwen2.5-7b'
    safetensors_file = os.path.join(weights_path, 'model-00001-of-00004.safetensors')

    logger.info(f"Loading parameter file: {safetensors_file}")
    weights = load_file(safetensors_file)
    
    # Check all parameters
    logger.info(f"Found {len(weights.keys())} parameters in weight map")
    
    # Check for embedding parameter
    embed_keys = [key for key in weights.keys() if 'embed_tokens' in key]
    logger.info(f"Embedding parameters: {embed_keys}")
    
    if 'model.embed_tokens.weight' in weights:
        logger.info("Embedding parameter exists in safetensors file")
        try:
            embed_shape = weights['model.embed_tokens.weight'].shape
            logger.info(f"Embedding shape: {embed_shape}")
        except Exception as e:
            logger.error(f"Error getting shape: {e}")
    
    # Convert to flattened dict for JAX/Flax format
    try:
        # Create a nested structure with proper keys
        flat_params = {}
        
        # For each parameter in weights
        for key, value in weights.items():
            # Convert from PyTorch to Flax naming
            # For example: model.embed_tokens.weight -> model/embed_tokens/embedding
            parts = key.split('.')
            
            # Special handling for embedding parameter
            if key == 'model.embed_tokens.weight':
                # Use 'embedding' as the parameter name for the embedding layer
                flax_path = ('params', 'model', 'embed_tokens', 'embedding')
            else:
                # For other parameters, replace '.' with '/' and use 'kernel' instead of 'weight'
                if parts[-1] == 'weight':
                    parts[-1] = 'kernel'
                
                # Create path tuple with 'params' as the first element
                flax_path = tuple(['params'] + parts)
            
            # Add to flattened params dictionary
            flat_params[flax_path] = value
        
        # Log first few keys to check structure
        logger.info(f"First few flattened parameter keys: {list(flat_params.keys())[:5]}")
        
        # Unflatten for standard Flax format
        nested_params = unflatten_dict(flat_params)
        
        # Flatten again for inspection
        flattened_params = flatten_dict(nested_params)
        
        # Check that our parameter exists with various paths
        logger.info("Testing parameter lookup...")
        test_keys = [
            ('params', 'model', 'embed_tokens', 'embedding'),
            ('params', 'model', 'embed_tokens', 'weight'),
            ('params', 'model', 'embed_tokens', 'kernel'),
        ]
        
        for key in test_keys:
            value = flattened_params.get(key, None)
            logger.info(f"Lookup with {key}: {'Found' if value is not None else 'None'}")
        
        # Verify the 'params' dictionary structure
        if 'params' in nested_params:
            logger.info("'params' key exists in nested params")
            if 'model' in nested_params['params']:
                logger.info("'model' key exists under 'params'")
                if 'embed_tokens' in nested_params['params']['model']:
                    logger.info("'embed_tokens' key exists under 'params/model'")
                    embed_params = nested_params['params']['model']['embed_tokens']
                    logger.info(f"Keys under 'params/model/embed_tokens': {list(embed_params.keys())}")
            
    except Exception as e:
        logger.error(f"Error processing parameters: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 