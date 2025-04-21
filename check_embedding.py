from safetensors import safe_open
import os
import json

def check_embedding_params(weights_path):
    # Find all safetensors files
    safetensors_files = []
    for root, dirs, files in os.walk(weights_path):
        for file in files:
            if file.endswith('.safetensors'):
                safetensors_files.append(os.path.join(root, file))
    
    print(f"Found {len(safetensors_files)} safetensors files: {safetensors_files}")
    
    # Also check for an index file
    index_file = os.path.join(weights_path, 'model.safetensors.index.json')
    if os.path.exists(index_file):
        print(f"\nFound index file: {index_file}")
        try:
            with open(index_file, 'r') as f:
                index_data = json.load(f)
                print(f"Index structure: {list(index_data.keys())}")
                if 'weight_map' in index_data:
                    embed_entries = {k: v for k, v in index_data['weight_map'].items() if 'embed' in k}
                    print(f"Embedding entries in weight map: {embed_entries}")
        except Exception as e:
            print(f"Error reading index file: {e}")
    
    # Check each file for embedding parameters
    for file_path in safetensors_files:
        print(f"\nChecking file: {file_path}")
        try:
            with safe_open(file_path, framework='numpy') as f:
                # Just print the keys to see what's available
                all_keys = list(f.keys())
                print(f"Number of keys in file: {len(all_keys)}")
                print(f"First few keys: {all_keys[:5] if len(all_keys) > 5 else all_keys}")
                
                embed_keys = [k for k in all_keys if 'embed' in k]
                print(f"Embedding keys: {embed_keys}")
        except Exception as e:
            print(f"Error opening file {file_path}: {e}")

if __name__ == "__main__":
    check_embedding_params('/Users/lu/Documents/tt-bounty-1/qwen2.5-7b') 