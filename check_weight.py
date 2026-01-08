import safetensors.numpy
import os
import numpy as np

model_path = "/home/algo/chunzhuang/qwen/Qwen3-VL-4B-Instruct"
files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
if not files:
    print("No safetensors found")
    exit()

path = os.path.join(model_path, files[0])
print(f"Inspecting {path}...")
with safetensors.numpy.safe_open(path, framework="numpy", device="cpu") as f:
    keys = f.keys()
    print(f"Found {len(keys)} tensors")
    first_key = list(keys)[0]
    try:
        tensor = f.get_tensor(first_key)
        print(f"Key: {first_key}")
        print(f"Dtype: {tensor.dtype}")
        print(f"Shape: {tensor.shape}")
        print(f"Sample: {tensor.flatten()[:5]}")
    except Exception as e:
        print(f"Error loading tensor: {e}")
