"""
权重加载工具
从 SafeTensors 文件加载模型权重
使用 PyTorch 加载以支持 BFloat16，然后转换为指定精度
"""
import torch
from safetensors import safe_open
import os

def load_and_convert_weights(model_path, device="cuda", dtype=torch.bfloat16):
    """
    从 .safetensors 文件加载权重到 PyTorch Tensor
    
    参数:
        model_path: 模型文件夹路径，包含 .safetensors 文件
        device: 目标设备 ("cuda" 或 "cpu")
        dtype: 目标数据类型 (默认 bfloat16 以节省显存)
        
    返回:
        字典 {权重名称: torch.Tensor}
    """
    weights = {}
    # 查找所有 safetensors 文件
    files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    files.sort()
    
    print(f"正在从 {model_path} 加载权重...")
    
    for file in files:
        path = os.path.join(model_path, file)
        print(f"正在处理 {file}...")
        # 使用 PyTorch 框架加载，直接支持 bfloat16
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                # 转换到目标设备和数据类型
                weights[key] = tensor.to(device=device, dtype=dtype)
                        
    print(f"共加载 {len(weights)} 个张量到 {device}。")
    return weights