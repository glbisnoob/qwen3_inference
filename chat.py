"""
Qwen3-VL 对话推理入口
支持交互式多轮对话
"""
import os
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer

from config import QwenConfig
from utils import load_and_convert_weights
from model import Qwen3VLForConditionalGeneration

def sample_greedy(logits):
    """
    贪心采样: 选择概率最大的 token
    
    参数:
        logits: 模型输出，形状 [1, seq_len, vocab_size]
        
    返回:
        下一个 token ID，形状 [1]
    """
    return torch.argmax(logits[:, -1, :], dim=-1).cpu().numpy()  # torch.argmax: 返回指定维度上最大值的索引; .cpu(): 将张量从GPU移至CPU; .numpy(): 将张量转换为numpy数组


def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description="Qwen3-VL 纯 NumPy 推理")
    parser.add_argument("--model_path", type=str, default="/home/algo/chunzhuang/qwen/Qwen3-VL-4B-Instruct",
                        help="模型文件夹路径")
    args = parser.parse_args()

    # 检查GPU可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # torch.cuda.is_available(): 检查CUDA是否可用
    print(f"使用设备: {device}")

    # 1. 加载配置
    config_path = os.path.join(args.model_path, "config.json")
    print(f"正在从 {config_path} 加载配置...")
    config = QwenConfig(config_path)

    # 2. 加载分词器
    print("正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 3. 加载权重
    print("正在加载模型权重...")
    weights = load_and_convert_weights(args.model_path)
    
    # 4. 初始化模型
    print("正在初始化模型...")
    model = Qwen3VLForConditionalGeneration(config, device)
    model.load_weights(weights)
    
    # 清理权重字典以节省内存
    del weights
    import gc
    gc.collect()

    print("\n模型加载完成！输入 'exit' 或 'quit' 退出对话。")
    
    # 5. 对话循环
    while True:
        try:
            user_input = input("\n用户: ")
            if user_input.lower() in ["exit", "quit"]:
                print("再见！")
                break
                
            # 构建消息
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            messages.append({"role": "user", "content": user_input})
            
            # 使用分词器的对话模板
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # 分词
            model_inputs = tokenizer([text], return_tensors="np")
            input_ids = torch.tensor(model_inputs["input_ids"], device=device)  # torch.tensor: 从数据创建张量并指定设备
            
            # 初始化 KV 缓存 (每层一个 (k, v) 元组)
            kv_cache = [(None, None) for _ in range(config.text_config.num_hidden_layers)]
            
            print("助手: ", end="", flush=True)
            
            # 生成循环
            max_new_tokens = 128
            generated_ids = []
            
            curr_input_ids = input_ids
            position_ids = None  # 由模型自动计算
            
            for _ in range(max_new_tokens):
                # 前向传播
                logits = model.forward(curr_input_ids, position_ids=position_ids, kv_cache=kv_cache)
                
                # 采样下一个 token
                next_token_id = sample_greedy(logits)
                
                # 打印生成的文本
                token_text = tokenizer.decode(next_token_id, skip_special_tokens=True)
                print(token_text, end="", flush=True)
                
                generated_ids.append(next_token_id[0])
                
                # 检查是否遇到结束符
                if next_token_id[0] == config.text_config.eos_token_id or \
                   next_token_id[0] == getattr(config, "vision_end_token_id", -1) or \
                   next_token_id[0] in tokenizer.all_special_ids:
                    break
                
                # 更新输入为新生成的 token
                curr_input_ids = torch.tensor(next_token_id.reshape(1, 1), device=device)  # torch.tensor: 从数据创建张量并指定设备
                position_ids = None  # 由模型根据 KV 缓存自动计算
                
            print()  # 换行
            
        except KeyboardInterrupt:
            print("\n用户中断，退出...")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()