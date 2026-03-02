"""
完整的 Qwen3-VL 推理示例
此脚本展示了加载权重并使用 model_new.py 进行推理的完整流程。
为教学目的，包含详细的注释。

运行方式:
  python inference_complete.py --model_path <模型文件夹路径>
"""

import os
import argparse
import torch
from transformers import AutoTokenizer

from config import QwenConfig
from utils import load_and_convert_weights

from model_new import Qwen3VLForConditionalGeneration

def sample_greedy(logits):
    """
    贪心采样 (Greedy Sampling): 选择概率最大的 token
    
    原理: 模型输出为形状 [batch, seq_len, vocab_size] 的 logits。
    我们在最后一个位置取出每个词的预测分数，然后选取得分最高的词汇 ID。
    
    参数:
        logits: 模型输出的未归一化分数，形状 [1, seq_len, vocab_size]
        
    返回:
        预测的下一个 token ID，形状 [1]
    """
    # [:, -1, :] 表示取 batch 的所有元素、最后一个时间步（最新预测）、所有词汇分数
    # argmax 寻找最后一个维度（vocab_size）上的最大值索引
    next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
    
    # 转换为 numpy 数组返回，以兼容现有逻辑
    return next_token_id.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 完整推理示例")
    parser.add_argument("--model_path", type=str, default="/home/algo/chunzhuang/qwen/Qwen3-VL-4B-Instruct",
                        help="包含 config.json 和 safetensors 的模型文件夹路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="运行设备，默认为 cuda (若可用) 否则为 cpu")
    args = parser.parse_args()

    device = args.device

    config_path = os.path.join(args.model_path, "config.json")
    if not os.path.exists(config_path):
        print(f"错误: 找不到配置文件 {config_path}，无法继续。")
        print("注意：如果没有真实模型权重，由于是从本地加载配置，脚本无法运行。本脚本提供展示流程之用。")
        return

    config = QwenConfig(config_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # -------- 正确的显存控制顺序 --------
    # 1. 权重先加载到 CPU（不占 GPU 显存）
    weights = load_and_convert_weights(args.model_path, device="cpu")
    
    # 2. 模型结构初始化（skip_init，不分配随机权重）
    model = Qwen3VLForConditionalGeneration(config, device="cpu")
    
    # 3. 把空结构从 CPU 搬到目标 GPU（此时 GPU 几乎不用额外显存，因为 skip_init 的内容未初始化）
    model.to(device)
    
    # 4. copy_ 把 CPU 上的权重逐层拷进 GPU 上的模型（跨设备 copy_ PyTorch 原生支持）
    model.load_from_weight_dict(weights)
    
    # 5. 释放 CPU 上的权重字典，避免白占内存
    del weights
    import gc; gc.collect()

    model.eval()
    
    while True:
        try:
            user_input = input("\n[You]: ")
            if user_input.lower() in ["exit", "quit"]:
                print("结束对话。")
                break
            if not user_input.strip():
                continue
                
            # --- 4.1 Tokenization 阶段 ---
            # 按照模型的 System Prompt 和 Chat Template 构建对话历史
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ]
            
            # apply_chat_template 依据模型词表里特殊 token 的结构（如 <|im_start|>，<|im_end|> 等）渲染出字符串
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # 将字符串转换为模型可识别的 token id
            # 返回的是字典如 {'input_ids': [[...]]} 包含 batch 维度
            model_inputs = tokenizer(prompt_text, return_tensors="pt")
            input_ids = model_inputs["input_ids"].to(device)
            
            # --- 4.2 初始化 KV Cache ---
            # KV Cache 用于保存历史的 Key 和 Value 矩阵，避免重复计算，加速生成速度
            # 格式: list[tuple[Tensor, Tensor]] 每层一个 (K_cache, V_cache)，初始全部为 None
            num_layers = config.text_config.num_hidden_layers
            kv_cache = [(None, None)] * num_layers
            
            print("[Qwen]: ", end="", flush=True)
            
            # --- 4.3 生成循环（Prefill + Decode 结合） ---
            max_new_tokens = 256
            generated_ids = []
            
            # 首次输入是完整的 Prompt 序列（Prefill阶段）
            curr_input_ids = input_ids
            # 由内部逻辑根据 kv_cache 内容决定使用从 0 开始自增，还是衔接上次的
            position_ids = None 
            
            # 需关闭梯度计算以降低显存和加速
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    # 前向传播 (Prefill/Decode)
                    # Prefill 阶段: curr_input_ids 为完整 prompt, seq_len > 1
                    # Decode 阶段: curr_input_ids 为前一轮生成的 1 个 token, seq_len == 1
                    logits = model.forward(
                        input_ids=curr_input_ids,
                        position_ids=position_ids,
                        kv_cache=kv_cache
                    )
                    
                    # 使用贪心采样（只取概率最大者，无需考虑温度或 top-p）
                    next_token_id = sample_greedy(logits)
                    generated_ids.append(next_token_id[0])
                    
                    # 流式解码当前的 1 个 token 文本并打印
                    token_text = tokenizer.decode(next_token_id, skip_special_tokens=True)
                    print(token_text, end="", flush=True)
                    
                    # 检查停止条件
                    if (next_token_id[0] == config.text_config.eos_token_id or
                        next_token_id[0] == getattr(config, "vision_end_token_id", -1) or
                        next_token_id[0] in tokenizer.all_special_ids):
                        break
                    
                    # 更新当前输入给下一轮循环（仅仅是刚预测出的那 1 个 token）
                    curr_input_ids = torch.tensor([[next_token_id[0]]], device=device)
                    # 重置为 None 后 forward 内部会依据 KV Cache 的长度推断正确的绝对位置
                    position_ids = None 
                    
            print() # 补充换行
            
        except KeyboardInterrupt:
            print("\n>> 用户强行中断。结束。")
            break
        except Exception as e:
            print(f"\n>> 错误发生: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
