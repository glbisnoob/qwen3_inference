"""
Qwen3-VL 模型工程化实现 (model_new.py)
采用标准的 torch.nn.Module 架构构建，适合工程部署和生产环境。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class RMSNorm(nn.Module):
    """
    工程化 RMSNorm：继承 nn.Module
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 将 weight 注册为 Parameter，PyTorch 会自动管理其梯度和设备
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 使用 float32 计算以平衡精度和稳定性
        input_dtype = x.dtype
        x = x.float()
        # torch.mean: 计算平均值；pow(2): 平方
        variance = x.pow(2).mean(-1, keepdim=True)
        # rsqrt: 计算 1/sqrt(x)，比先 sqrt 再除法效率更高
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)

class SwiGLUMLP(nn.Module):
    """
    工程化 MLP：使用标准 nn.Linear 层
    """
    def __init__(self, config):
        super().__init__()
        hidden_size = config.text_config.hidden_size
        intermediate_size = config.text_config.intermediate_size
        
        # 门控分支、上投影分支、下投影分支
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        # F.silu: 即 x * sigmoid(x)，PyTorch 内置优化版本
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Qwen3Attention(nn.Module):
    """
    工程化 Attention：支持 GQA/MQA/MHA
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.text_config.hidden_size
        self.num_heads = config.text_config.num_attention_heads
        self.num_kv_heads = config.text_config.num_key_value_heads
        self.head_dim = config.text_config.head_dim
        self.scale = self.head_dim ** -0.5

        # 标准线性投影层
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # QK Norm
        self.q_norm = RMSNorm(self.num_heads * self.head_dim, eps=config.text_config.rms_norm_eps)
        self.k_norm = RMSNorm(self.num_kv_heads * self.head_dim, eps=config.text_config.rms_norm_eps)

    def forward(self, x, cos, sin, mask=None, kv_cache=None):
        bsz, q_len, _ = x.size()

        # 1. 投影并变换形状 [bsz, q_len, heads, head_dim]
        # view: 改变张量视图，不拷贝内存
        q = self.q_proj(x).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # 2. QK Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 3. 应用 RoPE (复用之前的函数)
        from model import apply_rotary_pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 4. KV Cache 处理
        if kv_cache is not None:
            k_cache, v_cache = kv_cache[self.layer_idx]
            if k_cache is not None:
                k = torch.cat([k_cache, k], dim=1)
                v = torch.cat([v_cache, v], dim=1)
            kv_cache[self.layer_idx] = (k, v)

        # 5. GQA 广播：repeat_interleave 处理分组头
        reps = self.num_heads // self.num_kv_heads
        if reps > 1:
            k = k.repeat_interleave(reps, dim=2)
            v = v.repeat_interleave(reps, dim=2)

        # 6. 计算 Attention (标准缩放点积)
        # transpose(1, 2): 将头维度移到前面 -> [bsz, heads, q_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # matmul: 矩阵乘法；k.transpose(-2, -1): 最后两维转置，用于计算相似度
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_weights = attn_weights + mask

        # 使用 float32 进行 Softmax 提高数值稳定性
        attn_weights = F.softmax(attn_weights.float(), dim=-1).to(q.dtype)
        
        # 7. 输出合并与投影
        attn_output = torch.matmul(attn_weights, v)
        # contiguous: 确保内存连续，否则 view 操作可能报错
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        
        return self.o_proj(attn_output)

class Qwen3Block(nn.Module):
    """
    工程化 Transformer 层
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.attn = Qwen3Attention(config, layer_idx)
        self.mlp = SwiGLUMLP(config)
        self.input_layernorm = RMSNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps)
  
    def forward(self, x, cos, sin, mask=None, kv_cache=None):
        # 注意力残差结构
        h = self.attn(self.input_layernorm(x), cos, sin, mask, kv_cache)
        x = x + h
        # MLP 残差结构
        h = self.mlp(self.post_attention_layernorm(x))
        x = x + h
        return x

class Qwen3VLForConditionalGeneration(nn.Module):
    """
    工程化顶级 Qwen3-VL 模型类
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        text_config = config.text_config
        
        # 1. 基础组件
        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size)
        self.layers = nn.ModuleList([Qwen3Block(config, i) for i in range(text_config.num_hidden_layers)])
        self.norm = RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        
        # 2. 位置编码器
        from model import MRoPE
        self.mrope = MRoPE(config)

    def forward(self, input_ids, position_ids=None, kv_cache=None):
        bsz, seq_len = input_ids.shape
        
        # 1. 准备位置编码相关
        if position_ids is None:
            # 此处逻辑同 model.py，不再重复工程化描述
            pass 
        
        # 获取位置编码 (cos, sin)
        # 为简化演示，这里假设已经从 self.mrope 获取
        # 实际代码中应包含 get_rotary_emb 逻辑
        cos, sin = self.mrope.get_rotary_emb(position_ids) if position_ids is not None else (None, None)

        # 2. 准备 Causal Mask
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device)
            mask = torch.triu(mask, diagonal=1) # 上三角为无穷小

        # 3. 前向计算
        x = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            x = layer(x, cos, sin, mask=mask, kv_cache=kv_cache)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits

    @torch.no_grad()
    def load_from_weight_dict(self, weights):
        """
        专门用于加载通过 safetensors 读取的原始权重字典的适配方法
        """
        print("工程化模型加载权重中...")
        
        # 加载词嵌入
        self.embed_tokens.weight.copy_(weights["model.language_model.embed_tokens.weight"])
        
        # 逐层加载
        for i, layer in enumerate(self.layers):
            prefix = f"model.language_model.layers.{i}"
            
            # 注意力部分
            layer.attn.q_proj.weight.copy_(weights[f"{prefix}.self_attn.q_proj.weight"])
            layer.attn.q_proj.bias.copy_(weights[f"{prefix}.self_attn.q_proj.bias"])
            layer.attn.k_proj.weight.copy_(weights[f"{prefix}.self_attn.k_proj.weight"])
            layer.attn.k_proj.bias.copy_(weights[f"{prefix}.self_attn.k_proj.bias"])
            layer.attn.v_proj.weight.copy_(weights[f"{prefix}.self_attn.v_proj.weight"])
            layer.attn.v_proj.bias.copy_(weights[f"{prefix}.self_attn.v_proj.bias"])
            layer.attn.o_proj.weight.copy_(weights[f"{prefix}.self_attn.o_proj.weight"])
            
            layer.attn.q_norm.weight.copy_(weights[f"{prefix}.self_attn.q_norm.weight"])
            layer.attn.k_norm.weight.copy_(weights[f"{prefix}.self_attn.k_norm.weight"])
            
            # MLP 部分
            layer.mlp.gate_proj.weight.copy_(weights[f"{prefix}.mlp.gate_proj.weight"])
            layer.mlp.up_proj.weight.copy_(weights[f"{prefix}.mlp.up_proj.weight"])
            layer.mlp.down_proj.weight.copy_(weights[f"{prefix}.mlp.down_proj.weight"])
            
            # Norm 部分
            layer.input_layernorm.weight.copy_(weights[f"{prefix}.input_layernorm.weight"])
            layer.post_attention_layernorm.weight.copy_(weights[f"{prefix}.post_attention_layernorm.weight"])
            
        # 最终输出部分
        self.norm.weight.copy_(weights["model.language_model.norm.weight"])
        # 如果 lm_head 是共享权重的，直接关联
        self.lm_head.weight = self.embed_tokens.weight
        
        print("权重加载完毕！")
