"""
Qwen3 模型核心实现 (仅文本部分)
"""
import torch
import numpy as np
import math

class RMSNorm:
    """
    均方根归一化 (Root Mean Square Normalization)
    公式: x * weight / sqrt(mean(x^2) + eps)
    """
    def __init__(self, weight, eps=1e-6):
        """
        参数:
            weight: 可学习的缩放权重，形状 [hidden_size]
            eps: 防止除零的小常数
        """
        self.weight = weight
        self.eps = eps

    def __call__(self, x):
        """
        前向计算
        
        参数:
            x: 输入张量，形状 [batch, seq_len, dim] 或 [dim]
            
        返回:
            归一化后的张量，形状与输入相同
        """
        # 保持输入的数据类型
        input_dtype = x.dtype
        # 确保x是float类型进行计算
        x_float = x.float()
        
        # 计算方差 (沿最后一个维度) 
        variance = torch.mean(x_float**2, dim=-1, keepdim=True)  # torch.mean: 计算张量的平均值，dim=-1表示沿最后一个维度计算
        # 归一化
        hidden_states = x_float * (1.0 / torch.sqrt(variance + self.eps))  # torch.sqrt: 计算张量元素的平方根
        # 转换回原始类型
        return (self.weight * hidden_states).to(input_dtype)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device='cpu'):
    """
    预计算旋转位置编码 (RoPE) 的频率张量
    
    参数:
        dim: 维度 (通常是 head_dim)
        end: 最大序列长度
        theta: RoPE 基础频率
        device: 计算设备
        
    返回:
        (freqs_cos, freqs_sin): 两个形状为 [end, dim//2] 的张量
    """
    # 计算频率: 1 / (theta^(2i/dim)), i = 0, 1, ..., dim/2-1
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float().to(device) / dim))  # torch.arange: 创建等差数列张量
    # 位置索引
    t = torch.arange(end, dtype=torch.float32, device=device)  # torch.arange: 创建等差数列张量
    # 外积得到 [end, dim//2] 的矩阵
    freqs = torch.outer(t, freqs)  # torch.outer: 计算两个向量的外积
    # 计算 cos 和 sin
    freqs_cos = torch.cos(freqs)   # torch.cos: 计算张量元素的余弦值
    freqs_sin = torch.sin(freqs)   # torch.sin: 计算张量元素的正弦值
    return freqs_cos, freqs_sin

class MRoPE:
    """
    M-RoPE 旋转位置编码
    
    将 head_dim 分成三段，每段独立计算位置编码:
    - 段 1: 序列位置 (使用实际 position_ids)
    - 段 2: 固定位置 0
    - 段 3: 固定位置 0
    
    这种分段方式来自 Qwen 模型架构设计
    """
    def __init__(self, config, device='cpu'):
        """
        参数:
            config: QwenConfig 配置对象
            device: 计算设备
        """
        self.head_dim = config.text_config.head_dim
        self.mrope_section = config.text_config.rope_scaling['mrope_section']
        self.rope_theta = config.text_config.rope_theta
        self.device = device
        
        # mrope_section 是 [24, 20, 20]，表示复数对的数量
        # 实际维度是每个数 * 2，所以 48 + 40 + 40 = 128 = head_dim
        
        # 预计算各段的频率张量
        MAX_POS = 8192  # 最大位置，足够大多数场景
        
        # 段 1: 序列位置 (主要位置编码)
        dim_t = self.mrope_section[0] * 2
        self.cos_t, self.sin_t = precompute_freqs_cis(dim_t, MAX_POS, self.rope_theta, device)
        
        # 段 2: 固定位置 0
        dim_h = self.mrope_section[1] * 2
        self.cos_h, self.sin_h = precompute_freqs_cis(dim_h, MAX_POS, self.rope_theta, device)

        # 段 3: 固定位置 0
        dim_w = self.mrope_section[2] * 2
        self.cos_w, self.sin_w = precompute_freqs_cis(dim_w, MAX_POS, self.rope_theta, device)
        
    def get_rotary_emb(self, position_ids):
        """
        获取旋转位置编码
        
        参数:
            position_ids: 位置索引，形状 [batch, seq_len]
            
        返回:
            (cos, sin): 两个形状为 [1, seq_len, 1, head_dim//2] 的张量
        """
        seq_len = position_ids.shape[1]
        
        # 简化处理 (batch=1)
        indices = position_ids[0]  # [seq_len]
        
        # 段 T: 使用实际位置
        cos_t = self.cos_t[indices]  # [seq_len, 24]
        sin_t = self.sin_t[indices]
        
        # 段 2 和 3: 使用固定位置 0
        zeros = torch.zeros_like(indices)
        cos_h = self.cos_h[zeros]
        sin_h = self.sin_h[zeros]
        
        cos_w = self.cos_w[zeros]
        sin_w = self.sin_w[zeros]
        
        # 拼接得到完整的 [seq_len, 64] (24+20+20 = 64 对)
        cos = torch.cat([cos_t, cos_h, cos_w], dim=-1)  # torch.cat: 沿指定维度拼接张量
        sin = torch.cat([sin_t, sin_h, sin_w], dim=-1)  # torch.cat: 沿指定维度拼接张量
        
        # 添加 batch 和 head 维度，用于广播
        # 输出形状: [1, seq_len, 1, 64]
        return cos[None, :, None, :], sin[None, :, None, :]

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    将旋转位置编码应用到 Query 和 Key
    
    参数:
        q: Query 张量，形状 [batch, seq_len, num_heads, head_dim]
        k: Key 张量，形状 [batch, seq_len, num_kv_heads, head_dim]
        cos, sin: 旋转编码，形状 [batch, seq_len, 1, head_dim//2]
        
    返回:
        旋转后的 (q, k)，形状不变
    """
    # Q 的维度
    b_q, s_q, h_q, d_q = q.shape
    # K 的维度 (可能有不同的头数)
    b_k, s_k, h_k, d_k = k.shape
    
    # 重塑为复数对形式: [..., d//2, 2] 其中最后一维是 (实部, 虚部)
    q_pairs = q.reshape(b_q, s_q, h_q, d_q // 2, 2)
    k_pairs = k.reshape(b_k, s_k, h_k, d_k // 2, 2)
    
    # 复数旋转: (x + iy) * (cos + isin) = (x*cos - y*sin) + i(x*sin + y*cos)
    # x = [..., 0], y = [..., 1]
    
    q_out = torch.zeros_like(q_pairs)
    q_out[..., 0] = q_pairs[..., 0] * cos - q_pairs[..., 1] * sin
    q_out[..., 1] = q_pairs[..., 0] * sin + q_pairs[..., 1] * cos
    
    k_out = torch.zeros_like(k_pairs)
    k_out[..., 0] = k_pairs[..., 0] * cos - k_pairs[..., 1] * sin
    k_out[..., 1] = k_pairs[..., 0] * sin + k_pairs[..., 1] * cos
    
    return q_out.reshape(b_q, s_q, h_q, d_q), k_out.reshape(b_k, s_k, h_k, d_k)


class Attention:
    """
    分组查询注意力 (Grouped Query Attention, GQA)
    
    Q 有 num_heads 个头，K/V 有 num_kv_heads 个头
    K/V 头被复制以匹配 Q 头数
    """
    def __init__(self, config, layer_idx, device='cpu'):
        """
        参数:
            config: 模型配置
            layer_idx: 当前层索引 (用于 KV 缓存)
            device: 计算设备
        """
        self.layer_idx = layer_idx
        self.head_dim = config.text_config.head_dim
        self.num_heads = config.text_config.num_attention_heads
        self.num_kv_heads = config.text_config.num_key_value_heads
        self.hidden_size = config.text_config.hidden_size
        self.scale = self.head_dim ** -0.5  # 注意力缩放因子
        self.device = device
        
        # 权重矩阵 (外部赋值)
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None
        # QK Norm (Qwen3 特有)
        self.q_norm = None
        self.k_norm = None

    # ============================================================================
    # 三种注意力机制实现:
    # 1. MHA (Multi-Head Attention): Q, K, V 都有 num_heads 个头
    # 2. MQA (Multi-Query Attention): Q 有 num_heads 个头, K 和 V 只有 1 个头  
    # 3. GQA (Grouped Query Attention): Q 有 num_heads 个头, K 和 V 有 num_kv_heads 个头
    # 
    # 当前运行: GQA
    # ============================================================================
    
    # def forward_mha(self, x, cos, sin, mask=None, kv_cache=None):
    #     """
    #     Multi-Head Attention (MHA) 前向计算
    #     所有投影都使用 num_heads 个头,K 和 V 完全独立
    #     
    #     参数:
    #         x: 输入张量,形状 [batch, seq_len, hidden_size]
    #         cos, sin: 旋转位置编码
    #         mask: 注意力掩码,形状 [1, 1, seq_len, seq_len]
    #         kv_cache: KV 缓存列表
    #         
    #     返回:
    #         输出张量,形状 [batch, seq_len, hidden_size]
    #     """
    #     b, s, _ = x.shape
    #     
    #     # QKV 投影
    #     q = torch.matmul(x, self.q_proj.T)  # [b, s, num_heads * head_dim]
    #     k = torch.matmul(x, self.k_proj.T)  # [b, s, num_heads * head_dim]
    #     v = torch.matmul(x, self.v_proj.T)  # [b, s, num_heads * head_dim]
    #     
    #     # 重塑为多头形式 - MHA: K 和 V 也使用 num_heads
    #     q = q.reshape(b, s, self.num_heads, self.head_dim)
    #     k = k.reshape(b, s, self.num_heads, self.head_dim)  # 注意: 使用 num_heads 而非 num_kv_heads
    #     v = v.reshape(b, s, self.num_heads, self.head_dim)  # 注意: 使用 num_heads 而非 num_kv_heads
    #     
    #     # QK Norm (Qwen3 特有)
    #     if self.q_norm is not None:
    #         q = self.q_norm(q)
    #     if self.k_norm is not None:
    #         k = self.k_norm(k)
    #     
    #     # 应用旋转位置编码
    #     q, k = apply_rotary_pos_emb(q, k, cos, sin)
    #     
    #     # 更新 KV 缓存
    #     if kv_cache is not None:
    #         k_cache, v_cache = kv_cache[self.layer_idx]
    #         if k_cache is not None:
    #             k = torch.cat([k_cache, k], dim=1)
    #             v = torch.cat([v_cache, v], dim=1)
    #         kv_cache[self.layer_idx] = (k, v)
    #     
    #     # MHA: 无需复制 K 和 V,因为它们已经有 num_heads 个头
    #     # 直接转置即可
    #     q = q.transpose(1, 2)  # [b, num_heads, seq_len, head_dim]
    #     k = k.transpose(1, 2)  # [b, num_heads, seq_len, head_dim]
    #     v = v.transpose(1, 2)  # [b, num_heads, seq_len, head_dim]
    #     
    #     # 计算注意力分数
    #     att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
    #     
    #     # 应用掩码
    #     if mask is not None:
    #         att = att + mask
    #     
    #     # Softmax
    #     att_float = att.float()
    #     att_max = torch.max(att_float, dim=-1, keepdim=True)[0]
    #     att_exp = torch.exp(att_float - att_max)
    #     att_weights = att_exp / torch.sum(att_exp, dim=-1, keepdim=True)
    #     att_weights = att_weights.to(v.dtype)
    #     
    #     # 加权求和
    #     out = torch.matmul(att_weights, v)
    #     
    #     # 合并多头
    #     out = out.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
    #     
    #     # 输出投影
    #     return torch.matmul(out, self.o_proj.T)
    
    # def forward_mqa(self, x, cos, sin, mask=None, kv_cache=None):
    #     """
    #     Multi-Query Attention (MQA) 前向计算
    #     Q 有 num_heads 个头,K 和 V 只有 1 个头,需要广播到所有查询头上
    #     
    #     参数:
    #         x: 输入张量,形状 [batch, seq_len, hidden_size]
    #         cos, sin: 旋转位置编码
    #         mask: 注意力掩码,形状 [1, 1, seq_len, seq_len]
    #         kv_cache: KV 缓存列表
    #         
    #     返回:
    #         输出张量,形状 [batch, seq_len, hidden_size]
    #     """
    #     b, s, _ = x.shape
    #     
    #     # QKV 投影
    #     q = torch.matmul(x, self.q_proj.T)  # [b, s, num_heads * head_dim]
    #     k = torch.matmul(x, self.k_proj.T)  # [b, s, 1 * head_dim]
    #     v = torch.matmul(x, self.v_proj.T)  # [b, s, 1 * head_dim]
    #     
    #     # 重塑为多头形式 - MQA: K 和 V 只有 1 个头
    #     q = q.reshape(b, s, self.num_heads, self.head_dim)
    #     k = k.reshape(b, s, 1, self.head_dim)  # 只有 1 个头
    #     v = v.reshape(b, s, 1, self.head_dim)  # 只有 1 个头
    #     
    #     # QK Norm (Qwen3 特有)
    #     if self.q_norm is not None:
    #         q = self.q_norm(q)
    #     if self.k_norm is not None:
    #         k = self.k_norm(k)
    #     
    #     # 应用旋转位置编码
    #     q, k = apply_rotary_pos_emb(q, k, cos, sin)
    #     
    #     # 更新 KV 缓存
    #     if kv_cache is not None:
    #         k_cache, v_cache = kv_cache[self.layer_idx]
    #         if k_cache is not None:
    #             k = torch.cat([k_cache, k], dim=1)
    #             v = torch.cat([v_cache, v], dim=1)
    #         kv_cache[self.layer_idx] = (k, v)
    #     
    #     # MQA: 将单个 K 和 V 头复制到所有查询头
    #     # k: [b, seq, 1, head_dim] -> [b, seq, num_heads, head_dim]
    #     k = torch.repeat_interleave(k, self.num_heads, dim=2)
    #     v = torch.repeat_interleave(v, self.num_heads, dim=2)
    #     
    #     # 转置以计算注意力
    #     q = q.transpose(1, 2)  # [b, num_heads, seq_len, head_dim]
    #     k = k.transpose(1, 2)  # [b, num_heads, seq_len, head_dim]
    #     v = v.transpose(1, 2)  # [b, num_heads, seq_len, head_dim]
    #     
    #     # 计算注意力分数
    #     att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
    #     
    #     # 应用掩码
    #     if mask is not None:
    #         att = att + mask
    #     
    #     # Softmax
    #     att_float = att.float()
    #     att_max = torch.max(att_float, dim=-1, keepdim=True)[0]
    #     att_exp = torch.exp(att_float - att_max)
    #     att_weights = att_exp / torch.sum(att_exp, dim=-1, keepdim=True)
    #     att_weights = att_weights.to(v.dtype)
    #     
    #     # 加权求和
    #     out = torch.matmul(att_weights, v)
    #     
    #     # 合并多头
    #     out = out.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
    #     
    #     # 输出投影
    #     return torch.matmul(out, self.o_proj.T)
    
    def forward(self, x, cos, sin, mask=None, kv_cache=None):
        """
        Grouped Query Attention (GQA) 前向计算 - 当前运行的版本
        Q 有 num_heads 个头,K 和 V 有 num_kv_heads 个头 (介于 MHA 和 MQA 之间)
        
        参数:
            x: 输入张量,形状 [batch, seq_len, hidden_size]
            cos, sin: 旋转位置编码
            mask: 注意力掩码,形状 [1, 1, seq_len, seq_len]
            kv_cache: KV 缓存列表
            
        返回:
            输出张量,形状 [batch, seq_len, hidden_size]
        """
        b, s, _ = x.shape
        
        # QKV 投影 - 使用基本的矩阵乘法而非torch.nn.Linear
        # 矩阵乘法: [b, s, h] @ [h, out]^T -> [b, s, out]
        q = torch.matmul(x, self.q_proj.T)  # torch.matmul: 执行矩阵乘法
        k = torch.matmul(x, self.k_proj.T)  # torch.matmul: 执行矩阵乘法
        v = torch.matmul(x, self.v_proj.T)  # torch.matmul: 执行矩阵乘法
        
        # 重塑为多头形式
        # q: [b, s, num_heads, head_dim]
        q = q.reshape(b, s, self.num_heads, self.head_dim)  # reshape: 重塑张量形状
        k = k.reshape(b, s, self.num_kv_heads, self.head_dim)  # reshape: 重塑张量形状
        v = v.reshape(b, s, self.num_kv_heads, self.head_dim)  # reshape: 重塑张量形状
        
        # QK Norm (Qwen3 特有) - 在 RoPE 之前应用
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        
        # 应用旋转位置编码
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # 更新 KV 缓存
        if kv_cache is not None:
            k_cache, v_cache = kv_cache[self.layer_idx]
            if k_cache is not None:
                # 拼接历史缓存
                k = torch.cat([k_cache, k], dim=1)  # torch.cat: 沿指定维度拼接张量
                v = torch.cat([v_cache, v], dim=1)  # torch.cat: 沿指定维度拼接张量
            
            # 更新缓存
            kv_cache[self.layer_idx] = (k, v)
        
        # 分组查询注意力 (GQA): 复制 KV 头以匹配 Q 头数
        # k: [b, total_seq, kv_heads, head_dim] -> [b, total_seq, num_heads, head_dim]
        reps = self.num_heads // self.num_kv_heads
        if reps > 1:
            k = torch.repeat_interleave(k, reps, dim=2)  # torch.repeat_interleave: 沿指定维度重复张量元素
            v = torch.repeat_interleave(v, reps, dim=2)  # torch.repeat_interleave: 沿指定维度重复张量元素
        
        # 转置以计算注意力: [b, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)  # transpose: 交换张量的两个维度
        k = k.transpose(1, 2)  # transpose: 交换张量的两个维度
        v = v.transpose(1, 2)  # transpose: 交换张量的两个维度
        
        # 计算注意力分数
        # q @ k.T -> [b, heads, s_q, s_k]
        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # torch.matmul: 执行矩阵乘法
        
        # 应用掩码 (如有)
        if mask is not None:
            # mask: [1, 1, s_q, s_k]
            att = att + mask  # 张量加法
        
        # 稳定 Softmax: 减去最大值防止溢出 - 手动实现而非使用torch.softmax
        att_float = att.float()  # 转换为 float32 进行 softmax 计算
        att_max = torch.max(att_float, dim=-1, keepdim=True)[0]
        att_exp = torch.exp(att_float - att_max)
        att_weights = att_exp / torch.sum(att_exp, dim=-1, keepdim=True)
        att_weights = att_weights.to(v.dtype)  # 转换回原始类型

        # 推荐写法
        #att_weights = torch.softmax(att.float(), dim=-1).to(v.dtype)
        
        # 加权求和
        # att @ v -> [b, heads, s_q, head_dim]
        out = torch.matmul(att_weights, v)
        
        # 合并多头
        # Qwen3-VL 特殊: num_heads * head_dim (4096) != hidden_size (2560)
        # 所以需要 reshape 到 [b, s, num_heads * head_dim],然后由 o_proj 投影回 hidden_size
        out = out.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
        
        # 输出投影: [b, s, num_heads * head_dim] @ [hidden_size, num_heads * head_dim]^T -> [b, s, hidden_size]
        return torch.matmul(out, self.o_proj.T)

class MLP:
    """
    SwiGLU 前馈网络
    公式: (Swish(x @ gate) * (x @ up)) @ down
    其中 Swish = SiLU = x * sigmoid(x)
    """
    def __init__(self, config):
        """权重矩阵 (外部赋值)"""
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None
    
    def forward(self, x):
        """
        前向计算
        
        参数:
            x: 输入张量，形状 [batch, seq_len, hidden_size]
            
        返回:
            输出张量，形状相同
        """
        # 门控和上投影
        gate = torch.matmul(x, self.gate_proj.T)  # torch.matmul: 执行矩阵乘法
        up = torch.matmul(x, self.up_proj.T)  # torch.matmul: 执行矩阵乘法
        
        # SiLU 激活: x * sigmoid(x) - 手动实现而非使用torch.nn.SiLU
        gate = gate * torch.sigmoid(gate)  # torch.sigmoid: 计算张量元素的sigmoid值
        
        # 下投影
        return torch.matmul((gate * up), self.down_proj.T)  # torch.matmul: 执行矩阵乘法

class QwenBlock:
    """
    Transformer 层 (Decoder Block)
    结构: x + Attn(Norm1(x)) + MLP(Norm2(x))
    """
    def __init__(self, config, layer_idx, device='cpu'):
        """
        参数:
            config: 模型配置
            layer_idx: 当前层索引
            device: 计算设备
        """
        self.layer_idx = layer_idx
        self.norm1 = None  # 注意力前的 RMSNorm
        self.norm2 = None  # MLP 前的 RMSNorm
        self.attn = Attention(config, layer_idx, device)
        self.mlp = MLP(config)
        
    def forward(self, x, cos, sin, mask=None, kv_cache=None):
        """
        前向计算
        
        参数:
            x: 输入张量
            cos, sin: 旋转位置编码
            mask: 注意力掩码
            kv_cache: KV 缓存
            
        返回:
            输出张量
        """
        # 注意力 + 残差
        h = self.attn.forward(self.norm1(x), cos, sin, mask, kv_cache)
        x = x + h  # 张量加法
        
        # MLP + 残差
        h = self.mlp.forward(self.norm2(x))
        x = x + h  # 张量加法
        return x

class Qwen3VLModel:
    """Qwen3-VL 模型主体 (不含 LM Head)"""
    
    def __init__(self, config, device='cpu'):
        """
        参数:
            config: 模型配置
            device: 计算设备
        """
        self.config = config
        self.embed_tokens = None  # 词嵌入矩阵 [vocab_size, hidden_size]
        self.layers = [QwenBlock(config, i, device) for i in range(config.text_config.num_hidden_layers)]
        self.norm = None  # 最终 RMSNorm
        self.device = device
        
    def forward(self, input_ids, position_ids=None, kv_cache=None, mask=None):
        """
        前向计算 (供内部使用)
        
        参数:
            input_ids: 输入 token ID，形状 [batch, seq_len]
            
        返回:
            隐藏状态
        """
        # 1. 词嵌入
        x = self.embed_tokens[input_ids]  # [batch, seq_len, hidden_size]
        
        # 2. RoPE 频率 (由调用者提供)
        pass

class Qwen3VLForConditionalGeneration:
    """
    Qwen3-VL 条件生成模型
    包含完整的推理流程: 嵌入 -> Transformer -> LM Head
    """
    
    def __init__(self, config, device='cpu'):
        """
        参数:
            config: QwenConfig 配置对象
            device: 计算设备
        """
        self.config = config
        self.model = Qwen3VLModel(config, device)
        self.lm_head = None  # 语言模型头 [vocab_size, hidden_size]
        self.mrope = MRoPE(config, device)  # 多模态位置编码
        self.device = device
        
    def load_weights(self, weights):
        """
        加载模型权重
        
        参数:
            weights: 字典 {权重名称: torch.Tensor}
        """
        print("正在分配权重...")
        
        # 1. 词嵌入 - 权重已经是 Tensor，直接使用
        # 注意: Qwen3-VL 使用 model.language_model. 前缀
        self.model.embed_tokens = weights["model.language_model.embed_tokens.weight"]
        
        # 2. 各层权重
        for i, layer in enumerate(self.model.layers):
            prefix = f"model.language_model.layers.{i}"
            
            # 注意力权重
            layer.attn.q_proj = weights[f"{prefix}.self_attn.q_proj.weight"]
            layer.attn.k_proj = weights[f"{prefix}.self_attn.k_proj.weight"]
            layer.attn.v_proj = weights[f"{prefix}.self_attn.v_proj.weight"]
            layer.attn.o_proj = weights[f"{prefix}.self_attn.o_proj.weight"]
            
            # QK Norm (Qwen3 特有)
            layer.attn.q_norm = RMSNorm(weights[f"{prefix}.self_attn.q_norm.weight"], 
                                        eps=self.config.text_config.rms_norm_eps)
            layer.attn.k_norm = RMSNorm(weights[f"{prefix}.self_attn.k_norm.weight"], 
                                        eps=self.config.text_config.rms_norm_eps)
            
            # MLP 权重
            layer.mlp.gate_proj = weights[f"{prefix}.mlp.gate_proj.weight"]
            layer.mlp.up_proj = weights[f"{prefix}.mlp.up_proj.weight"]
            layer.mlp.down_proj = weights[f"{prefix}.mlp.down_proj.weight"]
            
            # 归一化权重
            layer.norm1 = RMSNorm(weights[f"{prefix}.input_layernorm.weight"], 
                                  eps=self.config.text_config.rms_norm_eps)
            layer.norm2 = RMSNorm(weights[f"{prefix}.post_attention_layernorm.weight"], 
                                  eps=self.config.text_config.rms_norm_eps)
            
        # 3. 最终归一化 & LM Head
        self.model.norm = RMSNorm(weights["model.language_model.norm.weight"], 
                                  eps=self.config.text_config.rms_norm_eps)
        # LM Head 与词嵌入共享 (tie_word_embeddings=true)
        self.lm_head = self.model.embed_tokens
        
        print("权重分配完成。")
        
    def forward(self, input_ids, position_ids=None, kv_cache=None):
        """
        前向计算
        
        参数:
            input_ids: 输入 token ID，形状 [batch, seq_len]
            position_ids: 位置 ID (可选，自动计算)
            kv_cache: KV 缓存 (用于自回归生成)
            
        返回:
            logits: 形状 [batch, seq_len, vocab_size]
        """
        batch, seq_len = input_ids.shape
        
        # 计算位置 ID (如未提供)
        if position_ids is None:
            start_pos = 0
            if kv_cache is not None:
                # 从 KV 缓存获取已处理的序列长度
                if kv_cache[0][0] is not None:
                    start_pos = kv_cache[0][0].shape[1]
            
            if start_pos > 0:
                # 解码阶段: 新增 token 的位置
                position_ids = torch.arange(start_pos, start_pos + seq_len, device=self.device).reshape(1, seq_len)  # torch.arange: 创建等差数列张量; reshape: 重塑张量形状
            else:
                # 预填充阶段
                position_ids = torch.arange(seq_len, device=self.device).reshape(1, seq_len)  # torch.arange: 创建等差数列张量; reshape: 重塑张量形状
        
        # 构建因果掩码 (仅在预填充时需要)
        mask = None
        if seq_len > 1:
            # 下三角掩码: 可见位置为 0，被遮蔽位置为 -1e9
            mask = torch.full((1, 1, seq_len, seq_len), -1e9, dtype=torch.float32, device=self.device)  # torch.full: 创建指定值填充的张量
            mask = torch.triu(mask, diagonal=1)  # torch.triu: 返回矩阵的上三角部分
        else:
            mask = torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=self.device)  # torch.zeros: 创建零张量
        
        # 获取旋转位置编码
        cos, sin = self.mrope.get_rotary_emb(position_ids)
        
        # 前向传播
        x = self.model.embed_tokens[input_ids]
        
        for i, layer in enumerate(self.model.layers):
            x = layer.forward(x, cos, sin, mask, kv_cache)
            
        # 最终归一化
        x = self.model.norm(x)
        
        # LM Head: 将隐藏状态映射到词表
        # x: [batch, seq_len, hidden_size]
        logits = torch.matmul(x, self.lm_head.T)  # torch.matmul: 执行矩阵乘法
        
        return logits

        