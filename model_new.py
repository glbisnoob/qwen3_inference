"""
Qwen3-VL 模型工程化实现 (model_new.py)
采用标准的 torch.nn.Module 架构构建，适合工程部署和生产环境。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import skip_init   # 延迟初始化，避免显存翻倍
from typing import Optional, Tuple, List

# =============================================================================
# RoPE（旋转位置编码）相关实现
# 原本在 model.py 中，迁移至此以使 model_new.py 完全自给自足
# =============================================================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device='cpu'):
    """
    预计算旋转位置编码 (RoPE) 的频率张量

    参数:
        dim: 维度 (通常是 head_dim)
        end: 最大序列长度
        theta: RoPE 基础频率（Qwen 使用 1,000,000）
        device: 计算设备

    返回:
        (freqs_cos, freqs_sin): 两个形状为 [end, dim//2] 的张量
    """
    # 计算频率: 1 / (theta^(2i/dim)), i = 0, 1, ..., dim/2-1
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float().to(device) / dim))
    # 位置索引 0, 1, ..., end-1
    t = torch.arange(end, dtype=torch.float32, device=device)
    # 外积: [end, dim//2]，每个位置的旋转角度
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


class MRoPE:
    """
    M-RoPE 多模态旋转位置编码

    将 head_dim 分成三段，分别编码文本位置/图像高度/图像宽度:
      - 段 T (文本):    使用真实的 position_ids
      - 段 H (图像高): 纯文本输入时固定为位置 0
      - 段 W (图像宽): 纯文本输入时固定为位置 0
    mrope_section 例如 [24, 20, 20] 表示每段的复数对数量，
    实际维度 = 各值 × 2，共 48+40+40 = 128 = head_dim。
    """
    def __init__(self, config, device='cpu'):
        self.head_dim = config.text_config.head_dim
        self.mrope_section = config.text_config.rope_scaling['mrope_section']
        self.rope_theta = config.text_config.rope_theta
        self.device = device

        MAX_POS = 8192  # 支持的最大序列长度

        dim_t = self.mrope_section[0] * 2
        dim_h = self.mrope_section[1] * 2
        dim_w = self.mrope_section[2] * 2

        # 预计算三段的 cos/sin 查找表
        self.cos_t, self.sin_t = precompute_freqs_cis(dim_t, MAX_POS, self.rope_theta, device)
        self.cos_h, self.sin_h = precompute_freqs_cis(dim_h, MAX_POS, self.rope_theta, device)
        self.cos_w, self.sin_w = precompute_freqs_cis(dim_w, MAX_POS, self.rope_theta, device)

    def get_rotary_emb(self, position_ids):
        """
        根据 position_ids 从查找表中提取对应位置的旋转编码。

        参数:
            position_ids: 形状 [batch, seq_len]
        返回:
            (cos, sin): 形状均为 [1, seq_len, 1, head_dim//2]
        """
        # MRoPE 不是 nn.Module，cos/sin 查找表不会随 model.to(device) 自动迁移。
        # 这里手动将查找表移到与 position_ids 相同的设备上（首次调用有少量开销，之后不再搬运）。
        dev = position_ids.device
        if self.cos_t.device != dev:
            self.cos_t = self.cos_t.to(dev)
            self.sin_t = self.sin_t.to(dev)
            self.cos_h = self.cos_h.to(dev)
            self.sin_h = self.sin_h.to(dev)
            self.cos_w = self.cos_w.to(dev)
            self.sin_w = self.sin_w.to(dev)

        indices = position_ids[0]          # 简化: batch=1, shape [seq_len]
        zeros   = torch.zeros_like(indices)

        # 段 T: 查真实位置
        cos_t = self.cos_t[indices]        # [seq_len, 24]
        sin_t = self.sin_t[indices]
        # 段 H/W: 纯文本时固定查位置 0
        cos_h = self.cos_h[zeros]
        sin_h = self.sin_h[zeros]
        cos_w = self.cos_w[zeros]
        sin_w = self.sin_w[zeros]

        cos = torch.cat([cos_t, cos_h, cos_w], dim=-1)  # [seq_len, 64]
        sin = torch.cat([sin_t, sin_h, sin_w], dim=-1)

        # 添加 batch 和 head 维度，便于后续广播
        return cos[None, :, None, :], sin[None, :, None, :]  # [1, seq_len, 1, 64]


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    将旋转位置编码施加到 Query 和 Key 上（复数旋转）。

    参数:
        q: [batch, seq_len, num_heads, head_dim]
        k: [batch, seq_len, num_kv_heads, head_dim]
        cos, sin: [1, seq_len, 1, head_dim//2]
    返回:
        旋转后的 (q, k)，形状不变
    """
    b_q, s_q, h_q, d_q = q.shape
    b_k, s_k, h_k, d_k = k.shape

    # 将向量拆成「复数对」形式: [..., d//2, 2]
    q_pairs = q.reshape(b_q, s_q, h_q, d_q // 2, 2)
    k_pairs = k.reshape(b_k, s_k, h_k, d_k // 2, 2)

    # 复数旋转: (x + iy)(cosθ + i sinθ) = (x cosθ - y sinθ) + i(x sinθ + y cosθ)
    q_out = torch.zeros_like(q_pairs)
    q_out[..., 0] = q_pairs[..., 0] * cos - q_pairs[..., 1] * sin
    q_out[..., 1] = q_pairs[..., 0] * sin + q_pairs[..., 1] * cos

    k_out = torch.zeros_like(k_pairs)
    k_out[..., 0] = k_pairs[..., 0] * cos - k_pairs[..., 1] * sin
    k_out[..., 1] = k_pairs[..., 0] * sin + k_pairs[..., 1] * cos

    return q_out.reshape(b_q, s_q, h_q, d_q), k_out.reshape(b_k, s_k, h_k, d_k)


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

class MoEMLP(nn.Module):
    """
    【教学版】MoE (Mixture of Experts) 架构示例
    将传统的 MLP 替换为多个平行的 Expert，并由 Router (门控网络) 决定每个 token 激活哪些 Expert。
    大模型（如 Mixtral 8x7B, Qwen1.5-MoE）常用此结构来在不显著增加推理计算量的前提下增加参数量。
    """
    def __init__(self, config, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = config.text_config.hidden_size
        
        # 1. 路由器 (Router / Gate)：
        # 一个非常简单的线性层，将隐状态映射为分配给各个专家的“意愿得分”
        self.router = skip_init(nn.Linear, self.hidden_size, self.num_experts, bias=False)
        
        # 2. 专家网络列表 (Experts)：
        # 这里复用了刚刚讲过的 SwiGLUMLP，将 8 个完整的 MLP 平行摆放
        self.experts = nn.ModuleList([SwiGLUMLP(config) for _ in range(num_experts)])

    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_size]
        bsz, seq_len, hidden_dim = x.shape
        x = x.view(-1, hidden_dim)  # 展平成 [batch_size * seq_len, hidden_size]，逐 token 处理
        
        # --- 第1步：Router 选择专家 ---
        # router_logits shape: [num_tokens, num_experts] (每个 token 对 8 个专家的打分)
        router_logits = self.router(x)
        
        # 将打分变为概率分布，并选出概率最大的 top_k 个专家（例如选前 2 个）
        routing_weights = F.softmax(router_logits, dim=-1)
        # topk_weights: 选出的权重, topk_indices: 选出的专家索引
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # 归一化选出专家的权重（使得这 2 个专家的重要性加起来为 1）
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # --- 第2步：分发给专家计算 ---
        final_output = torch.zeros_like(x)  # 初始化输出容器
        
        # TODO: 实际的工程化 MoE 这一步会高度并行化 (比如使用 Triton kernel 或 batched matmul)
        # 这里为了演示清晰，使用人类易读的循环方式
        for i, expert in enumerate(self.experts):
            # 找到当前专家 i 被哪些 token 选中了
            # mask 是一个布尔矩阵，告诉我们哪些 token 的 top_k 里面包含了当前的 expert i
            expert_mask = (topk_indices == i).any(dim=-1)
            
            # 提取真正在这里工作的 token
            flat_tokens = x[expert_mask]
            
            if flat_tokens.numel() > 0: # 如果有 token 分配给这个专家
                # 专家进行正常的 SwiGLU 处理
                expert_out = expert(flat_tokens)
                
                # 找到这些 token 对应的分配权重
                # torch.nonzero 告诉我们对应元素的索引；提取其分发权重
                weight_idx = (topk_indices[expert_mask] == i).nonzero(as_tuple=True)[1]
                weights = topk_weights[expert_mask, weight_idx].unsqueeze(-1)
                
                # 特征加权后，加回最终的结果中
                final_output[expert_mask] += expert_out * weights
                
        # 恢复成原来的三维结构 [batch_size, seq_len, hidden_size]
        return final_output.view(bsz, seq_len, hidden_dim)

class SwiGLUMLP(nn.Module):
    """
    工程化 MLP：使用标准 nn.Linear 层
    """
    def __init__(self, config):
        super().__init__()
        hidden_size = config.text_config.hidden_size
        intermediate_size = config.text_config.intermediate_size
        
        # 同样使用 skip_init，避免随机初始化浪费显存
        self.gate_proj = skip_init(nn.Linear, hidden_size, intermediate_size, bias=False)
        self.up_proj   = skip_init(nn.Linear, hidden_size, intermediate_size, bias=False)
        self.down_proj  = skip_init(nn.Linear, intermediate_size, hidden_size, bias=False)

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

        # skip_init: 创建 nn.Linear 但跳过随机权重初始化，只占内存布局
        # 避免 __init__ 时分配一次随机权重 + load_weights 时再 copy_ 一次导致显存翻倍
        self.q_proj = skip_init(nn.Linear, self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = skip_init(nn.Linear, self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = skip_init(nn.Linear, self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = skip_init(nn.Linear, self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # QK Norm：在 Q/K reshape 成多头后，对每个头的 head_dim 维度单独做归一化
        # 所以 dim 应该是 head_dim，而非 num_heads * head_dim
        self.q_norm = RMSNorm(self.head_dim, eps=config.text_config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.text_config.rms_norm_eps)

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

        # 3. 应用 RoPE
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
        
        # 默认使用单体 MLP (Dense 模型)
        self.mlp = SwiGLUMLP(config)
        
        # 【MoE 切换示例】：如果你想把模型变成 Mixture of Experts
        # 你只需要把上面那行注释掉，打开下面这行：
        # self.mlp = MoEMLP(config, num_experts=8, top_k=2)

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
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        text_config = config.text_config
        
        # skip_init 延迟初始化：不分配随机权重，只占内存布局
        # 对于几十亿参数的大模型，这可以节省一半的显存峰值
        self.embed_tokens = skip_init(nn.Embedding, text_config.vocab_size, text_config.hidden_size)
        self.layers = nn.ModuleList([Qwen3Block(config, i) for i in range(text_config.num_hidden_layers)])
        self.norm = RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.lm_head = skip_init(nn.Linear, text_config.hidden_size, text_config.vocab_size, bias=False)
        
        # 2. 位置编码器 (M-RoPE: 多模态旋转位置编码)
        self.mrope = MRoPE(config, device)

    def forward(self, input_ids, position_ids=None, kv_cache=None):
        bsz, seq_len = input_ids.shape
        
        # 1. 自动计算 position_ids（如未提供）
        if position_ids is None:
            # start_pos: 当前已处理的 token 数量（从 KV Cache 中读取）
            start_pos = 0
            if kv_cache is not None:
                # kv_cache[0] 是第 0 层的 (k_cache, v_cache)
                # k_cache.shape[1] 即缓存的序列长度
                if kv_cache[0][0] is not None:
                    start_pos = kv_cache[0][0].shape[1]
            
            if start_pos > 0:
                # 解码阶段 (Decoding): 每次只处理 1 个新 token
                # position_ids 从上次结束的位置继续递增
                position_ids = torch.arange(
                    start_pos, start_pos + seq_len,
                    device=input_ids.device
                ).reshape(1, seq_len)  # [1, seq_len]
            else:
                # 预填充阶段 (Prefill): 处理完整的输入序列
                # position_ids 从 0 开始
                position_ids = torch.arange(
                    seq_len,
                    device=input_ids.device
                ).reshape(1, seq_len)  # [1, seq_len]
        
        # 2. 获取旋转位置编码 (cos, sin)
        # get_rotary_emb 内部将 position_ids 映射到预计算的频率表
        # 返回形状: [1, seq_len, 1, head_dim//2]
        cos, sin = self.mrope.get_rotary_emb(position_ids)

        # 3. 准备因果掩码 (Causal Mask)
        # 预填充时 seq_len > 1，需要掩码防止未来 token 泄露
        # 解码时 seq_len == 1，无需掩码
        mask = None
        if seq_len > 1:
            # -inf 掩码：在 Attention 计算 softmax 前加到原始分数上
            # 这样被掩盖的位置 softmax 后就会变成 0，即完全不关注该位置
            mask = torch.full(
                (1, 1, seq_len, seq_len), float("-inf"),
                device=input_ids.device
            )
            # triu 保留上三角（包含对角线），diagonal=1 表示从对角线上一格开始
            # 因此，第 i 行的第 j(j>i) 列会被置为 0，这正是我们需要 mask 掉的未来信息
            mask = torch.triu(mask, diagonal=1)
        else:
             # 解码阶段：由于每次只输入 1 个 token，它自己注意自己和它之前的历史 token（都在 KV Cache 中）
             # 所以不需要对当前长度为 1 的序列做 mask
             mask = torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=input_ids.device)

        # 4. 前向计算
        x = self.embed_tokens(input_ids)  # [bsz, seq_len, hidden_size]
        
        for layer in self.layers:
            x = layer(x, cos, sin, mask=mask, kv_cache=kv_cache)
            
        x = self.norm(x)
        logits = self.lm_head(x)  # [bsz, seq_len, vocab_size]
        
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
            
            # 辅助：按需加载（有些模型版本没有 bias，直接跳过即可）
            def _copy(dst_tensor, key):
                if key in weights:
                    dst_tensor.copy_(weights[key])

            # 注意力部分
            layer.attn.q_proj.weight.copy_(weights[f"{prefix}.self_attn.q_proj.weight"])
            _copy(layer.attn.q_proj.bias, f"{prefix}.self_attn.q_proj.bias")
            layer.attn.k_proj.weight.copy_(weights[f"{prefix}.self_attn.k_proj.weight"])
            _copy(layer.attn.k_proj.bias, f"{prefix}.self_attn.k_proj.bias")
            layer.attn.v_proj.weight.copy_(weights[f"{prefix}.self_attn.v_proj.weight"])
            _copy(layer.attn.v_proj.bias, f"{prefix}.self_attn.v_proj.bias")
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
