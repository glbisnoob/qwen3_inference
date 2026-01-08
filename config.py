"""
模型配置加载器
从 config.json 加载 Qwen3-VL 模型配置
"""
import json
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class VisionConfig:
    """视觉编码器配置"""
    depth: int                    # 视觉编码器层数
    hidden_act: str               # 激活函数
    hidden_size: int              # 隐藏层维度
    in_channels: int              # 输入通道数
    initializer_range: float      # 初始化范围
    intermediate_size: int        # 前馈网络中间层维度
    model_type: str               # 模型类型
    num_heads: int                # 注意力头数
    num_position_embeddings: int  # 位置嵌入数量
    out_hidden_size: int          # 输出隐藏层维度
    patch_size: int               # 图像块大小
    spatial_merge_size: int       # 空间合并大小
    temporal_patch_size: int      # 时间维度块大小
    deepstack_visual_indexes: Optional[List[int]] = None  # 深度堆叠视觉索引

@dataclass
class TextConfig:
    """文本模型配置"""
    attention_bias: bool          # 注意力是否使用偏置
    attention_dropout: float      # 注意力 Dropout 率
    bos_token_id: int             # 起始 token ID
    dtype: str                    # 数据类型
    eos_token_id: int             # 结束 token ID
    head_dim: int                 # 每个注意力头的维度
    hidden_act: str               # 激活函数
    hidden_size: int              # 隐藏层维度
    initializer_range: float      # 初始化范围
    intermediate_size: int        # 前馈网络中间层维度
    max_position_embeddings: int  # 最大位置嵌入
    model_type: str               # 模型类型
    num_attention_heads: int      # 注意力头数
    num_hidden_layers: int        # Transformer 层数
    num_key_value_heads: int      # KV 头数 (用于 GQA)
    rms_norm_eps: float           # RMSNorm 的 epsilon
    rope_scaling: Optional[dict]  # RoPE 缩放配置
    rope_theta: float             # RoPE 基础频率
    tie_word_embeddings: bool     # 是否共享词嵌入和 LM Head
    use_cache: bool               # 是否使用 KV 缓存
    vocab_size: int               # 词表大小

class QwenConfig:
    """Qwen3-VL 模型配置类"""
    
    def __init__(self, config_path):
        """
        从 config.json 加载配置
        
        参数:
            config_path: config.json 文件路径
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 顶层配置
        self.architectures = self.config.get("architectures", [])
        self.image_token_id = self.config.get("image_token_id")
        self.model_type = self.config.get("model_type")
        self.video_token_id = self.config.get("video_token_id")
        self.vision_start_token_id = self.config.get("vision_start_token_id")
        self.vision_end_token_id = self.config.get("vision_end_token_id")

        # 子配置
        self.text_config = TextConfig(**self.config.get("text_config", {}))
        self.vision_config = VisionConfig(**self.config.get("vision_config", {}))
