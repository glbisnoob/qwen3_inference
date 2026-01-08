# Qwen3-VL 推理系统 (PyTorch GPU 版本)

从零实现的 Qwen3-VL (4B) 大语言模型推理系统。使用 PyTorch 进行 GPU 加速，但核心计算手动实现，避免使用 `torch.nn` 模块。

## 项目结构

```
├── config.py     # 模型配置加载器
├── utils.py      # 权重加载工具 (SafeTensors -> PyTorch)
├── model.py      # 核心模型实现 (Attention, MLP, RMSNorm, M-RoPE)
├── chat.py       # 交互式对话入口
└── README.md     # 本文档
```

## 核心实现

### 手动实现的组件 (不使用 torch.nn)

| 组件 | 实现方式 |
|------|----------|
| **RMSNorm** | `x * weight / sqrt(mean(x²) + eps)` |
| **Softmax** | `exp(x - max) / sum(exp(x - max))` |
| **SiLU** | `x * sigmoid(x)` |
| **矩阵乘法** | `torch.matmul(x, weight.T)` |
| **M-RoPE** | 三段式旋转位置编码 (T/H/W) |
| **GQA** | 分组查询注意力 (repeat KV heads) |
| **QK Norm** | Qwen3 特有的 Query/Key 归一化 |

### Qwen3-VL 特殊配置

| 参数 | 值 | 说明 |
|------|-----|------|
| hidden_size | 2560 | 隐藏维度 |
| num_attention_heads | 32 | 注意力头数 |
| head_dim | 128 | 每头维度 (**非标准: 32×128=4096 ≠ 2560**) |
| num_kv_heads | 8 | KV 头数 (GQA) |
| num_hidden_layers | 36 | 层数 |
| intermediate_size | 9728 | FFN 中间层 |
| vocab_size | 151936 | 词表大小 |

## 使用方法

```bash
# 运行对话
python chat.py --model_path /home/algo/chunzhuang/qwen/Qwen3-VL-4B-Instruct
```

## 已知限制

1. **重复生成** - 贪心采样可能导致输出重复
2. **EOS 检测** - 当前不会在检测到 EOS 时停止
3. **仅支持文本** - 未实现视觉编码器

## 技术细节

- **权重格式**: SafeTensors (BFloat16)
- **推理精度**: BFloat16 (GPU)
- **KV Cache**: 支持自回归生成
- **权重共享**: `lm_head` 与 `embed_tokens` 共享 (tie_word_embeddings)