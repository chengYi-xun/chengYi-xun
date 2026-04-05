---
title: 笔记｜多模态融合（五）：原生多模态——从 Flamingo 到 Chameleon
date: 2026-04-06 00:15:00
categories:
 - Tutorials
tags:
 - Flamingo
 - Chameleon
 - Perceiver Resampler
 - VQ-VAE
 - Native Multimodal
series: "多模态融合"
mathjax: true
---

> **论文**：
> - *Flamingo: a Visual Language Model for Few-Shot Learning*（Alayrac et al., 2022, DeepMind）
> - *Chameleon: Mixed-Modal Early-Fusion Foundation Models*（Meta, 2024）
>
> **代码**：[lucidrains/flamingo-pytorch](https://github.com/lucidrains/flamingo-pytorch) · [facebookresearch/chameleon](https://github.com/facebookresearch/chameleon)
> **前置知识**：[上一篇：LLaVA](posts/32-llava/)

---

## 0. 翻译官 vs 双语母语者

前面三篇介绍的模型——CLIP、BLIP-2、LLaVA——都遵循同一个范式：**视觉编码器和语言模型是独立预训练的**，通过某种中间模块（对比学习/Q-Former/MLP）将它们连接起来。

这就像两个只说各自语言的专家，中间需要一个翻译官。翻译官再好，也会有信息损失。

有没有一种模型，**从第一层开始就天然理解图像和文字**，就像一个双语母语者？

本篇介绍两种接近这个目标的方案：

| 模型 | 方案 | 类比 |
|------|------|------|
| Flamingo | 在冻结 LLM 中插入交叉注意力层 | 翻译官驻场在一个专家旁边 |
| Chameleon | 将图像 token 化后与文本共享同一个 Transformer | 真正的双语母语者 |

---

## 1. Flamingo：门控交叉注意力

### 1.1 核心问题

Flamingo 面对的场景是：你有一个**超大的冻结 LLM**（如 Chinchilla-70B），不想动它的参数。如何让它处理视觉信息？

LLaVA 的方案是把视觉 token 直接拼在文本前面。但 Flamingo 认为这不够优雅——它希望视觉信息能在 LLM 的**每一层**被参考到，而不只是作为输入前缀。

### 1.2 Perceiver Resampler

首先需要将不定长的视觉特征压缩为定长表示。Flamingo 使用 **Perceiver Resampler**——一个基于交叉注意力的压缩模块。

设视觉编码器输出 $\mathbf{F} \in \mathbb{R}^{N_v \times d_v}$，Perceiver Resampler 使用 $N_q$ 个可学习的查询向量 $\mathbf{Q}_0 \in \mathbb{R}^{N_q \times d}$（Flamingo 中 $N_q = 64$）。

经过 $L_r$ 层交叉注意力 + 自注意力：

$$
\begin{aligned}
\tilde{\mathbf{Q}}_l &= \mathbf{Q}_{l-1} + \text{CrossAttn}(\mathbf{Q}_{l-1}, \mathbf{F}) \\
\mathbf{Q}_l &= \tilde{\mathbf{Q}}_l + \text{SelfAttn}(\tilde{\mathbf{Q}}_l) + \text{FFN}(\tilde{\mathbf{Q}}_l)
\end{aligned}
$$

最终输出 $\mathbf{Q}_L \in \mathbb{R}^{64 \times d}$——将视觉特征压缩为 64 个固定维度的 token。

**与 Q-Former 的区别**：

| 维度 | Q-Former (BLIP-2) | Perceiver Resampler (Flamingo) |
|------|-------------------|-------------------------------|
| 查询数 | 32 | 64 |
| 与文本的交互 | 有（共享 self-attention） | 无（独立压缩） |
| 交叉注意力来源 | 冻结 ViT | 冻结 NFNet / ViT |
| 输出用途 | LLM 的 soft prompt | Gated Cross-Attention 的 KV |

### 1.3 Gated Cross-Attention

Flamingo 的核心创新：在冻结 LLM 的每一层之间，插入**新的交叉注意力层**。

设 LLM 第 $l$ 层的 self-attention 输出为 $\mathbf{x}_l \in \mathbb{R}^{L \times d}$（文本 token），Perceiver Resampler 的视觉输出为 $\mathbf{Q} \in \mathbb{R}^{64 \times d}$。

Gated Cross-Attention 的计算：

$$
\begin{aligned}
\mathbf{x}_l' &= \mathbf{x}_l + \tanh(\alpha_{\text{attn}}) \cdot \text{CrossAttn}(\mathbf{x}_l, \mathbf{Q}) \\
\mathbf{x}_l'' &= \mathbf{x}_l' + \tanh(\alpha_{\text{ffn}}) \cdot \text{FFN}(\mathbf{x}_l')
\end{aligned}
$$

其中 $\alpha_{\text{attn}}$ 和 $\alpha_{\text{ffn}}$ 是**可学习的标量门控参数**，初始化为 0。

**tanh 门控的精妙之处**：

1. **初始化为 0**：$\tanh(0) = 0$，意味着训练开始时交叉注意力的贡献为零——模型的行为与原始冻结 LLM 完全一致
2. **渐进式引入**：随着训练进行，$\alpha$ 可以逐渐增大，让视觉信息缓慢地"渗透"进 LLM
3. **稳定性**：$\tanh$ 的值域为 $(-1, 1)$，防止视觉信息过度干扰 LLM 的原始行为

这种设计哲学是：**不破坏预训练 LLM 的知识，而是在其基础上"附加"视觉理解能力。**

### 1.4 交错的图文输入

Flamingo 的另一个特点是支持**交错的图文输入**——文本中可以穿插多张图片：

```
<image_1> 这是一只猫。
<image_2> 这也是一只猫。
问：两只猫有什么区别？
```

实现方式：每个 `<image>` 标记关联一组视觉 token（通过 Perceiver Resampler 产生），在 Gated Cross-Attention 中，每个文本 token 只关注**它之前最近的那张图片**的视觉 token。

### 1.5 Few-Shot 能力

交错图文输入直接支持了 **few-shot learning**：

```
<image_1> 这是一只暹罗猫。
<image_2> 这是一只英短蓝猫。
<image_3> 这是什么品种的猫？
```

模型从前两个图文示例中学到"模式"，然后应用到第三个问题上——无需任何微调。

### 1.6 代码实现

```python
import torch
import torch.nn as nn

class PerceiverResampler(nn.Module):
    """Perceiver Resampler：将视觉特征压缩为固定长度"""
    def __init__(self, d_model, n_queries=64, n_layers=6, n_heads=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.layers = nn.ModuleList([
            PerceiverLayer(d_model, n_heads) for _ in range(n_layers)
        ])

    def forward(self, visual_features):
        # visual_features: [B, N_v, d]
        B = visual_features.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, 64, d]

        for layer in self.layers:
            q = layer(q, visual_features)

        return q  # [B, 64, d]


class PerceiverLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, queries, context):
        q = self.norm1(queries)
        queries = queries + self.cross_attn(q, context, context)[0]
        q = self.norm2(queries)
        queries = queries + self.self_attn(q, q, q)[0]
        queries = queries + self.ffn(self.norm3(queries))
        return queries


class GatedCrossAttentionBlock(nn.Module):
    """Flamingo 的门控交叉注意力块"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 门控参数，初始化为 0 → tanh(0) = 0
        self.gate_attn = nn.Parameter(torch.zeros(1))
        self.gate_ffn = nn.Parameter(torch.zeros(1))

    def forward(self, x, visual_tokens):
        # x: [B, L, d] — LLM 中间层的文本 token
        # visual_tokens: [B, 64, d] — Perceiver 输出
        
        residual = x
        x_norm = self.norm1(x)
        attn_out = self.cross_attn(x_norm, visual_tokens, visual_tokens)[0]
        x = residual + torch.tanh(self.gate_attn) * attn_out

        residual = x
        x = residual + torch.tanh(self.gate_ffn) * self.ffn(self.norm2(x))

        return x
```

---

## 2. Chameleon：统一 Token 的原生多模态

### 2.1 从模块化到原生

Flamingo 虽然让 LLM "感知"到了图像，但它本质上仍然是模块化的——视觉编码器、Perceiver Resampler、冻结 LLM 各司其职。

Chameleon（Meta, 2024）提出了一个更激进的方案：**把图像也变成离散 token，和文本 token 放在同一个序列中，用同一个 Transformer 处理。**

### 2.2 图像 Token 化：从连续到离散

文本天然是离散的——每个词/子词对应词汇表中的一个整数 ID。但图像是连续的。如何将图像"量化"为离散 token？

答案是 **VQ-VAE**（Vector Quantized Variational Autoencoder）。

#### VQ-VAE 的数学

设编码器 $E$、解码器 $D$、码本（codebook）$\mathcal{C} = \{e_k\}_{k=1}^K$（$K$ 个可学习的嵌入向量），$e_k \in \mathbb{R}^{d_c}$。

对图像 $\mathbf{I}$ 的编码过程：

$$
\mathbf{z} = E(\mathbf{I}) \in \mathbb{R}^{h \times w \times d_c}
$$

其中 $h \times w$ 是编码后的空间分辨率。对每个空间位置 $(i,j)$，找到码本中最近的向量：

$$
k_{ij} = \arg\min_{k \in \{1,...,K\}} \|\mathbf{z}_{ij} - e_k\|_2
$$

量化后的表示：$\hat{\mathbf{z}}_{ij} = e_{k_{ij}}$。

解码器重建图像：$\hat{\mathbf{I}} = D(\hat{\mathbf{z}})$。

**VQ-VAE 的训练损失**：

$$
\mathcal{L}_{\text{VQ}} = \underbrace{\|\mathbf{I} - \hat{\mathbf{I}}\|_2^2}_{\text{重建损失}} + \underbrace{\|\text{sg}[\mathbf{z}] - e\|_2^2}_{\text{码本损失}} + \underbrace{\beta \|\mathbf{z} - \text{sg}[e]\|_2^2}_{\text{commitment 损失}}
$$

其中 $\text{sg}[\cdot]$ 表示 stop-gradient。码本损失让码本向量靠近编码器输出；commitment 损失防止编码器输出偏离码本太远。

> **定理（VQ 的离散化误差界, van den Oord et al. 2017; Gray & Neuhoff 1998）**：设连续表示 $\mathbf{z}$ 服从分布 $p(\mathbf{z})$，码本大小为 $K$，维度为 $d_c$。在高维设定下，最优 $K$-means 量化的均方误差满足：
>
> $$\mathbb{E}[\|\mathbf{z} - \hat{\mathbf{z}}\|^2] \propto K^{-2/d_c}$$
>
> 即码本越大、维度越低，量化误差越小。这解释了为什么 Chameleon 使用 $K = 8192$ 的大码本和 $d_c = 256$ 的较低维度。

#### Chameleon 的图像 Token 化

Chameleon 使用 Meta 开发的 image tokenizer（基于 Make-A-Scene 的改进版 VQ-VAE）：

- **码本大小**：$K = 8192$
- **图像分辨率**：512×512
- **Token 序列长度**：1024 个 token（$32 \times 32$ 的空间网格）
- **重建质量**：FID ≈ 1.0（接近无损）

每张图片被编码为 1024 个离散整数，每个取值在 $\{0, 1, \ldots, 8191\}$ 中——和文本 token 完全类似。

### 2.3 统一的自回归训练

有了离散的图像 token 后，Chameleon 将图像和文本混合成一个 token 序列：

$$
\mathbf{x} = (\underbrace{w_1, w_2, \ldots}_{\text{文本 token}}, \underbrace{[\text{IMG}]}_{\text{起始标记}}, \underbrace{i_1, i_2, \ldots, i_{1024}}_{\text{图像 token}}, \underbrace{[\text{/IMG}]}_{\text{结束标记}}, \underbrace{w_k, w_{k+1}, \ldots}_{\text{文本 token}})
$$

词汇表扩展为 $\mathcal{V}_{\text{text}} \cup \mathcal{V}_{\text{image}}$，总大小约 $65536 + 8192 = 73728$。

训练目标：标准的 next-token prediction：

$$
\mathcal{L} = -\sum_{t} \log P(x_t \mid x_{<t})
$$

不区分文本 token 和图像 token——模型需要学会**预测下一个 token 是什么，无论它是一个单词还是一个像素块**。

### 2.4 训练稳定性挑战

混合模态的自回归训练面临严重的稳定性问题。Chameleon 发现了两个关键挑战并提出解决方案：

**问题 1：梯度范数发散**

图像 token 和文本 token 的 loss 尺度不同，导致梯度范数不稳定。

**解决方案——QK-Norm**：

对 attention 中的 Query 和 Key 做 L2 归一化：

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{\hat{Q} \hat{K}^\top}{\sqrt{d_k}}\right) V, \quad \hat{Q} = \frac{Q}{\|Q\|}, \;\hat{K} = \frac{K}{\|K\|}
$$

这限制了 attention logits 的范围在 $[-1/\sqrt{d_k}, 1/\sqrt{d_k}]$，防止极端值。

**问题 2：softmax 数值溢出**

混合模态时，某些 token 的 logit 异常大（尤其是在图像→文本转换边界）。

**解决方案——Layer Norm 重排**：将 RMSNorm 替换为标准 LayerNorm，并调整其在 attention 前后的位置。

### 2.5 Chameleon 的能力

作为真正的原生多模态模型，Chameleon 可以：

1. **图像理解**：看图回答问题、描述图像
2. **图像生成**：通过自回归预测图像 token 后用 VQ-VAE 解码器重建
3. **混合生成**：输出交错的图文内容（如"以下是三种猫的图片和介绍..."）
4. **纯文本**：保持与 LLaMA-2 相当的文本能力

| 任务 | Chameleon-34B | Flamingo-80B | LLaVA-1.5-13B |
|------|-------------|-------------|---------------|
| VQAv2 | 73.5 | 56.3 | 80.0 |
| 图像描述 (COCO CIDEr) | 130.5 | 138.1 | - |
| 文本生成 (MMLU) | 63.2 | - | - |
| **图像生成** (FID↓) | **5.8** | ✗ | ✗ |

Chameleon 的独特优势：**同一个模型**既能理解也能生成图像——Flamingo 和 LLaVA 都无法生成图像。

---

## 3. Adapter vs Native：本质差异

### 3.1 融合深度的数学分析

设 Transformer 有 $L$ 层。不同方案的"融合深度"可以量化为：

| 方案 | 首次融合层 | 融合总层数 | 融合密度 |
|------|----------|-----------|---------|
| CLIP | 第 $L$ 层（最终） | 0 层 | 0% |
| LLaVA | 第 1 层（前缀） | $L$ 层（self-attn） | 100%（间接） |
| Flamingo | 第 1 层到第 $L$ 层 | 交替 | ~50% |
| Chameleon | 第 1 层到第 $L$ 层 | 每层 | 100%（直接） |

**直觉**：融合越早、越密，模态间的信息交换越充分。但也越难训练，越容易遗忘单模态知识。

### 3.2 表达能力

> **命题（早期融合的表达优势）**：设 $f_{\text{early}}: \mathcal{X}_v \times \mathcal{X}_t \to \mathcal{Y}$ 为一个 $L$ 层 Transformer 在拼接输入上的映射，$f_{\text{late}}: \mathcal{X}_v \times \mathcal{X}_t \to \mathcal{Y}$ 为两个独立 $L$ 层 Transformer 的晚期融合。则 $f_{\text{early}}$ 的函数族严格包含 $f_{\text{late}}$（在参数数量相当的条件下）。
>
> **证明思路**：$f_{\text{late}}$ 等价于 self-attention 中将跨模态注意力权重强制设为 0 的 $f_{\text{early}}$——这是一个严格的约束子集。

但更强的表达能力也意味着更大的搜索空间和更高的优化难度。实践中，Chameleon-34B 在纯图像理解任务上仍不如 LLaVA-1.5-13B——可能是因为模块化方案可以利用更强的预训练组件（如 CLIP-ViT 的视觉表征已经非常好了）。

---

## 4. 代码实现对比

### 4.1 Flamingo 式推理

```python
# 使用 flamingo-pytorch 简化版
from flamingo_pytorch import PerceiverResampler, GatedCrossAttentionBlock

# 假设已有冻结 LLM 和视觉编码器
class FlamingoLikeModel(nn.Module):
    def __init__(self, llm, d_model, n_media_tokens=64):
        super().__init__()
        self.llm = llm  # 冻结
        self.perceiver = PerceiverResampler(
            dim=d_model, depth=6, num_latents=n_media_tokens
        )
        # 在 LLM 的每隔一层插入门控交叉注意力
        self.gated_blocks = nn.ModuleList([
            GatedCrossAttentionBlock(dim=d_model, dim_head=64, heads=8)
            for _ in range(len(llm.layers) // 2)
        ])

    def forward(self, images, text_ids):
        # 视觉编码 + Perceiver 压缩
        vis_features = self.vision_encoder(images)
        media_tokens = self.perceiver(vis_features)  # [B, 64, d]

        # LLM 前向，交替插入门控交叉注意力
        x = self.llm.embed(text_ids)
        gate_idx = 0
        for i, layer in enumerate(self.llm.layers):
            x = layer(x)  # 原始 LLM 层（冻结）
            if i % 2 == 1 and gate_idx < len(self.gated_blocks):
                x = self.gated_blocks[gate_idx](x, media_tokens)
                gate_idx += 1

        return self.llm.head(x)
```

### 4.2 Chameleon 式统一序列

```python
class ChameleonLikeModel(nn.Module):
    def __init__(self, vocab_size_text, vocab_size_image, d_model, n_layers):
        super().__init__()
        total_vocab = vocab_size_text + vocab_size_image + 2  # +2: [IMG], [/IMG]
        self.embedding = nn.Embedding(total_vocab, d_model)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead=16, batch_first=True),
            num_layers=n_layers
        )
        self.head = nn.Linear(d_model, total_vocab)

    def forward(self, token_ids):
        """
        token_ids: [B, S] — 混合的文本和图像 token ID
        文本 token: [0, vocab_size_text)
        图像 token: [vocab_size_text, vocab_size_text + vocab_size_image)
        """
        x = self.embedding(token_ids)

        # 因果掩码（标准自回归）
        mask = nn.Transformer.generate_square_subsequent_mask(token_ids.shape[1])
        x = self.transformer(x, x, tgt_mask=mask)

        logits = self.head(x)  # [B, S, total_vocab]
        return logits

    def generate_image(self, text_prompt, vq_decoder, num_image_tokens=1024):
        """文本条件下自回归生成图像"""
        # 1. 编码文本 prompt
        token_ids = tokenize(text_prompt)
        # 2. 追加 [IMG] token
        token_ids.append(IMG_TOKEN_ID)
        # 3. 自回归采样 1024 个图像 token
        for _ in range(num_image_tokens):
            logits = self.forward(torch.tensor([token_ids]))
            next_token = logits[0, -1].argmax()
            token_ids.append(next_token.item())
        # 4. 提取图像 token 并用 VQ-VAE 解码
        image_tokens = token_ids[-1024:]
        image = vq_decoder(image_tokens)
        return image
```

---

## 5. 多模态融合范式的演进总结

```
2019   ViLBERT ── 双流交叉注意力（需要联合预训练）
  │
2021   CLIP ────── 完全晚期融合（对比学习）
  │
2022   Flamingo ── 冻结 LLM + 门控交叉注意力（Adapter 范式开端）
  │
2023   BLIP-2 ──── 冻结 ViT + 冻结 LLM + Q-Former
  │    LLaVA ───── 冻结 ViT + MLP + 可训练 LLM
  │
2024   Chameleon ── 原生早期融合（统一 token）
  │
2025+  InternVL, Qwen-VL ── 混合策略（动态分辨率 + 原生视频）
```

每一步演进都在回答同一个问题的不同侧面：**模态之间的信息应该在何时、以何种方式交换？**

下一篇将介绍最新一代的多模态模型——它们如何结合这些范式的优点，在实际工程中取得最佳效果。

> **下一篇**：[笔记｜多模态融合（六）：2026 前沿——InternVL、Qwen-VL、Mamba 与多模态的未来](posts/34-multimodal-frontier/)

---

**参考文献**

1. Alayrac, J.-B., et al. (2022). *Flamingo: a Visual Language Model for Few-Shot Learning*. NeurIPS 2022.
2. Team Chameleon (2024). *Chameleon: Mixed-Modal Early-Fusion Foundation Models*. arXiv:2405.09818.
3. van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). *Neural Discrete Representation Learning*. NeurIPS 2017.
4. Jaegle, A., et al. (2021). *Perceiver: General Perception with Iterative Attention*. ICML 2021.
