---
title: 笔记｜多模态融合（一）：从特征拼接到注意力融合——多模态学习基础
date: 2026-04-05 23:55:00
categories\n
Tutorials
tags\n
Multimodal Learning
 - Fusion
 - Cross-Attention
 - Computer Vision
 - NLP
series: "多模态融合"
mathjax: true
---

> **系列说明**：本文是多模态融合系列的第一篇，从最基本的概念出发，建立三级融合的数学框架，为后续 CLIP、BLIP-2、LLaVA 等模型打下理论基础。
> **前置知识**：线性代数基础、Transformer 注意力机制（可参考本站第十一篇 UIT/DiT 详解）。

---

## 0. 从一个图文检索任务说起

假设你手上有一个小型数据集：5 张图片和 5 段文字描述。

| 编号 | 图片内容 | 文字描述 |
|------|---------|---------|
| 1 | 一只橘猫趴在窗台上 | "an orange cat lying on a windowsill" |
| 2 | 一辆红色跑车停在路边 | "a red sports car parked on the street" |
| 3 | 一碗拉面冒着热气 | "a steaming bowl of ramen noodles" |
| 4 | 一片雪山风景 | "a snowy mountain landscape" |
| 5 | 一个小女孩在画画 | "a little girl painting on a canvas" |

你的任务是：给定一张新的图片（比如另一只猫），从这 5 段文字中找到最匹配的描述。

这个问题的核心挑战是——**图片和文字是完全不同的信号**。图片是一个三维张量 $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$，文字是一个变长离散序列 $\mathbf{w} = (w_1, w_2, \ldots, w_L)$，$w_i \in \mathcal{V}$（词汇表）。如何让它们"对话"？

这就是**多模态融合**（Multimodal Fusion）要解决的核心问题。

---

## 1. 什么是"模态"？

**模态**（Modality）指信息的来源或表现形式。在机器学习中，常见模态包括：

| 模态 | 原始形式 | 典型编码器 | 编码后维度 |
|------|---------|-----------|-----------|
| 视觉（图像） | $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$ | ViT, ResNet | $\mathbf{v} \in \mathbb{R}^{N_v \times d_v}$ |
| 语言（文本） | $(w_1, \ldots, w_L), w_i \in \mathcal{V}$ | BERT, GPT | $\mathbf{t} \in \mathbb{R}^{N_t \times d_t}$ |
| 音频 | $\mathbf{a} \in \mathbb{R}^{T \times F}$（频谱） | Whisper, HuBERT | $\mathbf{a} \in \mathbb{R}^{N_a \times d_a}$ |
| 3D 点云 | $\mathbf{P} \in \mathbb{R}^{N \times 3}$ | PointNet, PCT | $\mathbf{p} \in \mathbb{R}^{N_p \times d_p}$ |

关键观察：不同模态经过各自的编码器后，都变成了**序列化的向量表示**。模态之间的差异体现在两个维度上：

1. **序列长度不同**：$N_v \neq N_t$（ViT-B/16 对 224×224 图片产生 196 个 patch token，而一段文字可能只有 20 个 token）
2. **特征维度不同**：$d_v \neq d_t$（ViT-B 的 $d_v = 768$，而某些文本编码器 $d_t = 512$）

**多模态融合的目标**：将来自不同模态的表示整合为统一的联合表示，使下游任务（分类、检索、生成）能够同时利用多个模态的信息。

数学化地说，我们要找到一个融合函数：

$$
\mathcal{F}: \underbrace{\mathbb{R}^{N_v \times d_v}}_{\text{视觉}} \times \underbrace{\mathbb{R}^{N_t \times d_t}}_{\text{语言}} \longrightarrow \underbrace{\mathbb{R}^{N_z \times d_z}}_{\text{联合表示}}
$$

不同的 $\mathcal{F}$ 设计，构成了多模态融合的不同流派。

---

## 2. 三级融合框架

按照融合发生的时机，可以将多模态融合分为三个层级。用一个直观的类比：

| 层级 | 类比 | 信息交换时机 |
|------|------|------------|
| 早期融合 | 把中文和英文混在一起写 | 原始特征层 |
| 中期融合 | 两个翻译官反复交流后各自总结 | 中间表征层 |
| 晚期融合 | 各自写完报告再合并结论 | 决策/输出层 |

### 2.1 早期融合（Early Fusion）

**核心思想**：在编码的最早阶段就将不同模态的特征合并，然后用统一的网络处理。

**最简单的形式——拼接**：

给定视觉特征 $\mathbf{v} \in \mathbb{R}^{d_v}$ 和文本特征 $\mathbf{t} \in \mathbb{R}^{d_t}$（这里先考虑单向量的简化场景），拼接融合为：

$$
\mathbf{z}_{\text{concat}} = [\mathbf{v}; \mathbf{t}] \in \mathbb{R}^{d_v + d_t}
$$

然后通过 MLP 映射到任务空间：$\hat{y} = \text{MLP}(\mathbf{z}_{\text{concat}})$。

**问题**：拼接是线性组合，无法建模模态之间的**交互**。"猫"和"窗台"的组合含义不等于"猫"的含义加上"窗台"的含义。

**改进——双线性池化（Bilinear Pooling）**：

双线性池化通过外积捕获二阶交互：

$$
\mathbf{z}_{\text{bilinear}} = \mathbf{v}^\top \mathbf{W} \mathbf{t}
$$

其中 $\mathbf{W} \in \mathbb{R}^{d_v \times d_t}$ 是可学习的权重矩阵。完整的双线性池化输出维度为 $d_v \times d_t$：

$$
z_{ij} = \sum_k v_k W_{ki} t_j = \mathbf{v}^\top \mathbf{W}_{\cdot, j} \cdot t_j
$$

用外积的形式表达更清晰——令 $\tilde{\mathbf{z}} = \mathbf{v} \otimes \mathbf{t} \in \mathbb{R}^{d_v \times d_t}$，则 $\tilde{z}_{ij} = v_i \cdot t_j$。这个 $d_v \times d_t$ 维的向量能表达所有**二阶交互项**。

> **定理（双线性模型的表达能力）**：设 $\phi(\mathbf{v}, \mathbf{t}) = \mathbf{w}^\top (\mathbf{v} \otimes \mathbf{t})$ 为双线性分类器，则其等价于在 $\mathbf{v}$ 和 $\mathbf{t}$ 的所有分量对之间施加线性权重。形式化地，$\phi(\mathbf{v}, \mathbf{t}) = \sum_{i,j} W_{ij} v_i t_j$，即一个关于输入的**二次函数**。（参考：Tenenbaum & Freeman, 2000）

**问题**：当 $d_v = d_t = 768$ 时，$\tilde{\mathbf{z}}$ 的维度高达 $768^2 \approx 59$ 万，计算和存储开销巨大。

**解决方案——紧凑双线性池化（Compact Bilinear Pooling）**：

利用 Count Sketch 投影将外积压缩到低维：

$$
\mathbf{z}_{\text{compact}} = \text{CountSketch}(\mathbf{v}) \circledast \text{CountSketch}(\mathbf{t}) \in \mathbb{R}^{d_z}
$$

其中 $\circledast$ 表示卷积（等价于 FFT 域的逐元素乘法），$d_z \ll d_v \cdot d_t$。这利用了如下性质：

> **定理（Count Sketch 的卷积性质）**：若 $\psi_h^s(\mathbf{x})$ 为 Count Sketch 投影（由哈希函数 $h$ 和符号函数 $s$ 定义），则 $\psi_{h_1}^{s_1}(\mathbf{v}) \circledast \psi_{h_2}^{s_2}(\mathbf{t})$ 是 $\mathbf{v} \otimes \mathbf{t}$ 的 Count Sketch 投影的无偏估计。（Pham & Pagh, 2013）

实际计算通过 FFT 加速：$\mathbf{z}_{\text{compact}} = \text{FFT}^{-1}(\text{FFT}(\psi(\mathbf{v})) \odot \text{FFT}(\psi(\mathbf{t})))$。

**PyTorch 实现**：

```python
import torch
import torch.nn as nn
import torch.fft

class EarlyFusionConcat(nn.Module):
    """最简单的早期融合：拼接 + MLP"""
    def __init__(self, d_v, d_t, d_out):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_v + d_t, 512),
            nn.ReLU(),
            nn.Linear(512, d_out),
        )

    def forward(self, v, t):
        # v: [B, d_v], t: [B, d_t]
        z = torch.cat([v, t], dim=-1)  # [B, d_v + d_t]
        return self.mlp(z)


class BilinearPooling(nn.Module):
    """全双线性池化（用于小维度场景）"""
    def __init__(self, d_v, d_t, d_out):
        super().__init__()
        self.bilinear = nn.Bilinear(d_v, d_t, d_out)

    def forward(self, v, t):
        # v: [B, d_v], t: [B, d_t]
        return self.bilinear(v, t)  # [B, d_out]


class CompactBilinearPooling(nn.Module):
    """紧凑双线性池化（Count Sketch + FFT）"""
    def __init__(self, d_v, d_t, d_out):
        super().__init__()
        self.d_out = d_out
        # 为每个模态生成随机 hash 和 sign
        self.register_buffer('h_v', torch.randint(0, d_out, (d_v,)))
        self.register_buffer('s_v', 2 * torch.randint(0, 2, (d_v,)).float() - 1)
        self.register_buffer('h_t', torch.randint(0, d_out, (d_t,)))
        self.register_buffer('s_t', 2 * torch.randint(0, 2, (d_t,)).float() - 1)

    def count_sketch(self, x, h, s):
        # x: [B, d], h: [d], s: [d]
        B, d = x.shape
        sketch = torch.zeros(B, self.d_out, device=x.device)
        signed_x = x * s.unsqueeze(0)
        sketch.scatter_add_(1, h.unsqueeze(0).expand(B, -1), signed_x)
        return sketch

    def forward(self, v, t):
        # Count Sketch 投影
        sketch_v = self.count_sketch(v, self.h_v, self.s_v)
        sketch_t = self.count_sketch(t, self.h_t, self.s_t)
        # FFT 域逐元素相乘 = 时域卷积 ≈ 外积的 sketch
        fft_v = torch.fft.fft(sketch_v, dim=-1)
        fft_t = torch.fft.fft(sketch_t, dim=-1)
        z = torch.fft.ifft(fft_v * fft_t, dim=-1).real
        return z  # [B, d_out]
```

用前面的例子验证：

```python
# 模拟 5 个图文对
B = 5
d_v, d_t = 768, 512

v = torch.randn(B, d_v)  # 视觉特征
t = torch.randn(B, d_t)  # 文本特征

model_concat = EarlyFusionConcat(d_v, d_t, d_out=128)
model_bilinear = BilinearPooling(d_v, d_t, d_out=128)
model_compact = CompactBilinearPooling(d_v, d_t, d_out=8192)

z1 = model_concat(v, t)    # [5, 128]
z2 = model_bilinear(v, t)  # [5, 128]
z3 = model_compact(v, t)   # [5, 8192]

print(f"拼接融合: {z1.shape}, 参数量: {sum(p.numel() for p in model_concat.parameters()):,}")
print(f"双线性融合: {z2.shape}, 参数量: {sum(p.numel() for p in model_bilinear.parameters()):,}")
print(f"紧凑双线性: {z3.shape}, 参数量: 0 (无可学习参数)")
```

### 2.2 晚期融合（Late Fusion）

**核心思想**：每个模态用独立的编码器处理到最终表示，然后在**决策层**合并。

设视觉编码器 $f_v: \mathbb{R}^{H \times W \times 3} \to \mathbb{R}^{d}$ 和文本编码器 $f_t: \mathcal{V}^* \to \mathbb{R}^{d}$ 分别将输入映射到共享的 $d$ 维空间。

**Score-level 融合**（对检索任务）：

$$
\text{sim}(\mathbf{I}, \mathbf{w}) = \frac{f_v(\mathbf{I})^\top f_t(\mathbf{w})}{\|f_v(\mathbf{I})\| \cdot \|f_t(\mathbf{w})\|}
$$

这正是 CLIP 采用的方式（下一篇详细介绍）。

**Decision-level 融合**（对分类任务）：

$$
\hat{y} = \lambda \cdot f_v^{\text{cls}}(\mathbf{I}) + (1 - \lambda) \cdot f_t^{\text{cls}}(\mathbf{w})
$$

其中 $f_v^{\text{cls}}$ 和 $f_t^{\text{cls}}$ 分别输出各模态的 logits，$\lambda$ 为混合权重。

**优点**\n
各模态编码器可以独立预训练
- 推理时可以单模态部署（图像检索只需图像编码器）
- 计算效率高——编码后只需一次点积

**缺点**\n
模态之间的交互仅限于最终的相似度计算
- 无法捕获细粒度的跨模态对应关系（如"红色"对应图片中车的颜色）

**PyTorch 实现**：

```python
class LateFusion(nn.Module):
    """晚期融合：独立编码 + 余弦相似度"""
    def __init__(self, d_v, d_t, d_shared):
        super().__init__()
        self.proj_v = nn.Linear(d_v, d_shared)
        self.proj_t = nn.Linear(d_t, d_shared)

    def encode_image(self, v):
        z = self.proj_v(v)
        return z / z.norm(dim=-1, keepdim=True)

    def encode_text(self, t):
        z = self.proj_t(t)
        return z / z.norm(dim=-1, keepdim=True)

    def forward(self, v, t):
        v_emb = self.encode_image(v)   # [B, d_shared]
        t_emb = self.encode_text(t)    # [B, d_shared]
        sim = v_emb @ t_emb.T         # [B, B] 相似度矩阵
        return sim
```

### 2.3 中期融合（Mid Fusion / Cross-Modal Attention）

**核心思想**：在编码过程的中间层引入跨模态信息交换，让不同模态在处理过程中**互相看到**彼此。

这是现代多模态模型最常用的策略，核心工具是**交叉注意力**（Cross-Attention）。

#### 交叉注意力的数学推导

回顾标准的自注意力（Self-Attention）。给定输入序列 $\mathbf{X} \in \mathbb{R}^{N \times d}$：

$$
\text{SelfAttn}(\mathbf{X}) = \text{softmax}\!\left(\frac{\mathbf{X}\mathbf{W}_Q (\mathbf{X}\mathbf{W}_K)^\top}{\sqrt{d_k}}\right) \mathbf{X}\mathbf{W}_V
$$

其中 $\mathbf{W}_Q, \mathbf{W}_K \in \mathbb{R}^{d \times d_k}$，$\mathbf{W}_V \in \mathbb{R}^{d \times d_v}$。

**交叉注意力**的关键修改：**Query 来自一个模态，Key 和 Value 来自另一个模态**。

设视觉 token 序列 $\mathbf{V} \in \mathbb{R}^{N_v \times d}$，文本 token 序列 $\mathbf{T} \in \mathbb{R}^{N_t \times d}$，则"以视觉查询文本"的交叉注意力为：

$$
\text{CrossAttn}(\mathbf{V}, \mathbf{T}) = \text{softmax}\!\left(\frac{(\mathbf{V}\mathbf{W}_Q)(\mathbf{T}\mathbf{W}_K)^\top}{\sqrt{d_k}}\right) \mathbf{T}\mathbf{W}_V
$$

拆解每个视觉 token $\mathbf{v}_i$ 的输出：

$$
\text{out}_i = \sum_{j=1}^{N_t} \alpha_{ij} \cdot (\mathbf{t}_j \mathbf{W}_V)
$$

其中注意力权重为：

$$
\alpha_{ij} = \frac{\exp\!\left(\frac{(\mathbf{v}_i \mathbf{W}_Q)(\mathbf{t}_j \mathbf{W}_K)^\top}{\sqrt{d_k}}\right)}{\sum_{k=1}^{N_t} \exp\!\left(\frac{(\mathbf{v}_i \mathbf{W}_Q)(\mathbf{t}_k \mathbf{W}_K)^\top}{\sqrt{d_k}}\right)}
$$

**几何直觉**：视觉 token $\mathbf{v}_i$ 通过 $\mathbf{W}_Q$ 生成一个"问题"向量，文本 token $\mathbf{t}_j$ 通过 $\mathbf{W}_K$ 生成一个"钥匙"向量。点积 $(\mathbf{v}_i \mathbf{W}_Q)(\mathbf{t}_j \mathbf{W}_K)^\top$ 衡量这个"问题"和"钥匙"的匹配程度。匹配度高的文本 token 贡献更多的信息（通过 $\mathbf{W}_V$ 投影后加权求和）。

> **定理（多层交叉注意力的 Bayes 最优性）**：考虑线性化的多模态 in-context learning 设置，单层线性自注意力**不能**对所有任务分布一致地达到 Bayes 最优预测。但一个包含交叉注意力 + 自注意力 + 残差连接的多层模型，在梯度流优化下可以收敛到 Bayes 最优的 in-context 预测器。（引自：Akyürek et al., 2025, arXiv:2602.04872）

这个定理表明，交叉注意力不仅是一种工程直觉，它在理论上也是多模态信息融合的最优选择（至少在线性化设置下）。

#### 双向交叉注意力

在实践中，通常采用双向交叉注意力——视觉看文本，文本也看视觉：

$$
\begin{aligned}
\mathbf{V}' &= \mathbf{V} + \text{CrossAttn}(\mathbf{V}, \mathbf{T}) \\
\mathbf{T}' &= \mathbf{T} + \text{CrossAttn}(\mathbf{T}, \mathbf{V})
\end{aligned}
$$

这种双向设计让两个模态的信息**对称流动**。ViLBERT（2019）是最早采用这种设计的模型之一。

#### 计算复杂度分析

| 操作 | 复杂度 |
|------|--------|
| 自注意力（视觉） | $O(N_v^2 \cdot d)$ |
| 自注意力（文本） | $O(N_t^2 \cdot d)$ |
| 交叉注意力（V→T） | $O(N_v \cdot N_t \cdot d)$ |
| 拼接后自注意力 | $O((N_v + N_t)^2 \cdot d)$ |

当 $N_v \gg N_t$ 时（ViT 产生 196 个 token，文本只有 20 个），交叉注意力 $O(N_v \cdot N_t \cdot d)$ 比拼接后自注意力 $O((N_v + N_t)^2 \cdot d)$ 更高效。

**PyTorch 实现**：

```python
class CrossAttention(nn.Module):
    """交叉注意力模块"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, context):
        """
        query:   [B, N_q, d_model]  — 发出"问题"的模态
        context: [B, N_c, d_model]  — 提供"答案"的模态
        """
        B, N_q, _ = query.shape
        _, N_c, _ = context.shape

        Q = self.W_q(query).view(B, N_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(context).view(B, N_c, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(context).view(B, N_c, self.n_heads, self.d_k).transpose(1, 2)

        # Q: [B, H, N_q, d_k], K: [B, H, N_c, d_k]
        attn = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)  # [B, H, N_q, N_c]
        attn = attn.softmax(dim=-1)

        out = (attn @ V).transpose(1, 2).reshape(B, N_q, -1)
        return self.W_o(out)


class MidFusionBlock(nn.Module):
    """中期融合块：双向交叉注意力 + FFN"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.cross_attn_v2t = CrossAttention(d_model, n_heads)
        self.cross_attn_t2v = CrossAttention(d_model, n_heads)
        self.norm_v = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)
        self.ffn_v = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        self.ffn_t = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )

    def forward(self, v, t):
        # 双向交叉注意力
        v = v + self.cross_attn_v2t(self.norm_v(v), t)  # 视觉查询文本
        t = t + self.cross_attn_t2v(self.norm_t(t), v)  # 文本查询视觉
        # FFN
        v = v + self.ffn_v(self.norm_v(v))
        t = t + self.ffn_t(self.norm_t(t))
        return v, t
```

---

## 3. 三种融合的对比实验

回到第 0 节的图文检索任务。用一个简化实验来对比三种融合方式：

```python
import torch
import torch.nn.functional as F

# 模拟数据：5 个图文对
B = 5
d_v, d_t, d = 768, 512, 256

v_feats = torch.randn(B, d_v)  # 假设已经过图像编码器
t_feats = torch.randn(B, d_t)  # 假设已经过文本编码器

# --- 早期融合：拼接 ---
early_proj = torch.nn.Linear(d_v + d_t, d)
z_early = early_proj(torch.cat([v_feats, t_feats], dim=-1))  # [5, 256]

# --- 晚期融合：独立投影 + 余弦相似度 ---
proj_v = torch.nn.Linear(d_v, d)
proj_t = torch.nn.Linear(d_t, d)
v_emb = F.normalize(proj_v(v_feats), dim=-1)
t_emb = F.normalize(proj_t(t_feats), dim=-1)
sim_late = v_emb @ t_emb.T  # [5, 5] 相似度矩阵

# --- 中期融合：交叉注意力 ---
# 需要序列化特征 -> 升维为 [B, 1, d] 只是示意
v_seq = proj_v(v_feats).unsqueeze(1)  # [5, 1, 256]
t_seq = proj_t(t_feats).unsqueeze(1)  # [5, 1, 256]
cross_attn = CrossAttention(d_model=d, n_heads=4)
v_fused = v_seq + cross_attn(v_seq, t_seq)  # [5, 1, 256]

print(f"早期融合输出: {z_early.shape}")
print(f"晚期融合相似度矩阵:\n{sim_late}")
print(f"中期融合输出: {v_fused.shape}")
```

**直观对比**：

| 维度 | 早期融合 | 中期融合 | 晚期融合 |
|------|---------|---------|---------|
| 交互深度 | 一阶/二阶 | 注意力级别 | 仅相似度 |
| 计算开销 | $O(d_v \cdot d_t)$ | $O(N_v \cdot N_t \cdot d)$ | $O(d)$ |
| 预训练独立性 | ✗ 需要联合训练 | 部分独立 | ✓ 完全独立 |
| 细粒度对齐 | ✗ | ✓ token 级 | ✗ |
| 检索效率 | ✗ 需要联合编码 | ✗ | ✓ 可预计算 |

**关键洞察**：没有"最好的"融合策略，选择取决于任务需求。

- **检索任务**（需要在百万数据库中快速搜索）→ 晚期融合（可预计算嵌入，线下建索引）
- **VQA 任务**（需要理解"图片左边的红色物体是什么"）→ 中期融合（需要 token 级交互）
- **简单分类任务**（已知模态组合固定）→ 早期融合（参数量最少）

---

## 4. 从融合框架到模型演进

上面介绍的三级融合框架，恰好对应了多模态大模型的三条演进路线：

```
晚期融合 ─────────> CLIP (2021)
                     对比学习, 双编码器
                          │
中期融合 ─────────> BLIP-2 (2023)
(Cross-Attention)    Q-Former 交叉注意力
                          │
                     LLaVA (2023)
                     MLP 投影 + LLM
                          │
早期融合 ─────────> Chameleon (2024)
                     统一 token, 共享 Transformer
```

| 时间线 | 模型 | 融合策略 | 核心特点 |
|--------|------|---------|---------|
| 2019 | ViLBERT / LXMERT | 中期（双流交叉注意力） | 首次将 BERT 扩展到视觉-语言 |
| 2021 | CLIP | 晚期（对比学习） | 4 亿图文对, 零样本迁移 |
| 2022 | Flamingo | 中期（Gated Cross-Attention） | 少样本学习, 冻结预训练模型 |
| 2023 | BLIP-2 | 中期（Q-Former） | 冻结 ViT + 冻结 LLM, 轻量桥接 |
| 2023 | LLaVA | 中期（MLP 投影） | 视觉指令微调, 简洁架构 |
| 2024 | **SD3 / Flux (MMDiT)** | **中期（Joint Attention）** | **扩散模型中的多模态融合** |
| 2024 | Chameleon | 早期（统一 token） | 图像/文本共享 next-token-prediction |
| 2025 | InternVL 2.5 | 中期（ViT-MLP-LLM） | MMMU 70.1%, 开源 SOTA |
| 2025 | Qwen2.5-VL | 中期 + 早期混合 | 原生视频理解, 动态分辨率 |

值得特别一提的是 **MMDiT**（Multimodal Diffusion Transformer），它是 Stable Diffusion 3 和 Flux 的核心架构，将多模态融合直接嵌入了扩散模型的去噪过程中（详见本站[第十四篇 SD3 架构解析](posts/15-sd3/)和[第十五篇 Flux 架构解析](posts/16-flux/)）。

MMDiT 的融合策略结合了双流和单流设计：

- **Double Stream**（SD3 / Flux 前 19 层）：文本和图像各自拥有独立的 QKV 投影和 MLP，但在注意力计算时**拼接 KV 做 Joint Attention**——让文本 token 能关注图像 token，反之亦然
- **Single Stream**（Flux 后 38 层）：文本和图像完全合并为一个序列，共享所有参数

这种"先分后合"的混合设计，正是中期融合在生成式 AI 中的典型落地形式：网络前期保留模态特性（各自的 MLP 处理不同的特征分布），后期充分融合（共享参数减少冗余）。

这个演进并非单向的——**早期融合回归**。最初人们用早期融合但效果差（因为缺乏大规模预训练）；CLIP 证明了晚期融合 + 大数据的威力；BLIP-2/LLaVA 用中期融合实现了更细粒度的理解；而 Chameleon 又回到了早期融合——但这次有了 VQ-VAE 图像 token 化和海量数据，效果远超当年。

---

## 5. Tensor Fusion Network：一个完整的数学例子

为了展示如何从数学出发设计融合方法，这里详解 Tensor Fusion Network（TFN, Zadeh et al., 2017），它是理解更高阶交互的经典框架。

### 5.1 问题设定

三模态情感分析任务：给定视频中某一时刻的**视觉表情** $\mathbf{v} \in \mathbb{R}^{d_v}$、**语音语调** $\mathbf{a} \in \mathbb{R}^{d_a}$、**文字内容** $\mathbf{t} \in \mathbb{R}^{d_t}$，预测情感极性 $y \in \{-1, +1\}$。

### 5.2 张量融合

TFN 的核心是**三阶张量积**。首先，为每个模态增加一个常数维度（对应偏置项）：

$$
\tilde{\mathbf{v}} = [\mathbf{v}; 1] \in \mathbb{R}^{d_v + 1}, \quad \tilde{\mathbf{a}} = [\mathbf{a}; 1] \in \mathbb{R}^{d_a + 1}, \quad \tilde{\mathbf{t}} = [\mathbf{t}; 1] \in \mathbb{R}^{d_t + 1}
$$

张量融合计算三者的外积：

$$
\mathbf{Z} = \tilde{\mathbf{v}} \otimes \tilde{\mathbf{a}} \otimes \tilde{\mathbf{t}} \in \mathbb{R}^{(d_v+1) \times (d_a+1) \times (d_t+1)}
$$

展开后，$\mathbf{Z}$ 包含了所有可能的交互项：

$$
Z_{ijk} = \tilde{v}_i \cdot \tilde{a}_j \cdot \tilde{t}_k
$$

由于加了常数 1，$\mathbf{Z}$ 实际上包含\n
**一阶项**：$v_i, a_j, t_k$（当其他两个取到常数 1 时）
- **二阶项**：$v_i a_j, v_i t_k, a_j t_k$
- **三阶项**：$v_i a_j t_k$

> **命题（TFN 的完备性）**：张量融合 $\mathbf{Z} = \tilde{\mathbf{v}} \otimes \tilde{\mathbf{a}} \otimes \tilde{\mathbf{t}}$ 包含了三个模态之间**所有**一阶、二阶和三阶多项式交互项。

这个命题的证明是直接的：展开 $(d_v+1)(d_a+1)(d_t+1)$ 维的张量积，按常数 1 出现的位置分类即可得到上述三类项。

**PyTorch 实现**：

```python
class TensorFusionNetwork(nn.Module):
    """三模态张量融合"""
    def __init__(self, d_v, d_a, d_t, d_out):
        super().__init__()
        fusion_dim = (d_v + 1) * (d_a + 1) * (d_t + 1)
        self.fc = nn.Linear(fusion_dim, d_out)

    def forward(self, v, a, t):
        B = v.shape[0]
        # 增加常数维度
        ones = torch.ones(B, 1, device=v.device)
        v_aug = torch.cat([v, ones], dim=-1)  # [B, d_v+1]
        a_aug = torch.cat([a, ones], dim=-1)  # [B, d_a+1]
        t_aug = torch.cat([t, ones], dim=-1)  # [B, d_t+1]

        # 三阶外积（通过两次 einsum 实现）
        va = torch.einsum('bi,bj->bij', v_aug, a_aug)       # [B, d_v+1, d_a+1]
        vat = torch.einsum('bij,bk->bijk', va, t_aug)       # [B, d_v+1, d_a+1, d_t+1]
        z = vat.reshape(B, -1)                                # [B, (d_v+1)(d_a+1)(d_t+1)]

        return self.fc(z)

# 示例
d_v, d_a, d_t = 32, 16, 64  # 小维度示意
model = TensorFusionNetwork(d_v, d_a, d_t, d_out=2)
v, a, t = torch.randn(4, d_v), torch.randn(4, d_a), torch.randn(4, d_t)
logits = model(v, a, t)  # [4, 2]
print(f"融合维度: {(d_v+1)*(d_a+1)*(d_t+1)} = {33*17*65}")
```

注意当 $d_v = d_a = d_t = 768$ 时，融合维度为 $(769)^3 \approx 4.5 \times 10^8$——完全不可行。这就是为什么实际应用中需要低秩近似（Low-Rank Tensor Fusion, LMF）或直接转向注意力机制。

---

## 6. 总结与下一篇预告

| 概念 | 数学表达 | 适用场景 |
|------|---------|---------|
| 拼接融合 | $\mathbf{z} = [\mathbf{v}; \mathbf{t}]$ | 简单分类, 维度小 |
| 双线性池化 | $z_{ij} = v_i t_j$ | 二阶交互, 中等维度 |
| 紧凑双线性 | FFT + Count Sketch | 高维二阶交互 |
| 张量融合 | $\mathbf{Z} = \tilde{\mathbf{v}} \otimes \tilde{\mathbf{a}} \otimes \tilde{\mathbf{t}}$ | 多模态高阶交互 |
| 晚期融合 | $\text{sim} = \mathbf{v}^\top \mathbf{t} / \|\mathbf{v}\|\|\mathbf{t}\|$ | 检索, 零样本迁移 |
| 交叉注意力 | $\text{softmax}(QK^\top / \sqrt{d_k})V$ | 细粒度对齐, VQA |

本文建立了多模态融合的基本数学框架。但一个关键问题还没回答：**不同模态的编码器从哪里来？** 如果两个编码器各自独立训练，它们输出的特征空间毫无对齐——"猫"的视觉特征和"cat"的文本特征可能方向完全不同。

下一篇我们将看到 **CLIP** 如何通过对比学习，在 4 亿图文对上训练出天然对齐的视觉和语言编码器，从而让晚期融合的余弦相似度真正有意义。

> **下一篇**：[笔记｜多模态融合（二）：CLIP——对比学习连接视觉与语言](posts/30-clip/)

---

**参考文献**

1. Tenenbaum, J. B., & Freeman, W. T. (2000). *Separating style and content with bilinear models*. Neural Computation.
2. Gao, Y., et al. (2016). *Compact Bilinear Pooling*. CVPR 2016.
3. Zadeh, A., et al. (2017). *Tensor Fusion Network for Multimodal Sentiment Analysis*. EMNLP 2017.
4. Lu, J., et al. (2019). *ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations*. NeurIPS 2019.
5. Akyürek, E., et al. (2025). *Multi-layer Cross-Attention is Provably Optimal for Multi-modal In-context Learning*. arXiv:2602.04872.
