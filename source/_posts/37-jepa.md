---
title: 笔记｜世界模型（三）：JEPA——在嵌入空间预测世界
date: 2026-04-06 00:40:00
categories:
 - Tutorials
tags:
 - JEPA
 - V-JEPA
 - Self-Supervised Learning
 - World Model
series: "世界模型"
mathjax: true
---

> **核心论文**：
> - I-JEPA: arXiv:2301.08243 (CVPR 2023)
> - V-JEPA: arXiv:2404.08471 (2024)
> - V-JEPA 2: arXiv:2506.09985 (2025)
>
> **代码**：[facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) · [facebookresearch/jepa](https://github.com/facebookresearch/jepa)
> **前置知识**：[上一篇：Dreamer 系列](posts/36-dreamer/)

---

## 0. 预测每片树叶的飘动是愚蠢的

观察窗外的一棵树。树叶在风中随机飘动——每片叶子的具体轨迹本质上不可预测。但你能预测"树叶还会继续飘动"、"风变大了树会摇得更厉害"。

这就是 JEPA 的核心直觉：**好的预测不需要重建每个像素细节，只需要在语义层面预测未来。**

Dreamer 和 Sora 都在某种程度上试图"重建"观测——前者通过解码器重建图像训练世界模型，后者直接生成视频。JEPA 走了一条完全不同的路：**丢弃不可预测的细节，只在嵌入空间中预测。**

---

## 1. LeCun 的认知架构提案

![Generative vs Contrastive vs JEPA](/img/jepa_vs_generative.png)

### 1.1 三种学习架构

2022 年，Yann LeCun 在白皮书 *"A Path Towards Autonomous Machine Intelligence"* 中系统对比了三种架构：

| 架构 | 预测对象 | 问题 |
|------|---------|------|
| **生成式（Generative）** | 预测原始输入 $\hat{x}_{t+1}$ | 浪费容量在不可预测细节上 |
| **联合嵌入（Contrastive/JE）** | 拉近正样本嵌入、推远负样本 | 需要负样本，训练不稳定 |
| **JEPA（联合嵌入预测架构）** | 在嵌入空间预测目标表征 | 无需重建、无需负样本 |

### 1.2 JEPA 的形式化定义

给定上下文输入 $x^{\text{ctx}}$ 和目标输入 $x^{\text{tgt}}$（可以是被遮挡的图像块、未来的视频帧等）：

$$
\begin{aligned}
s_x &= f_\theta(x^{\text{ctx}}) \quad &\text{（上下文编码器）} \\
s_y &= f_{\bar{\theta}}(x^{\text{tgt}}) \quad &\text{（目标编码器，EMA 更新）} \\
\hat{s}_y &= g_\phi(s_x) \quad &\text{（预测器）}
\end{aligned}
$$

训练目标：

$$
\mathcal{L}_{\text{JEPA}} = \|\hat{s}_y - \text{sg}(s_y)\|_2^2
$$

其中 $\text{sg}(\cdot)$ 是 stop-gradient。目标编码器 $f_{\bar{\theta}}$ 通过 EMA（指数移动平均）更新：

$$
\bar{\theta} \leftarrow \tau \bar{\theta} + (1 - \tau) \theta, \quad \tau \in [0.996, 1)
$$

### 1.3 为什么不在像素空间预测？

从信息论角度分析。设输入 $x$ 包含两部分信息：

$$
H(x) = \underbrace{H_{\text{semantic}}}_{\text{可预测的语义}} + \underbrace{H_{\text{stochastic}}}_{\text{不可预测的噪声}}
$$

- **像素重建**（MAE/Dreamer 解码器）：被迫建模 $H(x)$ 的全部，包括 $H_{\text{stochastic}}$
- **JEPA**：编码器 $f_\theta$ 可以学会丢弃 $H_{\text{stochastic}}$，只在嵌入空间保留 $H_{\text{semantic}}$

> **命题（JEPA 的信息选择性）**：设 $s = f_\theta(x)$ 为 JEPA 编码器的输出，$y$ 为预测目标。JEPA 的最优编码器满足：
>
> $$I(s; y) \to \max, \quad I(s; x \mid y) \to 0$$
>
> 即 $s$ 保留了所有与 $y$ 相关的信息，同时丢弃与 $y$ 无关的信息。这正好是**最小充分统计量**（Minimal Sufficient Statistics）的定义。

### 1.4 防止崩塌

JEPA 面临的最大风险是**表征崩塌**——编码器输出常数向量也能让损失为零。

三种防崩塌机制：

1. **EMA 目标编码器**：目标侧参数缓慢更新，提供稳定的预测目标
2. **掩码策略**：让预测任务足够困难（不是简单复制）
3. **方差正则**（可选）：确保嵌入维度间有足够方差

---

## 2. I-JEPA：图像 JEPA

### 2.1 架构

I-JEPA 处理图像：遮挡部分 patch，从可见 patch 预测被遮挡 patch 的**表征**（不是像素）。

1. **输入**：图像被分为 $N$ 个 patch（如 ViT 的 16×16 patch）
2. **上下文 $x^{\text{ctx}}$**：可见的 patch 子集
3. **目标 $x^{\text{tgt}}$**：被遮挡的若干目标块（每块包含多个连续 patch）
4. **预测器** $g_\phi$：从上下文编码预测目标块的嵌入

### 2.2 掩码策略

I-JEPA 的掩码策略强调**语义级别**而非纹理级别：

- **目标块较大**（如图像面积的 15%-20%），迫使模型理解语义而非纹理
- **多个目标块**（通常 4 个），从不同位置预测
- **上下文块**也是连续的大块

这与 MAE 的随机掩码形成对比——MAE 遮挡 75% 的随机 patch，I-JEPA 遮挡较少但更大的连续区域。

### 2.3 与 MAE 的对比

| 维度 | MAE | I-JEPA |
|------|-----|--------|
| 预测目标 | 像素重建 | 嵌入预测 |
| 解码器 | 需要（像素解码） | 不需要 |
| 掩码比例 | 75% 随机 | 15-20% 大块 |
| 数据增强 | 必需 | 不需要 |
| ImageNet 线性评估 | 75.5% (ViT-H) | **77.5%** (ViT-H) |
| GPU 小时 | 1600 | **1200** |

I-JEPA 在性能更高的同时训练更快（不需要像素解码器）。

### 2.4 代码核心

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class IJEPA(nn.Module):
    def __init__(self, encoder, predictor, embed_dim, ema_decay=0.996):
        super().__init__()
        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        self.predictor = predictor
        self.ema_decay = ema_decay

        # 目标编码器不计算梯度
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA 更新目标编码器"""
        for p, tp in zip(self.context_encoder.parameters(),
                         self.target_encoder.parameters()):
            tp.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    def forward(self, images, context_masks, target_masks):
        """
        images: [B, C, H, W]
        context_masks: [B, N_ctx] — 可见 patch 的索引
        target_masks: list of [B, N_tgt] — 目标块的索引
        """
        # 上下文编码（只编码可见 patch）
        context_tokens = self.context_encoder(images, mask=context_masks)

        # 目标编码（编码全图，取目标位置）
        with torch.no_grad():
            target_tokens = self.target_encoder(images)

        loss = 0
        for tgt_mask in target_masks:
            # 预测器：从上下文预测目标
            pred = self.predictor(context_tokens, tgt_mask)
            # 取出目标位置的真实嵌入
            target = target_tokens[tgt_mask]
            # L2 损失
            loss += F.mse_loss(pred, target.detach())

        return loss / len(target_masks)
```

---

## 3. V-JEPA：视频 JEPA

### 3.1 从图像到视频

V-JEPA 将 I-JEPA 的思想扩展到视频——在**时空**维度上掩码并预测。

输入：视频片段 $V \in \mathbb{R}^{T \times H \times W \times 3}$

1. **时空分块**：将视频切分为 $(t, h, w)$ 的 3D patch
2. **时空掩码**：遮挡一些时空区域（如遮挡未来几帧的某个空间区域）
3. **预测**：从可见的时空 patch 预测被遮挡区域的**嵌入**

### 3.2 核心特点

V-JEPA 有几个重要设计选择：

1. **纯特征预测**：不用任何像素重建、对比学习、负样本、预训练图像编码器或文本
2. **冻结评估**：下游任务只训练一个轻量探针（linear probe 或 attention probe），backbone 完全冻结
3. **多种掩码策略**：短程掩码（同帧内空间掩码）和长程掩码（跨帧时间掩码）

### 3.3 V-JEPA vs 视频生成

| 维度 | V-JEPA | 视频生成模型 (Sora等) |
|------|--------|---------------------|
| 预测目标 | 嵌入向量 | 像素/视频帧 |
| 解码器 | 无 | 需要（扩散/自回归） |
| 训练目标 | MSE on embeddings | 扩散 ELBO / 自回归 NLL |
| 输出 | 不可视化的表征 | 可视化的视频 |
| 适用场景 | 表征学习、理解 | 生成、仿真 |
| 不可预测细节 | 丢弃 | 必须建模 |

### 3.4 V-JEPA 的数学

设视频被分为 $N$ 个时空 patch，上下文集合 $\mathcal{C}$，目标集合 $\mathcal{T}$：

$$
\begin{aligned}
\mathbf{s}_{\mathcal{C}} &= f_\theta(\{x_i\}_{i \in \mathcal{C}}) \in \mathbb{R}^{|\mathcal{C}| \times d} \\
\mathbf{s}_{\mathcal{T}} &= f_{\bar{\theta}}(\{x_j\}_{j \in \mathcal{T}}) \in \mathbb{R}^{|\mathcal{T}| \times d} \\
\hat{\mathbf{s}}_{\mathcal{T}} &= g_\phi(\mathbf{s}_{\mathcal{C}}, \text{pos}_{\mathcal{T}}) \in \mathbb{R}^{|\mathcal{T}| \times d}
\end{aligned}
$$

预测器 $g_\phi$ 接收上下文嵌入和目标位置编码（告诉它要预测"哪里"），输出目标位置的嵌入预测。

损失函数：

$$
\mathcal{L} = \frac{1}{|\mathcal{T}|} \sum_{j \in \mathcal{T}} \|\hat{s}_j - \text{sg}(s_j)\|_2^2
$$

---

## 4. V-JEPA 2：走向世界模型

### 4.1 从表征到规划

V-JEPA 2（2025）将 V-JEPA 从"自监督表征学习"推进到了"世界模型"领域：

- **动作条件预测**：给定动作 $a_t$，在嵌入空间预测未来状态
- **规划能力**：通过在嵌入空间中 rollout 来评估不同动作序列
- **多尺度预测**：不同时间分辨率的嵌入预测

### 4.2 V-JEPA 2 作为世界模型的数学

$$
\begin{aligned}
\text{编码:} \quad & s_t = f_\theta(o_t) \\
\text{预测:} \quad & \hat{s}_{t+1} = g_\phi(s_t, a_t) \\
\text{规划:} \quad & a^* = \arg\max_a \mathcal{R}(g_\phi(s_t, a))
\end{aligned}
$$

与 Dreamer 的对比：

| 维度 | Dreamer | V-JEPA 2 |
|------|---------|----------|
| 预测空间 | 潜变量 + 解码器 | 纯嵌入空间 |
| 训练信号 | ELBO（重建+KL） | MSE（嵌入预测） |
| 解码器 | 需要 | 不需要 |
| 奖励模型 | 显式预测 | 隐式（通过探针） |
| 物理/机器人评估 | DayDreamer | V-JEPA 2 规划评估 |

### 4.3 V-JEPA 2.1：稠密特征

V-JEPA 2.1（2026）进一步增强了稠密预测能力：

- **稠密预测损失**：在 patch 级别而非序列级别做预测
- **跨层自监督**：利用不同 Transformer 层的特征
- **统一图像/视频 tokenizer**

---

## 5. JEPA vs 生成式世界模型的深层分析

### 5.1 损失景观的差异

**生成式模型**（Dreamer, Sora）的损失函数本质上在优化：

$$
\mathcal{L}_{\text{gen}} = -\mathbb{E}_{p_{\text{data}}}[\log p_\theta(x)]
$$

这要求模型对所有 $x$ 都分配合理的概率密度——包括那些随机的、不可预测的细节。

**JEPA** 的损失函数：

$$
\mathcal{L}_{\text{JEPA}} = \mathbb{E}[\|g_\phi(f_\theta(x^{\text{ctx}})) - \text{sg}(f_{\bar{\theta}}(x^{\text{tgt}}))\|^2]
$$

编码器 $f_\theta$ 可以自由选择丢弃哪些信息——只要目标编码器也丢弃了同样的信息。

### 5.2 信息选择性的数学

> **定理（JEPA 编码器的信息瓶颈性质）**：在 JEPA 框架下，最优编码器 $f^*$ 是 $x$ 关于 $y$（预测目标）的**最小充分统计量**：
>
> $$f^* = \arg\min_{f: I(f(x); y) = I(x; y)} H(f(x))$$
>
> 即在保持对预测目标的全部信息的同时，最小化自身的熵。

这与 Information Bottleneck（Tishby et al., 2000）一脉相承。直觉：如果树叶的具体飘动方向与预测下一帧的语义内容无关，JEPA 编码器会自动忽略它。

### 5.3 实际对比

| 任务 | 生成式更好 | JEPA 更好 |
|------|-----------|----------|
| 图像/视频生成 | ✓ | ✗（不能生成） |
| 图像分类 | - | ✓ |
| 视频理解 | - | ✓ |
| 仿真数据生成 | ✓ | ✗ |
| 机器人规划 | ✓（可渲染） | ✓（更高效） |
| 物理推理 | 待定 | 待定 |

---

## 6. 总结

| 维度 | 要点 |
|------|------|
| **核心思想** | 在嵌入空间预测，丢弃不可预测的细节 |
| **损失函数** | $\|\hat{s} - \text{sg}(s_{\text{tgt}})\|^2$（简单的 MSE） |
| **防崩塌** | EMA 目标编码器 + 大块掩码 |
| **I-JEPA** | 图像块掩码 → 嵌入预测 |
| **V-JEPA** | 时空块掩码 → 嵌入预测，无像素重建 |
| **V-JEPA 2** | 动作条件 + 规划能力 → 世界模型 |
| **vs Dreamer** | 无解码器，纯嵌入空间 |
| **vs Sora** | 不生成视频，只预测语义 |

JEPA 代表了一种优雅的世界模型哲学：**不需要能"画出"未来，只需要能"理解"未来。** 但它有一个明显的限制——无法生成可视化的预测。下一篇将介绍走向另一个极端的方案：**视频生成即世界模拟**。

> **下一篇**：[笔记｜世界模型（四）：视频生成即世界模拟——从 Sora 到 Genie 与 Cosmos](posts/38-video-world-model/)

---

**参考文献**

1. LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence*. OpenReview.
2. Assran, M., et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. CVPR 2023.
3. Bardes, A., et al. (2024). *Revisiting Feature Prediction for Learning Visual Representations from Video*. arXiv:2404.08471.
4. Tishby, N., Pereira, F., & Bialek, W. (2000). *The Information Bottleneck Method*. arXiv:physics/0004057.
