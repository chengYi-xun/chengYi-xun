---
title: 笔记｜多模态融合（二）：CLIP——对比学习连接视觉与语言
date: 2026-04-06 00:00:00
categories:
 - Tutorials
tags:
 - CLIP
 - Contrastive Learning
 - InfoNCE
 - SigLIP
 - Vision-Language Model
series: "多模态融合"
mathjax: true
---

> **论文**：*Learning Transferable Visual Models From Natural Language Supervision*（Radford et al., 2021, OpenAI）
> **代码**：[openai/CLIP](https://github.com/openai/CLIP) · [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
> **前置知识**：[上一篇：多模态融合基础](posts/29-multimodal-basics/)
>
> ⬅️ 上一篇：[笔记｜多模态融合（一）：从特征拼接到注意力融合——多模态学习基础](/chengYi-xun/posts/29-multimodal-basics/)
> ➡️ 下一篇：[笔记｜多模态融合（三）：从 BLIP 到 BLIP-2——Q-Former 与交叉注意力的艺术](/chengYi-xun/posts/31-blip2-qformer/)


---

![CLIP 对比式预训练与零样本推理示意图（摘自 Radford et al., arXiv:2103.00020 图 1）](/chengYi-xun/img/clip_arch.png)


## 0. 零样本分类：一个不可能的任务？

假设你训练了一个图像分类器，类别是 ImageNet 的 1000 类。现在来了一张**鸭嘴兽**的图片——这个类别不在训练集中。传统分类器只能把它归入某个已知类别（也许是"水獭"），因为它的输出层只有 1000 个 logit。

但如果换一种思路：不用固定的类别标签，而是用**自然语言描述**来定义类别呢？

| 候选描述 | 与鸭嘴兽图片的匹配度 |
|---------|-------------------|
| "a photo of a platypus" | **0.92** |
| "a photo of an otter" | 0.71 |
| "a photo of a beaver" | 0.68 |
| "a photo of a duck" | 0.55 |

只要图像和文本共享同一个特征空间，就可以通过**余弦相似度**找到最匹配的描述——即使模型从未见过鸭嘴兽的标注图片。

这就是 CLIP 的核心能力：**零样本迁移**（zero-shot transfer）。

---

## 1. CLIP 的架构

CLIP 是典型的**晚期融合**（Late Fusion）架构——两个完全独立的编码器，只在最终的相似度计算处交汇：

$$
\text{CLIP}: \quad \mathbf{v} = f_{\theta_v}(\mathbf{I}), \quad \mathbf{t} = f_{\theta_t}(\mathbf{w}), \quad \text{sim}(\mathbf{I}, \mathbf{w}) = \frac{\mathbf{v}^\top \mathbf{t}}{\|\mathbf{v}\| \cdot \|\mathbf{t}\|}
$$

### 1.1 图像编码器 $f_{\theta_v}$

CLIP 论文尝试了两种架构：

- **ResNet 系列**（ResNet-50, RN50x4, RN50x16, RN50x64）：使用全局平均池化后接线性投影

- **ViT 系列**（ViT-B/32, ViT-B/16, ViT-L/14）：使用 [CLS] token 的输出接线性投影

以 ViT-B/16 为例，对 224×224 的图像：

$$
\mathbf{I} \in \mathbb{R}^{224 \times 224 \times 3} \xrightarrow{\text{PatchEmbed}} \mathbf{X} \in \mathbb{R}^{197 \times 768} \xrightarrow{\text{12 层 Transformer}} \mathbf{H} \in \mathbb{R}^{197 \times 768} \xrightarrow{\text{取 [CLS]}} \mathbf{h}_0 \in \mathbb{R}^{768} \xrightarrow{W_v} \mathbf{v} \in \mathbb{R}^{512}
$$

### 1.2 文本编码器 $f_{\theta_t}$

CLIP 使用一个 12 层、宽度 512、8 头注意力的 Transformer：

$$
\mathbf{w} = (w_1, \ldots, w_L) \xrightarrow{\text{Tokenizer}} \mathbf{E} \in \mathbb{R}^{77 \times 512} \xrightarrow{\text{12 层 Transformer}} \mathbf{H}_t \in \mathbb{R}^{77 \times 512} \xrightarrow{\text{取 [EOS]}} \mathbf{h}_{\text{eos}} \in \mathbb{R}^{512} \xrightarrow{W_t} \mathbf{t} \in \mathbb{R}^{512}
$$

上下文长度固定为 77 个 token（含 [BOS] 和 [EOS]）。

### 1.3 共享嵌入空间

两个编码器的输出经过各自的线性投影后，映射到**同一个** $d$ 维空间（CLIP 中 $d = 512$）。关键的是，在计算相似度前会进行 **L2 归一化**：

$$
\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}, \quad \hat{\mathbf{t}} = \frac{\mathbf{t}}{\|\mathbf{t}\|_2}
$$

归一化后，余弦相似度退化为点积：$\text{sim}(\hat{\mathbf{v}}, \hat{\mathbf{t}}) = \hat{\mathbf{v}}^\top \hat{\mathbf{t}}$。

---

## 2. InfoNCE Loss：从互信息到对比学习

CLIP 的训练目标是**对比学习**：给定一个 batch 中的 $N$ 个图文对 $\{(\mathbf{I}_i, \mathbf{w}_i)\}_{i=1}^N$，让匹配的对（对角线）相似度高，不匹配的对（非对角线）相似度低。

### 2.1 从互信息说起

对比学习的理论根基是**互信息最大化**。两个随机变量 $X, Y$ 的互信息定义为：

$$
I(X; Y) = \mathbb{E}_{p(x,y)}\left[\log \frac{p(x,y)}{p(x)p(y)}\right]
$$

直觉：$I(X;Y)$ 衡量知道 $X$ 后对 $Y$ 的不确定性减少了多少。

> **定理（互信息的变分下界, Barber & Agakov 2003; Oord et al. 2018）**：对任意函数 $f: \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$，有
> $$I(X; Y) \geq \mathbb{E}_{p(x,y)}\left[f(x,y)\right] - \log \mathbb{E}_{p(x)p(y)}\left[e^{f(x,y)}\right]$$
> 当 $f(x,y) = \log \frac{p(y|x)}{p(y)} + C$ 时取等。

InfoNCE 是这个下界的一个特殊形式。选择 $f(x,y) = \frac{g(x)^\top h(y)}{\tau}$（即缩放后的点积），并用 batch 中的其他样本近似 $p(x)p(y)$：

### 2.2 InfoNCE 推导

设 batch 大小为 $N$，第 $i$ 对的图像嵌入为 $\hat{\mathbf{v}}_i$，文本嵌入为 $\hat{\mathbf{t}}_i$。

**图像到文本方向**（Image-to-Text, i2t）：对于第 $i$ 张图像，在 $N$ 个文本中找到匹配的那个：

$$
p_{i \to j}^{\text{i2t}} = \frac{\exp(\hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}_j / \tau)}{\sum_{k=1}^{N} \exp(\hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}_k / \tau)}
$$

这是一个 $N$ 分类的 softmax，正确类别是 $j = i$。对应的交叉熵损失：

$$
\mathcal{L}_i^{\text{i2t}} = -\log p_{i \to i}^{\text{i2t}} = -\log \frac{\exp(\hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}_i / \tau)}{\sum_{k=1}^{N} \exp(\hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}_k / \tau)}
$$

展开对数：

$$
\mathcal{L}_i^{\text{i2t}} = -\frac{\hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}_i}{\tau} + \log \sum_{k=1}^{N} \exp\!\left(\frac{\hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}_k}{\tau}\right)
$$

第一项推大正样本的相似度，第二项（LogSumExp）推小所有样本的相似度——两者的竞争使得正样本相对于负样本脱颖而出。

**文本到图像方向**（Text-to-Image, t2i）完全对称：

$$
\mathcal{L}_i^{\text{t2i}} = -\log \frac{\exp(\hat{\mathbf{t}}_i^\top \hat{\mathbf{v}}_i / \tau)}{\sum_{k=1}^{N} \exp(\hat{\mathbf{t}}_i^\top \hat{\mathbf{v}}_k / \tau)}
$$

**CLIP 的最终损失**——对称 InfoNCE：

$$
\boxed{\mathcal{L}_{\text{CLIP}} = \frac{1}{2N} \sum_{i=1}^{N} \left(\mathcal{L}_i^{\text{i2t}} + \mathcal{L}_i^{\text{t2i}}\right)}
$$

### 2.3 温度参数 $\tau$ 的角色

温度 $\tau > 0$ 控制 softmax 分布的"锐度"：

| $\tau$ 值 | 效果 | 风险 |
|-----------|------|------|
| $\tau \to 0$ | softmax 趋近 argmax，只关注最相似的样本 | 梯度爆炸，训练不稳定 |
| $\tau = 1$ | 标准 softmax | 可能不够"尖锐" |
| $\tau \to +\infty$ | softmax 趋近均匀分布 | 无法区分正负样本 |

数学上，$\tau$ 出现在梯度中：

$$
\frac{\partial \mathcal{L}_i^{\text{i2t}}}{\partial \hat{\mathbf{v}}_i} = \frac{1}{\tau}\left(\sum_{k=1}^{N} p_{i \to k}^{\text{i2t}} \cdot \hat{\mathbf{t}}_k - \hat{\mathbf{t}}_i\right)
$$

当 $\tau$ 小时，梯度被 $1/\tau$ 放大，但同时 $p_{i \to k}$ 更集中于 hard negative，相当于做了**困难样本挖掘**。

CLIP 中 $\tau$ 是**可学习参数**（以 $\log \tau$ 形式参数化），初始化为 $\tau = 1/0.07 \approx 14.3$，训练中会自适应调整。实际代码中参数化为 `logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))`。

### 2.4 Batch Size 的关键作用

InfoNCE 的负样本来自 batch 内的其他样本。batch 越大，负样本越丰富，对比学习效果越好。

> **命题（InfoNCE 作为互信息下界的紧度）**：InfoNCE loss 满足 $\mathcal{L}_{\text{InfoNCE}} \geq -I(X;Y) + \log N$。因此 $I(X;Y) \geq \log N - \mathcal{L}_{\text{InfoNCE}}$。当 $N \to \infty$ 时，下界可以无限紧。（Oord et al., 2018）

这个命题揭示了 batch size 的重要性——$N$ 越大，互信息的估计越准确。CLIP 使用了 $N = 32{,}768$ 的超大 batch（在 256 个 V100 GPU 上通过分布式训练实现）。

---

## 3. InfoNCE 最优嵌入的几何结构

一个深刻的理论问题：当 InfoNCE loss 达到最优时，嵌入向量在几何上呈什么结构？

> **定理（InfoNCE 最优嵌入的 ETF 结构, Graf et al. 2021）**：在有 $N$ 个类别、嵌入维度 $d \geq N-1$ 的设定下，InfoNCE loss 的全局最优嵌入构成**单纯形等角紧框架**（Simplex Equiangular Tight Frame, ETF）。即 $N$ 个归一化嵌入向量 $\{\hat{\mathbf{z}}_i\}_{i=1}^N$ 满足：
>
> $$\hat{\mathbf{z}}_i^\top \hat{\mathbf{z}}_j = \begin{cases} 1, & i = j \\ -\frac{1}{N-1}, & i \neq j \end{cases}$$
>
> 它们在超球面上**均匀分布**，且任意两个不同向量的夹角相等：$\cos\theta = -\frac{1}{N-1}$。

直觉：ETF 是"在球面上尽可能均匀撒开"的最优解——每个类别的嵌入尽量远离其他类别。当 $N$ 很大时，$\cos\theta \approx 0$，即不同类别的嵌入几乎正交。

---

## 4. SigLIP：从 Softmax 到 Sigmoid

CLIP 的 InfoNCE loss 有一个实际问题：softmax 的分母需要对**整个 batch** 的指数求和，这要求全局 batch 同步——在分布式训练中开销很大。

### 4.1 SigLIP 的 Sigmoid Loss

SigLIP（Zhai et al., 2023）将对比学习从 $N$ 分类问题转变为 $N^2$ 个独立的**二元分类**问题：

对于 batch 中的每一对 $(i, j)$，定义标签 $y_{ij} = \begin{cases} 1, & i = j \\ -1, & i \neq j \end{cases}$

$$
\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N^2} \sum_{i=1}^{N} \sum_{j=1}^{N} \log \sigma\!\left(y_{ij} \cdot (\hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}_j / \tau - b)\right)
$$

其中 $\sigma(x) = \frac{1}{1 + e^{-x}}$ 是 sigmoid 函数，$b$ 是可学习的偏置。

展开正样本和负样本：

$$
\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N^2}\left[\sum_{i=1}^{N} \log \sigma\!\left(\frac{\hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}_i}{\tau} - b\right) + \sum_{i \neq j} \log \sigma\!\left(-\frac{\hat{\mathbf{v}}_i^\top \hat{\mathbf{t}}_j}{\tau} + b\right)\right]
$$

### 4.2 InfoNCE vs SigLIP 的数学对比

| 维度 | InfoNCE (CLIP) | Sigmoid (SigLIP) |
|------|----------------|-----------------|
| 分类形式 | $N$-way softmax | $N^2$ 个二元分类 |
| 归一化 | 全局（分母对所有样本求和） | 局部（每对独立） |
| 梯度同步 | 需要全局 all-gather | 不需要 |
| 最优几何 | 严格 ETF | ETF → Antipodal（依赖 $\tau$） |

> **定理（Sigmoid Loss 的最优几何, Lee et al., AISTATS 2024）**：Sigmoid loss 的最优嵌入几何取决于温度 $\tau$。存在临界温度 $\tau^*$：
>
> - 当 $\tau > \tau^*$ 时，最优嵌入为**单纯形 ETF**（与 InfoNCE 相同）
>
> - 当 $\tau < \tau^*$ 时，最优嵌入向**对跖结构**（antipodal）过渡：匹配对的余弦相似度趋近 +1，非匹配对趋近 -1
>
> 这通过 Constant Embedding Model (CCEM) 框架证明，临界温度由类别数 $N$ 和正负样本比例共同决定。

**2025 年最新理论进展**：
2025 年 9 月，一项发表在 NeurIPS 的研究（Global Minimizers of Sigmoid Contrastive Loss, 2025）进一步深化了对 SigLIP 的理论理解。该研究引入了 $(\mathsf{m}, \mathsf{b}_{\mathsf{rel}})$-Constellations（一种与球面编码相关的组合对象），从数学上严格证明了 SigLIP 中**可学习的逆温度和偏置项**的必要性。
理论分析表明：
1. **模态鸿沟（Modality Gap）的必然性**：在 SigLIP 和 CLIP 中观察到的图像和文本嵌入分离的现象，实际上是优化过程达到全局极小值时的必然几何特征。
2. **维度要求**：为了获得高质量的表征（即避免特征坍缩），嵌入空间的维度必须达到一定的理论下界。
3. 这些理论工具不仅解释了为什么 SigLIP 在检索任务上表现优异，还提出了一种带有显式相对偏置的改进版 Sigmoid 损失重参数化方法。

### 4.3 实际性能

SigLIP 在效率上的优势显著：

| 配置 | CLIP | SigLIP |
|------|------|--------|
| 训练设备 | 256 TPUv3, 5-10 天 | 32 TPUv4, 5 天 |
| ImageNet 零样本 | ~76% (ViT-B/16) | 73.4% (ViT-B/16) |
| Batch 同步 | 需要全局 all-gather | 不需要 |

性能接近但训练成本降低 8 倍。

---

## 5. 零样本迁移的数学机制

CLIP 最引人注目的能力是零样本迁移。其数学机制优雅地利用了共享嵌入空间。

### 5.1 文本作为分类器

对于一个有 $C$ 个类别的分类任务（如 ImageNet 的 1000 类），CLIP 将每个类别名转换为文本描述：

$$
\mathbf{t}_c = f_{\theta_t}(\text{"a photo of a [class\_name]}_c\text{"})
$$

这 $C$ 个文本嵌入 $\{\hat{\mathbf{t}}_1, \ldots, \hat{\mathbf{t}}_C\}$ 构成一组**权重向量**，与传统分类器的权重矩阵 $\mathbf{W} \in \mathbb{R}^{C \times d}$ 等价。

### 5.2 推理过程

对新图像 $\mathbf{I}$：

$$
\hat{y} = \arg\max_{c \in \{1,\ldots,C\}} \frac{\exp(\hat{\mathbf{v}}^\top \hat{\mathbf{t}}_c / \tau)}{\sum_{c'=1}^{C} \exp(\hat{\mathbf{v}}^\top \hat{\mathbf{t}}_{c'} / \tau)}
$$

这正是一个标准的 softmax 分类器——区别在于权重 $\hat{\mathbf{t}}_c$ 不是从训练数据中学到的，而是由文本编码器即时生成的。

### 5.3 Prompt Engineering

类别描述的措辞（prompt）对零样本性能影响很大。CLIP 论文发现使用 prompt ensemble 效果更好：

```python
# 论文共使用 80 个手工设计的上下文模板（此处仅示意）。
templates = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a sculpture of a {}.",
]
```

对每个类别，用所有模板编码后取平均：

$$
\hat{\mathbf{t}}_c = \frac{1}{|\mathcal{T}|} \sum_{\text{template} \in \mathcal{T}} f_{\theta_t}(\text{template}(\text{class}_c))
$$

再归一化。Radford et al. (2021) 在 ImageNet 上对 **80** 个上下文 prompt 的嵌入做平均（embedding-space ensemble），相较**单一默认 prompt** 可再提升约 **3.5%**；最佳配置在论文表 1 中达到 **76.2%** 的零样本精度。

---

## 6. 代码实现

### 6.1 CLIP 前向传播（核心代码）

以下是 CLIP 训练循环的核心逻辑，基于 OpenCLIP 简化：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CLIPModel(nn.Module):
    """Minimal CLIP-style dual encoder with a learnable logit scale."""

    def __init__(self, vision_encoder, text_encoder, embed_dim):
        super().__init__()
        self.visual = vision_encoder
        self.text = text_encoder
        # Learnable temperature in log-space (OpenCLIP-style init).
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image):
        """Encode images to L2-normalized embeddings.

        Args:
            image: Float tensor of shape ``(B, C, H, W)``.

        Returns:
            Tensor of shape ``(B, d)``, L2-normalized along the last dim.
        """
        features = self.visual(image)  # (B, d_v)
        features = F.normalize(features, dim=-1)
        return features

    def encode_text(self, text):
        """Encode text token ids to L2-normalized embeddings.

        Args:
            text: Long tensor of shape ``(B, L)`` (token ids).

        Returns:
            Tensor of shape ``(B, d)``, L2-normalized along the last dim.
        """
        features = self.text(text)  # (B, d_t)
        features = F.normalize(features, dim=-1)
        return features

    def forward(self, image, text):
        """Pairwise scaled cosine logits between batch image and text rows.

        Args:
            image: Float tensor ``(B, C, H, W)``.
            text: Long tensor ``(B, L)``.

        Returns:
            Logits of shape ``(B, B)`` where ``[i, j]`` scores image i vs text j.
        """
        v = self.encode_image(image)  # (B, d)
        t = self.encode_text(text)  # (B, d)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * (v @ t.T)  # (B, B)
        return logits


def clip_loss(logits):
    """Symmetric InfoNCE (image-to-text + text-to-image).

    Args:
        logits: Square tensor ``(N, N)`` of pairwise similarity scores.

    Returns:
        Scalar mean of the two cross-entropy terms.
    """
    N = logits.shape[0]
    labels = torch.arange(N, device=logits.device)
    # Each row: classify which text matches this image (N-way).
    loss_i2t = F.cross_entropy(logits, labels)
    # Each column: classify which image matches this text (N-way).
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2
```

### 6.2 SigLIP Loss 实现

```python
def siglip_loss(logits, temperature=1.0, bias=0.0):
    """Pairwise sigmoid (SigLIP-style) loss over a square logits grid.

    Args:
        logits: Tensor ``(N, N)`` of unscaled similarity scores.
        temperature: Positive scalar temperature.
        bias: Optional scalar bias inside the sigmoid.

    Returns:
        Scalar mean loss over all ``N * N`` pairs.
    """
    N = logits.shape[0]
    # +1 on diagonal (matched pairs), -1 off-diagonal (negatives).
    labels = 2 * torch.eye(N, device=logits.device) - 1  # (N, N)
    loss = -F.logsigmoid(labels * (logits / temperature - bias))
    return loss.mean()
```

### 6.3 零样本分类

```python
@torch.no_grad()
def zero_shot_classify(model, image, class_names, templates):
    """Zero-shot classify one image against templated class prompts.

    Args:
        model: Object with ``encode_image``, ``encode_text``, ``logit_scale``.
        image: Float tensor ``(1, C, H, W)`` after preprocessing.
        class_names: Length-``C`` list of class name strings.
        templates: List of format strings, each with one ``{}`` slot.

    Returns:
        Tuple ``(pred, probs)`` where ``pred`` is ``(1,)`` class indices and
        ``probs`` is ``(1, C)`` softmax probabilities.
    """
    text_embeddings = []
    for class_name in class_names:
        texts = [template.format(class_name) for template in templates]
        # Pseudocode: replace with tokenizer + ``encode_text`` on ids.
        t = model.encode_text(texts)  # (n_templates, d)
        t = t.mean(dim=0)  # (d,)
        t = F.normalize(t, dim=-1)
        text_embeddings.append(t)

    text_embeddings = torch.stack(text_embeddings)  # (C, d)
    v = model.encode_image(image)  # (1, d)
    logit_scale = model.logit_scale.exp()
    logits = logit_scale * (v @ text_embeddings.T)  # (1, C)
    probs = logits.softmax(dim=-1)
    pred = probs.argmax(dim=-1)
    return pred, probs
```

### 6.4 使用 OpenCLIP 的完整示例

```python
import open_clip
import torch.nn.functional as F
from PIL import Image

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    "ViT-B-16", pretrained="laion2b_s34b_b88k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-16")

image = preprocess_val(Image.open("platypus.jpg")).unsqueeze(0)
image_features = model.encode_image(image)
image_features = F.normalize(image_features, dim=-1)  # (1, d)

texts = tokenizer(["a platypus", "an otter", "a beaver", "a duck"])
text_features = model.encode_text(texts)
text_features = F.normalize(text_features, dim=-1)  # (4, d)

similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
print(f"相似度: {similarity}")
# 例: tensor([[0.85, 0.08, 0.04, 0.03]]) -> 预测为 platypus
```

---

## 7. CLIP 的训练规模与局限

### 7.1 训练规模

| 配置 | 值 |
|------|-----|
| 训练数据 | WIT-400M（4 亿图文对） |
| Batch Size | 32,768 |
| 训练步数 | ~12.8B 样本（32 个 epoch） |
| GPU | 256 × V100 |
| 训练时间 | ~12 天（ViT-B/16） |
| 温度参数 | $\log \tau$ 初始化为 $\log(1/0.07)$，可学习 |

### 7.2 局限性

1. **细粒度理解不足**：CLIP 只做全局对比，无法理解"红色车的左后轮"这种细粒度描述

2. **组合推理差**：对"一只猫骑在狗背上"这种组合性描述的理解力有限

3. **计数和空间关系**：CLIP 难以区分"两只猫"和"三只猫"

4. **单向信息流**：图像和文本编码器完全独立，无法做到"根据问题聚焦图像的特定区域"

这些局限性本质上源于**晚期融合**的架构——两个编码器在处理过程中无法互相参考。

下一篇我们将看到 **BLIP-2** 如何通过 Q-Former 引入交叉注意力来突破这些限制。

---

## 8. 总结

| 维度 | 要点 |
|------|------|
| **架构** | 双编码器（ViT + Transformer），晚期融合 |
| **损失函数** | 对称 InfoNCE：$\mathcal{L} = \frac{1}{2}(\mathcal{L}^{\text{i2t}} + \mathcal{L}^{\text{t2i}})$ |
| **温度参数** | 可学习的 $\tau$，控制 softmax 锐度 |
| **最优几何** | ETF 结构：嵌入在超球面上均匀分布 |
| **SigLIP 改进** | Sigmoid loss，无需全局同步，效率提升 8× |
| **核心能力** | 零样本迁移：文本嵌入 = 分类器权重 |
| **核心限制** | 无细粒度交互，无组合推理能力 |

> 参考资料：
>
> 1. Radford, A., ... & Sutskever, I. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML 2021.
> 2. Oord, A. v. d., Li, Y., & Vinyals, O. (2018). *Representation Learning with Contrastive Predictive Coding*. arXiv:1807.03748.
> 3. Zhai, X., ... & Houlsby, N. (2023). *Sigmoid Loss for Language Image Pre-Training*. ICCV 2023.
> 4. Graf, F., ... & Roth, K. (2021). *Dissecting Supervised Contrastive Learning*. ICML 2021.
> 5. Lee, J., ... & Papailiopoulos, V. (2024). *Analysis of Using Sigmoid Loss for Contrastive Learning*. AISTATS 2024.
> 6. *Global Minimizers of Sigmoid Contrastive Loss*. (2025). NeurIPS 2025. (arXiv:2509.18552).

> 下一篇：[笔记｜多模态融合（三）：从 BLIP 到 BLIP-2——Q-Former 与交叉注意力的艺术](/chengYi-xun/posts/31-blip2-qformer/)
