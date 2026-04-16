---
title: 笔记｜多模态融合（三）：从 BLIP 到 BLIP-2——Q-Former 与交叉注意力的艺术
date: 2026-04-06 00:05:00
categories:
 - Tutorials
tags:
 - BLIP-2
 - Q-Former
 - Cross-Attention
 - Vision-Language Model
 - Multimodal Learning
series: "多模态融合"
mathjax: true
---

> **论文**：*BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*（Li et al., 2023, Salesforce）
> **代码**：[salesforce/LAVIS](https://github.com/salesforce/LAVIS) · `lavis/models/blip2_models/blip2_qformer.py`
> **前置知识**：[上一篇：CLIP 对比学习](posts/30-clip/)

---

## 0. 一个尴尬的场景

你手上有一个冻结的 ViT-g（14 亿参数的视觉编码器）和一个冻结的 FlanT5-XXL（110 亿参数的语言模型），两者都是各自领域的顶尖模型。

现在你想让它们协作完成 VQA（视觉问答）：看一张图片，回答"图中有几个人？"

问题是——ViT-g 输出的是 $\mathbb{R}^{257 \times 1408}$ 的视觉 token，FlanT5 期望的输入是 $\mathbb{R}^{* \times 2048}$ 的文本 token。**维度不同、语义空间不同、序列长度不同**。

最暴力的方案是把两者解冻后联合微调——但 12.1B 参数的微调成本巨大，而且可能破坏预训练学到的知识。

BLIP-2 的方案：在两个巨人之间放一个**翻译官**——只有 188M 参数的 **Q-Former**。

| 组件 | 参数量 | 是否训练 |
|------|--------|---------|
| ViT-g（视觉编码器） | 1.1B | 冻结 |
| FlanT5-XXL（语言模型） | 11B | 冻结 |
| **Q-Former（桥接模块）** | **188M** | **训练** |
| **总计** | 12.1B | 仅训练 1.5% |

---

## 1. Q-Former 架构详解

![Q-Former 结构与三种自注意力掩码（摘自 Li et al., arXiv:2301.12597 图 2）](/chengYi-xun/img/blip2_qformer.png)

### 1.1 整体设计

Q-Former 基于 BERT-base 初始化（12 层 Transformer，hidden size 768），但做了关键修改：

1. **引入可学习查询向量**：32 个查询 $\mathbf{q}_1, \ldots, \mathbf{q}_{32} \in \mathbb{R}^{768}$

2. **在每隔一层插入交叉注意力**：查询通过 cross-attention 从冻结 ViT 中提取视觉信息

3. **双分支共享自注意力**：查询分支和文本分支共用 self-attention 层（通过不同的掩码控制交互）

### 1.2 Token 序列构成

Q-Former 的输入序列由两部分组成：

$$
\mathbf{X}_{\text{Q-Former}} = [\underbrace{q_1, \ldots, q_{32}}_{\text{查询 token}}, \underbrace{w_1, \ldots, w_L}_{\text{文本 token}}]
$$

两组 token 共享同一个 self-attention 层，但**是否互相可见由任务决定**（见第 2 节）。

### 1.3 交叉注意力连接

每隔一个 Transformer block，Q-Former 插入一个交叉注意力层（随机初始化），让查询 token 从冻结 ViT 的输出中提取信息：

$$
\mathbf{q}_i' = \mathbf{q}_i + \text{CrossAttn}(\mathbf{q}_i \mathbf{W}_Q, \;\mathbf{F}_{\text{ViT}} \mathbf{W}_K, \;\mathbf{F}_{\text{ViT}} \mathbf{W}_V)
$$

其中 $\mathbf{F}_{\text{ViT}} \in \mathbb{R}^{257 \times 1408}$ 是冻结 ViT 的输出特征。

关键设计：**只有查询 token 参与交叉注意力，文本 token 不直接访问视觉特征**。这意味着所有视觉信息都必须"经过"查询 token，再通过自注意力传递给文本——查询 token 是信息的**瓶颈**。

这个瓶颈是有意为之的：它迫使 Q-Former 在 32 个查询中压缩最关键的视觉信息，避免将 257 个冗余的 ViT patch token 直接灌给 LLM。

### 1.4 输出

Q-Former 的输出是 32 个查询 token 的最终表示：$\mathbf{Z} \in \mathbb{R}^{32 \times 768}$。这 32 个向量浓缩了图像中与任务相关的核心信息。

---

## 2. 三个预训练目标与注意力掩码

第一阶段的核心在于**用三个互补的目标**训练 Q-Former，每个目标使用**不同的自注意力掩码**来控制查询与文本之间的交互。

### 2.1 ITC：图文对比学习

**目标**：让查询输出 $\mathbf{Z}$ 与对应文本的 [CLS] 嵌入 $\mathbf{t}$ 在共享空间中对齐。

**掩码策略——单模态掩码（Uni-modal mask）**：查询 token 和文本 token **互不可见**。

$$
\text{Mask}_{\text{ITC}} = \begin{bmatrix} \mathbf{1}_{32 \times 32} & \mathbf{0}_{32 \times L} \\ \mathbf{0}_{L \times 32} & \mathbf{1}_{L \times L} \end{bmatrix}
$$

其中 $\mathbf{1}$ 表示"可见"，$\mathbf{0}$ 表示"不可见"。

**相似度计算**：由于有 32 个查询，论文取**最大**作为图文相似度：

$$
s(\mathbf{I}, \mathbf{w}) = \max_{k \in \{1,...,32\}} \text{sim}(\mathbf{z}_k, \mathbf{t})
$$

直觉：不同查询可能捕获图像的不同方面（如前景物体、背景场景、颜色信息），取最大值相当于找到与文本最相关的那个方面。

**损失函数**——标准 InfoNCE：

$$
\mathcal{L}_{\text{ITC}} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log \frac{\exp(s_i^{+}/\tau)}{\sum_j \exp(s_{ij}/\tau)} + \log \frac{\exp(s_i^{+}/\tau)}{\sum_j \exp(s_{ji}/\tau)}\right]
$$

### 2.2 ITM：图文匹配

**目标**：二分类——判断给定的图文对是否匹配。

**掩码策略——双向掩码（Bi-directional mask）**：查询 token 和文本 token **完全互相可见**。

$$
\text{Mask}_{\text{ITM}} = \mathbf{1}_{(32+L) \times (32+L)}
$$

此时查询可以直接"看到"文本，文本也可以"看到"查询——实现最充分的跨模态交互。

**分类器**：每个查询输出过一个二类线性头，取所有查询的 logit 均值作为匹配分数：

$$
s_{\text{match}} = \frac{1}{32}\sum_{k=1}^{32} \mathbf{w}_{\text{cls}}^\top \mathbf{z}_k
$$

**损失函数**——二元交叉熵 + 困难负样本挖掘：

$$
\mathcal{L}_{\text{ITM}} = -\left[y \log \sigma(s_{\text{match}}) + (1-y) \log(1 - \sigma(s_{\text{match}}))\right]
$$

负样本通过 ITC 的相似度矩阵选择 batch 中**最混淆的非匹配对**（hard negative mining）。

### 2.3 ITG：图文生成

**目标**：以图像为条件生成文本描述（image-grounded text generation）。

**掩码策略——因果掩码（Causal mask）**：

$$
\text{Mask}_{\text{ITG}}[i][j] = \begin{cases}
1, & \text{如果 } i,j \in \text{query 区域} \\
1, & \text{如果 } i \in \text{text 区域}, j \in \text{query 区域} \\
1, & \text{如果 } i \in \text{text 区域}, j \in \text{text 区域}, j \leq i \\
0, & \text{其他}
\end{cases}
$$

即：查询之间互相可见（不看文本），文本 token 可以看到所有查询 + 左侧文本（因果生成）。[CLS] token 被替换为 [DEC] token 作为解码起始标记。

**损失函数**——标准因果语言建模：

$$
\mathcal{L}_{\text{ITG}} = -\sum_{t=1}^{L} \log P(w_t \mid \mathbf{Z}, w_1, \ldots, w_{t-1})
$$

### 2.4 三种掩码的直觉对比

| 目标 | 查询→文本 | 文本→查询 | 直觉 |
|------|----------|----------|------|
| ITC | ✗ | ✗ | 独立编码再对比，避免捷径 |
| ITM | ✓ | ✓ | 深度融合，判断是否匹配 |
| ITG | ✗ | ✓（因果） | 视觉条件下自回归生成 |

三个目标**共享所有参数**（包括 self-attention），仅通过掩码切换行为——巧妙地用一个模型同时学习对比、匹配和生成能力。

**第一阶段联合损失**：

$$
\mathcal{L}_{\text{stage1}} = \mathcal{L}_{\text{ITC}} + \mathcal{L}_{\text{ITM}} + \mathcal{L}_{\text{ITG}}
$$

---

## 3. 两阶段训练

### 3.1 第一阶段：视觉-语言表示学习

| 配置 | 值 |
|------|-----|
| 训练数据 | ~129M 图像（COCO, VG, CC3M, CC12M, SBU, LAION-400M 子集） |
| 训练步数 | 250k |
| Batch Size | 2320 (ViT-L) / 1680 (ViT-g) |
| 图像分辨率 | 224 × 224 |
| 优化器 | AdamW, $\beta_1=0.9, \beta_2=0.98$, weight decay 0.05 |
| 学习率 | cosine schedule, peak 1e-4, warmup 2k steps |

此阶段只训练 Q-Former（188M），ViT 完全冻结。

### 3.2 第二阶段：视觉到语言的生成学习

Q-Former 的 32 个输出 token 通过一个**全连接层**投影到 LLM 的词嵌入维度：

$$
\mathbf{Z}_{\text{LLM}} = \mathbf{Z} \mathbf{W}_{\text{proj}} + \mathbf{b}_{\text{proj}}, \quad \mathbf{W}_{\text{proj}} \in \mathbb{R}^{768 \times d_{\text{LLM}}}
$$

投影后的视觉 token 被**前置**到 LLM 的输入序列中，充当 soft visual prompts：

$$
\text{LLM 输入} = [\underbrace{\mathbf{z}_1', \ldots, \mathbf{z}_{32}'}_{\text{视觉 prompt}}, \underbrace{w_1, \ldots, w_L}_{\text{文本 token}}]
$$

**对 Decoder-only LLM（如 OPT）**：用标准因果语言建模损失。

**对 Encoder-Decoder LLM（如 FlanT5）**：文本被拆分为 prefix（与视觉 token 一起进入 encoder）和 suffix（作为 decoder 的生成目标）。

| 配置 | 值 |
|------|-----|
| 训练步数 | 80k |
| Batch Size | 1920 (OPT) / 1520 (FlanT5) |
| 学习率 | cosine schedule, 最小 5e-5 |
| 训练时间 | < 6 天（阶段一）+ < 3 天（阶段二），16×A100 |

---

## 4. 实验结果

### 4.1 核心数字

| 模型 | 可训练参数 | VQAv2 (零样本) | NoCaps CIDEr | Flickr30k TR@1 |
|------|-----------|---------------|-------------|----------------|
| BLIP | - | 56.3 | 113.2 | 96.7 |
| Flamingo-80B | 10.2B | 56.3 | - | - |
| **BLIP-2 (ViT-g + FlanT5-XXL)** | **188M** | **65.0** | **121.6** | **97.6** |

BLIP-2 仅用约 **1/54** 的可训练参数（188M vs 10.2B），在零样本 VQAv2（test-dev）上以 **65.0** 对 **56.3** 超过 Flamingo-80B **8.7** 个百分点（Li et al., 2023）。

### 4.2 为什么 Q-Former 比直接投影好？

论文通过消融实验表明，32 个查询相当于对 257 个 ViT patch token 做了**自适应压缩**：

1. **信息瓶颈（Information Bottleneck）**：迫使模型选择性地提取最关键的视觉信息。由于查询 token 数量远小于图像 patch 数量，Q-Former 作为一个信息漏斗，过滤掉了与文本无关的冗余视觉细节（如纯色背景），只保留了高度语义化的特征。
2. **交叉注意力的可学习性**：不同查询可以自发地关注图像的不同区域（类似 DETR 中 object query 的分工）。
3. **维度匹配**：768 维 → LLM 维度的投影比 1408 维 → LLM 维度更轻量。

**理论局限性与反思（2024-2025 视角）**：

尽管 Q-Former 在 BLIP-2 中取得了巨大成功，但后续研究（如 LLaVA 的出现）揭示了其潜在的理论局限。Q-Former 的"信息瓶颈"是一把双刃剑：

- **优点**：大幅减少了输入给 LLM 的 token 数量（32 vs 257），降低了 LLM 的计算开销，特别是在长视频或多图理解任务中优势明显。
- **缺点**：强制压缩会导致**细粒度空间信息的丢失**。当任务需要精确的局部理解（如 OCR 读取图片上的小字、密集的物体计数、复杂的空间关系推理）时，32 个 token 往往无法承载足够的细节。这也是为什么后来的 LLaVA 放弃了 Q-Former，直接用 MLP 将所有 patch token 喂给 LLM，从而在 TextVQA 等细粒度任务上取得了更好的成绩。

此外，从 Transformer 的信息流角度来看，Q-Former 这种固定数量的查询机制，在处理信息密度极高的图像时，容易遇到类似于图神经网络中的"过度挤压"（Over-squashing）问题——过多的局部信息被迫挤入有限的查询向量中，导致表征崩塌。

---

## 5. 代码解析

### 5.1 Q-Former 核心结构（简化版）

```python
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class QFormer(nn.Module):
    """Simplified Q-Former: learned queries + BERT stack with cross-attention."""

    def __init__(
        self,
        num_queries=32,
        hidden_dim=768,
        num_layers=12,
        visual_dim=1408,
        cross_attention_freq=2,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, hidden_dim))
        nn.init.normal_(self.query_tokens, std=0.02)

        config = BertConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=12,
            intermediate_size=hidden_dim * 4,
            encoder_width=visual_dim,
            add_cross_attention=True,
            cross_attention_freq=cross_attention_freq,
        )
        self.bert = BertModel(config)

    def forward(
        self,
        text_input_ids,
        text_attention_mask,
        visual_features,
        attention_mask_type="bidirectional",
    ):
        """Run Q-Former and return final query token states.

        Args:
            text_input_ids: Long tensor ``(B, L)`` (text token ids).
            text_attention_mask: Unused in this stub; real code passes BERT mask.
            visual_features: Float tensor ``(B, T_v, D_v)`` from a frozen ViT
                (paper uses ``T_v = 257``, ``D_v = 1408`` for ViT-g).
            attention_mask_type: One of ``"unimodal"``, ``"bidirectional"``,
                ``"causal"`` (ITC / ITM / ITG style masks).

        Returns:
            Float tensor of shape ``(B, num_queries, hidden_dim)``.
        """
        B = text_input_ids.shape[0]
        L = text_input_ids.shape[1]
        query_tokens = self.query_tokens.expand(B, -1, -1)  # (B, N_q, H)

        query_attn_mask = self._build_attention_mask(
            B, self.num_queries, L, attention_mask_type, text_input_ids.device
        )

        outputs = self.bert(
            query_embeds=query_tokens,
            input_ids=text_input_ids,
            attention_mask=query_attn_mask,
            encoder_hidden_states=visual_features,
        )
        query_output = outputs.last_hidden_state[:, : self.num_queries]  # (B,N_q,H)
        return query_output

    def _build_attention_mask(self, B, N_q, N_t, mask_type, device):
        """Build an extended self-attention mask over ``[queries | text]``.

        Args:
            B: Batch size (mask is broadcast along batch).
            N_q: Number of query tokens.
            N_t: Text length.
            mask_type: ``"unimodal"`` | ``"bidirectional"`` | ``"causal"``.
            device: Torch device for new tensors.

        Returns:
            Float/bool mask of shape ``(B, N_q + N_t, N_q + N_t)``.
        """
        total = N_q + N_t
        if mask_type == "unimodal":
            mask = torch.zeros(total, total, device=device)
            mask[:N_q, :N_q] = 1
            mask[N_q:, N_q:] = 1
        elif mask_type == "bidirectional":
            mask = torch.ones(total, total, device=device)
        elif mask_type == "causal":
            mask = torch.zeros(total, total, device=device)
            mask[:N_q, :N_q] = 1
            mask[N_q:, :N_q] = 1
            causal = torch.tril(torch.ones(N_t, N_t, device=device))
            mask[N_q:, N_q:] = causal
        else:
            raise ValueError(mask_type)
        return mask.unsqueeze(0).expand(B, -1, -1)
```

### 5.2 两阶段训练伪代码

```python
# --- Stage 1: representation learning (train Q-Former only) ---
vit = load_frozen_vit("eva_vit_g")
qformer = QFormer(num_queries=32, visual_dim=1408)

for images, texts in dataloader_stage1:
    with torch.no_grad():
        visual_features = vit(images)  # (B, 257, 1408)

    # ITC: queries and text do not attend to each other (uni-modal mask).
    z_itc = qformer(texts, visual_features, mask="unimodal")
    t_cls = text_encoder(texts)  # (B, H)
    loss_itc = infonce_loss(z_itc, t_cls)

    # ITM: full bidirectional mixing + binary match head.
    z_itm = qformer(texts, visual_features, mask="bidirectional")
    loss_itm = binary_ce_loss(classifier(z_itm), labels)

    # ITG: causal LM on text with queries as prefix.
    z_itg = qformer(texts, visual_features, mask="causal")
    loss_itg = causal_lm_loss(z_itg, texts)

    loss = loss_itc + loss_itm + loss_itg
    loss.backward()
    optimizer.step()

# --- Stage 2: bootstrap frozen LLM (train Q-Former + projection) ---
llm = load_frozen_llm("flan-t5-xxl")
proj = nn.Linear(768, llm.config.d_model)

for images, texts in dataloader_stage2:
    visual_features = vit(images)
    z = qformer(texts, visual_features, mask="causal")
    z_llm = proj(z)  # (B, 32, d_llm)

    text_embeds = llm.embed_tokens(texts)  # (B, L, d_llm)
    inputs_embeds = torch.cat([z_llm, text_embeds], dim=1)

    outputs = llm(inputs_embeds=inputs_embeds, labels=labels)
    loss = outputs.loss
    loss.backward()
```

### 5.3 使用 LAVIS 库的完整推理

```python
from lavis.models import load_model_and_preprocess
from PIL import Image

model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device="cuda"
)

image = vis_processors["eval"](Image.open("cat.jpg")).unsqueeze(0).to("cuda")
question = txt_processors["eval"]("How many cats are in the image?")

answer = model.generate({"image": image, "prompt": question})
print(answer)  # "there is one cat in the image"
```

---

## 6. 从 CLIP 到 BLIP-2 的范式转变

| 维度 | CLIP | BLIP-2 |
|------|------|--------|
| 融合策略 | 晚期（余弦相似度） | 中期（Q-Former 交叉注意力） |
| 视觉-文本交互 | 无（独立编码） | 多层交叉注意力 |
| LLM 利用 | 无 | 冻结 LLM + soft prompt |
| VQA 能力 | 无（仅检索/分类） | 零样本问答 |
| 可训练参数 | ~400M（全量训练） | 188M（仅 Q-Former） |

BLIP-2 证明了一个重要观点：**不需要从头训练巨大的多模态模型，只需要在冻结的视觉和语言模型之间插入一个轻量级的桥接模块。**

但 Q-Former 的 188M 参数和复杂的三目标训练流程仍然不够简单。下一篇我们将看到 LLaVA 如何用**一层 MLP**替代整个 Q-Former，取得同等甚至更好的效果。

> 参考资料：
>
> 1. Li, J., Li, D., Savarese, S., & Hoi, S. (2023). *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*. ICML 2023.
> 2. Li, J., ... & Hoi, S. (2022). *BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation*. ICML 2022.
> 3. Dai, W., ... & Hoi, S. (2023). *InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning*. NeurIPS 2023.
> 4. Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2024). *Visual Instruction Tuning*. NeurIPS 2024. (LLaVA - 揭示了保留所有视觉 patch token 的优势)
> 5. Alon, U., & Yahav, E. (2021). *On the Bottleneck of Graph Neural Networks and its Practical Implications*. ICLR 2021. (关于信息瓶颈和过度挤压的理论基础)

> 下一篇：[笔记｜多模态融合（四）：LLaVA——用一层 MLP 让大模型"看懂"图片](/chengYi-xun/posts/32-llava/)
