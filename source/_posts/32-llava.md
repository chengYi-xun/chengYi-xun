---
title: 笔记｜多模态融合（四）：LLaVA——用一层 MLP 让大模型"看懂"图片
date: 2026-04-06 00:10:00
categories:

 - Tutorials

tags:

 - LLaVA

 - Visual Instruction Tuning

 - Multimodal LLM

 - Vision-Language Model
series: "多模态融合"
mathjax: true
---

> **论文**：*Visual Instruction Tuning*（Liu et al., 2023, UW-Madison / Microsoft）
> **代码**：[haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA)
> **前置知识**：[上一篇：BLIP-2 / Q-Former](posts/31-blip2-qformer/)

---

## 0. 一个极简的想法

上一篇介绍了 BLIP-2 的 Q-Former——188M 参数、32 个可学习查询、三个预训练目标、两阶段训练。Q-Former 在冻结的 ViT 和 LLM 之间架起了桥梁。

但 LLaVA 的作者问了一个大胆的问题：**如果把 Q-Former 换成一层 MLP 呢？**

|  | Q-Former (BLIP-2) | MLP Projector (LLaVA) |
|--|-------------------|-----------------------|
| 参数量 | 188M | ~20M |
| 结构 | 12 层 Transformer + Cross-Attention | 2 层全连接 |
| 可学习查询 | 32 个 | 无（直接使用所有 patch token） |
| 预训练目标 | ITC + ITM + ITG | 仅因果语言建模 |
| 训练复杂度 | 高（三种掩码切换） | 低（标准自回归） |

结果？LLaVA-1.5 在多数 benchmark 上**持平甚至超越** BLIP-2 和 InstructBLIP。

这揭示了一个深刻的洞察：**当 LLM 足够强大时，复杂的桥接模块可能并非必要——一个简单的线性投影就能让视觉 token "说上" LLM 的语言。**

---

## 1. LLaVA 架构

### 1.1 三个组件

LLaVA 的架构极其简洁，只有三个组件：

```
图像 → [CLIP-ViT] → 视觉 token → [MLP 投影] → 视觉嵌入 ─┐
                                                           ├─→ [LLM (Vicuna/LLaMA)] → 回答
文本 → [Tokenizer] → 文本嵌入 ───────────────────────────────┘
```

![LLaVA 架构示意（摘自 Liu et al., *Visual Instruction Tuning*, NeurIPS 2023；图源 [llava-vl.github.io](https://llava-vl.github.io/)）](/chengYi-xun/img/llava_arch.png)

**数学描述**：

设 CLIP-ViT 的视觉编码器为 $g_\phi$，MLP 投影为 $W$，LLM 为 $f_\theta$。

$$
\begin{aligned}
\mathbf{Z}_v &= g_\phi(\mathbf{I}) \in \mathbb{R}^{N_v \times d_v} \quad &\text{(视觉编码)} \\
\mathbf{H}_v &= \mathbf{Z}_v \mathbf{W}_1 \xrightarrow{\text{GELU}} \mathbf{W}_2 \in \mathbb{R}^{N_v \times d_t} \quad &\text{(MLP 投影)} \\
\mathbf{X}_q &= [\mathbf{H}_v; \;\mathbf{E}_t] \in \mathbb{R}^{(N_v + L) \times d_t} \quad &\text{(拼接)} \\
\mathbf{Y} &= f_\theta(\mathbf{X}_q) \quad &\text{(LLM 自回归生成)}
\end{aligned}
$$

其中 $N_v = 576$（ViT-L/14 对 336×336 图像的 patch 数），$d_v = 1024$，$d_t = 4096$（LLaMA-7B/13B 的隐藏维度）。

### 1.2 为什么 MLP 就够了？

从信息论的角度，BLIP-2 的 Q-Former 做了两件事：

1. **压缩**：257 个 ViT token → 32 个查询 token

2. **对齐**：将视觉特征从 ViT 空间投影到 LLM 空间

LLaVA 的 MLP 只做第 2 件事——不压缩，直接把所有 576 个 patch token 投影到 LLM 空间。

为什么不压缩反而更好？

> **假说（MLP 投影的充分性）**：当 LLM 具有足够的上下文窗口和推理能力时，它可以自行从 576 个视觉 token 中"注意到"与问题相关的部分（通过 self-attention 的 QKV 机制），而无需外部模块预先筛选。Q-Former 的压缩虽然降低了 LLM 的输入长度，但可能丢失了细粒度的空间信息。

实验验证：LLaVA-1.5 在 TextVQA（需要阅读图片中的文字）上的表现显著优于 BLIP-2，正是因为保留了所有 patch token 中的细粒度文字信息。

### 1.3 投影层的数学

LLaVA 原版使用单层线性投影：$\mathbf{H}_v = \mathbf{Z}_v \mathbf{W}$

LLaVA-1.5 升级为两层 MLP + GELU 激活：

$$
\mathbf{H}_v = \text{GELU}(\mathbf{Z}_v \mathbf{W}_1) \mathbf{W}_2
$$

其中 $\mathbf{W}_1 \in \mathbb{R}^{d_v \times d_h}$，$\mathbf{W}_2 \in \mathbb{R}^{d_h \times d_t}$，$d_h$ 通常设为 $d_t$。

参数量：$d_v \times d_h + d_h \times d_t = 1024 \times 4096 + 4096 \times 4096 \approx 21M$。相比 Q-Former 的 188M，轻了近 10 倍。

---

## 2. 视觉指令数据

LLaVA 的另一个核心贡献是**视觉指令数据**的构造方法。

### 2.1 数据生成流程

利用 GPT-4（纯文本版）自动生成视觉对话数据：

```
输入给 GPT-4:

  - 图像的 Caption（如 COCO caption）
  - 图像中物体的 Bounding Box 坐标
  - 任务类型提示（对话/详细描述/复杂推理）

GPT-4 输出:

  - 多轮对话（如 Q: "图片里的人在做什么？" A: "..."）
  - 详细描述（一段关于图片内容的长文本）
  - 推理问题（需要常识或逻辑推理的问答）
```

生成了三种类型共 **158K** 条指令数据：

| 类型 | 数量 | 示例 |
|------|------|------|
| 对话（Conversation） | 58K | 多轮问答 |
| 详细描述（Detail） | 23K | 段落级图片描述 |
| 复杂推理（Reasoning） | 77K | 需要逻辑推理的问答 |

### 2.2 为什么这种方式有效？

传统的视觉-语言数据（如 COCO Caption）格式单一——"一张图配一段描述"。而 LLaVA 的指令数据模拟了**人类与 AI 的自然对话**，包含追问、推理、细节描述等多种交互模式。

这种**指令跟随**（Instruction Following）的数据格式，与 LLM 在文本域的指令微调（如 Alpaca、Vicuna）一脉相承——本质上是把视觉信息也纳入了"指令-回答"的范式中。

---

## 3. 两阶段训练

### 3.1 第一阶段：特征对齐（Pre-alignment）

| 配置 | 值 |
|------|-----|
| 目标 | 让 MLP 学会将视觉特征投影到 LLM 的输入空间 |
| 数据 | 595K 图文对（CC3M 子集过滤后） |
| 训练什么 | **仅 MLP 投影层**（ViT 和 LLM 全部冻结） |
| 任务 | 单轮图像描述生成 |
| 步数 | 1 epoch |
| 学习率 | 1e-3 |
| Batch Size | 256 |

此阶段的训练目标是标准的因果语言建模。给定图像 $\mathbf{I}$ 和对应描述 $\mathbf{w} = (w_1, \ldots, w_L)$：

$$
\mathcal{L}_{\text{align}} = -\sum_{t=1}^{L} \log P_\theta(w_t \mid \mathbf{H}_v, w_1, \ldots, w_{t-1})
$$

注意这与 BLIP-2 第一阶段的三目标训练形成鲜明对比——LLaVA 只用一个最简单的语言建模损失。

### 3.2 第二阶段：指令微调（Instruction Tuning）

| 配置 | 值 |
|------|-----|
| 目标 | 让模型学会遵循多模态指令 |
| 数据 | 158K 视觉指令数据 + 学术 VQA 数据 |
| 训练什么 | **MLP + LLM 全参数**（ViT 冻结） |
| 任务 | 多轮对话、VQA、图像描述 |
| 步数 | 1 epoch |
| 学习率 | 2e-5 |

对多轮对话 $\{(q_1, a_1), (q_2, a_2), \ldots\}$，损失函数只在**回答部分**计算：

$$
\mathcal{L}_{\text{instruct}} = -\sum_{\text{round } r} \sum_{t \in \text{answer}_r} \log P_\theta(w_t \mid \mathbf{H}_v, \text{context}_{<t})
$$

其中 $\text{context}_{<t}$ 包括视觉 token + 之前所有轮次的问答 + 当前轮的问题。

**关键**：问题部分的 token 不计入损失（仅作为上下文），只有回答部分贡献梯度。这与 ChatGPT 的 SFT 阶段一致。

---

## 4. LLaVA-1.5 的改进

LLaVA 原版（2023 年 4 月）虽然概念优雅，但在某些 benchmark 上不如 BLIP-2 系列。LLaVA-1.5（2023 年 10 月）通过以下改进实现了全面超越：

| 改进项 | LLaVA | LLaVA-1.5 |
|--------|-------|-----------|
| 投影层 | 单层线性 | 2 层 MLP + GELU |
| 图像分辨率 | 224×224 | **336×336** |
| 视觉 token 数 | 256 | **576** |
| LLM | Vicuna-7B/13B | Vicuna-7B/13B |
| 学术数据 | 无 | VQAv2, GQA, OKVQA, TextVQA 等 |
| ShareGPT 数据 | 无 | 40K 纯文本对话 |

### 4.1 分辨率的影响

ViT-L/14 处理 336×336 图像时产生 $(336/14)^2 = 576$ 个 patch token，比 224×224 的 $256$ 个多了一倍多。更多 token = 更多细节信息。

这对 **TextVQA**（识别图片中的文字）等任务影响巨大：

| 模型 | TextVQA Acc |
|------|------------|
| BLIP-2 | 42.5 |
| InstructBLIP | 50.1 |
| **LLaVA-1.5** | **58.2** |

高分辨率保留了文字区域的像素细节，而 Q-Former 的 32-token 压缩可能会丢失这些信息。

### 4.2 LLaVA-NeXT：动态分辨率

LLaVA-NeXT（2024）进一步支持**任意分辨率**输入：

1. 将高分辨率图像切割为多个 336×336 的子图

2. 每个子图独立编码，得到各自的 576 个 patch token

3. 附加一张全局缩略图（下采样到 336×336）

4. 所有 token 拼接后送入 LLM

$$
N_{\text{total}} = (n_{\text{tiles}} + 1) \times 576
$$

例如 672×672 图像 → 4 个子图 + 1 个缩略图 → $5 \times 576 = 2880$ 个视觉 token。

---

## 5. 代码实现

### 5.1 LLaVA 核心模块

```python
import torch
import torch.nn as nn
from typing import Optional

from transformers import CLIPVisionModel, LlamaForCausalLM


class LLaVAModel(nn.Module):
    """CLIP ViT patch tokens + MLP projector + LLaMA-style causal LM."""

    def __init__(self, vision_tower_name: str, llm_name: str) -> None:
        super().__init__()
        # 视觉编码器：推理期常冻结；此处演示 ``requires_grad`` 开关。
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name)
        self.vision_tower.requires_grad_(False)

        d_v = self.vision_tower.config.hidden_size  # e.g. 1024 (ViT-L)

        self.llm = LlamaForCausalLM.from_pretrained(llm_name)
        d_t = self.llm.config.hidden_size  # e.g. 4096 (LLaMA-7B)

        # 可训练 connector：LLaVA-1.5 风格 2-layer MLP。
        self.mm_projector = nn.Sequential(
            nn.Linear(d_v, d_t),
            nn.GELU(),
            nn.Linear(d_t, d_t),
        )

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """ViT patch features from the second-to-last block (drop CLS).

        Args:
            images: Pixel batch ``(B, 3, H, W)`` (336px → ``N_v=576`` for L/14).

        Returns:
            Patch tokens ``(B, N_v, d_v)`` where ``N_v`` depends on resolution.
        """
        outputs = self.vision_tower(images, output_hidden_states=True)
        # 倒数第二层 hidden state；丢弃 CLS（``[:, 1:]``）。
        features = outputs.hidden_states[-2][:, 1:]
        return features  # (B, N_v, d_v)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """Project vision tokens, concat with text embeddings, run LLM.

        Args:
            images: ``(B, 3, H, W)``.
            input_ids: Text token ids ``(B, L)``.
            attention_mask: Mask ``(B, L)``; 真实训练需按拼接长度扩展。
            labels: Causal LM 标签 ``(B, L)`` 或带 ignore_index 的对齐张量。

        Returns:
            ``LlamaForCausalLM`` 输出（含 ``loss`` / ``logits`` 等）。
        """
        # 1. 视觉编码
        image_features = self.encode_images(images)  # (B, N_v, d_v)
        # 2. MLP 投影
        image_embeds = self.mm_projector(image_features)  # (B, N_v, d_t)
        # 3. 文本嵌入
        text_embeds = self.llm.model.embed_tokens(input_ids)  # (B, L, d_t)
        # 4. 拼接
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        # (B, N_v+L, d_t)
        # 5. LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs
```

### 5.2 两阶段训练

```python
# === 第一阶段：特征对齐 ===
model = LLaVAModel(
    "openai/clip-vit-large-patch14-336",
    "lmsys/vicuna-7b-v1.5",
)

# 冻结 ViT 与 LLM，仅训练投影层
model.vision_tower.requires_grad_(False)
for param in model.llm.parameters():
    param.requires_grad = False
for param in model.mm_projector.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(model.mm_projector.parameters(), lr=1e-3)

for images, captions in pretrain_dataloader:
    outputs = model(
        images,
        captions.input_ids,
        captions.attention_mask,
        labels=captions.labels,
    )
    outputs.loss.backward()
    optimizer.step()

# === 第二阶段：指令微调 ===
# 解冻 LLM 全参数
for param in model.llm.parameters():
    param.requires_grad = True

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=2e-5)

for images, conversations in instruct_dataloader:
    # labels 仅在 assistant 回答位置非 -100
    outputs = model(
        images,
        conversations.input_ids,
        conversations.attention_mask,
        labels=conversations.labels,
    )
    outputs.loss.backward()
    optimizer.step()
```

### 5.3 推理

```python
@torch.no_grad()
def chat(model, tokenizer, image, question: str) -> str:
    """单轮对话占位：需将 ``<image>`` 处替换为 ``image_embeds`` 再 ``generate``。

    Args:
        model: ``LLaVAModel`` 实例。
        tokenizer: 与 Vicuna/LLaMA 对齐的分词器。
        image: ``(3, H, W)`` 单张图（已做与训练一致的归一化）。
        question: 用户问题字符串。

    Returns:
        解码后的 assistant 回复（示意；真实管线需拼 ``inputs_embeds``）。
    """
    # 构造对话模板
    prompt = f"<image>\nUSER: {question}\nASSISTANT:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    # ViT → projector
    image_features = model.encode_images(image.unsqueeze(0).cuda())
    image_embeds = model.mm_projector(image_features)  # (1, N_v, d_t)

    # 将 ``<image>`` 行替换为 ``image_embeds`` 再构造 ``inputs_embeds``；
    # 示意代码略去索引对齐。
    text_embeds = model.llm.model.embed_tokens(input_ids)  # (1, L, d_t)
    inputs_embeds = text_embeds

    output_ids = model.llm.generate(
        inputs_embeds=inputs_embeds,
        max_new_tokens=512,
        temperature=0.2,
    )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer
```

---

## 6. 实验结果与对比

### 6.1 Benchmark 全面对比

| 模型 | VQAv2 | GQA | TextVQA | POPE | MME |
|------|-------|-----|---------|------|-----|
| BLIP-2 (FlanT5-XXL) | 65.0 | 44.7 | 42.5 | 85.3 | - |
| InstructBLIP (Vicuna-7B) | - | 49.2 | 50.1 | - | - |
| **LLaVA-1.5 (Vicuna-7B)** | **78.5** | **62.0** | **58.2** | **85.9** | **1510** |
| **LLaVA-1.5 (Vicuna-13B)** | **80.0** | **63.3** | **61.3** | **85.9** | **1531** |

LLaVA-1.5 在几乎所有指标上全面超越 Q-Former 方案——用更简单的架构取得了更好的结果。

### 6.2 投影层消融实验

| 投影类型 | 参数量 | VQAv2 | GQA |
|---------|--------|-------|-----|
| 线性投影 | ~4M | 74.5 | 57.8 |
| **2 层 MLP + GELU** | **~21M** | **78.5** | **62.0** |
| 3 层 MLP | ~38M | 78.3 | 61.8 |

2 层 MLP 是最佳的性价比选择——比线性投影好很多，但增加到 3 层没有额外收益。

---

## 7. 从 BLIP-2 到 LLaVA 的启示

LLaVA 的成功传递了几个深层信息：

1. **简单架构 + 好数据 > 复杂架构**：Q-Former 的三目标训练虽然优雅，但 LLaVA 用一层 MLP + 高质量指令数据就超越了它

2. **不要压缩视觉信息**：保留所有 576 个 patch token 比压缩到 32 个查询更好——LLM 有足够的注意力能力自行筛选

3. **LLM 本身就是最好的融合器**：只要视觉 token 被正确投影到 LLM 的嵌入空间，LLM 的 self-attention 就天然地实现了"中期融合"

但 LLaVA 仍然是一个**模块化**方案——视觉编码器和 LLM 是分开预训练的，通过 MLP "粘合"。这引出了一个更根本的问题：**能不能训练一个真正的"原生多模态"模型，从第一层开始就同时处理图像和文本？**

下一篇将介绍 Flamingo 的门控交叉注意力和 Chameleon 的统一 token 方案。

> **下一篇**：[笔记｜多模态融合（五）：原生多模态——从 Flamingo 到 Chameleon](posts/33-native-multimodal/)

---

**参考文献**

1. Liu, H., et al. (2023). *Visual Instruction Tuning*. NeurIPS 2023.

2. Liu, H., et al. (2023). *Improved Baselines with Visual Instruction Tuning* (LLaVA-1.5). arXiv:2310.03744.

3. Liu, H., et al. (2024). *LLaVA-NeXT: Improved Reasoning, OCR, and World Knowledge*. Blog post.
