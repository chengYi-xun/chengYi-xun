---
title: 笔记｜多模态融合（六）：2026 前沿——InternVL、Qwen-VL、Mamba 与多模态的未来
date: 2026-04-06 00:20:00
categories:
 - Tutorials
tags:
 - InternVL
 - Qwen-VL
 - Mamba
 - State Space Model
 - Multimodal Frontier
series: "多模态融合"
mathjax: true
---

> **核心论文**：
> - *InternVL 2.5*（Chen et al., 2024, Shanghai AI Lab）
>
> ⬅️ 上一篇：[笔记｜多模态融合（五）：原生多模态——从 Flamingo 到 Chameleon](/chengYi-xun/posts/33-native-multimodal/)
>
> ➡️ 下一篇：[笔记｜世界模型（一）：什么是世界模型？从认知科学到深度学习](/chengYi-xun/posts/35-world-model-basics/)
>
> - *Qwen2.5-VL*（Bai et al., 2025, Alibaba）
>
> - *OmniMamba*（Chen et al., 2025）
>
> - *FUSION*（Li et al., 2025）
>
> **前置知识**：[上一篇：Flamingo 与 Chameleon](posts/33-native-multimodal/)

---

## 0. 一张表格暴露的差距

文档问答基准 **DocVQA（test）** 可粗略反映模型在扫描件/表格类图像上的细粒度阅读能力。下表数据摘自 InternVL 2.5 技术报告 Table 7 与 Qwen2.5-VL 技术报告（GPT-4o 为 `20240513` 快照）。

| 模型 | DocVQA (test) % | 模型参数 | 开源 |
|------|----------------|---------|------|
| GPT-4o-20240513 | 92.8 | 未知 | ✗ |
| Claude 3.5 Sonnet | 95.2 | 未知 | ✗ |
| **InternVL 2.5-78B** | **95.1** | 78B | **✓** |
| Qwen2.5-VL-72B | 96.4 | 72B | ✓ |
| Qwen2-VL-72B | 96.5 | 72B | ✓ |

两年前，开源模型和闭源模型之间在文档理解上差距明显。到 2024–2025 年，InternVL 2.5 在 **MMMU（val）** 上达到 **70.1%**，略高于同期报告的 GPT-4o（**69.1%**），并成为首个在该基准上超过 70% 的开源 MLLM。

这一篇梳理推动这些进展的关键技术创新。

---

## 1. InternVL 2.5：ViT-MLP-LLM 的极致

### 1.1 架构概览

![InternVL 2.5 等在 OpenCompass 上的表现与部分闭源模型对照（摘自 Chen et al., arXiv:2412.05271 文内图 1）](/chengYi-xun/img/internvl_arch.png)

InternVL 2.5 延续了 LLaVA 式的 **ViT-MLP-LLM** 三段式架构，但在每个组件上做到了极致：

$$
\text{InternViT-6B}(\mathbf{I}) \xrightarrow{N_v \text{ tokens}} \text{MLP Projector} \xrightarrow{N_v \text{ tokens}} \text{InternLM2-Chat-20B/76B}
$$

| 组件 | InternVL 2.5 | LLaVA-1.5 |
|------|-------------|-----------|
| 视觉编码器 | InternViT-6B（自研，6B 参数） | CLIP-ViT-L（0.3B） |
| 投影层 | 2 层 MLP | 2 层 MLP |
| LLM | InternLM2-Chat-20B/76B | Vicuna-7B/13B |
| 动态分辨率 | ✓（动态切块；单图训练常见上界约 36 个 448×448 tile） | ✗（固定 336×336） |

### 1.2 动态分辨率

这是 InternVL 2.5 的核心创新之一。不同任务需要不同分辨率：

- **文档 OCR**：需要极高分辨率读取小字

- **场景理解**：中等分辨率即可

- **图标分类**：低分辨率足够

InternVL 2.5 的动态分辨率策略：

1. **输入**：任意分辨率的图像 $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$

2. **计算最优切分**：根据图像宽高比，选择不超过配置上限的 $n$ 个 448×448 子图（tile；InternVL 2.5 训练配置中单图常见上界约 **36**）

3. **编码**：每个 tile 独立通过 InternViT-6B，产生 256 个 token

4. **缩略图**：附加一张下采样到 448×448 的全局缩略图

5. **拼接**：所有 token 送入 LLM

$$
N_{\text{total}} = (n_{\text{tiles}} + 1) \times 256
$$

在单图训练常见上界 $n_{\max}=36$ 时，最多约 $37 \times 256 = 9472$ 个视觉 token。

### 1.3 训练策略

InternVL 2.5 在 InternVL 2.0 的基础上改进了三个方面：

1. **数据质量**：更严格的数据过滤和质量控制

2. **Chain-of-Thought 推理**：在训练数据中加入推理链，引导模型逐步思考

3. **测试时计算缩放（Test-Time Scaling）**：以 CoT 等方式扩大推理计算；在 MMMU（val）上，CoT 相对直接作答约 **+3.7** 个百分点，并可与多数投票等策略叠加

### 1.4 性能

| Benchmark | InternVL 2.5-78B | GPT-4o | Claude 3.5 Sonnet |
|-----------|-----------------|--------|-------------------|
| MMMU (val) | **70.1** | 69.1 | 68.3 |
| MathVista | 72.3 | 63.8 | 67.7 |
| OCRBench | 854 | 736 | 788 |
| DocVQA (test) | 95.1 | 92.8 | 95.2 |

首个在 MMMU（val）上超过 70% 的开源 MLLM；在 MathVista（test-mini）等推理类指标上亦不低于同期报告的 GPT-4o / Claude 3.5 Sonnet。文档类子项上各家互有高低（同一报告 Table 7 中 DocVQA、TextVQA 等列并不总是同向领先）。

---

## 2. Qwen2.5-VL：原生视频理解

### 2.1 核心创新

Qwen2.5-VL（2025 年 2 月技术报告）的特点不在于刷高单项 benchmark，而在于**能力的广度**：

1. **原生视频理解**：支持任意长度视频输入，无需逐帧提取

2. **文档解析**：结构化输出表格、图表信息

3. **空间推理**：理解物体之间的空间关系和位置

4. **多语言**：中英日韩等多语言的多模态理解

### 2.2 视觉编码

Qwen2.5-VL 使用改进的 ViT 编码器，支持**动态分辨率和动态帧率**：

对视频输入 $\mathbf{V} \in \mathbb{R}^{T \times H \times W \times 3}$：

$$
\text{ViT}_{\text{Qwen}}(\mathbf{V}) = \text{Concat}\left[\text{ViT}(\mathbf{V}_{t_1}), \text{ViT}(\mathbf{V}_{t_2}), \ldots, \text{ViT}(\mathbf{V}_{t_K})\right]
$$

其中 $\{t_1, \ldots, t_K\}$ 是动态采样的关键帧——运动剧烈的片段采样更密集，静态片段更稀疏。

### 2.3 实际应用场景

```python
# Qwen2.5-VL：典型加载与消息构造
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

model_id = "Qwen/Qwen2.5-VL-72B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

# 文档解析
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "invoice.png"},
        {
            "type": "text",
            "text": "请将这张发票中的所有字段提取为 JSON 格式",
        },
    ],
}]

# 视频理解
messages = [{
    "role": "user",
    "content": [
        {"type": "video", "video": "cooking.mp4"},
        {"type": "text", "text": "这段视频中厨师做了哪些步骤？"},
    ],
}]
```

---

## 3. Mamba 多模态：线性复杂度的未来

### 3.1 Transformer 的瓶颈

标准 Transformer self-attention 的复杂度为 $O(N^2 \cdot d)$，其中 $N$ 为序列长度。

对多模态输入：$N = N_{\text{visual}} + N_{\text{text}}$。当 InternVL 使用 9472 个视觉 token + 2048 个文本 token 时，$N = 11520$，self-attention 的开销已经很大。

对视频理解更是灾难级的：100 帧 × 256 token/帧 = 25,600 个 token → $O(N^2) \approx 6.5 \times 10^8$。

### 3.2 State Space Model 的 $O(N)$ 复杂度

**Mamba**（Gu & Dao, 2023）基于 Selective State Space Model，复杂度为线性 $O(N \cdot d)$。

核心递推方程：

$$
\begin{aligned}
h_t &= \bar{A} h_{t-1} + \bar{B} x_t \\
y_t &= C h_t
\end{aligned}
$$

其中 $h_t \in \mathbb{R}^{d_{\text{state}}}$ 是隐状态，$\bar{A}, \bar{B}$ 是离散化后的状态转移矩阵。Mamba 的创新在于让 $\bar{A}, \bar{B}, C$ **依赖于输入** $x_t$（selective），从而获得类似注意力的"选择性关注"能力。

### 3.3 多模态 Mamba 架构

**VL-Mamba**（Qiao et al., 2024）：最早将 Mamba 应用于多模态学习
- 使用 Vision Selective Scan (VSS) 模块处理 2D 图像特征
- Mamba-2 backbone 替代 Transformer
- 在 LLaVA-1.5 的 benchmark 上达到相当性能，但推理速度更快

**OmniMamba**（Chen et al., 2025 年 3 月）：首个基于线性架构的多模态**理解与生成统一**模型

| 特点 | OmniMamba | LLaVA-NeXT | Show-o |
|------|-----------|------------|--------|
| 架构 | Mamba-2 | Transformer | Transformer |
| 推理加速 | **119.2×** (长序列) | 基准 | - |
| GPU 内存节省 | **63%** | 基准 | - |
| 训练数据 | **2M** 图文对 | 2B 图文对 | 2B 图文对 |

核心设计与理论突破：
1. **解耦词汇表与模态特定 LoRA**：由于文本和图像 token 在特征空间和转移矩阵 $\bar{A}, \bar{B}$ 上的最优动态特性不同，OmniMamba 避免了 Chameleon 式的完全参数共享，而是使用独立的词汇表和任务特定的 LoRA 适配器，有效缓解了模态竞争。
2. **解耦两阶段训练**：为了解决图文生成任务之间的数据不平衡，采用先理解后生成的两阶段策略，仅用 200 万图文对（比 Show-o 少 1000 倍）就达到了极具竞争力的性能。
3. **统一的自回归生成**：在 Mamba-2 的线性状态空间中，通过 next-token prediction 同时实现文本和图像的生成，彻底摆脱了 Transformer 的二次复杂度诅咒。

**mmMamba**（2025）：通过**知识蒸馏**从 Transformer MLLM 到 Mamba

$$
\mathcal{L}_{\text{distill}} = \text{KL}(P_{\text{Mamba}}(\cdot | x) \;\|\; P_{\text{Transformer}}(\cdot | x))
$$

混合架构（mmMamba-hybrid）保留少量 Transformer 层处理关键位置的全局依赖：

| 变体 | 加速 | 内存节省 | 性能（vs Transformer） |
|------|------|---------|---------------------|
| mmMamba-linear | 20.6× | 75.9% | -5.2% |
| mmMamba-hybrid | 13.5× | 60.2% | -1.8% |

---

## 4. 最新融合策略（2025-2026）

### 4.1 MMDiT / Joint Attention：扩散模型中的多模态融合

在讨论多模态融合时不能忽略**扩散模型**领域的重要进展。Stable Diffusion 3 的 **MMDiT**（Multimodal Diffusion Transformer）和 Flux 的混合架构，将多模态融合直接嵌入了去噪过程。

#### 设计溯源

MMDiT 的设计并非凭空产生，它是多条技术路线的汇合点：

| 来源 | 贡献 | 时间 |
|------|------|------|
| **DiT**（Peebles & Xie） | 证明 Transformer 可以替代 U-Net 作为扩散骨干 | 2023 |
| **ViLBERT**（Lu et al.） | 双流 Transformer + 交叉注意力的多模态架构 | 2019 |
| **Cross-Attention 条件化** | U-Net 中文本通过 Cross-Attention 指导图像生成（SD 1.x/2.x） | 2022 |
| **Pooled CLIP Embedding** | 全局文本向量通过 AdaLN 注入（DiT） | 2023 |

SD 1.x/2.x 的条件化方式是**单向 Cross-Attention**——图像特征作为 Query，文本特征作为 Key/Value：

$$
\text{SD 1.x}: \quad \mathbf{x}' = \text{SelfAttn}(\mathbf{x}) + \text{CrossAttn}(\underbrace{\mathbf{x}}_Q, \underbrace{\mathbf{c}_{\text{text}}}_{K,V})
$$

这是不对称的：图像可以"看到"文本，但**文本看不到图像**。而语言理解本身可能需要根据图像上下文调整——比如"realistic"在不同图像内容下应强调不同的视觉属性。

MMDiT 通过 **Joint Attention** 打破了这种不对称：

$$
\begin{aligned}
Q_v, K_v, V_v &= \text{Linear}_v(\mathbf{x}_{\text{img}}), \quad Q_t, K_t, V_t = \text{Linear}_t(\mathbf{x}_{\text{txt}}) \\
[Q, K, V] &= [\text{Concat}(Q_v, Q_t),\; \text{Concat}(K_v, K_t),\; \text{Concat}(V_v, V_t)] \\
[\mathbf{y}_v; \mathbf{y}_t] &= \text{SelfAttn}(Q, K, V)
\end{aligned}
$$

文本和图像各自拥有独立的投影权重（保留模态特性），但在注意力矩阵中**共享同一个 softmax**（实现深度双向交互）。计算完成后再拆分，分别通过各自的 MLP。

**Flux 的混合策略**更进一步：前 19 层用 Double Stream（双流 Joint Attention），后 38 层切换为 Single Stream（完全共享参数）——前期保留模态差异，后期充分融合。

#### MMDiT vs LLM 多模态融合：本质差异

MMDiT 和 LLM 方案（LLaVA/BLIP-2）虽然都在做多模态融合，但设计哲学截然不同：

| 维度 | MMDiT (SD3/Flux) | LLM 方案 (LLaVA/BLIP-2) |
|------|-----------------|------------------------|
| **任务** | 图像**生成**（去噪） | 图像**理解**（问答/描述） |
| **融合对称性** | **双向对称**——文本影响图像，图像也影响文本 | **单向**——视觉 token 被 LLM 处理，但视觉编码器不更新 |
| **模态参数** | 各自独立的 Linear/MLP，共享注意力 | 视觉编码器通常冻结，LLM 有自己的参数 |
| **文本角色** | 条件信号（被持续修改和细化） | 指令/上下文（固定不变） |
| **迭代次数** | 每个去噪步重复完整的多模态融合 | 单次前向传播 |

数学上，关键区别体现在**文本表征是否被视觉信息更新**：

$$
\begin{aligned}
\text{LLaVA}: \quad &\mathbf{t}' = \mathbf{t} \quad \text{(文本嵌入不变)} \\
\text{MMDiT}: \quad &\mathbf{t}' = \mathbf{t} + \text{Attn}(\mathbf{t}, [\mathbf{v};\mathbf{t}]) \quad \text{(文本被图像更新)}
\end{aligned}
$$

在 LLaVA 中，视觉 token 被投影后拼接在文本前面，进入 LLM 的 self-attention。虽然 self-attention 理论上让文本也能"看到"视觉 token，但**视觉编码器的输出是固定的**——它在编码图像时完全不知道用户的问题是什么。

而在 MMDiT 中，文本和图像的表征在**每一层都被对方更新**。这对图像生成至关重要：在去噪的早期步，模型需要理解文本的整体语义（"一只猫在月光下"）；在后期步，需要细化文本中的具体属性（"月光是蓝色的"）。文本表征随图像特征的变化而动态调整，使条件化更加精确。

这种差异解释了为什么 SD3/Flux 在复杂 prompt 的遵循度上远超 SD 1.x——双向融合让模型对"一只**红色**的猫坐在**蓝色**椅子旁边"这种属性绑定描述有了更好的理解。

关于 MMDiT 的详细架构和代码实现，可参考本站[第十四篇 SD3 架构解析](posts/15-sd3/)和[第十五篇 Flux 架构解析](posts/16-flux/)。

### 4.2 FUSION：文本引导的视觉编码

传统方案中视觉编码器独立于文本，FUSION（Li et al., 2025）打破了这个假设：

**Text-Guided Vision Encoding**：在视觉编码过程中就注入文本信息：

$$
\mathbf{V}' = \text{ViT}(\mathbf{I}, \mathbf{t}_{\text{query}})
$$

让视觉编码器"知道"用户在问什么，从而提取更相关的视觉特征。

**效果**：FUSION-3B 超越了 Cambrian-1-8B 和 Florence-VL-8B，仅使用 630 个视觉 token。

### 4.3 BRIDGE：轻量双向交叉注意力

BRIDGE（2025）提出了一种介于早期融合和晚期融合之间的"中间地带"：

$$
\text{BRIDGE}: \quad \mathbf{V}' = \mathbf{V} + \text{BiCrossAttn}(\mathbf{V}, \mathbf{T}), \quad \mathbf{T}' = \mathbf{T} + \text{BiCrossAttn}(\mathbf{T}, \mathbf{V})
$$

在视觉和文本编码器的**顶部几层**放置轻量的双向交叉注意力，保持双编码器的检索效率，同时获得深度融合的性能。

### 4.4 Falcon Perception：早期融合 + 3D RoPE

Falcon Perception（TII, 2026）是最新的早期融合方案：

- **600M 参数**的统一 dense Transformer

- 图像 patch 和文本 token 从第一层起共享参数

- **3D Rotary Position Embedding (GGROPE)**：保持图像的 2D 空间关系

- **混合注意力**：图像用双向注意力，文本用因果注意力

$$
\text{RoPE}_{3D}(i, j, k) = \text{RoPE}_{\text{row}}(i) \otimes \text{RoPE}_{\text{col}}(j) \otimes \text{RoPE}_{\text{seq}}(k)
$$

### 4.5 统一生成模型

| 模型 | 能力 | 架构 |
|------|------|------|
| X-Fusion (ICCV 2025) | 文→图 + 图→文 | 双塔 + 冻结 LLM |
| Qwen3.5 Omni (2026.3) | 文/图/音/视频 全模态 | Thinker-Talker + MoE |
| Gemini 2.5 Pro | 1M context + 多模态 | 原生多模态 |

---

## 5. Benchmark 综合对比

### 5.1 图像理解

| 模型 | 参数量 | MMMU | MathVista | DocVQA | OCRBench | 架构 |
|------|--------|------|-----------|--------|----------|------|
| GPT-4o | ? | 69.1 | 63.8 | 92.8 | 736 | 原生 |
| Gemini 2.5 Pro | ? | 82.0 | 80.4 | - | - | 原生 |
| **InternVL 2.5-78B** | 78B | **70.1** | 72.3 | **95.1** | **854** | ViT-MLP-LLM |
| Qwen2.5-VL-72B | 72B | 70.2 | 74.8 | 96.4 | 885 | ViT-MLP-LLM |
| LLaVA-NeXT-34B | 34B | 51.1 | 46.5 | 74.6 | - | ViT-MLP-LLM |
| Chameleon-34B | 34B | - | - | - | - | 原生 token |

### 5.2 效率对比

| 模型 | 推理延迟（相对） | GPU 内存 | 长序列支持 |
|------|----------------|---------|-----------|
| InternVL 2.5 (Transformer) | 1× | 1× | 8K token |
| OmniMamba (Mamba-2) | **0.008×** | **0.37×** | **103K token** |
| mmMamba-hybrid | 0.07× | 0.40× | 103K token |

---

## 6. 未来展望

### 6.1 原生多模态 vs 模块化

当前领先模型（InternVL、Qwen-VL）仍然是模块化的 ViT-MLP-LLM 方案。原生多模态模型（Chameleon）虽然概念优美，但在纯理解任务上暂时落后。

**预测**：随着训练数据和计算量的增加，原生方案将逐步追平并超越模块化方案——因为它没有信息瓶颈。

### 6.2 高效架构

Mamba/SSM 在长序列处理上有巨大的效率优势（线性复杂度），但在"需要全局比较"的任务上（如检索、匹配）表现稍弱。

**预测**：混合架构（少量 Transformer 层 + 大量 Mamba 层）将成为主流——在效率和性能之间取得平衡。

### 6.3 世界模型

当多模态模型可以同时理解和生成图像、视频、文本、音频时，它本质上已经接近一个"世界模型"——能够通过内部模拟来理解物理世界。

```
文本 ("把红球推向蓝球") → 世界模型 → 视频（红球滚动撞击蓝球的物理过程）
```

Qwen3.5 Omni 和 Gemini 2.5 Pro 已经在这个方向上迈出了第一步。

### 6.4 具身智能

多模态融合的终极应用是**具身智能**（Embodied AI）：机器人需要同时处理视觉（摄像头）、触觉（力传感器）、语言（指令）、空间（3D 点云）等多种模态。

这将是多模态融合领域的下一个大战场。

---

## 7. 系列总结

六篇文章走完了多模态融合从基础到前沿的完整路径：

| 篇章 | 核心内容 | 关键公式/概念 |
|------|---------|-------------|
| **一：基础理论** | 三级融合框架 | 早期/中期/晚期融合, Cross-Attention |
| **二：CLIP** | 对比学习 | InfoNCE Loss, SigLIP, ETF 几何 |
| **三：BLIP-2** | Q-Former 交叉注意力 | ITC/ITM/ITG 三目标, 注意力掩码 |
| **四：LLaVA** | 视觉指令微调 | MLP 投影, 两阶段训练 |
| **五：原生多模态** | Flamingo + Chameleon | Gated Cross-Attn, VQ-VAE |
| **六：前沿** | InternVL, Mamba, MMDiT, FUSION | 动态分辨率, SSM $O(N)$, Joint Attention |

多模态融合的发展史，就是一部**如何让不同信号形态的数据在同一个计算框架中高效协作**的探索史。从最初的特征拼接，到对比学习的共享空间，到交叉注意力的深度交互，再到统一 token 的原生融合——每一步都在追求更深层、更自然的模态间信息交换。

> 参考资料：
>
> 1. Chen, Z., ... & Wang, W. (2024). *InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks*. CVPR 2024.
> 2. Chen, Z., ... & Wang, W. (2024). *Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling* (InternVL 2.5). arXiv:2412.05271.
> 3. Bai, J., ... & Zhou, J. (2025). *Qwen2.5-VL Technical Report*. arXiv:2502.13923.
> 4. Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752.
> 5. Chen, J., ... & Wang, Y. (2025). *OmniMamba: Efficient and Unified Multimodal Understanding and Generation via State Space Models*. arXiv:2503.08686. (华中科技大学 & 字节跳动)
> 6. Li, Y., ... & Zhang, Z. (2025). *FUSION: Fully Integration of Vision-Language Representations for Deep Cross-Modal Understanding*. arXiv:2504.09925.

> 下一篇：[笔记｜世界模型（一）：什么是世界模型？从贝叶斯推断到联合嵌入](/chengYi-xun/posts/35-world-model-basics/)
