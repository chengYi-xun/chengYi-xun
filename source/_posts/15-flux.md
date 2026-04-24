---
title: 笔记｜生成模型（十五）：Flux 架构解析
date: 2025-08-15 10:00:00
cover: false
mathjax: true
categories:
 - Notes
tags:
 - Deep learning
 - Generative models theory
 - Diffusion Models
 - Flux
series: Diffusion Models theory
---

> 本文为生成模型系列第十五篇。继 Stable Diffusion 3（SD3）之后，由原 SD 核心团队创立的 Black Forest Labs 推出了 Flux 系列模型。Flux 沿用了 Flow Matching 与 Diffusion Transformer（DiT）的底层框架，但在特征对齐、位置编码、网络结构以及蒸馏策略上进行了深度的重构与优化。本文将从数学原理与网络设计的角度，全面解析 Flux 的核心架构。
>
> ⬅️ 上一篇：[笔记｜生成模型（十四）：Stable Diffusion 3 架构解析 (MMDiT)](/chengYi-xun/posts/14-sd3/)
>
> ➡️ 下一篇：[笔记｜强化学习（一）：强化学习基础与策略梯度](/chengYi-xun/posts/51-rl-basics/)

## 一、 引言：Flux 的定位与变体

在 2024 年 8 月发布时，FLUX.1 是开源社区规模最大的文本到图像（Text-to-Image）生成模型之一 [1]。相比于 2B 参数量级的 SD3，FLUX.1 将 Transformer 骨干网络的参数量大幅扩展至 **12B（120 亿）**。相关研究表明，这种对模型容量的暴力扩展（Scaling up）能够显著提升模型对复杂物理规律、空间关系以及长文本指令的遵循能力 [3, 5]。

FLUX.1 包含三个主要变体，它们共享相同的 12B 参数基础架构，但在推理效率与开源协议上有所不同：

- **FLUX.1 [pro]**：闭源的最强基座模型，采用标准的无分类器指引（Classifier-Free Guidance, CFG）进行多步采样。
- **FLUX.1 [dev]**：开源非商用版本。通过**指引蒸馏（Guidance Distillation）**技术 [6]，将 CFG 的双路前向计算压缩为单路，在保持生成质量的同时将推理效率提升一倍。
- **FLUX.1 [schnell]**：开源商用版本（Apache 2.0）。通过**潜空间对抗扩散蒸馏（LADD）**技术 [7]，将原本需要数十步的采样过程极速压缩至 1~4 步。

## 二、 Flux 整体架构与数据流解析

Flux 的核心架构是一种混合了双流（Double Stream）与单流（Single Stream）的 Transformer 网络 [5]。我们首先通过数据流的视角，梳理其从文本/图像输入到最终输出的完整计算图。

![Flux 整体架构（图源：周弈帆博客）](/chengYi-xun/img/flux_architecture_new.jpg)

### 2.1 文本编码与全局条件注入（Text Encoding & Conditioning）

Flux 采用了双文本编码器策略，但相比 SD3 进行了精简，移除了 OpenCLIP-bigG，仅保留 **CLIP-L/14** 与 **T5-v1.1-XXL** [5]。

1. **Token 级特征（细粒度语义）**：T5-XXL 提取长度为 512 的文本特征序列，主要负责长指令遵循与复杂逻辑解析。该序列将直接作为 Transformer 的文本流输入。
2. **全局条件向量（Global Conditioning Vector）**：模型需要一个全局向量来统领整个生成过程。Flux 将以下三种标量/向量映射到相同的隐藏维度（3072维）并求和，构建出统一的全局条件向量 $\mathbf{c}_{\text{global}}$（在 diffusers 源码中常被称为 `temb`）：

   - **时间步 $t$**：经过正弦位置编码（Sinusoidal Embedding）与 MLP 映射。
   - **指引强度 $s$**（仅限 [dev] 版本）：同样经过正弦编码与 MLP 映射。
   - **全局文本语义**：CLIP-L 提取的 Pooled Output 经 MLP 映射。

$$ \mathbf{c}_{\text{global}} = \text{MLP}(\text{Sinusoidal}(t)) + \text{MLP}(\text{Sinusoidal}(s)) + \text{MLP}(\text{CLIP}_{\text{pooled}}) $$

这个 $\mathbf{c}_{\text{global}}$ 向量极其关键，它将通过自适应层归一化（Adaptive Layer Normalization, AdaLN）机制 [8]，在 Transformer 的每一层中动态调制（Modulate）特征的缩放（Scale）与平移（Shift）。

### 2.2 图像潜空间与 Pixel Unshuffle

在图像输入端，Flux 使用预训练的 VAE 将 RGB 图像压缩到潜空间（Latent Space）。假设输入图像分辨率为 $H \times W$，VAE 编码后的 Latent 形状为 $C \times \frac{H}{8} \times \frac{W}{8}$（Flux 中 $C=16$）。

为了将 2D 的 Latent 转换为 1D 的 Token 序列，传统方法（如 ViT 或 SD3）通常使用步长与核大小相同的卷积层（Patch Embedding） [8]。而 Flux 摒弃了卷积，采用了一种无参数的 **Pixel Unshuffle** 操作 [5]：将空间维度上 $2 \times 2$ 的相邻像素块展平，并堆叠至通道维度。

数学上，这一操作将张量从 $(N, 16, h, w)$ 重塑为 $(N, 64, \frac{h}{2}, \frac{w}{2})$。随后，通过一个简单的线性投影层将其映射至 3072 维。
**设计动机**：Pixel Unshuffle 彻底移除了卷积操作带来的归纳偏置（Inductive Bias），使得模型完全依赖 Transformer 的全局自注意力机制来学习空间关系，这在数据量与模型参数量足够大时，往往能带来更高的上限 [5]。

### 2.3 三维旋转位置编码（3D RoPE）

SD3 采用的是传统的加性二维正弦位置编码 [8]。Flux 则引入了自然语言处理中广泛使用的**旋转位置编码（Rotary Positional Embedding, RoPE）** [2]，并将其扩展至三维空间 [5]。

RoPE 的核心思想是不在输入端直接叠加位置向量，而是在计算注意力机制的 Query ($\mathbf{q}$) 和 Key ($\mathbf{k}$) 时，左乘一个与位置相关的正交旋转矩阵 $\mathbf{R}_{\Theta, m}$。这使得内积 $\mathbf{q}_m^\top \mathbf{k}_n$ 能够自然地表达 Token 之间的**相对位置关系** [2]。

在 Flux 中，每个 Token 被赋予一个三维坐标 $(t, y, x)$：

- **文本 Token**：没有空间概念，坐标统一设为 $(0, 0, 0)$。
- **图像 Token**：位于特征图第 $i$ 行 $j$ 列的 Token，坐标设为 $(0, i, j)$。

这三个维度的坐标分别被编码为长度为 16、56、56 的 RoPE 向量，拼接后总长度为 128（恰好等于单个注意力头的特征维度）。3D RoPE 赋予了 Flux 极强的空间外推能力，使其能够完美适应任意长宽比与高分辨率的图像生成 [5]。

## 三、 混合 Transformer 骨干网络

Flux 的 Transformer 骨干网络由 19 层双流块（Double Stream Blocks）和 38 层单流块（Single Stream Blocks）串联而成 [5]。这种混合架构在表达能力与计算效率之间取得了绝佳的平衡。

### 3.1 双流块（Double Stream Blocks）：跨模态对齐

网络的前 19 层采用了与 SD3 MMDiT 结构一致的双流块 [8]。在这一阶段，文本序列 $\mathbf{X}_{\text{txt}}$ 与图像序列 $\mathbf{X}_{\text{img}}$ 保持物理上的隔离。

**计算流程**：

1. **独立调制与投影**：文本和图像分别通过各自的 AdaLN 层（受 $\mathbf{c}_{\text{global}}$ 调制），并使用独立的权重矩阵投影出各自的 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$。
2. **联合注意力（Joint Attention）**：将两者的 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ 在序列维度上拼接，执行全局的自注意力计算。这一步是模态间信息交换的唯一通道。
3. **独立前馈网络（MLP）**：注意力计算完成后，序列被重新拆分为文本和图像两部分，分别输入各自的 MLP 进行非线性变换。

**设计动机**：在网络浅层，文本（离散语义）与图像（连续像素）的特征空间存在巨大的语义鸿沟（Modality Gap）。使用独立的权重矩阵处理不同的模态，能够提供足够的自由度让模型完成初步的特征对齐 [5, 8]。

### 3.2 单流块（Single Stream Blocks）：深度融合与并行计算

在经过 19 层双流块的初步对齐后，网络后段的 38 层无缝切换至单流块 [5]。

![Single Stream Block 结构（图源：Demystifying Flux Architecture, arXiv:2507.09595）](/chengYi-xun/img/flux_single_stream.jpg)

**计算流程**：

1. **序列拼接**：文本与图像序列在进入单流块前被永久拼接为一个统一的长序列 $\mathbf{X}_{\text{joint}} = [\mathbf{X}_{\text{txt}}, \mathbf{X}_{\text{img}}]$。
2. **共享权重**：所有 Token 共享同一套 AdaLN、自注意力权重与 MLP 权重。
3. **并行计算架构**：Flux 借鉴了 ViT-22B 的设计 [3]，打破了传统 Transformer 中 Attention 和 MLP 串行计算的惯例。在单流块中，归一化后的特征同时输入给 Attention 层和 MLP 层，两者的输出在通道维度拼接后，通过一个统一的线性层进行融合：
   $$ \mathbf{X}_{\text{out}} = \mathbf{X}_{\text{in}} + \text{Linear}(\text{Concat}(\text{Attention}(\mathbf{X}_{\text{norm}}), \text{MLP}(\mathbf{X}_{\text{norm}}))) $$

**设计动机**：在网络深层，多模态特征已经实现了高层次的语义融合，此时继续维持两套权重不仅造成参数冗余，还会阻碍特征的进一步交融。单流架构显著提升了参数利用率；而并行计算设计则大幅减少了计算图的深度，提高了硬件执行效率 [3, 5]。

## 四、 核心创新点与数学原理解析

除了骨干网络的重构，Flux 在训练目标、蒸馏策略与训练稳定性上也引入了多项前沿技术。

### 4.1 蒸馏技术：从指引蒸馏到 LADD

传统扩散模型依赖无分类器指引（CFG）来增强文本条件的影响力。数学上，CFG 要求在每个时间步计算两次速度场（或噪声），并进行外推：
$$ \hat{v} = v_{\text{uncond}} + s \cdot (v_{\text{cond}} - v_{\text{uncond}}) $$
其中 $s$ 为指引强度（Guidance Scale）。这种双路计算导致推理时间直接翻倍。Flux 提供了两种截然不同的蒸馏方案来解决这一痛点：

**方案一：指引蒸馏（Guidance Distillation, 用于 FLUX.1 [dev]）**
该方案的目标是**消除 CFG 的双倍计算开销**，但保持原有的采样步数（约 50 步） [6]。

- **教师模型（FLUX.1 [pro]）的计算**：在训练的每一步，给定文本条件 $c$、当前潜变量 $z_t$ 和时间步 $t$，教师模型首先进行两次前向传播，分别计算无条件输出 $v_{\text{uncond}}$ 和有条件输出 $v_{\text{cond}}$。随后，在预设的区间内（如 $[1.0, 10.0]$）**随机采样一个指引强度标量 $s$**（例如抽到 $s=3.5$），利用 CFG 公式计算出融合后的目标速度场 $\hat{v}_{\text{teacher}}$。*注：教师模型的全局条件向量 $\mathbf{c}_{\text{global}}$ 仅包含时间步 $t$ 和文本语义，不包含 $s$。*
- **学生模型（FLUX.1 [dev]）的计算**：学生模型同样接收文本条件 $c$、潜变量 $z_t$ 和时间步 $t$。但与教师不同的是，学生模型**将刚才采样到的标量 $s$ 作为一个额外的输入参数**（经过正弦编码与 MLP 后，融入全局条件向量 $\mathbf{c}_{\text{global}}$ 中）。学生模型仅进行**一次前向传播**，输出 $v_{\text{student}}$。
- **优化目标**：通过均方误差损失（MSE），迫使单次前向的 $v_{\text{student}}$ 逼近教师模型双次前向组合出的 $\hat{v}_{\text{teacher}}$。
- **结果**：蒸馏完成后，推理时只需将期望的 CFG 强度 $s$ 传给模型，即可在单次前向计算中直接获得等效于该 CFG 强度的生成结果，推理速度翻倍。

**方案二：潜空间对抗扩散蒸馏（LADD, 用于 FLUX.1 [schnell]）**
该方案的目标是**极大地压缩采样步数**（从 50 步压缩至 1~4 步） [7]。为了理解 LADD，我们需要先了解它试图解决的痛点：

- **早期对抗蒸馏（如 ADD）的痛点**：早期的加速方案会让学生模型在 1~4 步内生成图像，然后用一个外部的、预训练好的图像分类器（如 DINOv2）作为“判别器”来打分。但这就要求学生模型必须先把潜变量（Latent）通过 VAE 解码成真正的像素图像（Pixel Space），才能喂给 DINOv2。这个解码过程极其消耗显存和计算资源，而且 DINOv2 通常只能处理固定分辨率，导致模型很难支持任意长宽比的生成。
- **LADD 的核心创新：全程在潜空间（Latent Space）完成对抗**。LADD 彻底抛弃了外部的像素级判别器，直接利用强大的**教师模型（FLUX.1 [pro]）本身**来充当判别器的特征提取器 [7]。

**LADD 的具体训练流程如下**：

1. **生成阶段（Generator）**：学生模型（[schnell]）接收纯噪声，试图在极少的步数（如 1~4 步）内，直接预测出一个高质量的潜变量 $\hat{z}_0$。
2. **判别阶段（Discriminator）**：将学生生成的 $\hat{z}_0$ 和真实的潜变量 $z_0$ 分别输入给冻结的教师模型（[pro]）。LADD 并不看教师模型的最终输出，而是**提取教师模型内部注意力层（Attention Layers）的中间特征**。在这些特征之上，接一个轻量级的判别器网络，用来分辨输入到底是“学生生成的 Fake Latent”还是“真实的 Real Latent” [7]。
3. **双重优化目标**：
   - **对抗损失（Adversarial Loss）**：学生模型需要不断优化自身，努力生成极其逼真的潜变量，以“骗过”基于教师特征的判别器。这保证了模型在 1~4 步内就能生成具有极高清晰度和丰富细节（高频信息）的图像。
   - **分数蒸馏采样（Score Distillation Sampling, SDS）**：除了对抗，学生模型还要最小化 SDS 损失。SDS 利用教师模型预测的速度场/噪声，为学生模型提供明确的梯度方向，确保学生生成的图像在整体语义和结构上（低频信息）严格遵循教师模型的分布 [7]。
4. **合成数据训练（Synthetic Data Training）**：为了保证学生模型在极少步数下依然拥有顶级的图文对齐能力，LADD 的训练数据并非真实图像，而是由教师模型（[pro]）提前生成好的高质量合成数据。因为真实图文对往往存在描述缺失或错位的问题，而合成数据则能提供“完美对齐”的监督信号。
5. **结果**：蒸馏后的 [schnell] 模型不仅摆脱了 CFG 的束缚（无需输入 Guidance），还能在 1~4 步内生成极高质量的图像，且天然支持多分辨率和任意长宽比。

### 4.2 分辨率自适应噪声调度（Resolution-Adaptive Noise Scheduling）

在 Flow Matching 框架下，时间步 $t \in [0, 1]$ 决定了当前状态是偏向纯噪声（$t=0$）还是清晰图像（$t=1$）。Flux 引入了基于图像分辨率（即 Token 总数 $N$）的动态时间步调度偏移机制，这一思想最初在 SD3 论文中被提出 [8]。

给定基础时间步 $t$，模型通过一个非线性变换函数将其映射为实际使用的时间步 $t'$。Flux 根据 Token 数量 $N$ 线性插值计算出一个偏移参数 $\mu$：
$$ \mu = m \cdot N + b, \quad m = \frac{1.16 - 0.5}{4096 - 256}, \quad b = 0.5 - m \cdot 256 $$
（其中 256 对应 $16 \times 16$ 的 Latent，4096 对应 $64 \times 64$ 的 Latent）。

**数学直觉**：$\mu$ 越大，时间步调度的曲线越向高噪声区域凸起。这意味着对于高分辨率（大 $N$）图像，采样器会在高噪声区间分配更多的去噪步数。其理论依据在于：高分辨率图像包含更密集的信号与更复杂的全局结构，需要更大幅度的噪声才能实现完全破坏；依据对称性，反向生成时也必须在高噪声阶段投入更多的积分步数以恢复其全局拓扑结构 [8]。

### 4.3 QK-Norm 与 12B 模型训练稳定性

在将 Transformer 扩展至 12B 参数时，注意力机制的数值不稳定性会急剧放大。在标准的 Attention 中，注意力分数由 $\mathbf{Q}\mathbf{K}^\top / \sqrt{d}$ 计算得出。当特征维度或参数量极大时，点积结果容易出现极值，导致 Softmax 函数进入饱和区（梯度消失）或引发梯度爆炸 [3]。

Flux 引入了 **QK-Norm** 技术 [3, 5]：在计算点积之前，先对 Query 和 Key 分别应用 RMSNorm（均方根归一化）：
$$ \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \text{RMSNorm}(\mathbf{Q}) \cdot \text{RMSNorm}(\mathbf{K})^\top \right) \mathbf{V} $$
这一操作强制 $\|\mathbf{q}\| \approx 1$ 且 $\|\mathbf{k}\| \approx 1$，严格限制了注意力分数的数值上限，从根本上消除了 Softmax 饱和现象，确保了 12B 极大规模模型的训练稳定性 [3]。

## 五、 延伸：FLUX.2 的架构演进与能力飞跃

2025 年 11 月，Black Forest Labs 推出了全新预训练的 FLUX.2 [9]。它绝非 FLUX.1 的简单微调版，而是在底层架构与产品能力上进行了一次“脱胎换骨”的重构 [10]。为了让大家更直观地理解这次升级，我们将从“底层架构”和“外在能力”两个维度进行剖析。

### 5.1 底层架构的激进重构

1. **文本编码器大换血：从“关键词匹配器”到“原生多模态大模型”**
   FLUX.1 使用了 CLIP + T5 的组合，这俩本质上还是传统的文本编码器，对复杂物理规律（比如“镜子里的倒影”、“半透明材质”）的理解比较吃力。
   FLUX.2 直接掀桌子，换成了一个 **24B 参数的视觉语言大模型（VLM）—— Mistral Small 3.1** [9]。VLM 是看过海量图文交织数据的，它拥有强大的“世界知识”。这意味着 FLUX.2 不仅能听懂你的提示词，还能像人类一样理解空间关系和物理属性。为了达到最佳效果，FLUX.2 巧妙地提取了 Mistral 中间层（第 10、20、30 层）的特征进行堆叠，因为中间层往往保留了最适合生成图像的语义信息 [10]。

2. **骨干网络大倾斜：单流块的全面胜利**
   FLUX.1 证明了一件事：双流块（文本和图像分开处理）虽然好，但太占参数了。只要在网络最前面用几个双流块把特征对齐，后面的工作完全可以交给更高效的单流块（文本图像拼在一起处理）。
   因此，FLUX.2 将 19 层双流 + 38 层单流的比例，极其激进地调整为 **8 层双流 + 48 层单流** [10]。在 FLUX.2 庞大的 **32B** 参数中，约 73% 的参数都被分配给了融合更深入、计算更高效的单流块。

3. **极致的架构微调（Tweaks）**

   - **全面移除 Bias 参数**：让网络更纯粹，减少冗余。
   - **共享 AdaLN 调制**：双流块和单流块内部不再各自为战，而是共享调制参数，大幅降低了条件注入的开销。
   - **升级 SwiGLU 激活函数**：抛弃了老旧的 GELU，换上了在大语言模型（LLM）中大放异彩的 SwiGLU。
   - **全新 VAE (AutoencoderKLFlux2)**：重新训练了 VAE，极大地提升了对高频细节（如微小的文字、织物纹理）的压缩与重建质量 [9, 10]。

### 5.2 外在能力的史诗级跨越

得益于上述 32B 庞大参数和 VLM 的加持，FLUX.2 在实际使用中展现出了令人惊叹的新特性：

1. **原生 400 万像素（4MP）生成**：FLUX.1 原生只能生成 100 万像素（1MP）左右的图像，而 FLUX.2 直接跨越到了原生 4MP，无需任何后期的放大（Upscaling）处理，直接输出极度清晰的壁纸级大图 [9]。
2. **多图参考与极致的一致性（Multi-Reference）**：FLUX.2 原生支持同时输入**多达 10 张参考图**。这意味着你可以轻松保持同一个角色、同一件商品或同一种画风在不同场景下的高度一致性，这对于商业落地（如电商模特、漫画分镜）是革命性的 [10]。
3. **生成与编辑的“大一统”**：在 FLUX.1 时代，生成（Generation）和局部重绘（Inpainting/Editing）往往需要不同的模型或复杂的工作流。FLUX.2 将它们统一在了一个架构下，实现了无缝的生成与编辑 [10]。
4. **设计级的精准控制**：

   - **HEX 色值识别**：你可以直接在提示词里写 `#FF5733`，模型就能精准还原这个颜色。
   - **结构化 Prompting**：模型甚至能看懂 JSON 格式的提示词，允许你像写代码一样精确控制相机角度、光影布局和物体位置 [9, 10]。

### 5.3 FLUX.1 与 FLUX.2 核心参数对比

| 架构维度 | FLUX.1 | FLUX.2 |
| :--- | :--- | :--- |
| **骨干参数量** | 12B | **32B** |
| **文本编码器** | CLIP-L + T5-XXL | **Mistral Small 3.1 (24B)** |
| **双流/单流层数** | 19 / 38 | **8 / 48** |
| **双流参数占比** | ~54% | **~24%** |
| **Bias 参数** | 存在 | **全部移除** |
| **AdaLN 调制** | 逐层独立生成 | **块间共享** |
| **MLP 激活函数** | GELU | **SwiGLU** |
| **VAE** | FLUX.1 VAE | **AutoencoderKLFlux2** |
| **原生分辨率** | ~1MP | **4MP** |
| **参考图支持** | 依赖外部 ControlNet | **原生支持最多 10 张** |

> **参考文献：**
> 1. Black Forest Labs. (2024). *Flux.1 [dev]*. https://blackforestlabs.ai/
> 2. Su, J. et al. (2024). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. Neurocomputing, 568, 127063. arXiv:2104.09864.
> 3. Dehghani, M., Djolonga, J., Mustafa, B., et al. (2023). *Scaling Vision Transformers to 22 Billion Parameters*. ICML 2023. arXiv:2302.05442.
> 4. 周弈帆. (2024). *Stable Diffusion 3「精神续作」FLUX.1 源码深度前瞻解读*. https://zhouyifan.net/2024/09/03/20240809-flux1/
> 5. Greenberg, O. (2025). *Demystifying Flux Architecture*. arXiv:2507.09595.
> 6. Meng, C., et al. (2023). *On Distillation of Guided Diffusion Models*. CVPR 2023. arXiv:2210.03142.
> 7. Sauer, A., Lorenz, D., Blattmann, A., Rombach, R. (2024). *Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation*. arXiv:2403.12015.
> 8. Esser, P., Kulal, S., Blattmann, A., et al. (2024). *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*. ICML 2024. arXiv:2403.03206.
> 9. Black Forest Labs. (2025). *FLUX.2: Frontier Visual Intelligence*. https://bfl.ai/blog/flux-2
> 10. HuggingFace. (2025). *Diffusers welcomes FLUX-2*. https://huggingface.co/blog/flux-2

> ➡️ 下一篇：[笔记｜强化学习（一）：强化学习基础与策略梯度](/chengYi-xun/posts/51-rl-basics/)