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

> 本文为生成模型系列第十五篇。继 SD3 之后，Black Forest Labs（由原 Stable Diffusion 核心团队创立）推出了 Flux 系列模型。Flux 同样基于 Flow Matching + DiT 架构，但在多处细节上做了不同的设计选择。本文将解析 Flux 的核心架构及其与 SD3 的异同。
>
> ⬅️ 上一篇：[笔记｜生成模型（十四）：Stable Diffusion 3 架构解析 (MMDiT)](/chengYi-xun/posts/14-sd3/)
>
> ➡️ 下一篇：[笔记｜强化学习（一）：强化学习基础与策略梯度](/chengYi-xun/posts/51-rl-basics/)


# Flux 整体架构概览

Black Forest Labs 的成员基本上都是 SD3 的作者，其中三名元老级成员还是初代 Stable Diffusion 论文的作者，因此 FLUX.1 可以看作 SD3 的「精神续作」。Flux 的参数量达到 **12B（120 亿）**，是目前规模最大的开源图像生成模型之一。

FLUX.1 有三个变体：

- **FLUX.1 [pro]**：最强版本，仅通过付费 API 使用
- **FLUX.1 [dev]**：从 pro 模型指引蒸馏（guidance-distilled）而来，质量接近但推理更高效，开源可下载
- **FLUX.1 [schnell]**：从 pro 模型时间步蒸馏（timestep-distilled）而来，仅需 1~4 步即可生成，Apache 2.0 许可证

三个变体共享同一套 12B 参数架构，区别只在蒸馏方式和许可证。

![Flux 整体架构（图源：周弈帆博客）](/chengYi-xun/img/flux_architecture_new.jpg)

Flux 的核心骨干网络由两部分组成：

1. **MM-DiT-Block（双流块）× 19 层**：与 SD3 的 MMDiT 相同，文本和图像序列分别用各自的权重处理，但在注意力层拼接做联合注意力。
2. **Single-DiT-Block（单流块）× 38 层**：文本和图像序列拼接为统一序列，共享同一组权重通过注意力层和 MLP。

右侧 **Ids + RoPE** 分支把位置信息注入每一层的注意力计算，底部 **×T** 循环表示采样器的多步迭代去噪。


# 相对 SD3 的关键改动

在深入细节之前，先梳理 Flux 相对 SD3 的几处改动。后面各节会分别展开。

**1. 图块化策略不同**

SD3 在 Transformer 网络内部使用一个步长为 2 的卷积层（`PatchEmbed`）把 VAE 输出的 16 通道 latent 映射为 token 序列（`in_channels=16`）。Flux 则在进入 Transformer 之前先做 Pixel Unshuffle：把每个 $2 \times 2$ 的相邻像素块拆到通道维度上。如果 VAE 输出 16 通道，经过 $2 \times 2$ unshuffle 后变成 $16 \times 4 = 64$ 通道（`in_channels=64`），空间分辨率缩小一半。之后再用线性层映射到 Transformer 的隐藏维度。这样做的好处是不引入卷积的归纳偏置，且实现更简单。

**2. 文本编码器精简**

SD3 同时使用三个文本编码器（CLIP-L/14 + OpenCLIP-bigG/14 + T5-v1.1-XXL），Flux 去掉了 OpenCLIP-bigG，只保留 CLIP-L/14 和 T5-v1.1-XXL。CLIP-L 的 Pooled output 用于全局语义条件化（通过 AdaLN），T5-XXL 的 token 级输出作为文本流直接进入 Transformer。实测来看，T5-XXL 对长指令遵循和文字拼写的贡献更大，因此去掉 OpenCLIP-bigG 对性能影响有限，还能减少显存开销。

**3. 指引蒸馏替代 CFG**

传统 CFG 需要两次前向：一次用空文本（无条件），一次用给定文本（有条件），然后按 $\hat{\epsilon} = \epsilon_\text{uncond} + s \cdot (\epsilon_\text{cond} - \epsilon_\text{uncond})$ 组合输出。这使推理计算量翻倍。Flux 的 [dev] 模型通过指引蒸馏（guidance distillation），让学生模型直接学会 CFG 的组合效果。蒸馏后，指引强度 $s$ 作为一个实数条件输入模型（和 timestep 一样走正弦编码 + MLP），一次前向就能得到带指引的结果。

**4. RoPE 替换正弦位置编码**

SD3 用的是传统的二维正弦位置编码（加法形式），在 `PatchEmbed` 中一次性加到输入上。Flux 换成了 RoPE（Rotary Positional Embedding）：不是把位置编码加到输入上，而是在每次注意力 Q、K 求内积前，给它们乘上和位置相关的旋转矩阵。这样注意力得分中自然编码了 token 间的相对位置关系，更有利于处理不同分辨率和长宽比的输入。

**5. 新增 Single Stream Block**

SD3 全部使用 Double Stream（MMDiT）块，文本和图像始终在独立的流中处理。Flux 在 19 层 Double Stream 块之后，接上了 38 层 Single Stream 块。在 Single Stream 中，文本和图像被拼接成统一序列，共享同一组权重，并且注意力和 MLP 是并行计算的。这种设计在后期特征已经充分融合时更高效。

**6. 分辨率自适应噪声调度**

Flux 在推理时根据图像分辨率（具体说是 token 数量）动态调整噪声调度的偏移量。分辨率越高，token 越多，偏移量越大，使得采样器在高噪声区域花更多步数。直觉上，高分辨率图像需要更强的噪声才能完全破坏信号，因此应把更多步数分配给高噪声阶段。

# Double Stream 与 Single Stream 的混合设计

## Double Stream Blocks（双流块）

前 19 层采用 Double Stream Blocks，**结构与 SD3 的 MMDiT 完全一致**（详见[上一篇 SD3 解析](/chengYi-xun/posts/14-sd3/)）：

- 文本和图像各自有独立的 AdaLayerNorm、QKV 投影和 MLP
- 计算注意力时，把文本和图像的 Q、K、V 拼接起来做联合注意力
- 注意力输出拆分回各自的流，分别过各自的 MLP
- 时刻编码 `temb`（timestep + guidance + CLIP pooled embedding 合成）通过 AdaLN 调制每一层

![SD3 MMDiT 架构图（图源：SD3 论文 arXiv:2403.03206）](/chengYi-xun/img/sd3_mmdit.png)

**AdaLN（Adaptive Layer Normalization）** 是其中的关键调制机制：与标准 LayerNorm 不同，AdaLN 根据条件向量 `temb` 动态生成 scale、shift、gate 参数，分别用于调制注意力层和 MLP 的输入/输出。

这种设计在早期阶段让文本和图像在各自的模态空间中保持独立表示，同时通过联合注意力做双向信息交换。

## Single Stream Blocks（单流块）

后 38 层切换到 Single Stream Blocks，这是 Flux 相对 SD3 最主要的架构创新。

- 进入 Single Stream 之前，文本和图像特征被拼接成一个长序列
- 之后不再区分文本和图像，共享同一组权重
- 来自 [Scaling Vision Transformers to 22 Billion Parameters](https://arxiv.org/abs/2302.05442) 的设计：**注意力和 MLP 并行执行**，而非传统 Transformer 的串行

![Single Stream Block 结构（图源：Demystifying Flux Architecture, arXiv:2507.09595）](/chengYi-xun/img/flux_single_stream.jpg)

传统 Transformer 的计算顺序是先 Attention 再 MLP，并行设计则让二者同时计算，最后把输出拼接后过一个统一的投影层。下面是 `diffusers` 中的简化实现：

```python
class FluxSingleTransformerBlock(nn.Module):
    def forward(self, hidden_states, temb, image_rotary_emb=None):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
```

过完所有单流块后，文本 token 部分被丢弃，只保留图像 token 做最终输出。

## 为什么要混合两种 Block？

在网络前期，文本和图像的语义差异大，需要独立的权重分别处理（Double Stream）；到了后期，多模态信息已经充分融合，共享权重的 Single Stream 既能减少参数量，又能进一步促进特征交融。arXiv:2507.09595 也做了类似的分析：Double Stream 侧重「专门化和表达力」，Single Stream 侧重「效率和简洁」。

两种 Block 的衔接方式可以在 `diffusers/models/transformers/transformer_flux.py` 中看到：

```python
for block in self.transformer_blocks:
    encoder_hidden_states, hidden_states = block(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        image_rotary_emb=image_rotary_emb,
    )

hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

for block in self.single_transformer_blocks:
    hidden_states = block(
        hidden_states=hidden_states,
        temb=temb,
        image_rotary_emb=image_rotary_emb,
    )

hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]
```

# 核心创新点

## 1. RoPE（旋转式位置编码）的引入

SD3 使用的是传统的二维正弦位置编码（加法形式），在 `PatchEmbed` 中一次性加到输入上。Flux 换成了来自 NLP 领域的 **RoPE（Rotary Positional Embedding）**，最初由苏剑林在 [RoFormer](https://arxiv.org/abs/2104.09864) 中提出。

RoPE 的核心思路：不提前把位置编码加到输入上，而是在每次注意力 Q、K 求内积前，给它们乘上和位置相关的旋转矩阵。这样注意力得分中自然包含 token 间的**相对位置关系**，而不只是绝对位置。

**Flux 中的三维位置编号方案**：

- 每个**文本 token** 的位置编号为 $(0, 0, 0)$
- 位于第 $i$ 行第 $j$ 列的**图像 token** 编号为 $(0, i, j)$

三个维度分别编码为长度 16、56、56 的 RoPE 向量，拼接后正好是 128 维，等于每个注意力头的特征维度：

```python
self.pos_embed = EmbedND(dim=self.inner_dim, theta=10000, axes_dim=[16, 56, 56])
```

RoPE 增强了模型对空间关系的感知，特别有利于处理不同长宽比的图像。

## 2. 指引蒸馏与文本编码器

**文本编码器**方面，Flux 用两个编码器：

- **CLIP L/14**：提取 Pooled output（全局语义），经 MLP 后与时刻编码相加，通过 AdaLN 影响所有层
- **T5-v1.1-XXL**：提取 Token 级别特征序列，作为文本流直接输入 Transformer

去掉 OpenCLIP-bigG 后，Flux 更依赖 T5-XXL 的文本理解能力，这也是它在长指令遵循和文字拼写上表现突出的原因。

**指引蒸馏**方面，FLUX.1 [dev] 是指引蒸馏模型。传统 CFG 需要跑两次前向（一次无条件、一次有条件），指引蒸馏让模型直接学会 CFG 的效果，一次前向搞定：

```python
guidance = torch.tensor([guidance_scale], device=device)
temb = self.time_text_embed(timestep, guidance, pooled_projections)
```

`guidance` 和 `timestep` 一样，以实数形式输入模型，经正弦编码和 MLP 后加到时刻编码上。

## 3. 分辨率自适应噪声调度

Flux 用了 SD3 论文中提出的 *Resolution-dependent shifting of timestep schedules*。根据输入图像的 token 数 $N$，通过线性插值算偏移量 $\mu$：

$$\mu = m \cdot N + b, \quad m = \frac{1.16 - 0.5}{4096 - 256}, \quad b = 0.5 - m \cdot 256$$

token 数范围为 256（$16 \times 16$）到 4096（$64 \times 64$）。$\mu$ 越大，sigma 曲线越上凸，在高噪声区域花更多步数。道理很直接：分辨率越高的图像需要更强的噪声才能完全破坏信号，所以需要把更多步数分配给高噪声阶段。

## 4. QK-Norm 与训练稳定性

为了稳定 12B 参数模型的训练，Flux 在注意力计算中引入了 **QK-Norm**（对 Q 和 K 做 RMSNorm），同样来自 [Scaling Vision Transformers to 22 Billion Parameters](https://arxiv.org/abs/2302.05442)。

具体做法：在 $QK^T$ 点积之前，先对 $Q$ 和 $K$ 分别做 RMSNorm，使 $\|q\| \approx 1$，$\|k\| \approx 1$。这限制了注意力分数的范围，防止 softmax 饱和和梯度异常。

# 总结

Flux 在 SD3 基础上做了几处有针对性的改动：

| 维度 | SD3 | Flux |
|:---|:---|:---|
| 参数量 | ~2B | **12B** |
| 文本编码器 | CLIP-L + OpenCLIP-bigG + T5-XXL | CLIP-L + T5-XXL |
| 位置编码 | 2D 正弦（加法） | **3D RoPE（乘法）** |
| Transformer 块 | MM-DiT × d | **MM-DiT × 19 + Single-DiT × 38** |
| 指引方式 | CFG（两次前向） | **指引蒸馏（一次前向）** |
| 图块化 | 网络内部卷积 | 网络外部 Pixel Unshuffle |
| 噪声调度 | 分辨率相关 | **分辨率相关 + 推理时动态偏移** |

整体来看，Flux 通过 Double Stream + Single Stream 的混合架构、RoPE 和 Flow Matching，在 12B 的参数规模上取得了很好的生成质量。它也是后续做 Flow-GRPO 等强化学习微调的基础模型。

> 参考资料：
>
> 1. Black Forest Labs. (2024). *Flux.1 [dev]*. https://blackforestlabs.ai/
> 2. Su, J. et al. (2024). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. Neurocomputing, 568, 127063. arXiv:2104.09864.
> 3. Dehghani, M., Djolonga, J., Mustafa, B., et al. (2023). *Scaling Vision Transformers to 22 Billion Parameters*. ICML 2023. arXiv:2302.05442.
> 4. 周弈帆. (2024). *Stable Diffusion 3「精神续作」FLUX.1 源码深度前瞻解读*. https://zhouyifan.net/2024/09/03/20240809-flux1/
> 5. Greenberg, O. (2025). *Demystifying Flux Architecture*. arXiv:2507.09595.
> 6. Meng, C., et al. (2023). *On Distillation of Guided Diffusion Models*. CVPR 2023.
> 7. Esser, P., Kulal, S., Blattmann, A., et al. (2024). *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*. ICML 2024. arXiv:2403.03206.

> 下一篇：[笔记｜强化学习（一）：强化学习基础与策略梯度](/chengYi-xun/posts/51-rl-basics/)
