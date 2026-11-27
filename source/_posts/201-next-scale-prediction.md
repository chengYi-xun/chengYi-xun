---
title: 讨论｜Next-scale Prediction：从 VAR 到长时程世界模型
date: 2026-09-08 12:00:00
cover: false
mathjax: true
description: 从 VAR 的多尺度量化、训练和推理出发，对比 MAR、BAR、GRN 的预测方式，再看 InfinityStar 和 WALL-SS 怎样把尺度预测用于视频与世界模型。
categories:
 - Notes
tags:
 - Next-scale Prediction
 - Autoregressive Models
 - Visual Generation
 - World Model
series: 视觉自回归
---

> 如果把图像表示拆成由粗到细的几层，生成模型就可以一轮预测整层。本文从 VAR 的训练和推理讲起，再看 MAR、BAR、GRN 如何改变表示或预测方式，以及 InfinityStar、WALL-SS 如何把下一尺度预测用于视频和世界模型。
>
> 相关基础：[VAE 理论](/chengYi-xun/posts/4-VAE-theory/)、[Flow Matching](/chengYi-xun/posts/13-flow_matching/)、[世界模型概览](/chengYi-xun/posts/101-world-model-basics/)。

## 一、图像生成，为什么要预测“下一尺度”？

把语言模型的逐 token 预测用到图像上，可以先把图像编码成二维 token 网格，再按从左到右、从上到下的顺序逐个生成。这个顺序容易实现，但画面中相隔很远的区域也可能有关联，例如人物两侧的光照和阴影。

VAR（Visual AutoRegressive Modeling）把生成单位换成一整张潜空间 token 网格。先生成粗尺度网格，再据此给出细尺度网格中各位置的码字分布。同一尺度可以并行采样，尺度之间依次推进。这就是 **Next-scale Prediction（下一尺度预测）**。[VAR 论文 §3.2](https://arxiv.org/abs/2404.02905v2)

<!-- more -->

![VAR 原论文图 2：语言逐 token 生成、图像逐 token 生成与下一尺度预测的比较](/chengYi-xun/img/next-scale-prediction/var-order.png)

*VAR 论文图 2。右侧用逐步清晰的图像示意生成顺序，模型实际预测的是各尺度的离散潜编码。*

| 推理时 | 逐 token 图像自回归 | VAR | 常见的单网格扩散模型 |
| --- | --- | --- | --- |
| 每轮输出 | 下一个位置的码字分布 | 当前尺度各位置的码字分布 | 当前噪声状态下各位置的去噪预测，例如噪声或速度 |
| 串行顺序 | 依次生成空间位置 | 依次生成尺度 | 依次执行去噪迭代 |

VAR 在每轮并行处理同一尺度的位置。常见的单网格扩散模型从噪声出发，每次去噪并行处理整张潜网格。比较延迟时，要同时看串行轮数和每轮开销。网格越大，一轮需要处理的位置就越多。

## 二、从 Next-token 到 Next-scale：概率分解变了什么？

### 2.1 按位置分解

把一张图的 $N$ 个 token 依次记为 $y_1,\ldots,y_N$，把类别或文本等外部条件记为 $c$。逐 token 模型把联合概率分解为每个位置在已知前缀下的条件概率。

$$
p_\theta(y_1,\ldots,y_N\mid c)
=\prod_{i=1}^{N}p_\theta(y_i\mid y_{<i},c).
$$

因果 Transformer 训练时可以用真实的 $y_{<i}$ 并行计算各位置的损失。常规逐 token 采样则要等 $y_i$ 采出后才能开始生成 $y_{i+1}$，因此 $16\times16$ 潜网格需要 256 轮。

### 2.2 按尺度分解

把图像表示为 $R=(r_1,\ldots,r_K)$，其中 $r_k$ 是大小为 $h_k\times w_k$ 的码字索引网格，分辨率从粗到细。每轮在已有尺度 $r_{<k}$ 的条件下生成整张 $r_k$。原始 VAR 的实验用 ImageNet 类别标签作为 $c$。[VAR 论文 §3.2](https://arxiv.org/abs/2404.02905v2)

$$
p_\theta(R\mid c)
=\prod_{k=1}^{K}p_\theta(r_k\mid r_{<k},c).
$$

上式只规定尺度之间的顺序。原始 VAR 的分类头进一步对当前尺度的每个位置 $j$ 给出一个分布，再分别采样。[VAR 推理实现](https://github.com/FoundationVision/VAR/blob/main/models/var.py#L155-L182)

$$
p_\theta(r_k\mid r_{<k},c)
=\prod_{j=1}^{h_kw_k}p_\theta(r_{k,j}\mid r_{<k},c).
$$

这里的独立是**给定粗尺度和条件后的采样独立**。例如，左眼和右眼位置可以通过注意力读取同一个脸部布局，两个位置的分布也可以一起计算。一旦开始采样，左眼刚抽中的码字要等下一尺度才会进入模型的输入。粗尺度留下了多少未建模的关系，会影响这种并行采样能保留多少细节一致性。

## 三、VAR 的关键：多尺度残差量化

VAR 的量化 tokenizer 用多张码字索引网格共同表示一张潜特征图。每个尺度对前面尺度尚未解释的那部分特征做量化。[VAR 论文算法 1、2](https://arxiv.org/abs/2404.02905v2)

### 3.1 下一尺度预测的是尚未解释的特征

编码器把 RGB 图像 $x$ 压缩成连续潜特征 $f$。

$$
f=E(x),\qquad f\in\mathbb{R}^{h_K\times w_K\times C}.
$$

共享码本 $Z\in\mathbb{R}^{V\times C}$ 包含 $V$ 个长度为 $C$ 的码字。$Q$ 为当前网格的每个位置选择最近码字，$Z[r_k]$ 把索引查回码字向量。初始残差为 $\varepsilon_0=f$，累计重建为 $\hat f_0=0$。

$$
\begin{aligned}
r_k&=Q\!\left(\operatorname{Resize}_{h_k,w_k}(\varepsilon_{k-1})\right),\\
\Delta f_k&=\phi_k\!\left(\operatorname{Resize}_{h_K,w_K}(Z[r_k])\right),\\
\hat f_k&=\hat f_{k-1}+\Delta f_k,\\
\varepsilon_k&=\varepsilon_{k-1}-\Delta f_k.
\end{aligned}
$$

码字向量先缩放到最终潜网格，再经过残差卷积变换。论文为每个尺度记作 $\phi_k$，官方 VQ-VAE 默认用 4 个变换供多个尺度共享。最后，解码器将累计特征还原为 RGB 图像。[官方 tokenizer 实现](https://github.com/FoundationVision/VAR/blob/main/models/vqvae.py#L13-L46)

$$
\hat x=D(\hat f_K)=D\!\left(\sum_{k=1}^{K}\Delta f_k\right).
$$

### 3.2 680 个 token，10 轮生成

原始 VAR 的 tokenizer 使用 16 倍空间下采样和 4096 个共享码字。$256\times256$ 图像对应 $16\times16$ 的最终潜网格。[VAR 论文 §4](https://arxiv.org/abs/2404.02905v2)

官方代码为这个分辨率安排了 10 个尺度。[默认尺度配置](https://github.com/FoundationVision/VAR/blob/main/utils/arg_util.py)

```text
尺度边长：  1, 2, 3, 4, 5, 6, 8, 10, 13, 16
token 数：  1, 4, 9, 16, 25, 36, 64, 100, 169, 256
```

每层的位置数相加，得到总长度 $L$。

$$
L=\sum_{k=1}^{10}h_kw_k=680.
$$

同样以 $16\times16$ 为最终网格，单尺度逐 token 生成要采样 256 个位置，分 256 轮。VAR 的多尺度表示有 680 个位置，分 10 轮。两种表示的总 token 数和串行轮数是两组独立的量。

### 3.3 推理时怎样从 1×1 生成到 16×16？

第一轮以类别嵌入构造起始块 $u_1$，采出 $1\times1$ 网格里的一个码字索引 $r_1$。这个码字向量上采样到 $16\times16$，经过 $\phi_1$ 后形成第一份累计特征 $\hat f_1$。类别嵌入也用于 Transformer 各层的自适应层归一化（AdaLN）。[VAR 推理实现](https://github.com/FoundationVision/VAR/blob/main/models/var.py#L147-L186)

第二轮从 $\hat f_1$ 缩放到 $2\times2$，再投影并加位置与尺度编码，得到输入块 $u_2$。Transformer 结合 $u_2$ 和已有的 KV cache，为这 4 个位置各输出 4096 个码字的 logits。采出的 4 个索引组成 $r_2$，查码本、上采样、卷积后得到 $\Delta f_2$，加进 $\hat f_1$。后面的轮次复用同一个过程。

| 轮次 | 当前输入块的来源 | 新采出的索引 | 累计潜特征 |
| --- | --- | --- | --- |
| 1 | 类别起始嵌入 | $1\times1$，共 1 个 | $\hat f_1=\Delta f_1$ |
| 2 | $\hat f_1$ 缩放并投影 | $2\times2$，共 4 个 | $\hat f_2=\hat f_1+\Delta f_2$ |
| 3 | $\hat f_2$ 缩放并投影 | $3\times3$，共 9 个 | $\hat f_3=\hat f_2+\Delta f_3$ |
| 10 | $\hat f_9$ 投影 | $16\times16$，共 256 个 | $\hat f_{10}=\hat f_9+\Delta f_{10}$ |

最终输出来自解码器 $D(\hat f_{10})$。如果想观察每一轮解决了什么，可以把中间的 $\hat f_k$ 单独送进解码器。模型生成下一尺度时用的仍然是累计潜特征和缓存。

## 四、训练与推理：输入和目标怎样错开？

![VAR 原论文图 4：多尺度 tokenizer 与 VAR Transformer 的两阶段训练](/chengYi-xun/img/next-scale-prediction/var-training.png)

*VAR 论文图 4。左侧训练 tokenizer，右侧训练生成器。右图的 `[S]` 用于预测 $r_1$，$e_1$ 和 $e_2$ 是分别用于预测 $r_2$ 和 $r_3$ 的输入特征块。两行 $r_k$ 和 $\hat r_k$ 分别表示监督目标和预测。*

### 4.1 训练数据、监督信号与两个阶段的分工

同一张训练图经过 tokenizer，就能得到所有尺度的码字索引。第二阶段固定 tokenizer，用这些索引监督 VAR Transformer。原论文的两个阶段使用了不同数据集。[VAR 论文 §4](https://arxiv.org/abs/2404.02905v2)

| 阶段 | 论文训练数据 | 更新的模块 | 训练目标 |
| --- | --- | --- | --- |
| 多尺度 VQ tokenizer | OpenImages 图像 | 编码器、码本、残差卷积、解码器和判别器 | 用多尺度离散编码重建图像 |
| VAR Transformer | ImageNet 训练图及类别标签 | Transformer、投影、嵌入和 AdaLN 等生成器参数 | 从类别和真实粗尺度预测下一尺度的码字索引 |

论文使用 AdamW，$(\beta_1,\beta_2)=(0.9,0.95)$，权重衰减为 0.05。以 batch size 256 为基准的学习率为 $10^{-4}$，各模型实际 batch size 为 768 到 1024，训练 200 到 350 个 epoch。[VAR 论文 §4](https://arxiv.org/abs/2404.02905v2)

对单张图，第二阶段的基本损失是所有尺度位置上的交叉熵平均值。

$$
\mathcal L_{\mathrm{AR}}
=-\frac{1}{L}\sum_{k=1}^{K}\sum_{j=1}^{h_kw_k}
\log p_\theta(r_{k,j}\mid r_{<k},c).
$$

每个位置等权时，默认调度下最后一层占 $256/680$ 的损失权重，最粗一层占 $1/680$。如果希望每个尺度等权，可以先在尺度内取平均，再对各尺度的损失取平均。这会改变训练中各尺度的相对权重。[VAR 训练实现](https://github.com/FoundationVision/VAR/blob/main/trainer.py#L39-L45)

### 4.2 第一阶段如何训练 tokenizer？

论文公式（5）用像素、潜特征、感知和对抗损失描述 tokenizer 的训练目标。下面把比较原图 $x$ 和重建图 $\hat x$ 的关系显式写出来。

$$
\mathcal L_{\mathrm{tok}}
=\underbrace{\lVert x-\hat x\rVert_2^2}_{\text{像素重建}}
+\underbrace{\lVert f-\hat f\rVert_2^2}_{\text{量化/潜特征误差}}
+\lambda_P\underbrace{\mathcal L_P(x,\hat x)}_{\text{感知损失}}
+\lambda_G\underbrace{\mathcal L_G(\hat x)}_{\text{对抗损失}}.
$$

官方代码在每个尺度比较累计重建与原潜特征，并将量化损失取平均。码本项更新码字及其重建变换，commitment 项约束编码器的输出。最近邻选码不可导，解码路径另外使用直通估计将重建梯度传回编码器。[VAR 论文公式（5）](https://arxiv.org/abs/2404.02905v2) · [官方量化实现](https://github.com/FoundationVision/VAR/blob/main/models/quant.py#L61-L99)

### 4.3 第二阶段为何可以一次训练全部尺度？

训练时，冻结的 tokenizer 已算出所有真实索引。用其中更粗的索引构造下一尺度的输入，就是这里的 teacher forcing。起始块 $u_1$ 来自类别嵌入，$u_2$ 由真实 $r_1$ 构造，$u_3$ 则由真实 $r_1$ 和 $r_2$ 构造。所有输入块可以提前拼好，一次送入 Transformer。

训练时把下面的偏置加到注意力分数上。其中 $a$ 是查询所在的尺度块，$b$ 是键所在的尺度块。

$$
M_{ab}=\begin{cases}
0,& b\leq a,\\
-\infty,& b>a.
\end{cases}
$$

例如，$2\times2$ 块里的 4 个位置可以彼此注意，因为它们的输入都来自 $r_1$。真实的 $r_2$ 用于本块的交叉熵，从 $u_3$ 开始才参与构造输入，掩码会挡住这些更细的块。**输入与目标错一尺度，加上分块因果掩码**，才使训练能并行预测而不泄漏目标。[VAR 论文图 4](https://arxiv.org/abs/2404.02905v2) · [官方注意力掩码](https://github.com/FoundationVision/VAR/blob/main/models/var.py#L103-L108)

### 4.4 Tokenizer 训练伪代码：学会多尺度离散表示

下面把论文算法 1、算法 2 和复合损失合成一次训练迭代。`vq_loss` 和 `straight_through` 沿用上一节的定义。辅助函数表示对应的算法操作。

```python
# RGB 图像先编码成连续潜特征，起初全部特征都待量化。
features = encoder(images)
residual = features
reconstruction = zeros_like(features)
token_maps = []
loss_vq = 0

for k, shape in enumerate(scale_shapes):  # 从粗到细
    # 将剩余特征缩到当前尺度，为每个位置选最近的码字。
    scaled_residual = resize(residual, shape)
    indices_k, codes_k = quantize(scaled_residual, codebook)
    token_maps.append(indices_k)  # 冻结后用同样的流程生成 VAR 标签

    # 码字向量先放回最终潜网格，再经过该尺度的卷积变换。
    delta_k = phi[k](resize(codes_k, final_latent_shape))
    reconstruction = reconstruction + delta_k
    residual = residual - delta_k  # 剩下的特征留给下一尺度
    loss_vq += vq_loss(features, reconstruction)  # 约束累计重建

# 解码器接收量化后的特征。重建梯度经直通估计回到编码器。
decoder_input = straight_through(reconstruction, features)
reconstructed_images = decoder(decoder_input)

# 像素和感知损失分别比较像素和视觉特征，对抗损失约束真实感。
loss_rec = pixel_loss(images, reconstructed_images)
loss_vq = loss_vq / len(scale_shapes)
loss_perc = perceptual_loss(images, reconstructed_images)
loss_adv = adversarial_generator_loss(reconstructed_images)
loss_gen = loss_rec + loss_vq + lambda_p * loss_perc + lambda_g * loss_adv
update(encoder, codebook, phi, decoder, using=loss_gen)

# 判别器用真实图和重建图单独更新，截断重建图的梯度。
loss_disc = adversarial_discriminator_loss(
    images, stop_gradient(reconstructed_images)
)
update(discriminator, using=loss_disc)
```

### 4.5 VAR 训练伪代码：一次前向覆盖全部尺度

下面以一张图像说明输入和标签如何对齐，省略 batch 维度和条件丢弃。输入构造对应论文 §3.2 和图 4。[官方输入构造](https://github.com/FoundationVision/VAR/blob/main/models/quant.py#L162-L178)

```python
# 冻结的 tokenizer 给出真实索引 [r1, r2, r3, ...]。
# r1 是 1×1 网格，r2 是 2×2 网格，依此类推。
targets = frozen_tokenizer.encode_multiscale(image)

# 最粗尺度从类别条件的起始 embedding 开始。
input_blocks = [start_embedding(condition)]
accumulated = zeros(final_latent_shape)

for prev_idx, next_shape in enumerate(scale_shapes[1:]):
    # 查码本，先上采样到最终潜网格，再卷积得到该尺度的贡献。
    delta = frozen_tokenizer.residual_to_final_grid(
        targets[prev_idx], prev_idx
    )
    accumulated = accumulated + delta

    # 把累积的真实前缀缩放、投影成下一输入块（teacher forcing）。
    # 2×2 输入只用 r1，r2 的贡献从 3×3 输入块起才用到。
    prefix = resize(accumulated, next_shape)
    input_blocks.append(project_to_model_dim(prefix))

# 展平各尺度的空间位置，按从粗到细的顺序拼接并加位置和尺度编码。
inputs = add_position_and_scale(concat_flattened(input_blocks))
target_ids = concat_flattened(targets)  # 标签与各输出位置按同样顺序对齐

# 每个尺度块只看自己和更粗的输入块。类别条件通过 AdaLN 传入。
mask = block_causal_mask(scale_shapes)
logits = transformer(inputs, condition, mask)  # 一次前向覆盖所有尺度

# 2×2 块输出 4 个码字分布，各自与 r2 中对应的索引计算交叉熵。
loss = cross_entropy(logits, target_ids)

# 清空旧梯度，反传并更新 VAR 参数。tokenizer 保持冻结。
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 4.6 推理伪代码：一步生成一个尺度

推理时用模型刚采出的索引更新累计特征，所以要等当前尺度采样完才能构造下一输入块。伪代码省略了 CFG。使用 CFG 时，先组合有条件与无条件的 logits，再采样。[官方采样实现](https://github.com/FoundationVision/VAR/blob/main/models/var.py#L147-L186)

```python
# 先用类别条件构造 1×1 的起始输入，累计潜网格从零开始。
block = start_embedding(condition)
accumulated = zeros(final_latent_shape)
cache = empty_kv_cache()

for k in range(len(scale_shapes)):
    # 加入当前尺度的位置和尺度编码。cache 复用历史尺度的键和值。
    inputs = add_position_and_scale(block, k)
    logits, cache = transformer_step(inputs, condition, cache)

    # 同一尺度的各位置并行采样。例如 2×2 一次采出 4 个索引。
    sampled_ids = sample_categorical(logits, top_k, top_p)
    delta = tokenizer.residual_to_final_grid(sampled_ids, k)
    accumulated = accumulated + delta

    if k + 1 < len(scale_shapes):
        # 用包含本轮新结果的累计特征构造下一尺度的输入。
        next_shape = scale_shapes[k + 1]
        prefix = resize(accumulated, next_shape)
        block = project_to_model_dim(prefix)

# 所有尺度累加完成后，将完整潜特征解码成 RGB 图像。
image = tokenizer.decode(accumulated)
```

## 五、串行步数、推理耗时和实验结果

### 5.1 串行步数、注意力复杂度和延迟是三件事

固定模型宽度和层数，只比较密集自注意力的计算量怎样随最终潜网格边长 $n$ 增长。设单尺度网格包含 $N=n^2$ 个位置。VAR 论文附录 B 用每轮重算完整前缀的算法推导复杂度。[VAR 论文附录 B](https://arxiv.org/abs/2404.02905v2)

逐 token AR 重算完整前缀时，第 $i$ 轮计算 $i$ 个查询和 $i$ 个键。使用 KV cache 后，该轮只需要为一个新查询计算与 $i$ 个键的注意力。

$$
\begin{aligned}
\text{重算前缀}\quad &\sum_{i=1}^{N} O(i^2)=O(n^6),\\
\text{使用 KV cache}\quad &\sum_{i=1}^{N} O(i)=O(n^4).
\end{aligned}
$$

对 VAR，记第 $k$ 个尺度有 $m_k=h_kw_k$ 个位置，前 $k$ 个尺度共有 $S_k=\sum_{i=1}^{k}m_i$ 个位置。使用 KV cache 时，第 $k$ 轮的 $m_k$ 个新查询可以看前面和当前尺度的 $S_k$ 个键。总量满足

$$
\sum_{k=1}^{K} O(m_k S_k)=O(L^2),\qquad L=S_K.
$$

当尺度边长按固定倍率增长时，$L=O(n^2)$，所以这个 VAR 算法也是 $O(n^4)$。这个调度有 $O(\log n)$ 个尺度，每轮重算前缀的 $\sum_k O(S_k^2)$ 同样由最后几轮主导，得到论文的 $O(n^4)$。**在这些假设下，两种方法都用 KV cache 时，密集注意力随 $n$ 增长的阶相同。**

VAR 把顺序依赖从逐位置缩短到逐尺度，给每轮留下更多并行空间。实际耗时还受模型大小、每轮 token 数、KV 读取、采样和解码开销影响。比较延迟时需要报告硬件、batch size、精度、缓存与采样设置。

### 5.2 1.73 FID 对应哪个版本？

下表摘自 VAR v2 的表 1，任务为 **ImageNet $256\times256$ 类别条件生成**。相对时间以表中的 VAR-d30 为 1。[VAR 论文表 1](https://arxiv.org/abs/2404.02905v2)

| 模型 | 参数量 | FID ↓ | IS ↑ | 表中步数 | 相对时间 |
| --- | --- | --- | --- | --- | --- |
| VQGAN | 1.4B | 15.78 | 74.3 | 256 | 24 |
| DiT-XL/2 | 675M | 2.27 | 278.2 | 250 | 45 |
| VAR-d24 | 1.0B | 2.09 | 312.9 | 10 | 0.6 |
| VAR-d30 | 2.0B | 1.92 | 323.1 | 10 | 1 |
| VAR-d30-re | 2.0B | 1.73 | 350.2 | 10 | 1 |

同一个 VAR-d30 在表中有两条记录。未做拒绝采样时，FID 为 1.92。带 **拒绝采样（rejection sampling）** 的 `-re` 记录为 1.73。两条记录的 10 步都按单条轨迹的尺度数计算。表 1 没有列出候选预算、接受率或独立的筛选耗时，无法用时间栏核算每张接受样本的端到端耗时。

表中的相对耗时和质量分数只适用于这组论文对照。评估文本遵循、物理一致性或闭环控制时，还要使用对应任务的指标。

### 5.3 Scaling law 提供的是经验规律

VAR 论文训练了 12 个从 18.5M 到 2.0B 参数的模型，在 50,000 张 ImageNet 验证图上测量交叉熵和 token 错误率。在这组模型上，所有尺度平均交叉熵对参数量的拟合指数为 $-0.20$，平均 token 错误率的指数为 $-0.010$。这些拟合的适用范围取决于 tokenizer、数据和训练预算。换任务以后，应重新测量拟合和下游表现。[VAR 论文 §5.2](https://arxiv.org/abs/2404.02905v2)

## 六、从图像生成到机器人世界模型

比较这几项工作时，可以先找它的预测单位。这个单位可能是空间位置、一个 token 的若干 bit、整张潜网格，也可能是一个视频片段里的残差尺度。

| 工作 | 生成怎样推进 | 对每一步建模的方法 |
| --- | --- | --- |
| VAR（2024） | 空间尺度从粗到细 | 同尺度逐位置分类和采样 |
| MAR（2024） | 随机顺序，每轮补一组位置 | 条件去噪网络采样连续 token |
| BAR（2026） | 外层沿 token 序列推进 | 内层分批填充一个 token 的 bit |
| GRN（2026） | 迭代修正整张 token 网格 | 每轮重预测全图，再随机保留一部分预测 |
| InfinityStar（2025） | 片段间按时间，片段内从粗到细 | 基于文本和可见历史预测视频残差块 |
| WALL-SS（2026） | 依次预测多视角未来片段 | 每个尺度读取匹配的动作条件和有界历史 |

### 6.1 MAR 用扩散头生成连续 token

Li 等人用条件去噪来预测连续 token。以一轮补 4 个位置为例，Transformer 根据已生成位置和待补位置的编码，为每个待补位置输出条件向量 $z_i$。小型去噪 MLP 在 $z_i$ 的条件下分别采出连续向量 $v_i$。训练时给真实 $v_i$ 加噪并回归噪声，这就是 Diffusion Loss。推理时从高斯噪声出发，多步去噪得到 $v_i$。[MAR 论文 §3.2–3.3](https://arxiv.org/abs/2406.11838v3)

**外层自回归决定这一轮补哪些位置，内层扩散决定这些位置上的向量怎么采样。** MAR 在固定网格上按随机顺序逐组解除掩码。若把类似的连续预测头接到下一尺度模型上，需要把外层轮数和内层去噪轮数都计入耗时。[MAR 论文 §4.1–4.2](https://arxiv.org/abs/2406.11838v3)

### 6.2 BAR 把一个离散 token 拆成多轮 bit 预测

Yu 等人的实验把潜空间的名义 bit 预算纳入比较。若有 $N$ 个位置，词表大小为 $V$，固定长度编码至少用 $b=\lceil\log_2 V\rceil$ 个 bit 表示一个 token，总预算 $B=Nb$。论文用有限标量量化（FSQ），通过增减二值通道来调整 $b$。这描述编码占用，实际信息量还受码字使用率和训练约束影响。[BAR 论文 §3.2–3.3](https://arxiv.org/abs/2602.09024v1)

例如，一个 16 bit token 可以表示 65,536 种组合，直接分类需要给每种组合输出一个 logit。BAR 把这一步拆给 **Masked Bit Modeling（MBM）预测头**。训练时随机遮住 token 的一部分 bit，用逐 bit 交叉熵监督。采样时从全部遮住开始，每轮填入一批 bit，后面的轮次能读到前面已经填入的 bit。Transformer 在外层提供 token 间的上下文。[BAR 论文 §3.4、§4.2](https://arxiv.org/abs/2602.09024v1)

增大 bit 预算后，论文在这组 tokenizer 实验中观察到重建质量改善。rFID 用编码后解码的图像衡量重建分布，gFID 用模型从头采样的图像衡量生成分布。要判断增加 bit 后的生成效果，还要在可比的训练和采样预算下比较 gFID。[BAR 论文图 4、图 6](https://arxiv.org/abs/2602.09024v1)

### 6.3 GRN 反复修正整张潜网格

**层级二值量化（HBQ）**先用 $\tanh$ 将每个潜特征标量 $f$ 压到 $(-1,1)$，然后按当前区间的中点逐轮二分。令初始中点 $\hat f_0=0$。第 $m$ 轮用一个 bit 选择半区，再把中点移动到这个半区的中心。[GRN 论文 §3.1](https://arxiv.org/abs/2604.13030v2)

$$
\begin{aligned}
q_m&=\mathbf{1}\{f>\hat f_{m-1}\},\\
\hat f_m&=\hat f_{m-1}+(2q_m-1)2^{-m}.
\end{aligned}
$$

比如 $f=0.2$，前两轮的中点依次为 $0.5$ 和 $0.25$。做完 $M$ 轮后，潜特征的标量量化误差至多为 $2^{-M}$。这个界描述 $\tanh$ 后的潜特征，RGB 重建还取决于编码器和解码器。HBQ 每多一层增加的是同一位置的数值精度。

**全局修正**从随机 token 网格出发。每轮重新预测整张网格，再随机决定各位置采用本轮预测，还是填回随机 token。采用新预测的概率逐轮增加，位置可以重新抽选，所以前几轮填过的内容可以更新或擦除。训练用真实 token 与随机 token 混合成输入，要求模型预测完整的真实网格。自适应采样则在预热后用预测分布的平均熵调节这个概率的增长速度，给高熵样本分配更多迭代。[GRN 论文 §3.2、附录 B](https://arxiv.org/abs/2604.13030v2)

这给出两种处理早期错误的方式。原始 VAR 固定已经采出的粗尺度索引，通过叠加细尺度残差改变重建。GRN 可以直接替换较早轮次填入的 token。

### 6.4 InfinityStar 在视频片段内预测下一尺度

![InfinityStar 原论文图 1：首帧图像金字塔和后续视频片段金字塔组成的时空自回归过程](/chengYi-xun/img/next-scale-prediction/infinitystar-pyramid.png)

*InfinityStar 论文图 1。模型用一个首帧金字塔和多个后续片段金字塔组织视频。下方展示同一套生成顺序如何用于文生图、文生视频、图生视频和视频续写。*

InfinityStar 把 Infinity 的下一尺度框架扩展到视频。首帧的潜时间维度为 1，后续片段有更长的潜时间维度。每个片段对应一组 $(T_u,h_k,w_k)$ 的网格，片段内固定 $T_u$，逐尺度增加空间分辨率。这样首帧可以提供外观，后续片段在自己的时间窗口里补充运动和细节。[InfinityStar 论文 §3.1–3.2](https://arxiv.org/abs/2511.04675v2)

设 $R$ 包含 $U$ 个片段的全部残差编码。$r_{u,k}$ 表示第 $u$ 个片段第 $k$ 轮的编码，$K_u$ 是该片段的轮数，$H_u$ 是从先前片段选入上下文的历史。按时间再按尺度的生成顺序可写为

$$
\begin{aligned}
p_\theta(R\mid c)&=\prod_{u=1}^{U}\prod_{k=1}^{K_u}p_{u,k},\\
p_{u,k}&=p_\theta(r_{u,k}\mid r_{u,<k},H_u,c).
\end{aligned}
$$

这里 $c$ 是文本条件。论文 §3.4 的时空稀疏注意力把跨片段的直接读取限制在**前一片段的最后一个尺度**。更早的信息可以随生成结果间接传下去。论文的长视频交互扩展另用滑动窗口，并额外引入最初画面与前一片段的条件。[InfinityStar 论文 §3.4、§5.1](https://arxiv.org/abs/2511.04675v2)

- **随机量化深度（SQD）**在 tokenizer 训练时让最后若干尺度各自有一定概率被跳过，使较粗的累计特征也要承担重建。论文在图 3 对比了只用早期尺度的重建结果。[InfinityStar 论文 §3.3](https://arxiv.org/abs/2511.04675v2)
- **语义尺度重复（SSR）**让生成器在早期网格尺寸上多做几轮残差预测，再进入更细网格。多轮预测可以共用网格尺寸，每轮仍有新的残差编码。论文单独消融了这一设计。[InfinityStar 论文 §3.4、§4.4](https://arxiv.org/abs/2511.04675v2)

两项设计分别影响粗尺度在重建中承担的信息，以及生成器在粗尺度上分配的计算。连续生成更长视频时，历史覆盖和误差累积还需要单独评估。

### 6.5 WALL-SS 用动作预测多视角未来

WALL-SS 从 InfinityStar 初始化视觉生成器。它用初始多视角观测、任务指令和预定动作来预测同步的多视角未来。比如从同一个抓取前的状态开始，给出闭合夹爪和保持张开两套控制，模型需要生成与各自动作相符的未来。[WALL-SS 论文 §3.1、§6.2](https://arxiv.org/abs/2608.26239v1)

用 $R_u$ 表示待预测片段的残差编码，$K_u$ 表示尺度数。$\mathcal M_u$ 是生成前保留的记忆，$A_u$ 是这一片段的预定控制，$c$ 是任务指令。省略视角与同尺度残差重复的下标后，视觉编码的条件分解为

$$
\begin{aligned}
p_\theta(R_u\mid\mathcal M_u,A_u,c)&=\prod_{k=1}^{K_u}p_{u,k},\\
p_{u,k}&=p_\theta(r_{u,k}\mid r_{u,<k},\mathcal M_u,
A_u^{(\kappa(k))},c).
\end{aligned}
$$

$A_u^{(\kappa(k))}$ 来自将预定控制按相机标定投影到对应的画面，再沿同一时间线做因果编码。$\kappa(k)$ 选取不细于当前视觉尺度的最细可用动作尺度。注意力只直接读取同片段、同视角对应的动作块，跨视角适配器交换同步的因果视觉状态。完整的训练目标先对每个预测块的有效 token 取平均，再对片段、尺度和残差重复等权平均。[WALL-SS 论文 §3.2](https://arxiv.org/abs/2608.26239v1)

1. **时间与尺度记忆。** 模型在生成时记录指定尺度边界上的因果状态，并与同一时段的动作绑定。推理按片段年龄选择已记录的细尺度或粗尺度状态，淘汰超过保留期限的记录，同时一直保留初始观测作为锚点。[WALL-SS 论文 §3.3](https://arxiv.org/abs/2608.26239v1)
2. **尺度级 dream forcing。** 在干净历史之外，训练还使用按片段年龄扰动的历史，以及较早版本模型在流式推理中生成的历史。后一种历史会截断梯度。三类历史都以真实未来为监督，训练模型从不完美的前缀接着预测。[WALL-SS 论文 §3.3、§5.1](https://arxiv.org/abs/2608.26239v1)
3. **视觉生成策略的奖励对齐。** 固定起点、指令和动作，采样一组视觉轨迹，用动作遵循和长程一致性来打分，再做组相对策略优化。KL 约束和真实轨迹回放帮助保留原有视觉分布。这个阶段更新视觉生成器和条件适配器，任务进展和成功率留给闭环评估。[WALL-SS 论文 §3.5、§5.2](https://arxiv.org/abs/2608.26239v1)

闭环评估时，论文把同一个冻结的外部策略分别接到世界模型和真机上，让策略在各自环境里根据当前画面继续选动作，再比较任务结果和策略检查点的排序。[WALL-SS 论文 §6.5](https://arxiv.org/abs/2608.26239v1)

## 七、哪些问题值得单独做对照？

VAR 的空间尺度改变网格分辨率，HBQ 的量化层级改变同一位置的精度，WALL-SS 的记忆调度则按历史片段的年龄选择保留哪一层状态。要看这些设计分别起了什么作用，可以做下面几组对照。

以下是实验建议。每组对照只改变要考察的因素，并记录训练预算、推理时间和质量指标。

| 问题 | 对照怎么做 | 看什么结果 |
| --- | --- | --- |
| 粗尺度里保留了什么？ | 逐步截断真实尺度的累计重建。另做一组实验，替换生成的粗尺度并重新采样后续层 | 截断重建用来看布局和细节何时出现。重采样用来看生成器能否补救粗尺度偏差 |
| 同尺度独立采样会造成什么错误？ | 固定 tokenizer、数据和模型预算，比较一次并行预测与少量尺度内修正 | 结构错误率，以及达到相同质量需要的端到端延迟 |
| 粗尺度学得少，问题出在哪？ | 固定 tokenizer 比较损失权重。另固定生成训练配方，比较有无 SQD 的 tokenizer | 同时看各尺度损失、截断重建和最终生成质量 |
| 世界模型确实使用了动作吗？ | 固定初始观测、指令和采样种子，比较两条可执行且有真实轨迹可查的动作 | 对照预测和真实的位移、接触事件与任务结果 |

我会先做第一组。截断重建能看出粗尺度留下了多少结构，重新采样能看出生成器遇到粗尺度偏差时能补救多少。再测尺度内修正，就能判断新增的计算有没有减少这类错误。

## 参考文献与阅读定位

1. **Tian et al. (2024).** *Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction.* §3、算法 1–2 和图 4 讲模型，表 1、§5.2 和附录 B 分别给出生成结果、规模拟合和复杂度推导。[arXiv:2404.02905v2](https://arxiv.org/abs/2404.02905v2) · [代码](https://github.com/FoundationVision/VAR)
2. **Li et al. (2024).** *Autoregressive Image Generation without Vector Quantization.* §3.2–3.3 和 §4.2 讲 Diffusion Loss、连续 token 和掩码采样。[arXiv:2406.11838v3](https://arxiv.org/abs/2406.11838v3) · [代码](https://github.com/LTH14/mar)
3. **Yu et al. (2026).** *Autoregressive Image Generation with Masked Bit Modeling.* §3.2–3.4 和图 4–6 讲 bit 预算、重建实验和 MBM 预测头。[arXiv:2602.09024v1](https://arxiv.org/abs/2602.09024v1) · [项目页](https://bar-gen.github.io/)
4. **Han et al. (2026).** *Generative Refinement Networks for Visual Synthesis.* §3.1–3.2 和附录 B 讲 HBQ、全局修正和熵引导调度。[arXiv:2604.13030v2](https://arxiv.org/abs/2604.13030v2) · [代码](https://github.com/bytedance/GRN)
5. **Liu et al. (2025).** *InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation.* 图 1 和 §3.2–3.4 讲金字塔及 SQD、SSR，§5.1 讲滑动窗口续写。[arXiv:2511.04675v2](https://arxiv.org/abs/2511.04675v2) · [代码](https://github.com/FoundationVision/InfinityStar)
6. **X Square Robot Team (2026).** *WALL-SS: Scaling Long-horizon World Models via Next-Scale Autoregression.* §3.2–3.5 和 §5 讲动作条件、记忆、训练和推理。[arXiv:2608.26239v1](https://arxiv.org/abs/2608.26239v1) · [项目页](https://github.com/X-Square-Robot/wall-ss)
