---
title: 笔记｜生成模型（十一）：UIT和DiT架构详解
date: 2025-08-11 10:00:00
cover: false
mathjax: true
categories:
 - Notes
tags:
 - Deep learning
 - Diffusion models
 - Generative models
series: Diffusion Models theory
---
>
> ⬅️ 上一篇：[笔记｜生成模型（十）：Classifier-Free Guidance 理论与实现](/chengYi-xun/posts/10-classifier-free-guidance-for-diffusion-models/)
>
> ➡️ 下一篇：[笔记｜生成模型（十二）：Normalizing Flow理论与实现](/chengYi-xun/posts/12-Normalizing-flow/)

# UIT

> 论文链接：*[All are Worth Words: A ViT Backbone for Diffusion Models](https://arxiv.org/abs/2209.12152)*
>
> 官方实现：**[baofff/U-ViT](https://github.com/baofff/U-ViT)**

扩散模型自从被提出后，主干网络一直都是各种基于卷积的 UNet 的变体。而在其他领域，Transformer 架构则更加流行，尤其是由于 Transformer 多模态性能和缩放能力都很强，因此把 Transformer 架构用于扩散模型是很值得尝试的。这篇 U-ViT 的工作就是一个不错的尝试。

# U-ViT 的设计

在开始具体的介绍之前，可以先看一下 U-ViT 整体的架构。可以看出其有几个主要的特点：

1. 所有的元素，包括 latent、timestep、condition 等都以 token 的形式进行了 embedding；
2. 类似于 UNet，在不同的 Transformer Block 层之间添加了长跳跃连接。

虽然理论上来说这两个点都比较简单，但作者进行了一系列实验来选择比较好的设计。

![uvit-framework](/chengYi-xun/img/uvit-framework.jpg)

## 长跳跃连接的实现

将主分支和长跳跃连接分支的特征分别记为 $h_m$ 和 $h_s$。作者选取了几种不同的实现方式进行实验：

1. 将两个特征拼接起来然后用一个线性层做 projection：$\mathrm{Linear}(\mathrm{Concat}(h_m,h_s))$；
2. 两者直接相加：$h_m+h_s$；
3. 对长跳跃连接分支做 projection 再相加：$h_m+\mathrm{Linear}(h_s)$；
4. 先相加再做 projection：$\mathrm{Linear}(h_m+h_s)$；
5. 直接去掉长跳跃连接（这个相当于对照组）。

因为 Transformer block 中本来就有短跳跃连接，所以 $h_m$ 本身就含有一部分 $h_s$ 的信息，直接将两者相加意义不大。最后经过实验（可以看下方图里的 (a)）发现第一种设计的效果最好。

## Time Embedding 的实现

时间步的嵌入可以用 ViT 的风格也可以用 UNet 的风格，具体来说是：

1. 把 time embedding 当成一个 token 输入；
2. 在 transformer block 的 layer normalization 的位置嵌入，也就是使用 AdaLN：$\mathrm{AdaLN}(h,y)=y_s\mathrm{LayerNorm}(h)+y_b$，其中 $y_s$ 和 $y_b$ 是两个 time embedding 的 projection。这个相当于用 time embedding 对 layer normalization 的结果进行 affine。

最后发现第一种方法更佳有效，如下图中的 (b) 所示。

## 在 Transformer 后使用卷积的方式

作者也尝试了三种方法：

1. 直接在 linear projection 后使用 3x3 卷积，把 token 转换为 image patch；
2. 在 projection 前先把 token embedding 转换为二维，卷积后再 projection；
3. 直接不加入额外的卷积层。

最后发现第一种方式效果最好，如下图中的 (c) 所示。

## Patch Embedding 的实现

有两种方法：

1. 直接用 linear projection 进行 embedding；
2. 用一系列 3x3 卷积+1x1 卷积进行 embedding。

最后发现第一种方法的效果更好，如下图中的 (d)。

## Position Embedding 的实现

1. 和 ViT 的 setting 相同，使用一维的可学习向量；
2. 使用 NLP 领域常用的 sinusoidal position embedding 的二维形式。

经过实验发现前者效果更好，如下图的 (e) 所示。

![消融实验的结果](/chengYi-xun/img/uvit-design-ablation.jpg)

总而言之作者通过一系列实验确定了一个比较好的 U-ViT 的设计。

## 讨论：U-ViT 的缩放能力

作者尝试了更深/更宽的模型架构，以及更大的 patch，最后发现性能随着深度和宽度的增加，并不是单调上升的，最佳的效果都在中等宽度/中等深度的网络中取得。

除此之外，作者发现最小的 patch 能够取得最佳的结果，这可能是因为预测噪声是比较 low-level 的任务，所以更小的 patch 更合适。由于对于高分辨率图像使用小 patch 比较消耗资源，所以也需要先将图像转换到低维度的隐空间中再进行建模。

# 总结

除了 U-ViT 之外，同期的 DiT 也把 transformer 架构引入到了扩散模型中，虽然感觉作者的实验思路非常简单粗暴，但最后的效果还是不错的。从后续的工作也可以看出，这类 transformer 架构的方法在某些任务上（例如视频生成）有取代 UNet 的趋势。

# DIT

> 论文链接：*[Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)*
>
> 官方实现：**[facebookresearch/DiT](https://github.com/facebookresearch/DiT)**

Transformer 在许多领域都有很不错的表现，尤其是近期大语言模型的成功证明了 scaling law 在 NLP 领域的效果。Diffusion Transformer（DiT）把 transformer 架构引入了扩散模型中，并且试图用同样的 scaling 方法提升扩散模型的效果。DiT 提出后就受到了很多后续工作的 follow，例如比较有名的视频生成方法 sora 就采取了 DiT 作为扩散模型的架构。

# Diffusion Transformer

在正式开始介绍 DiT 之前，需要先了解一下 DiT 使用的扩散模型架构。DiT 使用的是 latent diffusion，VAE 采用和 Stable Diffusion 相同的 KL-f8，并且使用了 Improved DDPM（详细介绍见[这个链接](https://littlenyima.github.io/posts/15-improved-denoising-diffusion-probabilistic-models/)），同时预测噪声的均值和方差。

![DiT 的架构](/chengYi-xun/img/dit-framework.jpg)

## Patchify

由于 DiT 使用了 latent diffusion，对于 $256\times256\times3$ 的输入图像，首先使用 VAE 转换为 $32\times32\times4$ 的 latent，在 latent 输入到 DiT 之前，首先需要将其转换为 token，也就是 patchify。DiT 的 patchify 方式和 ViT 基本上相同，也就是将图像变成 $(I/p)^2$ 个 patch，其中 $I$ 是 latent 的 spatial size，$p$ 是每个 patch 的大小，这里 $p$ 的取值使用了 2、4、8 几个取值。在转换为 patch 后，还需要给每个 patch 加上位置编码，这里的位置编码使用的是二维的不可学习 sin-cos 位置编码。

## DiT Block Design

DiT 的 block 大部分结构都可以直接沿用 ViT 的结构，不过和 ViT 不同的是，DiT 除了需要接收 token 的输入，还需要将 time embedding 和生成条件也进行嵌入。为此，作者设计了四种方式：

1. In context conditioning：用类似 ViT 的方法，把 time embedding 和 condition 作为额外的 token 输入到 DiT 中，并且在最后把这两部分 token 去掉。这个做法和 U-ViT 是一样的，不过会增加一部分额外的计算开销；
2. Cross-attention block：使用和原版 Transformer 类似的设计，在 self-attention 层后方再加一个 cross-attention 层，把 time embedding 和 condition 作为一个长度为 2 的序列输入，latent 作为 query，条件作为 key 和 value，这种方式引入的额外开销比较大；
3. Adaptive layer norm：将 transformer block 中的 LayerNorm 换成 AdaLayerNorm，用 time embedding 和 condition embedding 回归 shift 和 scale 参数，用来对 latent 进行 affine；
4. AdaLN-Zero block：根据一些现有的工作的经验，把残差连接前的最后一个卷积初始化为 0 对训练比较有利，因此这里将 AdaLN 中的线性层（对应上图中的 $\gamma$ 和 $\beta$）进行 0 初始化，并且在最后加入另一个可学习的 scale（对应上图的 $\alpha$），也初始化为 0。

经过实验发现 AdaLN-Zero 的设计是最好的，因此 DiT 最终采用的是这种设计方案。

## 其他

和 ViT 相同，通过修改 transformer block 的数量、隐变量的维度，以及 attention head 的数量，可以得到不同大小的 DiT（DiT-S、DiT-B、DiT-L 和 DiT-XL），这些模型的 flop 从 0.3 GFLOPs 到 118.6 GFLOPs 不等。

对于 decoder，DiT 使用一个 LayerNorm（或 AdaLN）以及一个线性层对输出进行编码。最后得到的输出的大小为 $p\times p\times 2C$，通道数为 $2C$ 是因为包括了均值和方差。

## 讨论

作者进行了一系列实验来研究不同 DiT 设计之间的区别，主要包括以下几个方面：

1. DiT block 的设计：经过实验可以发现 AdaLN 的效果比其他的条件嵌入方式更好，并且初始化方式也很重要，AdaLN 将每个 DiT block 初始化为恒等映射，能取得更好的效果；不过对于比较复杂的条件，比如 text，可能用 cross-attention 更好；
2. 缩放模型大小和 patch 大小：实验发现增大模型大小并减小 patch 大小可以提高性能；
3. 提高 GFLOPs 是改善模型性能的关键；
4. 更大的 DiT 模型的计算效率更高；

# DiT 的代码实现

主要需要关注的是模型的 DiT block 和 decoder，根据[官方实现](https://github.com/facebookresearch/DiT/blob/main/models.py)：

```python
def modulate(x, shift, scale):
    """
    自适应调制函数：对输入进行缩放和平移变换
    Args:
        x: 输入特征 [batch, seq_len, dim]
        shift: 平移参数 [batch, dim]
        scale: 缩放参数 [batch, dim]
    Returns:
        调制后的特征
    """
    # 先进行缩放变换 (1 + scale)，再进行平移变换 + shift
    # unsqueeze(1) 将 [batch, dim] 扩展为 [batch, 1, dim] 以匹配x的维度
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    DiT (Diffusion Transformer) 块，使用自适应层归一化零初始化 (adaLN-Zero) 条件化
    这是Transformer架构的一个变体，专门为扩散模型设计
    """
    
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        
        # 第一个层归一化：用于多头自注意力之前，不使用可学习的仿射参数
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # 多头自注意力模块
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        
        # 第二个层归一化：用于MLP之前，不使用可学习的仿射参数
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # 计算MLP的隐藏层维度（通常是输入维度的4倍）
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        
        # 定义近似GELU激活函数（使用tanh近似以提高计算效率）
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        
        # MLP模块：两层全连接网络，中间使用GELU激活，无dropout
        self.mlp = Mlp(in_features=hidden_size, 
                      hidden_features=mlp_hidden_dim, 
                      act_layer=approx_gelu, 
                      drop=0)
        
        # 自适应层归一化调制网络
        # 输出6个参数：注意力的shift/scale/gate + MLP的shift/scale/gate
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),  # Swish激活函数 (x * sigmoid(x))
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)  # 线性层输出6组参数
        )

    def forward(self, x, c):
        """
        前向传播
        Args:
            x: 输入特征 [batch, seq_len, hidden_size]
            c: 条件信息 [batch, hidden_size]（如时间嵌入、类别嵌入等）
        """
        
        # 通过调制网络生成6组参数，每组大小为 [batch, hidden_size]
        # chunk(6, dim=1) 将 [batch, 6*hidden_size] 分割为6个 [batch, hidden_size] 的张量
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        # 第一个残差连接：多头自注意力分支
        # 1. 对x进行层归一化
        # 2. 使用shift_msa和scale_msa进行自适应调制
        # 3. 通过自注意力模块
        # 4. 使用gate_msa进行门控（控制信息流量）
        # 5. 添加残差连接
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        
        # 第二个残差连接：MLP分支
        # 1. 对x进行层归一化
        # 2. 使用shift_mlp和scale_mlp进行自适应调制
        # 3. 通过MLP模块
        # 4. 使用gate_mlp进行门控
        # 5. 添加残差连接
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x
```

可以看到所有的 affine 参数都是由同一个 MLP 学习的，推理时被 6 等分，然后对 latent 进行 affine。

Decoder 则是由 AdaLN + Linear 组成：

```python
class FinalLayer(nn.Module):
    """
    DiT (Diffusion Transformer) 的最终输出层
    将隐藏特征转换为最终的输出patch，通常用于重建图像或预测噪声
    """
    
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        
        # 最终层归一化：不使用可学习的仿射参数，与DiTBlock保持一致
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # 线性投影层：将隐藏特征映射到输出patch
        # 输出维度 = patch_size² × out_channels (每个patch的像素数 × 通道数)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        
        # 自适应层归一化调制网络（简化版，只有shift和scale参数）
        # 输出2组参数：shift 和 scale，用于条件化最终的层归一化
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),  # Swish激活函数 (x * sigmoid(x))
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)  # 输出2倍hidden_size的特征
        )

    def forward(self, x, c):
        """
        前向传播
        Args:
            x: 输入特征 [batch, num_patches, hidden_size] - 来自DiT块的输出
            c: 条件信息 [batch, hidden_size] - 时间嵌入、类别嵌入等条件
        Returns:
            x: 输出特征 [batch, num_patches, patch_size² × out_channels]
        """
        
        # 通过调制网络生成shift和scale参数
        # chunk(2, dim=1) 将 [batch, 2*hidden_size] 分割为2个 [batch, hidden_size] 的张量
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        
        # 对输入进行层归一化，然后使用条件信息进行自适应调制
        # modulate函数：x * (1 + scale) + shift
        x = modulate(self.norm_final(x), shift, scale)
        
        # 通过线性层将特征映射到最终输出维度
        # [batch, num_patches, hidden_size] -> [batch, num_patches, patch_size² × out_channels]
        x = self.linear(x)
        
        return x
```

这些 `adaLN_modulation` 层在创建时被零初始化：

```python
# 零初始化 DiT 块中的 adaLN 调制层：
for block in self.blocks:
    # 将每个 DiT 块的 adaLN 调制网络的最后一层（线性层）权重初始化为 0
    # adaLN_modulation[-1] 是 Sequential 中的最后一个模块（nn.Linear）
    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
    
    # 将每个 DiT 块的 adaLN 调制网络的最后一层偏置初始化为 0
    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

# 零初始化输出层：
# 将最终层的 adaLN 调制网络的线性层权重初始化为 0
nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)

# 将最终层的 adaLN 调制网络的线性层偏置初始化为 0
nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

# 将最终层的输出线性层权重初始化为 0
nn.init.constant_(self.final_layer.linear.weight, 0)

# 将最终层的输出线性层偏置初始化为 0
nn.init.constant_(self.final_layer.linear.bias, 0)
```

adaLN 调制层零初始化的作用：

```python
# 当权重和偏置都为0时：
# adaLN_modulation 输出 = SiLU(0) + Linear(input, weight=0, bias=0) = 0
# 这意味着：
shift_msa = scale_msa = gate_msa = 0  # 对于DiT块
shift_mlp = scale_mlp = gate_mlp = 0  # 对于DiT块
shift = scale = 0                     # 对于最终层

```

# 总结

个人感觉 DiT 相比于 U-ViT 是更成功的，因为我们使用 transformer 架构比较注重的就是缩放能力，而 DiT 的实验表明，其性能能够随着模型的缩放（模型规模/模型计算量）而受益。


> 参考资料：
>
> 1. Bao, F., Nie, S., Xue, J., Cao, Y., Li, C., Su, H., & Zhu, J. (2023). *All are Worth Words: A ViT Backbone for Diffusion Models*. CVPR 2023.
> 2. Peebles, W., & Xie, S. (2023). *Scalable Diffusion Models with Transformers*. ICCV 2023.
> 3. [UIT](https://littlenyima.github.io/posts/29-uvit-a-vit-backbone-for-diffusion-models/)
> 4. [DIT](https://littlenyima.github.io/posts/32-dit-scalable-diffusion-with-transformers/)

> 下一篇：[笔记｜生成模型（十三）：流模型 (Normalizing Flow)](/chengYi-xun/posts/12-Normalizing-flow/)