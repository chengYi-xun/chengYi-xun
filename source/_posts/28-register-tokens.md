---
title: 笔记｜Vision Transformers Need Registers：用 Register Tokens 治愈 ViT 的"注意力伪影"
date: 2026-04-05 23:50:00
categories:
 - Tutorials
tags:
 - Vision Transformer
 - Register Tokens
 - DINOv2
 - Self-Supervised Learning
 - Computer Vision
series: "杂谈"
---

> **论文**：*Vision Transformers Need Registers*（Darcet et al., ICLR 2024 Outstanding Paper）
> **代码**：[facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) · [HuggingFace DINOv2-with-Registers](https://huggingface.co/docs/transformers/model_doc/dinov2_with_registers)
> **参考资料**：本文结合论文原文与 DINOv2 开源代码进行讲解。

---

## 0. 从一张"坏掉的注意力图"说起

假设你用 DINOv2-g（ViT-giant）提取一张猫咪图片的特征，然后可视化最后一层 self-attention 中 `[CLS]` token 对所有 patch token 的注意力权重。你期望看到的结果是：注意力集中在猫咪身上，背景区域的权重较低——这正是初代 DINO 擅长的事情，漂亮的注意力图甚至可以直接当作无监督分割结果。

但实际出来的图，却在墙壁、天空等均匀背景区域出现了几个异常的"亮点"：

| 位置 | 期望注意力 | 实际注意力 |
|------|-----------|-----------|
| 猫的头部 | 高 | 高 |
| 猫的身体 | 中 | 中 |
| 纯色墙壁 patch $(i,j)=(3,12)$ | 低 | **异常高** |
| 纯色天花板 patch $(i,j)=(1,5)$ | 低 | **异常高** |
| 地板花纹 | 低 | 低 |

更奇怪的是，这些异常 patch 的 **输出特征 L2 范数** 比正常 patch 高出约 **10 倍**（正常 patch 范数 ≈ 50-100，异常 patch ≈ 500+）。

这就是本文要解决的核心问题——**Artifact Tokens（伪影 Token）**。

---

## 1. 问题分析：谁在制造伪影？

### 1.1 现象的量化

论文对 DINOv2-g 进行了系统性分析，定义了一个简单的判别准则：**输出特征 L2 范数 > 150 的 patch token 即为 outlier**。在 ImageNet 数据集上统计，这类 outlier 约占总 patch 数的 **2.37%**。

范数的分布呈现明显的**双峰结构**：

$$
p(\|z_i\|_2) \approx \underbrace{p_{\text{normal}}(\|z_i\|_2)}_{\text{主峰, } \|z_i\|_2 \in [0, 100]} + \underbrace{p_{\text{outlier}}(\|z_i\|_2)}_{\text{副峰, } \|z_i\|_2 \in [400, 600]}
$$

这个现象并不局限于 DINOv2——DeiT-III（有监督）、OpenCLIP（文本-图像对齐监督）同样存在。**唯独初代 DINO 不存在**。

### 1.2 三个关键实验

论文通过三组实验揭示了 outlier token 的本质：

**实验 1：位置信息探测（Position Probing）**

在 outlier 和正常 patch 上分别训练线性分类器预测自身位置：

| Token 类型 | 位置预测准确率（Top-1） |
|-----------|----------------------|
| 正常 patch | 41.7% |
| Outlier patch | 22.8% |

Outlier token **丢失了自身的位置信息**。虽然位置编码在输入时就注入了，但在经过网络后，这些 token 不再"记得"自己在图像中的位置。

**实验 2：像素重建探测（Pixel Reconstruction）**

训练线性模型从 patch embedding 重建原始像素值：

| Token 类型 | 重建 L2 误差（↓更好） |
|-----------|---------------------|
| 正常 patch | 18.38 |
| Outlier patch | 25.23 |

Outlier token **丢失了局部像素信息**。

**实验 3：图像分类探测（Image Classification）**

随机选取一个 outlier 或正常 patch 作为**整张图片的表征**，训练线性分类器：

| Token 类型 | ImageNet Top-1 | Aircraft | Cars | CUB |
|-----------|---------------|----------|------|-----|
| 正常 patch | 65.8% | 17.1% | 10.8% | 18.6% |
| Outlier patch | **69.0%** | **79.1%** | **85.2%** | **84.9%** |
| [CLS] token | 86.0% | 87.3% | 91.5% | 91.3% |

Outlier token 的分类精度**远高于**正常 patch，接近 [CLS] token 的水平。

### 1.3 浮现的解释

三组实验指向同一个结论：

> **模型在训练过程中，学会了识别"信息冗余"的 patch（如均匀背景），把它们劫持成全局信息的容器——丢弃局部内容，转而存储全局语义。**

从信息论的角度可以这样理解。设 $z_i^{(l)}$ 为第 $l$ 层第 $i$ 个 token 的表征。Self-attention 是一个全局信息交换机制：

$$
z_i^{(l+1)} = z_i^{(l)} + \text{Attn}(Q_i, K, V)
$$

其中 $\text{Attn}(Q_i, K, V) = \sum_{j} \alpha_{ij} V_j$ 聚合了所有 token 的信息。当模型需要在某处"暂存"全局统计量（比如"这张图整体是室内场景"、"光照偏暖"）时，它没有专用的存储位置。于是模型学到了一个 hack——选择那些与邻居高度相似（余弦相似度高）、丢失后对重建损失影响最小的 patch，把它们变成"全局寄存器"。

论文验证了这一点：outlier patch 与其 4-邻域的**输入层余弦相似度**显著高于正常 patch，说明它们确实位于"信息冗余"区域。

### 1.4 伪影什么时候出现？

论文进一步刻画了出现条件：

| 条件 | 细节 |
|------|------|
| 网络层数 | 在 40 层 ViT-g 的约**第 15 层**开始分化 |
| 训练进度 | 训练到约 **1/3** 时开始出现 |
| 模型规模 | ViT-Tiny/Small/Base 不出现，**Large 及以上**才出现 |

这说明伪影是大模型在充分训练后的**涌现行为**（emergent behavior），小模型的容量不足以"奢侈地浪费" patch 来存储全局信息。

---

## 2. 解决方案：Register Tokens

### 2.1 核心思想

既然模型缺少存储全局信息的专用位置，那就**显式地提供**：在输入序列中追加 $N$ 个可学习的 token（称为 register token），让模型自由地在这些 token 上存储和处理全局信息，而不必劫持 patch token。

修改后的 token 序列变为：

$$
\mathbf{x} = [\underbrace{[\text{CLS}]}_{\text{类别 token}},\; \underbrace{[\text{REG}_1], \ldots, [\text{REG}_N]}_{\text{register tokens}},\; \underbrace{p_1, p_2, \ldots, p_M}_{\text{patch tokens}}]
$$

**关键设计细节**：

1. **Register token 不携带位置编码**——它们不对应图像中的任何空间位置
2. **输出时丢弃 register token**——只保留 [CLS] 和 patch token 作为最终表征
3. **参数初始化**与 [CLS] token 相同：$\text{init} \sim \mathcal{N}(0, 10^{-6})$

### 2.2 DINOv2 中的代码实现

在 DINOv2 的 `DinoVisionTransformer` 类中，register token 的实现简洁而优雅：

**参数定义**：

```python
# facebookresearch/dinov2 - vision_transformer.py
class DinoVisionTransformer(nn.Module):
    def __init__(self, ..., num_register_tokens=0, ...):
        ...
        self.num_register_tokens = num_register_tokens
        
        # [CLS] token 和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Register tokens: shape [1, N, D]
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens else None
        )
    
    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
```

**前向传播中的拼接**：

```python
def prepare_tokens_with_masks(self, x, masks=None):
    B, nc, w, h = x.shape
    x = self.patch_embed(x)                          # [B, M, D]
    
    # 拼接 [CLS] token 并加入位置编码
    x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)  # [B, 1+M, D]
    x = x + self.interpolate_pos_encoding(x, w, h)   # 位置编码只覆盖 [CLS] + patch
    
    # 在 [CLS] 和 patch 之间插入 register tokens
    if self.register_tokens is not None:
        x = torch.cat(
            (
                x[:, :1],                                          # [CLS]
                self.register_tokens.expand(B, -1, -1),            # [REG_1, ..., REG_N]
                x[:, 1:],                                          # patch tokens
            ),
            dim=1,
        )  # 最终: [B, 1 + N + M, D]
    
    return x
```

**输出时分离**：

```python
def forward_features(self, x, masks=None):
    x = self.prepare_tokens_with_masks(x, masks)
    
    for blk in self.blocks:
        x = blk(x)
    
    x_norm = self.norm(x)
    return {
        "x_norm_clstoken": x_norm[:, 0],
        "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
        "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
    }
```

整个改动只涉及三处：**定义参数**、**输入时拼接**、**输出时切片**——对已有架构的侵入性极低。

### 2.3 为什么 Register Token 不使用位置编码？

这是一个值得思考的设计选择。Register token 的职责是存储**全局**信息，而位置编码绑定的是**空间**信息。如果给 register token 添加位置编码，相当于人为地将它与某个空间位置关联，这会：

1. **与其职责矛盾**——全局容器不应偏向特定位置
2. **限制模型灵活性**——模型可能被迫在特定 register 中存储特定区域的信息

在 DINOv2 代码中，位置编码 `self.pos_embed` 的形状是 `[1, 1 + M, D]`（1 个 [CLS] + M 个 patch），在 `prepare_tokens_with_masks` 中先加位置编码再插入 register token，自然地实现了"register 无位置编码"。

VGGT 中更显式地处理了这一点：

```python
# VGGT - aggregator.py
if self.patch_start_idx > 0:
    # register 和 camera token 的位置编码设为 0
    pos = pos + 1
    pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(device)
    pos = torch.cat([pos_special, pos], dim=1)
```

---

## 3. 实验验证

### 3.1 伪影是否消失了？

论文在三种训练范式下验证——DeiT-III（有监督）、OpenCLIP（文本监督）、DINOv2（自监督）。加入 4 个 register token 后，所有模型的 **输出特征范数分布变为单峰**，高范数 outlier 完全消失。

### 3.2 性能是否退化？

这是最关键的问题。加入 register token 增加了序列长度（从 $1 + M$ 变为 $1 + N + M$），会不会带来性能损失？

| 模型 | ImageNet（Top-1） | ADE20k（mIoU） | NYUd（rmse↓） |
|------|-----------------|---------------|---------------|
| DeiT-III | 84.7 | 38.9 | 0.511 |
| DeiT-III + reg | 84.7 | 39.1 | 0.512 |
| OpenCLIP | 78.2 | 26.6 | 0.702 |
| OpenCLIP + reg | 78.1 | 26.7 | **0.661** |
| DINOv2 | 84.3 | 46.6 | 0.378 |
| DINOv2 + reg | **84.8** | **47.9** | **0.366** |

结论：**没有退化，反而在密集预测任务上有显著提升**（DINOv2 的 ADE20k mIoU 从 46.6 → 47.9，NYUd rmse 从 0.378 → 0.366）。

### 3.3 需要多少个 Register Token？

论文用 DINOv2 ViT-L 进行了 $N \in \{0, 1, 2, 4, 8, 16\}$ 的消融实验：

| $N$ | 伪影消除 | ImageNet Top-1 | ADE20k mIoU | NYUd rmse↓ |
|-----|---------|---------------|-------------|------------|
| 0 | × | 84.4 | ~66.0 | ~2.85 |
| 1 | ✓ | 84.5 | ~66.5 | ~2.76 |
| 2 | ✓ | 84.6 | ~66.6 | ~2.74 |
| **4** | **✓** | **84.7** | **~66.7** | **~2.73** |
| 8 | ✓ | 84.7 | ~66.5 | ~2.76 |
| 16 | ✓ | 84.8 | ~66.2 | ~2.79 |

两个发现：

1. **1 个 register 就足以消除伪影**，但性能提升需要更多
2. **密集预测任务存在最优点**（约 4 个），过多反而有微小退化；**分类任务随数量单调提升**
3. 论文最终选择 $N = 4$ 作为默认配置

计算开销方面，$N = 4$ 时 FLOPs 增加不到 2%，参数量增加可忽略。

### 3.4 无监督目标发现

Register token 的一个重要下游收益体现在无监督目标发现（如 LOST 算法）上：

| 模型 | VOC 2007 (corloc) | VOC 2012 | COCO 20k |
|------|-------------------|----------|----------|
| DINOv2 | 35.3 | 40.2 | 26.9 |
| DINOv2 + reg | **55.4** | **60.0** | **42.0** |
| DeiT-III | 11.7 | 13.1 | 10.7 |
| DeiT-III + reg | **27.1** | **32.7** | **25.1** |

DINOv2 + reg 在 VOC 2007 上提升了 **+20.1 corloc**，这是因为 LOST 依赖于干净的特征图来定位物体，而 outlier token 会严重干扰这一过程。

### 3.5 Register Token 学到了什么？

论文可视化了各 register token 的注意力模式，发现了一个有趣的现象：**不同 register 自发地出现了分工**。

- 某些 register 关注图像中心区域（通常是物体所在位置）
- 某些 register 关注边缘区域（通常是背景）
- 它们的注意力模式与 [CLS] token 类似——支撑区域很大，但各自有所侧重

这种**自发分工**（emergent specialization）从未被显式要求，完全由端到端训练涌现。它类似于 Slot Attention 中 slot 对不同物体的自发绑定。

---

## 4. 更深的理解：为什么 DINO 不需要 Register？

这可能是论文中最引人深思的问题。所有现代 ViT——DeiT-III、OpenCLIP、DINOv2——都出现了伪影，**唯独** DINO 没有。为什么？

论文给出了一些线索但没有完整回答。结合后续研究，一种可能的解释是：

1. **DINO 的模型规模较小**（论文验证的 DINO 是 ViT-B/16），而伪影只在 ViT-L 及以上才出现
2. **DINO 的训练时间较短**，伪影需要充分训练才涌现（约训练 1/3 后）
3. **MAE（Masked Autoencoder）也不出现伪影**——因为它使用**局部重建损失**，不需要模型聚合全局信息

这暗示了一个统一解释：**伪影的根源是模型在训练目标中需要聚合全局信息，但缺少专用容器。** DINO 的 ViT-B 规模太小、训练不够长，模型还没学会这个"hack"。而 MAE 的训练目标本身就是局部的（重建被遮盖的 patch），根本不需要全局聚合。

---

## 5. 后续发展与影响

Register Token 自提出以来（2023 年 9 月）影响深远，被多项重要工作采纳，也催生了多条后续研究线：

### 5.1 直接采纳

- **DINOv2 官方模型**：Meta 发布了带 register 的 DINOv2 预训练权重，成为视觉基础模型的标配
- **VGGT**（Wang et al., 2025）：在 3D 视觉重建 Transformer 中使用 register token 存储全局几何信息
- **Mamba-Reg**（CVPR 2025）：将 register token 扩展到 Vision Mamba 架构，在输入序列中均匀插入 register，ViT-B 级别的精度从 81.8% → 83.0%

### 5.2 训练无关方案

*Vision Transformers Don't Need Trained Registers*（2025）提出了一种免训练替代方案：通过识别负责产生高范数激活的"register neuron"，在推理时将 outlier 激活转移到额外的未训练 token 上。这使得**已经训练好的 ViT 也能受益**，无需重新训练。

### 5.3 质疑与反思

*Do All Vision Transformers Need Registers?*（2025）对原论文的结论进行了跨架构验证，发现：
- 原论文的核心发现在 DINOv2 上高度可复现
- 但某些结论（如 outlier 始终出现在低信息区域）并不能普遍推广到所有架构
- 对于某些文本监督模型（如 OpenCLIP），register 的收益并不稳定

*Vision Transformers Need More Than Registers*（2025）则认为伪影的深层原因是"lazy aggregation behavior"——模型倾向于就近聚合信息而非均匀利用所有 token——单靠 register token 不能完全解决。

---

## 6. 从工程角度看：如何使用 Register Token

### 6.1 使用预训练的 DINOv2-with-Registers

通过 HuggingFace 可以直接加载带 register 的模型：

```python
from transformers import AutoModel, AutoImageProcessor

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large-with-registers")
model = AutoModel.from_pretrained("facebook/dinov2-large-with-registers")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# outputs.last_hidden_state 的 token 顺序：
# [CLS] [REG_1] [REG_2] [REG_3] [REG_4] [patch_1] ... [patch_M]

cls_token = outputs.last_hidden_state[:, 0]        # 全局表征
patch_tokens = outputs.last_hidden_state[:, 5:]    # 空间特征（跳过 1 CLS + 4 REG）
```

### 6.2 在自定义 ViT 中添加 Register Token

如果你在训练自己的 ViT，添加 register token 只需三步：

```python
class MyViTWithRegisters(nn.Module):
    def __init__(self, embed_dim, num_patches, num_registers=4):
        super().__init__()
        self.num_registers = num_registers
        
        # 步骤 1：定义参数
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_registers, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        
        nn.init.normal_(self.register_tokens, std=1e-6)
    
    def forward(self, patch_tokens):
        B = patch_tokens.shape[0]
        
        # 步骤 2：输入时拼接（位置编码不覆盖 register）
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patch_tokens], dim=1)
        x = x + self.pos_embed
        
        reg = self.register_tokens.expand(B, -1, -1)
        x = torch.cat([x[:, :1], reg, x[:, 1:]], dim=1)
        
        # ... transformer blocks ...
        
        # 步骤 3：输出时分离
        cls_out = x[:, 0]
        patch_out = x[:, 1 + self.num_registers:]
        return cls_out, patch_out  # 丢弃 register token
```

---

## 7. 总结

| 维度 | 要点 |
|------|------|
| **问题** | 大规模 ViT 在低信息 patch 上出现高范数 outlier token（伪影） |
| **原因** | 模型缺少存储全局信息的专用位置，劫持冗余 patch 作为"寄存器" |
| **方案** | 显式添加 $N$ 个可学习 register token，不携带位置编码，输出时丢弃 |
| **效果** | 伪影完全消除，密集预测任务性能提升，无监督目标发现大幅改善 |
| **开销** | $N=4$ 时 FLOPs < 2%，参数量可忽略 |
| **默认配置** | $N = 4$（消除伪影 + 性能最优的平衡点） |

Register Token 揭示的深层洞察是：Transformer 的 self-attention 机制天然具有"全局信息聚合"的需求，但标准架构没有为此提供专用通道。当模型足够大、训练足够久时，它会自发地在低信息区域"创造"出这些通道——而这个过程会污染 patch token 的局部表征。Register Token 的优雅之处在于：它没有改变 Transformer 的计算逻辑，只是显式地提供了模型本就需要的东西。

---

**参考文献**

1. Darcet, T., Oquab, M., Mairal, J., & Bojanowski, P. (2024). *Vision Transformers Need Registers*. ICLR 2024 (Outstanding Paper).
2. Oquab, M., et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision*. arXiv:2304.07193.
3. Caron, M., et al. (2021). *Emerging Properties in Self-Supervised Vision Transformers*. ICCV 2021.
4. Wang, J., et al. (2025). *VGGT: Visual Geometry Grounded deep Transformer*. arXiv:2503.11651.
5. Wang, J., et al. (2025). *Mamba-Reg: Vision Mamba Also Needs Registers*. CVPR 2025.
