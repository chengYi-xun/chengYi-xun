---
title: 笔记｜多模态融合（一）：VGGT——用一个 Transformer 完成所有 3D 视觉任务
date: 2026-04-05 20:00:00
cover: false
mathjax: true
categories:
 - Notes
tags:
 - Deep learning

 - 3D Vision

 - Transformer

 - Multi-modal Fusion
series: Multi-modal Fusion
---

> CVPR 2025 Best Paper Award
>
> ⬅️ 上一篇：[笔记｜MemoryBank：用艾宾浩斯遗忘曲线赋予 LLM 长期记忆](/chengYi-xun/posts/26-memory-bank/)
>
> ➡️ 下一篇：[笔记｜Vision Transformers Need Registers：用 Register Tokens 治愈 ViT 的"注意力伪影"](/chengYi-xun/posts/28-register-tokens/)
>
> 论文：[VGGT: Visual Geometry Grounded Transformer](https://arxiv.org/abs/2503.11651)（Oxford VGG + Meta AI）
>
> 代码：[github.com/facebookresearch/vggt](https://github.com/facebookresearch/vggt)

# 一个模型，四个任务

**从一个例子开始。** 假设你有一组室内场景的照片（10 张不同角度），传统做法需要：

| 任务 | 传统方法 | 耗时 |
|:---:|:---|:---:|
| 相机位姿估计 | COLMAP（特征匹配 → 三角化 → BA） | ~15s |
| 深度图估计 | MVSNet（需要已知相机参数） | 依赖上一步 |
| 点云重建 | DUSt3R（两两配对 → 全局对齐） | ~200s（32帧） |
| 点追踪 | CoTracker（逐帧处理） | 另起一个模型 |

四个任务、四个模型、流水线式的依赖、分钟级的耗时。

**VGGT 的做法**：一个前馈 Transformer，输入 10 张图，**0.2 秒**输出全部四种 3D 属性：

$$
f\left((I_i)_{i=1}^{N}\right) = \left(g_i,\, D_i,\, P_i,\, T_i\right)_{i=1}^{N}
$$

其中 $g_i \in \mathbb{R}^9$ 是相机参数（旋转四元数 + 平移 + 视场角），$D_i \in \mathbb{R}^{H \times W}$ 是深度图，$P_i \in \mathbb{R}^{3 \times H \times W}$ 是点云图，$T_i \in \mathbb{R}^{C \times H \times W}$ 是追踪特征。

更惊人的是——它在多个任务上**超越了**需要后处理优化的专用方法。

---

# 架构：Alternating Attention

VGGT 的核心思想极其简洁：**不设计专门的 3D 归纳偏置，让一个大 Transformer 从数据中学习 3D 几何**。

![VGGT 方法概览（摘自 Wang et al., arXiv:2503.11651 首页图）](/chengYi-xun/img/vggt_arch.png)

## Token 构成

每帧图像 $I_i$ 经过冻结的 DINOv2 编码为 $K$ 个 patch tokens $t_i^I \in \mathbb{R}^{K \times C}$。然后为每帧添加两种特殊 Token：

| Token 类型 | 数量 | 作用 |
|:---:|:---:|:---|
| Camera Token $t_i^g$ | 1 | 承载相机参数预测 |
| Register Tokens $t_i^R$ | 4 | 防止 patch tokens 被劫持存储全局信息 |
| Patch Tokens $t_i^I$ | $K$ | 编码图像局部视觉信息 |

拼接后的序列为 $[t_i^g,\, t_i^R,\, t_i^I]$，所有帧一起送入主 Transformer。

代码中的实现（`aggregator.py`）：

```python
# 第一帧与后续帧各用一套可学习 camera / register；
# 张量中有一维大小为 2，对应参考帧与其余帧
self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
self.register_token = nn.Parameter(
    torch.randn(1, 2, num_register_tokens, embed_dim)
)
# 1 个 camera + num_register 个 register 之后才是 patch
self.patch_start_idx = 1 + num_register_tokens  # = 5

camera_token = slice_expand_and_flatten(self.camera_token, B, S)  # (B*S,1,C)
register_token = slice_expand_and_flatten(self.register_token, B, S)
# register: (B*S, R, C)；patch: (B*S, K, C)，R/K 见上文
tokens = torch.cat(
    [camera_token, register_token, patch_tokens], dim=1
)  # (B*S, 1+R+K, C)
```

**第一帧 vs. 其他帧**：camera_token 和 register_token 的 shape 中有一个维度为 2，分别对应第一帧（参考帧）和其他帧。这允许模型区分参考坐标系——所有 3D 预测都在第一帧相机的坐标系下表达。

## Register Tokens 的必要性

Register Tokens 来自 DINOv2 团队的发现（[Darcet et al., 2023](https://arxiv.org/abs/2309.16588)）。在大规模 ViT 中，模型会**劫持低信息区域的 patch tokens**（如天空、墙壁）来存储全局信息，导致这些 token 的 L2 范数异常升高、特征图被污染。

Register Tokens 提供了专门的可学习 Token 作为"全局信息容器"，使 patch tokens 保持干净——这对 VGGT 的密集预测任务（深度图、点云图）至关重要。

## Alternating Attention

VGGT 的 Transformer 由 $L = 24$ 层**交替注意力**（Alternating Attention, AA）构成。每个 AA block 包含两种注意力：

1. **帧内注意力（Frame Attention）**：每帧的 token 独立做 self-attention，等价于对 $B \times S$ 个独立序列做 attention

$$
\text{Frame-Attn}(t_i) = \text{SelfAttn}([t_i^g,\, t_i^R,\, t_i^I]) \quad \forall i \in \{1, \ldots, N\}
$$

2. **全局注意力（Global Attention）**：所有帧的 token 混在一起做 self-attention

$$
\text{Global-Attn}(t) = \text{SelfAttn}([t_1^g, t_1^R, t_1^I, \ldots, t_N^g, t_N^R, t_N^I])
$$

两者交替执行：

```python
# Alternating Attention：通常重复 aa_block_num 次（如 24），
# 每轮按 aa_order 在 frame / global 间切换
for _ in range(self.aa_block_num):
    for attn_type in self.aa_order:  # e.g. ["frame", "global"]
        if attn_type == "frame":
            tokens = self._process_frame_attention(
                tokens, B, S, P, C, ...
            )  # (B, S, T, C)
        elif attn_type == "global":
            tokens = self._process_global_attention(
                tokens, B, S, P, C, ...
            )  # (B, S, T, C)
```

**为什么不只用全局注意力？** 消融实验（论文 Table 5）显示：

| 注意力方案 | 点云整体误差 (ETH3D) ↓ |
|:---:|:---:|
| Cross-Attention | 1.061 |
| 仅全局 Self-Attention | 0.827 |
| **Alternating Attention** | **0.709** |

帧内注意力允许模型在每帧内部"整理信息"，类似于在小组讨论前先让每个人整理自己的想法。

---

# 四个预测头

## 1. Camera Head：迭代精炼

Camera Head 从 camera token $\hat{t}_i^g$ 预测 $g_i = [q_i, t_i, f_i]$（四元数 + 平移 + 视场角）。采用**迭代精炼**策略：

```python
class CameraHead(nn.Module):
    def trunk_fn(self, pose_tokens, num_iterations=4):
        """迭代精炼相机位姿编码。

        行为与仓库 `camera_head.py` 中实现一致（示意摘录）。

        Args:
            pose_tokens: backbone 输出的 camera token，形状约 (B, T, C)。
            num_iterations: 迭代次数，常见为 4。

        Returns:
            list[torch.Tensor]: 每步经 `activate_pose` 后的位姿编码。
        """
        pred_pose_enc = None
        pred_pose_enc_list = []
        for _ in range(num_iterations):
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens)
            else:
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # 用上一步位姿生成 AdaLN 的 shift / scale / gate
            shift, scale, gate = self.poseLN_modulation(
                module_input
            ).chunk(3, dim=-1)
            modulated = gate * modulate(
                self.adaln_norm(pose_tokens), shift, scale
            )
            modulated = modulated + pose_tokens

            # 小型 trunk（若干层 Transformer）+ MLP 预测位姿增量
            modulated = self.trunk(modulated)
            delta = self.pose_branch(self.trunk_norm(modulated))
            pred_pose_enc = (
                delta if pred_pose_enc is None else pred_pose_enc + delta
            )

            # 四元数归一化、FOV 取正等
            activated = activate_pose(pred_pose_enc, ...)
            pred_pose_enc_list.append(activated)
        return pred_pose_enc_list
```

**关键设计**：

- 每次迭代预测的是**增量** $\Delta g$，逐步精炼

- `detach()` 阻断梯度回传到上一次迭代，避免计算图爆炸

- 使用 **AdaLN**（Adaptive Layer Normalization，与 DiT 中的设计相同）将上一步预测作为条件注入

## 2. DPT Head：密集预测

深度图 $D_i$、点云图 $P_i$ 和追踪特征 $T_i$ 都从 patch tokens $\hat{t}_i^I$ 经过 **DPT**（Dense Prediction Transformer）头预测。DPT 将 Transformer 的多尺度特征融合为全分辨率密集预测。

## 3. Tracking Head

追踪头基于 **CoTracker2** 架构。给定查询点 $y_q$ 在查询图像 $I_q$ 中的特征 $T_q(y_q)$，与所有其他帧的特征图 $T_i$ 计算相关性，再通过 self-attention 层预测对应点 $\hat{y}_i$。

---

# 冗余预测的力量

VGGT 的一个反直觉发现：**同时预测冗余的输出反而更好**。

点云图 $P_i$ 可以由深度图 $D_i$ 和相机参数 $g_i$ 通过几何关系精确计算得到（反投影）。按常理，预测 $P_i$ 是多余的。但消融实验表明（论文 Table 6）：

| 训练时是否包含 | 相机损失 | 深度损失 | 追踪损失 | 点云整体误差 ↓ |
|:---:|:---:|:---:|:---:|:---:|
| | ✗ | ✓ | ✓ | 0.834 |
| | ✓ | ✗ | ✓ | 0.727 |
| | ✓ | ✓ | ✗ | 0.790 |
| | ✓ | ✓ | ✓ | **0.709** |

**所有任务一起训练，每个任务都变好了**——即使任务之间存在数学冗余。这暗示多任务学习中的**隐式正则化效应**：相机参数和深度图的联合训练迫使模型学到更一致的 3D 表征。

更有趣的是，推理时用分别预测的深度图和相机参数**重新计算**点云（而非直接用点云头的输出），精度更高（0.677 vs. 0.709）。这说明将复杂任务分解为子问题，比端到端预测更鲁棒。

---

# 实验结果

## 相机位姿估计

在 RealEstate10K（零样本泛化）和 CO3Dv2 上，VGGT 的前馈结果超越所有需要后优化的方法：

| 方法 | Re10K AUC@30 ↑ | CO3Dv2 AUC@30 ↑ | 耗时 |
|:---:|:---:|:---:|:---:|
| COLMAP+SPSG | 45.2 | 25.3 | ~15s |
| DUSt3R | 67.7 | 76.7 | ~7s |
| MASt3R | 76.4 | 81.8 | ~9s |
| VGGSfM v2 | 78.9 | 83.4 | ~10s |
| **VGGT（前馈）** | **85.3** | **88.2** | **~0.2s** |
| VGGT + BA | 93.5 | 91.8 | ~1.8s |

**前馈推理 0.2 秒**，比 DUSt3R 快 35 倍，精度高 26%。加上 Bundle Adjustment 后进一步提升至 93.5。

## 点云重建

在 ETH3D 上，VGGT 无需已知相机参数，精度接近使用 GT 相机的专用 MVS 方法：

| 方法 | 已知 GT 相机 | 整体误差 ↓ | 耗时 |
|:---:|:---:|:---:|:---:|
| GeoMVSNet | ✓ | 0.295 | - |
| MASt3R | ✓ | 0.374 | ~9s |
| DUSt3R | ✗ | 1.741 | ~7s |
| **VGGT** | ✗ | **0.382** | ~0.2s |

## 作为特征骨干

VGGT 预训练的特征可以增强下游任务。替换 CoTracker 的骨干后，在 TAP-Vid RGB-S 上 $\delta_{\text{avg}}^{\text{vis}}$ 从 78.9 提升至 84.0。

---

# 总结与思考

VGGT 的核心贡献可以用一句话概括：**用足够大的 Transformer + 足够多的 3D 数据，替代手工设计的 3D 几何流水线**。

这与大语言模型的理念一脉相承——GPT 不需要语法树就能理解语言，VGGT 不需要 RANSAC 和 Bundle Adjustment 就能理解 3D 几何。

| 设计选择 | 传统方法 | VGGT |
|:---|:---|:---|
| 3D 归纳偏置 | 大量（对极约束、BA、PnP） | 几乎没有（仅 AA） |
| 任务耦合 | 串行流水线 | 并行多头预测 |
| 输入帧数 | 成对或固定数量 | 1 到数百 |
| 后处理 | 必须（全局对齐/BA） | 可选（已经很好） |
| 推理速度 | 秒~分钟级 | **亚秒级** |

VGGT 展示了"暴力"方法（更大的模型、更多的数据）在 3D 视觉中的潜力。但也留下了开放问题：这种方法能否推广到真正大规模的场景（城市级重建）？在极端视角变化或遮挡下，纯前馈方法能否匹敌迭代优化？

> 参考资料：
>
> 1. Wang, Y., ... & Vedaldi, A. (2025). *VGGT: Visual Geometry Grounded Transformer*. CVPR 2025.
> 2. Darcet, T., Oquab, M., Mairal, J., & Bojanowski, P. (2023). *Vision Transformers Need Registers*. arXiv:2309.16588.
> 3. Wang, S., ... & Reizenstein, J. (2023). *DUSt3R: Geometric 3D Vision Made Easy*. CVPR 2024.
> 4. Karaev, S., ... & Vedaldi, A. (2023). *CoTracker: It is Better to Track Together*. arXiv:2307.07635.
> 5. Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). *Vision Transformers for Dense Prediction*. ICCV 2021.
