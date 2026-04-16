---
title: 笔记｜世界模型（四）：视频生成即世界模拟——从 Sora 到 Genie 与 Cosmos
date: 2026-04-06 00:45:00
categories:
 - Tutorials
tags:
 - Sora
 - Genie
 - Cosmos
 - Video Generation
 - World Model
series: "世界模型"
mathjax: true
---

> **核心论文**：
>
> - Genie: arXiv:2402.15391 (ICML 2024)
>
> - Cosmos: arXiv:2501.03575 (2025)
>
> - UniSim: arXiv:2310.06114 (ICLR 2024)
>
> **前置知识**：[上一篇：JEPA](posts/37-jepa/)

---

## 0. 如果视频模型就是世界模型？

给一个视频生成模型一张图片和指令"向左走"，它生成了一段视角向左移动的视频——走廊延伸出去，墙上的画从右侧滑入视野，地板的透视关系正确变化。

这个视频模型"理解"了三维空间吗？它是不是已经在内部构建了一个"世界模型"？

2024 年 2 月，OpenAI 发布 Sora 时明确提出：**视频生成模型是世界模拟器的有前途的路径。** 这是与 JEPA 截然相反的立场——JEPA 说"不需要生成像素"，Sora 说"生成像素本身就是理解世界"。

---

## 1. Sora：视频生成模型作为世界模拟器

### 1.1 OpenAI 的定位

OpenAI 在 Sora 的技术报告中写道：

> *"我们发现视频模型在大规模训练后展现出了许多有趣的涌现能力。这些能力使 Sora 能够模拟物理世界中人、动物和环境的某些方面。"*

### 1.2 架构要点

Sora 的架构未完全公开，但从技术说明和第三方分析可推断：

1. **时空 Patch 化**：将视频分为 $T \times H \times W$ 的时空 patch

2. **潜在扩散**：在 VAE 的潜空间中做扩散去噪

3. **DiT 架构**：使用 Diffusion Transformer（与 SD3/Flux 的 MMDiT 类似）

4. **可变分辨率/时长**：支持不同宽高比和视频长度

### 1.3 "世界模拟器"的能力与局限

**涌现能力**（据官方报告）：

- 3D 一致性：摄像机移动时场景几何保持一致

- 对象持久性：物体被遮挡后重新出现时保持一致

- 简单物理交互：球的弹跳、液体的流动

**局限性**：

- 复杂物理推理失败：反重力、物体穿透

- 长程因果关系弱：前后帧的因果链容易断裂

- 本质上是**统计相关性**而非**物理因果模型**

### 1.4 数学分析与 2025 年的理论反思：为什么像素相关性 ≠ 物理理解？

设 $p_\theta(V)$ 为视频生成模型学到的分布，$p_{\text{real}}(V)$ 为真实视频分布。

即使 $D_{\text{KL}}(p_{\text{real}} \| p_\theta) \to 0$（模型完美拟合数据分布），也不意味着模型理解了物理：

> **命题（统计模拟的不充分性）**：存在视频分布 $p(V)$ 和两个模型 $M_1, M_2$，使得 $p_{M_1}(V) = p_{M_2}(V) = p(V)$，但 $M_1$ 内部包含正确的物理模型而 $M_2$ 不包含。两者在分布匹配意义下不可区分，但在反事实推理（"如果重力加倍会怎样？"）上表现不同。

**2024-2025 年的实证与机理研究**：

为了回答"Sora 等模型离真正的世界模型还有多远"，2024 年末的一项重要研究（*How Far is Video Generation from World Model: A Physical Law Perspective*）构建了一个受经典力学控制的 2D 碰撞测试床。研究发现：

1. **泛化能力的错觉**：模型在分布内（In-distribution）表现完美，在组合泛化上表现出可测量的缩放定律（Scaling law），但在分布外（Out-of-distribution）泛化上彻底失败。
2. **基于案例的模仿，而非规则抽象**：机理分析表明，视频生成模型并没有抽象出一般的物理规则（如动量守恒）。相反，它们展现出"基于案例（Case-based）"的泛化——遇到新场景时，模型只是在模仿训练集中最相似的例子。
3. **特征优先级的偏置**：在尝试泛化时，模型优先关注表观特征而非物理特征，优先级顺序为：颜色 > 大小 > 速度 > 形状。

**结论**：单靠扩大模型规模（Scaling alone）不足以让视频生成模型发现基础的物理定律。这直接挑战了"只要数据足够多，视频模型就能自然涌现出物理引擎"的乐观假设。

---

## 2. Genie：可交互的世界模型

![Genie Architecture](/img/genie_architecture.png)

### 2.1 核心创新：从视频中学习潜动作

Genie（DeepMind, 2024）解决了一个关键问题：大多数互联网视频**没有动作标注**。人玩游戏的视频只有画面，没有对应的按键记录。

Genie 的方案：从视频中**自动发现**潜在动作空间。

### 2.2 三模块架构

$$
\text{Genie} = \underbrace{\text{ST-Tokenizer}}_{\text{时空视频编码}} + \underbrace{\text{Latent Action Model}}_{\text{潜动作推断}} + \underbrace{\text{Dynamics Model}}_{\text{下一帧预测}}
$$

**时空视频 Tokenizer**：将视频帧序列编码为离散 token

$$
\mathbf{T}_t = \text{VQ}(\text{Enc}(I_t)) \in \{1, \ldots, K\}^{h \times w}
$$

**潜动作模型**：从连续两帧推断潜动作

$$
a_t = \text{ActionModel}(\mathbf{T}_t, \mathbf{T}_{t+1}) \in \{1, \ldots, N_a\}
$$

这里 $a_t$ 是离散的潜动作（通过 VQ 离散化），$N_a$ 是潜动作空间大小。关键：$a_t$ 不是人类标注的，而是模型自动从帧间变化中推断的。

**动力学模型**：给定当前帧和潜动作，预测下一帧

$$
\hat{\mathbf{T}}_{t+1} = \text{Dynamics}(\mathbf{T}_t, a_t)
$$

使用 MaskGIT 式的并行解码。

### 2.3 从观看到操控

训练完成后，用户可以通过输入潜动作来"操控"世界：

1. 给一张起始图片

2. 用户选择离散动作（如"向左"、"跳跃"）

3. 模型生成下一帧

4. 重复——形成可交互的"可玩"环境

### 2.4 规模与结果

| 配置 | 值 |
|------|-----|
| 模型参数 | 11B |
| 训练数据 | 大规模互联网 2D 平台游戏视频 |
| 潜动作数 | 8 个离散动作 |
| 帧率 | 1 FPS（受限于计算） |

Genie 证明了**无需动作标注就能从视频中学习可交互的世界模型**。

### 2.5 Genie 2 & 3

Genie 2（2024.12）大幅扩展：

- 从 2D 到 **3D** 环境

- 支持键盘鼠标控制

- 长时一致性（数十秒）

- 涌现物理行为和多智能体

Genie 3（2025-2026）进一步推进了规模和能力，但完整技术细节尚未公开。

---

## 3. Cosmos：世界基础模型平台

![Cosmos 世界基础模型与后训练应用概览（NVIDIA, 2025, Fig. 1）](/img/video_wm_overview.png)

### 3.1 NVIDIA 的定位

Cosmos 不只是一个模型，而是一个**平台**——为 Physical AI（机器人、自动驾驶）提供世界模型基础设施。

### 3.2 平台组成

```
Cosmos 平台
    ├── 视频策展管线（数据清洗、过滤、字幕生成）
    ├── 视频 Tokenizer（连续 / 离散）
    ├── Cosmos-Predict（预训练世界模型）
    ├── Cosmos-Transfer（条件化世界生成）
    └── 后训练工具（面向特定领域微调）
```

### 3.3 Cosmos-Predict

基于 Diffusion Transformer 的视频预测模型，预训练在大规模真实世界视频上：

$$
\hat{V}_{t+1:t+K} = \text{Cosmos-Predict}(V_{1:t}, c)
$$

其中 $c$ 为可选的条件信号（文本描述、动作指令等）。

### 3.4 Cosmos-Transfer1：多模态控制

Cosmos-Transfer1 支持多种**结构化控制条件**进行世界变换：

| 控制条件 | 含义 |
|---------|------|
| 深度图 | 场景的 3D 结构 |
| 语义分割 | 物体类别 |
| 边缘图 | 轮廓信息 |
| LiDAR 点云 | 激光雷达数据 |
| HD Map | 高精地图 |

核心用途：**Sim2Real**——将仿真器的结构化输出（深度、分割）转化为逼真的视频。

```python
"""Conceptual Cosmos-Transfer1-style inference API."""

from cosmos import CosmosTransfer


def synthesize_video(depth_map, seg_map, hd_map, num_frames: int = 30):
    """Generate video conditioned on structured maps.

    Args:
        depth_map: Depth sequence, shape (T, H, W).
        seg_map: Segmentation sequence, shape (T, H, W).
        hd_map: HD map tensor or raster for routing context.
        num_frames: Number of output frames to synthesize.

    Returns:
        Generated video tensor from the pretrained transfer model.
    """
    model = CosmosTransfer.from_pretrained("nvidia/cosmos-transfer1")

    conditions = {
        "depth": depth_map,  # (T, H, W)
        "segmentation": seg_map,  # (T, H, W)
        "hdmap": hd_map,
    }

    weights = {"depth": 0.8, "segmentation": 0.5, "hdmap": 0.3}

    return model.generate(
        conditions,
        weights=weights,
        num_frames=num_frames,
    )
```

---

## 4. UniSim：通用交互式模拟器

UniSim（Google/Berkeley, 2024）目标更宏大：用视频生成模型构建一个**通用的交互式真实世界模拟器**。

### 4.1 核心设计

UniSim 将多种数据源统一到同一个视频生成框架中：

- 互联网视频（无动作标注）

- 机器人操作视频（有动作标注）

- 导航视频（有 pose 标注）

- 仿真器渲染（有完整标注）

### 4.2 统一的条件生成

$$
V_{t+1:t+K} = \text{UniSim}(V_{1:t}, a_t, \text{text}_t, \text{pose}_t, \ldots)
$$

UniSim 支持多种条件信号的任意组合。

### 4.3 作为 RL 训练环境

UniSim 最激动人心的应用：**用视频生成模型替代传统仿真器训练 RL 智能体**。

传统流程：设计仿真器 → 训练策略 → Sim2Real 迁移
UniSim 流程：学习视频世界模型 → 在模型中训练策略 → 直接部署

---

## 5. 四大视频世界模型对比

| 维度 | Sora | Genie | Cosmos | UniSim |
|------|------|-------|--------|--------|
| 机构 | OpenAI | DeepMind | NVIDIA | Google/Berkeley |
| 开源 | ✗ | 部分 | ✓ | ✗ |
| 可交互 | ✗ | ✓ | ✗（预测模式） | ✓ |
| 动作空间 | 无 | 潜动作（自学习） | 条件信号 | 多类型动作 |
| 3D 一致性 | 有限 | Genie 2 支持 | Sim2Real | 有限 |
| 物理理解 | 统计相关 | 涌现行为 | 领域微调 | 统计相关 |
| 目标应用 | 内容创作 | 游戏/具身 AI | 机器人/驾驶 | 通用仿真 |

---

## 6. 视频世界模型的数学框架

### 6.1 统一形式化

所有视频世界模型都可以用条件视频分布统一描述：

$$
p_\theta(V_{t+1:t+H} \mid V_{1:t}, c)
$$

其中 $c$ 为条件信号（动作/文本/控制图等），$H$ 为预测时域。

不同模型的区别在于：

| 组件 | Sora | Genie | Cosmos |
|------|------|-------|--------|
| $V$ 的表示 | 潜在连续 (VAE) | 离散 token (VQ) | 两者兼有 |
| 生成方式 | 扩散去噪 | MaskGIT 并行解码 | 扩散去噪 |
| $c$ 的类型 | 文本 | 潜动作 | 多模态 |

### 6.2 与传统仿真器的对比

| 维度 | 传统仿真器 | 视频世界模型 |
|------|-----------|------------|
| 物理模型 | 显式方程（$F=ma$） | 隐式（从数据学习） |
| 视觉真实感 | 有限（需要手工设计） | 高（从真实视频学习） |
| 开发成本 | 高（每个场景需要建模） | 低（数据驱动） |
| 物理准确性 | 高（精确方程） | 低（统计近似） |
| 可验证性 | ✓（可检查方程） | ✗（黑盒） |
| 反事实推理 | ✓（改变参数即可） | ✗（需要重新训练） |

---

## 7. 总结

视频生成世界模型代表了一种大胆的假设：**通过学习预测像素，模型会隐式地学到世界的结构。** 这与 JEPA 的"丢弃像素细节"形成了鲜明对照。

当前的共识是：纯数据驱动的视频模型虽然能产生惊人的视觉效果，但在**物理准确性**和**可靠的因果推理**上还有显著差距。下一篇将介绍如何通过**显式嵌入物理定律**来弥补这个差距。

> 参考资料：
>
> 1. OpenAI (2024). *Video generation models as world simulators*. Technical Report.
> 2. Bruce, J., ... & Vinyals, O. (2024). *Genie: Generative Interactive Environments*. ICML 2024.
> 3. NVIDIA (2025). *Cosmos World Foundation Model Platform for Physical AI*. arXiv:2501.03575.
> 4. Yang, M., ... & Levine, S. (2024). *Learning Interactive Real-World Simulators*. ICLR 2024.
> 5. (2024). *How Far is Video Generation from World Model: A Physical Law Perspective*. arXiv:2411.02385. (关于视频生成模型物理泛化能力的实证分析)

> 下一篇：[笔记｜世界模型（五）：物理化的视频生成——让模型理解牛顿定律](/chengYi-xun/posts/39-physics-world-model/)
