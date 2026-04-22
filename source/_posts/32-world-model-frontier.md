---
title: 笔记｜世界模型（七）：前沿与统一视角——五条路线的收敛
date: 2026-04-06 01:00:00
categories:
 - Tutorials
tags:
 - World Model
 - 4D Generation
 - WorldScore
 - Embodied AI
 - Frontier
series: "世界模型"
mathjax: true
---

> **前置知识**：本文是世界模型系列的终篇。建议先阅读前六篇。
> - [（一）世界模型基础](/chengYi-xun/posts/26-world-model-basics/)
> - [（二）Dreamer 系列](/chengYi-xun/posts/27-dreamer/)
> - [（三）JEPA](/chengYi-xun/posts/28-jepa/)
> - [（四）视频生成世界模型](/chengYi-xun/posts/29-video-world-model/)
> - [（五）物理化世界模型](/chengYi-xun/posts/30-physics-world-model/)
> - [（六）自动驾驶世界模型](/chengYi-xun/posts/31-driving-world-model/)
>
> ⬅️ 上一篇：[笔记｜世界模型（六）：自动驾驶世界模型——从视频预测到占用预测](/chengYi-xun/posts/31-driving-world-model/)

## 0. 2026 年的世界模型：百花齐放还是趋于收敛？

回顾前六篇，我们沿着五条路线走了一遍世界模型的版图：

1. **Model-based RL**（Dreamer 系列）：在潜空间做想象训练
2. **JEPA**（I/V-JEPA）：在嵌入空间做预测，丢弃不可预测细节
3. **视频生成**（Sora、Genie、Cosmos）：用大规模视频模型涌现物理理解
4. **物理化生成**（PhysDreamer、NewtonGen）：显式嵌入牛顿定律
5. **自动驾驶**（GAIA-1、Vista、OccWorld）：像素/占用/结构化三种预测范式

截至 2026 年，我们看到一个明确的趋势：**路线开始交叉融合。** 本篇讨论四个前沿方向，然后从统一视角分析这种收敛。

## 1. 4D 生成：时空一体的世界构建

### 1.1 什么是 4D 生成？

4D = 3D 空间 + 时间。4D 生成的目标是**从文本/图像/视频生成随时间变化的 3D 场景**。

这与视频生成（2D+T）的本质区别：4D 生成的输出是一个可以从任意角度、任意时刻渲染的 3D 场景。

### 1.2 4D Gaussian Splatting

主流的 4D 表示基于 **4D Gaussian Splatting**：

每个高斯核的参数是时间的函数：

$$
\begin{aligned}
\mu_i(t) &= \mu_i^0 + \Delta\mu_i(t) \quad &\text{（位置随时间变化）} \\
\Sigma_i(t) &= R_i(t) S_i(t) S_i(t)^\top R_i(t)^\top \quad &\text{（形状随时间变化）} \\
\alpha_i(t) &= \sigma(\hat{\alpha}_i + \Delta\alpha_i(t)) \quad &\text{（透明度随时间变化）}
\end{aligned}
$$

时间函数 $\Delta\mu_i(t)$、$\Delta\alpha_i(t)$ 等通常用小型 MLP 或多项式参数化：

$$
\Delta\mu_i(t) = \text{MLP}_\theta(t, \mathbf{f}_i) \quad \text{或} \quad \Delta\mu_i(t) = \sum_{k=0}^{K} c_k^i t^k
$$

### 1.3 代表工作

| 方法 | 表示 | 时间建模 | 文本控制 |
|------|------|---------|---------|
| 4D-GS (2024) | 4D Gaussian | 多项式 | ✗ |
| DreamGaussian4D (2024) | 4D Gaussian | MLP | ✓ |
| Align Your Gaussians (Google, 2024) | 4D Gaussian | Score Distillation | ✓ |
| TC4D (2024) | 4D Gaussian + 轨迹控制 | 条件 MLP | ✓ |

### 1.4 4D 生成与世界模型的关系

4D 生成可以视为世界模型的一种**显式几何表示**：

$$
\text{世界模型}: s_{t+1} = f(s_t, a_t) \quad \leftrightarrow \quad \text{4D 生成}: \text{Scene}(t+1) = \text{Render}(\mathcal{G}(t+1))
$$

区别在于：传统世界模型（如 Dreamer）预测**潜变量**，4D 生成直接构建**可从任意视角渲染的 3D 场景**。这意味着 4D 世界模型天然具备多视角一致性，但在处理复杂动态（流体、柔性物体）时仍有局限。

## 2. 评估框架：WorldScore 与 WorldModelBench

### 2.1 如何评估世界模型？

一个根本问题：**FID 和 FVD 只衡量"生成的视频好不好看"，但世界模型的核心价值不是好看，而是物理正确、可控、可用于决策。** 用 FID 评估世界模型，就像用"字写得好不好看"来评估一篇论文的学术水平。

2025 年，学术界推出了专门评估世界模型的统一基准体系，代表作是 **WorldScore**（ICCV 2025）和 **WorldModelBench**（arXiv:2502.20694, NeurIPS 2025）。

### 2.2 WorldScore：统一 3D/4D 与视频生成

WorldScore 是首个统一评估世界生成的基准。它的核心创新在于：将"世界生成"分解为一系列**带有显式相机轨迹布局说明的下一场景生成任务**。这使得它能够用同一套标准评估 3D 场景生成、4D 场景生成和纯视频生成模型。

WorldScore 包含 3000 个精心策划的测试用例（涵盖室内/室外、静态/动态、写实/风格化），主要评估三个维度：
1. **可控性（Controllability）**：模型严格遵循控制输入（如相机轨迹矩阵、文本布局描述）的能力。
2. **质量（Quality）**：生成场景的保真度和多视角一致性。
3. **动力学（Dynamics）**：生成世界中运动的准确性和稳定性。

### 2.3 WorldModelBench：聚焦物理幻觉与常识

WorldModelBench（arXiv:2502.20694, NeurIPS 2025）更侧重于评估视频生成模型的"世界建模"能力，包含 350 个精心策划的图像-文本条件对，覆盖 7 个应用驱动领域（机器人、自动驾驶、工业、人类活动、游戏、动画、自然场景）和 56 个子领域。衡量三个核心能力：

1. **指令遵循（Instruction Following）**：物体和运动是否符合指令
2. **常识（Common Sense）**：时序一致性和视觉质量
3. **物理遵循（Physical Adherence）**：专门调查常见的物理幻觉（违反牛顿第一定律、物体穿模、物体大小突变等）

关键创新：为了克服传统自动化指标的缺陷，WorldModelBench 收集了 **6.7 万个人类标签**，评估了 14 个前沿模型，并微调了一个视觉语言裁判模型用于自动评估（在预测物理违规方面比 GPT-4o 更准确）。

### 2.4 评估结果的启示

根据这些最新基准的测试结果：
- **物理推理依然是短板**：即使是最先进的闭源视频模型，在 WorldModelBench 的"物理遵循"测试中也经常出现违背牛顿定律的幻觉。
- **4D 生成在一致性上占优，但在动态性上落后**：基于 4D Gaussian 的方法在 WorldScore 的"质量"（多视角一致性）上表现优异，但在处理复杂、非刚性的"动力学"时，往往不如大规模视频扩散模型自然。

## 3. 具身智能中的世界模型

### 3.1 Thinker-Talker 架构

2025-2026 年的具身智能研究中，一种流行的架构是 **Thinker-Talker**：

- **Thinker**（世界模型）：在潜空间中"想象"未来，生成预测和计划
- **Talker**（策略网络/LLM）：基于 Thinker 的预测做出决策

$$
\begin{aligned}
\text{Thinker:} \quad & z_{t+1:t+H} = \text{WorldModel}(z_t, a_{t:t+H-1}) \\
\text{Talker:} \quad & a^* = \text{Policy}(z_t, z_{t+1:t+H})
\end{aligned}
$$

### 3.2 LeWorldModel：从像素直接学习 JEPA 世界模型

LeWorldModel（arXiv:2603.19312, 2026）是第一个**稳定的端到端 JEPA 世界模型**——直接从原始像素学习，无需复杂的训练技巧。

核心创新：用 **SIGReg**（Sketched-Isotropic-Gaussian Regularizer）替代传统 JEPA 中防止坍缩的复杂机制（如 EMA、stop-gradient）。训练目标只有两项：

$$
\mathcal{L} = \underbrace{\|\hat{z}_{t+1} - z_{t+1}\|^2}_{\text{下一嵌入预测}} + \lambda \cdot \underbrace{\text{SIGReg}(z)}_{\text{高斯正则化}}
$$

| 特性 | 值 |
|------|-----|
| 参数量 | ~15M（极轻量） |
| 训练成本 | 单 GPU 数小时 |
| 规划速度 | 比基础模型类世界模型快 48 倍 |
| 物理理解 | 对物理不合理事件（如瞬移）给出更高的"惊讶值" |

LeWorldModel 的意义在于证明了：**JEPA 世界模型不需要庞大的参数和复杂的训练流程**，一个简洁的正则化就能让嵌入空间稳定地编码物理结构。

### 3.3 机器人操作中的世界模型

**UniPi**（2023）用视频生成模型做机器人规划：

1. 给定当前场景图像和目标描述（"把红色方块放到蓝色碗里"）
2. 用视频生成模型"想象"完成任务的视频
3. 用逆动力学模型从视频中提取动作序列
4. 执行提取的动作

$$
\text{Image}_0 + \text{Goal} \overset{\text{Video Gen}}{\longrightarrow} \hat{V}_{0:T} \overset{\text{Inverse Dynamics}}{\longrightarrow} a_{0:T-1} \overset{\text{Execute}}{\longrightarrow} \text{Robot}
$$

## 4. 五条路线的收敛

### 4.1 收敛趋势分析

```
2023:                    2026:
  
  MBRL ──────┐           
             │ 潜空间预测 ────→ 统一的
  JEPA ──────┘                 嵌入空间
                               世界模型
  Video Gen ─┐                    ↑
             │ 像素/token ──→ 可渲染+
  Physics ───┘  生成          可控+物理
                                 ↑
  AD WM ─────────────────→ 领域特化
```

### 4.2 三个融合点

**融合点 1：潜空间统一——RSSM 与 JEPA 的汇合**

Dreamer 的 RSSM 和 JEPA 看似截然不同，但本质上都在做同一件事：在潜空间中预测未来状态。

| | Dreamer (RSSM) | JEPA | 融合趋势 |
|--|---|---|---|
| 潜空间 | 确定性 $h_t$ + 随机性 $z_t$ | 嵌入 $z_t$ | 统一的潜在世界表征 |
| 解码器 | 有（重建观测） | 无（只预测嵌入） | LeWorldModel 证明无解码器也可行 |
| 动作条件 | ✓ | V-JEPA 2 加入 | 都支持 |

TD-MPC2 已经证明：无解码器的潜空间世界模型在 **104 个任务**上表现优异，不需要重建任何像素。

**融合点 2：生成 + 物理——让视频既好看又正确**

- 视频生成提供视觉逼真度
- 物理约束提供因果正确性
- 两种融合方式：NewtonGen（训练时嵌入物理方程）vs NewtonRewards（训练后用物理奖励对齐）

**融合点 3：通用 + 领域——Foundation Model 范式**

- 通用世界模型（Cosmos, Sora）提供基础能力
- 领域微调（驾驶、机器人、游戏）提供专业精度
- 类似 LLM 的 "Foundation Model + Fine-tuning" 范式——先在海量视频上预训练，再在特定领域微调

### 4.3 路线对比总结

| 路线 | 核心优势 | 核心局限 | 2026 状态 |
|------|---------|---------|----------|
| **Model-based RL** | 样本效率最高 | 仅潜空间，无视觉 | DreamerV3 固定超参横扫 |
| **JEPA** | 信息选择性，无需重建 | 不能生成，不能可视化 | V-JEPA 2 加入动作条件 |
| **视频生成** | 视觉逼真，涌现能力 | 物理不准确，计算昂贵 | Cosmos 开源，Genie 3 交互 |
| **物理化生成** | 物理正确性 | 场景受限，方法碎片化 | NewtonRewards 后训练范式 |
| **自动驾驶** | 领域优化，实用性强 | 泛化性差 | Vista/OccWorld 走向实用 |

## 5. 开放问题

### 5.1 世界模型需要多大的规模？

Sora 和 Cosmos 表明，**规模**是涌现物理理解的关键因素。但多大才"够"？

$$
\text{能力} \overset{?}{=} f(\text{参数量}, \text{数据量}, \text{计算量})
$$

是否存在类似 LLM 的 **Scaling Law** for World Models？目前没有定论。

### 5.2 统计相关性 vs 因果理解

这是世界模型领域**最根本的哲学问题**，也是上一篇自动驾驶部分提到的因果性反思的延续。

| 问题 | 统计模型能做到 | 需要因果模型 |
|------|-------------|------------|
| "球掉下来会怎样？" | ✓（训练数据中见过） | ✓ |
| "月球上球掉下来会怎样？" | ✗（需要外推） | ✓ |
| "如果重力加倍呢？" | ✗（反事实推理） | ✓ |
| "为什么球会掉？" | ✗（无解释能力） | ✓ |

Pearl 的因果层级（Ladder of Causation, Pearl 2009）：

$$
\underbrace{\text{关联}}_{\text{Level 1: } p(y|x)} \subset \underbrace{\text{干预}}_{\text{Level 2: } p(y|do(x))} \subset \underbrace{\text{反事实}}_{\text{Level 3: } p(y_x|x', y')}
$$

当前所有数据驱动的世界模型最多处于 Level 1（统计关联），偶尔涌现 Level 2 的能力（如 GAIA-1 改变车速时场景正确响应）。真正的"世界模型"需要达到 Level 3——而这可能需要架构层面的根本变革，而不仅仅是更多数据和更大模型。

### 5.3 评估难题

目前没有统一的世界模型评估标准：

- Dreamer 用 RL 任务的回报评估
- Sora 用 FVD 和人类评估
- OccWorld 用 mIoU 评估
- PhysDreamer 用物理准确性评估

WorldScore 是第一次尝试统一评估，但仍然不完善。

![世界模型技术演进全景图：从 Ha & Schmidhuber (2018) 到多模态大模型](/chengYi-xun/img/wm_timeline.png)

## 6. 技术演进全景图

```
1943: Craik, "心智模型"
  │
1989: Sutton, Dyna 架构（Model-based RL 开端）
  │
2015: Schmidhuber, RNN 世界模型
  │
2018: Ha & Schmidhuber, "World Models"（VAE + MDN-RNN + Controller）
  │
  ├── 2019: Dreamer v1（RSSM + 想象训练）
  │     ├── 2021: DreamerV2（离散潜变量）
  │     └── 2023: DreamerV3（固定超参，Minecraft 钻石）
  │
  ├── 2022: LeCun, JEPA 白皮书
  │     ├── 2023: I-JEPA（图像嵌入预测）
  │     ├── 2024: V-JEPA（视频嵌入预测）
  │     ├── 2025: V-JEPA 2（动作条件 + 规划）
  │     └── 2026: LeWorldModel（端到端 JEPA + SIGReg）
  │
  ├── 2023: GAIA-1（9B 驾驶世界模型）
  │     ├── 2023: DriveDreamer（结构化控制）
  │     ├── 2024: Vista（通用驾驶世界模型）
  │     └── 2024: OccWorld（3D 占用预测）
  │
  ├── 2024: Sora（视频生成即世界模拟）
  │     ├── 2024: Genie（可交互世界模型）
  │     └── 2025: Cosmos（世界基础模型平台）
  │
  └── 2024: PhysDreamer / PhysGen（物理化世界模型）
        ├── 2025: NewtonGen（潜空间牛顿方程）
        └── 2025: NewtonRewards（后训练物理奖励）
```

## 7. 系列总结

| 篇目 | 主题 | 核心信息 |
|------|------|---------|
| 第一篇 | 基础概念 | 世界模型 = 前向动力学 + 观测模型 + 奖励模型 |
| 第二篇 | Dreamer | RSSM 在潜空间想象训练策略 |
| 第三篇 | JEPA | 嵌入空间预测，丢弃不可预测细节 |
| 第四篇 | 视频生成 | 大规模视频模型展现涌现物理理解 |
| 第五篇 | 物理化 | 显式嵌入物理定律到生成过程 |
| 第六篇 | 自动驾驶 | 像素预测 / 占用预测 / 结构化预测 |
| 第七篇 | 前沿 | 4D 生成 / 评估框架 / 路线融合 |

世界模型是通向通用人工智能的核心模块之一。无论是 LeCun 的 JEPA 路线还是 OpenAI 的视频生成路线，最终目标一致：**构建一个能够预测、推理、规划的世界内部表征。**

用一句话总结这个系列的核心洞见：**世界模型的本质不是"预测像素"，而是"理解因果"。** 从 Dreamer 的潜空间想象，到 JEPA 的信息选择，到 Sora 的涌现物理，到 NewtonGen 的显式物理嵌入——每一步都是在逼近这个目标。真正的突破可能不会来自更大的模型或更多的数据，而是来自将因果推理能力融入世界模型的架构设计。

> 参考资料：
>
> 1. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
> 2. Ha, D. & Schmidhuber, J. (2018). *World Models*. arXiv:1803.10122.
> 3. LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence*. OpenReview.
> 4. Du, Y., Yang, M., Dai, B., Dai, H., Nachum, O., Tenenbaum, J. B., Schuurmans, D. & Abbeel, P. (2023). *Learning Universal Policies via Text-Guided Video Generation*. NeurIPS 2023.
> 5. NVIDIA (2025). *Cosmos World Foundation Model Platform for Physical AI*. arXiv:2501.03575.
> 6. Duan, H., et al. (2025). *WorldScore: A Unified Evaluation Benchmark for World Generation*. ICCV 2025.
> 7. WorldModelBench Team (2025). *WorldModelBench: Judging Video Generation Models As World Models*. arXiv:2502.20694, NeurIPS 2025.
> 8. LeWorldModel Team (2026). *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*. arXiv:2603.19312.
