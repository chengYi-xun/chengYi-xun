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
> - [（一）世界模型基础](posts/35-world-model-basics/)
> - [（二）Dreamer 系列](posts/36-dreamer/)
> - [（三）JEPA](posts/37-jepa/)
> - [（四）视频生成世界模型](posts/38-video-world-model/)
> - [（五）物理化世界模型](posts/39-physics-world-model/)
> - [（六）自动驾驶世界模型](posts/40-driving-world-model/)

---

## 0. 2026 年的世界模型：百花齐放还是趋于收敛？

回顾过去三年（2023-2026），世界模型领域从五条独立的技术路线发展而来。截至 2026 年，我们看到了一个明确的趋势：**路线开始交叉融合。**

本篇将讨论四个前沿方向，然后从统一视角分析路线收敛。

---

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

4D 生成可以视为世界模型的一种**显式表示**：

$$
\text{世界模型}: s_{t+1} = f(s_t, a_t) \quad \leftrightarrow \quad \text{4D 生成}: \text{Scene}(t+1) = \text{Render}(\mathcal{G}(t+1))
$$

区别在于：传统世界模型预测**潜变量**，4D 生成直接构建**可渲染的 3D 场景**。

---

## 2. 评估框架：WorldScore 与 WorldModelBench

### 2.1 如何评估世界模型？

传统指标（FID、FVD）只衡量生成质量，不衡量"物理正确性"。WorldScore（2025）提出了专门评估世界模型的指标体系。

### 2.2 WorldScore 的五个维度

| 维度 | 评测内容 | 量化方式 |
|------|---------|---------|
| **3D 一致性** | 多视角几何一致 | 重投影误差 + 深度一致性 |
| **物理合理性** | 运动符合物理定律 | 速度/加速度曲线分析 |
| **时序一致性** | 帧间内容连贯 | 光流平滑度 + 对象追踪 |
| **可控性** | 对输入条件的响应精度 | 动作-输出对应关系 |
| **多样性** | 不同采样的变化范围 | FID diversity |

### 2.3 综合评分

$$
\text{WorldScore} = \prod_{i=1}^{5} d_i^{w_i}, \quad \sum_i w_i = 1
$$

使用几何平均而非算术平均——**任何一个维度的严重失败都会拉低总分**，这符合安全关键场景的需求（一个物理离谱的帧就足以让仿真器不可用）。

### 2.4 各模型表现

| 模型 | 3D 一致性 | 物理合理性 | 时序一致性 | 综合 WorldScore |
|------|----------|-----------|-----------|----------------|
| Sora | 中 | 低 | 高 | 中 |
| Cosmos | 中 | 中 | 高 | 中上 |
| Genie 2 | 高 | 中 | 中 | 中上 |
| DreamerV3 | - | 高 | - | -（不生成像素） |

---

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

### 3.2 LeWorldModel：自动驾驶中的世界模型

LeWorldModel（2025）结合了 JEPA 和 BEV 预测：

- 在 BEV 嵌入空间（而非像素空间）做预测
- 用 JEPA 式训练（嵌入预测，无解码器）
- 在嵌入空间中做下游规划

$$
\begin{aligned}
z_t^{\text{BEV}} &= \text{BEVEncoder}(\{I_t^{(i)}\}_{i=1}^{N_{\text{cam}}}) \\
\hat{z}_{t+1}^{\text{BEV}} &= g_\phi(z_t^{\text{BEV}}, a_t)
\end{aligned}
$$

### 3.3 机器人操作中的世界模型

**UniPi**（2023）用视频生成模型做机器人规划：

1. 给定当前场景图像和目标描述（"把红色方块放到蓝色碗里"）
2. 用视频生成模型"想象"完成任务的视频
3. 用逆动力学模型从视频中提取动作序列
4. 执行提取的动作

$$
\text{Image}_0 + \text{Goal} \overset{\text{Video Gen}}{\longrightarrow} \hat{V}_{0:T} \overset{\text{Inverse Dynamics}}{\longrightarrow} a_{0:T-1} \overset{\text{Execute}}{\longrightarrow} \text{Robot}
$$

---

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

**融合点 1：潜空间统一**

- Dreamer 的 RSSM 潜空间 + JEPA 的嵌入空间 → 统一的潜在世界表征
- TD-MPC2 已经证明：无解码器的潜空间世界模型在多任务上表现优异

**融合点 2：生成 + 物理**

- 视频生成提供视觉逼真度
- 物理约束提供因果正确性
- NewtonGen 和 Cosmos-Transfer 代表了两种融合方式

**融合点 3：通用 + 领域**

- 通用世界模型（Cosmos, Sora）提供基础能力
- 领域微调（驾驶、机器人、游戏）提供专业精度
- 类似 LLM 的 "Foundation Model + Fine-tuning" 范式

### 4.3 路线对比总结

| 路线 | 核心优势 | 核心局限 | 2026 状态 |
|------|---------|---------|----------|
| **Model-based RL** | 样本效率最高 | 仅潜空间，无视觉 | DreamerV3 固定超参横扫 |
| **JEPA** | 信息选择性，无需重建 | 不能生成，不能可视化 | V-JEPA 2 加入动作条件 |
| **视频生成** | 视觉逼真，涌现能力 | 物理不准确，计算昂贵 | Cosmos 开源，Genie 3 交互 |
| **物理化生成** | 物理正确性 | 场景受限，方法碎片化 | NewtonRewards 后训练范式 |
| **自动驾驶** | 领域优化，实用性强 | 泛化性差 | Vista/OccWorld 走向实用 |

---

## 5. 开放问题

### 5.1 世界模型需要多大的规模？

Sora 和 Cosmos 表明，**规模**是涌现物理理解的关键因素。但多大才"够"？

$$
\text{能力} \overset{?}{=} f(\text{参数量}, \text{数据量}, \text{计算量})
$$

是否存在类似 LLM 的 **Scaling Law** for World Models？目前没有定论。

### 5.2 统计相关性 vs 因果理解

这是最根本的哲学问题：

| 问题 | 统计模型能做到 | 需要因果模型 |
|------|-------------|------------|
| "球掉下来会怎样？" | ✓（训练数据中见过） | ✓ |
| "月球上球掉下来会怎样？" | ✗（需要外推） | ✓ |
| "如果重力加倍呢？" | ✗（反事实推理） | ✓ |
| "为什么球会掉？" | ✗（无解释能力） | ✓ |

Pearl 的因果层级（Ladder of Causation）：

$$
\underbrace{\text{关联}}_{\text{Level 1: } p(y|x)} \subset \underbrace{\text{干预}}_{\text{Level 2: } p(y|do(x))} \subset \underbrace{\text{反事实}}_{\text{Level 3: } p(y_x|x', y')}
$$

当前所有数据驱动的世界模型最多处于 Level 1（统计关联），偶尔展现 Level 2 的能力。真正的"世界模型"需要达到 Level 3。

### 5.3 评估难题

目前没有统一的世界模型评估标准：

- Dreamer 用 RL 任务的回报评估
- Sora 用 FVD 和人类评估
- OccWorld 用 mIoU 评估
- PhysDreamer 用物理准确性评估

WorldScore 是第一次尝试统一评估，但仍然不完善。

---

![World Model Evolution Timeline](/img/wm_timeline.png)

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
  │     └── 2025: V-JEPA 2（动作条件 + 规划）
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

---

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

世界模型是通向通用人工智能的核心模块之一。无论是 LeCun 的 JEPA 路线还是 OpenAI 的视频生成路线，最终目标一致：**构建一个能够预测、推理、规划的世界内部表征。** 技术路线的收敛还需要时间，但方向已经越来越清晰。

---

**参考文献**

1. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
2. Ha, D. & Schmidhuber, J. (2018). *World Models*. arXiv:1803.10122.
3. LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence*. OpenReview.
4. Du, Y., et al. (2023). *UniPi: Learning Universal Policies via Text-Guided Video Generation*. NeurIPS 2023.
5. NVIDIA (2025). *Cosmos World Foundation Model Platform for Physical AI*. arXiv:2501.03575.
