---
title: 笔记｜世界模型（六）：自动驾驶世界模型——从视频预测到占用预测
date: 2026-04-06 00:55:00
categories:
 - Tutorials
tags:
 - GAIA-1
 - DriveDreamer
 - OccWorld
 - Vista
 - Autonomous Driving
 - World Model
series: "世界模型"
mathjax: true
---

> **核心论文**：GAIA-1 (arXiv:2309.17080, 2023)、DriveDreamer (arXiv:2309.09777, ECCV 2024)、Vista (arXiv:2405.17398, NeurIPS 2024)、OccWorld (arXiv:2311.16038, CVPR 2024)
>
> **前置知识**：[上一篇：物理化世界模型](/chengYi-xun/posts/30-physics-world-model/)
>
> ⬅️ 上一篇：[笔记｜世界模型（五）：物理化的视频生成——让模型理解牛顿定律](/chengYi-xun/posts/30-physics-world-model/)
>
> ➡️ 下一篇：[笔记｜世界模型（七）：前沿与统一视角——五条路线的收敛](/chengYi-xun/posts/32-world-model-frontier/)

## 0. 为什么自动驾驶特别需要世界模型？

假设你在开发自动驾驶系统，需要测试"行人突然从路边冲出"的场景。你有两个选择：

1. **真实路测**：让测试车上路，等这种场景自然发生——可能等几万公里都遇不到一次，而且有安全风险
2. **仿真器**：在 CARLA 等仿真器中构建场景——但仿真器的画面和真实世界差距太大（domain gap），训练出的模型可能无法迁移

世界模型提供了第三种选择：**从真实驾驶视频中学习环境动力学，生成无限的、逼真的、可控的驾驶场景。**

自动驾驶世界模型有三个独特需求：

1. **3D 几何**：必须理解三维空间（深度、遮挡、多视角一致性）
2. **可控性**：必须能根据驾驶指令生成对应场景（转弯、加速、变道）
3. **安全关键**：生成的场景必须用于安全决策，不能有离谱的物理错误

## 1. GAIA-1：最早的驾驶世界模型

### 1.1 核心思想

GAIA-1（Wayve, 2023）是最早的大规模驾驶世界模型之一。核心假设：**用足够大的 Transformer 和足够多的驾驶视频，模型会隐式学到驾驶场景的 3D 结构和运动规律。**

### 1.2 架构

```
Video Encoder → World Model (Transformer) → Video Decoder
     ↑                    ↑
  驾驶视频              条件输入
                   (文本 / 动作 / 地图)
```

**多模态输入**：

$$
\begin{aligned}
\mathbf{v}_t &= \text{VideoTokenizer}(I_t) \quad &\text{(视频 token)} \\
\mathbf{a}_t &= \text{ActionEncoder}(\text{speed}_t, \text{steer}_t) \quad &\text{(动作 token)} \\
\mathbf{c} &= \text{TextEncoder}(\text{prompt}) \quad &\text{(文本描述)}
\end{aligned}
$$

**自回归预测**：

$$
p(\mathbf{v}_{t+1} \mid \mathbf{v}_{\leq t}, \mathbf{a}_{\leq t}, \mathbf{c})
$$

### 1.3 规模

| 配置 | 值 |
|------|-----|
| 参数量 | 9B |
| 训练数据 | 英国城市驾驶视频（约 4700 小时） |
| 分辨率 | 288 × 512 |
| 预测时长 | 数秒 |

### 1.4 涌现能力

GAIA-1 展示了几个有趣的涌现行为：

- **3D 几何理解**：改变车速时，场景以正确的透视变化展开
- **交通规则**：红灯时周围车辆自动停车
- **天气变化**：通过文本控制可切换晴天/雨天/雪天
- **长程一致性**：道路和建筑在数秒内保持一致

## 2. DriveDreamer：结构化控制的驾驶生成

### 2.1 核心改进

DriveDreamer（2023）的核心改进：**用结构化的中间表示（HD Map + 3D Bbox + 交通元素）替代纯视频自回归，提供更精确的场景控制。**

### 2.2 两阶段流程

**阶段 1：结构化世界模型（预测布局）**

给定当前的结构化场景描述，预测未来的场景布局：

$$
(\hat{B}_{t+1}, \hat{M}_{t+1}, \hat{L}_{t+1}) = f_\theta(B_{\leq t}, M_{\leq t}, L_{\leq t}, a_t)
$$

其中 $B$ 是 3D 边界框、$M$ 是 HD Map 元素、$L$ 是车道线/交通灯状态。

**阶段 2：视频生成（渲染像素）**

将预测的结构化布局渲染为逼真视频：

$$
\hat{I}_{t+1} = g_\psi(\hat{B}_{t+1}, \hat{M}_{t+1}, \hat{L}_{t+1}, I_t, \text{noise})
$$

使用条件扩散模型，以结构化布局为控制信号。

### 2.3 数学：条件扩散

DriveDreamer 的视频生成阶段基于条件 DDPM：

$$
\mathcal{L} = \mathbb{E}_{t, \epsilon, I_0}\left[\|\epsilon - \epsilon_\theta(I_t^{\text{noisy}}, t, c_{\text{struct}})\|^2\right]
$$

其中 $c_{\text{struct}} = \text{Encode}(B, M, L)$ 是结构化条件编码。

### 2.4 DriveDreamer-2

DriveDreamer-2（2024）进一步引入 **LLM 作为交通管理器**：

$$
\text{LLM}(\text{当前场景描述}) \to \text{未来交通参与者轨迹}
$$

用 LLM 的常识推理能力生成合理的交通行为（如"前方有红灯，前车应该减速"），再交给视频模型渲染。

## 3. Vista：通用驾驶世界模型

### 3.1 定位

Vista（Tsinghua/Wayve, 2024）试图构建一个**通用的**驾驶世界模型——不局限于特定数据集或驾驶场景。

### 3.2 关键技术

**动态分辨率训练**：

$$
I_{\text{train}} \in \{480p, 720p, 1080p\} \times \{4:3, 16:9, 21:9\}
$$

通过在不同分辨率和宽高比上联合训练，模型学会处理各种驾驶摄像头配置。

**动作条件生成**：

$$
\hat{I}_{t+1:t+K} = \text{Vista}(I_{1:t}, \underbrace{(\text{speed}, \text{steer}, \text{curvature})}_{a_{t:t+K}})
$$

支持连续的**速度-转向-曲率**控制。

**长时预测**：通过自回归 rollout，Vista 能生成数十秒的驾驶视频。每步生成后将输出作为下一步的输入，但加入了**历史帧缓冲**来缓解误差累积。

### 3.3 作为仿真器训练自动驾驶

Vista 最大的贡献：**直接在世界模型生成的视频中训练自动驾驶策略**。

传统流程：

$$
\text{真实数据} \overset{\text{训练}}{\longrightarrow} \pi_\theta \overset{\text{部署}}{\longrightarrow} \text{真实车辆}
$$

Vista 流程：

$$
\text{真实数据} \overset{\text{训练 WM}}{\longrightarrow} \text{Vista} \overset{\text{生成训练数据}}{\longrightarrow} \pi_\theta \overset{\text{部署}}{\longrightarrow} \text{真实车辆}
$$

用 Vista 生成的数据训练的规划器，在 nuScenes 上的 L2 误差降低了 **15-20%**。

## 4. OccWorld：3D 占用空间的世界模型

### 4.1 从像素到体素

前面的方法都在**像素空间**（2D 图像/视频）做预测。OccWorld（THU, 2024）走了不同的路：**在 3D 占用空间中预测世界变化。**

### 4.2 什么是占用空间？

将 3D 空间离散化为体素网格 $\mathcal{O} \in \{0, 1, \ldots, C\}^{X \times Y \times Z}$：

- 每个体素 $(x, y, z)$ 的值表示该位置被什么占据（空气=0，车辆=1，行人=2，建筑=3，...）
- 分辨率通常为 $200 \times 200 \times 16$，覆盖车辆周围 $[-50m, 50m] \times [-50m, 50m] \times [-5m, 3m]$

### 4.3 OccWorld 架构

$$
\hat{\mathcal{O}}_{t+1}, \hat{f}_{t+1} = \text{OccWorld}(\mathcal{O}_{1:t}, f_{1:t}, a_t)
$$

其中 $f_t$ 是场景流（flow），描述每个体素的运动方向和速度。

**GPT 风格的自回归预测**：

OccWorld 将 3D 占用体素序列化为 token，用 GPT 式 Transformer 自回归预测：

1. **空间 VQ 编码**：将 3D 占用网格编码为离散 token

$$
\mathbf{T}_t = \text{VQ-VAE}(\mathcal{O}_t) \in \{1, \ldots, K\}^{n_x \times n_y \times n_z}
$$

2. **时序预测**：用 Transformer 预测下一时刻的 token

$$
\hat{\mathbf{T}}_{t+1} = \text{Transformer}(\mathbf{T}_{1:t}, a_t)
$$

3. **解码**：将 token 解码回 3D 占用网格

$$
\hat{\mathcal{O}}_{t+1} = \text{VQ-Decoder}(\hat{\mathbf{T}}_{t+1})
$$

![像素预测 vs 占用预测：两种自动驾驶世界模型输出范式对比](/chengYi-xun/img/pixel_vs_occupancy.png)

### 4.4 占用预测 vs 像素预测

| 维度 | 像素预测 (GAIA-1 等) | 占用预测 (OccWorld) |
|------|---------------------|-------------------|
| 预测空间 | 2D 像素 | 3D 体素 |
| 3D 信息 | 隐式（需要推断深度） | 显式（直接预测 3D） |
| 计算效率 | 高分辨率视频昂贵 | 离散体素相对高效 |
| 下游任务 | 需要额外感知模块 | 直接用于规划 |
| 视觉真实感 | 高 | 无（纯几何） |

### 4.5 与 BEV 预测的关系

OccWorld 与 BEV（Bird's Eye View）预测的关系：

$$
\text{BEV} \subset \text{Occupancy}
$$

BEV 只预测地面平面（$x$-$y$），而 Occupancy 预测完整的 3D 体积（$x$-$y$-$z$）。Occupancy 能区分"桥下的空间可以通过"和"前方墙壁不可通过"——这是 BEV 无法区分的。

## 5. 自动驾驶世界模型全景对比

| 模型 | 机构 | 年份 | 预测空间 | 控制条件 | 开源 |
|------|------|------|---------|---------|------|
| GAIA-1 | Wayve | 2023 | 像素 (2D) | 动作+文本 | ✗ |
| DriveDreamer | 清华 | 2023 | 像素 (2D) | 3D Box+HDMap | ✓ |
| DriveDreamer-2 | 清华 | 2024 | 像素 (2D) | LLM 交通管理 | ✓ |
| Vista | 清华/Wayve | 2024 | 像素 (2D) | 速度+转向 | 部分 |
| OccWorld | 清华 | 2024 | 3D 占用 | 自车动作 | ✓ |
| Drive-WM | 上交 | 2024 | 像素 (多视角) | 轨迹 | ✓ |
| GenAD | 浙大 | 2024 | 像素 (多视角) | 轨迹+地图 | ✗ |

## 6. 数学框架：驾驶世界模型的统一视角

### 6.1 统一形式化

所有驾驶世界模型都可以统一为：

$$
\hat{o}_{t+1:t+H} = \mathcal{W}_\theta(o_{1:t}, a_{t:t+H-1}, c)
$$

其中：

- $o_t$ 是观测（像素 / 点云 / 占用网格 / BEV）
- $a_t$ 是驾驶动作（速度、转向角）
- $c$ 是上下文（HD Map、天气、交通规则）
- $\hat{o}_{t+1:t+H}$ 是未来 $H$ 步的预测

### 6.2 用世界模型做规划

给定世界模型 $\mathcal{W}_\theta$，规划问题变为：

$$
a^*_{t:t+H-1} = \arg\max_{a_{t:t+H-1}} \mathbb{E}_{\hat{o} \sim \mathcal{W}_\theta}\left[\sum_{k=0}^{H-1} r(o_{t+k}, a_{t+k})\right]
$$

奖励函数 $r$ 包括：

- **安全**：$r_{\text{safe}} = -\mathbb{1}[\text{碰撞}]$（不碰撞）
- **舒适**：$r_{\text{comfort}} = -\|a_t - a_{t-1}\|^2$（动作平滑）
- **效率**：$r_{\text{progress}} = v_t \cos(\theta_t)$（沿目标方向前进）

### 6.3 Sim2Real 差距与 2025 年的因果性反思

世界模型作为仿真器的关键挑战在于 **Sim2Real Gap**：

$$
\text{Performance Gap} = |J(\pi, \text{Real}) - J(\pi, \text{WM})|
$$

其中 $J(\pi, \text{env})$ 是策略 $\pi$ 在环境 env 中的期望回报。传统的差距来源包括视觉差距（渲染不真实）和动力学差距（运动轨迹不准）。

**2025 年的理论突破：因果性与反事实推理的缺失**

最新的研究（如 2025 年的 *Beyond Simulation: Benchmarking World Models for Planning and Causality*）指出，当前自动驾驶世界模型面临的最致命差距并非视觉上的不真实，而是**因果推理（Causal Reasoning）和反事实生成（Counterfactual Generation）能力的缺失**。

- **统计相关性 vs 因果互动**：当前模型（如基于 Transformer 的 token 预测）主要依赖统计相关性。当测试车（Ego vehicle）改变动作时，世界模型往往无法正确预测周围车辆的**反应**（例如：我突然刹车，后车应该减速或变道）。研究发现，排名靠前的模型在重放原始轨迹时表现良好，但一旦引入扰动，预测就会崩溃。
- **反事实安全测试的失败**：为了生成"危险的边缘场景"（Corner cases），我们需要模型回答反事实问题："如果那辆车当时没有减速，会发生什么？" 2025 年的新框架（如基于因果交互图的生成式 BEV 世界模型）开始尝试显式建模智能体之间的动态依赖关系，通过最小干预让风险通过自然的交互传播自然涌现，而不是生硬地修改像素。

这表明，下一代自动驾驶世界模型必须从单纯的"视频预测器"进化为真正的"因果模拟器"。

## 7. 总结

| 路径 | 代表 | 优势 | 局限 |
|------|------|------|------|
| **像素预测** | GAIA-1, Vista | 视觉逼真，可直接端到端训练 | 3D 信息隐式，计算昂贵 |
| **结构化预测** | DriveDreamer | 精确可控，物理合理 | 依赖额外标注 |
| **占用预测** | OccWorld | 原生 3D，直接用于规划 | 无视觉输出，分辨率有限 |

自动驾驶世界模型正在从"展示酷炫视频"走向"实际替代仿真器训练规划器"。下一篇将总结世界模型所有路线，探讨它们的收敛趋势和未来方向。

> 参考资料：
>
> 1. Hu, A., ... & Kendall, A. (2023). *GAIA-1: A Generative World Model for Autonomous Driving*. arXiv:2309.17080.
> 2. Wang, X., ... & Zhao, H. (2023). *DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving*. ECCV 2024.
> 3. Gao, J., ... & Wang, N. (2024). *Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability*. NeurIPS 2024.
> 4. Zheng, W., ... & Lu, J. (2024). *OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving*. CVPR 2024.
> 5. (2025). *Beyond Simulation: Benchmarking World Models for Planning and Causality in Autonomous Driving*. arXiv:2508.01922. (关于世界模型因果性与反事实推理的最新研究)

> 下一篇：[笔记｜世界模型（七）：前沿与统一视角——五条路线的收敛](/chengYi-xun/posts/32-world-model-frontier/)
