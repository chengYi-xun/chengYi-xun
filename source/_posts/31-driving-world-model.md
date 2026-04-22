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

> **核心论文**：GAIA-1 (arXiv:2309.17080, 2023)、DriveDreamer (arXiv:2309.09777, ECCV 2024)、Vista (arXiv:2405.17398, NeurIPS 2024)、OccWorld (arXiv:2311.16038, ECCV 2024)
>
> **前置知识**：[上一篇：物理化世界模型](/chengYi-xun/posts/30-physics-world-model/)
>
> ⬅️ 上一篇：[笔记｜世界模型（五）：物理化的视频生成——让模型理解牛顿定律](/chengYi-xun/posts/30-physics-world-model/)
>
> ➡️ 下一篇：[笔记｜世界模型（七）：前沿与统一视角——五条路线的收敛](/chengYi-xun/posts/32-world-model-frontier/)

## 0. 为什么自动驾驶特别需要世界模型？

上一篇讨论了如何让视频模型遵守物理定律。但在所有世界模型的应用中，**自动驾驶是对物理准确性和安全性要求最高的领域**——这里的"物理错误"不是视觉瑕疵，而是可能导致事故的决策失误。

假设你在开发自动驾驶系统，需要测试"行人突然从路边冲出"的场景：

1. **真实路测**：等这种场景自然发生——几万公里都遇不到一次，有安全风险
2. **传统仿真器**（CARLA 等）：画面和真实世界差距太大（domain gap），训练出的模型可能无法迁移

世界模型提供了**第三条路**：从真实驾驶视频中学习环境动力学，生成无限的、逼真的、可控的驾驶场景。

自动驾驶对世界模型有三个独特的需求：

1. **3D 几何**：必须理解三维空间——深度、遮挡、多视角一致性
2. **精确可控**：必须能根据驾驶指令（转弯、加速、变道）生成对应场景
3. **安全可靠**：生成的场景要用于安全决策，不能有物理错误

## 1. GAIA-1：最早的驾驶世界模型

### 1.1 核心思想

GAIA-1（Wayve, 2023）是最早的大规模驾驶世界模型之一。核心假设：**用足够大的 Transformer 和足够多的驾驶视频，模型会隐式学到驾驶场景的 3D 结构和运动规律。**

### 1.2 架构：两阶段 token 预测 + 扩散解码

GAIA-1 的架构包含两大组件：

1. **World Model（6.5B 参数）**：自回归 Transformer，接收多模态 token 序列，预测未来视频 token
2. **Video Decoder（2.6B 参数）**：视频扩散模型，将离散 token 解码为连续视频帧

```
Video Encoder → World Model (Transformer, 6.5B) → Video Decoder (Diffusion, 2.6B)
     ↑                    ↑
  驾驶视频              条件输入
                   (文本 / 动作 / 地图)
```

**多模态 token 化**：GAIA-1 将所有输入统一为 token 序列（类似 LLM 处理文本），然后做 next-token prediction：

$$
\begin{aligned}
\mathbf{v}_t &= \text{VQ-VAE}(I_t) \quad &\text{（视频帧 → 离散 token）} \\
\mathbf{a}_t &= \text{ActionEncoder}(\text{speed}_t, \text{steer}_t) \quad &\text{（连续动作 → token）} \\
\mathbf{c} &= \text{TextEncoder}(\text{prompt}) \quad &\text{（自然语言 → token）}
\end{aligned}
$$

**自回归预测**（与 GPT 完全类似，只是预测的是视频 token 而非文字 token）：

$$
p(\mathbf{v}_{t+1} \mid \mathbf{v}_{\leq t}, \mathbf{a}_{\leq t}, \mathbf{c})
$$

### 1.3 规模与涌现

| 配置 | 值 |
|------|-----|
| 总参数量 | 9B（World Model 6.5B + Video Decoder 2.6B） |
| 训练数据 | 伦敦城市驾驶视频，约 4700 小时（2019-2023 年采集） |
| 分辨率 | 288 × 512 |
| 预测时长 | 数秒 |

当模型规模达到 9B 后，GAIA-1 展示了几个有趣的**涌现行为**——这些能力并非显式训练，而是从大规模数据中自动习得：

- **3D 几何理解**：改变车速时，场景以正确的透视变化展开（近处物体移动快，远处慢）
- **交通规则**：红灯时周围车辆自动停车，绿灯后恢复
- **天气变化**：通过文本控制可切换晴天/雨天/雪天，光影随之变化
- **长程一致性**：道路和建筑在数秒的生成中保持几何一致

> **启发**：GAIA-1 的核心哲学是"暴力出奇迹"——用足够大的 Transformer 和足够多的数据，让 3D 理解和交通规则从 2D 视频中自然涌现。这与 LLM 从文本中涌现推理能力的思路一脉相承。但它的局限也很明显：涌现不可控、物理正确性没有保证、且完全依赖闭源数据。

## 2. DriveDreamer：结构化控制的驾驶生成

### 2.1 核心改进

GAIA-1 的问题：场景可控性差。你能控制车速和天气，但无法精确指定"前方 50 米处有一辆卡车在变道"。

DriveDreamer（GigaAI & 清华, ECCV 2024）的核心改进：**用结构化的中间表示（HD Map + 3D Bbox + 交通元素）替代纯视频自回归，实现精确的场景级控制。**

思路类比：GAIA-1 像是"给一个画家一段语音描述，让他凭感觉画"，DriveDreamer 则是"先画好场景的线稿（结构化布局），再交给画家上色（视频渲染）"。

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

### 2.4 DriveDreamer-2：LLM 作为交通导演

DriveDreamer-2（2024）进一步引入 **LLM 作为交通管理器**——用语言模型的常识推理能力生成合理的交通行为：

$$
\text{LLM}(\text{当前场景描述}) \to \text{未来交通参与者轨迹}
$$

例如输入"前方路口有红灯，左侧车道有一辆公交车"，LLM 推理出"前车应减速停车，公交车缓慢靠站"，生成对应轨迹后交给视频模型渲染。

> **DriveDreamer 系列的启发**：将场景生成拆解为"理解 → 布局 → 渲染"三步，每一步都可以独立优化。结构化中间表示是精确可控的关键，但代价是需要 HD Map、3D 标注等额外数据。

## 3. Vista：通用驾驶世界模型

### 3.1 定位

GAIA-1 只在伦敦数据上训练，DriveDreamer 依赖 nuScenes 数据集。**能否构建一个跨数据集、跨场景的通用驾驶世界模型？**

Vista（OpenDriveLab / 上海AI Lab, NeurIPS 2024）试图回答这个问题。

### 3.2 三个关键技术

**（1）动态分辨率训练**：不同车型的摄像头分辨率和宽高比各异，Vista 通过多分辨率联合训练来适配各种配置：

$$
I_{\text{train}} \in \{480p, 720p, 1080p\} \times \{4:3, 16:9, 21:9\}
$$

**（2）多模态动作控制**：支持从高层意图（"左转"、目标点）到底层控制（轨迹、速度、转向角）的多种格式：

$$
\hat{I}_{t+1:t+K} = \text{Vista}(I_{1:t}, \underbrace{(\text{speed}, \text{steer}, \text{curvature})}_{a_{t:t+K}})
$$

**（3）长时预测**：通过自回归 rollout 生成数十秒的驾驶视频。关键技巧是**潜在替换（latent replacement）**——将历史帧作为先验注入，缓解误差累积。

### 3.3 作为奖励函数评估驾驶策略

Vista 的一个独特贡献：**首次利用世界模型自身的能力建立通用奖励函数**，无需真值数据即可评估真实世界中的驾驶动作质量。

传统流程：

$$
\text{真实数据} \overset{\text{训练}}{\longrightarrow} \pi_\theta \overset{\text{部署}}{\longrightarrow} \text{真实车辆}
$$

Vista 增强流程：

$$
\text{真实数据} \overset{\text{训练 WM}}{\longrightarrow} \text{Vista} \overset{\text{生成数据 + 奖励信号}}{\longrightarrow} \pi_\theta \overset{\text{部署}}{\longrightarrow} \text{真实车辆}
$$

在 nuScenes 验证集上，Vista 相比此前最优方法 FID 提升 **55%**，FVD 提升 **27%**；在跨数据集泛化（nuScenes、Waymo、CODA）中，超过 **70%** 的对比优于通用视频生成器。

## 4. OccWorld：3D 占用空间的世界模型

### 4.1 从像素到体素

前面的 GAIA-1、DriveDreamer、Vista 都在**像素空间**（2D 图像/视频）做预测。但自动驾驶的下游任务（规划、避障）需要的是 **3D 空间信息**——"前方 30 米处有没有障碍物？" 而不是"这个像素是什么颜色？"

OccWorld（清华, ECCV 2024）走了完全不同的路：**直接在 3D 占用空间中预测世界变化，跳过像素生成。**

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

| 模型 | 机构 | 会议/年份 | 预测空间 | 控制条件 | 开源 |
|------|------|---------|---------|---------|------|
| GAIA-1 | Wayve | arXiv 2023 | 像素 (2D) | 动作+文本 | ✗ |
| DriveDreamer | GigaAI/清华 | ECCV 2024 | 像素 (2D) | 3D Box+HDMap | ✓ |
| DriveDreamer-2 | GigaAI/清华 | 2024 | 像素 (2D) | LLM 交通管理 | ✓ |
| Vista | OpenDriveLab/上海AI Lab | NeurIPS 2024 | 像素 (2D) | 多模态动作 | ✓ |
| OccWorld | 清华 | ECCV 2024 | 3D 占用 | 自车动作 | ✓ |
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

### 6.3 Sim2Real 差距与因果性反思

世界模型作为仿真器的关键挑战在于 **Sim2Real Gap**：

$$
\text{Performance Gap} = |J(\pi, \text{Real}) - J(\pi, \text{WM})|
$$

其中 $J(\pi, \text{env})$ 是策略 $\pi$ 在环境 env 中的期望回报。传统的差距来源包括视觉差距（渲染不真实）和动力学差距（运动轨迹不准）。

**深层问题：因果推理能力的缺失**

最新的研究指出，当前自动驾驶世界模型面临的**最致命差距并非视觉上的不真实，而是因果推理（Causal Reasoning）和反事实生成（Counterfactual Generation）能力的缺失**。

用 Pearl 的因果层级来理解：

| 层级 | 能力 | 示例 | 当前世界模型 |
|------|------|------|------------|
| Level 1: 关联 $p(y \mid x)$ | 观察到 X 后预测 Y | "前车减速了，我也该减速" | ✓ 可以做到 |
| Level 2: 干预 $p(y \mid do(x))$ | 改变 X 后预测 Y | "如果我突然变道，后车会怎样反应？" | △ 偶尔涌现 |
| Level 3: 反事实 $p(y_x \mid x', y')$ | 回溯推理 | "如果那辆车当时没有减速，会发生碰撞吗？" | ✗ 做不到 |

当前模型主要依赖统计相关性：在重放原始轨迹时表现良好，但一旦引入扰动（改变自车动作），预测就会崩溃——因为它没有学到"我的动作会因果性地影响其他车辆的行为"。

这表明，下一代自动驾驶世界模型必须从"视频预测器"进化为"因果模拟器"。

## 7. 总结

| 路径 | 代表 | 核心思路 | 优势 | 局限 |
|------|------|---------|------|------|
| **像素预测** | GAIA-1, Vista | 预测未来视频帧 | 视觉逼真，可直接端到端 | 3D 信息隐式，计算昂贵 |
| **结构化预测** | DriveDreamer | 先预测布局再渲染 | 精确可控，物理合理 | 依赖 HD Map 等标注 |
| **占用预测** | OccWorld | 预测 3D 体素变化 | 原生 3D，直接用于规划 | 无视觉输出，分辨率有限 |

回顾本篇四个模型，可以看到自动驾驶世界模型的演进脉络：

1. **GAIA-1** 证明了大规模 Transformer + 视频数据可以涌现 3D 理解
2. **DriveDreamer** 引入结构化中间表示，解决了精确可控性
3. **Vista** 追求跨数据集的泛化能力，并首创世界模型作为奖励函数
4. **OccWorld** 跳出像素空间，直接在 3D 占用空间做预测

下一步的挑战在于**因果推理**：当前模型只能做统计性的"回放预测"，无法真正模拟"如果我做了不同的动作，世界会如何响应"。

> 参考资料：
>
> 1. Hu, A., Russell, L., Yeo, H., ... & Kendall, A. (2023). *GAIA-1: A Generative World Model for Autonomous Driving*. arXiv:2309.17080.
> 2. Wang, X., Zhu, Z., Huang, G., Chen, X., Zhu, J. & Lu, J. (2023). *DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving*. ECCV 2024.
> 3. Gao, S., Yang, J., Chen, L., Chitta, K., Qiu, Y., Geiger, A., Zhang, J. & Li, H. (2024). *Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability*. NeurIPS 2024.
> 4. Zheng, W., Chen, W., Huang, Y., Zhang, B., Duan, Y. & Lu, J. (2024). *OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving*. ECCV 2024.
> 5. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.

> 下一篇：[笔记｜世界模型（七）：前沿与统一视角——五条路线的收敛](/chengYi-xun/posts/32-world-model-frontier/)
