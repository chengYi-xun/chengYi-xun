---
title: 笔记｜世界模型（五）：物理化的视频生成——让模型理解牛顿定律
date: 2026-04-06 00:50:00
categories:
 - Tutorials
tags:
 - PhysDreamer
 - PhysGen
 - NewtonGen
 - Physics-Grounded
 - World Model
series: "世界模型"
---

> **核心论文**：
> - PhysDreamer: arXiv:2404.13026 (ECCV 2024)
> - PhysGen: arXiv:2409.18964 (ECCV 2024)
> - NewtonGen: arXiv:2509.21309 (2025)
> - NewtonRewards: arXiv:2512.00425 (2025)
>
> **代码**：[stevenlsw/physgen](https://github.com/stevenlsw/physgen) · [pandayuanyu/NewtonGen](https://github.com/pandayuanyu/NewtonGen)
> **前置知识**：[上一篇：视频生成世界模型](posts/38-video-world-model/)

---

## 0. Sora 生成的球为什么不遵守牛顿定律？

用 Sora 生成一段"球从桌子上滚下来"的视频。你可能会看到：球的轨迹看起来大致合理，但仔细观察——球在桌边没有加速（无视重力），落地后弹跳角度不对（违反反射定律），甚至可能穿过桌面。

纯数据驱动的视频模型学到了"球通常会往下掉"这种**统计规律**，但没有学到 $F = ma$ 这个**物理定律**。在训练数据覆盖的常见场景中统计规律足够，但一旦进入未见过的物理场景（如月球上的低重力），模型就会失败。

本篇介绍四种将物理知识嵌入视频生成的方案——从显式物理仿真到可验证奖励。

---

## 1. PhysDreamer：从视频先验蒸馏物理属性

### 1.1 核心思想

PhysDreamer（MIT, 2024）将问题分为两步：

1. **从视频模型中"蒸馏"出物理属性**（如材质的弹性模量）
2. **用显式物理仿真器驱动运动**（Material Point Method）

这是"数据驱动的物理参数估计 + 基于物理的仿真"的混合方案。

### 1.2 方法

给定一个静态 3D 物体（用 3D Gaussian Splatting 表示），PhysDreamer 的流程：

**Step 1：物理属性估计**

对 3D 场景中的每个 Gaussian 预测一个材质参数向量 $\mathbf{m}_i = (\mu_i, \lambda_i)$（杨氏模量和泊松比）：

$$
\mathbf{m}_i = \text{MLP}_\theta(\mathbf{x}_i, \mathbf{f}_i)
$$

其中 $\mathbf{x}_i$ 是空间位置，$\mathbf{f}_i$ 是外观特征。

如何训练这个 MLP？用**视频扩散模型的得分函数**作为监督信号（Score Distillation）：

$$
\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}_{t, \epsilon}\left[w(t) \cdot (\hat{\epsilon}_\phi(z_t, t, c) - \epsilon) \cdot \frac{\partial z}{\partial \theta}\right]
$$

直觉：视频扩散模型"知道"物体在外力作用下应该如何运动（从训练视频中学到），PhysDreamer 利用这个知识来估计材质参数。

**Step 2：物理仿真**

用 **Material Point Method (MPM)** 进行可微物理仿真：

$$
\begin{aligned}
\mathbf{v}^{n+1}_p &= \mathbf{v}^n_p + \Delta t \cdot \left(\mathbf{g} + \frac{1}{m_p} \mathbf{f}^n_p\right) \\
\mathbf{x}^{n+1}_p &= \mathbf{x}^n_p + \Delta t \cdot \mathbf{v}^{n+1}_p
\end{aligned}
$$

其中 $\mathbf{g}$ 是重力加速度，$\mathbf{f}^n_p$ 是弹性力（由材质参数决定），$m_p$ 是质点质量。

弹性力通过变形梯度 $\mathbf{F}$ 和应力张量 $\boldsymbol{\sigma}$ 计算：

$$
\boldsymbol{\sigma} = \frac{1}{J}\frac{\partial \Psi}{\partial \mathbf{F}} \mathbf{F}^\top
$$

其中 $\Psi$ 是弹性势能密度（由杨氏模量 $E$ 和泊松比 $\nu$ 决定），$J = \det(\mathbf{F})$。

**Step 3：渲染**

将仿真后的粒子位置更新到 3D Gaussian，用可微渲染生成视频。

### 1.3 效果

PhysDreamer 能让静态 3D 物体在外力作用下产生**物理合理**的形变和运动。例如推一只布料做的兔子，它会像真正的布料一样变形和回弹。

局限：偏向**慢速、平滑**的运动（弹性形变），无法处理碰撞、断裂等复杂物理。

---

## 2. PhysGen：刚体物理 + 扩散视频

### 2.1 混合管线

PhysGen（UIUC, 2024）走了一条不同的路——**先仿真后生成**：

```
图像理解 → 物理参数提取 → 刚体仿真 → 视频扩散细化
```

### 2.2 三阶段流程

**阶段 1：场景理解**

从单张图像中提取物理信息：

$$
\{\text{几何}, \text{材质}, \text{物理参数}\} = \text{SceneParser}(\mathbf{I})
$$

包括物体的 3D 形状、质量、摩擦系数等。

**阶段 2：图像空间物理仿真**

在 2D 图像空间中运行简化的刚体动力学：

$$
\begin{aligned}
\mathbf{F} &= m\mathbf{a} \quad &\text{（牛顿第二定律）} \\
\boldsymbol{\tau} &= I\boldsymbol{\alpha} \quad &\text{（旋转动力学）} \\
\mathbf{v}' &= -e \cdot \mathbf{v}_n + \mu \cdot \mathbf{v}_t \quad &\text{（碰撞响应）}
\end{aligned}
$$

其中 $e$ 是恢复系数，$\mu$ 是摩擦系数。仿真输出是**粗糙但物理正确的**运动轨迹。

**阶段 3：视频扩散细化**

用视频扩散模型将粗糙的仿真结果（线条/简笔画级别）"美化"为逼真视频：

$$
V_{\text{realistic}} = \text{VideoDiffusion}(V_{\text{sim}}, \mathbf{I}_0, \text{prompt})
$$

### 2.3 数学：力条件控制

PhysGen 允许用户指定**作用力**来控制生成：

$$
\mathbf{F}_{\text{ext}} = (F_x, F_y), \quad \text{作用点} = (x_0, y_0)
$$

物理仿真器根据这些力计算运动轨迹，再交给视频模型渲染。

---

## 3. NewtonGen：Neural Newtonian Dynamics

### 3.1 核心创新

NewtonGen（2025）直接在视频扩散模型的**潜空间**中嵌入牛顿运动方程：

$$
\text{Neural Newtonian Dynamics (NND)}: \quad z_{t+1} = z_t + \Delta t \cdot v_t + \frac{1}{2} (\Delta t)^2 \cdot a_t
$$

其中 $z_t$ 是潜变量，$v_t$ 和 $a_t$ 是在潜空间中学到的"速度"和"加速度"。

### 3.2 架构

NewtonGen 在标准视频扩散模型中加入一个 **NND 模块**：

$$
\begin{aligned}
v_t &= \text{VelocityNet}_\psi(z_t, t) \\
a_t &= \text{AccelNet}_\psi(z_t, v_t, t, F_{\text{ext}}) \\
z_{t+1}^{\text{physics}} &= z_t + \Delta t \cdot v_t + \frac{(\Delta t)^2}{2} a_t \\
z_{t+1} &= (1 - \lambda) \cdot z_{t+1}^{\text{diffusion}} + \lambda \cdot z_{t+1}^{\text{physics}}
\end{aligned}
$$

$\lambda$ 控制物理约束的强度——$\lambda = 0$ 回退为标准扩散，$\lambda = 1$ 完全由物理主导。

### 3.3 可控物理参数

NewtonGen 支持用户指定物理参数来控制生成：

| 参数 | 含义 | 效果 |
|------|------|------|
| 重力 $g$ | 重力加速度 | 改变物体下落速度 |
| 恢复系数 $e$ | 弹性碰撞程度 | 控制弹跳高度 |
| 摩擦系数 $\mu$ | 表面摩擦 | 影响滑动速度 |
| 外力 $\mathbf{F}_{\text{ext}}$ | 施加的力 | 推动物体运动 |

---

## 4. NewtonRewards：用后训练奖励教模型物理

### 4.1 另一种思路

前面三种方法都在**训练/推理过程**中嵌入物理。NewtonRewards（2025）走了一条更简单的路：**后训练**——用可验证的物理奖励对已有视频模型做 RLHF。

### 4.2 可验证物理奖励

**速度奖励**：通过光流估计速度，检查是否满足匀加速运动：

$$
r_{\text{vel}} = -\sum_t \|v_t - (v_0 + a \cdot t)\|^2
$$

如果物体做自由落体，$a = g$，奖励鼓励模型生成符合 $v = v_0 + gt$ 的运动。

**质量一致性奖励**：检测到的物体应有一致的质量表现：

$$
r_{\text{mass}} = -\text{Var}\left(\frac{F_i}{a_i}\right)_{i=1}^{N}
$$

如果同一个物体在不同碰撞中表现出不同的质量（$m = F/a$ 不一致），给低奖励。

**综合奖励**：

$$
r = w_1 r_{\text{vel}} + w_2 r_{\text{mass}} + w_3 r_{\text{quality}}
$$

### 4.3 后训练流程

1. 用已有视频模型生成 $N$ 个候选视频
2. 计算每个视频的物理奖励
3. 用 GRPO/DPO 等方法优化模型偏好

### 4.4 NewtonBench-60K

NewtonRewards 附带了一个评测基准——60K 个视频，标注了物理正确性：

| 类别 | 物理现象 | 示例 |
|------|---------|------|
| 自由落体 | $y = \frac{1}{2}gt^2$ | 球从高处落下 |
| 碰撞 | 动量守恒 | 台球碰撞 |
| 斜面运动 | $a = g\sin\theta$ | 物体沿斜面滑下 |
| 抛体运动 | 抛物线轨迹 | 投篮 |

---

## 5. 四种方案对比

| 方案 | 物理嵌入方式 | 优点 | 局限 |
|------|------------|------|------|
| **PhysDreamer** | 蒸馏材质参数 + MPM 仿真 | 物理准确的3D形变 | 仅弹性体，速度慢 |
| **PhysGen** | 先仿真后生成 | 力控制，刚体物理 | 仿真粗糙，仅2D |
| **NewtonGen** | 潜空间牛顿方程 | 端到端，参数可控 | 物理近似 |
| **NewtonRewards** | 后训练物理奖励 | 即插即用，不改架构 | 依赖奖励质量 |

---

## 6. 物理约束的数学框架

### 6.1 守恒律作为损失项

物理中最基本的约束是守恒律。可以将其作为正则化项加入训练损失：

**能量守恒**：

$$
\mathcal{L}_{\text{energy}} = \left|\sum_i \frac{1}{2}m_i v_i^2(t) + m_i g h_i(t) - E_0\right|^2
$$

**动量守恒**（无外力时）：

$$
\mathcal{L}_{\text{momentum}} = \left\|\sum_i m_i \mathbf{v}_i(t) - \mathbf{p}_0\right\|^2
$$

### 6.2 PINN 式混合动力学

Physics-Informed Neural Networks（PINN）的核心思想：让神经网络的输出满足物理 PDE。

对于视频世界模型，可以要求：

$$
\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{data}}}_{\text{拟合数据}} + \alpha \underbrace{\mathcal{L}_{\text{PDE}}}_{\text{满足物理方程}}
$$

例如，对流体场景，$\mathcal{L}_{\text{PDE}}$ 可以是 Navier-Stokes 方程的残差：

$$
\mathcal{L}_{\text{NS}} = \left\|\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} + \frac{1}{\rho}\nabla p - \nu \nabla^2 \mathbf{u}\right\|^2
$$

---

## 7. 总结与展望

物理化的视频生成代表了世界模型领域最重要的交叉方向之一——**数据驱动的学习**与**物理知识的注入**并非对立，而是互补。

| 趋势 | 含义 |
|------|------|
| 仿真→生成 | 用物理仿真器提供骨架，用生成模型填充视觉细节 |
| 奖励→对齐 | 用可验证的物理奖励"教"模型遵守物理定律 |
| 显式→隐式 | 从手写方程到可微仿真器到潜空间约束 |

下一篇将介绍世界模型在**自动驾驶**中的专业化应用——从像素级预测到 3D 占用预测。

> **下一篇**：[笔记｜世界模型（六）：自动驾驶世界模型——从视频预测到占用预测](posts/40-driving-world-model/)

---

**参考文献**

1. Zhang, T., et al. (2024). *PhysDreamer: Physics-Based Interaction with 3D Objects via Video Generation*. ECCV 2024.
2. Liu, S., et al. (2024). *PhysGen: Rigid-Body Physics-Grounded Image-to-Video Generation*. ECCV 2024.
3. Pan, Y., et al. (2025). *NewtonGen: Physics-Consistent and Controllable Text-to-Video Generation via Neural Newtonian Dynamics*. arXiv:2509.21309.
4. Duan, H., et al. (2025). *What about gravity in video generation? Post-Training Newton's Laws with Verifiable Rewards*. arXiv:2512.00425.
