---
title: 笔记｜强化学习（九）：DanceGRPO 与 MixGRPO——视觉生成 GRPO 的扩展与加速
date: 2026-04-05 16:00:00
cover: false
mathjax: true
categories:
 - Notes
tags:
 - Deep learning
 - Generative models theory
 - Reinforcement Learning
 - Flow Matching
series: Diffusion Models theory
---

> Flow-GRPO 证明了 GRPO 在图像生成上的有效性，但留下了两个方向的空白：**任务维度**（能否推广到视频？）和**效率维度**（全轨迹 SDE 的开销能否降低？）。
>
> 本文讲清楚两篇一脉相承的工作：DanceGRPO 解决了"广度"问题，将 GRPO 统一到 Diffusion + Flow Matching 双范式和图像+视频双模态；MixGRPO 解决了"效率"问题，用混合 ODE-SDE 和滑动窗口将训练时间砍掉 50%~71%。
>
> ⬅️ 上一篇：[笔记｜强化学习（八）：SuperFlow 与图像生成 RL 前沿（2026）](/chengYi-xun/posts/58-superflow/)
>
> ➡️ 下一篇：[笔记｜强化学习（十）：LLM 对齐中的 RL 方法全景对比——从 PPO 到 SuperFlow](/chengYi-xun/posts/60-rl-alignment-comparison/)
>
> 论文：
> - [DanceGRPO: Unleashing GRPO on Visual Generation](https://arxiv.org/abs/2505.07818)（ByteDance, 2025）
> - [MixGRPO: Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE](https://arxiv.org/abs/2507.21802)（Tencent Hunyuan, 2026）

---

## Part I: DanceGRPO 的理论分析

### 出发点：Flow-GRPO 留下了哪些空白

Flow-GRPO（前文已详细讨论）首次将 GRPO 引入 Flow Matching 模型，通过 ODE→SDE 转换实现了随机探索。它的理论框架（论文 Eq. 7）适用于 Flow Matching 模型全家族，而非仅限于 Rectified Flow 一种实例。但它存在两个显著局限：

{% note info no-icon %}
**Flow Matching 与 Rectified Flow 的最核心区别**：
简单来说，**Flow Matching 是一套通用的理论框架（“接口”）**，它告诉你如何通过回归速度场（Velocity Field）来训练连续正规化流（CNF），但它**不限制**数据到噪声的概率路径长什么样（可以是弯曲的、复杂的）。

除了 Rectified Flow，Flow Matching 框架下还有其他几种著名的实现（“实例”）：
- **Diffusion Flow Matching**：它故意把路径设计成和传统 DDPM（如 VP 或 VE 路径）一模一样，从而证明了 Diffusion 模型只是 Flow Matching 的一种特殊路径。
- **Optimal Transport Flow Matching (OT-CFM)**：它在训练时通过最优传输（Optimal Transport）算法，强制让每个噪声点去寻找离它“最近”的真实图像进行配对，从而在源头上减少路径的交叉。

而 **Rectified Flow（RF）是这个框架下最简单、最暴力的一个具体实现**。它的核心特征有两个：

1. **直线路径**：它强制规定从数据 $x_0$ 到噪声 $x_1 \sim \mathcal{N}(0, I)$ 的路径必须是最简单的直线：$x_t = (1-t)x_0 + t x_1$。
2. **独有的 Reflow 机制（从数学角度理解）**：
   - **问题**：在最初训练 RF 时（称为 1-Rectified Flow），噪声 $x_1$ 和图像 $x_0$ 是**独立随机配对**的。这导致不同图像的生成轨迹在空间中会发生**严重交叉**。一旦轨迹交叉，速度场就会变得极度非线性，ODE 求解器必须用非常小的步长（很多步）才能准确积分。
   - **Reflow 解决之道**：Reflow 是一种“用魔法打败魔法”的迭代微调技术。它先用训练好的 1-RF 模型生成一批图像，记录下每个噪声 $x_1$ 和它最终生成的图像 $x_0$ 的**确定性映射对**。然后，用这些**已经绑定好**的 $(x_1, x_0)$ 数据对去训练一个新的模型（2-Rectified Flow）。
   - **数学本质**：这个过程在数学上等价于在寻找**最优传输（Optimal Transport）**的解。通过 Reflow，原本交叉、弯曲的生成轨迹被彻底“拉直”成了互不交叉的平行直线。轨迹越直，ODE 积分误差越小，最终甚至只需要 1 步 Euler 步进（$x_0 = x_1 - v_\theta$）就能生成高质量图像。

当前最前沿的图像/视频大模型（如 Stable Diffusion 3、FLUX、HunyuanVideo）之所以快，正是因为它们采用了 Rectified Flow 的直线路径。因此，本文讨论的算法（如 Flow-GRPO）虽然在理论上适用于所有 Flow Matching 模型，但在工程落地时，都是在 Rectified Flow 的架构上进行验证的（详见[《笔记｜生成模型（十三）：Flow Matching理论与实现》](/chengYi-xun/posts/13-flow_matching/)）。
{% endnote %}

1. **范式局限**：Flow-GRPO 的 ODE→SDE 转换要求模型预测速度场 $v_t$，这是 Flow Matching 模型的特征。传统 Diffusion 模型（如 SD v1.4）使用噪声预测 $\epsilon$-prediction 和不同的前向过程（$z_t = \alpha_t x + \sigma_t \epsilon$），其 SDE 形式（Langevin 动力学）与 Flow Matching 的 SDE 在数学结构上不同。Flow-GRPO 未对 Diffusion 模型进行适配和验证。
2. **模态局限**：仅处理文生图（T2I），未触及视频生成（T2V、I2V），而视频的高维度、高算力需求和更复杂的评价标准是全新的挑战。
3. **方法局限**：先前的 RL 方法（DDPO、DPOK）在大规模 Prompt 集上训练不稳定，且未在 Flow Matching 模型上被充分验证。

DanceGRPO 的出发点就是：**能否用一个统一的 GRPO 框架，同时覆盖 Diffusion 和 Flow Matching（含 Rectified Flow）两种范式、图像和视频两种模态？**

### 关键贡献一：统一的 SDE 公式——从 Flow Matching 到 Diffusion

DanceGRPO 的第一个理论贡献是推导出覆盖**传统 Diffusion 模型**和**Flow Matching 模型**（含 Rectified Flow）的**统一 SDE 框架**。

**Diffusion Model** 的反向 SDE（基于 Score-based SDE 理论 [Song et al., 2020]）：

$$dz_t = \left(f_t z_t - \frac{1+\varepsilon_t^2}{2}g_t^2\nabla\log p_t(z_t)\right)dt + \varepsilon_t g_t\,dw$$

**Flow Matching / Rectified Flow** 的反向 SDE（基于 Stochastic Interpolant 理论 [Albergo et al., 2023]）：

$$dz_t = \left(u_t - \frac{1}{2}\varepsilon_t^2\nabla\log p_t(z_t)\right)dt + \varepsilon_t\,dw$$

DanceGRPO 通过论文 Appendix B 的详细推导，借助 Stochastic Interpolant 框架建立了两者在数学结构上的等价性。具体来说，Diffusion 模型的插值路径 $z_t = \alpha_t x + \sigma_t \epsilon$ 和 Rectified Flow 的线性插值 $z_t = (1-t)x + t\epsilon$ 都是 Stochastic Interpolant $z_t = \alpha(t)z_0 + \beta(t)z_1 + \gamma(t)\epsilon$ 的特例。如果设定 Diffusion 模型的系数 $f_t = \partial_t\log\alpha_t$ 和 $g_t^2 = 2\alpha_t\sigma_t\partial_t(\sigma_t/\alpha_t)$，并匹配噪声项 $\eta_t = \varepsilon_t / g_t$，则两种 SDE 产生**完全相同形式的策略概率** $\pi(a_t | s_t)$。

这种统一性通过一个优雅的公式体现：

$$\tilde{z}_s = \tilde{z}_t + \text{网络输出} \cdot (\eta_s - \eta_t)$$

对 $\varepsilon$-prediction（Diffusion），$\tilde{z} = z/\alpha$，$\eta = \sigma/\alpha$；对速度预测（Flow Matching / Rectified Flow），$\tilde{z} = z$，$\eta = t$。**这意味着同一套 GRPO 算法逻辑可以不修改地应用于两种范式**——DanceGRPO 是首个在 SD v1.4（Diffusion）和 FLUX/HunyuanVideo（Flow Matching）上统一验证的 GRPO 框架。

### 关键贡献二：面向视频的多维奖励机制

视频的评价维度远比图像复杂。DanceGRPO 使用 VideoAlign 作为奖励模型，它评估三个独立维度：
- **VQ（Visual Quality）**：画面质量、美学水平
- **MQ（Motion Quality）**：运动流畅度、物理合理性
- **TA（Text Alignment）**：文本-视频对齐度（实验中发现不稳定，被排除）

如果将 VQ 和 MQ 简单加权求和后再做组内标准化，VQ 的绝对值范围（约 [4, 8]）远大于 MQ（约 [0, 4]），会导致**尺度淹没**。DanceGRPO 提出**分项标准化**：

$$\hat{A}_i^{(d)} = \frac{r_i^{(d)} - \bar{r}_{\text{group}}^{(d)}}{\sigma_{\text{group}}^{(d)} + \epsilon}, \quad d \in \{\text{VQ}, \text{MQ}\}$$

每个维度**独立计算组内优势**后，各自应用 PPO clip loss，最后加权合并：$L = \alpha_\text{VQ} \cdot L^\text{(VQ)} + \alpha_\text{MQ} \cdot L^\text{(MQ)}$。

### 关键贡献三：共享初始噪声与训练稳定性

Flow-GRPO 对同一 Prompt 的 $G$ 个样本使用**不同的初始噪声** $x_T^{(i)} \sim \mathcal{N}(0, I)$，以增加探索多样性。其消融实验（论文 Fig. 10）证实这在图像生成中是有益的。

但 DanceGRPO 发现，在视频生成的高维空间中，不同初始噪声会导致严重的 **reward hacking**（论文 Fig. 8）——模型找到某些噪声模式能"欺骗"奖励模型，产生视觉上不合理但分数虚高的视频。

DanceGRPO 改为：同一 Prompt 的所有 $G$ 个样本共享**同一个初始噪声**，差异仅来自 SDE 过程中各步注入的随机噪声。这使得组内对比更加公平——差异反映的是**去噪策略的好坏**，而非初始条件的不同。

{% note warning no-icon %}
**Flow-GRPO vs DanceGRPO 的初始噪声策略对立**是一个值得注意的现象。矛盾的根源在于维度差异：视频 latent（如 $16 \times 14 \times 60 \times 60$）的维度远高于图像（如 $16 \times 64 \times 64$），不同初始噪声产生的轨迹差异过大，组内对比的信号被噪声淹没，优势估计方差激增。
{% endnote %}

### 关键贡献四：时间步随机抽样

完整的去噪轨迹需要 $T$ 步（如 25~50 步）。如果每步都反向传播计算梯度，显存需求将不可承受，尤其是视频任务。DanceGRPO 提出**随机子采样**：

1. 用 SDE 完整走完 $T$ 步采样（收集所有 log\_prob）
2. 随机打乱时间步顺序
3. 只取前 $K = \lfloor T \times \tau \rfloor$ 步进行梯度更新（默认 $\tau = 0.6$）
4. 每步单独 backward，立即释放计算图

论文 Fig. 4(b) 的消融实验揭示了一个关键规律：**前 30% 时间步（靠近噪声端、构图阶段）贡献最大**，后期步贡献递减。随机 60% 可以接近全量的性能。

### DanceGRPO 的实验成果

![DanceGRPO 图像生成奖励曲线](/chengYi-xun/img/dancegrpo_reward_t2i.jpg)

| 任务 | 基线 | DanceGRPO | 提升 |
|:---|:---|:---|:---|
| T2I HPS-v2.1（FLUX） | 0.304 | 0.372 | +22% |
| T2I CLIP Score（FLUX） | 0.405 | 0.427 | +5% |
| T2V VQ（HunyuanVideo） | 4.51 | 7.03 | +56% |
| T2V MQ（HunyuanVideo） | 1.37 | 3.85 | **+181%** |
| I2V MQ（SkyReels-I2V） | — | — | +118% |

DanceGRPO 在四个基础模型（SD v1.4、FLUX、HunyuanVideo、SkyReels-I2V）上均表现稳定，人类评估中 RLHF 优化后的模型在 T2I、T2V、I2V 三个任务上均被一致偏好。

与 DDPO、DPOK、ReFL、DPO 等方法相比，DanceGRPO 是唯一同时满足以下全部条件的方法：RL-based + 视频支持 + 大规模数据可扩展 + 显著奖励提升 + 同时兼容 Diffusion 和 Flow Matching（Rectified Flow）+ 不要求可微奖励。

### DanceGRPO 的遗留问题

尽管贡献显著，DanceGRPO 存在三个核心瓶颈——它们是 MixGRPO 的直接动机：

**问题一：全轨迹 SDE 的计算浪费。** DanceGRPO 在**所有** $T$ 步都注入 SDE 噪声，即使大部分步不参与优化。这导致 $\pi_{\theta_\text{old}}$ 的推理必须走完全部 $T$ 步 SDE，无法使用高阶 ODE 求解器（如 DPM-Solver++）加速。

**问题二：随机抽样的性能-效率困境。** 采样阶段对所有步注入了 SDE 噪声，但训练阶段只优化其中 $K$ 步。**未被选中的步的策略偏移没有被纠正。** 当优化步数从 14 降到 4 时，ImageReward 从 1.436 降到 1.335（下降 7%）。MixGRPO 原文评价：*"this approach does not fundamentally address the issue."*

**问题三：梯度冲突。** 前期步（构图/大结构）和后期步（细节/纹理）的梯度方向相互冲突，被随机混合在同一 batch 中优化时效率降低。

三个问题的共同根源：**在不需要探索的时间步上注入了不必要的随机性。**

---

## Part II: MixGRPO 的理论分析

### 出发点：重新审视 MDP 的范围

MixGRPO 从一个根本性的观察出发：**Flow-GRPO 和 DanceGRPO 将整条去噪轨迹 $(s_0, a_0, \ldots, s_T, a_T)$ 建模为 MDP，但并非所有步都需要是随机的。** ODE 步是确定性的，不产生探索——它们在 MDP 中贡献零信息量但消耗大量计算。

因此，MixGRPO 提出：**将 MDP 缩短到仅包含 SDE 步的子区间**，其余步退化为确定性 ODE。

### 关键贡献一：混合 ODE-SDE 采样

![MixGRPO 方法概览](/chengYi-xun/img/mixgrpo_method1.png)

MixGRPO 在去噪时间线上定义一个子区间 $S = [t_l, t_r) \subseteq [0, 1)$，在 $S$ 内用 SDE，在 $S$ 外用 ODE：

$$dx_t = \begin{cases} \left(v_\theta - \frac{1}{2}\sigma_t^2\,s_t(x_t)\right)dt + \sigma_t\,dw, & t \in S \\ v_\theta\,dt, & \text{otherwise} \end{cases}$$

其中 Score Function $s_t(x_t) = -\frac{x_t}{t} - \frac{1-t}{t}v_\theta(x_t, t)$ 由 Tweedie 公式给出。

Euler-Maruyama 离散化后：

$$x_{t+\Delta t} = \begin{cases} x_t + \mu_\theta(x_t, t)\Delta t + \sigma_t\sqrt{\Delta t}\,\epsilon, & t \in S \\ x_t + v_\theta(x_t, t)\Delta t, & \text{otherwise} \end{cases}$$

SDE 漂移项 $\mu_\theta = v_\theta + \frac{\sigma_t^2}{2t}(x_t + (1-t)v_\theta)$。

GRPO 优化的范围缩小为仅 $S$ 内的步：

$$J_\text{MixGRPO}(\theta) = \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^{N}\frac{1}{|S|}\sum_{t\in S}\min\left(r_t^i\hat{A}^i,\ \text{clip}(r_t^i, 1-\varepsilon, 1+\varepsilon)\hat{A}^i\right) - \beta J_\text{KL}\right]$$

**收敛性保证：** MixGRPO 在 Supplementary 中严格证明了混合 ODE-SDE 采样与纯 ODE 采样产生**完全相同的边缘分布**。证明利用 Kolmogorov 方程（Fokker-Planck 方程）：SDE 段的分布演化为

$$\frac{\partial q_t}{\partial t} = -\nabla\cdot\left[(f - g^2 s_t)\,q_t\right] + \frac{1}{2}g^2\nabla^2 q_t$$

展开后 Score 扩散项与噪声扩散项精确对消，得到与 ODE 的连续性方程 $\frac{\partial q_t}{\partial t} = -\nabla\cdot[f_\text{ODE}\,q_t]$ 完全一致的形式。因此，在任意位置切换 SDE↔ODE 不影响最终分布。

### 关键贡献二：滑动窗口作为优化调度器

$S$ 不是固定的——它是一个**滑动窗口** $W(l) = \{t_l, t_{l+1}, \ldots, t_{l+w-1}\}$，随训练进行从低 SNR 向高 SNR 移动：

$$l \leftarrow \min(l + s,\; T - w) \quad \text{every } \tau \text{ iterations}$$

这等价于一种**隐式课程学习**（analogous to temporal discounting in RL）：

| 训练阶段 | 窗口位置 | 优化内容 | 探索空间 |
|:---|:---|:---|:---|
| 早期 | 低 SNR（$l \approx 0$） | 全局构图、物体布局 | 大（高随机性） |
| 后期 | 高 SNR（$l \rightarrow T-w$） | 纹理细节、色彩精修 | 小（低随机性） |

MixGRPO 论文的消融实验验证了这种直觉：

| 窗口策略 | HPS-v2.1 | ImageReward |
|:---|:---:|:---:|
| Frozen（固定在开头） | 0.354 | 1.580 |
| Random（每轮随机位置） | 0.365 | 1.513 |
| Progressive + Constant τ | **0.367** | 1.629 |
| Progressive + Exp Decay τ | 0.360 | **1.632** |

Progressive 一致优于 Random，**验证了课程学习优于随机抽样的假设**。更值得注意的是，即使 Frozen 策略（仅优化最前面几步）也超过了 DanceGRPO（NFE=4 时的 1.335），说明**定向优化前几步比全轨迹 SDE + 随机抽样更有效**。

MixGRPO 还提出**指数衰减调度**：$\tau(l) = \tau_0 \cdot \exp(-k \cdot \text{ReLU}(l - \lambda_\text{thr}))$，让模型在影响最大的构图阶段停留更久。

### 关键贡献三：MixGRPO-Flash 高阶求解器加速

MixGRPO 的混合策略带来一个额外收益：窗口外的 ODE 步可以用**高阶求解器**加速。MixGRPO 将 DPM-Solver++（二阶多步法）适配到 Flow Matching（Rectified Flow）框架：

利用 $\hat{x}_0 = x_t - v_\theta \cdot t$（Flow Matching 中由速度场反推的 $x_0$-prediction 形式），二阶校正公式为：

$$D_i = \left(1 + \frac{h_i}{2h_{i-1}}\right)\hat{x}_0^{(i-1)} - \frac{h_i}{2h_{i-1}}\hat{x}_0^{(i-2)}$$

$$x_i = \frac{t_i}{t_{i-1}}x_{i-1} - (1-t_i)(e^{-h_i} - 1)D_i$$

其中 $h_i = \lambda_{t_i} - \lambda_{t_{i-1}}$，$\lambda_t = \log\frac{1-t}{t}$ 是 log-SNR。

**关键约束：只加速窗口后面（post-window）的 ODE 步。** 加速窗口前面的 ODE 会让数值误差被窗口内的 SDE 随机性放大，严重损害生成质量。

两种变体：
- **MixGRPO-Flash**：Progressive 策略 + 后窗口 DPM-Solver++ 加速
- **MixGRPO-Flash\***：Frozen 策略（窗口固定在开头） + 全后段加速，加速比 $S = T / (w + (T-w)\tilde{r})$

### 关键贡献四：CPS 采样替代标准 SDE

标准 SDE 的 Euler-Maruyama 离散化在每步注入独立噪声，容易产生高频"颗粒感"伪影，导致 reward hacking。MixGRPO 引入 **Coefficients-Preserving Sampling (CPS)**：

$$x_{t_{i-1}} = \frac{1-t_{i-1}}{1-t_i}x_{t_i} + \left(t_{i-1} - \frac{1-t_{i-1}}{1-t_i}t_i\right)v_{t_i} + \sigma_{t_i}\epsilon_i$$

CPS 的核心优势：它保持了 Rectified Flow 的**线性插值结构**（$x_t = (1-t)x_0 + tx_1$），噪声通过系数约束融入路径而非叠加在路径上，从而消除了采样伪影。

### MixGRPO 的实验成果

![MixGRPO 实验结果对比](/chengYi-xun/img/mixgrpo_experiment1.png)

#### 同条件定量对比（HPDv2 数据集，4 奖励模型联合）

| 方法 | NFE$_{\pi_{\theta_\text{old}}}$ | NFE$_{\pi_\theta}$ | 每轮时间 (s) | HPS-v2.1 | ImageReward |
|:---|:---:|:---:|:---:|:---:|:---:|
| FLUX 基线 | — | — | — | 0.313 | 1.088 |
| DanceGRPO（14步） | 25 | 14 | 291.3 | 0.356 | 1.436 |
| DanceGRPO（4步） | 25 | 4 | 150.0 | 0.334 | 1.335 |
| **MixGRPO** | **25** | **4** | **150.8** | **0.367** | **1.629** |
| MixGRPO-Flash | 16 (Avg) | 4 | 112.4 | 0.358 | 1.528 |
| MixGRPO-Flash\* | 8 | 4 | **83.3** | 0.357 | 1.624 |

**同样只优化 4 步**，MixGRPO 的 ImageReward 比 DanceGRPO 高出 22%（1.629 vs 1.335）。MixGRPO-Flash\* 训练时间仅为 DanceGRPO 的 **29%**（83.3s vs 291.3s），性能仍然更优。

#### CPS vs SDE 采样

| 方法 | Pick Score | ImageReward | HPS-v2.1 |
|:---|:---:|:---:|:---:|
| MixGRPO-SDE | 0.234 | 1.590 | 0.365 |
| MixGRPO-CPS | **0.238** | **1.645** | **0.369** |

CPS 在所有指标上全面超越标准 SDE，且生成图像视觉上更干净。

#### 视频生成（HunyuanVideo-1.5）

MixGRPO 在 HunyuanVideo-1.5 的 T2V 任务上展现了更强的稳定性：Flow-GRPO 在视频高维空间中出现训练不稳定和指标退化，而 MixGRPO 在 HPSv3、VQ、MQ、TA 四个维度上均**稳定单调上升**。

---

## Part III: 源码实现对比

理论讲清楚之后，我们来看代码层面三者的核心差异。

### SDE 单步去噪：`flux_step`（DanceGRPO / MixGRPO 共用）

这是所有方法共享的基础函数，实现 Flow Matching（Rectified Flow）的 SDE 单步更新：

```python
def flux_step(model_output, latents, eta, sigmas, index, prev_sample, grpo, sde_solver):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma  # 负值，表示向低噪声方向前进

    # ODE 步进：x_{t+Δt} = x_t + Δσ · v_θ（确定性部分）
    prev_sample_mean = latents + dsigma * model_output

    # Tweedie 公式反推干净图像：x̂₀ = x_t − σ_t · v_θ
    pred_original_sample = latents - sigma * model_output

    # SDE 噪声标准差：η · √Δt
    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t)

    # Score Function 修正（仅 SDE 模式）
    # s_t(x_t) = -(x_t - x̂₀·(1-σ)) / σ²
    # 加入到漂移项中：μ = v_θ - ½η²·s_t
    if sde_solver:
        score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    # rollout 阶段：注入 Gaussian 噪声
    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

    # 计算高斯 log π(a_t|s_t)
    if grpo:
        log_prob = (
            -((prev_sample.detach().float() - prev_sample_mean.float()) ** 2)
                / (2 * std_dev_t**2)
            - math.log(std_dev_t)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        # 对空间维度求均值，得到标量 log-prob
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean, pred_original_sample
```

### DanceGRPO 的采样循环：全轨迹 SDE

DanceGRPO 在采样时对**每一步都调用 SDE**：

```python
def run_sample_step(args, z, progress_bar, sigma_schedule, transformer, ...):
    all_latents = [z]
    all_log_probs = []

    for i in progress_bar:  # 遍历所有 T 步
        # transformer 前向推理得到 v_θ
        model_pred = transformer(hidden_states=z, timestep=timesteps, ...)

        # 关键：sde_solver=True 对每一步都注入噪声
        z, pred_original, log_prob = flux_step(
            model_pred, z, args.eta, sigma_schedule, i,
            prev_sample=None, grpo=True, sde_solver=True
        )

        all_latents.append(z)
        all_log_probs.append(log_prob)

    # 返回完整轨迹的 latent 和 log_prob
    return z, latents, torch.stack(all_latents, dim=1), torch.stack(all_log_probs, dim=1)
```

### DanceGRPO 的训练循环：随机抽样优化

```python
# ① 随机打乱时间步索引
perms = torch.stack([torch.randperm(T) for _ in range(batch_size)]).to(device)
for key in ["timesteps", "latents", "next_latents", "log_probs"]:
    samples[key] = samples[key][torch.arange(batch_size)[:, None], perms]

# ② 只取前 K 步训练
train_timesteps = int(T * args.timestep_fraction)  # 默认 0.6

for i, sample in enumerate(samples_batched_list):
    for t in range(train_timesteps):
        # 用当前策略重新计算 log_prob
        new_log_probs = grpo_one_step(
            args, sample["latents"][:, t], sample["next_latents"][:, t],
            encoder_hidden_states, encoder_attention_mask,
            transformer, sample["timesteps"][:, t], perms[i][t], sigma_schedule
        )

        # 重要性比率
        ratio = torch.exp(new_log_probs - sample["log_probs"][:, t])

        # 分维度 PPO clip loss（视频任务特有）
        vq_loss = torch.mean(torch.maximum(
            -vq_advantages * ratio,
            -vq_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        )) / (gradient_accumulation_steps * train_timesteps)

        mq_loss = torch.mean(torch.maximum(
            -mq_advantages * ratio,
            -mq_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        )) / (gradient_accumulation_steps * train_timesteps)

        # 加权组合并反传（每步独立 backward，释放计算图）
        final_loss = vq_coef * vq_loss + mq_coef * mq_loss
        final_loss.backward()
```

### MixGRPO 的采样循环：混合 ODE-SDE

MixGRPO 的关键差异在于**根据窗口位置选择性地启用 SDE**：

```python
# 滑动窗口状态管理
class GRPOStates:
    def get_current_timesteps(self):
        """返回当前窗口内的时间步索引"""
        return list(range(self.left, self.left + self.group_size))

    def update_iteration(self):
        """每 τ 轮移动一次窗口"""
        self.current_iteration += 1
        if self.current_iteration % self.iters_per_group == 0:
            self.left = min(self.left + self.stride, self.total_steps - self.group_size)

# 采样时根据窗口设置 deterministic 标志
current_window = grpo_states.get_current_timesteps()
for i in range(total_steps):
    # 窗口内用 SDE，窗口外用 ODE
    deterministic = (i not in current_window)

    if deterministic:
        # ODE 步进：无噪声，无 log_prob
        prev, pred = flux_step(..., sde_solver=False)
    else:
        # SDE 步进：注入噪声，记录 log_prob
        prev, pred, log_prob = flux_step(..., sde_solver=True)
```

### MixGRPO-Flash：DPM-Solver++ 加速 ODE 段

窗口后面的 ODE 步可以用高阶求解器压缩：

```python
# 判断当前步是否在窗口后、应用 DPM-Solver++ 加速
if dpm_apply_strategy == "post" and i >= window_end:
    # 二阶 DPM-Solver++ 多步法
    # 需要至少两个历史 x̂₀ 预测值
    x0_pred_current = latents - sigma * model_output  # 当前 x̂₀
    D = (1 + h_i/(2*h_prev)) * x0_pred_prev - (h_i/(2*h_prev)) * x0_pred_prev2
    next_latent = (sigma_next/sigma) * latents - (1-sigma_next) * (exp(-h_i)-1) * D
else:
    # 常规一阶 Euler 步进
    next_latent = latents + dsigma * model_output
```

### 三者的核心差异汇总

| 代码层面 | Flow-GRPO | DanceGRPO | MixGRPO |
|:---|:---|:---|:---|
| 采样时 `sde_solver` | 全部 `True` | 全部 `True` | 窗口内 `True`，窗口外 `False` |
| 训练步选择 | 全部 $T$ 步 | `randperm` + `timestep_fraction` | 窗口内 $w$ 步 |
| 初始噪声 | 不同 | 相同（`use_same_noise`） | 相同 |
| 高阶求解器 | 无 | 无 | 窗口后 DPM-Solver++ |
| 多维奖励 | 加权求和 | VQ/MQ 分项标准化 | VQ/MQ 分项标准化 |

---

## Part IV: 2026 年最新前沿——从稀疏奖励到密集奖励（Dense Reward）

在 DanceGRPO 和 MixGRPO 之后，2026 年初的最新研究主要聚焦于解决 GRPO 在图像/视频生成中的另一个核心痛点：**信用分配（Credit Assignment）问题**。

在标准的 Flow-GRPO 或 MixGRPO 中，奖励（Reward）只有在生成完最后一步（$t=0$）的完整图像/视频后才能由奖励模型给出。这种**稀疏奖励（Sparse Reward）**被平均分配给了去噪轨迹上的所有步骤。然而，去噪过程的每一步贡献是完全不同的：早期的高噪声阶段决定了全局结构和语义布局（低频信息），而后期的低噪声阶段主要负责纹理和细节的生成（高频信息）。如果某一步实际上破坏了图像结构，但最终生成的图像依然得到了高分，这一步的错误行为也会被错误地“奖励”。

为了实现精确的步级信用分配（Step-wise Credit Assignment），2026 年涌现了多篇重磅工作：

1. **DenseGRPO (arXiv:2601.20218, 2026年1月)**

   - **核心机制：预测步级奖励增益**。DenseGRPO 不再把最终奖励平摊给所有步骤，而是通过 ODE 求解器在中间步骤预测出干净图像，并用奖励模型对其打分。它将相邻两步的得分差值（Reward Gain）作为当前时间步的**密集奖励（Dense Reward）**。这样，如果某一步的去噪操作让图像质量变好了，它就会得到正奖励；如果变差了，就会得到负惩罚。
   - **自适应探索空间校准**：作者发现，在不同的时间步，模型需要的探索空间是不同的。DenseGRPO 引入了一种“奖励感知（Reward-aware）”的机制，根据当前时间步的奖励情况，**自适应地调整 SDE 采样器中的随机性（Stochasticity）大小**。在模型表现不佳的步骤注入更多噪声以鼓励探索，在表现较好的步骤减少噪声以稳定利用。

2. **Stepwise-Flow-GRPO (arXiv:2603.28718, 2026年3月)**

   - **核心机制：基于 Tweedie 公式的增益分配**。与 DenseGRPO 类似，它也致力于计算每一步的奖励增益 $g_t = r_{t-1} - r_t$。为了高效获得中间步骤的得分，它巧妙地利用了 **Tweedie 公式**（即我们在代码中看到的 `pred_original_sample = latents - sigma * model_output`），在任意中间步骤 $t$ 直接单步估计出干净图像 $\hat{x}_0$，并送入奖励模型打分。
   - **改进版 DDIM-SDE**：标准的 SDE 采样在中间步骤引入的噪声往往会导致 Tweedie 估计出的 $\hat{x}_0$ 质量较差，从而让奖励模型给出不准确的低分。为此，作者引入了一种受 DDIM 启发的改进版 SDE，它在保持策略梯度所需的随机性的同时，显著提高了中间步骤图像的质量，使得奖励模型的打分更加精准。

3. **TempFlow-GRPO (ICLR 2026)**

   - **核心机制：轨迹分支（Trajectory Branching）与过程监督**。它通过在中间步骤进行多次分支采样，显式地获取过程奖励（Process Reward），从而捕捉 Flow 生成过程中的时间结构。
   - **噪声感知加权方案**：提出了一种基于噪声水平的加权机制，强制让模型在影响最大的早期生成阶段（高噪声、决定全局构图的阶段）集中学习。这与 MixGRPO 的滑动窗口课程学习思想有异曲同工之妙，但它是通过显式的损失加权来实现的。

这些 2026 年的新工作标志着视觉生成领域的 RL 正在经历与 LLM 推理（如 OpenAI o1, DeepSeek-R1）类似的演进：从简单的“结果监督（Outcome Supervision, ORM）”全面转向更精细的“过程监督（Process Supervision, PRM）”。

---

## 总结：三代方法的技术演进

| 维度 | Flow-GRPO | DanceGRPO | MixGRPO |
|:---|:---|:---|:---|
| **核心出发点** | 将 GRPO 引入 Flow Matching | 统一 Diffusion + Flow Matching，推广到视频 | 缩短 MDP，混合 ODE-SDE |
| **适用范式** | Flow Matching（含 Rectified Flow） | Diffusion + Flow Matching（统一 SDE） | 通用概率流 ODE（理论），Flow Matching（实验） |
| **SDE 范围** | 全轨迹 | 全轨迹 | 仅窗口内 |
| **训练步选择** | 全部 | 随机子集 | 滑动窗口（课程学习） |
| **高阶求解器** | 不可用 | 不可用 | 窗口后可用 |
| **任务范围** | T2I | T2I + T2V + I2V | T2I + T2V |
| **收敛性证明** | Fokker-Planck | Stochastic Interpolant | 分段 Fokker-Planck（通用 ODE 形式） |
| **训练时间基线** | — | 291s/iter | **83s/iter**（Flash\*） |

{% note info no-icon %}
**关于适用范式的补充说明**：三篇工作的理论都不局限于 Rectified Flow。Flow-GRPO 的 ODE→SDE 转换（Eq. 7）适用于任意 Flow Matching 模型；DanceGRPO 通过 Stochastic Interpolant 框架统一了 Diffusion 和 Flow Matching；MixGRPO 的混合 ODE-SDE 理论（Eq. 2-4）从通用概率流 ODE 出发，其收敛性证明对任何满足 Fokker-Planck 方程的概率流成立。实验中选择 Rectified Flow（FLUX、SD3.5、HunyuanVideo）是因为它是当前 T2I/T2V 领域的主流架构。
{% endnote %}

从 Flow-GRPO 到 DanceGRPO 是**广度的扩展**（统一 Diffusion 与 Flow Matching 范式、拓展到视频模态）；从 DanceGRPO 到 MixGRPO 是**深度的优化**（更好的 MDP 建模、更聪明的采样策略）。三者共同构成了概率流模型 + GRPO 方向的完整技术图谱。

---

> 参考资料：
>
> 1. Xue, Z., et al. (2025). *DanceGRPO: Unleashing GRPO on Visual Generation*. arXiv:2505.07818.
> 2. Li, J., et al. (2026). *MixGRPO: Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE*. arXiv:2507.21802.
> 3. Liu, J., et al. (2025). *Flow-GRPO: Training Flow Matching Models via Online RL*. arXiv:2505.05470.
> 4. Wang, F., Yu, Z. (2025). *Coefficients-Preserving Sampling for RL with Flow Matching*. arXiv:2509.05952.
> 5. *DenseGRPO: From Sparse to Dense Reward for Flow Matching Model Alignment*. arXiv:2601.20218.
> 6. *Stepwise Credit Assignment for GRPO on Flow-Matching Models*. arXiv:2603.28718.
