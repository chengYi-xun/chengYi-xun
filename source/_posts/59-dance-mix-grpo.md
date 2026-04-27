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
- **Schrödinger Bridge**：它寻找两个分布之间最可能的随机演化路径，可以看作是 Flow Matching 的广义随机版本。

而 **Rectified Flow（RF）是这个框架下最简单、最暴力的一个具体实现**。它的核心特征有两个：

1. **直线路径**：它强制规定从数据 $x_0$ 到噪声 $x_1 \sim \mathcal{N}(0, I)$ 的路径必须是最简单的直线：$x_t = (1-t)x_0 + t x_1$。
2. **独有的 Reflow 机制（从数学角度理解）**：

   - **理论上的 Reflow**：在最初训练 RF 时（称为 1-Rectified Flow），噪声 $x_1$ 和图像 $x_0$ 是**独立随机配对**的。这导致不同图像的生成轨迹在空间中会发生**严重交叉**。一旦轨迹交叉，速度场就会变得极度非线性，ODE 求解器必须用非常小的步长（很多步）才能准确积分。为了解决这个问题，原论文提出了一种“用魔法打败魔法”的迭代微调技术：先用训练好的 1-RF 模型生成一批图像，记录下每个噪声 $x_1$ 和它最终生成的图像 $x_0$ 的**确定性映射对**。然后，用这些**已经绑定好**的数据对去训练一个新的模型（2-Rectified Flow）。这个过程在数学上等价于在寻找**最优传输（Optimal Transport）**的解，能把交叉的轨迹彻底“拉直”。
   - **工程实践中的演进（如 FLUX / SD3）**：虽然理论上的 Reflow 很优美，但在工业界训练大模型时，先生成几十亿张图片再去训练 2-RF 的成本太高了。因此，**当前最前沿的开源大模型（如 Stable Diffusion 3、FLUX）在实践中并没有使用多阶段的迭代 Reflow**。相反，它们在第一阶段训练（1-RF）时，直接在每个 Batch 内部使用**最优传输（Optimal Transport, OT）算法**对噪声和图像进行配对（即 OT-CFM）。通过在训练源头上强制让每个噪声点去寻找离它“最近”的真实图像，模型在单阶段训练中就能直接学到几乎不交叉的直线轨迹，从而实现极少步数的高质量生成。

因此，本文讨论的算法（如 Flow-GRPO）虽然在理论上适用于所有 Flow Matching 模型，但在工程落地时，都是在 Rectified Flow（结合 OT 配对）的架构上进行验证的（详见[《笔记｜生成模型（十三）：Flow Matching理论与实现》](/chengYi-xun/posts/13-flow_matching/)）。
{% endnote %}

1. **范式局限**：Flow-GRPO 的 ODE→SDE 转换要求模型预测速度场 $v_t$，这是 Flow Matching 模型的特征。传统 Diffusion 模型（如 SD v1.4）使用噪声预测 $\epsilon$-prediction 和不同的前向过程（$z_t = \alpha_t x + \sigma_t \epsilon$），其 SDE 形式（Langevin 动力学）与 Flow Matching 的 SDE 在数学结构上不同。Flow-GRPO 未对 Diffusion 模型进行适配和验证。
2. **模态局限**：仅处理文生图（T2I），未触及视频生成（T2V、I2V），而视频的高维度、高算力需求和更复杂的评价标准是全新的挑战。
3. **方法局限**：先前的 RL 方法（DDPO、DPOK）在大规模 Prompt 集上训练不稳定，且未在 Flow Matching 模型上被充分验证。

DanceGRPO 的出发点就是：**能否用一个统一的 GRPO 框架，同时覆盖 Diffusion 和 Flow Matching（含 Rectified Flow）两种范式、图像和视频两种模态？**

### 关键贡献一：统一的 SDE 公式——从 Flow Matching 到 Diffusion

DanceGRPO 的第一个理论贡献是推导出覆盖**传统 Diffusion 模型**和**Flow Matching 模型**（含 Rectified Flow）的**统一 SDE 框架**。为了理解这个贡献，我们需要先弄懂这两个公式背后的物理意义。

**1. 为什么需要 SDE（随机微分方程）？**
无论是 Diffusion 还是 Flow Matching，它们在推理时通常使用常微分方程（ODE）进行确定性采样。但强化学习（RL）的核心是**试错与探索（Exploration）**。如果每次给定相同的 Prompt 和初始噪声，模型生成的轨迹完全一样，RL 就无法通过对比“好”与“坏”的样本来学习。因此，必须将确定性的 ODE 转换为包含随机噪声的 SDE。

**2. 传统 Diffusion Model 的反向 SDE**（基于 Score-based SDE 理论）：
$$dz_t = \underbrace{f_t z_t dt}_{\text{收缩项}} - \underbrace{\frac{1}{2}g_t^2\nabla\log p_t(z_t)dt}_{\text{去噪引导项}} - \underbrace{\frac{\varepsilon_t^2}{2}g_t^2\nabla\log p_t(z_t)dt + \varepsilon_t g_t\,dw}_{\text{Langevin 随机探索项}}$$

- **物理意义**：前两项是确定性的，负责把纯噪声一步步还原成图像。最后面的 `Langevin 随机探索项` 是为了 RL 引入的“扰动”。它在注入随机噪声（$\varepsilon_t g_t\,dw$）让模型去探索不同画面的同时，又加入了一个基于 Score 的修正力（$-\frac{\varepsilon_t^2}{2}g_t^2\nabla\log p_t(z_t)dt$），确保这种随机扰动不会破坏图像原本的概率分布。

**3. Flow Matching / Rectified Flow 的反向 SDE**（基于 Stochastic Interpolant 理论）：
$$dz_t = \underbrace{u_t dt}_{\text{直线速度场}} - \underbrace{\frac{1}{2}\varepsilon_t^2\nabla\log p_t(z_t)dt + \varepsilon_t\,dw}_{\text{随机探索与修正项}}$$

- **物理意义**：与 Diffusion 类似，第一项 $u_t dt$ 是 Flow Matching 特有的直线速度场，直接指向目标图像。后面的项同样是注入随机噪声并用 Score 函数进行修正，以保证探索过程的合法性。

**4. 统一的数学本质：Stochastic Interpolant（随机插值）**
这两个 SDE 看起来长得不一样，但 DanceGRPO 借助 Stochastic Interpolant 框架证明了它们**在数学结构上是完全等价的**。
无论是 Diffusion 的弯曲路径 $z_t = \alpha_t x_0 + \sigma_t \epsilon$，还是 Rectified Flow 的直线路径 $z_t = (1-t)x_0 + t x_1$，它们本质上都是在对数据 $x_0$ 和噪声进行插值。只要对 Diffusion 的坐标系进行适当的缩放（除以 $\alpha_t$），并将时间轴映射为信噪比（$\sigma_t/\alpha_t$），这两种看似不同的物理过程就会坍缩成同一个极其优雅的公式：

$$\tilde{z}_s = \tilde{z}_t + \text{网络输出} \cdot (\eta_s - \eta_t)$$

- **物理意义**：在这个统一的变换空间里，无论是预测噪声（Diffusion）还是预测速度（Flow Matching），单步的去噪过程都变成了**当前状态加上网络输出乘以时间步长**。
- 对 $\varepsilon$-prediction（Diffusion），变换后的坐标 $\tilde{z} = z/\alpha$，时间 $\eta = \sigma/\alpha$。
- 对速度预测（Flow Matching / Rectified Flow），坐标无需变换 $\tilde{z} = z$，时间 $\eta = t$。

**结论**：这个统一公式的伟大之处在于，**同一套 GRPO 算法代码逻辑，不需要做任何架构上的修改，就可以直接通吃两种范式**。DanceGRPO 也因此成为首个在 SD v1.4（Diffusion 代表）和 FLUX/HunyuanVideo（Flow Matching 代表）上统一验证成功的 GRPO 框架。

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

1. **全轨迹采样与记录**：首先，模型在不计算梯度（`torch.no_grad()`）的情况下，用 SDE 完整走完 $T$ 步去噪采样过程，生成最终的图像/视频。在这个过程中，记录下每一步的隐状态（latent）、时间步（timestep）以及旧策略给出的动作概率（`log_prob`）。
2. **打乱时间步顺序（打破时间相关性）**：将这 $T$ 步的数据在时间维度上随机打乱。这一步非常关键，因为相邻时间步的梯度高度相关，打乱顺序可以打破这种相关性，让模型在优化时看到的样本更加独立（类似于强化学习中的经验回放池 Experience Replay）。
3. **按比例截断（随机子采样）**：由于视频生成所需显存极大，如果对所有 $T$ 步都计算梯度，显存会直接 OOM（Out Of Memory）。因此，DanceGRPO 设定了一个比例 $\tau$（默认 0.6），只取打乱后的前 $K = \lfloor T \times \tau \rfloor$ 步进行优化。相当于在整条轨迹中随机“抽查”了 60% 的步骤。
4. **单步独立反向传播（极致的显存优化）**：对于选中的这 $K$ 步，并不把它们拼接成一个巨大的计算图。相反，DanceGRPO 采用了一个极其节省显存的策略：每次只拿出一个时间步 $t$，用当前最新的策略网络重新计算其动作概率（`new_log_prob`），计算 PPO 的 Clip Loss，然后立刻执行 `.backward()` 反向传播，并**立即释放该步的计算图**。这样，无论 $K$ 有多大，显存占用始终只相当于单步前向+反向的开销。

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

**问题一：全轨迹 SDE 带来的“推理算力浪费”**
在强化学习中，我们需要用旧策略 $\pi_{\theta_\text{old}}$ 生成一批图像供奖励模型打分（即 Rollout 数据收集阶段）。DanceGRPO 为了让整个轨迹满足强化学习的 MDP（马尔可夫决策过程），强制要求在 $T$ 步去噪的**每一步**都注入 SDE 随机噪声。
这就带来了一个致命问题：SDE 必须老老实实地一步步积分（比如走完 25 步）。我们原本可以用高阶 ODE 求解器（如 DPM-Solver++，只需要 8-10 步就能生成高质量图像）来大幅加速这个 Rollout 过程，但因为 SDE 噪声的干扰，高阶求解器完全失效了。这导致数据收集阶段耗时极长。

**问题二：随机抽样导致的“策略偏移（Policy Shift）”**
DanceGRPO 在采样时，整条轨迹 $T$ 步都注入了探索噪声；但在训练时为了省显存，只随机挑选了其中 $K$ 步（比如 4 步）进行梯度更新。
这就好比一个学生做了一套包含 25 道题的试卷，老师最后给了一个总分，但却只随机挑了 4 道题给他讲解正确答案。那剩下的 21 道题呢？模型在这些未被选中的步骤上产生的错误探索（策略偏移）完全没有得到纠正。
实验证明，这种“偷工减料”会直接损害模型性能：当优化步数从 14 降到 4 时，ImageReward 显著下降了 7%。正如 MixGRPO 论文一针见血的评价：*“这种方法并没有从根本上解决计算开销的问题。”*

**问题三：随机打乱带来的“梯度冲突（Gradient Conflict）”**
DanceGRPO 把整条轨迹的 $T$ 步打乱，随机抽取 $K$ 步放在同一个 Batch 里优化。
然而，扩散/流模型的生成过程具有极强的**阶段性特征**：

- **早期高噪声阶段**（如 $t \to 1$）：模型在努力确定画面的**全局构图和大结构**。
- **后期低噪声阶段**（如 $t \to 0$）：模型在努力刻画**局部纹理和光影细节**。
这两种任务的目标截然不同，所需的梯度更新方向甚至可能是相反的。把它们强行塞进同一个 Batch 里同时优化，就像让一个人左手画圆右手画方，会导致严重的梯度冲突，拖慢模型的收敛速度。

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

**收敛性保证：** MixGRPO 在 Supplementary 中严格证明了混合 ODE-SDE 采样与纯 ODE 采样产生**完全相同的边缘分布**。证明的核心在于利用 Kolmogorov 方程（Fokker-Planck 方程）来分析概率密度 $q_t(x)$ 随时间的演化。

**严格的数学推导如下：**

对于纯 ODE 采样：$dx_t = [f(x_t, t) - \frac{1}{2}g^2(t)\nabla \log q_t(x_t)]dt$，其对应的连续性方程为：
$$\frac{\partial q_t(x)}{\partial t} = -\nabla_x \cdot \left[ \left( f(x, t) - \frac{1}{2}g^2(t)\nabla_x \log q_t(x) \right) q_t(x) \right]$$

对于 SDE 采样：$dx_t = [f(x_t, t) - g^2(t)\nabla \log q_t(x_t)]dt + g(t)dw$，根据 Fokker-Planck 方程，其概率密度演化为：
$$\frac{\partial q_t(x)}{\partial t} = \underbrace{-\nabla_x \cdot \left[ \left( f(x, t) - g^2(t)\nabla_x \log q_t(x) \right) q_t(x) \right]}_{\text{漂移项贡献（向内收拢）}} + \underbrace{\frac{1}{2}g^2(t)\nabla_x^2 q_t(x)}_{\text{扩散噪声项贡献（向外扩散）}}$$

利用数学恒等式 $\nabla_x \log q_t(x) = \frac{\nabla_x q_t(x)}{q_t(x)}$，以及拉普拉斯算子定义 $\nabla_x^2 q_t(x) = \nabla_x \cdot \nabla_x q_t(x)$，我们可以将上式展开并合并：
$$
\begin{aligned}
\frac{\partial q_t(x)}{\partial t} &= -\nabla_x \cdot [f(x, t)q_t(x)] + g^2(t)\nabla_x \cdot [\nabla_x q_t(x)] + \frac{1}{2}g^2(t)\nabla_x \cdot [\nabla_x q_t(x)] \\
&= -\nabla_x \cdot [f(x, t)q_t(x)] + \frac{1}{2}g^2(t)\nabla_x \cdot [\nabla_x q_t(x)] \\
&= -\nabla_x \cdot \left[ f(x, t)q_t(x) - \frac{1}{2}g^2(t)\nabla_x q_t(x) \right] \\
&= -\nabla_x \cdot \left[ \left( f(x, t) - \frac{1}{2}g^2(t)\nabla_x \log q_t(x) \right) q_t(x) \right]
\end{aligned}
$$

可以看到，SDE 中多出来的漂移项（$-g^2 \nabla \log q_t$）与注入的随机噪声项（$\frac{1}{2}g^2 \nabla^2 q_t$）在 Fokker-Planck 方程中**精确对消了一半**，最终化简得到的形式与纯 ODE 的连续性方程**完全一致**。因此，无论在哪个时间段切换 SDE 或 ODE，整个系统在宏观上的概率密度演化轨迹 $q_t(x)$ 是完全相同的。

### 关键贡献二：滑动窗口作为优化调度器

$S$ 不是固定的——它是一个**滑动窗口** $W(l) = \{t_l, t_{l+1}, \ldots, t_{l+w-1}\}$，随训练进行从低 SNR 向高 SNR 移动：

$$l \leftarrow \min(l + s,\; T - w) \quad \text{every } \tau \text{ iterations}$$

{% note info no-icon %}
**什么是 SNR（信噪比）？**

SNR（Signal-to-Noise Ratio）即**信噪比**，用来衡量当前图像中“真实信号”与“随机噪声”的能量比例。

- **低 SNR（低信噪比）**：噪声远多于信号。这通常对应于去噪过程的**早期**（$t \approx 1$），此时图像几乎全是纯噪声，模型正在努力“无中生有”地勾勒出大体的轮廓和构图。
- **高 SNR（高信噪比）**：信号远多于噪声。这通常对应于去噪过程的**后期**（$t \approx 0$），此时图像的主体已经非常清晰，模型只是在做最后的微调，比如增加毛发纹理、修正光影细节。
{% endnote %}

这等价于一种**隐式课程学习**（analogous to temporal discounting in RL）：

- **在训练早期**，窗口位于低 SNR 区域（靠近纯噪声端，$l \approx 0$）。此时模型主要优化全局构图和物体布局，探索空间大，具有较高的随机性。
- **在训练后期**，窗口滑动到高 SNR 区域（靠近干净图像端，$l \rightarrow T-w$）。此时模型主要优化纹理细节和色彩精修，探索空间小，随机性更低。

MixGRPO 论文的消融实验验证了这种直觉。作者对比了四种不同的窗口策略：

1. **Frozen（固定在开头）**：窗口始终停留在低 SNR 区域。这种策略在 HPS-v2.1（0.354）和 ImageReward（1.580）上的表现最差，说明缺乏后期细节优化的探索。
2. **Random（每轮随机位置）**：窗口在每次迭代时随机跳跃。虽然 HPS-v2.1 有所提升（0.365），但 ImageReward 却大幅下降（1.513），说明随机跳跃破坏了学习的连贯性。
3. **Progressive + Constant $\tau$（恒定步长滑动）**：窗口按照固定的迭代次数 $\tau$ 逐步向后滑动。这种策略在 HPS-v2.1 上取得了最高分（0.367），ImageReward 也达到了 1.629。
4. **Progressive + Exp Decay $\tau$（指数衰减滑动）**：滑动速度越来越快。这种策略在 ImageReward 上取得了最高分（1.632），HPS-v2.1 为 0.360。

Progressive 一致优于 Random，**验证了课程学习优于随机抽样的假设**。更值得注意的是，即使 Frozen 策略（仅优化最前面几步）也超过了 DanceGRPO（NFE=4 时的 1.335），说明**定向优化前几步比全轨迹 SDE + 随机抽样更有效**。

MixGRPO 还提出**指数衰减调度**：$\tau(l) = \tau_0 \cdot \exp(-k \cdot \text{ReLU}(l - \lambda_\text{thr}))$，让模型在影响最大的构图阶段停留更久。

### 关键贡献三：MixGRPO-Flash 高阶求解器加速

MixGRPO 的混合策略带来了一个巨大的工程收益：**窗口外的 ODE 步可以用高阶求解器（如 DPM-Solver++）来加速！**

在 Flow-GRPO 和 DanceGRPO 中，由于全轨迹都注入了 SDE 噪声，模型必须老老实实地走完几十步（比如 25 步），无法使用任何加速算法。而在 MixGRPO 中，只有滑动窗口 $W$ 内是 SDE 随机探索，窗口外都是确定性的 ODE。

MixGRPO 将 DPM-Solver++ 适配到了 Flow Matching 框架中。简单来说，DPM-Solver++ 是一种“预测-校正”算法，它利用前几步的速度场信息，可以一步跨越原来需要好几步才能走完的距离，从而大幅缩短推理时间。

**关键约束：只加速窗口后面（post-window）的 ODE 步。**
为什么不能加速窗口前面的 ODE 步？

- **窗口前的 ODE**：此时图像还处于高噪声阶段，如果用大步长（高阶求解器）跨越，不可避免会产生一些数值计算误差。当这些带有误差的图像进入滑动窗口（SDE 阶段）时，SDE 的随机噪声注入会把这些微小的误差**成倍放大**，最终导致生成的图像崩溃。
- **窗口后的 ODE**：此时 SDE 探索已经结束，图像的大体结构已经定型。此时再用高阶求解器加速，即使有一点微小的数值误差，也不会被后续的随机性放大，对最终生成质量的影响微乎其微。

基于这个发现，作者提出了两种加速变体：

- **MixGRPO-Flash**：使用 Progressive（渐进式滑动）策略，仅对滑动窗口之后的 ODE 步进行 DPM-Solver++ 加速。
- **MixGRPO-Flash\***：为了追求极致速度，采用 Frozen 策略（把窗口死死固定在最开头）。这样一来，除了开头几步是 SDE，后面所有的步骤都可以用高阶求解器全速狂飙，训练时间被压缩到了极致。

**哪种效果更好？**
直觉上，MixGRPO-Flash 使用了更科学的课程学习（滑动窗口），效果应该更好。但论文的实验结果（Table 9）却给出了一个反直觉的结论：**MixGRPO-Flash\* 的整体表现反而更优！**

| 方法 | NFE（采样步数） | 单图耗时 (s) | HPS-v2.1 | ImageReward |
|:---|:---:|:---:|:---:|:---:|
| DanceGRPO (基线) | 25 | 9.30 | 0.334 | 1.335 |
| MixGRPO-Flash | 16 (平均) | 6.43 | **0.362** | 1.578 |
| MixGRPO-Flash\* | **8** | **3.79** | 0.357 | **1.624** |

**为什么 MixGRPO-Flash\* 更强？**
1. **极致的效率**：因为窗口固定在开头，它能最大程度地利用高阶求解器。单图生成耗时从基线的 9.3 秒降到了 3.79 秒（加速近 60%），采样步数从 25 步降到了 8 步。
2. **早期探索的决定性作用**：虽然它放弃了后期的滑动优化，但实验证明，**图像的全局构图和核心语义（由早期低 SNR 阶段决定）对最终奖励分数的影响最大**。只要在开头几步（窗口内）做好了 SDE 探索和优化，即使后面全用确定性加速跳过，依然能拿到极高的 ImageReward 分数（1.624）。
3. **权衡**：MixGRPO-Flash 在 HPS-v2.1 上略高一点（0.362 vs 0.357），说明滑动窗口确实对某些细节指标有帮助。但综合考虑巨大的时间收益和 ImageReward 的显著提升，MixGRPO-Flash\* 是性价比最高的选择。

### 关键贡献四：CPS 采样替代标准 SDE

从直觉上看，CPS 公式与之前的 SDE 似乎都在路径上注入了噪声 $\epsilon_i$（以提供 RL 所需的探索空间）。但两者的核心差异在于：**在注入噪声后，如何保证生成轨迹的概率分布不被破坏？**

1. **标准 SDE 的做法（强行叠加 + 估算补丁）**：
   标准 SDE（Euler-Maruyama 离散化）是在确定性步进上**强行叠加**一个独立的噪声：
   $$x_{t_{i-1}} = x_{t_i} - v_{t_i} \Delta t + \underbrace{\sqrt{2\sigma^2 \Delta t} \epsilon_i}_{\text{强行加噪}} - \underbrace{\frac{1}{2}\sigma^2 s_{t_i}(x_{t_i}) \Delta t}_{\text{Score 修正项}}$$
   由于强行加噪会让总方差变大（图像变糊），SDE 必须引入一个基于 Score Function（分数函数 $s_t$）的**漂移修正项**。但 Score 往往是靠 Tweedie 公式估算出来的，存在极大的理论和数值误差。

   {% note info no-icon %}
   **深度解析：Score 估算为什么存在误差？**
   
   SDE 的漂移修正项依赖于精确的 Score（数据分布的对数梯度 $\nabla_{x_t} \log p_t(x_t)$）。在实际工程中，我们通常用神经网络预测出干净图像 $\hat{x}_0$，再通过 Tweedie 公式反推 Score：$s_t(x_t) \approx -(x_t - \hat{x}_0) / \sigma_t^2$。这种估算在底层存在三大致命缺陷：
   
   1. **理论陷阱（Tweedie 的后验均值坍缩）**：数学上，神经网络预测的 $\hat{x}_0$ 实际上是给定带噪图像 $x_t$ 时，所有可能干净图像的**后验期望（均值）** $\mathbb{E}[x_0 | x_t]$。当噪声较大时，真实的后验分布是多峰的（一个带噪轮廓既可能是黑猫也可能是白猫）。用一个平滑的“均值”（灰色猫）去替代真实的多峰分布来计算 Score，这在数学上是一种极大的近似，丢失了高频细节信息。
   2. **模型盲区（神经网络的频谱偏好）**：深度学习模型（如 U-Net 或 DiT）天生具有“低频偏好”，更容易拟合轮廓和颜色，而难以完美预测毛发、纹理等高频细节。SDE 注入的是全频段的纯高斯白噪声，当 Score 试图将其抵消时，低频噪声被成功去除了，但**高频噪声没能被完全抵消，残留在了图像里**。
   3. **数值截断（大步长离散化）**：SDE 在数学上建立在连续时间（$\Delta t \to 0$）上。但在现代大模型极少步数（大 $\Delta t$）的采样中，我们用 Euler-Maruyama 离散化，假设在这一大步内 Score 的方向恒定不变。在高维非线性空间中，沿着一个错误的、恒定的方向走一大步，会直接导致样本偏离真实的数据流形（Data Manifold）。
   
   **恶性循环与 Reward Hacking**：SDE 强行注入全频段噪声（推离流形），Score 试图拉回但由于上述三大误差拉不准（尤其是高频细节拉不回）。每走一步，高频噪声就残留一点。步数走完，图像上就铺满了一层“颗粒感伪影”。像 PickScore 这样的奖励模型对高频纹理非常敏感，当图像出现这种异常的高频颗粒时，奖励模型会被“欺骗”，误以为这是某种丰富的纹理细节，从而给出虚高分（Reward Hacking）。
   {% endnote %}

2. **CPS 的做法（系数守恒，重新分配）**：
   MixGRPO 引入的 **Coefficients-Preserving Sampling (CPS，系数保持采样)** 借鉴了 DDIM 的核心思想（**待定系数法**）。正如 DDIM 在 Diffusion 中通过待定系数法（令 $x_{t-1} = \lambda x_0 + k x_t + \sigma \epsilon$）来严格保证边缘分布一致，CPS 也在 Flow Matching 中使用了同样的思路：它**完全抛弃了误差极大的 Score 修正项**，而是通过待定系数法，严格解出了在注入噪声 $\sigma_{t_i}\epsilon_i$ 时，图像信号 $x_{t_i}$ 和速度场 $v_{t_i}$ 应该缩小的精确比例。

   **CPS 的待定系数法推导过程：**
   
   在 Flow Matching 中，理论上的理想路径（边缘分布）为：
   $$x_t = (1-t)x_0 + t \epsilon \quad (\text{假设 } t=0 \text{ 是干净图像，} t=1 \text{ 是纯噪声})$$
   当我们从 $t_i$ 步走到下一步 $t_{i-1}$ 时，我们希望生成的 $x_{t_{i-1}}$ 必须严格满足这个理论分布：
   $$x_{t_{i-1}} = (1-t_{i-1})x_0 + t_{i-1} \cdot \text{Noise}$$
   
   在 $t_i$ 时刻，我们已知当前状态 $x_{t_i}$、模型预测的速度场 $v_{t_i}$，以及我们想注入的新随机噪声 $\epsilon_i$。CPS 假设下一步状态是这三者的线性组合：
   $$x_{t_{i-1}} = A \cdot x_{t_i} + B \cdot v_{t_i} + C \cdot \epsilon_i$$
   
   根据 Flow Matching 的定义，当前状态和速度场可以表示为：

   - $x_{t_i} = (1-t_i)x_0 + t_i \hat{\epsilon}$ （$\hat{\epsilon}$ 是模型预测出的原有噪声）
   - $v_{t_i} = \hat{\epsilon} - x_0$ （假设速度场指向噪声方向）
   
   将这两个式子代入假设公式中，并合并 $x_0$ 和噪声项：
   $$x_{t_{i-1}} = [A(1-t_i) - B] \cdot x_0 + [A t_i + B] \cdot \hat{\epsilon} + C \cdot \epsilon_i$$
   
   为了让结果严格符合目标分布，我们必须让系数一一对应：
   1. **图像信号的系数必须守恒**：
      $$A(1-t_i) - B = 1 - t_{i-1}$$
   2. **总噪声的方差必须守恒**（因为 $\hat{\epsilon}$ 和 $\epsilon_i$ 都是独立的高斯噪声，其方差之和必须等于目标噪声系数的平方）：
      $$(A t_i + B)^2 + C^2 = t_{i-1}^2$$
   
   设定 $C = \sigma_{t_i}$（即我们想要的随机性大小），联立上述方程，即可严格解出 $A$ 和 $B$：
   - $A = \frac{1-t_{i-1}}{1-t_i}$
   - $B = t_{i-1} - \frac{1-t_{i-1}}{1-t_i}t_i$
   
   将 $A, B, C$ 代回原式，便得到了 CPS 的最终采样公式：
   $$x_{t_{i-1}} = \underbrace{\frac{1-t_{i-1}}{1-t_i}x_{t_i} + \left(t_{i-1} - \frac{1-t_{i-1}}{1-t_i}t_i\right)v_{t_i}}_{\text{按比例缩小的确定性信号}} + \underbrace{\sigma_{t_i}\epsilon_i}_{\text{新注入的噪声}}$$
   
   **核心优势**：通过待定系数法，CPS 严格保持了 Rectified Flow 的**线性插值结构**（$x_t = (1-t)x_0 + tx_1$）。无论注入多大的噪声 $\sigma_{t_i}$，公式都能自动调整 $x_{t_i}$ 和 $v_{t_i}$ 的权重，确保最终图像的方差和均值守恒。这彻底消除了 SDE 带来的颗粒感伪影，提供了更干净的图像供奖励模型打分。

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
