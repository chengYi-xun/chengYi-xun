---
title: 笔记｜强化学习（二）：信任区域与近端策略优化 (从 TRPO 到 PPO)
date: 2025-08-17 10:00:00
cover: false
mathjax: true
categories:
 - Notes
tags:
 - Deep learning
 - Generative models theory
 - Reinforcement Learning
series: Diffusion Models theory
---

> 本文为系列第二篇。在上一篇中，我们介绍了策略梯度和 Actor-Critic 架构。然而，包括 REINFORCE 在内的所有基础策略梯度方法，都存在更新步长难以控制、训练不稳定的核心困境。本文将首先深入剖析这一不稳定性的三个层面，然后详细推导如何通过限制策略更新幅度来保证训练的单调递增，从 TRPO 的数学思想一路演进到目前大模型 RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）的基石——PPO 算法。
>
> ⬅️ 上一篇：[笔记｜强化学习（一续）：从 REINFORCE 到 Actor-Critic](/chengYi-xun/posts/51-reinforce-ac/)
>
> ➡️ 下一篇：[笔记｜强化学习（三）：大模型对齐的另一条路：DPO (Direct Preference Optimization)](/chengYi-xun/posts/53-dpo/)

**核心摘要：**
PPO 的核心是根据 TRPO 发展而来的。TRPO 在 Actor-Critic 的基础上，通过限制更新步长（引入 KL 散度约束），使得策略更新单调递增。其最大的贡献在于**从数学上精确求解了参数空间与分布空间（策略变化）的映射关系（即 Fisher 信息矩阵，它是 KL 散度对网络参数的二阶导数矩阵。具体来说，是计算新策略和旧策略之间的 KL 散度，然后对新策略的参数 $\theta$ 求二阶偏导数）**，从而避免了使用启发式学习率导致的两个空间（参数空间和分布空间）优化步长不匹配的问题。然而，TRPO 引入了二阶优化。虽然在实际工程中，TRPO 并没有直接去求那个极其庞大的二阶矩阵的逆（而是巧妙地用了共轭梯度法 Conjugate Gradient 算矩阵向量乘积），但它依然需要进行多次反向传播来计算二阶信息，这对于动辄百亿参数的大模型来说计算代价是不可接受的。PPO 继承了 TRPO 限制分布变化步长的核心思想，但巧妙地通过**裁剪（Clip）重要性采样比率**，避开了二阶矩阵的求解。它仅用一阶优化的计算成本就实现了类似的安全更新效果，从而成为了目前大模型 RLHF 的基石算法。

# 关键概念回顾：Q 函数与优势函数

在深入本文之前，我们先回顾上一篇中两个最重要的概念。

**动作价值函数 $Q^\pi(s, a)$** 衡量的不是"当前这一步动作有多好"，而是**"选择这个动作后，一直走完全程能拿多少分"**。具体来说：在状态 $s$ 下选择动作 $a$，然后从下一步开始按策略 $\pi$ 继续行动直到结束，所获得的**期望累积奖励**。

以面试为例（假设面试有 3 轮）：$Q^\pi(s_1, \text{讲项目}) = \underbrace{r_1}_{\text{第1轮得分}} + \gamma \cdot \underbrace{\mathbb{E}[r_2 + \gamma r_3]}_{\text{后续按策略}\pi\text{走的期望得分}}$。它评估的是"从选择这个动作开始的整个未来"，而非仅仅当前一步的得分。

**优势函数 $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$** 则回答：**这个动作比"盲选"好多少？** $V^\pi(s)$ 是按策略随机选动作的平均期望得分，$Q^\pi(s,a)$ 是指定选动作 $a$ 的期望得分。两者之差 $A$ 就是动作 $a$ 的"优势"——$A > 0$ 说明比平均好，$A < 0$ 说明比平均差。

但要注意：在实际训练中，我们**算不出真实的** $Q$ 和 $V$，只能用有限的采样数据去**估计**它们。上一篇介绍了三种从粗到精的估计方法：

1. **REINFORCE（蒙特卡洛估计）**：跑完一整局面试，用实际拿到的总分 $G_t = r_1 + r_2 + \cdots$ 直接替代 $Q(s_t, a_t)$。无偏但方差极大——后续轮次的随机性全混入了对当前动作的评价。
2. **REINFORCE + Baseline**：用 $G_t - V_\phi(s_t)$ 近似优势 $A$，其中 $V_\phi$ 是一个 Critic 神经网络通过大量采样拟合出的"各状态平均得分"。减去基线后信号在零附近波动，方差显著降低，但仍需等到面试结束才能计算。
3. **Actor-Critic（TD 估计）**：用**自举**——只走一步就估计 $Q(s_t, a_t) \approx r_t + \gamma V_\phi(s_{t+1})$，然后 $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ 近似优势 $A$。不用等到结束就能更新，但因为 $V_\phi$ 本身是估计值，所以引入了偏差。

无论哪种方法，估计都是有噪声的、不精确的。这一点在理解下面的"步长控制"问题时至关重要。

---

# 策略梯度的核心困境：为什么"迈大步"会让训练崩溃

在上一篇中，我们讲到了如何通过策略梯度让模型"变聪明"。在所有策略梯度方法——包括最基础的 REINFORCE——中，我们通过梯度上升来更新策略参数 $\theta$：
$$
\theta_{\text{new}} = \theta_{\text{old}} + \alpha \nabla_\theta J(\theta)
$$

这里的学习率 $\alpha$ 决定了更新的**步长**。但策略梯度的更新幅度真的只由 $\alpha$ 决定吗？把梯度展开来看：

$$\nabla_\theta J(\theta) = \mathbb{E}\left[ \underbrace{\nabla_\theta \log \pi_\theta(a|s)}_{\text{方向}} \cdot \underbrace{A(s,a)}_{\text{幅度权重}} \right]$$

这里形成的是一个**两级步长控制**：

- **$\nabla_\theta \log \pi_\theta(a|s)$**（score function）：告诉你"往参数空间的哪个方向调，能让这个动作的概率上升"。它是一个单位方向指示器——梯度上升和下降是沿这个方向还是其反方向。

- **$A(s,a)$**（优势函数）：告诉你在**这个样本上**，沿该方向走多"狠"。优势大，这条样本的梯度分量就猛；优势接近零或为负，分量就小甚至反向。

- **$\alpha$**（学习率）：在**所有样本梯度聚合之后**，再做一次全局缩放。它不区分哪个样本更值得信任，而是一刀切地控制整体更新的激进程度。

所以说"对数梯度决定方向，优势决定步长"是就**单条样本对梯度的贡献**而言的。而 $\alpha$ 是跨所有样本的全局闸门。

但问题的要害恰恰在这里——优势函数给出的"单样本步长"是极不可靠的。同一个采样批次里，某条轨迹可能回报异常高（运气好），另一条可能很低，对应的梯度分量量级能差几个数量级。这种情况下，你调 $\alpha$ 也无济于事：设大了会因少数高优势样本导致灾难性更新，设小了又让正常样本的训练几乎停滞。

一旦因为单样本优势的剧烈波动导致"步子迈太大"，整个训练就会陷入崩溃。下面我们从具体的例子出发，逐步揭示这种"大步更新"是如何从三个层面摧毁策略梯度训练的。

## 第一层：优势估计失效

**先用上一篇的面试例子来理解这个问题。**

还记得我们训练的 AI 面试助手吗？假设经过几轮训练，AI 在第 1 轮（自我介绍）已经学到了一个不错的策略：

- $\pi_\text{old}$("讲项目经历" $| s_1$) = 70%
- $\pi_\text{old}$("讲兴趣爱好" $| s_1$) = 30%

为了量化这一点，假设各动作的 Q 值为 $Q(s_1, \text{讲项目}) = 15$，$Q(s_1, \text{讲兴趣}) = 5$。在旧策略下，基准线为：

$$V^{\pi_\text{old}}(s_1) = 0.7 \times 15 + 0.3 \times 5 = 12$$

所以 $A^{\pi_\text{old}}(s_1, \text{讲项目}) = 15 - 12 = +3$，$A^{\pi_\text{old}}(s_1, \text{讲兴趣}) = 5 - 12 = -7$。

**太激进的更新会怎样？** 如果学习率 $\alpha$ 设得太大，一次更新后策略可能变成：

- $\pi_\text{new}$("讲项目经历" $| s_1$) = 99%
- $\pi_\text{new}$("讲兴趣爱好" $| s_1$) = 1%

看起来没毛病？但问题在于——**$\hat{A}$ 的计算前提是"策略不变"**。回忆上面的定义，$\hat{A}$ 是在旧策略 $\pi_\text{old}$（70%/30%）下估算出来的，它的基准线 $V^{\pi_\text{old}}(s_1)$ 也是旧策略下的平均得分。如果策略发生了大幅变化，$\hat{A}$ 赖以成立的前提就被打破了，估计值就不再准确。

具体来看，新策略（99/1）下基准线变成了：

$$V^{\pi_\text{new}}(s_1) = 0.99 \times 15 + 0.01 \times 5 = 14.9$$

所以 $A^{\pi_\text{new}}(s_1, \text{讲项目}) = 15 - 14.9 = +0.1$。同一个动作，优势从 $+3$ 暴跌到 $+0.1$——因为当你几乎只选"讲项目"时，它已经就是平均水平了，自然没什么"优势"可言。但在更新过程中，我们用的始终是 $\hat{A} = +3$（旧策略下的值），相当于用了一个比真实值大 30 倍的信号来推动参数，导致更新幅度远超合理范围。

更关键的是，在实际训练中，策略是一个参数共享的神经网络 $\pi_\theta(a|s)$。当你大幅修改参数 $\theta$ 以改变某个状态下的策略时，所有其他状态下的策略也会被连带改变。例如，你猛拉参数让第 1 轮 99% 选"讲项目"，但同一个网络在第 2 轮（技术面）的输出也被打乱了——整个策略可能全面崩溃。

这就引出了第一个问题：**$\hat{A}$ 是在"假设策略不变"的前提下估计的，但更新本身又会改变策略**。步子越大，策略变化越大，$\hat{A}$ 就越不准确，更新就越不可靠。但这只是问题的冰山一角。

## 第二层：参数空间与策略空间的鸿沟

上面的例子展示了"步子迈太大"的危险。但为什么控制步长在策略梯度中如此困难？根本原因在于：**参数空间中的"距离"和策略空间中的"距离"没有简单的对应关系。**

学习率 $\alpha$ 控制的是参数 $\theta$ 在欧几里得空间中移动的幅度。但策略 $\pi_\theta(a|s)$ 是通过 softmax 等非线性函数从 $\theta$ 映射出来的概率分布。这种非线性映射导致：**参数空间中大小相同的一步，在策略空间中引起的分布变化可能截然不同**——沿某些方向，参数微调就会导致概率分布剧变（例如 softmax 在接近饱和区时的极端敏感性）；沿另一些方向，参数大幅变化却几乎不影响输出分布。

打个比方：想象你在山上徒步，只有一个指南针（梯度方向）和一个固定步幅（学习率）。往北走 10 米可能让你翻过一座山脊（策略剧变），往东走 10 米却只是在平地上挪了一下（策略几乎不变）。**没有任何固定的步幅能在所有方向上都安全。** 这正是后文 TRPO 引入 Fisher 信息矩阵的根本原因——它描述了"地形的局部曲率"，告诉我们哪些方向敏感、哪些方向安全。

## 第三层：正反馈崩溃循环——强化学习独有的致命问题

如果仅仅是"更新幅度不好控制"，那和监督学习中调学习率的困难并无本质区别。**真正让策略梯度训练如此脆弱的，是强化学习独有的特性——数据分布的非平稳性（Non-stationarity）。**

在监督学习中，训练数据是固定的数据集。即使某一步更新方向有误、模型暂时变差，下一步还能从同一批正确的数据中恢复。但在强化学习中，**训练数据是由策略自身产生的**——策略决定了智能体与环境的交互方式，从而决定了它能看到什么数据。这就形成了一个致命的**正反馈崩溃循环**（参考 [The Problem with Standard Policy Gradients](https://codefrydev.in/Reinforcement/curriculum/volume-05/chapter-01/)）：

1. **大步更新** → 策略发生大幅变化
2. **策略变差** → 采集到的轨迹质量下降（因为坏策略做出坏决策）
3. **垃圾数据** → 基于这些数据计算出的梯度方向更差
4. **更差的梯度** → 参数被推向更糟糕的区域
5. **回到第 1 步** → 恶性循环，训练不可逆转地崩溃

在监督学习中，这个循环不存在——你训练一个图像分类器，即使某一步更新让准确率下降了，ImageNet 数据集还在那里，下一批数据照样是正确的标注。但在强化学习中，一旦策略崩溃，采到的全是垃圾数据，模型就再也学不回来了。

## 实验证据：REINFORCE 有多不稳定

上述问题不只是理论推导，实验数据也清楚地展示了这一点。在经典的 CartPole 环境中，[Trust Region Methods: From REINFORCE to TRPO to PPO](https://sesen.ai/blog/trust-region-methods-reinforce-trpo-ppo) 给出了一组精心控制的对比实验——三种方法使用**完全相同的网络结构和超参数**（两层 64 单元的 tanh MLP），唯一的区别是更新规则：

| 方法 | 达到 400 分所需迭代 | 训练稳定性 | 步长控制机制 |
|:---:|:---:|:---:|:---:|
| REINFORCE | ~79 轮 | 全程剧烈震荡 | **无**——完全依赖手动调学习率 |
| TRPO | ~18 轮 | 较稳定，偶有波动 | KL 散度硬约束（$\delta = 0.01$） |
| PPO | ~15 轮 | 最稳定，达标后持续保持 | clip 裁剪（$r \in [0.8, 1.2]$） |

KL 散度监控进一步揭示了根源：**REINFORCE 每次更新对策略的改变幅度是无界的**（unbounded KL）。虽然学习率限制了参数空间中的步长，但如上节所述，这并不等于限制了策略空间中的变化。相比之下，TRPO 将 KL 散度显式约束在 $\delta \approx 0.01$ 附近，PPO 通过裁剪机制隐式控制在约 0.02 以下——两者都给出了策略空间中的"护栏"。

## 问题小结：为什么我们需要信赖域？

> **💡 核心死局与破局之道**
> 
> 总结来看，策略梯度方法面临的步长困境是一个**由浅入深、逐层叠加的死局**：
> 
> 1. **表层原因（估计失效）**：优势 $\hat{A}$ 的计算依赖"策略不变"假设，步子一旦迈大，作为更新依据的 $\hat{A}$ 本身就成了错的。
> 2. **深层原因（空间错位）**：参数空间的一小步，可能是策略空间的一大步。固定的学习率 $\alpha$ 根本无法在崎岖的策略空间中保证每一步都安全。
> 3. **致命打击（数据非平稳）**：一次大步更新导致策略变差，就会采到垃圾数据，进而算出更差的梯度，陷入**不可逆转的崩溃循环**。
> 
> 前文提到的"两级步长控制"（优势函数管单样本，$\alpha$ 管全局）在此面前苍白无力：优势估计的高方差让单样本步长极不可靠，而全局的 $\alpha$ 又无法兼顾不同方向的敏感度。
> 
> **破局之路**：要打破这个死局，唯一的方法就是**放弃在参数空间中盲目摸索，直接在策略空间中给更新幅度画一条不可逾越的"红线"**。这正是 **REINFORCE 不稳定性的根源**，也是**信赖域（Trust Region）**思想的起源。

接下来的两节，我们将看到 TRPO 是如何严谨地画出这条红线，而 PPO 又是如何用大道至简的"裁剪"艺术实现同样效果的。

---

# TRPO：画个圈圈，在圈里找最优解

前面我们从三个层面剖析了策略梯度的步长困境。现在来看解决方案。Schulman 等人在 2015 年提出了 TRPO (Trust Region Policy Optimization) 算法（[原始论文](https://arxiv.org/abs/1502.05477)），核心思路可以拆解为两步：

1. **构造替代目标函数**：利用重要性采样，使得一批旧数据可以被复用多次，解决样本效率问题。
2. **施加 KL 散度约束**：直接在策略空间（而非参数空间）中限制每次更新的幅度，解决步长控制的核心问题。

**核心思想用例子说**：既然怕 AI 面试助手一次更新走得太远（从 70% 跳到 99%），那我就给它画一个"信任区域"——你每次更新后的新策略，和旧策略之间的"差距"不能超过一个阈值 $\delta$。

## 替代目标函数 (Surrogate Objective)

**为什么叫"替代"？** 我们**真正想优化**的目标是新策略的期望回报 $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \gamma^t r_t]$。但这个真实目标**没法直接优化**——计算它需要用新策略 $\pi_\theta$ 去采集数据，而参数 $\theta$ 每一步都在变，不可能每次梯度更新都重新采样。实际训练中我们手头只有旧策略 $\pi_\text{old}$ 采集的数据。因此我们需要构造一个"替代品"——一个可以用旧数据计算、且在旧策略邻域内能忠实反映真实目标的近似函数。如何用这批旧数据来评估并优化新策略 $\pi_\text{new}$ 的效果？

关键工具是**重要性比率（Probability Ratio）**：
$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

$r$ 衡量的是新策略相对于旧策略对某个动作的"偏好变化倍率"。在面试例子中，假设新策略把"讲项目经历"的概率从 70% 提升到 84%：
$$r = \frac{0.84}{0.70} = 1.2$$

$r = 1.2$ 意味着新策略选这个动作的概率是旧策略的 1.2 倍。TRPO 用 $r$ 乘以旧策略下估计的优势 $\hat{A}$，构建**替代目标函数**来近似新策略的实际表现：
$$\mathcal{L}^{\text{CPI}}(\theta) = \mathbb{E}_{t} \left[ r_t(\theta) \hat{A}_t \right]$$

上标 $\text{CPI}$ 来自 Kakade & Langford (2002) 提出的 **Conservative Policy Iteration**（保守策略迭代），这是最早引入替代目标思想的工作。

直觉很简单：如果 $\hat{A} > 0$（好动作），$r > 1$（新策略增加了这个动作的概率），那么 $r \cdot \hat{A}$ 就是一个正的信号，说明新策略比旧策略更好。

**"替代"的直观几何含义：类比泰勒展开。** 替代目标 $\mathcal{L}^{\text{CPI}}(\theta)$ 和真实目标 $J(\theta)$ 之间的关系，可以完美类比为**函数的一阶泰勒近似**。

设 $f(x)$ 是原函数，$g(x)$ 是它在 $x_0$ 处的线性近似：
$$g(x) = f(x_0) + f'(x_0)(x - x_0)$$
在 $x_0$ 处，这两条曲线**值相等**（$g(x_0) = f(x_0)$）且**一阶导数对齐**（$g'(x_0) = f'(x_0)$）——即它们在 $x_0$ 处相切。

替代目标正是扮演了这个"一阶近似"的角色：

1. **在旧策略处两者相等**：当 $\theta = \theta_\text{old}$ 时，$r_t = 1$，$\mathcal{L}^{\text{CPI}}(\theta_\text{old}) = \mathbb{E}_t[\hat{A}_t] = 0$。真实目标的改进量同样为 $0$。
2. **在旧策略处梯度相同**：$\nabla_\theta \mathcal{L}^{\text{CPI}} \big|_{\theta=\theta_\text{old}} = \nabla_\theta J(\theta) \big|_{\theta=\theta_\text{old}}$。这意味着两条曲线在旧策略处**相切**。在 $\theta_\text{old}$ 邻域内，沿着替代目标上升，就是在沿着真实目标上升。
3. **远离旧策略后偏离加剧**：就像用直线去拟合曲线，当 $x$ 偏离 $x_0$ 越远，误差就越大。当 $\theta$ 偏离 $\theta_\text{old}$ 越远，替代目标对真实表现的近似就越不准确——这正是后文必须引入 KL 约束和裁剪机制的根本原因。

但为什么乘一个比率 $r$ 就能用旧数据评估新策略？这背后的数学工具叫做**重要性采样（Importance Sampling）**。

**为什么不直接用新策略重新采样？** 最直觉的方案是：既然想评估新策略 $\pi_\theta$，就让它去采集新数据。但在实际训练中，每一步梯度更新都会改变参数 $\theta$。如果每次更新后都要重新采样，训练流程就变成：

> 1. 用当前策略 $\pi_\theta$ 采集一批数据 ← **开销巨大！**
> 2. 用这批数据计算梯度，更新一步 $\theta \to \theta'$
> 3. 丢掉所有数据（因为数据是 $\pi_\theta$ 采的，对 $\pi_{\theta'}$ 已经"过期"了）
> 4. 回到第 1 步，用 $\pi_{\theta'}$ 重新采集……

这正是基础策略梯度（REINFORCE）的做法，**每批数据只用一次就废弃**，样本效率极低。对大语言模型来说，采集一条样本意味着完整的自回归生成，成本极高。

**重要性采样的数学原理。** 重要性采样提供了一条"用旧数据评估新策略"的严格路径。设 $q$ 为实际采样用的分布，$p$ 为想要评估的分布，对任意函数 $f$，有恒等变换：
$$
\mathbb{E}_{x \sim p}[f(x)] = \sum_x p(x) f(x) = \sum_x q(x) \cdot \frac{p(x)}{q(x)} \cdot f(x) = \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right]
$$

其中 $\frac{p(x)}{q(x)}$ 称为**重要性权重（Importance Weight）**，它修正了两个分布之间的偏差。在策略优化中，令 $q = \pi_{\theta_\text{old}}$（旧策略），$p = \pi_\theta$（新策略），$f = \hat{A}_t$（优势函数），代入得：

$$
\mathbb{E}_{(s,a) \sim \pi_\theta}[\hat{A}_t] = \mathbb{E}_{(s,a) \sim \pi_{\theta_\text{old}}}\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)} \hat{A}_t\right] = \mathbb{E}_{(s,a) \sim \pi_{\theta_\text{old}}}[r_t(\theta) \hat{A}_t]
$$

因此 $r_t(\theta)$ 并非一个随意取的比值，而恰恰是重要性采样中的**重要性权重**。替代目标 $\mathcal{L}^{\text{CPI}}(\theta) = \mathbb{E}_t[r_t(\theta) \hat{A}_t]$ 是一个重要性采样估计量——用旧策略的样本配合权重 $r_t(\theta)$，**无偏地**估计新策略下的期望优势。

**用面试例子具象化。** 旧策略（70/30）采集了 1000 次模拟面试，其中 700 次选了"讲项目"，300 次选了"讲兴趣"。新策略想变成 84%/16%。直接用这 1000 条数据算平均优势，得到的是旧策略下的期望——"讲项目"占了 70% 的权重。但新策略选"讲项目"的频率更高，这类样本应该占更大权重。重要性权重 $r = 0.84/0.70 = 1.2$ 把每条"讲项目"的样本**放大 1.2 倍**；"讲兴趣"的权重 $r = 0.16/0.30 \approx 0.53$ 则**缩小到约 0.53 倍**。加权后，虽然数据来自旧策略，但期望等价于在新策略下计算的结果。

有了重要性采样，训练流程可以做到"一鱼多吃"：

> 1. 用当前策略 $\pi_\text{old}$ 采集一批数据 ← **只采一次！**
> 2. 用这批数据做 $K$ 次梯度更新：
>    - 每次通过 $r = \pi_\theta / \pi_\text{old}$ 修正分布偏差
>    - 通过约束机制保证 $\pi_\theta$ 不会偏离 $\pi_\text{old}$ 太远（确保 IS 估计仍然可靠）
> 3. 更新完毕后，$\pi_\text{old} \leftarrow \pi_\theta$，回到第 1 步

同一批数据被反复利用 $K$ 次，样本效率大幅提升。但重要性采样有一个已知缺陷：当新旧策略差距过大时，权重 $r$ 的方差会爆炸，估计变得极不稳定。与此同时，替代目标本身对 $r$ 也没有任何限制——如果直接最大化 $\mathcal{L}^{\text{CPI}}$，优化器会无限增大好动作的概率（让 $r \to \infty$），又回到了"步子迈太大"的问题。

## KL 散度约束

**KL 散度的定义**：KL 散度（Kullback-Leibler Divergence）用于度量两个概率分布之间的"距离"。对于两个离散分布 $P$ 和 $Q$：

$$\text{KL}[P \| Q] = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

它可以理解为：**如果真实分布是 $P$，用 $Q$ 去近似 $P$ 会损失多少信息量**。KL 散度越大，两个分布差距越大；当 $P = Q$ 时，$\text{KL} = 0$。

**用面试例子计算**：旧策略 $\pi_\text{old} = (0.7, 0.3)$，假设新策略 $\pi_\text{new} = (0.84, 0.16)$：

$$\text{KL}[\pi_\text{old} \| \pi_\text{new}] = 0.7 \log \frac{0.7}{0.84} + 0.3 \log \frac{0.3}{0.16} = 0.7 \times (-0.18) + 0.3 \times 0.63 \approx 0.063$$

如果新策略变成 $\pi_\text{new} = (0.99, 0.01)$：

$$\text{KL}[\pi_\text{old} \| \pi_\text{new}] = 0.7 \log \frac{0.7}{0.99} + 0.3 \log \frac{0.3}{0.01} = 0.7 \times (-0.35) + 0.3 \times 3.40 \approx 0.78$$

KL 散度从 0.063 飙升到 0.78——数值直接反映了策略偏移的剧烈程度。

**TRPO 的做法**：将替代目标和 KL 约束结合，构成一个**带约束的优化问题**——在保证新旧策略的 KL 散度不超过阈值 $\delta$ 的前提下，最大化替代目标函数：

$$
\max_\theta \; \mathbb{E}_{t} \left[ r_t(\theta) \hat{A}_t \right]
$$
$$
\text{subject to} \quad \mathbb{E}_t \left[ \text{KL}[\pi_{\theta_{\text{old}}}(\cdot|s_t) \| \pi_\theta(\cdot|s_t)] \right] \le \delta
$$

$\delta$ 通常设为 0.01 这样的小值。如果以面试例子来说，当 $\delta = 0.01$ 时，策略从 70%/30% 最多只能调到大约 74%/26%——每次只迈一小步。这样就保证了 $\hat{A}$ 在更新前后几乎不变，旧的优势估计始终有效。

### 为什么不能直接构造 KL loss 用 backward？

到此为止，我们已经写出了完整的约束优化问题。一个非常自然的疑问浮现：既然 KL 散度在 PyTorch 中一行 `kl_divergence()` 就能算出来，**为什么不能把它当一个 loss 项、用标准的 `loss.backward()` 一步搞定？** 具体来说，把约束优化强行改成无约束优化，直接加一个 KL 惩罚项：

$$\mathcal{L}_\text{total}(\theta) = \mathbb{E}_t[r_t(\theta) \hat{A}_t] - \beta \cdot \text{KL}[\pi_{\theta_\text{old}} \| \pi_\theta]$$

然后 `loss.backward()` + Adam 更新，十行代码搞定。跟 L2 正则化一个逻辑。**这种做法技术上完全可行**，它已经有名字了——**KL-regularized policy gradient**。PPO 原始论文（[Schulman et al., 2017](https://arxiv.org/abs/1707.06347)）的第一个方案 **PPO-Penalty**（Section 4: Adaptive KL Penalty Coefficient）正是这个思路，配合自适应调整 $\beta$ 的启发式规则。

但**约束优化和惩罚项是两个完全不同的数学问题**，核心矛盾在于 **$\beta$ 没法确定**。PPO 论文原文直接指出了这一点："*it is hard to choose a single value of $\beta$ that performs well across different problems—or even within a single problem, where the characteristics change over the course of learning*"。$\beta$ 是你手动设的自由参数，不是从数学中推出来的。而同一个 $\beta$ 在不同训练阶段的效果截然不同——reward 的尺度在变、策略的 curvature 在变、advantage 的分布也在变：

| $\beta$ 取值 | 效果 |
|:---:|:---|
| 太大 | 策略几乎不更新，学不到东西 |
| 太小 | KL 惩罚形同虚设，策略更新过猛，回到崩溃问题 |
| "合适" | 能工作——但什么值算"合适"是未知的，且它随训练进程不断变化 |

如果试图自适应地调 $\beta$（KL 过大就增大 $\beta$，过小就减小），本质上你是在用数值方法**隐式地求解约束优化问题**——找一个 $\beta$ 使得最终 KL 落在 $\delta$ 附近。

这就引出了 TRPO 的核心选择：**与其让你手动搜索 $\beta$，不如把它当变量直接解出来。** 而要做到这一点，就必须引入二阶优化。

### TRPO 如何用二阶优化精确求解约束？

**追踪推导：二阶信息从哪里进入？** TRPO 选择正面求解 KL 约束优化问题，而不是用惩罚项去近似它。沿着推导主线，可以精确地标出二阶信息进入的那一步：

1. **建立替代目标函数**：构造 $\mathcal{L}(\theta) = \mathbb{E}_t[r_t(\theta) \hat{A}_t]$。只是用重要性采样改写了期望，涉及的全是策略概率的**一阶信息**。没碰二阶。
2. **加上 KL 约束**：要求 $\mathbb{E}_t[\text{KL}[\pi_{\theta_\text{old}} \| \pi_\theta]] \le \delta$。到这里仍然没有二阶——约束只是定义了"距离"，**但还没说怎么解它**。
3. **对目标和约束分别做泰勒展开——二阶就是在这一步进入的。** 为了把约束优化转为可解的形式，令 $d = \theta - \theta_\text{old}$ 表示参数更新量，做两件事：
   - **目标函数**做一阶展开：$\mathcal{L}(\theta_\text{old} + d) \approx \mathcal{L}(\theta_\text{old}) + g^T d$，其中 $g = \nabla_\theta \mathcal{L}\big|_{\theta_\text{old}}$ 是策略梯度向量。这只用了一阶信息——把"复杂的非线性目标"近似为"沿梯度方向的线性增量"。
   - **KL 约束**做二阶展开：$\text{KL} \approx \frac{1}{2} d^T F d$，其中 $F$ 是 Fisher 信息矩阵（$N \times N$，$N$ 为参数数量）。零阶项和一阶项恰好都是零（零阶：自己和自己的 KL 为零；一阶：得分函数期望恒为零），**只剩下这个二次项**。正是这一步把二阶矩阵 $F$ 引入了问题。
4. **拉格朗日对偶求解**：经过上面的近似，原始问题化简为"在椭球约束 $\frac{1}{2}d^T F d \le \delta$ 下，最大化线性目标 $g^T d$"——即在一个由 $F$ 定义的椭球面上，找到与梯度 $g$ 内积最大的点。这是一个有封闭解的经典优化问题，用拉格朗日乘子法即可直接解出最优方向和精确步长。

**一句话总结：正是对 KL 散度的二阶泰勒展开，把 Fisher 信息矩阵引入了更新公式。在此之前的所有推导都是一阶的。** 展开后，KL 约束从一个抽象的"不准走太远"变成了一个具体可解的二次型 $\frac{1}{2}d^T F d$，而这个二次型的系数矩阵就是二阶信息。PPO 原文也印证了这一点——TRPO 通过 "a linear approximation to the objective and **a quadratic approximation to the constraint**" 来近似求解（[Schulman et al., 2017, Section 2.2](https://arxiv.org/abs/1707.06347)）。

> **Fisher 矩阵 vs. 传统 Hessian：** 值得强调的是，$F$ 不是目标函数 $\mathcal{L}(\theta)$ 对 $\theta$ 的 Hessian（即传统意义上的"损失函数的二阶导"），而是 **KL 散度对 $\theta$ 的 Hessian**。从信息几何的角度看，$F$ 是策略分布参数空间上的**黎曼度量（Riemannian metric）**，衡量的是 $\theta$ 空间中一个微小位移在概率分布空间中对应多大的距离（[Amari, 1998](https://doi.org/10.1162/089976698300017746)）。这正是它作为 KL 的 Hessian 而非目标函数 Hessian 出现的深层原因。

**回到三种方案的全景对比：**

| 方案 | 对待 $\beta$ 的方式 | 需要二阶信息？ |
|:---|:---|:---:|
| KL loss + backward（PPO-Penalty） | $\beta$ 是超参数，手动调或启发式自适应 | 不需要 |
| TRPO | $\beta$ 是约束条件的拉格朗日乘子，数学解出 | **需要** |
| PPO-Clip | 重新设计目标函数，根本不需要 $\beta$ | 不需要 |

PPO 论文的实验（Table 1）直接验证了这一点：PPO-Clip（$\epsilon = 0.2$，平均归一化得分 0.82）显著优于 PPO-Penalty（Adaptive KL 最高 0.74）和 Fixed KL（最高 0.72），因此 PPO-Penalty 在实践中被淘汰。PPO-Clip 走了一条更彻底的路：既不去精确求解 KL 约束（需要二阶），也不去调 $\beta$（不够稳定），而是直接用 $\text{clip}(r, 1-\epsilon, 1+\epsilon)$ 在目标函数层面限制概率比率的变化范围。$\epsilon$ 可以固定为常数（通常 0.2），因为 clip 作用于每个动作的概率比率 $r$，天然 scale-free——不管 reward 尺度如何变化，$\epsilon = 0.2$ 始终翻译为"不许把某个动作的概率改超过约 20%"。

**几何直觉：为什么需要 $F^{-1}$？** 一阶梯度 $g$ 只告诉你"参数空间中哪个方向提升目标最快"，但 TRPO 的约束加在**分布空间**（KL 散度）上。问题是：参数空间中等长的一步，在分布空间中的效果可能天差地别——某些方向上参数动一点分布就剧变（$F$ 特征值大），另一些方向动很多分布才微变（$F$ 特征值小）。$F$ 正是这种不均匀性的度量，$F^{-1}$ 则做了校正：压缩敏感方向、放大迟钝方向，使更新在分布空间中最高效地利用 KL 预算。

### 自然梯度的结果与直觉

**我们在求什么？** 每一轮更新中，我们要求出参数更新量 $d = \theta_\text{new} - \theta_\text{old}$。普通梯度下降是 $d = \alpha \cdot g$（学习率 $\alpha$ 需要调参）；TRPO 通过约束优化直接算出最优的 $d$。

经过对目标的一阶近似（$\mathcal{L} \approx g^Td$）和 KL 的二阶近似（$\text{KL} \approx \frac{1}{2}d^TFd$，其中零阶项和一阶项都恰好为零），问题简化为：

$$\max_d \; g^T d \quad \text{s.t.} \quad \frac{1}{2} d^T F d \le \delta$$

用拉格朗日乘子法求解（构造 $L = g^Td - \lambda(\frac{1}{2}d^TFd - \delta)$，对 $d$ 求导令其为零），直接得到：

$$\theta_{\text{new}} = \theta_\text{old} + \underbrace{\sqrt{\frac{2\delta}{g^T F^{-1} g}}}_{\text{步长（从约束精确算出）}} \cdot \underbrace{F^{-1} g}_{\text{方向（自然梯度）}}$$

这个结果有两层含义：

1. **方向 $F^{-1}g$**：不是沿原始梯度 $g$ 走，而是用 Fisher 矩阵的逆做了"地形校正"——分布敏感的方向被压缩、迟钝的方向被放大，使更新在分布空间中各方向均衡推进。
2. **步长 $\sqrt{2\delta/(g^TF^{-1}g)}$**：完全由约束 $\delta$ 决定，不需要手动设学习率。拉格朗日乘子 $\lambda$ 在数学上和 PPO-Penalty 中的超参数 $\beta$ 扮演相同角色，但 TRPO 把它从约束中精确解出，不再是超参数。

> **延伸阅读：** 完整推导（含 KL 散度零阶/一阶为零的证明、拉格朗日求解的中间步骤）见 [SpinningUp TRPO](https://spinningup.openai.com/en/latest/algorithms/trpo.html) 和 [Natural Gradient Descent — Agustinus Kristiadi](https://agustinus.kristia.de/blog/natural-gradient/)。

### TRPO 的局限性

公式 $\theta_\text{new} = \theta_\text{old} + \sqrt{2\delta/(g^TF^{-1}g)} \cdot F^{-1}g$ 看起来优雅，但在实际工程中面临三层困难：

**1. 计算代价高昂。** $F$ 是 $N \times N$ 的矩阵（$N$ = 参数量），百万参数的网络 $F$ 就有 $10^{12}$ 个元素，无法存储更无法求逆。TRPO 使用**共轭梯度法（CG）** 绕过显式求逆：只需要能计算"$F$ 乘以任意向量 $v$"（通过两次反向传播实现），迭代约 10 步近似得到 $F^{-1}g$。即便如此，每次策略更新仍需约 **20 次反向传播 + 一轮线搜索**，而 PPO 只需 1 次反向传播——计算开销差了一个量级。

**2. 与现代网络结构不兼容。** Fisher 信息矩阵的估计依赖于网络的确定性结构和样本独立性假设，而现代深度网络恰恰打破了这些假设：

| 组件 | 与 FIM 的冲突 |
|:---|:---|
| **Dropout** | 每次前向传播随机丢弃神经元，有效网络结构不断变化，FIM 估计不稳定 |
| **BatchNorm** | 归一化统计量依赖当前 mini-batch 中的所有样本，引入样本间相互依赖 |
| **Transformer** | 多头注意力 + 残差连接使 Hessian 结构极其复杂，二阶导数计算既慢又不稳定 |

**3. 无法扩展到大模型。** 上述问题叠加在一起，使得 TRPO 在参数动辄上亿的现代网络上几乎无法实际使用。**TRPO 从未被应用于大模型 RLHF**——当 2022-2023 年 RLHF 成为大模型对齐的核心技术时，学界直接选择了 PPO 而跳过了 TRPO。

这正是 PPO 诞生的全部意义——用一阶裁剪替代二阶约束，使信任区域方法能够扩展到工业级大模型。关于 TRPO 的完整实现细节（共轭梯度 + 线搜索 + Fisher-向量积），可参考 [SpinningUp TRPO 源码](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/trpo) 和 [Trust Region Methods: From REINFORCE to TRPO to PPO](https://sesen.ai/blog/trust-region-methods-reinforce-trpo-ppo)。

---

# PPO：大道至简的"裁剪"艺术

OpenAI 在 2017 年给出了答案：**PPO (Proximal Policy Optimization)**（[原始论文](https://arxiv.org/abs/1707.06347)）。核心思路：不去精确求解 KL 约束（那需要 Fisher 矩阵 + 共轭梯度），而是直接用 $\text{clip}(r, 1-\epsilon, 1+\epsilon)$ 限制每个动作的概率比率变化幅度。这个操作只需要**标准的一阶梯度 + Adam 优化器**，对网络结构没有任何限制。

## 裁剪目标函数 (Clipped Surrogate Objective)

PPO 的核心公式：
$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_{t} \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$
其中 $\epsilon$ 是一个超参数（通常设为 0.2）。

这个公式有三个部分：

1. **第一项**：$r_t(\theta) \hat{A}_t$ 是正常的替代目标（与 TRPO 相同）。
2. **第二项**：$\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t$ 将重要性比率强行截断在 $[0.8, 1.2]$ 之间。
3. **取最小值 (min)**：构建悲观的下界（Pessimistic Bound）。

**分情况讨论**：

- **$\hat{A}_t > 0$（好动作）**：我们想增加其概率。但如果 $r_t(\theta) > 1+\epsilon$（概率涨太多了），裁剪项生效，梯度变为 0——**见好就收**。
- **$\hat{A}_t < 0$（坏动作）**：我们想降低其概率。但如果 $r_t(\theta) < 1-\epsilon$（概率降太多了），裁剪项生效，梯度变为 0——**防止矫枉过正**。

裁剪还带来一个重要的副产品：$r \in [1-\epsilon, 1+\epsilon]$ 同时将重要性权重限制在安全范围内，防止了 IS 估计的方差爆炸。这使得 PPO 可以对同一批采样数据安全地进行**多个 epoch 的小批量更新**（通常 3\~10 个 epoch），相比 REINFORCE 每批数据只用一次就丢弃，样本效率提升了数倍。

## PPO 的完整损失函数与代码实现

在大模型微调中，PPO 是 **RLHF** 流程的核心算法。RLHF 的基本思路是：先用人类标注的偏好数据（"回答 A 比回答 B 好"）训练一个**奖励模型（Reward Model）**，然后将该奖励模型作为环境的"评委"，用 PPO 训练语言模型生成更符合人类偏好的回答。在这一流程中，PPO 的总损失函数通常包含三部分：
$$
\mathcal{L}^{\text{PPO}}(\theta) = \mathcal{L}^{\text{CLIP}}(\theta) - c_1 \mathcal{L}^{\text{VF}}(\theta) + c_2 S[\pi_\theta](s_t)
$$

1. **策略损失 $\mathcal{L}^{\text{CLIP}}$**：即上文推导的裁剪目标函数：

   $$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[\min\!\left(r_t(\theta)\,\hat{A}_t,\;\text{clip}\!\left(r_t(\theta),\,1-\epsilon,\,1+\epsilon\right)\hat{A}_t\right)\right]$$

   其中 $r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是新旧策略的概率比率，$\hat{A}_t$ 是 GAE 计算的优势估计。$\min$ 加上裁剪构成了"悲观下界"——当优势为正（好动作）时阻止概率涨过 $1+\epsilon$，当优势为负（坏动作）时阻止概率降过 $1-\epsilon$。这确保了每次更新不会偏离旧策略太远。

2. **价值损失 $\mathcal{L}^{\text{VF}}$**：Critic 网络的回归目标，用均方误差衡量 Critic 预测值与回报目标之间的差距：

   $$\mathcal{L}^{\text{VF}}(\theta) = \mathbb{E}_t \left[\left(V_\theta(s_t) - V_t^{\text{target}}\right)^2\right]$$

   其中 $V_t^{\text{target}}$ 是 Critic 的学习目标，定义为：

   $$V_t^{\text{target}} = \hat{A}_t + V_{\theta_{\text{old}}}(s_t)$$

   即 GAE 估计的优势加上旧 Critic 的预测值。直觉上：$V_{\theta_\text{old}}(s_t)$ 是 Critic 上一轮的预测，$\hat{A}_t$ 是"实际表现比预测好（或差）了多少"，两者相加就是"真实应该是多少"——这正是代码中 `returns = advantages + old_values` 的含义。通过最小化该项，Critic 学会准确预估每个状态的长期收益，从而为 Actor 提供更低方差的优势估计。系数 $c_1$（通常 0.5）控制价值损失在总损失中的权重。

3. **熵奖励（Entropy Bonus）$S[\pi_\theta]$**：策略熵的定义为：

   $$S[\pi_\theta](s_t) = -\sum_{a \in \mathcal{A}} \pi_\theta(a \mid s_t) \log \pi_\theta(a \mid s_t)$$

   其中 $\mathcal{A}$ 是**动作空间**（所有可选动作的集合），$a$ 是其中的单个动作。在传统 RL 中，$\mathcal{A}$ 是有限动作集（如游戏的上/下/左/右按键，$|\mathcal{A}|=4$）；在 LLM-RL 中，$\mathcal{A}$ 就是整个词表（vocabulary），$|\mathcal{A}|$ 为词表大小（3~13 万），每个动作 $a$ 对应一个 token，而状态 $s_t$ 则是当前已生成的上下文。

   熵衡量的是**策略在当前状态下对动作空间的概率分布的不确定性**，其取值范围为 $[0, \log|\mathcal{A}|]$：

   - **上限 $\log|\mathcal{A}|$（均匀分布）**：当策略完全随机，即对所有动作 $a$ 都有 $\pi_\theta(a|s_t) = \frac{1}{|\mathcal{A}|}$ 时，熵达到最大。
     - **直觉计算**：代入熵公式 $S = -\sum_{a=1}^{|\mathcal{A}|} \frac{1}{|\mathcal{A}|}\log\frac{1}{|\mathcal{A}|}$。因为求和项里没有跟 $a$ 相关的变量，这就相当于把 $\frac{1}{|\mathcal{A}|}\log\frac{1}{|\mathcal{A}|}$ 累加了 $|\mathcal{A}|$ 次。即 $S = -|\mathcal{A}| \cdot \left( \frac{1}{|\mathcal{A}|}\log\frac{1}{|\mathcal{A}|} \right)$。前面的 $|\mathcal{A}|$ 和分母抵消，剩下 $-\log\frac{1}{|\mathcal{A}|}$，根据对数性质即等于 $\log|\mathcal{A}|$。
     - **数学证明（拉格朗日乘数法）**：这是一个带约束的优化问题。目标是最大化熵 $H(p) = -\sum_i p_i \log p_i$，约束条件是概率之和为 1（$\sum_i p_i = 1$）。构造拉格朗日函数 $L(p, \lambda) = -\sum_i p_i \log p_i + \lambda (\sum_i p_i - 1)$。对每个 $p_i$ 求偏导并令其为 0，得到 $\frac{\partial L}{\partial p_i} = -\log p_i - 1 + \lambda = 0$，解得 $p_i = e^{\lambda - 1}$。这说明要使熵最大，所有的 $p_i$ 必须相等（因为它们都等于同一个常数 $e^{\lambda - 1}$）。再结合概率和为 1 的约束，必然得出每个 $p_i = \frac{1}{|\mathcal{A}|}$，从而证明了均匀分布是**全局最大熵分布**。
   - **下限 $0$（确定性策略）**：当存在某个 $a_k$ 使得 $\pi_\theta(a_k|s_t)=1$，其余 $\pi_\theta(a|s_t)=0$ 时，代入得 $S = -(1\cdot\log 1 + 0\cdot\log 0 + \cdots) = 0$。这里约定 $0\log 0 = 0$，与极限 $\lim_{x\to 0^+} x\log x = 0$ 一致。

   损失函数中以 $+c_2 \cdot S$ 的形式出现（注意正号），因为最小化 loss 时该项等价于**最大化熵**——鼓励策略保持随机性、避免过早坍缩到确定性行为，这正是**探索与利用（exploration vs. exploitation）权衡**的体现。系数 $c_2$（通常 0.01）较小，确保探索激励不会盖过策略优化信号。

下面用 PyTorch 风格的伪代码展示 PPO 的完整训练流程。这段代码展示了**一个完整的 PPO 训练循环**，包括外层的全局迭代（数据采样）和内层的多次小批量参数更新。

```python
import torch
import torch.nn.functional as F

# ============================================================
# 模型定义与超参数 (与 TRPO 相同: Actor + Critic)
# ============================================================
actor = ActorModel(state_dim, action_dim)
critic = CriticModel(state_dim)
optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)

clip_range = 0.2      # ε：比率 r 允许偏离 1 的幅度，对应 clip 区间 [1-ε, 1+ε]
vf_coef = 0.5         # 价值损失权重 c₁
entropy_coef = 0.01   # 熵奖励权重 c₂
K_epochs = 4          # 同一批 rollout 数据做 K 轮 epoch，配合 clip 复用样本、提高样本效率
batch_size = 64
total_training_steps = 10000

# 辅助函数：生成小批量索引
def minibatch_indices(total, batch_size):
    """将 [0, total) 随机打乱后按 batch_size 切分，yield 每个小批次的索引"""
    perm = torch.randperm(total)                  # 随机排列所有样本下标
    for start in range(0, total, batch_size):     # 按 batch_size 步进切片
        yield perm[start : start + batch_size]    # 返回一组索引，shape = [batch_size]

# ============================================================
# 真实的训练外层大循环
# ============================================================
state = env.reset()
for global_step in range(total_training_steps):
    
    # ============================================================
    # Step 1: 数据采集 (同 TRPO)
    # ============================================================
    buffer = []
    for t in range(T):
        with torch.no_grad():
            dist = actor(state)       # 当前策略下动作分布
            action = dist.sample()
            log_prob = dist.log_prob(action)  # 采样时策略的 log π，用作 PPO 旧概率
            value = critic(state)     # 同时记下 V，用于 GAE
        next_state, reward, done = env.step(action)
        buffer.append((state, action, reward, log_prob, value, done))  # 同时存 done
        state = next_state if not done else env.reset()
    
    states, actions, rewards, old_log_probs, old_values, dones = collate(buffer)
    
    # ============================================================
    # Step 2: 优势估计 (同 TRPO)
    # ============================================================
    with torch.no_grad():
        advantages = compute_gae(rewards, old_values, gamma=0.99, lam=0.95, dones=dones)  # GAE 估计 Â，供裁剪目标使用
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 优势标准化，稳定训练
        returns = advantages + old_values  # Critic 的回归目标
    
    # ============================================================
    # Step 3: 多 epoch 小批量更新 (PPO 核心 — 替代了 TRPO 的 Step 4~5)
    # 同一批数据安全地复用 K 次, 每次用 clip 防止偏离过远
    # ============================================================
    for epoch in range(K_epochs):
        for idx in minibatch_indices(len(states), batch_size):
            s, a = states[idx], actions[idx]
            old_lp, adv, ret = old_log_probs[idx], advantages[idx], returns[idx]
    
            # --- 前向传播 ---
            dist = actor(s)                           # 新策略分布 π_θ(·|s)
            new_log_probs = dist.log_prob(a)          # log π_θ(a|s)
            values = critic(s).squeeze()              # V_φ(s)
            # 策略熵的 batch 均值：鼓励探索，避免过早确定性策略（乘以系数后从 loss 中减去即熵奖励）
            entropy = dist.entropy().mean()           # 策略熵 S[π_θ]
    
            # --- 策略损失 (裁剪) ---
            ratio = torch.exp(new_log_probs - old_lp) # r = π_θ / π_old，重要性采样权重
            surr1 = ratio * adv                        # 未裁剪的替代项 r·Â
            # 将 r 限制在 [1-ε,1+ε] 再乘 Â，过大偏离时梯度被截断，隐式信任区域
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
            # 取 min 再取负：对优化器而言是最大化悲观下界，防止过度乐观的重要性加权
            policy_loss = -torch.min(surr1, surr2).mean()
    
            # --- 价值损失 ---
            value_loss = F.mse_loss(values, ret)
    
            # --- 总损失: L^CLIP - c₁·L^VF + c₂·S ---
            # 策略项 + 价值回归 − 熵奖励（entropy 越大 loss 越小，等价鼓励高熵）
            loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy
    
            # --- 反向传播 + 梯度裁剪 + 更新 (标准一阶优化!) ---
            optimizer.zero_grad()
            loss.backward()
            # 全局梯度范数裁剪：抑制异常大更新，与大模型/深层网络训练常见做法一致
            torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), max_norm=0.5
            )
            optimizer.step()
```

**开源代码参考：** Vanilla PPO 在经典 RL（游戏、机器人控制等）中的主流实现是 **Stable-Baselines3** 库（`stable_baselines3.PPO`），其核心逻辑与上述代码一致。

上面的代码展示的是经典强化学习场景中的 PPO，只需要 Actor 和 Critic 两个模型。但在大模型 RLHF 中，PPO 实际上需要**四个模型**同时在线。

## RLHF 中的 PPO：四模型架构

在 RLHF 场景下，PPO 的"环境"不再是游戏或物理模拟器，而是**语言生成 + 奖励模型打分**。整个系统需要四个模型协同工作：

1. **Actor（策略模型）** $\pi_\theta$：正在训练的语言模型，负责根据 prompt 生成回答。
2. **Critic（价值模型）** $V_\phi$：估计当前生成状态的价值，通常与 Actor 共享底座（backbone），只在顶部加一个标量输出头。
3. **Reference 模型** $\pi_\text{ref}$：**冻结的 SFT 模型**，即 Actor 训练前的初始状态。它的作用是防止 Actor 在 RL 训练过程中"学偏"——如果没有约束，Actor 可能学会生成一些得分很高但语无伦次的"奖励黑客"（Reward Hacking）输出。通过惩罚 Actor 偏离 Reference 的程度（KL 散度），可以确保生成的回答保持合理的语言质量。
4. **Reward 模型** $R_\psi$：**冻结的奖励模型**，用人类偏好数据预训练而成，负责为 Actor 的回答打分。

**为什么 RLHF 多了 KL 惩罚，而 vanilla PPO 没有？** 这是两者最本质的区别，源于奖励信号的可靠性差异：

- **Vanilla PPO（游戏/机器人）**：奖励来自**真实环境**——CartPole 真的平衡了 200 步，机器人真的到达了目标。奖励信号是"地面真值"，策略再怎么优化也无法"欺骗"物理定律。PPO 的 clip 机制控制**每步更新步长**就足够了。
- **RLHF-PPO（大模型微调）**：奖励来自一个**学出来的奖励模型** $R_\psi$，它是人类偏好的**不完美近似**。没有额外约束，策略会发现奖励模型的"盲区"——生成一些 RM 打高分但实际上语无伦次或重复套话的输出，这就是**奖励黑客（Reward Hacking）**。KL 惩罚 $\beta \cdot \text{KL}[\pi_\theta \| \pi_\text{ref}]$ 的作用是把策略"拴"在 SFT 参考模型附近，防止因追逐不可靠的奖励而走偏。

> **PPO clip 与 KL 惩罚的分工**：clip 管的是"每步别迈太大"（步长控制），KL 惩罚管的是"总体别跑太远"（分布正则化）。两者解决的是不同层面的问题。

**RLHF-PPO 的奖励公式**与 Vanilla PPO 存在巨大差异。由于语言生成是逐 Token 进行的，但奖励模型 RM 只能在句末给出一个整句的标量评分，我们需要将这个“大结算”拆分到每一个步骤中。具体的，第 $t$ 步（即生成第 $t$ 个 Token）的奖励 $r_t$ 构造如下：

$$
r_t = \begin{cases}
-\beta \log \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)} & \text{如果 } t < T \text{ (非最后一个 Token)} \\
R_\psi(\text{prompt}, \text{response}) - \beta \log \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)} & \text{如果 } t = T \text{ (最后一个 Token)}
\end{cases}
$$

其中 $\beta$ 控制 KL 惩罚的强度：$\beta$ 越大，Actor 越不敢偏离 Reference 模型。

有了逐步的奖励 $r_t$ 后，我们利用 GAE（Generalized Advantage Estimation）计算每个 Token 的优势 $\hat{A}_t$。与传统强化学习不同，在语言模型中我们通常将时间折扣因子 $\gamma$ 设为 1.0，因为长句子的第一个词和最后一个词同样重要：

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$
$$
\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l} \quad (\text{其中 } \gamma = 1.0)
$$

### Critic 的“地狱难度”与工业界解法

仔细观察上面的公式，你会发现一个巨大的挑战：**Critic 模型 $V_\phi$ 必须在只看到前面几个词的时候，就准确预估出整句话最终在句末能拿多少分**。这就是经典的“长序列信用分配（Credit Assignment）”难题。如果 Critic 预测不准，算出来的 Advantage 就是纯噪音，训练会立刻崩溃。

为了克服这个困难，工业界通常采用以下几种策略：

1. **用奖励模型初始化 Critic**：Critic 绝对不能从随机权重开始训练。通常直接复制 RM 的权重作为 Critic 的初始化，因为它本来就具备极强的“看前半句猜全句得分”的能力。
2. **过程奖励模型（PRM）**：像 OpenAI 的 Let's Verify Step by Step 中，把单纯的句末打分改成了每个推理步骤结束时都打分，给 Critic 提供更密集的中间监督信号。
3. **彻底抛弃 Critic（GRPO 路线）**：由于预测长序列的未来回报太难，而且 Critic 还要吃掉一倍的显存，2024年以后的新范式（如 DeepSeek-R1 使用的 GRPO 算法）直接抛弃了 Critic。它通过“对同一问题生成多条回答，句末打分后在组内计算 z-score”来替代 GAE 的优势估计，既解决了预测难的问题，又省下了一半的显存开销。

### RLHF-PPO 的总损失函数

在完成了上面的优势估计后，RLHF-PPO 的优化目标由三部分组成：策略损失（带裁剪）、价值网络损失（回归长期回报）和熵正则项：

$$
\mathcal{L}^{\text{PPO}}(\theta, \phi) = \underbrace{\mathcal{L}^{\text{CLIP}}(\theta)}_{\text{第一项：策略损失}} - c_1 \underbrace{\mathbb{E}\left[(V_\phi(s_t) - V_t^{\text{target}})^2\right]}_{\text{第二项：价值损失}} + c_2 \underbrace{S[\pi_\theta]}_{\text{第三项：熵奖励}}
$$

下面逐项展开：

**第一项：裁剪策略损失 $\mathcal{L}^{\text{CLIP}}(\theta)$**——与 Vanilla PPO 完全相同，只是代入公式的 $\hat{A}_t$ 是基于 KL 修正后的奖励计算而来的：

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[\min\!\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}\,\hat{A}_t,\;\text{clip}\!\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)},\,1-\epsilon,\,1+\epsilon\right)\hat{A}_t\right)\right]
$$

裁剪比率 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$ 被限制在 $[1-\epsilon, 1+\epsilon]$ 内。注意这里的 $a_t$ 在 LLM 场景下就是第 $t$ 个生成的 Token，$s_t$ 则是 Prompt + 前 $t-1$ 个已生成 Token 构成的上下文。

**第二项：价值损失 $\mathcal{L}^{\text{VF}}(\phi)$**——Critic 网络的学习目标，用均方误差让 Critic 的预测值逼近真实回报：

$$
\mathcal{L}^{\text{VF}}(\phi) = \mathbb{E}_t \left[\left(V_\phi(s_t) - V_t^{\text{target}}\right)^2\right]
$$

其中回归目标 $V_t^{\text{target}} = \hat{A}_t + V_{\phi_\text{old}}(s_t)$（GAE 估计的优势 + 旧 Critic 的预测值）。直觉上：$V_{\phi_\text{old}}(s_t)$ 是 Critic 上一轮对这个状态的估值，$\hat{A}_t$ 是 "实际表现比估值好（或差）了多少"，两者加起来就是 "实际应该值多少"。通过最小化该项，Critic 学会更准确地预估每个 Token 位置的长期收益，进而为 Actor 提供更低方差的优势估计。系数 $c_1$（通常 0.5）控制其在总损失中的权重。

**第三项：熵奖励 $S[\pi_\theta]$**——鼓励策略保持探索性：

$$
S[\pi_\theta](s_t) = -\sum_{a \in \mathcal{V}} \pi_\theta(a \mid s_t) \log \pi_\theta(a \mid s_t)
$$

其中 $\mathcal{V}$ 是整个词表（vocabulary），$|\mathcal{V}|$ 通常为 3\~13 万。熵越大表示策略越"犹豫"（分布越均匀），熵越小表示策略越"确定"（集中在少数 Token 上）。在总损失中以 $+c_2 S$ 出现（正号），因为我们最小化 loss 时正号等价于**最大化熵**——防止策略过早坍缩到确定性行为，保持探索空间。系数 $c_2$（通常 0.01）确保探索激励不会盖过策略优化信号。

**三项合在一起的直觉**：第一项让 Actor "做得更好"（强化好动作、抑制坏动作）；第二项让 Critic "看得更准"（准确估值）；第三项让策略 "别太武断"（保持多样性）。三者协同，构成了 RLHF-PPO 的完整优化目标。

下面是 RLHF-PPO 的完整代码实现。对比上一节的 Vanilla 版本，核心差异就在于上述的奖励构造（Step 2）以及免采样的 Teacher Forcing 前向传播。

**代码结构差异——为什么 RLHF 没有 `dist.sample()`？** Vanilla PPO 是**单步决策**：`dist = actor(s)` 返回一个分布对象，然后 `dist.sample()` 采一个动作、`dist.log_prob()` 算概率、`dist.entropy()` 算熵——一个 `dist` 对象就搞定一切。但语言模型生成回答是**逐 token 自回归**的：生成 $T$ 个 token 需要循环 $T$ 次，每一步的分布都依赖上一步的采样结果，不存在一个 `dist` 能代表整个回答。因此 RLHF 将采样封装在 `actor.generate()` 内部，训练时使用 **teacher forcing** 一次前向传播同时得到所有位置的 log_prob 和 entropy。

> **什么是 Teacher Forcing？** 生成回答时，模型必须**逐 token 循环**——第 $t$ 步的输入依赖第 $t-1$ 步采样出的 token，无法并行，这就是 `generate()` 很慢的原因。但在训练阶段计算 `log_probs()` 和 `compute_entropy()` 时，回答已经生成好了（`responses` 是已知的固定序列）。此时可以把完整的 `[prompt, response]` **一次性喂给模型**，让模型在每个位置预测"下一个 token 应该是什么"——因为正确答案（response 中的实际 token）已经摆在那里当"老师"，所以叫 teacher forcing。这样只需**一次前向传播**就能并行算出所有 $T$ 个位置的 logits，再从中提取 log_prob 和 entropy，比自回归循环快 $T$ 倍。
>
> **为什么"一次前向传播"就能得到所有位置的概率？** 这靠的是 Transformer 的**因果遮罩（causal mask）**。直觉上语言模型是"输入序列 → 预测下一个 token"，但实际上 Transformer **同时**为序列中每个位置都预测了"下一个 token"。以输入 `[你, 好, 吗, 我]` 为例：
>
> | 位置 | 因果遮罩允许看到的内容 | 该位置的输出 |
> |:---:|:---|:---|
> | 0 ("你") | \[你\] | 预测"你"后面的 token |
> | 1 ("好") | \[你, 好\] | 预测"好"后面的 token |
> | 2 ("吗") | \[你, 好, 吗\] | 预测"吗"后面的 token |
> | 3 ("我") | \[你, 好, 吗, 我\] | 预测"我"后面的 token |
>
> 每个位置只能 attend 到**自己和之前的位置**（通过 attention 中的三角遮罩矩阵实现），所以位置 2 的输出和"只输入 `[你, 好, 吗]`"时完全相同——不会因为后面还有 `我` 而"偷看"未来信息。这 4 个位置的预测是**并行计算**的。这就是为什么**生成时必须逐步循环**（因为下一个 token 还不知道），但**训练时可以一次性并行**（所有 token 都已知，因果遮罩保证不作弊）。

代码中 `generate()`、`log_probs()` 和 `compute_entropy()` 的实现都展示在下方：

下面用 PyTorch 风格的伪代码展示大模型 RLHF 中 PPO 的完整训练流程。这段代码展示了**一个完整的 PPO 训练循环**，包括外层的全局迭代（数据采样）和内层的多次小批量参数更新。

```python
import torch
import torch.nn.functional as F

# ============================================================
# RLHF-PPO 的四模型架构
# ============================================================
actor = LanguageModel(...)                 # 正在训练的策略 π_θ
critic = ValueHead(actor.backbone)         # 价值网络 V_φ, 通常共享 Actor 底座
ref_model = LanguageModel(...)             # 冻结的 SFT 模型 π_ref
ref_model.requires_grad_(False)            # 不参与反传与更新，仅作 KL 锚点、防止偏离人类对齐初稿太远
reward_model = RewardModel(...)            # 冻结的奖励模型 R_ψ
reward_model.requires_grad_(False)         # 奖励固定，只 forward 打分

optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-5)

clip_range = 0.2
kl_coef = 0.1      # β, KL 惩罚系数
K_epochs = 4
batch_size = 64
total_training_steps = 10000

# 辅助函数：生成小批量索引
def minibatch_indices(total, batch_size):
    """将 [0, total) 随机打乱后按 batch_size 切分，yield 每个小批次的索引"""
    perm = torch.randperm(total)
    for start in range(0, total, batch_size):
        yield perm[start : start + batch_size]

# ------ compute_token_rewards 的实现 ------
# RM 给的是整句分数（一个标量），但 GAE 需要逐 token 的奖励。
# 标准做法（InstructGPT）：中间 token 的奖励 = −KL 惩罚，
# 最后一个 token 额外加上 RM 分数，相当于"句末结算"。
def compute_token_rewards(scores, kl_penalty):
    """
    scores:     Tensor [B]       每个回答的 RM 序列级评分
    kl_penalty: Tensor [B, T]    每个 token 位置的 KL 惩罚项
    return:     Tensor [B, T]    逐 token 奖励
    """
    rewards = -kl_penalty                       # 大部分 token：奖励 = −KL（偏离越大扣分越多）
    rewards[:, -1] += scores                    # 最后一个 token：叠加 RM 整句分数（句末结算）
    return rewards

# ============================================================
# 真实的训练外层大循环
# ============================================================
for global_step in range(total_training_steps):
    
    # ============================================================
    # Step 1: 数据采集 — "环境"是: prompt → 生成回答 → 奖励模型打分
    # ============================================================
    prompts = sample_prompts(dataset)          # [B, L]  B 个 prompt，每个长度 L
    with torch.no_grad():
        # 自回归生成完整回答，并记录采样时策略的 log π_θ（供 PPO 比率与 KL 项）
        # generate() 内部逐 token 循环：logits → softmax → 采样 → 拼接，共 T 步
        responses, old_log_probs = actor.generate(prompts, return_log_probs=True)
        # responses: [B, T]        T 个生成 token
        # old_log_probs: [B, T]    每个 token 位置的 log π_θ_old
        old_values = critic(prompts, responses)  # [B, T] 每个生成位置的价值估计
        
        # 注意：这里的 ref_log_probs 和 old_log_probs 在第一轮（Step 0）时是相等的，
        # 因为此时 actor 刚刚从 ref_model 初始化。
        # 但从第二轮（Step 1）开始，actor 被更新过，而 ref_model 被冻结，两者就不再相等了。
        # old_log_probs 用于限制单次更新步长（PPO Clip），ref_log_probs 用于防止模型偏离初始状态（KL 惩罚）。
        ref_log_probs = ref_model.log_probs(prompts, responses)  # [B, T] π_ref 的对数概率
        scores = reward_model(prompts, responses)                 # [B]    RM 整句打分（序列级标量）
    
    # ============================================================
    # Step 2: 计算 KL 惩罚修正后的奖励 (vanilla PPO 没有这一步!)
    # r = R_ψ - β·(log π_θ - log π_ref)
    # ============================================================
    # β·(log π_θ − log π_ref) 近似逐 token KL 惩罚，拉大与参考模型差异则扣分
    kl_penalty = kl_coef * (old_log_probs - ref_log_probs)      # [B, T] 逐 token KL 惩罚
    # 将序列级 RM 分数与逐 token 惩罚合成 token 级奖励
    adjusted_rewards = compute_token_rewards(scores, kl_penalty)  # [B, T] 逐 token 奖励
    
    # ============================================================
    # Step 3: 优势估计 (同 vanilla PPO)
    # ============================================================
    with torch.no_grad():
        # γ=1：序列短、主要关心整句质量时通常不做时间折扣，与未来 token 权重一致
        # 不传 dones：每个 prompt→response 就是一个完整回合，无中途终止
        advantages = compute_gae(adjusted_rewards, old_values, gamma=1.0, lam=0.95)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + old_values
    
    # ============================================================
    # Step 4: 多 epoch 小批量更新 (同 vanilla PPO)
    # ============================================================
    for epoch in range(K_epochs):
        for idx in minibatch_indices(len(prompts), batch_size):
            new_log_probs = actor.log_probs(prompts[idx], responses[idx])  # 更新后策略对已生成 token 的 log π
            values = critic(prompts[idx], responses[idx])  # 当前 Critic 对同一前缀的价值预测
            entropy = compute_entropy(actor, prompts[idx], responses[idx])  # 策略熵，鼓励多样性
    
            ratio = torch.exp(new_log_probs - old_log_probs[idx])  # 与 vanilla PPO 相同的重要性权重 r
            surr1 = ratio * advantages[idx]  # 未裁剪替代项
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages[idx]  # 裁剪后的悲观项
            policy_loss = -torch.min(surr1, surr2).mean()  # PPO-Clip 策略损失
    
            value_loss = F.mse_loss(values, returns[idx])  # Critic 回归 adjusted return
            loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy  # 与 vanilla 相同的三项组合
    
            optimizer.zero_grad()
            loss.backward()
            # 防止 RLHF 长序列梯度爆炸，限制 Actor+Critic 总梯度范数
            torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()), max_norm=1.0
            )
            optimizer.step()
```

**开源代码参考：** 上述伪代码的生产级实现可参考 **[TRL](https://github.com/huggingface/trl)**（Hugging Face）和 **[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)**，两者都采用 prompt+response 拼接后一起前向传播、再通过 mask 提取 response 部分的方式计算 log_prob 和 entropy。

## PPO 在大模型微调中的痛点

PPO 凭借其简单、高效、稳定的特点，成为了 ChatGPT 等大模型 RLHF 的标准算法。

然而，从上面的四模型架构可以看出，RLHF-PPO 的**显存开销巨大**：需要同时加载 Actor、Critic、Reference、Reward 四个模型。当模型参数量飙升到百亿（10B）甚至千亿级别时，即便 Reference 和 Reward 不需要梯度，仅它们的前向推理也会占据大量显存，加上 Actor 和 Critic 的参数、梯度和优化器状态，这往往远超单张甚至多张 GPU 的显存极限。

为了解决这个问题，学术界演化出了两条不同的路线：

1. **绕过强化学习**：直接使用偏好数据优化语言模型，即 **DPO (Direct Preference Optimization)** 算法。
2. **改进强化学习**：丢弃 Critic 网络，通过组内相对评分估计优势，即 **GRPO (Group Relative Policy Optimization)** 算法。

接下来的两篇文章，我们将分别探讨这两条激动人心的前沿路线。

> 参考资料：
>
> 1. Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). *Trust Region Policy Optimization*. ICML 2015.
> 2. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
> 3. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., ... & Lowe, R. (2022). *Training language models to follow instructions with human feedback*. NeurIPS 2022.
> 4. Amari, S. (1998). *Natural Gradient Works Efficiently in Learning*. Neural Computation, 10(2), 251-276.
> 5. [Trust Region Methods: From REINFORCE to TRPO to PPO](https://sesen.ai/blog/trust-region-methods-reinforce-trpo-ppo)
> 6. [Natural Gradient Descent — Agustinus Kristiadi](https://agustinus.kristia.de/blog/natural-gradient/)

> 下一篇：[笔记｜强化学习（三）：大模型对齐的另一条路：DPO (Direct Preference Optimization)](/chengYi-xun/posts/53-dpo/)
