---
title: 笔记｜生成模型（十七）：信任区域与近端策略优化 (从 TRPO 到 PPO)
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

这里的学习率 $\alpha$ 决定了更新的**步长**。策略梯度只给出了"哪个方向能提升策略"的指引，但**完全没有告诉我们"沿这个方向能安全地走多远"**——这种信息的缺失，是所有策略梯度方法不稳定性的根源。下面我们从具体的例子出发，逐步揭示这个问题的三个层面。

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

## 问题小结

总结来看，策略梯度方法面临的步长困境有三个层面，逐层叠加：

1. **优势估计的前提被破坏**：$\hat{A}$ 在旧策略下计算，策略大幅变化后估计失效
2. **参数-策略映射的非线性**：固定学习率无法在所有方向上保证安全步长
3. **正反馈崩溃循环**：坏更新 → 坏数据 → 更坏的更新 → 不可逆转的崩溃

这三个问题对**所有策略梯度方法普遍存在**，包括最基础的 REINFORCE。REINFORCE 的特殊困境在于：它**没有任何步长控制机制**，只能依赖手动调节的学习率，但正如第二层所分析的，没有任何固定学习率能在所有情况下都安全。

我们需要一种方法来**直接限制策略的变化幅度**（而非参数的变化幅度），保证模型稳扎稳打地变好。接下来的两节，我们将看到 TRPO 和 PPO 分别如何解决这个问题。

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

**"替代"的数学含义。** 替代目标和真实目标 $J(\theta)$ 之间有严格的数学关系，就像泰勒展开中的一阶近似和原函数的关系：

1. **在旧策略处两者相等**：当 $\theta = \theta_\text{old}$ 时，$r_t = 1$，$\mathcal{L}^{\text{CPI}}(\theta_\text{old}) = \mathbb{E}_t[\hat{A}_t] = 0$。真实目标的改进量同样为 $0$（策略没变，表现不变）。
2. **在旧策略处梯度相同**：$\nabla_\theta \mathcal{L}^{\text{CPI}} \big|_{\theta=\theta_\text{old}} = \nabla_\theta J(\theta) \big|_{\theta=\theta_\text{old}}$。这保证了在旧策略的邻域内，替代目标的上升方向就是真实目标的上升方向。
3. **远离旧策略后偏离加剧**：当 $\theta$ 偏离 $\theta_\text{old}$ 越远，替代目标对真实表现的近似就越不准确——这正是后文 KL 约束和裁剪机制存在的根本原因。

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

**TRPO 的缺点**：TRPO 在理论上非常严谨，保证了策略的单调改进。但它的求解过程要求**二阶优化**，这在大规模深度网络上带来了严重的实际困难。

**为什么不能只用一阶梯度？** 前文"第二层"已经指出，参数空间和策略空间的几何结构截然不同。TRPO 的约束加在**概率分布空间**上（KL 散度），而梯度下降操作在**参数空间**中。一阶梯度只能告诉你"哪个方向能提升目标函数"，但无法告诉你"沿这个方向走多远，分布才会变化到 KL 约束的边界"。

还是用前面的徒步比喻来说明：假设 KL 约束是"只能离当前位置走 100 米"。一阶梯度告诉你"正北方向最陡"，于是你往北走 100 米。但地形的不均匀性意味着——往北走 1 米海拔就升 50 米（分布极敏感），往东走 100 米海拔才升 1 米（分布不敏感）——径直往北走 100 米，等于在分布空间中远远飞出了信任区域。

**Fisher 信息矩阵** $F$ 恰恰描述了这种"地形"——参数空间到分布空间的**局部映射关系**。$F$ 在某个方向的值越大，说明分布对该方向的参数变化越敏感（只能走小步）；值越小，说明越不敏感（可以走大步）。下面我们通过严格推导，展示自然梯度 $F^{-1}g$ 是如何从 TRPO 的约束优化问题中自然涌现的。

### 自然梯度 $F^{-1}g$ 的推导

TRPO 的优化目标是：在 KL 散度不超过 $\delta$ 的约束下，找到使替代目标最大化的参数更新方向 $d = \theta - \theta_\text{old}$。将两个函数在 $\theta_\text{old}$ 处做泰勒展开：

**替代目标**的一阶展开（$d$ 很小时，一阶近似足够）：
$$\mathcal{L}(\theta_\text{old} + d) \approx \mathcal{L}(\theta_\text{old}) + g^T d$$
其中 $g = \nabla_\theta \mathcal{L}|_{\theta_\text{old}}$ 是策略梯度。

**KL 散度**的二阶展开。记 $D(\theta) := \text{KL}[\pi_{\theta_\text{old}} \| \pi_\theta]$，在 $\theta = \theta_\text{old}$ 处对 $d = \theta - \theta_\text{old}$ 做标准泰勒展开：

$$D(\theta_\text{old} + d) = \underbrace{D(\theta_\text{old})}_{\text{零阶项}} + \underbrace{\nabla_\theta D(\theta)\big|_{\theta_\text{old}}^T d}_{\text{一阶项}} + \underbrace{\frac{1}{2} d^T \nabla_\theta^2 D(\theta)\big|_{\theta_\text{old}} d}_{\text{二阶项}} + O(\|d\|^3)$$

下面逐项说明为什么前两项恰好为零，只剩下二阶项：

**(1) 零阶项 $= 0$**：$D(\theta_\text{old}) = \text{KL}[\pi_{\theta_\text{old}} \| \pi_{\theta_\text{old}}] = 0$，自己和自己的 KL 散度为零。

**(2) 一阶项 $= 0$**：这一步的关键不是"极值点性质"，而是一个适用于任意 $\theta$ 的恒等式——**得分函数（Score Function）的期望恒为零**：

$$\mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(x)] = 0$$

**证明**：从期望定义出发，用对数导数恒等式 $\nabla_\theta \log \pi_\theta = \frac{\nabla_\theta \pi_\theta}{\pi_\theta}$ 代入：

$$\mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(x)] = \sum_x \pi_\theta(x) \cdot \frac{\nabla_\theta \pi_\theta(x)}{\pi_\theta(x)} = \sum_x \nabla_\theta \pi_\theta(x) = \nabla_\theta \underbrace{\sum_x \pi_\theta(x)}_{=1} = \nabla_\theta(1) = 0$$

最后一步利用了概率分布的归一化约束 $\sum_x \pi_\theta(x) = 1$。直觉上，$\nabla_\theta \log \pi_\theta$ 告诉我们"参数 $\theta$ 往哪个方向调，能增加 $x$ 的概率"。但在自身分布下取期望，"增加某些 $x$ 概率"的推力和"减少另一些 $x$ 概率"的推力恰好抵消——因为概率总和始终为 1。

KL 散度的一阶导数 $\nabla_\theta D(\theta)\big|_{\theta_\text{old}} = -\mathbb{E}_{\pi_{\theta_\text{old}}}[\nabla_\theta \log \pi_\theta]\big|_{\theta = \theta_\text{old}}$，代入 $\theta = \theta_\text{old}$ 后正好套用上述恒等式，所以一阶项 $= 0$。

**(3) 二阶项的 Hessian $= F$（Fisher 信息矩阵）**：KL 散度对 $\theta$ 的 Hessian 在 $\theta_\text{old}$ 处等于：

$$\nabla_\theta^2 D(\theta)\big|_{\theta_\text{old}} = F := \mathbb{E}_{\pi_{\theta_\text{old}}}\left[\nabla_\theta \log \pi_\theta \cdot (\nabla_\theta \log \pi_\theta)^T\right]\bigg|_{\theta_\text{old}}$$

这就是 **Fisher 信息矩阵**。它是一个半正定矩阵，描述了策略分布对参数变化的局部敏感程度。

**三项代回**，得到 KL 散度的严格二阶近似：
$$\text{KL}[\pi_{\theta_\text{old}} \| \pi_{\theta_\text{old}+d}] = 0 + 0 + \frac{1}{2} d^T F d + O(\|d\|^3) \approx \frac{1}{2} d^T F d$$

$\frac{1}{2}d^TFd$ 是一个二次型，描述了"沿方向 $d$ 走一步，策略分布会偏移多少"。

将近似代入 TRPO 的约束优化问题，得到：
$$\max_d \; g^T d \quad \text{s.t.} \quad \frac{1}{2} d^T F d \le \delta$$

用**拉格朗日乘子法**求解。构造拉格朗日函数 $L(d, \lambda) = g^T d - \lambda (\frac{1}{2} d^T F d - \delta)$，对 $d$ 求导并令其为零：
$$\nabla_d L = g - \lambda F d = 0 \implies d^* = \frac{1}{\lambda} F^{-1} g$$

**这就是自然梯度的由来**：最优更新方向 $d^*$ 正比于 $F^{-1}g$。常数因子 $\frac{1}{\lambda}$ 可以通过代入 KL 约束 $\frac{1}{2}(d^*)^T F d^* = \delta$ 确定（这就是代码中计算 `max_step` 的那一步）。

直觉上，$F^{-1}$ 对梯度做了"地形校正"：在分布敏感的方向上（$F$ 的特征值大），$F^{-1}$ 会压缩步长；在分布不敏感的方向上（$F$ 的特征值小），$F^{-1}$ 会放大步长——使得最终的更新在**分布空间中各方向均匀推进**，最高效地利用有限的 KL 预算。

> **延伸阅读：** 关于自然梯度的更完整推导（包括黎曼几何视角），推荐 [Natural Gradient Descent — Agustinus Kristiadi](https://agustinus.kristia.de/blog/natural-gradient/)。

### 共轭梯度法：不存 $F$，也能算 $F^{-1}g$

上面推导出最优方向是 $F^{-1}g$，但计算它等价于求解线性方程组 $Fx = g$。

**问题是：$F$ 太大了，根本存不下。** $F$ 是一个 $d \times d$ 的矩阵（$d$ 是参数数量，可达百万甚至上亿）。对于一个 100 万参数的网络，$F$ 有 $10^{12}$（一万亿）个元素，任何计算机都装不下，更不用说求逆了。

**共轭梯度法（Conjugate Gradient, CG）** 提供了一条捷径：**不需要知道整个 $F$，只需要能回答"给定任意方向 $v$，$Fv$ 等于多少？"这一个问题。**

用一个比喻来理解。想象你蒙着眼睛在一个碗形的山谷里找最低点（即 $Fx = g$ 的解）。你看不到整个地形（$F$ 太大了），但你有一根"探测杆"：把它插进地面的任意方向 $v$，它会告诉你"这个方向上地面有多陡"（这就是 $Fv$）。

**最笨的方法**是沿最陡方向一直走（梯度下降），但你会发现自己走出一条**锯齿形**路径——往东走一步，发现南北方向更陡了，于是拐弯往南走，结果又发现东西方向更陡了……反反复复，很慢才能到达谷底。

**共轭梯度法的聪明之处**在于：每一步不只是沿"最陡"方向走，而是选择一个**经过修正的方向**，使得这一步的进展**永远不会被后续步骤撤销**。数学上，这通过让搜索方向之间满足 $F$-正交（$p_i^T F p_j = 0$，称为"共轭"）来实现。结果是，CG 不走回头路，理论上最多 $d$ 步就能精确到达谷底；在实际 TRPO 中，**只需约 10 步就能得到足够好的近似解**。

每一步 CG 只需要做一次"插探测杆"操作（计算 $Fv$）。在 TRPO 中，$Fv$ 可以通过**两次反向传播**高效得到——第一次对 KL 散度求梯度，第二次对梯度和 $v$ 的内积再求梯度（参考 [Efficiently Computing the Fisher Vector Product in TRPO](https://www.telesens.co/2018/06/09/efficiently-computing-the-fisher-vector-product-in-trpo/)）。因此，10 步 CG 需要约 20 次反向传播——相比 PPO 每次更新只需 1 次反向传播，计算开销高出一个量级。

> **想深入了解 CG 的数学细节？** 推荐 [An Introduction to the Conjugate Gradient Method Without the Agonizing Pain](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)（CMU 经典教程），以及 [Wikipedia: Derivation of the Conjugate Gradient Method](https://en.wikipedia.org/wiki/Derivation_of_the_conjugate_gradient_method)。

**复杂网络结构的问题。** 这一要求使得 TRPO 难以与现代深度网络结合：**Dropout** 在每次前向传播中随机丢弃神经元，使有效网络结构不断变化，FIM 在随机子网络上的估计变得不稳定；**BatchNorm** 的归一化统计量依赖于当前 mini-batch 中的所有样本，引入了样本间的相互依赖，而 FIM 的推导假设样本独立；**Transformer** 中的多头注意力、残差连接等组件使得损失函数的 Hessian 结构极其复杂，二阶导数的计算既慢又不稳定。这些因素叠加在一起，使得 TRPO 在参数动辄上亿的现代深度网络上几乎无法实际使用。

在下面的 TRPO 和 PPO 代码中，优势函数 $\hat{A}_t$ 的估计都使用 **GAE（Generalized Advantage Estimation）**，其理论推导和代码实现详见[上一篇的 GAE 章节](/chengYi-xun/posts/17-rl-basics/#广义优势估计（GAE）：蒙特卡洛与-TD-的统一框架)。这里只回顾核心公式——GAE 通过参数 $\lambda$ 在偏差（$\lambda \to 0$，单步 TD）和方差（$\lambda \to 1$，蒙特卡洛）之间权衡：

$$
\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \, \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

## TRPO 的实现

下面用 PyTorch 风格的伪代码展示 TRPO 的完整训练流程。注意其中 Step 4 和 Step 5 的二阶优化和线搜索——这正是 TRPO 的计算瓶颈所在。

```python
import copy  # 深拷贝旧策略，供 rollout 与 KL 基准使用
import torch
import torch.nn.functional as F  # 价值损失等常用算子

# ============================================================
# 模型定义
# actor:  策略网络 π_θ(a|s), 输入状态, 输出动作概率分布
# critic: 价值网络 V_φ(s), 输入状态, 输出标量价值
# ============================================================
actor = ActorModel(state_dim, action_dim)   # 当前待优化的策略网络
critic = CriticModel(state_dim)             # 估计 V(s)，供 GAE 与回报回归
old_actor = copy.deepcopy(actor)   # 旧策略的冻结副本
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)  # 仅 Critic 用一阶优化
delta = 0.01  # KL 约束阈值

# ============================================================
# Step 1: 数据采集 (Rollout)
# 用旧策略 π_old 与环境交互, 采集一批轨迹 (trajectory)
# 每条数据是一个五元组: (状态 s_t, 动作 a_t, 奖励 r_t, log π_old(a_t|s_t), done_t)
# ============================================================
buffer = []                    # 存放本批轨迹样本
state = env.reset()            # 环境初始状态
for t in range(T):
    with torch.no_grad():      # 采样阶段不反传，节省计算
        dist = old_actor(state)  # 用旧策略 π_old 得到动作分布，保证数据与行为策略一致
        action = dist.sample()   # 从该分布采样动作 a_t
        log_prob = dist.log_prob(action)  # 记录 log π_old(a_t|s_t)，供后续重要性比率
    next_state, reward, done = env.step(action)  # 环境转移与即时奖励
    buffer.append((state, action, reward, log_prob, done))  # 同时存 done，供 GAE 切割回合边界
    state = next_state if not done else env.reset()  # 回合结束则重置

states, actions, rewards, old_log_probs, dones = collate(buffer)  # 整理为张量批次

# ============================================================
# Step 2: 优势估计 (GAE)
# ============================================================
with torch.no_grad():
    values = critic(states)  # 各状态价值估计 V_φ(s)
    # GAE：结合多步回报与 TD 残差，降低方差、估计 A^π 的近似 Â
    advantages = compute_gae(rewards, values, gamma=0.99, lam=0.95, dones=dones)
    # 优势标准化：零均值单位方差量级，稳定策略梯度尺度
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = advantages + values  # 用作 Critic 回归目标（广义回报）

# ============================================================
# Step 3: 计算替代目标和 KL 散度
# ============================================================
dist = actor(states)  # 当前参数 θ 下的新策略分布
new_log_probs = dist.log_prob(actions)  # log π_θ(a_t|s_t)
# 重要性采样比率 r = π_θ/π_old，用对数差避免数值下溢
ratio = torch.exp(new_log_probs - old_log_probs)
surrogate = (ratio * advantages).mean()  # 替代目标 L^CPI = E[r·Â]，近似策略改进

old_dist = old_actor(states)  # 旧策略在相同状态上的分布
# 批量平均 KL(π_old || π_θ)：度量新策略相对旧策略偏离多少（信任区域约束依据）
kl = torch.distributions.kl_divergence(old_dist, dist).mean()

# ============================================================
# Step 4: 自然梯度 — 求解 F⁻¹g (二阶优化!)
# ============================================================
# 4a. 策略梯度 g = ∇_θ L^CPI
g = flatten(torch.autograd.grad(surrogate, actor.parameters()))  # 将各层梯度拼成一向量

# 4b. Fisher-向量积 Fv (避免显式构建 d×d 的 F 矩阵)
def fisher_vector_product(v):
    # 先对 KL 求梯度，再与 v 做内积后对 θ 再求导，得到 Fv（Fisher×方向 v）
    kl_grad = flatten(torch.autograd.grad(kl, actor.parameters(), create_graph=True))
    return flatten(torch.autograd.grad(kl_grad @ v, actor.parameters())) + 0.1 * v  # 0.1*v 为阻尼项，数值稳定

# 4c. 共轭梯度法近似求解 F⁻¹g (迭代 10 次)
# 不显式求逆 F，迭代解 F·x≈g，得到自然梯度方向 x∝F⁻¹g
step_dir = conjugate_gradient(fisher_vector_product, g, max_iter=10)

# ============================================================
# Step 5: 线搜索确定步长 (确保 KL ≤ δ 且目标提升)
# ============================================================
# 由 KL≈½ d^T F d 与约束 δ 反推最大步长（自然梯度步长的尺度）
max_step = torch.sqrt(2 * delta / (step_dir @ fisher_vector_product(step_dir)))
old_params = flatten_params(actor).detach()  # 线搜索起点：当前 Actor 参数

for shrink in [1.0, 0.5, 0.25, 0.125]:  # 回溯系数：逐步缩小步长
    new_params = old_params + shrink * max_step * step_dir  # 沿自然梯度方向试探更新
    assign_params(actor, new_params)
    new_surr = compute_surrogate(actor, states, actions, old_log_probs, advantages)  # 试探点的替代目标
    new_kl = compute_kl(old_actor, actor, states)  # 试探点的平均 KL
    if new_kl <= delta and new_surr >= surrogate:  # 同时满足 KL 预算且 surrogate 不下降则接受
        break
    assign_params(actor, old_params)  # 不满足则回退

# ============================================================
# Step 6: 更新 Critic (标准一阶优化)
# ============================================================
critic_optimizer.zero_grad()
F.mse_loss(critic(states), returns).backward()  # Critic 拟合回报/价值目标
critic_optimizer.step()

# Step 7: 同步旧策略
old_actor.load_state_dict(actor.state_dict())  # 下一轮 rollout 以更新后的 π 为行为策略基准
```

注意：上面的代码是 vanilla RL（经典强化学习）场景中的 TRPO。**TRPO 从未被应用于大模型 RLHF**——原因正是前文详述的二阶优化瓶颈：Fisher 矩阵和共轭梯度法在数十亿参数的语言模型上根本无法计算。当 2022-2023 年 RLHF 成为大模型对齐的核心技术时，学界直接选择了 PPO 而跳过了 TRPO。这正是 PPO 诞生的全部意义——用一阶裁剪替代二阶约束，使信任区域方法能够扩展到工业级大模型。关于三种方法（REINFORCE、TRPO、PPO）在同一环境下的直观对比，可参考 [Trust Region Methods: From REINFORCE to TRPO to PPO](https://sesen.ai/blog/trust-region-methods-reinforce-trpo-ppo)，该博客提供了从零实现三种算法并在 CartPole 上对比的完整代码。

---

# PPO：大道至简的"裁剪"艺术

**核心思考出发点**：TRPO 虽然理论完美，但求解 KL 约束所需的二阶优化（Fisher 矩阵、共轭梯度）算得太慢了，根本没法用在参数动辄上亿的深度神经网络上。能不能用一种极其简单的方法，达到和 TRPO 一样的"不出圈"效果呢？

OpenAI 在 2017 年给出了答案：**PPO (Proximal Policy Optimization)**（[原始论文](https://arxiv.org/abs/1707.06347)）。PPO 的核心思路是：用简单的一阶裁剪操作替代 TRPO 的二阶 KL 约束。PPO 不去精确求解"KL 约束内的最优方向"（那需要 Fisher 矩阵来理解分布空间的几何），而是直接用 $\text{clip}(r, 1-\epsilon, 1+\epsilon)$ 限制每个动作的概率比率——这是一个粗糙但有效的近似，不需要理解参数-分布映射的精确关系。因此 PPO **只需要标准的一阶梯度**，可以直接使用 Adam 等优化器，对网络结构没有任何限制——Dropout、BatchNorm、多头注意力、残差连接等组件都不影响其正常工作。

## 裁剪目标函数 (Clipped Surrogate Objective)

**先看例子**：还是面试助手的场景。$\epsilon = 0.2$ 意味着我们允许重要性比率 $r$ 在 $[0.8, 1.2]$ 的范围内变化。

回到前面的数字——旧策略 70% 选"讲项目经历"，优势 $\hat{A} = +3$。

| 新策略概率 | $r = \pi_\text{new}/\pi_\text{old}$ | 裁剪前的信号 $r \cdot \hat{A}$ | 裁剪后的信号 | 发生了什么？ |
|:---:|:---:|:---:|:---:|:---|
| 77% | 1.1 | 3.3 | 3.3 | 在安全范围内，正常更新 |
| 84% | 1.2 | 3.6 | 3.6 | 恰好在边界，正常更新 |
| 91% | 1.3 | 3.9 | **3.6** | $r > 1.2$，被裁剪！梯度变为 0，**不许再往上推了** |
| 99% | 1.41 | 4.24 | **3.6** | 远超上界，完全被拦住 |

PPO 的裁剪机制就像一个"刹车"：当新策略试图偏离旧策略太远时（$r$ 超出 $[1-\epsilon, 1+\epsilon]$），梯度被截断为 0，阻止继续偏移。

**一般化的算法原理**：

PPO 的核心创新在于其裁剪目标函数：
$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_{t} \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$
其中 $\epsilon$ 是一个超参数（通常设为 0.2）。

这个公式有三个部分：

1. **第一项**：$r_t(\theta) \hat{A}_t$ 是正常的替代目标（与 TRPO 相同）。
2. **第二项**：$\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t$ 将重要性比率强行截断在 $[0.8, 1.2]$ 之间。
3. **取最小值 (min)**：构建悲观的下界（Pessimistic Bound）。

**分情况讨论**（结合面试的例子）：

- **当 $\hat{A}_t > 0$ 时**（如"讲项目经历"，$\hat{A} = +3$）：好动作，我们想增加其概率。
  - 如果 $r_t(\theta)$ 增大到超过 $1+\epsilon$（概率涨太多了），裁剪项生效，梯度变为 0。**见好就收，不许一次涨太多。**
- **当 $\hat{A}_t < 0$ 时**（如"讲兴趣爱好"，$\hat{A} = -3$）：坏动作，我们想降低其概率。
  - 如果 $r_t(\theta)$ 减小到低于 $1-\epsilon$（概率降太多了），裁剪项生效，梯度变为 0。**防止矫枉过正，不许一次降太多。**

通过这种简单的裁剪机制，PPO 成功地将新策略限制在旧策略的"信任区域"内，无需复杂的二阶优化计算，直接使用 Adam 等一阶优化器即可高效训练。从重要性采样的角度看，裁剪 $r \in [1-\epsilon, 1+\epsilon]$ 同时将重要性权重限制在了安全范围内，防止了 IS 估计的方差爆炸。这使得 PPO 可以对同一批采样数据安全地进行多个 epoch 的小批量更新（通常 3\~10 个 epoch），相比 REINFORCE 每批数据只用一次就丢弃，样本效率提升了数倍——这正是 PPO 能够高效训练大规模模型的关键之一。

## PPO 的完整损失函数与代码实现

在大模型微调中，PPO 是 **RLHF** 流程的核心算法。RLHF 的基本思路是：先用人类标注的偏好数据（"回答 A 比回答 B 好"）训练一个**奖励模型（Reward Model）**，然后将该奖励模型作为环境的"评委"，用 PPO 训练语言模型生成更符合人类偏好的回答。在这一流程中，PPO 的总损失函数通常包含三部分：
$$
\mathcal{L}^{\text{PPO}}(\theta) = \mathcal{L}^{\text{CLIP}}(\theta) - c_1 \mathcal{L}^{\text{VF}}(\theta) + c_2 S[\pi_\theta](s_t)
$$

1. **策略损失 $\mathcal{L}^{\text{CLIP}}$**：即上文推导的裁剪目标函数：

   $$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[\min\!\left(r_t(\theta)\,\hat{A}_t,\;\text{clip}\!\left(r_t(\theta),\,1-\epsilon,\,1+\epsilon\right)\hat{A}_t\right)\right]$$

   其中 $r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是新旧策略的概率比率，$\hat{A}_t$ 是 GAE 计算的优势估计。$\min$ 加上裁剪构成了"悲观下界"——当优势为正（好动作）时阻止概率涨过 $1+\epsilon$，当优势为负（坏动作）时阻止概率降过 $1-\epsilon$。这确保了每次更新不会偏离旧策略太远。

2. **价值损失 $\mathcal{L}^{\text{VF}}$**：Critic 网络的回归目标，用均方误差衡量 Critic 预测值与实际回报之间的差距：

   $$\mathcal{L}^{\text{VF}}(\theta) = \mathbb{E}_t \left[\left(V_\theta(s_t) - V_t^{\text{target}}\right)^2\right]$$

   其中 $V_\theta(s_t)$ 是 Critic 对状态 $s_t$ 的价值预测，$V_t^{\text{target}}$ 是实际折扣回报（或 GAE 目标值 $\hat{A}_t + V_{\theta_{\text{old}}}(s_t)$）。通过最小化该项，Critic 学会准确预估每个状态的长期收益，从而为 Actor 提供更低方差的优势估计 $\hat{A}_t$。系数 $c_1$（通常 0.5）控制价值损失在总损失中的权重。

3. **熵奖励（Entropy Bonus）$S[\pi_\theta]$**：策略熵的定义为：

   $$S[\pi_\theta](s_t) = -\sum_{a \in \mathcal{A}} \pi_\theta(a \mid s_t) \log \pi_\theta(a \mid s_t)$$

   其中 $\mathcal{A}$ 是**动作空间**（所有可选动作的集合），$a$ 是其中的单个动作。在传统 RL 中，$\mathcal{A}$ 是有限动作集（如游戏的上/下/左/右按键，$|\mathcal{A}|=4$）；在 LLM-RL 中，$\mathcal{A}$ 就是整个词表（vocabulary），$|\mathcal{A}|$ 为词表大小（3~13 万），每个动作 $a$ 对应一个 token，而状态 $s_t$ 则是当前已生成的上下文。

   熵衡量的是**策略在当前状态下对动作空间的概率分布的不确定性**，其取值范围为 $[0, \log|\mathcal{A}|]$：

   - **上限 $\log|\mathcal{A}|$（均匀分布）**：当 $\pi_\theta(a|s_t) = \frac{1}{|\mathcal{A}|}$ 对所有 $a$ 成立时，代入得 $S = -\sum_{a=1}^{|\mathcal{A}|} \frac{1}{|\mathcal{A}|}\log\frac{1}{|\mathcal{A}|} = -|\mathcal{A}| \cdot \frac{1}{|\mathcal{A}|} \cdot (-\log|\mathcal{A}|) = \log|\mathcal{A}|$。可以用拉格朗日乘数法证明，均匀分布是在 $\sum_a \pi(a)=1$ 约束下的**全局最大熵分布**。
   - **下限 $0$（确定性策略）**：当存在某个 $a_k$ 使得 $\pi_\theta(a_k|s_t)=1$，其余 $\pi_\theta(a|s_t)=0$ 时，代入得 $S = -(1\cdot\log 1 + 0\cdot\log 0 + \cdots) = 0$。这里约定 $0\log 0 = 0$，与极限 $\lim_{x\to 0^+} x\log x = 0$ 一致。

   损失函数中以 $+c_2 \cdot S$ 的形式出现（注意正号），因为最小化 loss 时该项等价于**最大化熵**——鼓励策略保持随机性、避免过早坍缩到确定性行为，这正是**探索与利用（exploration vs. exploitation）权衡**的体现。系数 $c_2$（通常 0.01）较小，确保探索激励不会盖过策略优化信号。

下面用 PyTorch 风格的伪代码展示 PPO 的完整训练流程。对比上面 TRPO 的实现，可以清楚地看到 PPO 的核心简化：**没有 Fisher 矩阵、没有共轭梯度、没有线搜索**——全部替换为简单的裁剪 + 标准 Adam 优化器。

```python
import torch
import torch.nn.functional as F

# ============================================================
# 模型定义 (与 TRPO 相同: Actor + Critic)
# ============================================================
actor = ActorModel(state_dim, action_dim)
critic = CriticModel(state_dim)
optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
clip_range = 0.2      # ε：比率 r 允许偏离 1 的幅度，对应 clip 区间 [1-ε, 1+ε]
vf_coef = 0.5         # 价值损失权重 c₁
entropy_coef = 0.01   # 熵奖励权重 c₂
K_epochs = 4          # 同一批 rollout 数据做 K 轮 epoch，配合 clip 复用样本、提高样本效率

# ============================================================
# Step 1: 数据采集 (同 TRPO)
# ============================================================
buffer = []
state = env.reset()
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
# minibatch_indices 把全量数据随机拆成若干小批次，每次返回一组整数索引
# 例如: len(states)=1024, batch_size=64 → 每 epoch 产生 16 组 idx
# idx 的 shape 是 [batch_size]，即 [64]，内容是随机打乱的样本下标
def minibatch_indices(total, batch_size):
    """将 [0, total) 随机打乱后按 batch_size 切分，yield 每个小批次的索引"""
    perm = torch.randperm(total)                  # 随机排列所有样本下标
    for start in range(0, total, batch_size):     # 按 batch_size 步进切片
        yield perm[start : start + batch_size]    # 返回一组索引，shape = [batch_size]

for epoch in range(K_epochs):
    for idx in minibatch_indices(len(states), batch_size=64):
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

**RLHF-PPO 的奖励公式**因此需要对原始 RM 分数做 KL 修正：
$$
r_t = R_\psi(\text{prompt}, \text{response}) - \beta \cdot \text{KL}[\pi_\theta \| \pi_\text{ref}]
$$
其中 $\beta$ 控制 KL 惩罚的强度：$\beta$ 越大，Actor 越不敢偏离 Reference。

下面是 RLHF-PPO 的完整实现。对比上面的 vanilla 版本，核心差异在 Step 1（数据采集方式）和 Step 2（KL 惩罚修正奖励）。

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
    # 参考策略在相同 (prompt, response) 上的 log π_ref，用于逐 token KL
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

# ------ actor.generate() 的内部实现 ------
# Vanilla PPO 是单步决策：dist = actor(s) → action = dist.sample()，一步完成。
# 而语言模型生成回答是逐 token 自回归的——每个位置产生一个分布，
# 后一个位置依赖前一个位置的采样结果，因此不存在一个 dist 对象能概括整个回答。
# generate() 将采样循环封装在内部：
def generate(self, prompts, return_log_probs=True, max_new_tokens=256):
    """
    prompts: [B, L]
    return:  responses [B, T], log_probs [B, T]
    """
    tokens, log_probs_list = [], []
    input_ids = prompts                                     # [B, L]
    for t in range(max_new_tokens):
        logits = self.forward(input_ids).logits[:, -1, :]   # 取最后位置 → [B, V]
        probs = torch.softmax(logits, dim=-1)               # [B, V]
        token = torch.multinomial(probs, num_samples=1)     # 采样 → [B, 1]
        lp = torch.log_softmax(logits, dim=-1)              # [B, V]
        token_lp = lp.gather(dim=-1, index=token)           # 取对应 token → [B, 1]
        tokens.append(token)
        log_probs_list.append(token_lp)
        input_ids = torch.cat([input_ids, token], dim=-1)   # 自回归拼接
    return torch.cat(tokens, dim=-1), torch.cat(log_probs_list, dim=-1)

# ------ compute_entropy() 的实现 ------
# 策略熵 S[π_θ](s) = −Σ_a π_θ(a|s) log π_θ(a|s)，衡量策略在当前状态下的"不确定性"。
# 熵越大 → 策略越随机（均匀分布时熵最大）；熵越小 → 策略越确定（概率集中在一个 token）。
# 在 PPO 损失中以负号出现（- entropy_coef * entropy），等价于鼓励策略保持多样性。
# Vanilla PPO 直接调用 dist.entropy()；RLHF 中因为动作空间是整个词表（3~13万 token），
# 需要从 logits 手动计算。
def compute_entropy(actor, prompts, responses):
    """
    计算策略在 response 部分、所有生成位置上的平均熵

    actor:     语言模型 π_θ
    prompts:   [B, L]     prompt token ids —— 必须传入，因为 response 位置的概率分布
                          取决于 prompt 提供的上下文（prompt 就是 RL 中的"状态 s"）
    responses: [B, T]     response token ids
    return:    标量，response 区域所有 batch × token 位置的平均熵
    """
    # prompt 和 response 拼接后一起输入：prompt 提供条件，response 提供预测目标
    input_ids = torch.cat([prompts, responses], dim=-1)  # [B, L+T] 拼接完整序列
    logits = actor(input_ids).logits                     # [B, L+T, V]  V = 词表大小
    # 只取 response 部分的 logits（teacher forcing：位置 L-1 到 L+T-2 的输出预测位置 L 到 L+T-1 的 token）
    resp_logits = logits[:, prompts.size(1)-1:-1, :]     # [B, T, V]
    probs = torch.softmax(resp_logits, dim=-1)           # [B, T, V]
    log_probs = torch.log_softmax(resp_logits, dim=-1)   # [B, T, V]  数值稳定版 log
    token_entropy = -(probs * log_probs).sum(dim=-1)     # [B, T]  逐位置: −Σ_v p(v) log p(v)
    return token_entropy.mean()                          # 标量

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

> 下一篇：[笔记｜生成模型（十八）：大模型对齐的另一条路：DPO (Direct Preference Optimization)](/chengYi-xun/posts/19-dpo/)
