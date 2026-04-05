---
title: 笔记｜生成模型（十六）：强化学习基础与策略梯度
date: 2025-08-16 10:00:00
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

> 本文为大模型对齐与强化学习系列的第一篇，主要回顾强化学习（RL）的基础概念，并严格推导策略梯度定理，从最基础的 REINFORCE 算法讲起，逐步引入 Actor-Critic 架构，为后续深入理解 PPO 和 GRPO 打下坚实的理论基础。

# 强化学习基础概念：从"训狗"说起

对于纯小白读者来说，理解强化学习（Reinforcement Learning, RL）最直观的例子就是**"训狗"**。
- **智能体（Agent）**：就是那只狗（在 AI 中就是我们的模型）。
- **环境（Environment）**：狗所处的现实世界。
- **状态（State）**：狗当前看到的画面、听到的口令（比如你喊"坐下"）。
- **动作（Action）**：狗做出的反应（比如坐下、趴下、或者跑开）。
- **奖励（Reward）**：如果狗做对了，你给它一块肉（正奖励）；做错了，你呵斥它（负奖励或零奖励）。

强化学习的核心目标，就是让这只狗（模型）在不断的"尝试-犯错-获得奖励"的过程中，自己摸索出一条规律：**在什么情况下，做什么动作，能吃到最多的肉**。这条规律，在数学上就叫做**策略（Policy）**。

![强化学习智能体与环境交互循环](/chengYi-xun/img/rl_loop.png)

## 马尔可夫决策过程 (MDP)

强化学习问题通常被建模为**马尔可夫决策过程（Markov Decision Process, MDP）**。一个 MDP 可以用一个元组 $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$ 来表示：

- $\mathcal{S}$：状态空间（State Space），包含所有可能的状态 $s$。
- $\mathcal{A}$：动作空间（Action Space），包含所有可能的动作 $a$。
- $\mathcal{P}$：状态转移概率（Transition Probability），$\mathcal{P}(s'|s, a)$ 表示在状态 $s$ 执行动作 $a$ 后转移到状态 $s'$ 的概率。
- $\mathcal{R}$：奖励函数（Reward Function），$\mathcal{R}(s, a)$ 表示在状态 $s$ 执行动作 $a$ 后获得的即时奖励。
- $\gamma$：折扣因子（Discount Factor），$\gamma \in [0, 1]$，用于权衡即时奖励与未来奖励的重要性。

智能体的目标是学习一个策略 $\pi(a|s)$，即在状态 $s$ 下选择动作 $a$ 的概率分布，使得期望的累积折扣奖励（Return）最大化：
$$
J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
$$
其中，**$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)$ 表示一条完整的交互轨迹（Trajectory）**。

## 价值函数与优势函数：怎么判断一个动作是好是坏？

在训狗的过程中，狗不仅要看眼前的肉，还要考虑长远的利益（比如现在乖乖坐下，等会儿可能有大骨头）。为了评估策略的好坏，我们引入两个重要的价值函数：

1. **状态价值函数（State-Value Function）** $V^\pi(s)$：表示在状态 $s$ 下，遵循策略 $\pi$ 所能获得的期望累积奖励。它实际上是该状态下所有可能动作的动作价值 $Q^\pi(s, a)$ 按照策略概率 $\pi(a|s)$ 的加权平均：
   $$
   V^\pi(s) = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \Big| s_t = s \right] = \sum_{a} \pi(a|s) Q^\pi(s, a)
   $$
2. **动作价值函数（Action-Value Function）** $Q^\pi(s, a)$：表示在状态 $s$ 下执行动作 $a$，随后遵循策略 $\pi$ 所能获得的期望累积奖励。
   $$
   Q^\pi(s, a) = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \Big| s_t = s, a_t = a \right]
   $$

**优势函数（Advantage Function）** $A^\pi(s, a)$ 定义为动作价值与状态价值之差，用于衡量在状态 $s$ 下执行特定动作 $a$ 相比于"让策略自己盲选动作的平均表现"（即 $V^\pi(s)$）的"优势"程度：
$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$
如果 $A^\pi(s, a) > 0$，说明动作 $a$ 比策略 $\pi$ 随机选出来的平均动作表现更好；反之则说明动作 $a$ 不如策略的平均水平。这里并不是指"执行动作"与"不执行动作"的对比，而是"执行特定动作 $a$"与"按策略概率分布执行动作"的对比。

---

# 策略梯度定理：如何让模型"变聪明"？

**核心思考出发点**：既然我们知道了什么是"好动作"（优势函数 $A > 0$），什么是"坏动作"（优势函数 $A < 0$），那我们该如何修改模型（神经网络）的参数，让它以后多做"好动作"，少做"坏动作"呢？

这就是**策略梯度（Policy Gradient）**的意义所在。它提供了一种数学上的指导，告诉模型参数应该往哪个方向调整。

在深度强化学习中，我们通常使用神经网络来参数化策略，记为 $\pi_\theta(a|s)$，其中 $\theta$ 是网络参数。我们的目标是找到最优参数 $\theta^*$，使得目标函数 $J(\theta)$ 最大化。这里的目标函数 $J(\theta)$ 通常定义为策略 $\pi_\theta$ 下的期望累积奖励（即初始状态的价值）：
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right] = V^{\pi_\theta}(s_0)
$$

### 为什么目标函数 $J(\theta)$ 是一个"加权平均"？

很多人可能会有疑问：既然 $J(\theta) = V^{\pi_\theta}(s_0)$，而 $V(s)$ 是所有动作价值的均值，那训练的目标难道是"让所有动作的价值都变大"吗？

其实不然。这里的均值是一个**加权平均（Weighted Average）**，而"权重"正是策略选择动作的**概率** $\pi_\theta(a|s)$：
$$
J(\theta) = V^{\pi_\theta}(s_0) = \sum_{a} \pi_\theta(a|s_0) Q^{\pi_\theta}(s_0, a)
$$

**举个直观的例子：**
假设在某个状态下只有两个动作：
*   动作 A 是"吃金币"，环境给的真实价值 $Q(s, A) = 10$。
*   动作 B 是"跳悬崖"，环境给的真实价值 $Q(s, B) = -10$。

动作本身的价值是由环境规则决定的，**我们无法改变 $Q$ 值本身**。
但我们可以改变神经网络的参数 $\theta$，从而改变策略输出的**概率（权重）**：
*   **训练前（瞎猜）**：策略可能各给 50% 的概率，此时 $J = 0.5 \times 10 + 0.5 \times (-10) = 0$。
*   **训练后（变聪明）**：策略学会了 99% 选 A，1% 选 B，此时 $J = 0.99 \times 10 + 0.01 \times (-10) = 9.8$。

**结论：**
强化学习的训练目标，**不是去放大动作本身的价值，而是通过调整网络参数 $\theta$，把高价值动作的"选择概率"调大，把低价值动作的"选择概率"调小**，从而让这个加权平均值 $J(\theta)$ 达到最大。

为了使用梯度上升法更新参数：$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$，我们需要计算目标函数的梯度 $\nabla_\theta J(\theta)$。

根据**策略梯度定理**，梯度的解析表达式为：
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot Q^{\pi_\theta}(s_t, a_t) \right]
$$

**推导的核心：对数导数技巧 (Log-Derivative Trick)**
为什么梯度中会出现 $\log$？（*注：本文及后续强化学习相关文章中，$\log$ 均指代以 $e$ 为底的自然对数 $\ln$*）我们可以直接从目标函数 $J(\theta)$ 的期望定义出发进行严格推导。

根据前面的定义，目标函数是整条轨迹累积奖励的期望：
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
$$
为了书写简洁，我们记轨迹 $\tau = (s_0, a_0, s_1, a_1, \dots)$ 的总累积奖励为 $R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t$。
由于期望的本质是对所有可能轨迹的概率加权求和（积分），且参数 $\theta$ 决定了轨迹出现的概率分布 $P(\tau|\theta)$，所以我们必须将期望展开为积分形式才能对 $\theta$ 求导：

$$
\begin{aligned}
\nabla_\theta J(\theta) &= \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)] \\
&= \nabla_\theta \int P(\tau|\theta) R(\tau) d\tau \\
&= \int \nabla_\theta P(\tau|\theta) R(\tau) d\tau
\end{aligned}
$$

此时，我们引入微积分中的链式法则 $\nabla f(x) = f(x) \nabla \log f(x)$（即**对数导数技巧**），将其应用于概率 $P(\tau|\theta)$：
$$
\begin{aligned}
\nabla_\theta J(\theta) &= \int P(\tau|\theta) \nabla_\theta \log P(\tau|\theta) R(\tau) d\tau \\
&= \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P(\tau|\theta) R(\tau) \right]
\end{aligned}
$$

接下来，我们需要计算 $\nabla_\theta \log P(\tau|\theta)$。根据马尔可夫决策过程，一条轨迹发生的概率可以分解为初始状态概率、策略选择概率和环境转移概率的乘积：
$$
P(\tau|\theta) = P(s_0) \prod_{t=0}^{\infty} \pi_\theta(a_t|s_t) P(s_{t+1}|s_t, a_t)
$$
对其两边取对数（$\log$ 将乘积变为求和）：
$$
\log P(\tau|\theta) = \log P(s_0) + \sum_{t=0}^{\infty} \log \pi_\theta(a_t|s_t) + \sum_{t=0}^{\infty} \log P(s_{t+1}|s_t, a_t)
$$
对参数 $\theta$ 求梯度时，由于环境的初始状态分布 $P(s_0)$ 和状态转移概率 $P(s_{t+1}|s_t, a_t)$ 都与我们的策略网络参数 $\theta$ 无关，它们的梯度为 0，直接消失：
$$
\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

将展开后的梯度代回期望公式，我们得到：
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \left( \sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t|s_t) \right) R(\tau) \right]
$$

最后一步，结合**因果性（Causality）**：未来的动作不能影响过去的奖励。即 $t$ 时刻的动作 $a_t$ 只能影响 $t$ 时刻及之后的累积奖励。因此，我们可以将总奖励 $R(\tau)$ 替换为从 $t$ 时刻开始的累积奖励（这正是动作价值 $Q^{\pi_\theta}(s_t, a_t)$ 的无偏估计），最终得到策略梯度定理的严谨表达式：
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot Q^{\pi_\theta}(s_t, a_t) \right]
$$

通过这个技巧，我们将原本难以计算的"概率梯度的积分"，转化为了可以通过**蒙特卡洛采样（让智能体自己在环境里跑几圈收集数据）**来近似计算的"期望值"。

### 梯度的直观理解

我们可以将上式中的 $Q^{\pi_\theta}(s_t, a_t)$ 替换为优势函数 $A^{\pi_\theta}(s_t, a_t)$。
为什么可以这样替换？因为 $A(s, a) = Q(s, a) - V(s)$，相当于减去了一个只与状态 $s$ 有关的**基线（Baseline）** $V(s)$。

**物理直觉：为什么减去基线不影响期望？**
想象一下，你在某个状态 $s$ 下，对所有可能的动作都加上了一个固定的奖励 $b(s)$（比如每个动作都多给 10 分）。因为所有动作的相对好坏没有改变，策略更新的方向也不应该改变。在数学上，由于策略是一个概率分布（所有动作的概率之和永远为 1），如果你试图把所有动作的概率都往上推，这种"整体推力"的总和必然是 0。

我们可以通过下面的推导，严格证明减去任何基线 $b(s)$ 后的期望梯度为 0。这里的推导关键在于：**基线 $b(s)$ 只与状态有关，在对动作 $a$ 求和时，它相当于一个常数，可以被直接提取到求和符号外面**：
$$
\begin{aligned}
\mathbb{E}_{a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot b(s) \right] 
&= \sum_a \pi_\theta(a|s) \left( \nabla_\theta \log \pi_\theta(a|s) \cdot b(s) \right) \\
&= b(s) \sum_a \pi_\theta(a|s) \left( \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)} \right) \\
&= b(s) \sum_a \nabla_\theta \pi_\theta(a|s) \\
&= b(s) \nabla_\theta \left( \sum_a \pi_\theta(a|s) \right) \\
&= b(s) \nabla_\theta (1) \\
&= 0
\end{aligned}
$$
既然减去 $V(s)$ 后期望不变，为什么还要减呢？因为这能**显著降低采样的方差**，让训练更加稳定。

**为什么减去基线能降低方差？**
想象一下，如果环境给的所有奖励都是非常大的正数（比如所有动作的 $Q$ 值都在 1000 左右）。那么每次采样，无论动作好坏，梯度都会被乘以一个巨大的正数，导致网络参数剧烈震荡（方差极大）。
当我们减去平均值 $V(s)$（假设也是 1000）后，原本的 $Q(s, a)$ 就变成了优势 $A(s, a)$，它的值会围绕 0 波动（比如 +5 或 -3）。这样一来，好动作会得到一个温和的正向推动，坏动作会得到一个温和的负向惩罚。梯度的绝对数值大大减小，更新过程自然就变得更加平稳了。

替换后的公式如下：
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^{\pi_\theta}(s_t, a_t) \right]
$$
这个公式的物理意义非常直观：
- $\nabla_\theta \log \pi_\theta(a_t|s_t)$ 指示了增加动作 $a_t$ 概率的参数更新方向。
- $A^{\pi_\theta}(s_t, a_t)$ 作为权重。如果优势为正，说明该动作好于平均，梯度会推动网络增加该动作的概率；如果优势为负，则降低该动作的概率。

---

# 从 REINFORCE 到 Actor-Critic：算法的演进

虽然我们在理论上推导出了应该用优势函数 $A(s_t, a_t)$ 来更新策略，但在实际写代码时，我们无法直接获得真实的 $A$ 值或 $Q$ 值，只能通过**采样（Sampling）**来近似估计。根据估计方法的不同，策略梯度算法经历了以下几个阶段的演进。

为了让讲解更直观，我们先设定一个贯穿本章的具体例子，然后用它来串起三种算法。

**例子：训练 AI 面试助手**

假设你在训练一个 AI 面试助手（智能体），面试共三轮。AI 在每轮做出一个决策，并获得面试官打分。我们让 AI 完成了一局面试（为简化，令 $\gamma = 1$）：

| 轮次 | 状态 | AI 的决策 | 该轮得分（$r_t$） |
|:---:|:---:|:---:|:---:|
| 第 1 轮（自我介绍） | $s_1$ | "讲项目经历" | $r_1 = +6$ |
| 第 2 轮（技术面） | $s_2$ | "用动态规划解题" | $r_2 = +2$ |
| 第 3 轮（HR 面） | $s_3$ | "聊职业规划" | $r_3 = +4$ |

同时，假设我们有一个经验丰富的面试顾问（Critic 网络 $V_\phi$），他对各状态的预估为：
- $V_\phi(s_1) = 9$（"从第 1 轮开始，历史面试者的平均总分大约是 9 分"）
- $V_\phi(s_2) = 5$（"从第 2 轮开始，平均还能拿 5 分"）

**核心问题：第 1 轮选择"讲项目经历"这个决策到底好不好？策略参数应该怎么调？** 接下来，我们看三种算法如何回答这个问题。

## 1. 最朴素的实现：REINFORCE 算法

**先看例子：** REINFORCE 的做法最简单——等三轮面试**全部结束**，算出总分，直接用总分评价第 1 轮的决策：

$$G_1 = r_1 + r_2 + r_3 = 6 + 2 + 4 = \mathbf{12}$$

然后用 $G_1 = 12$ 乘以梯度方向 $\nabla_\theta \log \pi_\theta(a_1|s_1)$ 来更新策略参数：让"讲项目经历"的概率变大。

**问题马上就来了：** 假设 AI 再跑一局，第 1 轮仍然选"讲项目经历"（$r_1 = +6$），但第 2、3 轮运气不同（$r_2 = 0, r_3 = +1$），总分变成了 $G_1 = 7$。两次信号（$12$ vs $7$）的差异，并非第 1 轮决策好坏所致，而是后面轮次的随机性造成的。而且信号都是较大的正数，被乘入梯度后，参数每次更新的幅度都很大且波动剧烈。

**一般化的算法原理：**

上面例子中 $G_1 = r_1 + r_2 + r_3$ 是 $\gamma = 1$ 时的特例。一般地，智能体在 $t$ 时刻执行动作 $a_t$ 后，实际获得的**累积折扣回报**记为：
$$
G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}
$$

REINFORCE 算法直接用 $G_t$ 替换策略梯度公式中的 $Q(s_t, a_t)$。之所以可以这样替换，是因为 $Q$ 的定义就是 $G_t$ 的期望（$Q(s_t, a_t) = \mathbb{E}[G_t]$），所以 $G_t$ 是 $Q$ 的一个**无偏估计（Unbiased Estimate）**——虽然每局的 $G_t$ 时高时低，但打足够多局取平均，就会精确收敛到理论上的 $Q$ 值。

**参数更新公式：**
$$
\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t
$$

回到面试的例子，REINFORCE 的**两个致命缺点**就非常清楚了：
1. **方差极大**：两局面试中 $G_1$ 分别是 $12$ 和 $7$，但第 1 轮的决策完全相同。这种波动来自后续轮次的随机性，却被当作第 1 轮决策的评价信号，导致梯度估计震荡剧烈。
2. **回合更新（Episodic）**：必须等三轮面试全部结束，才能算出 $G_1$ 并更新参数。如果面试有 100 轮呢？效率极低。

## 2. 引入基线 (REINFORCE with Baseline)

**先看例子：** 为了更准确地评价第 1 轮的决策，我们请出那位面试顾问（Critic）。他说："历史上从第 1 轮开始，面试者的平均总分是 $V_\phi(s_1) = 9$ 分。"

有了这个参考值，我们就可以算出"这局面试比平均水平好了多少"：

$$\text{更新信号} = G_1 - V_\phi(s_1) = 12 - 9 = \mathbf{+3}$$

$+3$ 的含义很清晰：**这局比平均高了 3 分，"讲项目经历"是好于平均的选择。**

如果另一局总分是 $7$ 分，信号就是 $7 - 9 = -2$（不如平均，应降低该决策的概率）。对比两种算法的信号：

| | 第 1 局 | 第 2 局 | 信号波动范围 |
|:---:|:---:|:---:|:---:|
| **REINFORCE** | $12$ | $7$ | $[7, 12]$（大正数） |
| **+ Baseline** | $+3$ | $-2$ | $[-2, +3]$（0 附近） |

减去基线后，信号的幅度从 $[7, 12]$ 缩小到了 $[-2, +3]$，梯度被乘以的数值大大减小，参数更新的震荡随之降低。而且正负号直接告诉你决策是好是坏，方向更清晰。

**一般化的算法原理：**

我们在 $G_t$ 的基础上减去一个状态价值的估计值 $V_\phi(s_t)$（用另一个神经网络拟合，参数为 $\phi$），用 $(G_t - V_\phi(s_t))$ 来近似优势函数 $A(s_t, a_t)$。

**参数更新公式：**
$$
\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - V_\phi(s_t))
$$

需要注意一个微妙之处：虽然信号的**绝对幅度**降低了，但 $G_t$ 本身的**随机波动**（它仍然包含 $r_{t+1}, r_{t+2}, \dots$ 等后续时步的随机性）并没有改变。基线解决的是"信号偏离零点太远"的问题，而非"信号内部的随机性太大"的问题。

回到面试的例子，这个方法**依然没有解决"回合更新"的问题**——计算 $G_1 = r_1 + r_2 + r_3$ 仍然需要等三轮面试全部结束。有没有办法**不等面试结束，走一步就能评价**呢？

## 3. Actor-Critic 架构 (时序差分与单步更新)

**先看例子：** Actor-Critic 的核心想法是——评价第 1 轮的决策，何必等到面试结束？第 1 轮结束后，我们已经知道两件事：这轮拿了 $r_1 = 6$ 分，且进入了状态 $s_2$。面试顾问（Critic）告诉我们 $V_\phi(s_2) = 5$，即"从 $s_2$ 开始平均还能拿 5 分"。

于是我们可以**立刻**估算：走完第 1 步后的"实际表现" $\approx r_1 + V_\phi(s_2) = 6 + 5 = 11$。而 Critic 之前对 $s_1$ 的预期是 $V_\phi(s_1) = 9$。两者之差：

$$\delta_1 = r_1 + \gamma V_\phi(s_2) - V_\phi(s_1) = 6 + 5 - 9 = \mathbf{+2}$$

$\delta_1 = +2$ 的含义是：实际表现比预期好了 2 分，"讲项目经历"是个不错的决策。关键是——**第 1 轮一结束就算出来了，不用等后面两轮！**

**一般化的算法原理：**

上面的 $\delta_1$ 就是著名的**时序差分误差（TD Error）**。其核心思想叫**自举（Bootstrapping）**：不再等待真实的完整回报 $G_t$，而是用 Critic 对下一状态的估值 $V_\phi(s_{t+1})$ 来"代替"未来的实际奖励：

$$Q(s_t, a_t) \approx r_t + \gamma V_\phi(s_{t+1})$$

将其代入优势函数 $A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$，就得到了 TD 误差：
$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

$\delta_t$ 近似了优势函数 $A(s_t, a_t)$，而且**只需要走一步（获得 $r_t$ 和 $s_{t+1}$）就能计算**。

在 Actor-Critic 架构中，我们同时训练两个神经网络：

1. **Actor（演员/策略网络）** $\pi_\theta(a|s)$：负责选择动作。用 Critic 算出的 $\delta_t$ 来更新策略：
   $$
   \theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta_t
   $$
2. **Critic（评委/价值网络）** $V_\phi(s)$：负责估计状态价值。目标是让自己的预测越来越准，即最小化 TD 误差的平方：
   $$
   \phi \leftarrow \phi - \beta \nabla_\phi \left( \frac{1}{2} \delta_t^2 \right)
   $$

**用面试的例子理解 Critic 的更新**：我们算出了 $\delta_1 = +2$，说明 Critic 之前对 $s_1$ 的预估 $V_\phi(s_1) = 9$ 偏低了——实际的单步证据显示"得了 6 分，且到达了一个价值 5 分的状态"，合计预期 $11$ 分，而 Critic 只预测了 $9$ 分。

Critic 的目标是**减小这个预测误差**，即最小化 $\frac{1}{2}\delta_t^2$。让我们展开梯度下降的计算，看看它是否真的会让 $V_\phi(s_1)$ 往上调：

$$\nabla_\phi \frac{1}{2}\delta_t^2 = \delta_t \cdot \nabla_\phi \delta_t$$

由于 $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$，其中"目标值" $r_t + \gamma V_\phi(s_{t+1})$ 在实际训练中被视为常数（停止梯度），所以：

$$\nabla_\phi \delta_t = -\nabla_\phi V_\phi(s_t)$$

代入更新公式（注意梯度下降的减号和链式法则的减号**相消**了）：

$$\phi \leftarrow \phi - \beta \cdot \delta_t \cdot (-\nabla_\phi V_\phi(s_t)) = \phi + \beta \delta_t \nabla_\phi V_\phi(s_t)$$

当 $\delta_t = +2 > 0$ 时，最终效果是**加上** $\beta \cdot 2 \cdot \nabla_\phi V_\phi(s_t)$，推动 $V_\phi(s_1)$ 增大——正好从 $9$ 往真实值 $11$ 靠拢。反之，如果 $\delta_t < 0$（预估偏高），则会推动 $V_\phi$ 减小。这就是标准梯度下降自动做到"偏低就调高、偏高就调低"的数学原因。

随着训练迭代，Critic 的预测会越来越准，$\delta_t$ 会越来越小，Actor 收到的信号也就越来越精确。

回到面试的例子，Actor-Critic 相比前两种方法的两个关键优势就很清楚了：
1. **单步更新**：第 1 轮一结束就能计算 $\delta_1$，无需等后续轮次。
2. **方差更低**：$\delta_1$ 仅依赖**单步**的随机性（$r_1$ 和 $s_2$），与第 2、3 轮的随机结果完全解耦。相比 $G_1$ 累积了三步的随机性，波动自然更小。

代价是引入了**偏差（Bias）**：如果 Critic 对 $V_\phi(s_2)$ 的估计不准（比如真实值是 $6$ 而非 $5$），$\delta_1$ 也会偏离真实优势值。这就是经典的**偏差-方差权衡（Bias-Variance Trade-off）**：Actor-Critic 用一定的偏差换来了更低的方差和更高的学习效率。随着训练推进，Critic 的估值会越来越准，偏差也会逐渐减小。

Actor-Critic 完美结合了策略梯度和价值函数的优势，极大地降低了方差，并实现了高效的在线单步更新。然而，它仍然面临着策略更新步长难以控制的问题，过大的更新可能导致策略崩溃。这就引出了下一篇要讲的 TRPO 和 PPO 算法。

## 对比总结

回到面试的例子，三种算法对第 1 轮"讲项目经历"的评价一目了然：

| | **REINFORCE** | **+ Baseline** | **Actor-Critic** |
|:---|:---:|:---:|:---:|
| **更新信号** | $G_t$ | $G_t - V_\phi(s_t)$ | $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ |
| **本例数值** | $12$ | $+3$ | $+2$ |
| **信号含义** | 绝对总回报 | 相对于平均的优势 | 单步 TD 误差 |
| **何时可计算** | 回合结束后 | 回合结束后 | 每走一步即可 |
| **梯度方差** | 高（信号幅度大 + 累积多步随机性） | 较低（信号幅度小，但仍累积多步随机性） | 低（信号幅度小 + 仅含单步随机性） |
| **偏差** | 无偏（$G_t$ 是 $Q$ 的无偏估计） | 无偏（减去常数不影响期望） | 有偏（取决于 $V_\phi$ 的精度） |

> 从 REINFORCE → + Baseline → Actor-Critic，算法的演进路线非常清晰：**先解决信号幅度过大的问题**（引入基线，将信号从绝对值拉到 0 附近），**再解决学习效率低的问题**（用 Critic 自举替代完整回报，将多步随机性压缩为单步，同时实现即时更新）。

**开源代码参考与算法对比：**
在实际应用中，REINFORCE 由于方差过大，几乎不再被单独用于复杂任务；Actor-Critic（特别是其变体 A2C/A3C）则被广泛使用。
目前最主流的强化学习开源库如 **Stable-Baselines3** 和 Hugging Face 的 **TRL (Transformer Reinforcement Learning)** 都提供了这些基础算法的实现。

以一个极简的 REINFORCE 策略更新代码为例（PyTorch 实现）：
```python
# log_probs: 策略网络输出的对数概率 log \pi(a|s)
# returns: 计算出的累积奖励 G_t
policy_loss = []
for log_prob, G in zip(log_probs, returns):
    policy_loss.append(-log_prob * G)

optimizer.zero_grad()
loss = torch.stack(policy_loss).sum()
loss.backward()
optimizer.step()
```

> 下一篇：[笔记｜生成模型（十七）：信任区域与近端策略优化 (从 TRPO 到 PPO)](/chengYi-xun/2026/04/03/18-trpo-ppo/)
