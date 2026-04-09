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

强化学习问题通常被建模为**马尔可夫决策过程（Markov Decision Process, MDP）**（参考 [CMU 15-281 Lecture Notes: MDPs](https://www.cs.cmu.edu/~15281-s25/coursenotes/mdps/index.html)）。一个 MDP 可以用一个元组 $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$ 来表示：

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

在深度强化学习中，我们通常使用神经网络来参数化策略，记为 $\pi_\theta(a|s)$，其中 $\theta$ 是网络参数。我们的目标是找到最优参数 $\theta^*$，使得目标函数 $J(\theta)$ 最大化。这里的目标函数 $J(\theta)$ 通常定义为策略 $\pi_\theta$ 下的**期望累积奖励（即初始状态的价值）**：
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right] = V^{\pi_\theta}(s_0)
$$

### 深入理解目标函数 $J(\theta)$

很多人可能会有疑问：既然 $J(\theta) = V^{\pi_\theta}(s_0)$，那训练的目标难道是"让所有动作的价值都变大"吗？为了理清这个问题，我们需要深入理解以下三点：

1. **"初始状态的价值" $V^{\pi_\theta}(s_0)$ 的真正含义**：
   在强化学习中，一个状态的"价值"就是指**从这个状态出发，按照策略一直执行下去，所能获得的"期望累积奖励"**。所以，目标函数 $J(\theta)$ 正是从初始状态 $s_0$ 开始，遵循策略 $\pi_\theta$ 所能获得的期望总奖励。

2. **做一系列动作的价值之和的期望**：
   当你从 $s_0$ 出发，策略 $\pi_\theta$ 会决定你采取哪些动作，每个动作会带来一个即时奖励并导致状态转移。随着你不断执行动作、获得奖励、转移状态，这些奖励累积起来就形成了"累积奖励"或"回报"（Return）。因为策略和环境具有随机性，每一次从 $s_0$ 出发可能都会得到不同的累积奖励。所以，$V^{\pi_\theta}(s_0)$ 取的是所有这些可能的回报的**期望值**。

3. **策略概率作为"加权平均"的权重**：
   在计算这个期望值时，这里的均值是一个**加权平均（Weighted Average）**，而"权重"正是策略选择动作的**概率** $\pi_\theta(a|s)$。对于初始状态 $s_0$，其价值可以展开为：
   $$
   J(\theta) = V^{\pi_\theta}(s_0) = \sum_{a} \pi_\theta(a|s_0) Q^{\pi_\theta}(s_0, a)
   $$
   通过这种加权平均，我们将所有可能动作及其后续发展都考虑进去，最终得到该状态的期望累积奖励。

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

> **延伸阅读：** 关于策略梯度定理的更详细推导和直觉理解，推荐以下两篇经典文章：
> - [Part 3: Intro to Policy Optimization — OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)：OpenAI 官方教程，从最简形式出发逐步引入 baseline 和优势函数。
> - [Policy Gradient Algorithms — Lil'Log](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)：Lilian Weng 的经典博客，覆盖从 REINFORCE 到 PPO 的完整策略梯度算法族。

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

关于 REINFORCE 算法的深入探讨可参考 [REINFORCE: Monte Carlo Policy Gradient Methods](https://rl-book.com/learn/policy_gradients/reinforce/)（含方差分析和代码示例）。回到面试的例子，REINFORCE 的**两个致命缺点**就非常清楚了：

1. **方差极大**：两局面试中 $G_1$ 分别是 $12$ 和 $7$，但第 1 轮的决策完全相同。这种波动来自后续轮次的随机性，却被当作第 1 轮决策的评价信号，导致梯度估计震荡剧烈。
2. **回合更新（Episodic）**：必须等三轮面试全部结束，才能算出 $G_1$ 并更新参数。如果面试有 100 轮呢？效率极低。

**REINFORCE 的完整实现：**

```python
import torch  # 张量与自动求导，用于实现策略网络与梯度更新

# ============================================================
# 模型定义
# actor: 策略网络 π_θ(a|s), 输入状态, 输出动作概率分布
# REINFORCE 只需要 Actor, 不需要 Critic
# ============================================================
actor = ActorModel(state_dim, action_dim)  # 策略网络：把状态映射为动作分布（如各类别的 logits → softmax）
optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)  # 只优化 Actor 参数 θ，无 Critic

# ============================================================
# Step 1: 数据采集 — 必须跑完一整局 (回合制)
# 数据格式: 一条完整轨迹 [(s₀,a₀,r₀), (s₁,a₁,r₁), ...]
# ============================================================
trajectory = []  # 存储每步的 log π 与即时奖励，供回合末算 G_t 与策略梯度
state = env.reset()  # 初始状态 s₀
done = False  # 是否已到达终止状态
while not done:
    dist = actor(state)  # 当前 s_t 下动作的概率分布 π_θ(·|s_t)
    action = dist.sample()  # 从策略分布采样 a_t，与环境交互并探索
    log_prob = dist.log_prob(action)  # log π_θ(a_t|s_t)，策略梯度里要对 θ 求导的项
    next_state, reward, done = env.step(action)  # 执行动作得 r_t、s_{t+1}、是否结束
    trajectory.append((log_prob, reward))  # 只记策略相关量与奖励，状态可从环境重放（此处未存）
    state = next_state  # 进入下一时刻状态

# ============================================================
# Step 2: 计算累积回报 G_t (回合结束后才能算!)
# G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
# ============================================================
returns = []  # 与 trajectory 时间顺序一致的各步回报 G_t
G = 0  # 从轨迹末尾反向累加的折扣回报
for _, reward in reversed(trajectory):
    G = reward + gamma * G  # 自后向前：G_t = r_t + γ·G_{t+1}，得到从 t 起的累积折扣回报
    returns.insert(0, G)  # 插在列表头部，把逆序递推结果还原成 t=0,1,... 的正序
returns = torch.tensor(returns)  # 转为张量，便于与 log_prob 逐元素相乘

# ============================================================
# Step 3: 策略梯度更新 (用 G_t 作为 Q 的无偏估计)
# loss = -Σ log π(a|s) · G_t
# ============================================================
policy_loss = 0  # 标量损失；最小化 −Σ log π·G_t 等价于沿 ∇log π·G_t 方向改进策略
for (log_prob, _), G_t in zip(trajectory, returns):
    policy_loss += -log_prob * G_t  # 累加策略梯度项：G_t 充当 Q 的蒙特卡洛无偏估计

optimizer.zero_grad()  # 清空旧梯度，避免累加
policy_loss.backward()  # 反传：G_t 视为常数，梯度来自各步 log π 对 θ 的导数
optimizer.step()  # Adam 更新 θ，实现 θ ← θ + α·∇_θ log π·G_t 这一类随机梯度上升
```

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

**REINFORCE with Baseline 的完整实现：**

```python
import torch  # 张量计算与自动求导
import torch.nn.functional as F  # 损失函数等（此处用 MSE 训练 Critic）

# ============================================================
# 模型定义 (新增 Critic 网络作为基线!)
# actor:  策略网络 π_θ(a|s)
# critic: 价值网络 V_φ(s), 用于估计状态价值作为基线
# ============================================================
actor = ActorModel(state_dim, action_dim)  # Actor：选动作；梯度里乘的是 (G−V) 而非裸 G，减小方差
critic = CriticModel(state_dim)  # Critic：估计 V_φ(s) 作基线；减 V 不改变策略梯度期望，只降方差
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)  # 更新策略参数 θ
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)  # 更新价值参数 φ

# ============================================================
# Step 1: 数据采集 — 仍需跑完一整局 (与 REINFORCE 相同)
# ============================================================
trajectory = []  # 每步保存 log π、r、以及当步采样的 V(s)，供优势与 Critic 监督
state = env.reset()
done = False
while not done:
    dist = actor(state)  # 当前状态下的动作分布 π_θ(·|s)
    action = dist.sample()  # 采样动作与环境交互
    log_prob = dist.log_prob(action)  # 策略项，供 Actor 反传
    value = critic(state).squeeze()  # V_φ(s)：基线估计；squeeze 去掉多余维度便于标量运算
    next_state, reward, done = env.step(action)
    trajectory.append((log_prob, reward, value))  # 保留 value，与后面 G_t 对齐到同一时刻 t
    state = next_state

# ============================================================
# Step 2: 计算累积回报 G_t (仍需等回合结束!)
# ============================================================
returns = []  # 各步蒙特卡洛回报，仍须整局结束后才能算全
G = 0
for _, reward, _ in reversed(trajectory):
    G = reward + gamma * G  # 反向递推 G_t = r_t + γ·G_{t+1}
    returns.insert(0, G)  # 插入表头，恢复时间正序与 trajectory 一一对应
returns = torch.tensor(returns)  # 作为 Critic 的回归目标（真值标签为 G_t）

# ============================================================
# Step 3: 用 (G_t - V(s)) 替代 G_t, 降低方差
# ============================================================
policy_loss = 0  # Actor 损失：−log π·(G_t − V)
value_loss = 0  # Critic 损失：让 V_φ(s) 拟合 G_t
for (log_prob, _, value), G_t in zip(trajectory, returns):
    advantage = G_t - value.detach()  # 优势近似 A≈G_t−V；detach 截断到 V 的梯度，避免 policy_loss 反传到 φ
    policy_loss += -log_prob * advantage  # 相对基线偏高则增大该动作概率，偏低则抑制
    value_loss += F.mse_loss(value, torch.tensor(G_t))  # Critic 用 MSE 把 V_φ(s_t) 拉向蒙特卡洛回报 G_t

actor_optimizer.zero_grad()  # 清 Actor 梯度
policy_loss.backward()  # 仅 θ 从 policy_loss 获得梯度
actor_optimizer.step()  # 更新 Actor

critic_optimizer.zero_grad()  # 清 Critic 梯度
value_loss.backward()  # φ 从 value_loss 获得梯度（advantage 里 value 已 detach，互不串扰）
critic_optimizer.step()  # 更新 Critic
```

## 3. Actor-Critic 架构 (时序差分与单步更新)

> 关于 Actor-Critic 方法及其变体（A2C、GAE）的系统介绍，推荐 [Actor-Critic Methods, A2C and GAE](https://avandekleut.github.io/a2c/)。

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

**Actor-Critic 的完整实现：**

```python
import torch  # 张量与自动求导
import torch.nn.functional as F  # 与前后代码块风格一致；需要时可接 F.smooth_l1_loss 等

# ============================================================
# 模型定义 (与 Baseline 相同: Actor + Critic)
# ============================================================
actor = ActorModel(state_dim, action_dim)  # 策略 π_θ(a|s)，用 TD 误差 δ 作优势信号更新
critic = CriticModel(state_dim)  # 价值 V_φ(s)，用自举目标训练，供一步 TD 目标
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

# ============================================================
# 训练循环: 每走一步就更新! (不需要等回合结束)
# ============================================================
state = env.reset()  # 当前状态 s_t
for step in range(max_steps):  # 固定步数或可与 done 组合使用
    # --- 前向传播 ---
    dist = actor(state)                             # π_θ(·|s)
    action = dist.sample()  # a_t ~ π_θ(·|s_t)
    log_prob = dist.log_prob(action)  # log π，供策略梯度
    value = critic(state).squeeze()                 # V_φ(s)：当前状态价值，TD 残差的一端

    next_state, reward, done = env.step(action)  # 单步转移：得 r_t、s_{t+1}、是否终止
    next_value = critic(next_state).squeeze().detach()  # V_φ(s')：自举用下一状态价值；detach 使 Actor 不把梯度灌进「目标」侧

    # ============================================================
    # 计算 TD 误差 (走一步就能算! 不用等回合结束)
    # δ = r + γ·V(s') - V(s) ≈ A(s,a)
    # ============================================================
    td_target = reward + gamma * next_value * (1 - done)  # 一步 TD 目标：终止则不再加未来项，(1−done) 截断 bootstrap
    delta = td_target - value  # TD 误差 δ ≈ 优势：实际单步回报+bootstrap 与当前 V(s) 的差

    # ============================================================
    # 同时更新 Actor 和 Critic (每一步都更新!)
    # ============================================================
    actor_loss = -log_prob * delta.detach()    # Actor：δ 当系数；detach 使 Critic 误差不通过 δ 回传到 θ（标准 Actor-Critic 分工）
    critic_loss = delta.pow(2)                 # Critic：最小化 δ²，等价于让 V(s) 去拟合 r+γV(s')（目标端已 detach）

    actor_optimizer.zero_grad()  # 清 Actor 梯度缓存
    actor_loss.backward()  # 反传更新 θ
    actor_optimizer.step()

    critic_optimizer.zero_grad()  # 清 Critic 梯度缓存
    critic_loss.backward()  # 梯度经 δ 回传到 V_φ(s)；next_value 已 detach，不把 Actor 侧目标再反传到 φ
    critic_optimizer.step()

    state = next_state if not done else env.reset()  # 未结束则继续 s_{t+1}；结束则新开一局
```

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

---

# 广义优势估计（GAE）：蒙特卡洛与 TD 的统一框架

上面三种算法分别用了蒙特卡洛（$G_t$）、蒙特卡洛减基线（$G_t - V$）、单步 TD（$\delta_t$）来估计优势。但蒙特卡洛无偏却方差大，TD 方差小却有偏——有没有一种方法能**在这两个极端之间自由调节**？这就是 **GAE（Generalized Advantage Estimation）**（[原始论文](https://arxiv.org/abs/1506.02438)）要解决的问题，它是后续 TRPO 和 PPO 算法的基础组件。

## 三种优势估计方法的关系

不同的估计方法，本质上是在回答同一个问题——**"估计 $Q(s_t, a_t)$ 时，我们应该多大程度地依赖真实观测（实际奖励），多大程度地依赖 Critic 的预测（$V_\phi$）？"**

**方法 1：蒙特卡洛估计（完全不信 Critic）。** 从 $t$ 时刻开始，一直走到回合结束，用实际获得的累积奖励替代 $Q$：

$$\hat{A}_t^{MC} = G_t - V(s_t) = \left(\sum_{l=0}^{T-t-1} \gamma^l r_{t+l}\right) - V(s_t)$$

因为 $G_t$ 是 $Q(s_t, a_t)$ 的无偏估计（$\mathbb{E}[G_t | s_t, a_t] = Q(s_t, a_t)$），所以 $\hat{A}_t^{MC}$ 是真实优势的**无偏估计**。但它累积了从 $t$ 到 $T$ 每一步的随机性，方差极大。用面试的例子说：$G_1 = r_1 + r_2 + r_3$，后两轮的随机性全部混进了对第 1 轮决策的评价。

**方法 2：单步 TD 误差（完全信任 Critic）。** 只看一步实际奖励，未来的部分全部交给 Critic 预测：

$$\hat{A}_t^{TD} = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$\delta_t$ 仅含一步随机性（$r_t$ 和 $s_{t+1}$），方差很低。但如果 $V(s_{t+1})$ 不准确，$\delta_t$ 就会有偏差。用面试的例子说：$\delta_1 = r_1 + V(s_2) - V(s_1) = 6 + 5 - 9 = +2$，这个 $+2$ 完全依赖 Critic 对 $V(s_2) = 5$ 的预估准不准。如果实际后续得分是 $6$ 而非 $5$，$\delta_1$ 就低估了真实优势。

**方法 3：n 步估计（折中方案）。** 自然的想法是：多看几步实际奖励，减少对 Critic 的依赖。n 步优势估计为：

$$\hat{A}_t^{(n)} = \left(\sum_{l=0}^{n-1} \gamma^l r_{t+l}\right) + \gamma^n V(s_{t+n}) - V(s_t)$$

看 $n$ 步真实奖励，然后用 Critic 估计剩余部分。$n$ 越大，偏差越小（用了更多真实数据），但方差越大（累积了更多随机性）。容易验证：$n = 1$ 时退化为 TD 误差 $\delta_t$，$n = T - t$ 时退化为蒙特卡洛估计 $G_t - V(s_t)$。

## n 步优势估计 = TD 误差的折扣累加

一个关键的数学等式是：**n 步优势估计恰好等于前 n 个 TD 误差的折扣累加**：

$$\hat{A}_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}$$

我们用面试例子（$n = 3$，三轮面试，令 $\gamma = 1$ 简化）来**逐项展开**验证。先把 3 个 TD 误差写出来：

$$\delta_1 = r_1 + V(s_2) - V(s_1), \quad \delta_2 = r_2 + V(s_3) - V(s_2), \quad \delta_3 = r_3 + V(s_4) - V(s_3)$$

把它们加起来，按类型分组（$\color{blue}{\text{蓝色 = 真实奖励}}$、$\color{green}{\text{绿色 = 正 V 项}}$、$\color{red}{\text{红色 = 负 V 项}}$）：

$$\delta_1 + \delta_2 + \delta_3 = \underbrace{\color{blue}{r_1 + r_2 + r_3}}_{\text{真实奖励}} + \underbrace{\color{green}{V(s_2) + V(s_3) + V(s_4)}}_{\text{来自 }+\gamma V(s_{t+l+1})} - \underbrace{\color{red}{V(s_1) + V(s_2) + V(s_3)}}_{\text{来自 }-V(s_{t+l})}$$

现在逐项比较$\color{green}{\text{绿色}}$和$\color{red}{\text{红色}}$两组（$\color{gray}{\text{灰色}}$表示已抵消的项）：

$$= \color{blue}{r_1 + r_2 + r_3} + \color{gray}{V(s_2)} + \color{gray}{V(s_3)} + \color{green}{V(s_4)} - \color{red}{V(s_1)} - \color{gray}{V(s_2)} - \color{gray}{V(s_3)}$$

$\color{gray}{V(s_2)}$ 在绿色组出现一次（$+$），在红色组也出现一次（$-$），抵消变灰；$\color{gray}{V(s_3)}$ 同理抵消。**中间所有 $V$ 项两两对消，只剩下首尾**：

$$= \color{blue}{(r_1 + r_2 + r_3)} + \color{green}{V(s_4)} - \color{red}{V(s_1)} = \hat{A}_1^{(3)} \quad \checkmark$$

这正是 3 步优势估计的定义！如果回合在第 3 步结束（$V(s_4) = 0$），则 $= G_1 - V(s_1)$，退化为蒙特卡洛。

**一般情况**（$\gamma \neq 1$）同理：每个正 $V$ 项 $\gamma^{l+1} V(s_{t+l+1})$ 与下一个负 $V$ 项 $\gamma^{l+1} V(s_{t+l+1})$ 配对抵消，最终只剩首尾：

$$\sum_{l=0}^{n-1} \gamma^l \delta_{t+l} = \sum_{l=0}^{n-1} \gamma^l r_{t+l} + \gamma^n V(s_{t+n}) - V(s_t) = \hat{A}_t^{(n)} \quad \square$$

**这个等式解释了一个常见的困惑：为什么"不属于第 $t$ 步"的后续 TD 误差 $\delta_{t+1}, \delta_{t+2}, \ldots$ 也被加进了对动作 $a_t$ 的优势评价？** 答案是：累加后续 TD 误差并不是在"重复衡量动作 $a_t$ 好不好"，而是在**用实际观测逐步替换 Critic 的预测**——每多加一个 $\delta$，就多看了一步真实奖励，少依赖一步 Critic 的猜测。

用面试例子来说：$\delta_1 = +2$ 完全信任 Critic 对 $V(s_2) = 5$ 的预估。如果面试继续走下去，发现 $\delta_2 > 0$（第 2 步实际表现比 Critic 预估好），这说明 $V(s_2)$ 被低估了——而这正是 $\delta_1$ 被低估的同一个原因。把 $\delta_2$ 加进来（2 步估计 $\hat{A}_1^{(2)} = \delta_1 + \gamma\delta_2$），就是用第 2 步的实际奖励替换了 Critic 对 $V(s_2)$ 的不完美预测，修正了偏差。

## 从 n 步估计到 GAE：指数加权平均

现在问题变成了：**n 取多少合适？** $n=1$ 偏差大方差小，$n=T-t$ 偏差小方差大，中间的 $n$ 是某种折中。但每个 $n$ 都给出了真实优势的一个合理估计，选哪个都有道理。

GAE 的巧妙之处在于：**不选任何单一的 $n$，而是把所有 n 步估计做指数加权平均**：

$$\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \hat{A}_t^{(n)}$$

展开来看，各个 n 步估计分到的权重为：

| n 步估计 | 含义 | 权重 | $\lambda = 0.95$ 时 |
|:---:|:---|:---:|:---:|
| $\hat{A}_t^{(1)} = \delta_t$ | 只看 1 步，完全信任 Critic | $(1-\lambda) = 0.05$ | 5% |
| $\hat{A}_t^{(2)} = \delta_t + \gamma\delta_{t+1}$ | 看 2 步 | $(1-\lambda)\lambda = 0.0475$ | 4.75% |
| $\hat{A}_t^{(3)} = \delta_t + \gamma\delta_{t+1} + \gamma^2\delta_{t+2}$ | 看 3 步 | $(1-\lambda)\lambda^2 = 0.0451$ | 4.51% |
| $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ |

权重总和为 $(1-\lambda)(1 + \lambda + \lambda^2 + \cdots) = (1-\lambda) \cdot \frac{1}{1-\lambda} = 1$。由于 $\lambda^n$ 随 $n$ 指数衰减，短步估计的权重总是更大。$\lambda$ 越接近 $0$，1 步估计的权重 $(1-\lambda)$ 越接近 $1$，几乎只用 $\delta_t$（低方差、高偏差）；$\lambda$ 越接近 $1$，各步估计的权重越均匀，等效于看更多步的实际奖励（低偏差、高方差）。

将 $\hat{A}_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}$ 代入，对于固定的 $l$，$\delta_{t+l}$ 出现在所有 $n \geq l+1$ 的估计中。交换求和顺序：

$$\hat{A}_t^{\text{GAE}} = (1-\lambda) \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} \sum_{n=l+1}^{\infty} \lambda^{n-1} = (1-\lambda) \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} \cdot \frac{\lambda^l}{1-\lambda} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

得到 GAE 的标准形式：

$$\boxed{\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \, \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)}$$

## 严格推导：$\lambda = 1$ 时 GAE 退化为蒙特卡洛（无偏）

当 $\lambda = 1$ 时，GAE 公式变为 $\hat{A}_t = \sum_{l=0}^{T-t-1} \gamma^l \delta_{t+l}$。这和上面 n 步估计的 telescoping **完全相同**——只是 $n$ 取到了最大值 $T - t$（看完全程）。

仍然用面试例子（$\gamma = 1$，3 轮面试，$T - t = 3$）具体展开：

$$\hat{A}_1^{\text{GAE}(1,1)} = \delta_1 + \delta_2 + \delta_3$$

$$= \underbrace{\color{blue}{r_1 + r_2 + r_3}}_{\text{真实奖励}} + \underbrace{\color{gray}{V(s_2)} + \color{gray}{V(s_3)} + \color{green}{V(s_4)}}_{\text{正 V 项}} - \underbrace{\color{red}{V(s_1)} + \color{gray}{V(s_2)} + \color{gray}{V(s_3)}}_{\text{负 V 项}}$$

$$= \color{blue}{(r_1 + r_2 + r_3)} + \color{green}{V(s_4)} - \color{red}{V(s_1)}$$

若 $s_4$ 为终止状态（$V(s_4) = 0$）：$= (r_1 + r_2 + r_3) - V(s_1) = G_1 - V(s_1)$，即蒙特卡洛估计。

**一般情况**（$\gamma \neq 1$）同理，抵消规律与上面完全一致（只是每项多了 $\gamma^l$ 系数，不影响配对抵消），最终只剩首尾：

$$\hat{A}_t^{\text{GAE}(\gamma,1)} = \sum_{l=0}^{T-t-1} \gamma^l r_{t+l} + \gamma^{T-t} V(s_T) - V(s_t)$$

若回合在 $T$ 时刻终止（终止状态价值为零，$V(s_T) = 0$）：

$$\hat{A}_t^{\text{GAE}(\gamma,1)} = \underbrace{\sum_{l=0}^{T-t-1} \gamma^l r_{t+l}}_{= \; G_t} - V(s_t) = G_t - V(s_t) = \hat{A}_t^{MC} \quad \square$$

所有 $V$ 项被望远镜抵消消灭，估计中不再依赖 Critic 的任何预测——这就是 $\lambda = 1$ 无偏的根本原因。类似地，$\lambda = 0$ 时 $\hat{A}_t^{\text{GAE}(\gamma,0)} = \delta_t$，退化为单步 TD。

## 蒙特卡洛、TD 误差与 GAE 的对比总结

| | **蒙特卡洛** $G_t - V(s_t)$ | **TD 误差** $\delta_t$ | **GAE** $\sum (\gamma\lambda)^l \delta_{t+l}$ |
|:---|:---:|:---:|:---:|
| **等价于** | $\lambda = 1$ 的 GAE | $\lambda = 0$ 的 GAE | $\lambda \in (0,1)$ 的加权混合 |
| **公式展开** | $\sum_{l} \gamma^l r_{t+l} - V(s_t)$ | $r_t + \gamma V(s_{t+1}) - V(s_t)$ | 所有 n 步估计的指数加权平均 |
| **信任 Critic 程度** | 完全不信（只用 $V(s_t)$ 作基线） | 完全信任（用 $V(s_{t+1})$ 替代全部未来） | 部分信任（$\lambda$ 调节） |
| **偏差** | 无（$G_t$ 是 $Q$ 的无偏估计） | 有（$V(s_{t+1})$ 可能不准） | 介于两者之间 |
| **方差** | 高（累积 $T-t$ 步随机性） | 低（仅 1 步随机性） | 介于两者之间 |
| **何时可计算** | 回合结束后 | 每走一步即可 | 回合结束后（需后续 $\delta$） |
| **物理意义** | 动作 $a_t$ 走完全程后**实际**比平均好多少 | 动作 $a_t$ 的**即时**表现比 Critic 预期好多少 | 综合即时和多步信息的**平滑**估计 |

实践中 $\lambda = 0.95$ 是一个好的折中——保留了 95% 的长期信息，同时通过 $0.95^l$ 的指数衰减使远处（方差大的）TD 误差权重递减，有效抑制方差。

## GAE 的递推实现

GAE 公式可以从后向前**递推**计算，避免显式求和：

$$
\hat{A}_t = \delta_t + \gamma \lambda \, \hat{A}_{t+1}
$$

下面是具体实现：

```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95, dones=None):
    """
    广义优势估计 (Generalized Advantage Estimation)

    rewards: Tensor [T]                  每步即时奖励
    values:  Tensor [T] 或 [T+1]         Critic 对各状态的价值估计
                                         推荐 T+1，最后一个元素是 bootstrap 值 V(s_T)
    gamma:   折扣因子，控制对远期奖励的衰减（通常 0.99）
    lam:     GAE 平滑参数 λ，越大方差越高但偏差越小（通常 0.95）
    dones:   Tensor [T] (0/1)            1 表示回合在该步结束
                                         用于在 episode 边界处截断 GAE 递推链
    """
    T = len(rewards)
    # zeros_like 自动继承 device/dtype，多 GPU 训练时不会出错
    advantages = torch.zeros_like(rewards)

    # 没有提供 dones 时，默认全部为 0（假设一个连续回合，不推荐）
    if dones is None:
        dones = torch.zeros_like(rewards)

    # 如果 values 长度为 T（没有 bootstrap），补一个 0
    # 什么是 bootstrap？当回合被截断（达到最大步数但游戏未结束）时，
    # 后面还有未知的奖励。我们用 Critic 的估计 V(s_T) 来"替代"看不到的未来回报，
    # 这个"用估计代替真实值"的操作就叫 bootstrap。
    # 截断场景下建议传 T+1 维度的 values，使末尾能正确 bootstrap
    if len(values) == T:
        values = torch.cat([values, torch.zeros(1, device=values.device)])

    gae = 0.0  # 递推中的累积量，从末尾往前滚动

    for t in reversed(range(T)):  # 从 t=T-1 向前递推到 t=0
        # TD 残差 δ_t = r_t + γ·V(s_{t+1})·(1-done_t) − V(s_t)
        # 当 done_t=1 时，回合已结束，V(s_{t+1}) 属于新回合，用 (1-done) 归零
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        # GAE 递推：Â_t = δ_t + γλ·(1-done_t)·Â_{t+1}
        # (1-done_t) 同时截断递推链，防止新回合的优势信号泄漏回旧回合
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    return advantages  # [Â_0, Â_1, ..., Â_{T-1}]
```

> **什么是 Bootstrap？** 在 TD 误差 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 中，$V(s_{t+1})$ 就是 bootstrap——我们不知道从 $s_{t+1}$ 开始真正能拿多少分，所以**用 Critic 的预测值 $V(s_{t+1})$ 来"代替"看不到的未来回报**，这种"用估计值替代真实值"的操作就叫 bootstrap（自举）。
>
> **为什么需要 `dones`？** 一个 rollout buffer 往往包含多个回合片段。用面试的例子来说：假设 AI 连续跑了两局面试，数据在 buffer 中紧挨着排列：
>
> | 位置 | $t=1$ | $t=2$ | $t=3$（第1局结束） | $t=4$（第2局开始） | $t=5$ | ... |
> |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
> | 状态 | $s_1^{(1)}$ | $s_2^{(1)}$ | $s_3^{(1)}$ | $s_1^{(2)}$ | $s_2^{(2)}$ | ... |
> | done | 0 | 0 | **1** | 0 | 0 | ... |
>
> **没有 `dones` 会怎样？** 计算 $t=3$（第 1 局最后一步）的 TD 误差时：$\delta_3 = r_3 + \gamma V(s_4) - V(s_3)$。但 $s_4$ 是第 2 局的初始状态——一场全新的面试！$V(s_4)$ 与第 1 局完全无关，不应被当作第 1 局末尾的 bootstrap 值。如果直接使用，等于把"第 2 局开局的预期分数"算进了第 1 局的优势评价，导致估计错误。
>
> **加上 `(1 - done_t)` 后**：当 $\text{done}_3 = 1$ 时，$(1 - \text{done}_3) = 0$，于是 $\delta_3 = r_3 + \gamma \cdot V(s_4) \cdot 0 - V(s_3) = r_3 - V(s_3)$。bootstrap 值被归零，第 1 局的最后一步不再"偷看"第 2 局的信息。同时，GAE 递推链 $\hat{A}_3 = \delta_3 + \gamma\lambda \cdot 0 \cdot \hat{A}_4 = \delta_3$ 也被截断，第 2 局的优势不会泄漏回第 1 局。

---

**开源代码参考：**
在实际应用中，REINFORCE 由于方差过大，几乎不再被单独用于复杂任务；Actor-Critic（特别是其变体 A2C/A3C）则被广泛使用。目前最主流的强化学习开源库如 **Stable-Baselines3** 和 Hugging Face 的 **TRL (Transformer Reinforcement Learning)** 都提供了这些基础算法的实现。上面三个算法各自的完整 PyTorch 实现已附在对应小节末尾，从模型定义、数据采集到参数更新的完整前向流程可以直接参考。

**参考资料：** 本文的理论基础主要来自 Sutton & Barto 的经典教材 [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)，其中 Chapter 13 详细讨论了策略梯度方法。更偏实践的入门推荐 [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)，提供了从概念到代码的完整路径。

**这些基础算法与大模型 RLHF 的关系：** 本文介绍的三种算法都是经典 RL 场景中的理论基础，它们本身没有直接的 RLHF 版本，但它们的核心思想直接催生了大模型对齐中的关键算法：

- **Actor-Critic → PPO**：PPO 本质上是加了裁剪机制的 Actor-Critic，是目前 RLHF 中最主流的算法。在 RLHF 场景下需要四个模型（Actor、Critic、Reference、Reward）协同工作，详见[下一篇](/chengYi-xun/posts/18-trpo-ppo/)。
- **REINFORCE → GRPO**：GRPO（Group Relative Policy Optimization）的核心思想来自 REINFORCE——对同一个 prompt 生成一组回答，用组内相对排名替代 Critic 的价值估计，彻底去掉了 Critic 模型。

> 下一篇：[笔记｜生成模型（十七）：信任区域与近端策略优化 (从 TRPO 到 PPO)](/chengYi-xun/posts/18-trpo-ppo/)
