---
title: 笔记｜强化学习（一）：强化学习基础与策略梯度
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

> 本文为大模型对齐与强化学习系列的第一篇，主要回顾强化学习（RL）的基础概念，并严格推导策略梯度定理，为后续深入理解 REINFORCE、Actor-Critic、PPO 和 GRPO 打下坚实的理论基础。
>
> ⬅️ 上一篇：[笔记｜生成模型（十五）：Flux 架构解析](/chengYi-xun/posts/15-flux/)
>
> ➡️ 下一篇：[笔记｜强化学习（一续）：从 REINFORCE 到 Actor-Critic](/chengYi-xun/posts/51-reinforce-ac/)


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
   注意，这个公式虽然只对 $s_0$ 下的**第一步动作**求和，但 $Q^{\pi_\theta}(s_0, a)$ 已经包含了"执行动作 $a$ 之后，沿整条轨迹遵循策略 $\pi_\theta$ 所能获得的全部未来奖励的期望"。所以这个加权平均的结果等于从 $s_0$ 出发、遵循策略 $\pi_\theta$ 的**整条轨迹的期望总回报**——$Q$ 值把未来"打包"进去了。

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

但这里面临一个根本性困难：$J(\theta)$ 是一个**期望值**——它取决于策略 $\pi_\theta$ 生成的轨迹分布，而轨迹中涉及随机采样和环境交互，无法直接对 $\theta$ 求导。**策略梯度定理（Policy Gradient Theorem）** 正是为了解决这个问题而诞生的：它给出了一个**不需要对环境求导**、可以通过采样近似的梯度表达式。换句话说，即使环境是一个完全不可微的黑盒（物理引擎、游戏模拟器、大模型评分器），我们依然能精确计算策略梯度的方向。

根据**策略梯度定理**，梯度的解析表达式为：
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot Q^{\pi_\theta}(s_t, a_t) \right]
$$

**推导的核心：对数导数技巧 (Log-Derivative Trick)**

上面的公式中出现了 $\nabla_\theta \log \pi_\theta$——为什么对概率分布求梯度会变成对 $\log$ 概率求梯度？（*注：本文及后续强化学习相关文章中，$\log$ 均指代以 $e$ 为底的自然对数 $\ln$*）下面的推导将严格证明这个结论，其核心贡献有三点：**(1)** 把"对轨迹概率分布求导"转化为"对 $\log$ 概率求期望"，从而可以通过**蒙特卡洛采样**近似；**(2)** 证明环境的状态转移概率 $P(s'|s,a)$ 对 $\theta$ 的梯度为零，因此**完全不需要知道环境的内部规则**——这为后面 REINFORCE 的代理损失（Surrogate Loss）设计提供了数学保证；**(3)** 引入因果性裁剪，降低梯度估计的方差。

我们从目标函数 $J(\theta)$ 的期望定义出发。根据前面的定义，目标函数是整条轨迹累积奖励的期望：
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

**逐项拆解这个公式：**

| 组成部分 | 写法 | 含义 |
|:---|:---|:---|
| 外层期望 | $\mathbb{E}_{\tau \sim \pi_\theta}[\cdots]$ | 用当前策略 $\pi_\theta$ 跑很多条轨迹，对括号内的量取平均 |
| 时间求和 | $\sum_{t=0}^{\infty}$ | 对轨迹上每个时间步 $t$ 的贡献求和 |
| 折扣因子 | $\gamma^t$（$\gamma \in [0,1)$） | 越远的未来贡献越小，$\gamma=0.99$ 表示 100 步后衰减到约 $\frac{1}{3}$ |
| 策略梯度方向 | $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ | "让动作 $a_t$ 在状态 $s_t$ 下更可能"的参数更新方向 |
| 信用分配权重 | $Q^{\pi_\theta}(s_t, a_t)$ | 在状态 $s_t$ 下执行动作 $a_t$ 后，**未来能获得的期望总回报** |

**整体物理意义**：对轨迹上的每个时刻 $t$——

$$
\underbrace{\gamma^t}_{\text{时刻}\,t\,\text{的重要性}} \times \underbrace{\nabla_\theta \log \pi_\theta(a_t|s_t)}_{\text{增大}\,a_t\,\text{概率的方向}} \times \underbrace{Q^{\pi_\theta}(s_t, a_t)}_{\text{这个动作有多好}}
$$

- 如果 $Q$ 值**大**（这个动作带来高回报）→ 沿梯度方向走大步 → **强化**这个动作
- 如果 $Q$ 值**小**（甚至为负） → 沿反方向走 → **抑制**这个动作
- $\gamma^t$ 有两个作用：(1) 越早的决策权重越大（因为早期决策影响整条轨迹）；(2) 让 $\sum_{t=0}^{\infty}$ 这个理论上的无穷求和收敛——因为 $\gamma < 1$，所以 $\gamma^t$ 指数级衰减到零，总和有界。实际应用中回答长度有限（如最多 2048 token），$\sum$ 自然截断，不需要担心发散

**因果性为什么重要？** 上一步的公式里，每个时刻的梯度都乘以**整条轨迹的总奖励** $R(\tau)$。但 $t=5$ 时刻的动作根本不可能影响 $t=0 \sim 4$ 时已经发生的奖励，把这些"过去的奖励"也算进去纯粹是噪声。因果性裁剪掉这些无关项，只保留 $t$ 时刻及之后的累积回报 $Q^{\pi_\theta}(s_t, a_t) = \mathbb{E}\left[\sum_{k=t}^{\infty} \gamma^{k-t} r_k \mid s_t, a_t\right]$——**方差更小，学习更快**。

通过这个技巧，我们将原本难以计算的"概率梯度的积分"，转化为了可以通过**蒙特卡洛采样（让智能体自己在环境里跑几圈收集数据）**来近似计算的"期望值"。

> **延伸阅读：** 关于策略梯度定理的更详细推导和直觉理解，推荐以下两篇经典文章：
>
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

> **参考资料：**
>
> 1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT press. Chapter 13.
> 2. [Part 3: Intro to Policy Optimization — OpenAI Spinning Up](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
> 3. [Policy Gradient Algorithms — Lil'Log](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

> 下一篇：[笔记｜强化学习（一续）：从 REINFORCE 到 Actor-Critic](/chengYi-xun/posts/51-reinforce-ac/)