---
title: 笔记｜生成模型（十六）：强化学习基础与策略梯度
date: 2026-04-03 10:00:00
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

# 强化学习基础概念：从“训狗”说起

对于纯小白读者来说，理解强化学习（Reinforcement Learning, RL）最直观的例子就是**“训狗”**。
- **智能体（Agent）**：就是那只狗（在 AI 中就是我们的模型）。
- **环境（Environment）**：狗所处的现实世界。
- **状态（State）**：狗当前看到的画面、听到的口令（比如你喊“坐下”）。
- **动作（Action）**：狗做出的反应（比如坐下、趴下、或者跑开）。
- **奖励（Reward）**：如果狗做对了，你给它一块肉（正奖励）；做错了，你呵斥它（负奖励或零奖励）。

强化学习的核心目标，就是让这只狗（模型）在不断的“尝试-犯错-获得奖励”的过程中，自己摸索出一条规律：**在什么情况下，做什么动作，能吃到最多的肉**。这条规律，在数学上就叫做**策略（Policy）**。

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
其中，$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)$ 表示一条完整的交互轨迹（Trajectory）。

## 价值函数与优势函数：怎么判断一个动作是好是坏？

在训狗的过程中，狗不仅要看眼前的肉，还要考虑长远的利益（比如现在乖乖坐下，等会儿可能有大骨头）。为了评估策略的好坏，我们引入两个重要的价值函数：

1. **状态价值函数（State-Value Function）** $V^\pi(s)$：表示在状态 $s$ 下，遵循策略 $\pi$ 所能获得的期望累积奖励。
   $$
   V^\pi(s) = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \Big| s_t = s \right]
   $$
2. **动作价值函数（Action-Value Function）** $Q^\pi(s, a)$：表示在状态 $s$ 下执行动作 $a$，随后遵循策略 $\pi$ 所能获得的期望累积奖励。
   $$
   Q^\pi(s, a) = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \Big| s_t = s, a_t = a \right]
   $$

**优势函数（Advantage Function）** $A^\pi(s, a)$ 定义为动作价值与状态价值之差，用于衡量在状态 $s$ 下执行动作 $a$ 相比于平均表现（即 $V^\pi(s)$）的“优势”程度：
$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$
如果 $A^\pi(s, a) > 0$，说明动作 $a$ 比策略 $\pi$ 的平均表现更好；反之则更差。

---

# 策略梯度定理：如何让模型“变聪明”？

**核心思考出发点**：既然我们知道了什么是“好动作”（优势函数 $A > 0$），什么是“坏动作”（优势函数 $A < 0$），那我们该如何修改模型（神经网络）的参数，让它以后多做“好动作”，少做“坏动作”呢？

这就是**策略梯度（Policy Gradient）**的意义所在。它提供了一种数学上的指导，告诉模型参数应该往哪个方向调整。

在深度强化学习中，我们通常使用神经网络来参数化策略，记为 $\pi_\theta(a|s)$，其中 $\theta$ 是网络参数。我们的目标是找到最优参数 $\theta^*$，使得目标函数 $J(\theta)$ 最大化。

为了使用梯度上升法更新参数：$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$，我们需要计算目标函数的梯度 $\nabla_\theta J(\theta)$。

根据**策略梯度定理**，梯度的解析表达式为：
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot Q^{\pi_\theta}(s_t, a_t) \right]
$$

**推导的核心：对数导数技巧 (Log-Derivative Trick)**
为什么梯度中会出现 $\log$？这是因为我们在计算期望的梯度时，使用了以下恒等式：
$$
\nabla_\theta \pi_\theta(a|s) = \pi_\theta(a|s) \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)} = \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)
$$
这个技巧允许我们将梯度重新写成期望的形式，从而可以通过蒙特卡洛采样来近似计算。

### 梯度的直观理解

我们可以将上式中的 $Q^{\pi_\theta}(s_t, a_t)$ 替换为优势函数 $A^{\pi_\theta}(s_t, a_t)$，这不会改变梯度的期望，但能显著降低方差：
$$
\nabla_\theta J(\theta) = \mathbb{E}_{s_t, a_t \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^{\pi_\theta}(s_t, a_t) \right]
$$
这个公式的物理意义非常直观：
- $\nabla_\theta \log \pi_\theta(a_t|s_t)$ 指示了增加动作 $a_t$ 概率的参数更新方向。
- $A^{\pi_\theta}(s_t, a_t)$ 作为权重。如果优势为正，说明该动作好于平均，梯度会推动网络增加该动作的概率；如果优势为负，则降低该动作的概率。

---

# 从 REINFORCE 到 Actor-Critic

## REINFORCE 算法

最基础的策略梯度算法是 **REINFORCE**（也称为 Monte-Carlo Policy Gradient）。它直接使用一条完整轨迹的实际累积奖励（Return）$G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$ 来无偏估计 $Q^{\pi_\theta}(s_t, a_t)$。

REINFORCE 的更新公式为：
$$
\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t
$$

**REINFORCE 的缺点：**
1. **方差极大**：一条轨迹的累积奖励受环境随机性和策略随机性的影响极大。
2. **回合更新（Episodic）**：必须等一个完整的 Episode 结束才能计算 $G_t$ 并更新参数，学习效率低。

## 引入基线 (Baseline)

为了降低方差，我们可以在 $G_t$ 中减去一个与动作 $a_t$ 无关的基线 $b(s_t)$。通常，我们选择状态价值函数 $V(s_t)$ 作为基线：
$$
\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - V(s_t))
$$
这里的 $(G_t - V(s_t))$ 实际上就是优势函数 $A(s_t, a_t)$ 的蒙特卡洛估计。

## Actor-Critic 架构

为了进一步提高采样效率并实现单步更新（Step-by-step），我们引入了 **Actor-Critic** 架构。

在 Actor-Critic 中，我们同时训练两个神经网络：
1. **Actor（策略网络）** $\pi_\theta(a|s)$：负责根据状态选择动作。
2. **Critic（价值网络）** $V_\phi(s)$：负责估计状态价值，用于计算优势函数。

我们使用时序差分（TD, Temporal Difference）误差来估计优势函数：
$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$
此时，Actor 的参数更新为：
$$
\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta_t
$$
Critic 的参数更新为最小化 TD 误差的平方：
$$
\phi \leftarrow \phi + \beta \nabla_\phi (\delta_t)^2
$$

Actor-Critic 极大地降低了方差，并允许在线单步更新。然而，它仍然面临着策略更新步长难以控制的问题，过大的更新可能导致策略崩溃。这就引出了下一篇要讲的 TRPO 和 PPO 算法。

**开源代码参考与算法对比：**
在实际应用中，REINFORCE 由于方差过大，几乎不再被单独用于复杂任务；Actor-Critic（特别是其变体 A2C/A3C）则被广泛使用。
目前最主流的强化学习开源库如 **Stable-Baselines3** 和 Hugging Face 的 **TRL (Transformer Reinforcement Learning)** 都提供了这些基础算法的实现。

以一个极简的 REINFORCE 策略更新代码为例（PyTorch 实现）：
```python
# log_probs: 策略网络输出的对数概率 log \pi(a|s)
# returns: 计算出的累积奖励 G_t
policy_loss = []
for log_prob, G in zip(log_probs, returns):
    # 梯度上升等价于最小化负对数概率乘以回报
    policy_loss.append(-log_prob * G)

optimizer.zero_grad()
loss = torch.stack(policy_loss).sum()
loss.backward()
optimizer.step()
```

> 下一篇：[笔记｜生成模型（十七）：信任区域与近端策略优化 (从 TRPO 到 PPO)](/chengYi-xun/2026/04/03/18-trpo-ppo/)
