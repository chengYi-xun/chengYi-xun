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

> 本文为系列第二篇。在上一篇中，我们介绍了策略梯度和 Actor-Critic 架构。然而，基础的策略梯度方法存在更新步长难以控制、训练不稳定的问题。本文将详细推导如何通过限制策略更新幅度来保证训练的单调递增，从 TRPO 的数学思想一路演进到目前大模型 RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）的基石——PPO 算法。

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

# 策略更新的痛点：步长控制

在上一篇中，我们讲到了如何通过策略梯度让模型"变聪明"。在标准的策略梯度方法中，我们通过梯度上升来更新策略参数 $\theta$：
$$
\theta_{\text{new}} = \theta_{\text{old}} + \alpha \nabla_\theta J(\theta)
$$

这里的学习率 $\alpha$ 决定了更新的**步长**。

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

**这就引出了核心矛盾**：$\hat{A}$ 是在"假设策略不变"的前提下估计的参数更新幅度，但更新本身又会改变策略。步子越大，策略变化越大，$\hat{A}$ 就越不准确，更新就越不可靠。而强化学习和监督学习的本质区别在于——**模型自己产生的数据，决定了它下一步能学到什么**。一旦因为不可靠的更新导致策略崩溃，采集到的数据也会变成垃圾，模型就再也学不回来了。

**意义**：我们需要一种方法来**限制每次策略更新的幅度**，保证模型稳扎稳打地变好。这就是 TRPO 和 PPO 诞生的核心驱动力。

---

# TRPO：画个圈圈，在圈里找最优解

为了解决上述问题，Schulman 等人在 2015 年提出了 TRPO (Trust Region Policy Optimization) 算法。

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

**为什么不能只用一阶梯度？** 根本原因在于 TRPO 的约束加在**概率分布空间**上（KL 散度），而梯度下降操作在**参数空间**中。这两个空间的"距离"没有简单的对应关系——参数空间中的一小步，可能在分布空间中造成巨大变化；反之亦然。一阶梯度只能告诉你"哪个方向能提升目标函数"，但无法告诉你"沿这个方向走多远，分布才会变化到 KL 约束的边界"。

打个比方：想象你在山上徒步，规定"只能离当前位置走 100 米"（KL 约束）。一阶梯度告诉你"正北方向最陡"，于是你往北走 100 米。但如果地形是这样的——往北走 1 米海拔就升 50 米（分布极敏感），往东走 100 米海拔才升 1 米（分布不敏感）——那径直往北走 100 米，等于在分布空间中远远飞出了信任区域。

**Fisher 信息矩阵** $F$ 恰恰描述了这种"地形"——参数空间到分布空间的**局部映射关系**。$F$ 在某个方向的值越大，说明分布对该方向的参数变化越敏感（只能走小步）；值越小，说明越不敏感（可以走大步）。**自然梯度** $F^{-1}g$ 将一阶梯度 $g$ 按照这个"地形"重新缩放，使更新方向在**分布空间**（而非参数空间）中最优，最高效地利用有限的 KL 预算。

具体来说，TRPO 采用自然梯度方法：先计算策略梯度 $g$，再通过 $F$——本质上是 KL 散度对参数 $\theta$ 的二阶导数——来求解 $F^{-1}g$ 作为更新方向。由于 $F$ 是一个 $d \times d$ 的矩阵（$d$ 为参数数量），对于大型网络根本无法显式存储，TRPO 使用**共轭梯度法**近似求解，但这需要反复计算 **Hessian-向量积**，依赖于高效的二阶导数计算。

**复杂网络结构的问题。** 这一要求使得 TRPO 难以与现代深度网络结合：**Dropout** 在每次前向传播中随机丢弃神经元，使有效网络结构不断变化，FIM 在随机子网络上的估计变得不稳定；**BatchNorm** 的归一化统计量依赖于当前 mini-batch 中的所有样本，引入了样本间的相互依赖，而 FIM 的推导假设样本独立；**Transformer** 中的多头注意力、残差连接等组件使得损失函数的 Hessian 结构极其复杂，二阶导数的计算既慢又不稳定。这些因素叠加在一起，使得 TRPO 在参数动辄上亿的现代深度网络上几乎无法实际使用。

## TRPO 的实现

下面用 PyTorch 风格的伪代码展示 TRPO 的完整训练流程。注意其中 Step 4 和 Step 5 的二阶优化和线搜索——这正是 TRPO 的计算瓶颈所在。

```python
import copy
import torch
import torch.nn.functional as F

# ============================================================
# 模型定义
# actor:  策略网络 π_θ(a|s), 输入状态, 输出动作概率分布
# critic: 价值网络 V_φ(s), 输入状态, 输出标量价值
# ============================================================
actor = ActorModel(state_dim, action_dim)
critic = CriticModel(state_dim)
old_actor = copy.deepcopy(actor)   # 旧策略的冻结副本
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
delta = 0.01  # KL 约束阈值

# ============================================================
# Step 1: 数据采集 (Rollout)
# 用旧策略 π_old 与环境交互, 采集一批轨迹 (trajectory)
# 每条数据是一个四元组: (状态 s_t, 动作 a_t, 奖励 r_t, log π_old(a_t|s_t))
# ============================================================
buffer = []
state = env.reset()
for t in range(T):
    with torch.no_grad():
        dist = old_actor(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
    next_state, reward, done = env.step(action)
    buffer.append((state, action, reward, log_prob))
    state = next_state if not done else env.reset()

states, actions, rewards, old_log_probs = collate(buffer)

# ============================================================
# Step 2: 优势估计 (GAE)
# ============================================================
with torch.no_grad():
    values = critic(states)
    advantages = compute_gae(rewards, values, gamma=0.99, lam=0.95)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = advantages + values

# ============================================================
# Step 3: 计算替代目标和 KL 散度
# ============================================================
dist = actor(states)
new_log_probs = dist.log_prob(actions)
ratio = torch.exp(new_log_probs - old_log_probs)
surrogate = (ratio * advantages).mean()

old_dist = old_actor(states)
kl = torch.distributions.kl_divergence(old_dist, dist).mean()

# ============================================================
# Step 4: 自然梯度 — 求解 F⁻¹g (二阶优化!)
# ============================================================
# 4a. 策略梯度 g = ∇_θ L^CPI
g = flatten(torch.autograd.grad(surrogate, actor.parameters()))

# 4b. Fisher-向量积 Fv (避免显式构建 d×d 的 F 矩阵)
def fisher_vector_product(v):
    kl_grad = flatten(torch.autograd.grad(kl, actor.parameters(), create_graph=True))
    return flatten(torch.autograd.grad(kl_grad @ v, actor.parameters())) + 0.1 * v

# 4c. 共轭梯度法近似求解 F⁻¹g (迭代 10 次)
step_dir = conjugate_gradient(fisher_vector_product, g, max_iter=10)

# ============================================================
# Step 5: 线搜索确定步长 (确保 KL ≤ δ 且目标提升)
# ============================================================
max_step = torch.sqrt(2 * delta / (step_dir @ fisher_vector_product(step_dir)))
old_params = flatten_params(actor).detach()

for shrink in [1.0, 0.5, 0.25, 0.125]:
    new_params = old_params + shrink * max_step * step_dir
    assign_params(actor, new_params)
    new_surr = compute_surrogate(actor, states, actions, old_log_probs, advantages)
    new_kl = compute_kl(old_actor, actor, states)
    if new_kl <= delta and new_surr >= surrogate:
        break
    assign_params(actor, old_params)  # 不满足则回退

# ============================================================
# Step 6: 更新 Critic (标准一阶优化)
# ============================================================
critic_optimizer.zero_grad()
F.mse_loss(critic(states), returns).backward()
critic_optimizer.step()

# Step 7: 同步旧策略
old_actor.load_state_dict(actor.state_dict())
```

注意：上面的代码是 vanilla RL（经典强化学习）场景中的 TRPO。**TRPO 从未被应用于大模型 RLHF**——原因正是前文详述的二阶优化瓶颈：Fisher 矩阵和共轭梯度法在数十亿参数的语言模型上根本无法计算。当 2022-2023 年 RLHF 成为大模型对齐的核心技术时，学界直接选择了 PPO 而跳过了 TRPO。这正是 PPO 诞生的全部意义——用一阶裁剪替代二阶约束，使信任区域方法能够扩展到工业级大模型。

---

# PPO：大道至简的"裁剪"艺术

**核心思考出发点**：TRPO 虽然理论完美，但求解 KL 约束所需的二阶优化（Fisher 矩阵、共轭梯度）算得太慢了，根本没法用在参数动辄上亿的深度神经网络上。能不能用一种极其简单的方法，达到和 TRPO 一样的"不出圈"效果呢？

OpenAI 在 2017 年给出了答案：**PPO (Proximal Policy Optimization)**。PPO 的核心思路是：用简单的一阶裁剪操作替代 TRPO 的二阶 KL 约束。PPO 不去精确求解"KL 约束内的最优方向"（那需要 Fisher 矩阵来理解分布空间的几何），而是直接用 $\text{clip}(r, 1-\epsilon, 1+\epsilon)$ 限制每个动作的概率比率——这是一个粗糙但有效的近似，不需要理解参数-分布映射的精确关系。因此 PPO **只需要标准的一阶梯度**，可以直接使用 Adam 等优化器，对网络结构没有任何限制——Dropout、BatchNorm、多头注意力、残差连接等组件都不影响其正常工作。

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

1. **策略损失 $\mathcal{L}^{\text{CLIP}}$**：即上述的裁剪目标函数。
2. **价值损失 $\mathcal{L}^{\text{VF}}$**：Critic 网络预测的价值 $V_\theta(s_t)$ 与实际回报 $V_t^{\text{target}}$ 的均方误差，用于训练 Critic。
3. **熵奖励（Entropy Bonus）$S[\pi_\theta]$**：鼓励策略保持一定的随机性，防止过早收敛到局部最优（探索与利用的权衡）。

下面用 PyTorch 风格的伪代码展示 PPO 的完整训练流程。对比上面 TRPO 的实现，可以清楚地看到 PPO 的核心简化：**没有 Fisher 矩阵、没有共轭梯度、没有线搜索**——全部替换为简单的裁剪 + 标准 Adam 优化器。

```python
import torch
import torch.nn.functional as F

# ============================================================
# 模型定义 (与 TRPO 相同: Actor + Critic)
# ============================================================
actor = ActorModel(state_dim, action_dim)
critic = CriticModel(state_dim)
optimizer = torch.optim.Adam(                    # 标准一阶优化器!
    list(actor.parameters()) + list(critic.parameters()), lr=3e-4
)
clip_range = 0.2      # ε, 裁剪范围
vf_coef = 0.5         # 价值损失权重 c₁
entropy_coef = 0.01   # 熵奖励权重 c₂
K_epochs = 4          # 同一批数据重复使用的 epoch 数

# ============================================================
# Step 1: 数据采集 (同 TRPO)
# ============================================================
buffer = []
state = env.reset()
for t in range(T):
    with torch.no_grad():
        dist = actor(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = critic(state)
    next_state, reward, done = env.step(action)
    buffer.append((state, action, reward, log_prob, value))
    state = next_state if not done else env.reset()

states, actions, rewards, old_log_probs, old_values = collate(buffer)

# ============================================================
# Step 2: 优势估计 (同 TRPO)
# ============================================================
with torch.no_grad():
    advantages = compute_gae(rewards, old_values, gamma=0.99, lam=0.95)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = advantages + old_values

# ============================================================
# Step 3: 多 epoch 小批量更新 (PPO 核心 — 替代了 TRPO 的 Step 4~5)
# 同一批数据安全地复用 K 次, 每次用 clip 防止偏离过远
# ============================================================
for epoch in range(K_epochs):
    for idx in minibatch_indices(len(states), batch_size=64):
        s, a = states[idx], actions[idx]
        old_lp, adv, ret = old_log_probs[idx], advantages[idx], returns[idx]

        # --- 前向传播 ---
        dist = actor(s)                           # 新策略分布 π_θ(·|s)
        new_log_probs = dist.log_prob(a)          # log π_θ(a|s)
        values = critic(s).squeeze()              # V_φ(s)
        entropy = dist.entropy().mean()           # 策略熵 S[π_θ]

        # --- 策略损失 (裁剪) ---
        ratio = torch.exp(new_log_probs - old_lp) # r = π_θ / π_old (重要性权重)
        surr1 = ratio * adv                        # 原始替代目标
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # --- 价值损失 ---
        value_loss = F.mse_loss(values, ret)

        # --- 总损失: L^CLIP - c₁·L^VF + c₂·S ---
        loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy

        # --- 反向传播 + 梯度裁剪 + 更新 (标准一阶优化!) ---
        optimizer.zero_grad()
        loss.backward()
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

**RLHF-PPO 的核心区别**在于奖励的计算。Vanilla PPO 的奖励直接来自环境，而 RLHF-PPO 的奖励经过 KL 惩罚修正：
$$
r_t = R_\psi(\text{prompt}, \text{response}) - \beta \cdot \text{KL}[\pi_\theta \| \pi_\text{ref}]
$$
其中 $\beta$ 控制 KL 惩罚的强度：$\beta$ 越大，Actor 越不敢偏离 Reference。

下面是 RLHF-PPO 的完整实现。对比上面的 vanilla 版本，核心差异在 Step 1（数据采集方式）和 Step 2（KL 惩罚修正奖励）：

```python
import torch
import torch.nn.functional as F

# ============================================================
# RLHF-PPO 的四模型架构
# ============================================================
actor = LanguageModel(...)                 # 正在训练的策略 π_θ
critic = ValueHead(actor.backbone)         # 价值网络 V_φ, 通常共享 Actor 底座
ref_model = LanguageModel(...)             # 冻结的 SFT 模型 π_ref
ref_model.requires_grad_(False)
reward_model = RewardModel(...)            # 冻结的奖励模型 R_ψ
reward_model.requires_grad_(False)

optimizer = torch.optim.Adam(
    list(actor.parameters()) + list(critic.parameters()), lr=1e-5
)
clip_range = 0.2
kl_coef = 0.1      # β, KL 惩罚系数

# ============================================================
# Step 1: 数据采集 — "环境"是: prompt → 生成回答 → 奖励模型打分
# ============================================================
prompts = sample_prompts(dataset)
with torch.no_grad():
    responses, old_log_probs = actor.generate(prompts, return_log_probs=True)
    old_values = critic(prompts, responses)
    ref_log_probs = ref_model.log_probs(prompts, responses)  # π_ref 的对数概率
    scores = reward_model(prompts, responses)                 # R_ψ 打分

# ============================================================
# Step 2: 计算 KL 惩罚修正后的奖励 (vanilla PPO 没有这一步!)
# r = R_ψ - β·(log π_θ - log π_ref)
# ============================================================
kl_penalty = kl_coef * (old_log_probs - ref_log_probs)      # 逐 token KL 惩罚
adjusted_rewards = compute_token_rewards(scores, kl_penalty) # 最后 token 加 R_ψ, 其余 -KL

# ============================================================
# Step 3: 优势估计 (同 vanilla PPO)
# ============================================================
with torch.no_grad():
    advantages = compute_gae(adjusted_rewards, old_values, gamma=1.0, lam=0.95)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = advantages + old_values

# ============================================================
# Step 4: 多 epoch 小批量更新 (同 vanilla PPO)
# ============================================================
for epoch in range(K_epochs):
    for idx in minibatch_indices(len(prompts), batch_size):
        new_log_probs = actor.log_probs(prompts[idx], responses[idx])
        values = critic(prompts[idx], responses[idx])
        entropy = compute_entropy(actor, prompts[idx])

        ratio = torch.exp(new_log_probs - old_log_probs[idx])
        surr1 = ratio * advantages[idx]
        surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages[idx]
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values, returns[idx])
        loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(actor.parameters()) + list(critic.parameters()), max_norm=1.0
        )
        optimizer.step()
```

**开源代码参考：** 目前大模型 RLHF 微调中最常用的 PPO 实现是 Hugging Face 的 **TRL (Transformer Reinforcement Learning)** 库（`trl.PPOTrainer`），其核心逻辑与上述代码一致。

## PPO 在大模型微调中的痛点

PPO 凭借其简单、高效、稳定的特点，成为了 ChatGPT 等大模型 RLHF 的标准算法。

然而，从上面的四模型架构可以看出，RLHF-PPO 的**显存开销巨大**：需要同时加载 Actor、Critic、Reference、Reward 四个模型。当模型参数量飙升到百亿（10B）甚至千亿级别时，即便 Reference 和 Reward 不需要梯度，仅它们的前向推理也会占据大量显存，加上 Actor 和 Critic 的参数、梯度和优化器状态，这往往远超单张甚至多张 GPU 的显存极限。

为了解决这个问题，学术界演化出了两条不同的路线：

1. **绕过强化学习**：直接使用偏好数据优化语言模型，即 **DPO (Direct Preference Optimization)** 算法。
2. **改进强化学习**：丢弃 Critic 网络，通过组内相对评分估计优势，即 **GRPO (Group Relative Policy Optimization)** 算法。

接下来的两篇文章，我们将分别探讨这两条激动人心的前沿路线。

> 下一篇：[笔记｜生成模型（十八）：大模型对齐的另一条路：DPO (Direct Preference Optimization)](/chengYi-xun/posts/19-dpo/)
