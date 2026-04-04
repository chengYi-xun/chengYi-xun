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

> 本文为系列第二篇。在上一篇中，我们介绍了策略梯度和 Actor-Critic 架构。然而，基础的策略梯度方法存在更新步长难以控制、训练不稳定的问题。本文将详细推导如何通过限制策略更新幅度来保证训练的单调递增，从 TRPO 的数学思想一路演进到目前大模型 RLHF 的基石——PPO 算法。

# 策略更新的痛点：步长控制（走得太快容易扯着蛋）

在上一篇中，我们讲到了如何通过策略梯度让模型“变聪明”。在标准的策略梯度方法中，我们通过梯度上升来更新策略参数 $\theta$：
$$
\theta_{\text{new}} = \theta_{\text{old}} + \alpha \nabla_\theta J(\theta)
$$

这里的学习率 $\alpha$ 决定了更新的**步长**。

**核心思考出发点**：强化学习和普通的监督学习（比如教小孩子认猫狗图片）有一个本质区别。在监督学习中，数据集是固定的，模型学歪了，大不了下一批数据再纠正回来。但在强化学习中，**模型自己产生的数据，决定了它下一步能学到什么（On-policy）**。

如果步长过大，新策略 $\pi_{\text{new}}$ 可能会偏离旧策略太远。打个比方，一个刚学会走路的婴儿，你突然让他跑 100 米，他大概率会摔个狗吃屎（策略崩溃）。一旦摔倒了，他以后连路都不敢走了，采集到的数据全是“摔倒”，模型就再也学不好了。

**意义**：为了防止这种灾难性的“策略崩溃”，我们需要一种方法来**限制每次策略更新的幅度**，保证模型是一步一个脚印、稳扎稳打地变好（单调递增），或者至少保证新旧策略不要相差太远。这就是 TRPO 和 PPO 诞生的核心驱动力。

---

# TRPO：画个圈圈，在圈里找最优解

为了解决上述问题，Schulman 等人在 2015 年提出了 TRPO (Trust Region Policy Optimization) 算法。

**核心思想**：既然怕走得太远摔倒，那我就在你脚下画一个“信任区域（Trust Region）”（比如半径 1 米的圆）。你只能在这个圈里找最好的下一步，绝对不能出圈。

## 替代目标函数 (Surrogate Objective)

TRPO 引入了**重要性采样（Importance Sampling）**。假设我们用旧策略 $\pi_{\theta_{\text{old}}}$ 采集了一批数据，我们希望用这批数据来评估并优化新策略 $\pi_\theta$。

我们定义重要性比率（Probability Ratio）：
$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

TRPO 构建了一个替代目标函数：
$$
\mathcal{L}^{\text{CPI}}(\theta) = \mathbb{E}_{t} \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t \right] = \mathbb{E}_{t} \left[ r_t(\theta) \hat{A}_t \right]
$$
其中 $\hat{A}_t$ 是使用旧策略估计出的优势函数（通常使用广义优势估计 GAE）。

## KL 散度约束

为了保证新旧策略相差不远，TRPO 巧妙地将问题转化为一个**带约束的优化问题**：在保证新旧策略的 KL 散度（Kullback-Leibler Divergence）不超过某个阈值 $\delta$ 的前提下，最大化替代目标函数。

$$
\max_\theta \mathbb{E}_{t} \left[ r_t(\theta) \hat{A}_t \right]
$$
$$
\text{subject to} \quad \mathbb{E}_t \left[ \text{KL}[\pi_{\theta_{\text{old}}}(\cdot|s_t), \pi_\theta(\cdot|s_t)] \right] \le \delta
$$

**TRPO 的缺点**：
TRPO 在理论上非常严谨，保证了策略的单调改进。但由于它需要求解带约束的优化问题（通常使用共轭梯度法和 Fisher 信息矩阵），计算极其复杂，难以与包含 Dropout、BatchNorm 等复杂结构的大型深度神经网络（如 Transformer）结合。

---

# PPO：大道至简的“裁剪”艺术

**核心思考出发点**：TRPO 虽然理论完美，但计算“KL 散度约束”就像是在解一道极其复杂的微积分方程，算得太慢了，根本没法用在参数动辄上亿的深度神经网络（比如 Transformer）上。能不能用一种极其简单、粗暴但有效的方法，达到和 TRPO 一样的“不出圈”效果呢？

OpenAI 在 2017 年给出了答案：**PPO (Proximal Policy Optimization)**。PPO 将 TRPO 复杂的硬约束，转化为了简单优雅的**“裁剪（Clipping）”机制**。

## 裁剪目标函数 (Clipped Surrogate Objective)

PPO 的核心创新在于其裁剪目标函数：
$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_{t} \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$
其中 $\epsilon$ 是一个超参数（通常设为 0.2）。

让我们详细解析这个公式的巧妙之处：

1. **第一项**：$r_t(\theta) \hat{A}_t$ 是正常的替代目标（与 TRPO 相同）。
2. **第二项**：$\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t$ 将重要性比率强行截断在 $[0.8, 1.2]$ 之间。
3. **取最小值 (min)**：这是最关键的一步，它构建了一个悲观的下界（Pessimistic Bound）。

**分情况讨论：**

- **当 $\hat{A}_t > 0$ 时**：该动作比平均好，我们希望增加其概率（即增大 $r_t(\theta)$）。
  - 如果 $r_t(\theta)$ 增加到超过 $1+\epsilon$，裁剪项生效，梯度变为 0。这意味着：虽然这个动作很好，但我们**不允许你一次性把它的概率增加太多**，见好就收。
- **当 $\hat{A}_t < 0$ 时**：该动作比平均差，我们希望降低其概率（即减小 $r_t(\theta)$）。
  - 如果 $r_t(\theta)$ 减小到低于 $1-\epsilon$，裁剪项生效，梯度变为 0。这意味着：虽然这个动作很差，但我们**不允许你一次性把它的概率降得太低**，防止矫枉过正。

通过这种简单的裁剪机制，PPO 成功地将新策略限制在旧策略的“信任区域”内，无需复杂的二阶优化计算，直接使用 Adam 等一阶优化器即可高效训练。

## PPO 的完整损失函数与代码实现

在实际应用（如大模型 RLHF）中，PPO 的总损失函数通常包含三部分：
$$
\mathcal{L}^{\text{PPO}}(\theta) = \mathcal{L}^{\text{CLIP}}(\theta) - c_1 \mathcal{L}^{\text{VF}}(\theta) + c_2 S[\pi_\theta](s_t)
$$
1. **策略损失 $\mathcal{L}^{\text{CLIP}}$**：即上述的裁剪目标函数。
2. **价值损失 $\mathcal{L}^{\text{VF}}$**：Critic 网络预测的价值 $V_\theta(s_t)$ 与实际回报 $V_t^{\text{target}}$ 的均方误差，用于训练 Critic。
3. **熵奖励（Entropy Bonus）$S[\pi_\theta]$**：鼓励策略保持一定的随机性，防止过早收敛到局部最优（探索与利用的权衡）。

**开源代码参考：**
目前大模型微调中最常用的 PPO 实现是 Hugging Face 的 **TRL (Transformer Reinforcement Learning)** 库（`trl.PPOTrainer`）。其核心的裁剪损失计算可以用以下 PyTorch 伪代码表示：

```python
# ratio: r_t(θ) = exp(log_prob - old_log_prob)
# advantages: 优势函数 A_t
ratio = torch.exp(log_prob - old_log_prob)

# 计算两个 surrogate
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages

# 取最小值并加负号（因为 PyTorch 优化器是梯度下降，而我们要最大化目标函数）
policy_loss = -torch.min(surr1, surr2).mean()

# 价值损失 (MSE)
value_loss = F.mse_loss(values, returns)

# 总损失
loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy
```

## PPO 在大模型微调中的痛点

PPO 凭借其简单、高效、稳定的特点，成为了 ChatGPT 等大模型 RLHF 的标准算法。

然而，随着模型参数量飙升到百亿（10B）甚至千亿级别，PPO 暴露出一个致命的缺点：**显存开销巨大**。
在 PPO 中，我们不仅需要加载庞大的 Actor 模型，还需要加载一个同样庞大的 Critic 模型来估计价值函数。这往往超出了单张甚至多张 GPU 的显存极限。

为了解决这个问题，学术界演化出了两条不同的路线：
1. **绕过强化学习**：直接使用偏好数据优化语言模型，即 **DPO (Direct Preference Optimization)** 算法。
2. **改进强化学习**：丢弃 Critic 网络，通过组内相对评分估计优势，即 **GRPO (Group Relative Policy Optimization)** 算法。

接下来的两篇文章，我们将分别探讨这两条激动人心的前沿路线。

> 下一篇：[笔记｜生成模型（十八）：大模型对齐的另一条路：DPO (Direct Preference Optimization)](/chengYi-xun/2026/04/03/19-dpo/)
