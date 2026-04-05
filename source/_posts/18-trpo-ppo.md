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

基于采集的数据，我们估算出"讲项目经历"的优势 $\hat{A} = +3$（好于平均），"讲兴趣爱好"的优势 $\hat{A} = -3$（差于平均）。

**太激进的更新会怎样？** 如果学习率 $\alpha$ 设得太大，一次更新后策略可能变成：
- $\pi_\text{new}$("讲项目经历" $| s_1$) = 99%
- $\pi_\text{new}$("讲兴趣爱好" $| s_1$) = 1%

看起来没毛病？但问题是——**优势 $\hat{A} = +3$ 是在旧策略 $\pi_\text{old}$（70%/30%）下估算出来的**。新策略几乎只选"讲项目经历"，数据分布已经完全不同了，旧的优势估计可能完全不准。这就好比一个刚学会走路的婴儿，你突然让他跑 100 米——大概率会摔个狗吃屎（策略崩溃）。一旦崩了，采集到的数据全是垃圾，模型就再也学不好了。

**核心困境**：强化学习和监督学习有一个本质区别——**模型自己产生的数据，决定了它下一步能学到什么**。如果一次更新走得太远，新策略采集到的数据分布和旧策略截然不同，旧数据上的优势估计就失效了，训练就会崩溃。

**意义**：我们需要一种方法来**限制每次策略更新的幅度**，保证模型稳扎稳打地变好。这就是 TRPO 和 PPO 诞生的核心驱动力。

---

# TRPO：画个圈圈，在圈里找最优解

为了解决上述问题，Schulman 等人在 2015 年提出了 TRPO (Trust Region Policy Optimization) 算法。

**核心思想用例子说**：既然怕 AI 面试助手一次更新走得太远（从 70% 跳到 99%），那我就给它画一个"信任区域"——你每次更新后的新策略，和旧策略之间的"差距"不能超过一个阈值 $\delta$。

## 替代目标函数 (Surrogate Objective)

**先看例子**：在面试的场景中，我们已经用旧策略 $\pi_\text{old}$（70% 选"讲项目经历"）采集了一批面试数据。现在我们想用这批旧数据来评估新策略 $\pi_\text{new}$ 的效果。

关键工具是**重要性比率（Probability Ratio）**：
$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

在我们的例子中，假设新策略把"讲项目经历"的概率从 70% 提升到了 84%：
$$r = \frac{0.84}{0.70} = 1.2$$

$r = 1.2$ 意味着新策略选"讲项目经历"的概率是旧策略的 1.2 倍。我们用 $r$ 乘以优势 $\hat{A}$ 来估计新策略的表现：
$$\mathcal{L}^{\text{CPI}}(\theta) = \mathbb{E}_{t} \left[ r_t(\theta) \hat{A}_t \right]$$

**一般化的算法原理**：TRPO 引入了**重要性采样（Importance Sampling）**。假设我们用旧策略 $\pi_{\theta_{\text{old}}}$ 采集了一批数据，我们希望用这批数据来评估并优化新策略 $\pi_\theta$。TRPO 构建了上述替代目标函数，其中 $\hat{A}_t$ 是使用旧策略估计出的优势函数（通常使用广义优势估计 GAE）。

## KL 散度约束

**先看例子**：回到面试助手。我们限制新策略和旧策略之间的 KL 散度不超过 $\delta$。KL 散度可以直观理解为新旧策略"分布差距"的度量。

如果旧策略是 70%/30%，新策略是 84%/16%，KL 散度约为 0.05（较小的差距）。
如果旧策略是 70%/30%，新策略是 99%/1%，KL 散度约为 0.88（巨大的差距）。

通过限制 $\text{KL} \le \delta$（比如 $\delta = 0.01$），我们就能保证新策略不会偏离旧策略太远。

**一般化的算法原理**：TRPO 将问题转化为一个**带约束的优化问题**：在保证新旧策略的 KL 散度不超过阈值 $\delta$ 的前提下，最大化替代目标函数。

$$
\max_\theta \mathbb{E}_{t} \left[ r_t(\theta) \hat{A}_t \right]
$$
$$
\text{subject to} \quad \mathbb{E}_t \left[ \text{KL}[\pi_{\theta_{\text{old}}}(\cdot|s_t), \pi_\theta(\cdot|s_t)] \right] \le \delta
$$

**TRPO 的缺点**：
TRPO 在理论上非常严谨，保证了策略的单调改进。但由于它需要求解带约束的优化问题（通常使用共轭梯度法和 Fisher 信息矩阵），计算极其复杂，难以与包含 Dropout、BatchNorm 等复杂结构的大型深度神经网络（如 Transformer）结合。

---

# PPO：大道至简的"裁剪"艺术

**核心思考出发点**：TRPO 虽然理论完美，但计算"KL 散度约束"就像是在解一道极其复杂的微积分方程，算得太慢了，根本没法用在参数动辄上亿的深度神经网络上。能不能用一种极其简单的方法，达到和 TRPO 一样的"不出圈"效果呢？

OpenAI 在 2017 年给出了答案：**PPO (Proximal Policy Optimization)**。PPO 将 TRPO 复杂的硬约束，转化为了简单优雅的**"裁剪（Clipping）"机制**。

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

通过这种简单的裁剪机制，PPO 成功地将新策略限制在旧策略的"信任区域"内，无需复杂的二阶优化计算，直接使用 Adam 等一阶优化器即可高效训练。

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
