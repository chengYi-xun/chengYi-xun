---
title: 笔记｜生成模型（十九）：大模型在线 RL 破局者：GRPO 算法详解
date: 2025-08-19 10:00:00
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

> 本文为系列第四篇。在了解了 PPO 的显存痛点和 DPO 的离线局限性后，我们终于迎来了目前大模型在线 RL 的最前沿破局者——GRPO（Group Relative Policy Optimization）。本文将详细推导 GRPO 的核心思想，看它是如何优雅地丢弃 Critic 网络，实现高效的在线强化学习的。

# 在线 RL 的不可替代性与 Critic 的累赘

正如上一篇所言，DPO 虽然简单省显存，但它只能"死记硬背"人类给出的标准答案（离线学习）。为了让模型产生"顿悟"和自我进化，我们必须回归**在线强化学习（Online RL）**。

然而，PPO 算法中的 Critic 网络（价值网络）成为了最大的绊脚石。对于百亿参数的大模型，多维护一个 Critic 意味着显存开销直接翻倍。

**核心思考出发点**：既然 Critic 只是为了给出一个"及格线"（基准值 $V(s)$），我们能不能**彻底去掉 Critic 模型**，用一种更简单的方法来估计这个"及格线"？

---

# GRPO 的核心思想：矮子里拔高个

**先用一个具体的数学题例子来理解。**

假设大模型遇到了这道数学题（Prompt $s$）：

> **题目**：求 $\int_0^1 x^2 dx$ 的值。

我们让模型尝试用 $G = 4$ 种不同的方式解题，得到 4 个回答：

| 回答 | 模型输出 | 判题结果 | 奖励 $r_i$ |
|:---:|:---|:---:|:---:|
| $o_1$ | "$\int_0^1 x^2 dx = \frac{x^3}{3}\Big|_0^1 = \frac{1}{3}$" | 正确 | $r_1 = 1$ |
| $o_2$ | "先换元 $u = x^2$……最终 $= \frac{1}{3}$" | 正确（方法迂回但对） | $r_2 = 1$ |
| $o_3$ | "$\int_0^1 x^2 dx = \frac{x^2}{2}\Big|_0^1 = \frac{1}{2}$" | **错误**（积分公式用错） | $r_3 = 0$ |
| $o_4$ | "不会做" | **错误** | $r_4 = 0$ |

**PPO 的做法**：需要一个 Critic 网络来预测"这道题的基准分 $V(s)$ 大概是多少"。训练这个 Critic 本身就很耗显存。

**GRPO 的做法**：不需要 Critic！直接用这 4 个回答自身的奖励来计算"及格线"：

$$\mu_R = \frac{r_1 + r_2 + r_3 + r_4}{4} = \frac{1 + 1 + 0 + 0}{4} = 0.5$$

$$\sigma_R = \sqrt{\frac{(1-0.5)^2 + (1-0.5)^2 + (0-0.5)^2 + (0-0.5)^2}{4}} = 0.5$$

每个回答的**相对优势**为：

| 回答 | 奖励 $r_i$ | 优势 $\hat{A}_i = \frac{r_i - \mu_R}{\sigma_R}$ | 含义 |
|:---:|:---:|:---:|:---|
| $o_1$（直接积分，正确） | 1 | $+1$ | 高于平均，**增大**这种回答的概率 |
| $o_2$（换元法，正确） | 1 | $+1$ | 高于平均，**增大**这种回答的概率 |
| $o_3$（公式用错） | 0 | $-1$ | 低于平均，**减小**这种回答的概率 |
| $o_4$（不会做） | 0 | $-1$ | 低于平均，**减小**这种回答的概率 |

**关键洞察**：即使这道题非常难，所有 4 个回答都做错了（$r = [0, 0, 0, 0]$），均值为 0，标准差也为 0，优势全为 0——模型不更新，没有噪声梯度。如果有一个碰巧推导多对了一步（$r = [0.1, 0, 0, 0]$），那它的优势就是正的，模型就会向这个"相对最好"的回答学习。这就是**矮子里拔高个**。

---

# GRPO 的数学推导与损失函数构建

## 1. 组内相对优势计算

**一般化的算法原理**：

给定一个输入状态 $s$，策略网络 $\pi_\theta$ 采样出 $G$ 个输出（通常 $G=4 \sim 8$）：
$$
o_1, o_2, \dots, o_G \sim \pi_\theta(\cdot|s)
$$

奖励模型对每个输出打分，得到奖励集合 $R = \{r_1, r_2, \dots, r_G\}$。

我们计算这组奖励的均值 $\mu_R$ 和标准差 $\sigma_R$：
$$
\mu_R = \frac{1}{G} \sum_{i=1}^G r_i, \quad \sigma_R = \sqrt{\frac{1}{G} \sum_{i=1}^G (r_i - \mu_R)^2}
$$

对于第 $i$ 个输出 $o_i$，其**相对优势**估计为：
$$
\hat{A}_i = \frac{r_i - \mu_R}{\sigma_R + \epsilon}
$$
其中 $\epsilon$ 是一个极小的常数，防止除以零。

回到数学题的例子：$\mu_R = 0.5$, $\sigma_R = 0.5$, 所以 $\hat{A}_1 = \hat{A}_2 = +1$, $\hat{A}_3 = \hat{A}_4 = -1$。正确答案获得正优势，错误答案获得负优势——无需 Critic 网络，纯粹靠组内对比。

## 2. KL 散度正则化

为了防止模型在追求高奖励的过程中"钻空子"（Reward Hacking）或丧失语言的连贯性，我们需要约束更新后的策略 $\pi_\theta$ 不要偏离初始参考策略 $\pi_{\text{ref}}$ 太远。

在 GRPO 中，KL 散度惩罚被直接集成到损失函数中，采用了一种无偏的估计量：
$$
\mathbb{D}_{\text{KL}}(\pi_\theta || \pi_{\text{ref}}) = \frac{\pi_{\text{ref}}(o_i|s)}{\pi_\theta(o_i|s)} - \log \frac{\pi_{\text{ref}}(o_i|s)}{\pi_\theta(o_i|s)} - 1
$$

## 3. GRPO 最终目标函数

结合 PPO 的裁剪机制和组内相对优势，GRPO 的最终目标函数（需要最大化）定义为：

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{s \sim P(S), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^G \left( \min \left( \rho_i(\theta) \hat{A}_i, \text{clip}(\rho_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta || \pi_{\text{ref}}) \right) \right]
$$

其中：
- $\rho_i(\theta) = \frac{\pi_\theta(o_i|s)}{\pi_{\theta_{\text{old}}}(o_i|s)}$ 是重要性采样比率（同 PPO）。
- $\epsilon$ 是裁剪阈值（如 0.2）。
- $\beta$ 是 KL 惩罚系数。

回到数学题的例子，这个公式做了两件事：
1. 用 PPO 的裁剪机制，限制正确答案 $o_1, o_2$ 的概率不会一次涨太多（$\hat{A}_i = +1$），错误答案 $o_3, o_4$ 的概率不会一次降太多（$\hat{A}_i = -1$）。
2. 用 KL 散度约束模型不要为了做对数学题而丧失自然语言能力。

---

# GRPO 与 PPO 的对比总结

| 特性 | PPO | GRPO / RLOO |
| :--- | :--- | :--- |
| **基线 (Baseline) 估计** | 依赖独立的 Critic 网络预测绝对价值 $V(s)$ | 依赖同一 Prompt 下 $G$ 个采样的经验均值 |
| **显存开销** | 极高（需要加载 Actor 和 Critic 两个大模型） | 显著降低（彻底抛弃 Critic 网络） |
| **计算开销** | 较低（每个 Prompt 采样 1 次即可更新） | 较高（每个 Prompt 需要采样 $G$ 次） |
| **优势估计方差** | 较低（Critic 经过充分训练后预测稳定） | 依赖于组大小 $G$，$G$ 越小方差越大，但无偏差 |
| **核心优势** | 经典、稳定，适用于所有 RL 任务 | 完美契合大模型生成任务，实现高效在线 RL |

**用数学题的例子理解核心区别**：PPO 需要一个 Critic 网络来回答"这道积分题的平均得分大概是多少"——这需要额外的显存和训练。GRPO 则说"不用猜了，我直接让模型做 4 遍，用这 4 次的实际得分算平均就行"——用计算（多次采样）换显存（去掉 Critic），这对生成式大模型来说是一笔极其划算的交易。

**开源代码参考：**
GRPO 随着 DeepSeek 的开源而爆火，目前 Hugging Face 的 **TRL** 库已经快速跟进并提供了 `trl.GRPOTrainer`。其核心的优势计算与损失函数的 PyTorch 伪代码如下：

```python
# rewards: 形状为 [batch_size, G] 的奖励张量
# log_probs, old_log_probs, ref_log_probs: 形状均为 [batch_size, G] 的对数概率

# 1. 计算组内相对优势
mean_rewards = rewards.mean(dim=1, keepdim=True)
std_rewards = rewards.std(dim=1, keepdim=True)
advantages = (rewards - mean_rewards) / (std_rewards + 1e-8)

# 2. 计算重要性采样比率
ratio = torch.exp(log_probs - old_log_probs)

# 3. 计算 PPO 风格的裁剪损失
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
policy_loss = -torch.min(surr1, surr2)

# 4. 计算 KL 散度惩罚
kl = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1.0

# 5. 总损失
loss = (policy_loss + beta * kl).mean()
```

GRPO 优雅地去除了 Critic 网络，用组内相对优势实现了高效的对比学习。它证明了在生成式大模型时代，简单的经验统计往往比复杂的神经网络预测更加鲁棒和高效。

> 下一篇：[笔记｜生成模型（二十）：Flow-GRPO 与图像生成应用（基于 Flux 的代码解析）](/chengYi-xun/posts/21-flow-grpo/)
