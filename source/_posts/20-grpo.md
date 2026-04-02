---
title: 笔记｜生成模型（十九）：大模型在线 RL 破局者：GRPO 算法详解
date: 2026-04-03 10:15:00
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

# 在线 RL 的不可替代性与 Critic 的累赘（为什么 DPO 不够用？）

正如上一篇所言，DPO 虽然简单省显存，但它只能“死记硬背”人类给出的标准答案（离线学习）。如果遇到极其复杂的数学推理或逻辑编程任务，**人类自己都不知道最优解是什么**，更别提给模型准备成千上万的偏好数据了。

为了让模型产生“顿悟”（Aha moment）和自我进化，我们必须回归**在线强化学习（Online RL）**。模型需要像做实验一样，不断生成新的答案，由基于规则的验证器（比如代码能不能跑通、数学答案对不对）给出实时反馈。

然而，PPO 算法中的 Critic 网络（价值网络）成为了最大的绊脚石。Critic 的作用是预测“这道题有多难”（状态价值 $V(s)$），以此作为基准线。对于百亿参数的大模型，多维护一个 Critic 意味着显存开销直接翻倍。

**核心思考出发点**：既然 Critic 只是为了给出一个“及格线”，我们能不能**彻底去掉 Critic 模型**，用一种更简单、更直观的方法来估计这个“及格线”，同时又能准确地判断一个回答是好是坏？

这就是 **GRPO (Group Relative Policy Optimization)** 算法诞生的初衷。*(注：与 GRPO 同期，Cohere 团队提出的 RLOO 算法也采用了几乎相同的思想，这也印证了该方向的正确性。)*

---

# GRPO 的核心思想：矮子里拔高个（组内相对评分）

GRPO 由 DeepSeek 团队在 2024 年提出，并在 DeepSeekMath 和 DeepSeek-R1 中大放异彩。它的核心思想非常直观：**通过组内相对评分（Group Relative Scoring）来替代 Critic 网络的绝对价值估计。**

在生成任务（如写代码、解数学题、画图）中，**“好”与“坏”往往是相对的**。
- 想象一场极难的数学考试，全班考得都很差，平均分只有 10 分。Critic 网络就像一个死板的老师，非要预测每个人的绝对分数。由于分数极低，老师很难准确预测。
- 但在 GRPO 中，我们不预测绝对分数。对于同一道题（Prompt）$s$，我们让模型（学生）**尝试用不同的思路解答多次（Group）**，得到一组答案 $\{o_1, o_2, \dots, o_G\}$。

然后，我们用判题系统给这 $G$ 个答案分别打分，得到一组奖励值 $\{r_1, r_2, \dots, r_G\}$。

**意义**：既然我们有了同一道题的多个真实得分，我们就可以直接用这组分数的**平均分**作为及格线 $V(s)$，并用标准差进行归一化。这就是所谓的“矮子里拔高个”——即使大家都没做对，只要你比平均分高一点点（比如推导多对了一步），你就是“好”的，模型就会向你学习。这彻底抛弃了笨重的 Critic 网络，实现了极其高效的在线对比学习。

---

# GRPO 的数学推导与损失函数构建

让我们用严谨的数学语言来描述这一过程。

## 1. 组内相对优势计算

给定一个输入状态 $s$，策略网络 $\pi_\theta$ 采样出 $G$ 个输出（通常 $G=4 \sim 8$）：
$$
o_1, o_2, \dots, o_G \sim \pi_\theta(\cdot|s)
$$

奖励模型对每个输出打分，得到奖励集合：
$$
R = \{r_1, r_2, \dots, r_G\}
$$

我们计算这组奖励的均值 $\mu_R$ 和标准差 $\sigma_R$：
$$
\mu_R = \frac{1}{G} \sum_{i=1}^G r_i, \quad \sigma_R = \sqrt{\frac{1}{G} \sum_{i=1}^G (r_i - \mu_R)^2}
$$

对于第 $i$ 个输出 $o_i$，其**相对优势（Advantage）**估计为：
$$
\hat{A}_i = \frac{r_i - \mu_R}{\sigma_R + \epsilon}
$$
其中 $\epsilon$ 是一个极小的常数，防止除以零。

**物理意义**：
- 如果 $\hat{A}_i > 0$，说明输出 $o_i$ 的奖励高于这一组的平均水平，是一个“好”的输出，梯度会推动模型多生成类似的输出。
- 如果 $\hat{A}_i < 0$，说明输出 $o_i$ 低于平均水平，是一个“差”的输出，梯度会抑制它。
- 即使所有回答都很差，只要其中有一个稍微好一点（例如推导多对了一步），它的相对优势就会是正的，模型就能从这个“相对较好”的样本中学习。

## 2. KL 散度正则化

在 RLHF 中，为了防止模型在追求高奖励的过程中“钻空子”（Reward Hacking）或丧失语言的连贯性，我们需要约束更新后的策略 $\pi_\theta$ 不要偏离初始参考策略 $\pi_{\text{ref}}$ 太远。

PPO 通常将 KL 散度作为惩罚项直接加到奖励函数中。而在 GRPO 中，KL 散度惩罚被直接集成到了损失函数中，采用了一种无偏的估计量：
$$
\mathbb{D}_{\text{KL}}(\pi_\theta || \pi_{\text{ref}}) = \frac{\pi_{\text{ref}}(o_i|s)}{\pi_\theta(o_i|s)} - \log \frac{\pi_{\text{ref}}(o_i|s)}{\pi_\theta(o_i|s)} - 1
$$

## 3. GRPO 最终目标函数

结合 PPO 的裁剪机制（Clipping）和组内相对优势，GRPO 的最终目标函数（需要最大化）定义为：

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{s \sim P(S), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^G \left( \min \left( \rho_i(\theta) \hat{A}_i, \text{clip}(\rho_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta || \pi_{\text{ref}}) \right) \right]
$$

其中：
- $\rho_i(\theta) = \frac{\pi_\theta(o_i|s)}{\pi_{\theta_{\text{old}}}(o_i|s)}$ 是重要性采样比率。
- $\epsilon$ 是裁剪阈值（如 0.2）。
- $\beta$ 是 KL 惩罚系数。

---

# GRPO 与 PPO 的对比总结

| 特性 | PPO | GRPO / RLOO |
| :--- | :--- | :--- |
| **基线 (Baseline) 估计** | 依赖独立的 Critic 网络预测绝对价值 $V(s)$ | 依赖同一 Prompt 下 $G$ 个采样的经验均值 |
| **显存开销** | 极高（需要加载 Actor 和 Critic 两个大模型） | 显著降低（彻底抛弃 Critic 网络） |
| **计算开销** | 较低（每个 Prompt 采样 1 次即可更新） | 较高（每个 Prompt 需要采样 $G$ 次） |
| **优势估计方差** | 较低（Critic 经过充分训练后预测稳定） | 依赖于组大小 $G$，$G$ 越小方差越大，但无偏差 |
| **核心优势** | 经典、稳定，适用于所有 RL 任务 | 完美契合大模型生成任务，实现高效在线 RL |

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

> 下一篇：[笔记｜生成模型（二十）：Flow-GRPO 与图像生成应用（基于 Flux 的代码解析）](/chengYi-xun/2026/04/03/21-flow-grpo/)
