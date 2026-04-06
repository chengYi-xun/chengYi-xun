---
title: 笔记｜生成模型（十八）：大模型对齐的另一条路：DPO (Direct Preference Optimization)
date: 2025-08-18 10:00:00
cover: false
mathjax: true
categories:
 - Notes
tags:
 - Deep learning
 - Generative models theory
 - Reinforcement Learning
 - DPO
series: Diffusion Models theory
---

> 本文为系列第三篇。在上一篇中，我们提到 PPO 算法虽然稳定，但在百亿参数大模型微调时面临着极大的显存压力（需要同时维护 Actor 和 Critic 模型）。为了解决这一痛点，斯坦福大学在 2023 年提出了一条完全不同于在线 RL 的路线——DPO。本文将简要介绍 DPO 算法，作为后续回归 RL 路线（GRPO）的对比铺垫。

# PPO 的繁琐与显存危机：大模型吃不消了

**先看例子**：假设我们要用 RLHF 微调一个大模型，让它学会写出更好的代码。传统流程分三步：

1. **SFT**：用大量代码问答数据做监督微调——教模型"怎么写代码"。
2. **RM**：给同一道编程题生成两份代码（A 和 B），让人类标注哪份更好，训练一个"代码评审员"（奖励模型）。
3. **RL**：让模型自己去写代码，"评审员"给分，模型根据分数用 PPO 算法调整自己。

这个流程极其繁琐，且在 PPO 阶段，显存中需要同时驻留**四个**庞大的模型：

- **Actor 模型**（正在训练的策略网络，即写代码的学生）
- **Critic 模型**（价值网络，通常与 Actor 同等规模，估计代码题的"难度"）
- **Reference 模型**（冻结的 SFT 模型，防止学生学偏）
- **Reward 模型**（冻结的奖励模型，即代码评审员）

四个大模型同台竞技，显存开销令人绝望。对于百亿参数（10B+）的模型，普通实验室根本玩不起。

---

# DPO：绕过奖励模型与 RL

**核心思考出发点**：既然我们的最终目的是让模型符合人类偏好（生成好答案的概率大于坏答案），我们为什么非要绕个大弯子，先训练一个"评审员"，再用 PPO 去教学生呢？能不能直接把"人类偏好"喂给学生，让他直接学？

**用一个具体例子理解 DPO 的想法**：

假设用户问："用 Python 写一个排序函数。" 模型生成了两个回答：

| | 回答 $y_w$（胜者） | 回答 $y_l$（败者） |
|:---:|:---|:---|
| **代码** | `def sort(arr): return sorted(arr)` | `def sort(arr): arr.sort(); return arr` |
| **人类评价** | 更好（纯函数，无副作用） | 较差（修改了原数组） |

在 PPO 流程中，你需要先训练一个奖励模型来给两个回答打分（比如 $r(y_w) = 0.8$, $r(y_l) = 0.3$），然后再用 PPO 去优化策略。

**DPO 的做法**：跳过奖励模型，直接告诉语言模型——"$y_w$ 比 $y_l$ 好，请调整你的参数，让 $y_w$ 的生成概率相对增大，$y_l$ 的相对减小。" 整个过程变成了一个简单的监督学习问题。

**意义**：DPO 将复杂的强化学习问题，巧妙地转化为了一个**监督学习分类问题**——不需要奖励模型，不需要 Critic，不需要 PPO 采样。只需要 Actor 模型和一个冻结的 Reference 模型即可。

## DPO 的数学推导

**先看例子**：在推导之前，先用直觉理解 DPO 损失函数的含义。

对于上面的排序函数例子，DPO 计算的核心信号是：

$$\text{信号} = \beta \left( \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} \right)$$

- $\log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)}$：当前模型相比参考模型，有多"偏爱"好答案 $y_w$。
- $\log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}$：当前模型相比参考模型，有多"偏爱"坏答案 $y_l$。
- 两者之差越大，说明模型越能区分好坏——DPO 的目标就是**最大化这个差值**。

假设训练前，当前模型就是参考模型（$\pi_\theta = \pi_\text{ref}$），那么两项都是 0，信号也是 0——模型还没学会区分好坏。训练之后，模型应该增大 $\pi_\theta(y_w|x)$、减小 $\pi_\theta(y_l|x)$，使信号变大。

**一般化的数学推导：**

在传统的 RLHF 中，我们的目标是最大化奖励，同时约束 KL 散度：
$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ r(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right]
$$

DPO 论文通过严谨的数学推导证明，上述 RL 优化问题的**最优策略 $\pi^*$ 有一个闭式解**：
$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x, y) \right)
$$
其中 $Z(x)$ 是配分函数。

通过对上式进行代数变换，我们可以**用策略的概率反向表示出奖励函数**：
$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

现在，我们考虑人类偏好数据。根据 Bradley-Terry (BT) 模型，人类偏好 $y_w$ 胜过 $y_l$ 的概率可以表示为：
$$
p(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))
$$
其中 $\sigma$ 是 Sigmoid 函数。

**关键一步**：将用策略表示的奖励代入 BT 模型中：
$$
\begin{aligned}
r(x, y_w) - r(x, y_l) &= \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} + \beta \log Z(x) \right) - \left( \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} + \beta \log Z(x) \right) \\
&= \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
\end{aligned}
$$
难以计算的配分函数 $\beta \log Z(x)$ 在相减时被完全抵消了！

于是，我们得到了 **DPO 的损失函数**（负对数似然）：
$$
\mathcal{L}_{\text{DPO}}(\theta) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

回到排序函数的例子，这个损失函数的含义就很清晰了：它通过 Sigmoid + 负对数的方式，推动模型让好答案（纯函数版排序）的"相对偏好度"大于坏答案（修改原数组版排序）。

**开源代码参考：**
DPO 的实现非常简单，目前最主流的开源实现来自 Hugging Face 的 **TRL** 库（`trl.DPOTrainer`）。其核心损失计算的 PyTorch 伪代码如下：

```python
# policy_chosen_logps: 当前模型对 y_w 的对数概率
# policy_rejected_logps: 当前模型对 y_l 的对数概率
# ref_chosen_logps: 参考模型对 y_w 的对数概率
# ref_rejected_logps: 参考模型对 y_l 的对数概率

# 1. 计算当前模型和参考模型的对数概率差
policy_logratios = policy_chosen_logps - policy_rejected_logps
ref_logratios = ref_chosen_logps - ref_rejected_logps

# 2. 计算 logits (乘以 beta)
logits = policy_logratios - ref_logratios

# 3. 计算负对数似然损失 (使用 log_sigmoid)
loss = -F.logsigmoid(beta * logits).mean()
```

## DPO 的优势与局限

**优势：**
1. **极其简单**：不需要训练奖励模型，不需要 Critic 网络，不需要 PPO 采样。
2. **显存友好**：只需要加载 Actor 模型和冻结的 Reference 模型。
3. **稳定性高**：本质上变成了一个监督学习（分类）问题，避免了 RL 的不稳定性。

**局限：**
1. **离线学习 (Offline)**：DPO 依赖于预先收集好的静态偏好数据集。如果模型在训练过程中产生了新的（分布外）输出，DPO 无法对其进行实时评估和纠正。
2. **上限较低**：在线 RL（如 PPO）允许模型在探索中发现比人类标注更好的答案（例如 AlphaGo 发现新定式，DeepSeek-R1 涌现出长思维链顿悟）。而 DPO 只能模仿数据集中已有的偏好，难以产生真正的"涌现"和"超越"。

**用例子理解这个局限**：假设一道很难的数学推理题，人类标注员自己都做不出来，无法提供正确的偏好标注。DPO 就束手无策了。而在线 RL 可以让模型自己反复尝试，一旦碰巧写出正确答案，就立刻强化它——这就是 DeepSeek-R1 中"顿悟时刻"的由来。

因此，虽然 DPO 在开源社区大火，但在追求极致推理能力的最前沿大模型中，**在线强化学习（Online RL）仍然是不可替代的王者**。

那么，如何解决在线 RL（PPO）的显存危机呢？这就引出了我们下一篇的主角：GRPO。

> 下一篇：[笔记｜生成模型（十九）：大模型在线 RL 破局者：GRPO 算法详解](/chengYi-xun/posts/20-grpo/)
