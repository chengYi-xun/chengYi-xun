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

GRPO 的思路极简：**对同一个 Prompt 采样 $G$ 个回答，用组内奖励的均值和标准差做标准化，得到每个回答的相对优势——高于均值的强化，低于均值的抑制。**

$$
\hat{A}_i = \frac{r_i - \mu_R}{\sigma_R}, \quad \mu_R = \frac{1}{G}\sum_{j=1}^G r_j, \quad \sigma_R = \text{std}(r_1, \dots, r_G)
$$

这就是"矮子里拔高个"：即使绝对水平不高，只要能分出高低，模型就有学习信号。而当所有回答奖励相同时（$\sigma_R = 0$），优势为零，模型不更新——避免了无区分信号时的噪声梯度。

---

# GRPO 的理论根源：从 REINFORCE 到组内相对优势

在深入数学推导之前，先理清 GRPO 的理论脉络——它并不是凭空发明的，而是 **REINFORCE with Baseline** 的一个聪明的工程变体。

回顾第一篇（RL 基础）中的 REINFORCE with Baseline：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t)) \right]
$$

其中基线 $b(s_t)$ 可以是任何不依赖于动作 $a_t$ 的量——经典选择是**学习一个价值网络** $V_\phi(s_t)$，这就是 Actor-Critic / PPO 的路线（需要 Critic 模型，显存翻倍）。

**GRPO 的关键洞察**：在语言模型场景中，基线有一个更自然的选择——**同一个 Prompt 下多个采样回答的经验均值**。这个选择的理论合理性在于：

- 数学上，$b(s) = \mathbb{E}_{o \sim \pi_\theta(\cdot|s)}[r(o)]$ 是最优基线（使策略梯度方差最小），而组内均值 $\mu_R = \frac{1}{G}\sum_i r_i$ 正是这个期望的**无偏蒙特卡洛估计**。

- 当 $G \to \infty$ 时，由大数定律 $\mu_R \to \mathbb{E}[r]$，组内相对优势 $\hat{A}_i = \frac{r_i - \mu_R}{\sigma_R}$ 收敛到标准化的优势函数。

- 标准化（除以 $\sigma_R$）确保优势的尺度不依赖于奖励的绝对值——无论奖励是 $[0, 1]$ 还是 $[-100, 100]$，优势都在 $[-3, 3]$ 左右，梯度尺度稳定。

**与 RLOO 的关系**：RLOO（REINFORCE Leave-One-Out）是另一种去 Critic 的基线选择，它用"除了第 $i$ 个之外其余回答的均值"作为第 $i$ 个回答的基线：$b_i = \frac{1}{G-1}\sum_{j \neq i} r_j$。RLOO 的优势估计方差理论上更低（因为基线与 $r_i$ 独立），但实践中 GRPO 的组内标准化效果同样优秀，且实现更简单。

---

# GRPO 的数学推导与损失函数构建

## 1. 组内相对优势计算

给定一个输入 Prompt $s$，策略网络 $\pi_\theta$ 采样出 $G$ 个输出（通常 $G=4 \sim 16$）：
$$
o_1, o_2, \dots, o_G \sim \pi_\theta(\cdot|s)
$$

奖励模型（或规则判题器）对每个输出打分，得到奖励集合 $R = \{r_1, r_2, \dots, r_G\}$。

计算组内均值和标准差：
$$
\mu_R = \frac{1}{G} \sum_{i=1}^G r_i, \quad \sigma_R = \sqrt{\frac{1}{G} \sum_{i=1}^G (r_i - \mu_R)^2}
$$

对于第 $i$ 个输出 $o_i$，其**相对优势**估计为：
$$
\hat{A}_i = \frac{r_i - \mu_R}{\sigma_R + \epsilon}
$$
其中 $\epsilon$ 是极小常数，防止除以零（当所有回答奖励相同时 $\sigma_R = 0$）。

**极端情况分析**：

- 全对 $r = [1,1,1,1]$：$\sigma_R = 0$，$\hat{A}_i = 0$ → 不更新（都对了，没什么可学的）。

- 全错 $r = [0,0,0,0]$：$\sigma_R = 0$，$\hat{A}_i = 0$ → 不更新（都错了，没有正样本可以学习）。

- 一对三错 $r = [1,0,0,0]$：$\hat{A}_1 = +1.73$，$\hat{A}_{2,3,4} = -0.58$ → 大力强化唯一的正确回答。

这种"全对/全错时不更新"的行为避免了在没有区分信号时引入噪声梯度。

## 2. KL 散度正则化：为什么用这个特殊形式？

为了防止策略"钻空子"（Reward Hacking）或丧失语言连贯性，需要约束 $\pi_\theta$ 不偏离参考策略 $\pi_{\text{ref}}$ 太远。

标准的 KL 散度定义为：

$$
D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_{o \sim \pi_\theta}\left[\log \frac{\pi_\theta(o|s)}{\pi_{\text{ref}}(o|s)}\right]
$$

它需要对 $\pi_\theta$ 的完整分布求期望，但我们手头只有来自 $\pi_{\theta_{\text{old}}}$ 的有限采样。直接用采样估计 $D_{\text{KL}}$ 会引入较大方差。GRPO 转而使用一种基于 **$f$-散度** 的替代量。

**推导**：$f$-散度的一般形式为 $D_f(P \| Q) = \mathbb{E}_{x \sim Q}[f(P(x)/Q(x))]$，其中 $f$ 是满足 $f(1) = 0$ 的凸函数。取 $f(u) = u - \log u - 1$（对应**反向 KL 散度**），令 $P = \pi_{\text{ref}}$、$Q = \pi_\theta$、$u = \frac{\pi_{\text{ref}}(o_i|s)}{\pi_\theta(o_i|s)}$，则单样本估计量为：

$$
\hat{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \underbrace{\frac{\pi_{\text{ref}}(o_i|s)}{\pi_\theta(o_i|s)}}_{u} - \underbrace{\log \frac{\pi_{\text{ref}}(o_i|s)}{\pi_\theta(o_i|s)}}_{\log u} - 1
$$

**为什么选 $f(u) = u - \log u - 1$？**

- $f(1) = 0$：当 $\pi_\theta = \pi_{\text{ref}}$ 时惩罚为零。

- $f''(u) = 1/u^2 > 0$：严格凸，$u = 1$ 是唯一最小值点。

- **双侧惩罚**：当 $\pi_\theta \ll \pi_{\text{ref}}$（$u \gg 1$）时 $f(u) \approx u$（线性增长），当 $\pi_\theta \gg \pi_{\text{ref}}$（$u \to 0$）时 $f(u) \approx -\log u$（对数增长）。两个方向的偏离都会被惩罚，防止概率塌缩或异常膨胀。

> **与 PPO 的 KL 惩罚对比**：PPO 使用 $\beta \cdot (\log \pi_\theta - \log \pi_{\text{ref}})$ 逐 token 加到奖励上（见上一篇的 `kl_penalty`）。GRPO 将 KL 惩罚直接作为损失的一部分，并使用 $f$-散度形式，对概率塌缩更敏感。

### Token 级别的操作

在语言模型中，一个回答 $o_i$ 是一个 token 序列 $o_i = (o_i^1, o_i^2, \dots, o_i^T)$。**重要性比率和 KL 散度都是在 token 级别计算后求和的**：

$$
\log \pi_\theta(o_i|s) = \sum_{t=1}^{T} \log \pi_\theta(o_i^t | s, o_i^{<t})
$$

$$
\rho_i(\theta) = \exp\left(\sum_{t=1}^{T} \left[\log \pi_\theta(o_i^t | s, o_i^{<t}) - \log \pi_{\theta_{\text{old}}}(o_i^t | s, o_i^{<t})\right]\right)
$$

实际实现中，通常对每个 token 分别计算比率后取平均（而非序列级乘积），以避免长序列导致比率指数级爆炸或塌缩。

## 3. GRPO 最终目标函数

结合 PPO 的裁剪机制和组内相对优势，GRPO 的最终目标函数（需要最大化）定义为：

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{s \sim P(S), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^G \left( \min \left( \rho_i(\theta) \hat{A}_i, \text{clip}(\rho_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta \hat{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right) \right]
$$

其中：

- $\rho_i(\theta) = \frac{\pi_\theta(o_i|s)}{\pi_{\theta_{\text{old}}}(o_i|s)}$ 是重要性采样比率（同 PPO）。

- $\epsilon$ 是裁剪阈值（如 0.2），防止单步更新过大。

- $\beta$ 是 KL 惩罚系数，控制偏离参考策略的代价。

- $\hat{A}_i$ 是组内归一化优势，$\hat{D}_{\text{KL}}$ 是 $f$-散度形式的 KL 估计。


---

# GRPO 的完整实现

以下是 GRPO 的完整 PyTorch 实现伪代码，包括数据准备、模型定义、采样、优势计算和训练循环。

**Step 1: 数据格式 — Prompt 数据集 + 奖励函数**

与 DPO 不同，GRPO 是**在线**算法：不需要预先收集偏好对，只需要 Prompt 和一个能打分的奖励函数。

```python
"""
GRPO 数据格式：只需要 Prompt 集合 + 奖励函数
- Prompt 集合: 来自训练数据（如数学题、编程题、对话 Prompt）
- 奖励函数: 可以是规则判题器（数学题判对错）、奖励模型、或两者结合

与 DPO 的本质区别:
  DPO: 离线 — 需要预先标注好的 (prompt, chosen, rejected) 三元组
  GRPO: 在线 — 只需要 prompt，模型自己生成 + 自己评判
"""
# 只有 Prompt, 无需预先标注偏好对
prompts = load_dataset("math_problems")

def reward_fn(prompt, response):
    """奖励函数: 规则匹配 / 奖励模型 / 混合"""
    return 1.0 if verify_math_answer(prompt, response) else 0.0
```

**Step 2: 模型定义 — 只需要两个模型！**

GRPO 的一大优势：与 DPO 一样只需要两个模型，但保留了在线 RL 的探索能力。

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型 1: 待训练的策略模型 (π_θ)
actor = AutoModelForCausalLM.from_pretrained("sft_checkpoint")
# 模型 2: 冻结的参考模型 (π_ref), KL 正则锚点
ref_model = AutoModelForCausalLM.from_pretrained("sft_checkpoint")
ref_model.requires_grad_(False)

# 对比: PPO 需要 4 个模型, DPO/GRPO 只需 2 个

tokenizer = AutoTokenizer.from_pretrained("sft_checkpoint")
optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-6)

# 超参数
G = 8             # 每个 Prompt 采样的回答数量
clip_range = 0.2  # PPO 裁剪阈值 ε
beta = 0.04       # KL 惩罚系数
K_epochs = 2      # 每批数据的更新轮数
```

**Step 3: 在线采样 + 奖励收集**

这是 GRPO 与 DPO 的核心区别——GRPO 用当前策略**在线生成**回答并即时评分：

```python
def collect_group_rollouts(actor, prompts_batch, G, reward_fn):
    """
    对每个 Prompt 采样 G 个回答并打分

    Returns:
        prompt_ids:    List[(1, L_p)]     每个 prompt 的 token id
        response_ids:  List[(1, L_r)]     每个回答的 token id
        rewards:       List[float]        标量奖励 r_i
        old_log_probs: List[(L_r,)]       采样时的 token 级 log π_old
        ref_log_probs: List[(L_r,)]       参考策略的 token 级 log π_ref
    """
    all_prompt_ids, all_response_ids, all_rewards = [], [], []
    all_old_log_probs, all_ref_log_probs = [], []

    actor.eval()
    with torch.no_grad():
        for prompt in prompts_batch:
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
            # prompt_ids: (1, L_p)

            # 同一 prompt 下独立采样 G 条回答
            for _ in range(G):
                # 采样一个回答 o_i ~ π_old(·|s)
                response_ids = actor.generate(
                    prompt_ids,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7
                )
                # response_ids: (1, L_p + L_r)

                response_text = tokenizer.decode(response_ids[0])

                # 奖励函数打分
                reward = reward_fn(prompt, response_text)  # scalar

                # 计算 token 级对数概率
                old_logps = compute_token_log_probs(actor, prompt_ids, response_ids)
                # old_logps: (L_r,)
                ref_logps = compute_token_log_probs(ref_model, prompt_ids, response_ids)
                # ref_logps: (L_r,)

                all_prompt_ids.append(prompt_ids)
                all_response_ids.append(response_ids)
                all_rewards.append(reward)
                all_old_log_probs.append(old_logps)
                all_ref_log_probs.append(ref_logps)

    actor.train()
    return all_prompt_ids, all_response_ids, all_rewards, all_old_log_probs, all_ref_log_probs
```

**Step 4: 组内相对优势计算**

```python
def compute_group_advantages(rewards, G):
    """
    组内相对优势计算

    Args:
        rewards: List[float], 长度 batch_size * G
        G:       int, 每个 prompt 的采样数
    Returns:
        (batch_size * G,)  标准化后的相对优势
    """
    # 重塑为 (batch_size, G), 每行是同一 prompt 下的 G 个奖励
    rewards = torch.tensor(rewards).reshape(-1, G)  # (batch_size, G)
    # 组内均值作基线 (替代 Critic)
    mean_r = rewards.mean(dim=1, keepdim=True)      # (batch_size, 1)
    # 组内标准差, 稳定梯度尺度
    std_r = rewards.std(dim=1, keepdim=True)         # (batch_size, 1)
    # 标准化优势; 全同奖励时 ε 避免除零, 优势为 0
    advantages = (rewards - mean_r) / (std_r + 1e-8) # (batch_size, G)
    return advantages.reshape(-1)                     # (batch_size * G,)
```

**Step 5: 完整训练循环**

```python
for step in range(total_steps):
    # ---- 阶段 1: 在线采样 (GRPO 独有, DPO 没有这一步) ----
    prompts_batch = sample_prompts(prompts, batch_size=8)
    prompt_ids, response_ids, rewards, old_log_probs, ref_log_probs = \
        collect_group_rollouts(actor, prompts_batch, G, reward_fn)
    # rewards: List[float], 长度 8*G

    # ---- 阶段 2: 计算组内相对优势 ----
    advantages = compute_group_advantages(rewards, G)
    # advantages: (8*G,)

    # ---- 阶段 3: 多 epoch 更新 (重要性采样允许复用数据) ----
    for epoch in range(K_epochs):
        for idx in minibatch_indices(len(response_ids), batch_size=16):
            # 当前 π_θ 重新计算 token 级对数概率
            new_log_probs = compute_token_log_probs(
                actor, prompt_ids[idx], response_ids[idx]
            )
            # new_log_probs: (minibatch, L_r)

            # 重要性采样比率 ρ = π_θ/π_old
            log_ratio = new_log_probs - old_log_probs[idx]  # (minibatch,)
            ratio = torch.exp(log_ratio)                     # (minibatch,)

            # PPO 裁剪目标
            adv = advantages[idx]                             # (minibatch,)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
            policy_loss = -torch.min(surr1, surr2).mean()    # scalar

            # KL 散度惩罚 (f-散度: u - log u - 1)
            log_ref_ratio = ref_log_probs[idx] - new_log_probs  # log(π_ref/π_θ)
            kl_penalty = (
                torch.exp(log_ref_ratio) - log_ref_ratio - 1.0
            ).mean()  # scalar

            # 总损失
            loss = policy_loss + beta * kl_penalty  # scalar

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            optimizer.step()
```

> **与 DPO 训练循环的关键对比**：
> - DPO 只有"前向传播 + 损失计算 + 反向传播"，与标准 SFT 训练几乎一样。
> - GRPO 多了"在线采样"阶段（调用 `actor.generate()`），这是计算开销的主要来源，但也是在线 RL 探索能力的来源。
> - DPO 的 batch 是固定的偏好对；GRPO 的 batch 是模型自己实时生成的，每步训练都能看到新的探索结果。

---

# GRPO 与 PPO / DPO 的全景对比

| 维度 | PPO (RLHF) | DPO | GRPO |
| :---: | :---: | :---: | :---: |
| **模型数量** | 4 (Actor+Critic+Ref+RM) | 2 (Actor+Ref) | 2 (Actor+Ref) + 外部奖励函数 |
| **训练方式** | 在线 RL | 离线监督学习 | 在线 RL |
| **基线估计** | Critic 网络 $V_\phi(s)$ | 无需基线 | 组内经验均值 $\mu_R$ |
| **显存开销** | 极高 (4 个大模型) | 低 (2 个大模型) | 低 (2 个大模型) |
| **计算开销** | 中等 (每 Prompt 采样 1 次) | 最低 (纯前向传播) | 较高 (每 Prompt 采样 G 次) |
| **探索能力** | 强 | 弱 (离线数据) | 强 |
| **核心优势** | 经典稳定 | 极简高效 | 省显存 + 在线探索 |


**开源代码参考：** GRPO 随 DeepSeek 开源而爆火，Hugging Face **TRL** 库 ([`trl.GRPOTrainer`](https://huggingface.co/docs/trl/grpo_trainer)) 提供了生产级实现。

GRPO 证明了在生成式大模型时代，简单的经验统计（组内均值）往往比复杂的神经网络预测（Critic）更加鲁棒和高效。

> 下一篇：[笔记｜生成模型（二十）：Flow-GRPO 与图像生成应用（基于 Flux 的代码解析）](/chengYi-xun/posts/21-flow-grpo/)
