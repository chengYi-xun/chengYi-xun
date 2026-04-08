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

# GRPO 的理论根源：从 REINFORCE 到组内相对优势

在深入数学推导之前，先理清 GRPO 的理论脉络——它并不是凭空发明的，而是 **REINFORCE with Baseline** 的一个聪明的工程变体。

回顾第一篇（RL 基础）中的 REINFORCE with Baseline：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t)) \right]
$$

其中基线 $b(s_t)$ 可以是任何不依赖于动作 $a_t$ 的量——经典选择是**学习一个价值网络** $V_\phi(s_t)$，这就是 Actor-Critic / PPO 的路线（需要 Critic 模型，显存翻倍）。

**GRPO 的关键洞察**：在语言模型场景中，基线有一个更自然的选择——**同一个 Prompt 下多个采样回答的经验均值**。这个选择的理论合理性在于：

- 数学上，$b(s) = \mathbb{E}_{o \sim \pi_\theta(\cdot|s)}[r(o)]$ 是最优基线（方差最小），而组内均值 $\mu_R = \frac{1}{G}\sum_i r_i$ 正是这个期望的无偏蒙特卡洛估计。
- 当 $G \to \infty$ 时，$\mu_R \to \mathbb{E}[r]$，组内相对优势 $\hat{A}_i = \frac{r_i - \mu_R}{\sigma_R}$ 收敛到标准化的优势函数。
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

回到数学题的例子：$\mu_R = 0.5$, $\sigma_R = 0.5$, 所以 $\hat{A}_1 = \hat{A}_2 = +1$, $\hat{A}_3 = \hat{A}_4 = -1$。正确答案获得正优势，错误答案获得负优势——无需 Critic 网络，纯粹靠组内对比。

**极端情况分析**：
- 全对 $r = [1,1,1,1]$：$\sigma_R = 0$，$\hat{A}_i = 0$ → 不更新（都对了，没什么可学的）。
- 全错 $r = [0,0,0,0]$：$\sigma_R = 0$，$\hat{A}_i = 0$ → 不更新（都错了，没有正样本可以学习）。
- 一对三错 $r = [1,0,0,0]$：$\hat{A}_1 = +1.73$，$\hat{A}_{2,3,4} = -0.58$ → 大力强化唯一的正确回答。

这种"全对/全错时不更新"的行为看似浪费，但实际上非常合理——它避免了在没有区分信号时引入噪声梯度。

## 2. KL 散度正则化：为什么用这个特殊形式？

为了防止策略"钻空子"（Reward Hacking）或丧失语言连贯性，需要约束 $\pi_\theta$ 不偏离参考策略 $\pi_{\text{ref}}$ 太远。

标准的 KL 散度 $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_{o \sim \pi_\theta}\left[\log \frac{\pi_\theta(o|s)}{\pi_{\text{ref}}(o|s)}\right]$ 需要对 $\pi_\theta$ 的完整分布求期望，但我们手头只有来自 $\pi_{\theta_{\text{old}}}$ 的采样。GRPO 使用了一种**基于样本的无偏 KL 估计量**：

$$
\hat{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \frac{\pi_{\text{ref}}(o_i|s)}{\pi_\theta(o_i|s)} - \log \frac{\pi_{\text{ref}}(o_i|s)}{\pi_\theta(o_i|s)} - 1
$$

**这个公式从何而来？** 令 $u = \frac{\pi_{\text{ref}}(o_i|s)}{\pi_\theta(o_i|s)}$，则上式就是 $f(u) = u - \log u - 1$。这是一个以 $u=1$ 为最小值点的非负凸函数（$f(1) = 0$, $f''(u) = 1/u^2 > 0$）。它本质上是**反向 KL 散度**的 $f$-散度形式。

更直观地理解：
- 当 $\pi_\theta = \pi_{\text{ref}}$ 时，$u = 1$，惩罚为 0——不偏离，不惩罚。
- 当 $\pi_\theta(o_i|s) \ll \pi_{\text{ref}}(o_i|s)$ 时，$u \gg 1$，$u - \log u - 1$ 近似为 $u$（线性增长）——模型大幅减小某些回答概率时受到惩罚。
- 当 $\pi_\theta(o_i|s) \gg \pi_{\text{ref}}(o_i|s)$ 时，$u \to 0$，$-\log u$ 主导（对数增长）——模型大幅增大某些回答概率时也受到惩罚。

> **与 PPO (RLHF) 的 KL 惩罚对比**：PPO 中的 KL 惩罚 $\beta \cdot (\log \pi_\theta - \log \pi_{\text{ref}})$ 是逐 token 加到奖励上的（见上一篇的 `kl_penalty`）。GRPO 则将 KL 惩罚直接作为损失的一部分，并使用了这种非对称的 $f$-散度形式，对"概率塌缩"（某些回答概率趋近 0）更加敏感。

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

回到数学题的例子，这个公式做了两件事：

1. **裁剪机制**：限制正确答案 $o_1, o_2$ 的概率不会一次涨太多（$\hat{A}_i = +1$），错误答案 $o_3, o_4$ 的概率不会一次降太多（$\hat{A}_i = -1$）——保证信任域约束。
2. **KL 正则**：约束模型不要为了做对数学题而丧失自然语言能力——维持生成质量。

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
# 仅加载问题文本 s；在线阶段由策略采样得到回答 o，无需预先标注偏好对
prompts = load_dataset("math_problems")  # 只有 Prompt，没有标注答案
# 将 (prompt, response) 映射为标量奖励 r，供同一 prompt 下 G 个样本做组内相对比较
def reward_fn(prompt, response):
    """奖励函数：判题器 (可以是规则匹配、奖励模型、或混合)"""
    # 二值奖励示例：也可换成 RM 输出的连续分，再进入组内标准化
    return 1.0 if verify_math_answer(prompt, response) else 0.0
```

**Step 2: 模型定义 — 只需要两个模型！**

GRPO 的一大优势：与 DPO 一样只需要两个模型，但保留了在线 RL 的探索能力。

```python
import torch  # 张量计算与自动微分，支撑策略梯度
import torch.nn.functional as F  # 常用算子（如 log_softmax），便于算 token 对数概率
from transformers import AutoModelForCausalLM, AutoTokenizer  # 加载因果 LM 与分词器

# 模型 1: 待训练的策略模型 (Actor)
actor = AutoModelForCausalLM.from_pretrained("sft_checkpoint")  # π_θ：被优化的策略，产生 log π_θ(o|s)
# 模型 2: 冻结的参考模型 (Reference) — SFT 后的快照
ref_model = AutoModelForCausalLM.from_pretrained("sft_checkpoint")  # π_ref：KL 正则锚点，抑制 reward hacking
ref_model.requires_grad_(False)  # 参考策略不更新，仅前向算 log π_ref 供 KL 项使用

# 对比:
#   PPO:  Actor + Critic + Reference + Reward Model = 4 个模型
#   DPO:  Actor + Reference = 2 个模型 (但是离线)
#   GRPO: Actor + Reference = 2 个模型 (在线!)

tokenizer = AutoTokenizer.from_pretrained("sft_checkpoint")  # 文本 ↔ token id，条件化生成与算对数概率共用
optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-6)  # 只更新 Actor，省显存（无 Critic）

# 超参数
G = 8            # 每个 Prompt 采样的回答数量
clip_range = 0.2 # PPO 裁剪阈值 ε
beta = 0.04      # KL 惩罚系数
K_epochs = 2     # 每批数据的更新轮数 (重要性采样允许多次更新)
```

**Step 3: 在线采样 + 奖励收集**

这是 GRPO 与 DPO 的核心区别——GRPO 用当前策略**在线生成**回答并即时评分：

```python
def collect_group_rollouts(actor, prompts_batch, G, reward_fn):
    """对每个 Prompt 采样 G 个回答并打分"""
    # 存放每条轨迹：条件 prompt、生成序列、标量奖励（供组内基线与优势）
    all_prompt_ids, all_response_ids, all_rewards = [], [], []
    # 采样时刻的策略对数概率（作 π_old）与参考策略对数概率（作 KL 惩罚）
    all_old_log_probs, all_ref_log_probs = [], []

    actor.eval()  # 推理模式：关闭 dropout，采样行为稳定
    with torch.no_grad():  # 本阶段不反传：固定采样分布，避免用同一批数据误算「采样梯度」
        for prompt in prompts_batch:  # 对每个环境状态 s（prompt）展开一组 rollout
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt")  # 将 s 编码为模型输入 token
            for _ in range(G):  # 同一 s 下独立采样 G 条 o_i，构成 GRPO 的「一组」样本
                # 用当前策略采样一个回答 (temperature sampling)
                response_ids = actor.generate(prompt_ids, max_new_tokens=512,
                                               do_sample=True, temperature=0.7)  # o_i ~ π_old(·|s)，在线探索来源
                response_text = tokenizer.decode(response_ids[0])  # 解码为字符串，供规则/RM 打分

                # 奖励函数打分 (规则判题器 / 奖励模型)
                reward = reward_fn(prompt, response_text)  # 标量 r_i：强化信号，随后与组内其他 r 比较

                # 计算当前策略和参考策略的 token 级对数概率
                old_logps = compute_token_log_probs(actor, prompt_ids, response_ids)  # ∑_t log π_old(o^t|·)，用于重要性比 ρ
                ref_logps = compute_token_log_probs(ref_model, prompt_ids, response_ids)  # ∑_t log π_ref，用于 f-散度 KL

                all_prompt_ids.append(prompt_ids)
                all_response_ids.append(response_ids)
                all_rewards.append(reward)
                all_old_log_probs.append(old_logps)
                all_ref_log_probs.append(ref_logps)

    actor.train()  # 回到训练模式，后续计算 new_log_probs 时需要梯度
    return all_prompt_ids, all_response_ids, all_rewards, all_old_log_probs, all_ref_log_probs
```

**Step 4: 组内相对优势计算**

```python
def compute_group_advantages(rewards, G):
    """
    将 rewards (长度为 batch_size * G) 重塑为 (batch_size, G)，
    在 G 维度上做标准化
    """
    # 每个 prompt 一行、每行 G 个 r_i，对应「矮子里拔高个」的同一组样本
    rewards = torch.tensor(rewards).reshape(-1, G)   # (batch_size, G)
    mean_r = rewards.mean(dim=1, keepdim=True)        # 组内均值 μ_R：无 Critic 时的基线（REINFORCE baseline 的 MC 估计）
    std_r = rewards.std(dim=1, keepdim=True)           # 组内标准差 σ_R：标准化优势尺度，稳定梯度
    advantages = (rewards - mean_r) / (std_r + 1e-8)  # 相对优势 = (r_i-μ_R)/(σ_R+ε)；全同奖励时 ε 避免除零、优势为 0
    return advantages.reshape(-1)                      # 展平回 (batch_size * G,) 与 flatten 后的轨迹索引对齐
```

**Step 5: 完整训练循环**

```python
for step in range(total_steps):  # 外循环：每步先采样新轨迹，再基于固定 π_old 做多轮更新
    # --- 阶段 1: 在线采样 (GRPO 独有的 — DPO 没有这一步) ---
    prompts_batch = sample_prompts(prompts, batch_size=8)  # 从题库抽一批 s，控制每步计算量
    prompt_ids, response_ids, rewards, old_log_probs, ref_log_probs = \
        collect_group_rollouts(actor, prompts_batch, G, reward_fn)  # 每题 G 条 o_i、r_i 及采样时 log π_old / log π_ref

    # --- 阶段 2: 计算组内相对优势 ---
    advantages = compute_group_advantages(rewards, G)  # 用组内统计替代 Critic，得到每个样本的相对优势

    # --- 阶段 3: 多 epoch 更新 (重要性采样允许复用数据) ---
    for epoch in range(K_epochs):  # 同一批数据上重复 K 次：依赖重要性采样校正分布偏移
        for idx in minibatch_indices(len(response_ids), batch_size=16):  # 小批量随机子集，降噪并省显存
            # 用当前 actor 重新计算 token 级对数概率
            new_log_probs = compute_token_log_probs(actor, prompt_ids[idx], response_ids[idx])  # 当前 π_θ 下轨迹对数概率

            # 重要性采样比率 (token 级求和后取 exp)
            log_ratio = new_log_probs - old_log_probs[idx]  # log π_θ - log π_old（与文中 ρ_i(θ) 一致）
            ratio = torch.exp(log_ratio)  # ρ = π_θ/π_old：把在 π_old 下采样的梯度校正为 π_θ 目标

            # PPO 裁剪目标
            adv = advantages[idx]  # 与当前 minibatch 对齐的组内标准化优势
            surr1 = ratio * adv  # 未裁剪 surrogate：重要性比 ρ 与相对优势的乘积
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv  # 裁剪后的 surrogate：限制单步策略变化（信任域）
            policy_loss = -torch.min(surr1, surr2).mean()  # 最小化负的 min(·)：实现「保守」的策略改进目标

            # KL 散度惩罚 (f-散度形式: u - log u - 1, 其中 u = π_ref/π_θ)
            log_ref_ratio = ref_log_probs[idx] - new_log_probs  # log(π_ref/π_θ)，在已采样 token 上估计 KL
            kl_penalty = (torch.exp(log_ref_ratio) - log_ref_ratio - 1.0).mean()  # f(u)=u-log u-1：拉回参考策略，防塌缩/乱说

            # 总损失
            loss = policy_loss + beta * kl_penalty  # 策略项 + β·KL：与文中最大化 J_GRPO 对应（最小化负目标）

            optimizer.zero_grad()
            loss.backward()  # 梯度仅流向 Actor（无 Critic）
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)  # 裁剪梯度范数，稳定大模型 RL
            optimizer.step()  # 更新 θ，使高相对优势的轨迹概率升、低相对优势的降（在 clip/KL 约束内）

    # --- 阶段 4: 同步旧策略 ---
    # old_policy ← current_policy (下一轮采样用更新后的策略)
    # 注意: 与 PPO 不同，GRPO 通常每轮都重新采样，不需要显式的 old_policy 拷贝
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

**用数学题的例子理解三者区别**：
- **PPO**：需要一个 Critic 网络回答"这道积分题的平均得分大概是多少"——需要额外显存训练 Critic。
- **DPO**：需要一份标注好的数据"正确答案 $\frac{1}{3}$ vs 错误答案 $\frac{1}{2}$"——需要预先标注，无法在线探索。
- **GRPO**：让模型做 8 遍，用 8 次实际得分算平均——用计算（多次采样）换显存（去掉 Critic），保留在线探索。

**开源代码参考：** GRPO 随 DeepSeek 开源而爆火，Hugging Face **TRL** 库 ([`trl.GRPOTrainer`](https://huggingface.co/docs/trl/grpo_trainer)) 提供了生产级实现。

GRPO 证明了在生成式大模型时代，简单的经验统计（组内均值）往往比复杂的神经网络预测（Critic）更加鲁棒和高效。

> 下一篇：[笔记｜生成模型（二十）：Flow-GRPO 与图像生成应用（基于 Flux 的代码解析）](/chengYi-xun/posts/21-flow-grpo/)
