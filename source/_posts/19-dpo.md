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

### Step 1：RLHF 的 KL 约束优化目标

在传统的 RLHF 中（上一篇 PPO 的四模型架构），我们的目标是最大化奖励，同时用 KL 散度约束策略不要偏离参考模型太远：
$$
\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi} \left[ r(x, y) \right] - \beta \cdot D_{\text{KL}}(\pi \| \pi_{\text{ref}})
$$

将 KL 散度展开，等价于：
$$
\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi} \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right]
$$

这正是上一篇 RLHF-PPO 中的奖励修正公式。PPO 用在线采样 + 裁剪来近似求解这个问题。**DPO 的出发点是：这个问题是否有解析解（闭式解）？**

### Step 2：推导最优策略的闭式解

对于固定的 Prompt $x$，上述目标关于 $\pi(\cdot|x)$ 的优化可以写成：

$$
\max_{\pi(\cdot|x)} \sum_y \pi(y|x) \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right]
$$

这是一个带归一化约束（$\sum_y \pi(y|x) = 1$）的凸优化问题。我们定义一个特殊的分布：

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left( \frac{r(x, y)}{\beta} \right), \quad Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\left( \frac{r(x, y)}{\beta} \right)
$$

其中 $Z(x)$ 是**配分函数（Partition Function）**，确保 $\pi^*$ 是一个合法的概率分布（所有 $y$ 的概率之和为 1）。

**为什么 $\pi^*$ 是最优解？** 将目标函数改写为与 $\pi^*$ 的 KL 散度（具体推导见下）：

$$
\begin{aligned}
&\sum_y \pi(y|x) \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right] \\
&= \sum_y \pi(y|x) \left[ \beta \log \pi^*(y|x) + \beta \log Z(x) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right] \\
&\quad \text{（因为 } r(x,y) = \beta \log \pi^*(y|x) - \beta \log \pi_{\text{ref}}(y|x) + \beta \log Z(x) \text{）} \\
&= \beta \log Z(x) + \beta \sum_y \pi(y|x) \log \frac{\pi^*(y|x)}{\pi(y|x)} \\
&= \beta \log Z(x) - \beta \cdot D_{\text{KL}}(\pi \| \pi^*)
\end{aligned}
$$

由于 $D_{\text{KL}}(\pi \| \pi^*) \geq 0$，且当且仅当 $\pi = \pi^*$ 时取等号，所以**目标函数在 $\pi = \pi^*$ 时取到最大值** $\beta \log Z(x)$。

**直觉理解 $\pi^*$**：最优策略是参考策略经过**奖励加权**后的版本——高奖励的回答被指数级放大（$\exp(r/\beta)$），低奖励的被压制。$\beta$ 控制放大的程度：$\beta$ 越小，最优策略越激进地集中在高奖励回答上；$\beta$ 越大，越接近原始参考策略。

### Step 3：用策略反向表示奖励

既然 $\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left( \frac{r(x, y)}{\beta} \right)$，对两边取对数并重排，我们可以**用最优策略的概率来表示奖励**：

$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

这一步至关重要：它建立了**奖励函数**和**策略概率**之间的双向映射。如果我们有最优策略 $\pi^*$，就不需要显式的奖励模型了——奖励已经被隐式地编码在策略的概率中。

### Step 4：Bradley-Terry 偏好模型

**Bradley-Terry (BT) 模型**是偏好学习中最经典的概率模型。它假设：人类选择 $y_w$ 胜过 $y_l$ 的概率，取决于两者奖励之差通过 Sigmoid 函数的映射：

$$
p(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))
$$

其中 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 是 Sigmoid 函数。

**直觉理解**：如果好回答的奖励远高于坏回答（$r(x, y_w) \gg r(x, y_l)$），Sigmoid 输出接近 1（人类几乎一定偏好 $y_w$）；如果两者奖励接近，Sigmoid 输出接近 0.5（人类难以区分）。BT 模型正是传统 RLHF 中训练奖励模型 $R_\psi$ 的理论基础——用人类偏好数据最大化上式的似然来拟合 $R_\psi$。

### Step 5：消除奖励模型——DPO 的关键一步

现在，DPO 论文的核心洞察来了：**将 Step 3 中用策略表示的奖励，代入 Step 4 的 BT 模型**，奖励模型就被完全消除了：

$$
\begin{aligned}
r(x, y_w) - r(x, y_l) &= \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} + \cancel{\beta \log Z(x)} \right) - \left( \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} + \cancel{\beta \log Z(x)} \right) \\
&= \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
\end{aligned}
$$

难以计算的配分函数 $\beta \log Z(x)$ 在相减时**被完全抵消了**！这个消除至关重要——$Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(r(x,y)/\beta)$ 需要对所有可能的回答 $y$ 求和，在语言模型中这个求和空间是天文数字级别的，根本无法计算。DPO 通过配对对比巧妙地绕过了这个问题。

### DPO 的最终损失函数

将上述结果代入 BT 模型的负对数似然，我们得到 **DPO 的损失函数**：
$$
\mathcal{L}_{\text{DPO}}(\theta) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

回到排序函数的例子，这个损失函数的含义就很清晰了：它通过 Sigmoid + 负对数的方式，推动模型让好答案（纯函数版排序）的"相对偏好度"大于坏答案（修改原数组版排序）。最小化这个损失，就是在最大化人类偏好 $y_w$ 被正确区分的概率。

### DPO 梯度分析：损失函数在做什么？

对 DPO 损失求梯度，可以得到：

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \mathbb{E}_{(x, y_w, y_l)} \left[ \underbrace{\sigma(-\hat{r}_\theta)}_{\text{权重}} \left( \underbrace{\nabla_\theta \log \pi_\theta(y_w|x)}_{\text{增大好答案概率}} - \underbrace{\nabla_\theta \log \pi_\theta(y_l|x)}_{\text{减小坏答案概率}} \right) \right]
$$

其中 $\hat{r}_\theta = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$ 是**隐式奖励差**。

**梯度揭示了 DPO 的自适应学习机制**：
- $\sigma(-\hat{r}_\theta)$ 是一个自动调节的权重。当模型已经学得很好（$\hat{r}_\theta$ 很大），$\sigma(-\hat{r}_\theta) \to 0$，梯度变小——模型不再在"已学会"的样本上浪费力气。
- 当模型还没学好（$\hat{r}_\theta$ 接近 0 或为负），$\sigma(-\hat{r}_\theta) \to 1$，梯度很大——模型集中火力攻克"还没学会"的样本。
- 这相当于一种**自动课程学习（Curriculum Learning）**，模型自动聚焦于最需要改进的偏好对。

### DPO 的完整实现

以下是 DPO 的完整 PyTorch 实现伪代码，包括数据加载、模型定义和完整的训练循环。

**Step 1: 数据格式 — 偏好对数据集**

与 PPO 不同，DPO 不需要在线采样，直接使用**预先收集好的偏好对**数据集：

```python
"""
DPO 数据格式：每条数据是一个三元组 (prompt, chosen, rejected)
例如（来自 Anthropic HH-RLHF 数据集）：
{
    "prompt": "用 Python 写一个排序函数。",
    "chosen": "def sort(arr): return sorted(arr)  # 纯函数，不修改原数组",
    "rejected": "def sort(arr): arr.sort(); return arr  # 有副作用，修改了原数组"
}
"""
# DPO 为离线对齐：无需环境交互，直接用人类标注的 (x, y_w, y_l) 做对比学习
preference_dataset = load_dataset("preference_pairs")  # List of (prompt, chosen, rejected)；每条对应一次“胜者 vs 败者”的 BT 似然项
```

**Step 2: 模型定义 — 只需要两个模型！**

DPO 最大的优势：相比 RLHF-PPO 的四模型架构，DPO 只需要两个模型：

```python
import torch  # 张量与自动求导：DPO 损失只对 π_θ 反传
import torch.nn.functional as F  # log_softmax / logsigmoid：实现序列 log π 与 BT 负对数似然
from transformers import AutoModelForCausalLM, AutoTokenizer  # 因果 LM：拟合策略分布 π(y|x)

# 模型 1: 待训练的策略模型 (Actor)
actor = AutoModelForCausalLM.from_pretrained("sft_checkpoint")  # 当前策略 π_θ，对齐要更新的对象
# 模型 2: 冻结的参考模型 (Reference) — 就是 SFT 后的快照
ref_model = AutoModelForCausalLM.from_pretrained("sft_checkpoint")  # 冻结的 π_ref，作 KL 正则锚点
ref_model.requires_grad_(False)  # 不训练参考模型，仅前向计算 log π_ref 以构造隐式奖励差

# 对比 RLHF-PPO：不需要 Critic 模型，不需要 Reward 模型！
# PPO 需要 4 个模型 → DPO 只需要 2 个，显存节省约 50%

tokenizer = AutoTokenizer.from_pretrained("sft_checkpoint")  # 将 prompt 与回答编成 token 序列，才能逐 token 累加 log π
optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-6)  # 仅优化 Actor；DPO 等价于对偏好对的监督目标做一阶梯度下降

beta = 0.1  # KL 惩罚系数，控制偏离参考模型的程度
```

**Step 3: 计算序列的对数概率**

语言模型中，一个回答的对数概率是所有 token 对数概率之和：

$$
\log \pi_\theta(y|x) = \sum_{t=1}^{T} \log \pi_\theta(y_t | x, y_{<t})
$$

```python
def compute_log_probs(model, input_ids, labels, attention_mask):
    """计算模型对给定序列的对数概率"""
    # ref 模型用 no_grad 省显存；训练中的 actor 需保留计算图以回传 DPO 梯度
    with torch.no_grad() if not model.training else torch.enable_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # 各步对词表的未归一化打分，用于下一 token 分布
    # logits: (batch, seq_len, vocab_size) → 取每个位置上真实 token 的对数概率
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # 因果 LM：位置 t 预测 t+1，故 logits 与 labels 错一位对齐
    # labels 向左移一位 (next-token prediction)
    per_token_log_probs = torch.gather(log_probs, dim=-1, index=labels[:, 1:].unsqueeze(-1)).squeeze(-1)  # 取出真实续写 token 的 log π(y_t|x,y_{<t})
    # 只对回答部分求和 (mask 掉 prompt 部分)
    return (per_token_log_probs * attention_mask[:, 1:]).sum(dim=-1)  # 序列级 log π(y|x)，供 log-ratio 与 β·(···) 使用
```

**Step 4: 完整训练循环**

```python
for epoch in range(num_epochs):  # 多轮扫偏好数据；无在线 rollout，与 SFT 式 epoch 类似
    for batch in dataloader(preference_dataset, batch_size=4):  # 批内并行多条 (x,y_w,y_l)，稳定梯度估计
        prompts, chosen_responses, rejected_responses = batch  # Unpack：提示 + 人类偏好胜/负回答

        # --- 拼接 prompt + response，分别对 chosen 和 rejected 做 tokenize ---
        chosen_ids, chosen_mask, chosen_labels = tokenize(prompts, chosen_responses)  # y_w 的 token 与 mask；labels 用于算 log π(y_w|x)
        rejected_ids, rejected_mask, rejected_labels = tokenize(prompts, rejected_responses)  # y_l 同理，与 y_w 共享同一 prompt x

        # --- 前向传播: 计算四组对数概率 ---
        # 当前策略 π_θ 对好/坏回答的对数概率
        policy_chosen_logps = compute_log_probs(actor, chosen_ids, chosen_labels, chosen_mask)  # log π_θ(y_w|x)，对齐要抬高的项
        policy_rejected_logps = compute_log_probs(actor, rejected_ids, rejected_labels, rejected_mask)  # log π_θ(y_l|x)，对齐要压低的项

        # 参考策略 π_ref 对好/坏回答的对数概率 (不需要梯度)
        with torch.no_grad():
            ref_chosen_logps = compute_log_probs(ref_model, chosen_ids, chosen_labels, chosen_mask)  # log π_ref(y_w|x)，隐式奖励里的基准
            ref_rejected_logps = compute_log_probs(ref_model, rejected_ids, rejected_labels, rejected_mask)  # log π_ref(y_l|x)；Z(x) 在差分中消掉

        # --- 计算 DPO 损失 (对应数学推导的最终公式) ---
        # 隐式奖励差: β · [log(π_θ(y_w)/π_ref(y_w)) - log(π_θ(y_l)/π_ref(y_l))]
        chosen_logratios = policy_chosen_logps - ref_chosen_logps  # log π_θ - log π_ref 在 y_w：相对 ref 更偏好好答案的程度
        rejected_logratios = policy_rejected_logps - ref_rejected_logps  # y_l 上的 log-ratio，与上式相减得 BT 指数内的标量
        logits = beta * (chosen_logratios - rejected_logratios)  # 论文中的 β·(log-ratio 差)，即代入 σ 前的“隐式奖励差”

        # 负对数似然: -log σ(logits)
        # 使用 F.logsigmoid 而非 log(sigmoid(x))，避免数值下溢
        loss = -F.logsigmoid(logits).mean()  # 最大化 σ(logits)≈p(y_w≻y_l|x)；最小化此项即拟合 Bradley-Terry 偏好

        # --- 反向传播 (标准的一阶梯度! 无需二阶优化) ---
        optimizer.zero_grad()  # 每步先清空旧梯度，避免累积
        loss.backward()  # 梯度含 σ(-r̂) 权重：难样本权重大，已对齐样本权重小（自适应课程）
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)  # 防止大模型对齐时梯度爆炸
        optimizer.step()  # 仅更新 π_θ；π_ref 始终冻结

        # --- 监控指标 ---
        with torch.no_grad():
            implicit_reward_diff = logits.mean().item()  # 隐式奖励差 (越大越好)；batch 内平均 BT 边强度
            accuracy = (logits > 0).float().mean().item()  # 分类准确率 (好答案得分是否 > 坏答案)；偏好方向是否判对
```

> **与 PPO 实现的关键对比**：
> - PPO 需要在线采样（用 Actor 生成回答 → Reward Model 打分 → GAE 估计优势 → 多 epoch 裁剪更新），代码约 100 行。
> - DPO 只需要前向传播 + 交叉引用四个对数概率 + 一行损失计算，代码不到 30 行。
> - PPO 涉及重要性采样比 $r(\theta)$、裁剪区间 $[1-\epsilon, 1+\epsilon]$、GAE $\lambda$ 等超参数；DPO 只有一个超参数 $\beta$。

**开源代码参考：** Hugging Face **TRL** 库 ([`trl.DPOTrainer`](https://huggingface.co/docs/trl/dpo_trainer)) 提供了生产级别的 DPO 实现，支持 LoRA、多 GPU 训练、梯度累积等。

## DPO 的优势与局限

### 优势

1. **极其简单**：DPO 将 RL 问题转化为**二元分类**问题。整个训练流程与 SFT 几乎一致——加载数据、前向传播、反向传播、更新参数。不需要训练奖励模型，不需要 Critic 网络，不需要 PPO 的在线采样、GAE、多 epoch 裁剪更新。

2. **显存友好**：DPO 只需要 2 个模型（Actor + Reference），而 PPO 需要 4 个（Actor + Critic + Reference + Reward Model）。对于 70B 参数模型（FP16 约 140GB），PPO 需要 4 × 140GB = 560GB 显存，DPO 只需要 2 × 140GB = 280GB——差距巨大。

3. **训练稳定**：DPO 的损失函数是平滑的（Sigmoid + 对数），梯度有良好的数学性质（自适应加权，见梯度分析）。而 PPO 涉及多个相互耦合的组件（Actor 更新 → Critic 更新 → 优势估计变化 → Actor 目标变化），任何一个环节不稳定都会导致训练崩溃。

4. **超参数少**：DPO 核心超参数只有 $\beta$（KL 惩罚系数）。PPO 则有裁剪区间 $\epsilon$、GAE 参数 $\lambda$、value function coefficient、entropy coefficient、mini-batch size、epoch 数等大量需要调优的参数。

### 局限

1. **离线学习的分布偏移问题 (Distribution Shift)**

DPO 依赖于预先收集好的静态偏好数据集，而这些偏好对通常是由参考策略 $\pi_{\text{ref}}$（或另一个模型）生成的。随着训练的进行，$\pi_\theta$ 逐渐偏离 $\pi_{\text{ref}}$，训练数据对当前策略而言越来越"过时"——这就是**分布偏移**问题。

数学上看，DPO 损失中的 $\log \pi_\theta(y_w|x)$ 和 $\log \pi_\theta(y_l|x)$ 是在**固定的** $(y_w, y_l)$ 上计算的。如果 $\pi_\theta$ 已经学到了与 $\pi_{\text{ref}}$ 非常不同的分布，那么数据集中的 $(y_w, y_l)$ 可能都落在 $\pi_\theta$ 的低概率区域，梯度信号变得极弱。这类似于重要性采样中权重方差爆炸的问题（见上一篇）。

> **改进尝试**：Iterative DPO / Online DPO 通过定期用当前策略重新生成偏好对来缓解分布偏移，但这本质上又引入了在线采样的开销。

2. **探索能力有限，难以产生涌现**

在线 RL（如 PPO）允许模型在探索中发现比人类标注更好的答案（例如 AlphaGo 发现新定式，DeepSeek-R1 涌现出长思维链"顿悟"）。DPO 只能模仿数据集中已有的偏好，难以产生真正的"涌现"和"超越"。

**用例子理解**：假设一道很难的数学推理题，人类标注员自己都做不出来，无法提供正确的偏好标注。DPO 就束手无策了。而在线 RL 可以让模型自己反复尝试，一旦碰巧写出正确答案，就立刻强化它——这就是 DeepSeek-R1 中"顿悟时刻"的由来。

3. **隐式奖励的局限性**

DPO 假设最优策略和奖励之间存在精确的双射关系（Step 2-3 的推导）。但实际中这个假设并不总是成立——当偏好数据有噪声、标注不一致、或存在多种同样好的回答风格时，DPO 可能学到一个扭曲的隐式奖励函数。PPO 的显式奖励模型可以单独训练和评估，更容易发现和修正奖励建模的问题。

### 全景对比

| 维度 | PPO (RLHF) | DPO |
|:---:|:---:|:---:|
| **模型数量** | 4 (Actor + Critic + Ref + RM) | 2 (Actor + Ref) |
| **训练方式** | 在线 RL（采样 → 评分 → 更新） | 离线监督学习（直接从偏好对学习） |
| **核心优化** | 裁剪后的策略梯度 + GAE | 负对数似然 (交叉熵变体) |
| **超参数** | 多 ($\epsilon$, $\lambda$, $\gamma$, lr, ...) | 少 ($\beta$, lr) |
| **探索能力** | 强（在线采样发现新解） | 弱（受限于离线数据集） |
| **稳定性** | 低（多组件耦合，易崩溃） | 高（标准监督学习） |
| **适用场景** | 推理型模型（数学、代码、长思维链） | 通用对齐（对话质量、安全性） |

因此，虽然 DPO 在开源社区大火，但在追求极致推理能力的最前沿大模型中，**在线强化学习（Online RL）仍然是不可替代的王者**。

那么，如何解决在线 RL（PPO）的显存危机呢？这就引出了我们下一篇的主角：GRPO。

> 下一篇：[笔记｜生成模型（十九）：大模型在线 RL 破局者：GRPO 算法详解](/chengYi-xun/posts/20-grpo/)
