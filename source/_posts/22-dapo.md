---
title: 笔记｜生成模型（二十一）：DAPO：从 GRPO 到大规模推理 RL 的工程实践
date: 2026-04-05 10:00:00
cover: false
mathjax: true
categories\n
Notes
tags\n
Deep learning
 - Generative models theory
 - Reinforcement Learning
series: Diffusion Models theory
---

> 本文为 RL 系列的第六篇。在上几篇中我们推导了 GRPO 的核心思想并将其应用于图像生成。本文将介绍 GRPO 的工程增强版——DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization），它是字节跳动 Seed 团队与清华 AIR 联合提出的大规模 LLM 强化学习算法，用 Qwen2.5-32B 基座模型在 AIME 2024 上达到 50 分（超过 DeepSeek-R1-Zero 的 47 分），且训练步数减少 50%。
>
> 论文：[DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)（2025.03）

# 先看问题：GRPO 在大规模训练中遇到了什么？

**还是用上一篇的数学题例子开场。** 假设我们用 GRPO 训练一个推理模型解数学竞赛题（AIME 级别），每道题让模型生成 $G = 16$ 个回答，然后用规则判题：对了奖励 $+1$，错了奖励 $-1$。

先看一组理想情况下的训练数据：

| 题目 | 回答数 | 正确/错误 | 奖励分布 | $\sigma_R$ |
|:---:|:---:|:---:|:---:|:---:|
| 竞赛题 A（模型完全不会） | 16 | 0/16 | 全是 $-1$ | $0$ |
| 竞赛题 B（模型基本掌握） | 16 | 12/4 | 12 个 $+1$，4 个 $-1$ | $>0$ |
| 竞赛题 C（模型轻松拿下） | 16 | 16/0 | 全是 $+1$ | $0$ |

**问题暴露了**：

1. **题 A 和题 C 的 $\sigma_R = 0$**（所有回答的奖励完全相同），GRPO 的优势公式 $\hat{A}_i = \frac{r_i - \mu_R}{\sigma_R}$ 中分母为零，梯度信号为零——这些样本完全浪费了！这不只是效率问题：大量的零梯度让有效 batch size 缩水，训练不稳定。

2. **熵崩溃（Entropy Collapse）**：PPO/GRPO 使用对称裁剪 $[1 - \varepsilon, 1 + \varepsilon]$。对于一个概率仅 $\pi_{\theta_{\text{old}}}(o_t | q) = 0.01$ 的低概率 Token，即使 GRPO 发现它出现在正确回答中应被鼓励，裁剪上界也只允许概率增长到 $0.01 \times (1 + 0.2) = 0.012$——几乎动不了。**探索性 Token 被压制**，模型越来越"保守"，策略熵持续下降，最终只会生成少数固定模式的回答。

3. **长回答的奖励噪声**：数学推理题目的回答动辄数千 Token。如果模型生成超长但错误的回答（比如陷入循环推理），传统的 $+1/-1$ 奖励一视同仁，导致训练信号噪声极大。

用原始 GRPO 训练 Qwen2.5-32B，团队在 AIME 2024 上只拿到了 30 分——远低于 DeepSeek-R1-Zero 的 47 分。**DAPO 的四个技术正是为解决以上三个问题而生**。

---

# DAPO 的四大核心技术

## 技术一：Clip-Higher（非对称裁剪）

**问题复盘**：对称裁剪 $[1-\varepsilon, 1+\varepsilon]$ 在抑制概率下降和鼓励概率上升时使用相同的力度。但"探索"需要让小概率 Token 涨上去，而"维稳"需要限制大概率 Token 不要突变。这两件事不应该用同一个阈值。

**用例子理解**：模型解题时正确用了"换元法"（概率仅 0.01），而"直接暴力展开"（概率 0.8）虽然也对但不够优雅。对称裁剪下，"换元法"的概率上界只能到 $0.01 \times 1.2 = 0.012$（增长 20%），而"暴力展开"的概率上界可到 $0.8 \times 1.2 = 0.96$（绝对增量 0.16）。低概率探索路径几乎无法获得实质性增长。

**Clip-Higher 的解决方案**：将裁剪范围解耦为两个独立参数 $\varepsilon_{\text{low}}$ 和 $\varepsilon_{\text{high}}$：

$$
\mathcal{J}_{\text{DAPO}}(\theta) = \mathbb{E}\left[\frac{1}{\sum_{i=1}^{G}|o_i|} \sum_{i=1}^{G}\sum_{t=1}^{|o_i|} \min\Big(r_{i,t}(\theta)\hat{A}_{i,t},\ \text{clip}\big(r_{i,t}(\theta),\, 1-\varepsilon_{\text{low}},\, 1+\varepsilon_{\text{high}}\big)\hat{A}_{i,t}\Big)\right]
$$

其中 $r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} | q, o_{<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{<t})}$ 是 Token 级别的重要性采样比率。

**关键参数选择**：DAPO 论文中使用 $\varepsilon_{\text{low}} = 0.2$，$\varepsilon_{\text{high}} = 0.28$。高侧放宽意味着概率上升的空间更大（允许比率最大到 $1.28$ 而不是 $1.2$），从而让"探索性"Token 有更大的成长空间，避免熵崩溃。

**直觉对比**：

| | 对称裁剪（PPO/GRPO） | Clip-Higher（DAPO） |
|---|---|---|
| 裁剪范围 | $[1-\varepsilon, 1+\varepsilon] = [0.8, 1.2]$ | $[1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}}] = [0.8, 1.28]$ |
| 概率 0.01 的 Token 可涨到 | $0.012$ | $0.0128$ |
| 效果 | 探索被压制，熵崩溃 | 探索空间更大，策略保持多样性 |

---

## 技术二：Dynamic Sampling（动态采样）

**问题复盘**：回到开头的例子——题 A（全错）和题 C（全对）的 16 个回答没有产生任何梯度信号。在大规模训练中，这种"零方差"样本可能占到 batch 的一大半，导致有效训练数据大幅缩水。

**Dynamic Sampling 的解决方案**：只保留组内奖励方差大于零（$\sigma_R > 0$）的 Prompt 组，过滤掉全对或全错的无效样本，然后累积有效样本直到达到目标 batch size。

**用例子理解**：假设目标 batch 需要 512 个 Prompt，每轮我们采样 $512 \times 3 = 1536$ 个 Prompt（`batch_multiplier = 3`）：

| 步骤 | 操作 | 结果 |
|:---:|:---|:---|
| 1 | 采样 1536 个 Prompt，每个生成 16 个回答 | 共 24576 个回答 |
| 2 | 对每组计算 $\sigma_R$ | 发现 400 组 $\sigma_R > 0$，1136 组 $\sigma_R = 0$ |
| 3 | 只保留 $\sigma_R > 0$ 的 400 组存入缓存 | 缓存有 400 组 |
| 4 | 缓存不足 512 组，继续采样新一批 Prompt | 新增 150 组有效样本 |
| 5 | 缓存达到 550 组（≥ 512），截取 512 组训练 | 开始梯度更新 |

**核心约束**：DAPO 同时要求每个保留的 Prompt 组中必须同时包含正确和错误的回答：

$$
0 < |\{o_i \mid \text{is\_correct}(a, o_i)\}| < G
$$

这确保了每个 Prompt 都能提供"做对了的回答应该被鼓励、做错了的回答应该被抑制"的双向梯度信号。

**伪代码**：
```python
cache = []

while len(cache) < target_batch_size:
    prompts = sample_prompts(batch_multiplier * target_batch_size)
    
    for prompt in prompts:
        responses = model.generate(prompt, num_return=G)
        rewards = judge(responses)
        
        if rewards.std() > 0:
            cache.append((prompt, responses, rewards))
    
    if generation_rounds >= max_gen_batches:
        raise RuntimeError("无法累积足够的有效样本")

train_batch = cache[:target_batch_size]
```

---

## 技术三：Token-Level Policy Gradient Loss（Token 级损失）

**问题复盘**：标准 GRPO 使用**回答级**（Sequence-Level）的损失归一化——每个回答的贡献权重相同，不论长短。但在数学推理场景中，不同回答的长度差异极大（短回答 50 Token，长回答 5000 Token）。如果按回答级归一化，一个 5000 Token 的长回答和一个 50 Token 的短回答对损失的贡献一样大，但长回答中每个 Token 分到的梯度只有短回答的 1/100。

**用例子理解**：

| 回答 | Token 数 | 正确性 | 回答级权重（GRPO） | Token 级权重（DAPO） |
|:---:|:---:|:---:|:---:|:---:|
| $o_1$（简洁解法） | 50 | 正确 | $\frac{1}{2}$ | $\frac{50}{5050} \approx 1\%$ |
| $o_2$（长推导） | 5000 | 正确 | $\frac{1}{2}$ | $\frac{5000}{5050} \approx 99\%$ |

GRPO 给了两个回答相同的权重，这意味着长回答的 5000 个 Token 平均每个只分到 $\frac{1}{2 \times 5000} = 0.0001$ 的梯度——信号太弱了。

**Token-Level Loss 的解决方案**：归一化因子从"回答数"改为"总 Token 数"：

$$
\text{GRPO 归一化因子} = \frac{1}{G} \sum_{i=1}^{G} \left(\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} L_{i,t}\right)
\quad \xrightarrow{\text{DAPO}} \quad
\text{DAPO 归一化因子} = \frac{1}{\sum_{i=1}^{G} |o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} L_{i,t}
$$

其中 $L_{i,t} = \min\big(r_{i,t}\hat{A}_{i,t},\ \text{clip}(r_{i,t},\, 1-\varepsilon_{\text{low}},\, 1+\varepsilon_{\text{high}})\hat{A}_{i,t}\big)$。

**直觉**：GRPO 先求每个回答的"平均 Token 损失"，再求回答间平均——长短回答贡献相同。DAPO 直接对所有 Token 做全局平均——每个 Token 的贡献相同，长回答中的推理步骤得到了应有的梯度信号。

---

## 技术四：Overlong Reward Shaping（超长回答奖励塑形）

**问题复盘**：推理模型有时会陷入"循环推理"（重复验证或反复改写），生成远超必要长度的回答。在纯粹的 $+1/-1$ 二值奖励下，一个 500 Token 的正确解和一个 15000 Token 的正确解获得相同的 $+1$ 奖励——模型没有任何动力去简洁作答，甚至可能越来越啰嗦（因为长回答中"凑巧"碰到正确答案的概率更高）。

**Overlong Reward Shaping 的解决方案**：对超过长度阈值 $L_{\text{max}}$ 的回答施加长度惩罚：

$$
r_{\text{shaped}} = \begin{cases}
r_{\text{original}} & \text{if } |o| \leq L_{\text{max}} - L_{\text{buffer}} \\
\min\left(r_{\text{original}},\ -\frac{|o| - (L_{\text{max}} - L_{\text{buffer}})}{L_{\text{buffer}}} \cdot p\right) & \text{if } |o| > L_{\text{max}} - L_{\text{buffer}}
\end{cases}
$$

其中 $L_{\text{buffer}}$ 是缓冲区长度（论文中设为 4096），$p$ 是每超出一个 Token 的惩罚系数（设为 1.0）。

**用例子理解**（$L_{\text{max}} = 20480$，$L_{\text{buffer}} = 4096$）：

| 回答 | Token 数 | 原始奖励 | 是否超阈值（$20480 - 4096 = 16384$） | 塑形后奖励 |
|:---:|:---:|:---:|:---:|:---:|
| $o_1$（简洁正确） | 500 | $+1$ | 否 | $+1$ |
| $o_2$（长但正确） | 10000 | $+1$ | 否 | $+1$ |
| $o_3$（超长正确） | 18000 | $+1$ | **是**（超出 1616） | $\min(+1, -0.395) = -0.395$ |
| $o_4$（超长错误） | 19000 | $-1$ | **是**（超出 2616） | $\min(-1, -0.639) = -1$ |

**效果**：超长且正确的回答反而被惩罚，迫使模型学习更简洁高效的推理路径。

---

# 完整 DAPO 算法与代码实现

## 算法全貌

将四项技术整合起来，DAPO 的完整训练目标为：

$$
\mathcal{J}_{\text{DAPO}}(\theta) = \mathbb{E}\left[\frac{1}{\sum_{i=1}^{G}|o_i|} \sum_{i=1}^{G}\sum_{t=1}^{|o_i|} \min\Big(r_{i,t}(\theta)\hat{A}_{i,t},\ \text{clip}\big(r_{i,t}(\theta),\, 1-\varepsilon_{\text{low}},\, 1+\varepsilon_{\text{high}}\big)\hat{A}_{i,t}\Big)\right]
$$

$$
\text{s.t.} \quad 0 < |\{o_i \mid \text{is\_correct}(a, o_i)\}| < G
$$

其中优势函数仍然使用 GRPO 的组内相对计算：$\hat{A}_i = \frac{r_i - \mu_R}{\sigma_R + \epsilon}$，不依赖 Critic 网络。

## DAPO 与 GRPO 的差异对比

| 特性 | GRPO | DAPO |
|:---|:---:|:---:|
| 裁剪方式 | 对称 $[1-\varepsilon, 1+\varepsilon]$ | **非对称** $[1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}}]$ |
| 采样策略 | 固定 batch，包含零方差组 | **动态采样**，过滤零方差组 |
| 损失归一化 | 回答级（每个回答等权） | **Token 级**（每个 Token 等权） |
| 长度控制 | 无 | **超长奖励惩罚** |
| KL 正则 | $\beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$ | **移除 KL**（依赖裁剪约束策略） |

注意 DAPO **移除了 KL 散度正则项**——通过 Clip-Higher 和动态采样的协同作用，策略已经被有效约束在合理范围内，不再需要显式的参考模型约束。

## 核心代码实现

```python
import torch
import torch.nn.functional as F

def dapo_loss(
    log_probs,          # 当前策略 log π_θ(o_t|q), shape: [B*G, T]
    old_log_probs,      # 采样策略 log π_old(o_t|q), shape: [B*G, T]
    advantages,         # 组内相对优势 Â_i (广播到每个 token), shape: [B*G, T]
    loss_mask,          # 有效 token 掩码 (排除 padding), shape: [B*G, T]
    eps_low=0.2,
    eps_high=0.28,
):
    ratio = torch.exp(log_probs - old_log_probs)
    
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * advantages
    
    token_loss = -torch.min(surr1, surr2)
    
    # Token-Level 归一化：除以所有有效 token 总数
    total_tokens = loss_mask.sum()
    loss = (token_loss * loss_mask).sum() / total_tokens
    
    return loss


def dynamic_sampling(model, dataset, target_size, G, batch_multiplier=3, max_rounds=10):
    """动态采样：只保留 σ_R > 0 的 Prompt 组"""
    cache = []
    
    for round_idx in range(max_rounds):
        prompts = dataset.sample(target_size * batch_multiplier)
        
        for prompt, answer in prompts:
            responses = model.generate(prompt, num_return_sequences=G)
            rewards = judge_responses(responses, answer)
            
            if rewards.std() > 0:
                n_correct = (rewards > 0).sum().item()
                if 0 < n_correct < G:
                    cache.append({
                        "prompt": prompt,
                        "responses": responses,
                        "rewards": rewards,
                    })
        
        if len(cache) >= target_size:
            return cache[:target_size]
    
    raise RuntimeError(f"经过 {max_rounds} 轮仍无法收集 {target_size} 个有效样本")


def overlong_reward_shaping(rewards, response_lengths, max_len=20480, buffer_len=4096, penalty=1.0):
    """超长奖励塑形"""
    threshold = max_len - buffer_len
    shaped = rewards.clone()
    
    for i in range(len(rewards)):
        if response_lengths[i] > threshold:
            excess = response_lengths[i] - threshold
            penalty_value = -(excess / buffer_len) * penalty
            shaped[i] = min(rewards[i].item(), penalty_value)
    
    return shaped
```

---

# DAPO 的训练效果

DAPO 在 Qwen2.5-32B 基座模型上的 AIME 2024 成绩：

| 方法 | AIME 2024 分数 | 训练步数 |
|:---:|:---:|:---:|
| 原始 GRPO | 30 | ~10000 |
| DeepSeek-R1-Zero-Qwen-32B | 47 | ~10000 |
| **DAPO** | **50** | **~5000** |

四项技术的消融实验（Ablation）显示每项技术都有独立贡献，其中 Clip-Higher 和 Dynamic Sampling 对性能提升最为显著。

---

# 更远的视野：2026 年 RL 前沿

DAPO 之后，强化学习领域仍在快速演进：

- **f-GRPO**：将 GRPO 推广到通用 f-散度框架，不局限于 KL 散度，适用于安全对齐等更广泛的任务。
- **2-GRPO**：研究发现仅用 2 个 rollout（而非 16 个）就能保留 GRPO 98.1% 的性能，训练时间降至 21%。
- **GIFT**：融合 GRPO 的在线采样和 DPO 的隐式奖励，将优化转化为稳定的 MSE 损失。
- **SuperFlow**：将 DAPO 类似的思想引入图像生成，使用方差感知采样和步级优势，在 SD3.5 上取得 4.6%-47.2% 的性能提升。
- **Flow-Factory / GenRL**：统一的图像/视频生成 RL 框架，支持 T2I、T2V、I2V 多种模态。
- **TRL v1.0**（Hugging Face, 2026.04）：生产级 RL 框架，统一 SFT → Reward Modeling → Alignment（DPO/GRPO/KTO）流水线。

强化学习已经从一个"理论优美但工程复杂"的技术，演变为大模型训练不可或缺的核心环节。从 REINFORCE 的简单直觉，到 PPO 的步长控制，到 GRPO 的去 Critic 化，再到 DAPO 的工程最佳实践——每一步都在让 RL 变得更简单、更高效、更可规模化。

> 参考资料：
> 
> 1. [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
> 2. [NVIDIA NeMo RL DAPO Guide](https://docs.nvidia.com/nemo/rl/latest/guides/dapo.html)
> 3. [Hugging Face TRL v1.0](https://www.marktechpost.com/2026/04/01/hugging-face-releases-trl-v1-0-a-unified-post-training-stack-for-sft-reward-modeling-dpo-and-grpo-workflows/)
> 4. [f-GRPO: Divergence-Based RL for General LLM Alignment](https://arxiv.org/abs/2602.05946)

> 下一篇：[笔记｜生成模型（二十二）：GRPO 的三重面孔——从 2-GRPO 到 f-GRPO 与 GIFT](/chengYi-xun/posts/23-grpo-variants/)
