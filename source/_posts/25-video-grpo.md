---
title: 笔记｜生成模型（二十四）：DanceGRPO——让视频生成模型"跳好舞"的强化学习框架
date: 2026-04-05 16:00:00
cover: false
mathjax: true
categories:
 - Notes
tags:
 - Deep learning
 - Generative models theory
 - Reinforcement Learning
 - Flow Matching
series: Diffusion Models theory
---

> 在前面的系列中，我们讨论过如何用强化学习（RL）来提升图像生成的质量（比如 Flow-GRPO）。本文我们将目光转向**视频生成**。
> 
> 相比图像，视频多了"时间"这个维度，这让强化学习的难度呈指数级上升。本文将通俗地解释 DanceGRPO 是如何利用 **GRPO（组相对策略优化）** 巧妙解决这些难题，为视频生成提供了一条稳定且可扩展的优化路径。
>
> ⬅️ 上一篇：[笔记｜生成模型（二十三）：SuperFlow 与图像生成 RL 前沿（2026）](/chengYi-xun/posts/24-superflow/)
>
> ➡️ 下一篇：[笔记｜MemoryBank：用艾宾浩斯遗忘曲线赋予 LLM 长期记忆](/chengYi-xun/posts/26-memory-bank/)
>
> 论文：[DanceGRPO: Unleashing GRPO on Visual Generation](https://arxiv.org/abs/2505.07818)（arXiv:2505.07818）
>
> 代码：[github.com/XueZeyue/DanceGRPO](https://github.com/XueZeyue/DanceGRPO)

# 引言：为什么需要 DanceGRPO？

简单来说，我们希望生成的视频既好看又符合提示词。但是，很难用传统的"标准答案"（监督学习）来教模型怎么做。因此，用"强化学习（RL）"——也就是给生成的视频打分，让模型自己摸索如何拿高分——是个非常自然的选择。

DanceGRPO 的核心贡献可以概括为三点：

- **稳且能打（优化稳定性与可扩展性）**：把 GRPO 算法成功搬到了视频生成上，在各种大模型（如 FLUX, HunyuanVideo, Wan-2.1）上都能稳定训练。

- **通用性强（范式覆盖）**：不管是传统的扩散模型（Diffusion），还是最新的流匹配（Flow Matching），这套框架都能无缝接入。

- **多管齐下（任务与奖励的多样性）**：支持文生图（T2I）、文生视频（T2V）、图生视频（I2V）。并且能同时兼顾"画质好"、"动作流畅"、"符合文本"等多种打分标准。

# 图像 RL 与 视频 RL：到底难在哪里？

无论是图像还是视频，生成的逻辑是一样的：都是一步步去噪，最后得到成品，然后给成品打个总分（终端奖励）。

但是，视频生成面临三个被严重放大的困难：

- **（1）动作空间太大（容易迷失）**：图像只是二维的，视频是三维的（加上了时间帧）。这意味着模型每一步要处理的数据量极大。在这么高维的空间里试错，很容易"学崩"（梯度方差极大）。

- **（2）奖励太笼统（不知道哪步做对了）**：视频不仅要画面好看，还要动作连贯。最后只给出一个总分，模型很难搞清楚到底是哪一帧、哪一步去噪做得好或不好。这就好比你考了低分，但老师不告诉你错在哪道题。

- **（3）既要又要还要（算力与多目标）**：视频的评价指标很多（画质、运动、文本对齐等）。而且，生成视频非常耗费算力，如果像图像那样一次生成好几个来对比，显存和时间都会爆炸。

# DanceGRPO 的破局之道

DanceGRPO 并不是凭空捏造了一套全新的算法，而是**继承了 Flow-GRPO 在图像上的成功经验，并针对视频的痛点做了关键升级**。

## 1. 继承自 Flow-GRPO 的"老本行"

- **组相对优势（GRPO）**：不追求绝对的高分，而是在"同一个提示词"生成的几个视频里比个高低。如果比同组的平均分高，就奖励；比平均分低，就惩罚。这完美避开了"有些提示词天生就难拿高分"的问题。

- **引入随机性（ODE → SDE）**：为了让模型能试错，它把原本确定的生成过程（ODE）变成了带点随机性的过程（SDE）。就像是让模型在原本的轨迹上稍微"抖一抖"，看看能不能得到更好的结果，这样才能计算出"采取某个动作的概率"。

## 2. DanceGRPO 针对视频的"新招数"

既然 Flow-GRPO 已经有了上述机制，为什么视频还需要 DanceGRPO？因为视频的维度和评价指标太复杂了：

- **分项打分，各自比较（解决多目标问题）**：视频的评价指标很多。如果直接把"画质分"和"运动分"加起来算一个总分，可能会出现"画质极好但完全不动"的视频因为总分高而骗过模型。DanceGRPO 的做法是：**先分别对"画质（VQ）"和"运动（MQ）"在同组视频里做标准化对比，算出各自的"优势"，然后再把这两个优势对应的 Loss 加起来。** 这样，模型必须在画质和运动上都比同组的平均水平好，才能真正降低 Loss。

- **抽样训练省算力（解决显存爆炸问题）**：
  在扩散/流匹配模型中，生成一个视频需要经过几十步（比如 50 步）的去噪。
  按照标准的强化学习，这 50 步的每一步都要计算 Loss 并反向传播。**在 Flow-GRPO（图像生成）中**，因为图像模型相对较小，有时可以硬扛下所有时间步的反向传播。后来为了加速，Flow-GRPO 团队也推出了 **Flow-GRPO-Fast** 版本，引入了"窗口机制（Window Mechanism）"来只训练部分时间步。
  **在 DanceGRPO（视频生成）中**，由于视频多了"帧数"这个维度，显存需求直接翻了几十倍。如果把 50 步的计算图全存下来，显存会直接爆炸。
  因此，DanceGRPO 的做法与 Flow-GRPO-Fast 的思路异曲同工：**生成视频时（推理阶段）走完完整的 50 步，拿到最终的视频并打分算出优势；但在真正更新模型（训练阶段）时，只从这 50 步里随机抽取一小部分（比如 10 步）来计算 Loss 并反向传播。** 这样既保证了梯度的方向是对的，又极大地节省了显存和算力。

# 数学原理与公式基础

关于**组相对优势（GRPO）的计算公式**、**从 ODE 到 SDE 的随机探索机制（$\log\pi$ 的推导）**，以及 **PPO/GRPO 的裁剪损失函数**，DanceGRPO 在数学底层上完全继承了 Flow-GRPO 的设定。

如果您对这些基础公式的推导感兴趣，请直接回顾我们的[上一篇关于 Flow-GRPO 的笔记](/chengYi-xun/posts/24-superflow/)。

DanceGRPO 在数学层面的**新贡献**，主要体现在对**多维奖励（Multi-dimensional Rewards）**的处理上。假设我们有画质（VQ）和运动（MQ）两个维度的奖励，DanceGRPO 会**分别**套用 GRPO 公式，算出画质优势 $\hat A_i^{(\mathrm{VQ})}$ 和运动优势 $\hat A_i^{(\mathrm{MQ})}$，然后再分别计算 Loss 并加权组合，而不是简单地把原始分数相加。

# 实验效果与总结

DanceGRPO 在多个榜单上（比如 HPS v2.1, CLIP Score, VideoAlign 等）都取得了巨大的提升（最高相对提升约 181%）。它证明了：**只要把"组内比较"和"多维奖励分离"做得足够好，强化学习完全可以驾驭高维的视频生成任务。**

---

# 附录：DanceGRPO 核心训练代码解析（源码精读）

下面的代码直接提取自 [DanceGRPO 官方仓库](https://github.com/XueZeyue/DanceGRPO)，围绕三个核心函数展开讲解。为突出重点，省略了分布式通信、数据加载等工程代码。

## 步骤 1：`flux_step` —— SDE 单步去噪 + 对数概率

> 源码位置：`fastvideo/train_grpo_hunyuan.py` L57-96（Flux / Wan / Qwen 等变体结构一致）

这是 GRPO 训练的"原子操作"——在标准 Flow Matching ODE 步进基础上，注入 SDE 随机项实现"探索"，并计算该动作的对数概率 $\log\pi$。

```python
# ===== 源码：fastvideo/train_grpo_hunyuan.py =====
import math, torch

def flux_step(model_output, latents, eta, sigmas, index, prev_sample, grpo, sde_solver):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma                   # Δσ（负值，σ 从 1→0 递减）
    prev_sample_mean = latents + dsigma * model_output    # ODE 步进：x_{t+1} = x_t + Δσ·v_θ
    pred_original_sample = latents - sigma * model_output # 预测干净样本 x̂₀

    delta_t = sigma - sigmas[index + 1]                   # 正的时间增量
    std_dev_t = eta * math.sqrt(delta_t)                  # SDE 噪声标准差（η 控制探索强度）

    # SDE 修正项：用 score function 修正 ODE 方向
    if sde_solver:
        score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    # 推理/rollout 阶段：prev_sample=None → 显式加噪采样
    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

    # GRPO 核心：计算该动作的高斯对数概率
    if grpo:
        log_prob = (
            -((prev_sample.detach().float() - prev_sample_mean.float()) ** 2)
                / (2 * std_dev_t**2)
            - math.log(std_dev_t)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))  # 空间维取均值→(B,)
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean, pred_original_sample
```

这个函数在训练中被调用**两次**，承担不同角色：

- **推理阶段**（`prev_sample=None`）：模型走完所有去噪步，**采样**出完整视频轨迹，同时记录每一步的 `log_prob_old`。
- **训练阶段**（`prev_sample=已记录的 latent`）：用**当前**模型权重重算同一步的 `log_prob_new`——两者之比 $\exp(\log\pi_\theta - \log\pi_{\text{old}})$ 就是 PPO 的重要性采样比（importance ratio）。

## 步骤 2：多维奖励的分项优势与组合 Loss

> 源码位置：`fastvideo/train_grpo_hunyuan.py` L370-477（`train_one_step` 函数内部）

DanceGRPO 的核心创新：**VQ（画质）和 MQ（运动）各自独立做组内标准化**，避免某个维度主导训练。

```python
# ===== 源码：fastvideo/train_grpo_hunyuan.py (train_one_step 内部) =====

# ── 第一步：VQ / MQ 分别组内标准化 ─────────────────────────────
n = len(vq_rewards) // num_generations           # 提示词组数
vq_advantages = torch.zeros_like(vq_rewards)
mq_advantages = torch.zeros_like(mq_rewards)

for i in range(n):
    s, e = i * num_generations, (i + 1) * num_generations
    group_vq = vq_rewards[s:e]                   # 同一提示词下 G 个视频的画质分
    vq_advantages[s:e] = (group_vq - group_vq.mean()) / (group_vq.std() + 1e-8)

for i in range(n):
    s, e = i * num_generations, (i + 1) * num_generations
    group_mq = mq_rewards[s:e]                   # 同一提示词下 G 个视频的运动分
    mq_advantages[s:e] = (group_mq - group_mq.mean()) / (group_mq.std() + 1e-8)

# ── 第二步：分别计算 PPO clip loss，再加权组合 ──────────────────
ratio = torch.exp(new_log_probs - old_log_probs) # 策略概率比 π_θ/π_old

vq_advantages = torch.clamp(vq_advantages, -adv_clip_max, adv_clip_max)
vq_unclipped_loss = -vq_advantages * ratio
vq_clipped_loss   = -vq_advantages * torch.clamp(ratio, 1-clip_range, 1+clip_range)
vq_loss = torch.mean(torch.maximum(vq_unclipped_loss, vq_clipped_loss))

mq_advantages = torch.clamp(mq_advantages, -adv_clip_max, adv_clip_max)
mq_unclipped_loss = -mq_advantages * ratio
mq_clipped_loss   = -mq_advantages * torch.clamp(ratio, 1-clip_range, 1+clip_range)
mq_loss = torch.mean(torch.maximum(mq_unclipped_loss, mq_clipped_loss))

final_loss = vq_coef * vq_loss + mq_coef * mq_loss  # 加权组合总损失
```

为什么不直接 `vq_coef * vq_reward + mq_coef * mq_reward` 合并后再标准化？因为两个维度的**尺度和分布不同**——VQ 分数通常较大，MQ 较小。如果合并后标准化，画质会淹没运动。分别标准化后，每个维度都在"同组中排第几"的公平尺度上竞争。

## 步骤 3：时间步随机抽样训练

> 源码位置：`fastvideo/train_grpo_hunyuan.py` L411-487（`train_one_step` 函数内部）

生成视频时走完全部 T 步去噪，但**训练时只在随机抽样的部分步上反向传播**——这是 DanceGRPO 解决视频显存爆炸的关键。

```python
# ===== 源码：fastvideo/train_grpo_hunyuan.py (train_one_step 内部) =====

# ── 随机打乱时间步顺序（每个样本独立打乱）────────────────────
perms = torch.stack(
    [torch.randperm(len(samples["timesteps"][0])) for _ in range(batch_size)]
).to(device)
for key in ["timesteps", "latents", "next_latents", "log_probs"]:
    samples[key] = samples[key][torch.arange(batch_size).to(device)[:, None], perms]

# ── 只训练 timestep_fraction 比例的时间步 ──────────────────────
train_timesteps = int(num_total_steps * args.timestep_fraction)

for i, sample in enumerate(samples_batched_list):       # 遍历每个样本
    for t in range(train_timesteps):                     # 只取前 train_timesteps 步
        new_log_probs = grpo_one_step(                   # 用当前模型重算 log_prob
            sample["latents"][:, t],
            sample["next_latents"][:, t],
            ..., transformer, sigma_schedule,
        )
        ratio = torch.exp(new_log_probs - sample["log_probs"][:, t])

        # ... 计算 VQ/MQ 分项 clip loss（同步骤 2）...
        final_loss = vq_coef * vq_loss + mq_coef * mq_loss
        final_loss.backward()                            # 仅对抽样时间步反传

    if (i + 1) % gradient_accumulation_steps == 0:       # 梯度累积后统一更新
        grad_norm = transformer.clip_grad_norm_(max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
```

这里的关键技巧是：先用 `torch.randperm` 打乱全部 T 步的顺序，再只取前 `train_timesteps = T × timestep_fraction` 步——**等效于从 T 步中随机不重复抽样**。由于打乱是每个样本独立进行的，不同样本训练的时间步组合也不同，进一步增加了梯度估计的多样性。

> 参考资料：
>
> 1. Xue, Z., et al. (2025). *DanceGRPO: Unleashing GRPO on Visual Generation*. arXiv:2505.07818.
> 2. Flow-GRPO：Liu, et al. *Flow-GRPO: Training Flow Matching Models via Online RL*，arXiv:2505.05470。
