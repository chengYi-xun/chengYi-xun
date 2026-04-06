---
title: 笔记｜生成模型（二十）：Flow-GRPO 与图像生成应用（基于 Flux 的代码解析）
date: 2025-08-20 10:00:00
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

> 本文为 RL 系列第五篇。在完整梳理了从 REINFORCE 到 PPO、DPO，再到最新 GRPO 的演进路线后，我们将目光转向图像生成领域。本文将结合 `flow_grpo` 开源代码库，深入解析如何将 GRPO 算法应用于基于 Flow Matching 的图像生成模型（如 Flux）的微调中。

# 图像生成中的强化学习

**先用一个例子理解为什么需要 RL。**

假设你用一个 Flux 模型生成图像，给定 Prompt："一只橘猫坐在蓝色沙发上"。模型可能生成以下几种结果：

| 生成结果 | 问题 |
|:---|:---|
| 一只白色猫坐在蓝色沙发上 | 颜色不对（应该是橘猫） |
| 一只橘猫站在蓝色沙发旁边 | 动作不对（应该是"坐在"） |
| 一只橘猫坐在蓝色沙发上，画面清晰 | 完美 |
| 一只橘猫坐在蓝色沙发上，但画面模糊 | 质量差 |

传统的训练方式（Flow Matching 损失）只是让模型学会"生成看起来像训练集的图像"。但训练集里可能有模糊的、构图差的、与 Prompt 不一致的图像——模型无法区分好坏。

**RL 的价值**：我们训练一个"美术老师"（奖励模型，如 PickScore 或 ImageReward）来给图像打分。模型自己生成图像 → 美术老师打分 → 模型根据分数调整自己。这就是 RLHF 在图像生成中的应用。

---

# Flow-GRPO 框架解析：图像版"矮子里拔高个"

**先看例子**：对于 Prompt "一只橘猫坐在蓝色沙发上"，我们让 Flux 模型生成 $G = 4$ 张图像，美术老师分别打分：

| 图像 | 描述 | 奖励 $r_i$ | 相对优势 $\hat{A}_i$ |
|:---:|:---|:---:|:---:|
| 图 1 | 橘猫坐沙发，画面清晰 | $r_1 = 0.9$ | $+1.27$ |
| 图 2 | 橘猫坐沙发，稍微模糊 | $r_2 = 0.6$ | $-0.12$ |
| 图 3 | 白猫坐沙发（颜色错） | $r_3 = 0.3$ | $-1.50$ |
| 图 4 | 橘猫坐沙发，普通水平 | $r_4 = 0.7$ | $+0.35$ |

（均值 $\mu_R = 0.625$，标准差 $\sigma_R \approx 0.22$）

跟上一篇 GRPO 的做法完全一样：图 1 和图 4 高于平均（正优势），模型学习生成更像它们的图；图 3 远低于平均（负优势），模型学习远离这种生成方式。**不需要 Critic 网络，只需要多生成几张图做对比。**

**核心思考出发点**：由于像 Flux 这样的图像生成模型参数量达到百亿级别，传统的 PPO 算法由于需要额外的 Critic 网络，显存根本无法承受。因此，Flow-GRPO 采用了 GRPO 算法——彻底抛弃 Critic，用"组内相对评分"来实现高效的在线强化学习。

## 核心挑战：连续时间步的动作空间

在 LLM 中，动作（Action）是离散的词表（Token），计算 $\log \pi_\theta(a|s)$ 很直接。而在 Flow Matching 中，生成过程是一个连续的常微分方程（ODE）求解过程。

**用例子理解**：LLM 生成文本就像逐字写作——每个字是一个离散的"动作"。而 Flux 生成图像像是画画——每个时间步的"动作"是在画布上做一次连续的涂抹（从噪声图向清晰图的一步变换）。

**如何定义图像生成中的 Action？**
Flow-GRPO 将**整个去噪轨迹**视为一个宏观的 Action。在计算对数概率 $\log \pi_\theta(a|s)$ 时，将其分解为每个时间步 $t$ 的转移概率 $\log p_\theta(x_{t-\Delta t} | x_t)$ 的累加。

---

# Flux 模型的 GRPO 训练流程

结合 `flow_grpo` 仓库中的代码，Flux 模型的 GRPO 训练流程可以用橘猫的例子串起来：

## 1. 奖励模型设计 (Reward Composition)

在 `flow_grpo/rewards.py` 中，框架支持组合多种奖励函数：

- **Aesthetic Score**：美学评分（画面构图、色彩是否和谐）
- **ImageReward / PickScore**：综合评分（是否符合人类偏好）
- **CLIP Score**：图文匹配度（图像是否忠实于 Prompt "一只橘猫坐在蓝色沙发上"）

在我们的例子中，图 3（白猫）的 CLIP Score 会很低（颜色不匹配），所以总奖励最低。

## 2. 组内采样与优势计算 (Group Sampling & Advantage)

对于每个 Prompt，模型并行生成 $G$ 张图像（通常 $G=4$ 或 $8$），然后用我们在上一篇推导的公式计算相对优势 $\hat{A}_i = \frac{r_i - \mu_R}{\sigma_R + \epsilon}$。

## 3. 损失计算与梯度反传 (Loss Computation)

这是最核心的部分。在代码实现中（参考 `flow_grpo/diffusers_patch/`）：

- 模型不仅输出生成的图像，还会记录生成轨迹中每一步的隐变量 $x_t$ 和模型预测的速度场 $v_\theta$。
- 通过计算 $\log p_\theta(x_{t-\Delta t} | x_t)$，累加得到整条轨迹的对数概率 $\log P_\theta$。
- 参考模型（冻结的预训练模型）同样计算这条轨迹的对数概率 $\log P_{\text{ref}}$。

**GRPO 损失函数代码级映射**：
$$
\mathcal{L} = - \frac{1}{G} \sum_{i=1}^G \left( \exp(\log P_\theta^{(i)} - \log P_{\theta_{\text{old}}}^{(i)}) \cdot \hat{A}_i - \beta (\log P_{\text{ref}}^{(i)} - \log P_\theta^{(i)}) \right)
$$
*(注：实际代码中会包含 PPO 的 Clip 操作，此处为简化表达)*

在我们的橘猫例子中：图 1（清晰橘猫，$\hat{A} = +1.27$）的正向梯度会推动模型在类似 Prompt 下生成更清晰、颜色更准确的图像；图 3（白猫，$\hat{A} = -1.50$）的负向梯度会抑制模型产生颜色错误的输出。

---

# 代码级解析：深入 `train_flux.py`

```python
# 1. 初始化模型
actor_model = FluxPipeline(...) # 需要微调的模型 (通常加 LoRA)
ref_model = FluxPipeline(...)   # 冻结的参考模型
reward_model = PickScore(...)   # 奖励模型

for batch in dataloader:
    prompts = batch["prompts"] # [Batch_size]
    
    # 2. 组内采样 (Group Sampling)
    duplicated_prompts = duplicate(prompts, G)
    
    with torch.no_grad():
        # actor_model 生成图像，并返回轨迹的 log_probs
        images, old_log_probs, trajectories = actor_model.generate_with_logprob(duplicated_prompts)
        
        # 计算奖励（美术老师给每张图打分）
        rewards = reward_model(images, duplicated_prompts)
        
        # 计算参考模型的 log_probs
        ref_log_probs = ref_model.compute_logprob(trajectories, duplicated_prompts)
    
    # 3. 计算相对优势 (Advantage) —— GRPO 的核心
    # rewards 形状为 [Batch_size, G]
    mean_rewards = rewards.mean(dim=1, keepdim=True)
    std_rewards = rewards.std(dim=1, keepdim=True)
    advantages = (rewards - mean_rewards) / (std_rewards + 1e-8)
    
    # 4. 策略更新 (Policy Update)
    for _ in range(ppo_epochs):
        current_log_probs = actor_model.compute_logprob(trajectories, duplicated_prompts)
        
        # 重要性采样比率
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # 裁剪目标（同 PPO）
        surr1 = ratio * advantages.view(-1)
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages.view(-1)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL 散度惩罚（此处使用一阶近似 E[log(π_θ/π_ref)]，
        # 与上一篇 GRPO 中的闭合形式 e^(log π_ref - log π_θ) - (log π_ref - log π_θ) - 1 等价于低阶展开，
        # 在连续动作空间中计算更稳定）
        kl_loss = (current_log_probs - ref_log_probs).mean()
        
        # 总损失
        loss = policy_loss + beta * kl_loss
        
        loss.backward()
        optimizer.step()
```

## Flow-GRPO-Fast 的优化

由于生成 $G$ 张图像需要进行几十步的 ODE 求解，采样过程极其耗时。`flow_grpo` 提出了 **Flow-GRPO-Fast** 变体。

**用例子理解**：生成一张 1024x1024 的图像，Flux 默认需要 50 步去噪。生成 4 张就是 200 步。Flow-GRPO-Fast 的做法是：不从纯噪声 $t=1$ 开始，而是从中间时间步 $t_{\text{start}}$ 开始（比如 $t=0.5$），或者减少每张图的采样步数。这极大加速了组内采样的速度。

**开源代码参考与算法对比：**
在图像生成的 RLHF 领域，之前主要使用 DPO（如 `Diffusion-DPO`）或 PPO（如 `DDPO`）。
- **Diffusion-DPO** 只能进行离线学习，无法在训练中探索新的图像空间。
- **DDPO** 使用了 PPO，但需要维护 Critic 网络，对于 Flux 这种 12B 的模型几乎不可能单卡运行。
- **Flow-GRPO** 完美结合了 Flow Matching 和 GRPO，彻底抛弃了 Critic，使得百亿参数图像大模型的在线 RL 成为可能。

---

# 系列总结

通过这五篇文章，我们从最基础的强化学习与策略梯度出发，推导了解决步长控制的 PPO 算法，探讨了绕开 RL 的 DPO 路线，最终迎来了解决大模型显存危机的 GRPO 算法，并成功将其落地到了最前沿的 Flow-GRPO 图像生成微调框架中。

强化学习与生成模型的结合，正在开启 AI 领域的新纪元。无论是语言模型中的深度思考（DeepSeek-R1），还是图像生成中的美学对齐（Flow-GRPO），在线强化学习都展现出了无与伦比的潜力。

> 下一篇：[笔记｜生成模型（二十一）：DAPO：从 GRPO 到大规模推理 RL 的工程实践](/chengYi-xun/posts/22-dapo/)
