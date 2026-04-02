---
title: 笔记｜生成模型（二十）：Flow-GRPO 与图像生成应用（基于 Flux 的代码解析）
date: 2026-04-03 10:20:00
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

> 本文为系列最终篇。在完整梳理了从 REINFORCE 到 PPO、DPO，再到最新 GRPO 的演进路线后，我们将目光转向图像生成领域。本文将结合 `flow_grpo` 开源代码库，深入解析如何将 GRPO 算法应用于基于 Flow Matching 的图像生成模型（如 Flux）的微调中。

# 图像生成中的强化学习（为什么不用监督学习？）

在传统的文本到图像（Text-to-Image, T2I）模型训练中，我们通常使用极大似然估计（MLE）或其变体（如 DDPM 的去噪得分匹配，Flow Matching 的向量场匹配）来拟合数据分布。

然而，这种基于数据的训练方式存在一个致命缺陷：**模型只是在机械地模仿训练集，而不知道人类真正喜欢什么样的图像。** 比如，训练集里可能有很多模糊、构图杂乱、甚至带有水印的图片。模型学会了生成这些图片，但这并不是我们想要的。

为了让模型“懂审美”、“对齐人类偏好”，研究者们引入了 **RLHF（基于人类反馈的强化学习）**。其基本流程如下：
1. **定义奖励模型（Reward Model）**：训练一个“美术老师”（判别模型，如 ImageReward, PickScore, Aesthetic Score），输入 Prompt 和生成的图像，输出一个标量分数，分数越高代表图像越符合人类偏好。
2. **强化学习微调**：将 T2I 模型视为策略网络（Actor，即画画的学生），Prompt 视为状态（State，即考题），生成的图像（或去噪轨迹）视为动作（Action，即答卷）。利用 RL 算法最大化“美术老师”给出的分数。

---

# Flow-GRPO 框架解析：图像版“矮子里拔高个”

**Flow-GRPO** 是一种专门为基于 Flow Matching 的生成模型（如 Flux, Stable Diffusion 3）设计的在线强化学习微调框架。

**核心思考出发点**：由于图像生成模型（特别是像 Flux 这样百亿参数级别的模型）极其庞大，传统的 PPO 算法由于需要额外的 Critic 网络，显存开销根本无法承受。因此，Flow-GRPO 采用了我们在上一篇文章中推导的 **GRPO** 算法——彻底抛弃 Critic，用“组内相对评分”（矮子里拔高个）来实现高效的在线强化学习。

## 核心挑战：连续时间步的动作空间

在 LLM 中，动作（Action）是离散的词表（Token）。而在扩散模型/Flow Matching 中，生成过程是一个连续的常微分方程（ODE）或随机微分方程（SDE）求解过程。

**如何定义图像生成中的 Action？**
Flow-GRPO 将**整个去噪轨迹（或向量场积分轨迹）**视为一个宏观的 Action。但在计算对数概率 $\log \pi_\theta(a|s)$ 时，我们需要将其分解为每个时间步 $t$ 的转移概率。

对于 Flow Matching，模型预测的是速度场 $v_\theta(x_t, t)$。在添加了 SDE 噪声扰动后，我们可以通过 Girsanov 定理或直接计算高斯转移概率，得到模型在每一步的 $\log p_\theta(x_{t-\Delta t} | x_t)$。

---

# Flux 模型的 GRPO 训练流程

结合 `flow_grpo` 仓库中的代码（特别是 `scripts/train_flux.py`），Flux 模型的 GRPO 训练流程可以分为以下几个关键步骤：

## 1. 奖励模型设计 (Reward Composition)

在 `flow_grpo/rewards.py` 中，框架支持组合多种奖励函数：
- **Aesthetic Score**：美学评分，鼓励模型生成更好看的图像。
- **ImageReward / PickScore**：基于人类偏好数据集训练的综合评分模型。
- **CLIP Score / OCR Score**：用于衡量图像与文本 Prompt 的对齐程度（如是否正确拼写了文字）。

对于一个 Prompt，模型生成 $G$ 张图像，奖励函数对这 $G$ 张图像分别打分，得到 $\{r_1, r_2, \dots, r_G\}$。

## 2. 组内采样与优势计算 (Group Sampling & Advantage)

在训练循环中，对于每个 Prompt：
1. 模型并行生成 $G$ 张图像（通常 $G=4$ 或 $8$）。
2. 计算这组图像的奖励均值 $\mu_R$ 和标准差 $\sigma_R$。
3. 计算每张图像的相对优势：$\hat{A}_i = \frac{r_i - \mu_R}{\sigma_R + \epsilon}$。

## 3. 损失计算与梯度反传 (Loss Computation)

这是最核心的部分。为了计算 GRPO 损失，我们需要知道当前策略 $\pi_\theta$ 和旧策略 $\pi_{\theta_{\text{old}}}$ 生成这 $G$ 张图像的概率。

在代码实现中（参考 `flow_grpo/diffusers_patch/` 中的 pipeline 修改）：
- 模型不仅输出生成的图像，还会记录生成轨迹中每一步的隐变量 $x_t$ 和模型预测的速度场 $v_\theta$。
- 通过计算 $\log \pi_\theta(x_{t-\Delta t} | x_t)$，累加得到整条轨迹的对数概率 $\log P_\theta$。
- 参考模型（Reference Model，通常是冻结的预训练模型）同样计算这条轨迹的对数概率 $\log P_{\text{ref}}$。

**GRPO 损失函数代码级映射**：
$$
\mathcal{L} = - \frac{1}{G} \sum_{i=1}^G \left( \exp(\log P_\theta^{(i)} - \log P_{\theta_{\text{old}}}^{(i)}) \cdot \hat{A}_i - \beta (\log P_{\text{ref}}^{(i)} - \log P_\theta^{(i)}) \right)
$$
*(注：实际代码中会包含 PPO 的 Clip 操作，此处为简化表达)*

---

# 代码级解析：深入 `train_flux.py`

让我们看看 `train_flux.py` 中的关键逻辑（伪代码抽象）：

```python
# 1. 初始化模型
actor_model = FluxPipeline(...) # 需要微调的模型 (通常加 LoRA)
ref_model = FluxPipeline(...)   # 冻结的参考模型
reward_model = PickScore(...)   # 奖励模型

for batch in dataloader:
    prompts = batch["prompts"] # [Batch_size]
    
    # 2. 组内采样 (Group Sampling)
    # 将 prompts 复制 G 份：[Batch_size * G]
    duplicated_prompts = duplicate(prompts, G)
    
    with torch.no_grad():
        # actor_model 生成图像，并返回轨迹的 log_probs
        images, old_log_probs, trajectories = actor_model.generate_with_logprob(duplicated_prompts)
        
        # 计算奖励
        rewards = reward_model(images, duplicated_prompts)
        
        # 计算参考模型的 log_probs
        ref_log_probs = ref_model.compute_logprob(trajectories, duplicated_prompts)
    
    # 3. 计算相对优势 (Advantage)
    # rewards 形状为 [Batch_size, G]
    mean_rewards = rewards.mean(dim=1, keepdim=True)
    std_rewards = rewards.std(dim=1, keepdim=True)
    advantages = (rewards - mean_rewards) / (std_rewards + 1e-8)
    
    # 4. 策略更新 (Policy Update)
    for _ in range(ppo_epochs):
        # 重新计算当前模型的 log_probs (因为模型参数在更新)
        current_log_probs = actor_model.compute_logprob(trajectories, duplicated_prompts)
        
        # 重要性采样比率
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # 裁剪目标
        surr1 = ratio * advantages.view(-1)
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages.view(-1)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL 散度惩罚
        kl_loss = (current_log_probs - ref_log_probs).mean()
        
        # 总损失
        loss = policy_loss + beta * kl_loss
        
        # 反向传播与优化
        loss.backward()
        optimizer.step()
```

## Flow-GRPO-Fast 的优化

由于生成 $G$ 张图像需要进行几十步的 ODE 求解，采样过程极其耗时。`flow_grpo` 提出了 **Flow-GRPO-Fast** 变体。
其核心思想是：**不从纯噪声 $t=1$ 开始采样，而是从中间时间步 $t_{start}$ 开始，或者减少采样步数。**
通过截断轨迹或使用一致性模型（Consistency Models）的思路，极大加速了组内采样的速度，使得 GRPO 在图像生成上的训练成本大幅降低。

**开源代码参考与算法对比：**
在图像生成的 RLHF 领域，之前主要使用 DPO（如 `Diffusion-DPO`）或 PPO（如 `DDPO`）。
- **Diffusion-DPO** 只能进行离线学习，无法在训练中探索新的图像空间。
- **DDPO** 使用了 PPO，但需要维护 Critic 网络，对于 Flux 这种 12B 的模型几乎不可能单卡运行。
- **Flow-GRPO** (`flow_grpo` 仓库，如 `scripts/train_flux.py` 和 `scripts/train_flux_fast.py`) 完美结合了 Flow Matching 和 GRPO，彻底抛弃了 Critic，使得百亿参数图像大模型的在线 RL 成为可能。

---

# 系列总结

通过这五篇文章，我们从最基础的强化学习与策略梯度出发，推导了解决步长控制的 PPO 算法，探讨了绕开 RL 的 DPO 路线，最终迎来了解决大模型显存危机的 GRPO 算法，并成功将其落地到了最前沿的 Flow-GRPO 图像生成微调框架中。

强化学习与生成模型的结合，正在开启 AI 领域的新纪元。无论是语言模型中的深度思考（DeepSeek-R1），还是图像生成中的美学对齐（Flow-GRPO），在线强化学习都展现出了无与伦比的潜力。
