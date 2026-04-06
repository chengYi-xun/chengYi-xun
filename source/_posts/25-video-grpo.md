---
title: 笔记｜生成模型（二十四）：DanceGRPO——视频生成的统一强化学习框架
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

> 在前面的系列中，我们将 GRPO 从语言模型（第 19 篇）推广到了图像生成（第 20、23 篇 Flow-GRPO / SuperFlow）。本文将目光转向**视频生成**——一个更具挑战性的领域。我们以 DanceGRPO 框架为核心，解析如何将 GRPO 统一应用于文生图（T2I）、文生视频（T2V）和图生视频（I2V）三种任务。
>
> 论文：[DanceGRPO: Unleashing GRPO on Visual Generation](https://arxiv.org/abs/2505.07818)（2025.05）
> 代码：[github.com/XueZeyue/DanceGRPO](https://github.com/XueZeyue/DanceGRPO)

# 从图像到视频：新的挑战

**先用一个例子理解为什么视频比图像更难。** 在 Flow-GRPO 中，我们对 Prompt "一只橘猫坐在蓝色沙发上" 生成 4 张图像，用奖励模型打分，计算优势，更新策略——整个流程清晰明了。

但如果 Prompt 变成 "一只橘猫跳上蓝色沙发并打了个哈欠"，事情就复杂了：

| 维度 | 图像生成 | 视频生成 |
|:---:|:---|:---|
| 输出 | 1 帧（$1024 \times 1024 \times 3$） | $N$ 帧（如 $49 \times 480 \times 720 \times 3$） |
| 计算量 | 50 步 ODE × 1 帧 | 50 步 ODE × $N$ 帧 |
| 奖励评估 | 美学 + 图文匹配 | 视觉质量(VQ) + 运动质量(MQ) + 文本对齐(TA) |
| 显存 | ~10GB（Flux） | ~40-80GB（HunyuanVideo） |
| 信用分配 | 某步去噪决定了"猫的颜色对不对" | 某步去噪同时影响"猫跳得流不流畅"和"沙发纹理清不清晰" |

**三大新挑战**：

1. **多维奖励**：一个视频的好坏不是单一维度的——画面可以很清晰（VQ 高）但动作很僵硬（MQ 低），或者动作流畅但偏离了 Prompt（TA 低）。
2. **计算代价**：生成一个 49 帧的视频比生成一张图像慢 $10\times$ 以上，组内采样 $G = 4$ 意味着 4 个视频的推理。
3. **稳定性**：之前的视频 RL 方法（如 DDPO、DPOK）在大规模 Prompt 集上容易崩溃——策略一旦偏离，视频质量会急剧下降且难以恢复。

---

# DanceGRPO 的 MDP 建模

DanceGRPO 将视频生成的去噪过程建模为 MDP，与 Flow-GRPO 类似但针对视频做了扩展：

## 状态、动作与策略

$$
\text{状态: } s_t = (c,\, t,\, z_t), \quad \text{动作: } a_t = z_{t-1}, \quad \text{策略: } \pi_\theta(a_t | s_t) = p_\theta(z_{t-1} | z_t, c)
$$

- **状态** $s_t$：条件信息 $c$（文本嵌入 + 可选的首帧图像）、时间步 $t$、当前噪声隐变量 $z_t$
- **动作** $a_t$：下一步的隐变量 $z_{t-1}$
- **策略** $\pi_\theta$：由去噪模型（Transformer）参数化

**关键区别**：与 LLM 中的离散 Token 动作不同，视频生成的动作空间是**连续的高维高斯分布**。

## 连续动作空间的对数概率

DanceGRPO 使用反向 SDE 采样器。在每一步，策略输出一个均值 $\mu_t$（由模型预测），然后加上高斯噪声：

$$
z_{t-1} = \mu_t + \sigma_t \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

对应的对数概率密度为：

$$
\log \pi_\theta(z_{t-1} | z_t, c) = -\frac{\|z_{t-1} - \mu_t\|^2}{2\sigma_t^2} - \frac{d}{2}\log(2\pi\sigma_t^2)
$$

其中 $d$ 是隐变量的维数。在代码实现中，对所有非 batch 维度取平均：

```python
def flux_step(model_output, latents, eta, sigmas, index, prev_sample, grpo, sde_solver):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output
    
    delta_t = sigmas[index + 1] - sigmas[index]
    std_dev_t = eta * torch.sqrt(torch.abs(delta_t))
    
    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t
    
    if grpo:
        log_prob = (
            -((prev_sample.detach().float() - prev_sample_mean.float()) ** 2) / (2 * std_dev_t**2)
            - math.log(std_dev_t)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
```

**与 LLM GRPO 的对比**：

| | LLM GRPO | DanceGRPO |
|---|---|---|
| 动作空间 | 离散词表，$\log \pi = \log \text{softmax}$ | 连续隐变量，$\log \pi = \text{高斯对数密度}$ |
| 单步对数概率 | 标量（一个 Token 的概率） | 向量平均（所有维度的平均对数密度） |
| 轨迹对数概率 | $\sum_t \log \pi(o_t)$（Token 序列） | $\sum_t \log \pi(z_{t-1} | z_t)$（去噪步骤序列） |

## 奖励结构：仅终端奖励

DanceGRPO 的奖励**只在轨迹末端**（去噪完成后）分配：

$$
R(s_t, a_t) = \begin{cases} r(z_0, c) & \text{if } t = 0 \text{（最终步）} \\ 0 & \text{otherwise} \end{cases}
$$

最终图像/视频 $x = \text{VAE\_decode}(z_0)$ 被送入冻结的奖励模型评分。这意味着**信用分配是稀疏的**——所有去噪步骤共享同一个轨迹级优势。

---

# 多维视频奖励：VideoAlign

图像奖励模型（HPS、PickScore）输出单一分数。但视频的质量是多维的。DanceGRPO 使用 **VideoAlign** 奖励模型，基于视觉语言模型（VLM），输出三个独立维度的评分：

| 维度 | 含义 | 评估内容 |
|:---:|:---|:---|
| **VQ**（Visual Quality） | 视觉质量 | 画面清晰度、色彩、细节 |
| **MQ**（Motion Quality） | 运动质量 | 动作流畅性、物理合理性 |
| **TA**（Text Alignment） | 文本对齐 | 视频内容是否符合 Prompt |

**用例子理解**：对于 Prompt "一只橘猫跳上蓝色沙发并打了个哈欠"：

| 视频 | VQ | MQ | TA | 总奖励 | 问题 |
|:---:|:---:|:---:|:---:|:---:|:---|
| 视频 1 | 0.9 | 0.8 | 0.9 | **2.6** | 质量优秀 |
| 视频 2 | 0.8 | 0.2 | 0.7 | 1.7 | 猫跳得很僵硬 |
| 视频 3 | 0.7 | 0.7 | 0.3 | 1.7 | 猫没打哈欠 |
| 视频 4 | 0.3 | 0.5 | 0.6 | 1.4 | 画面模糊 |

**多维优势的处理**：在 HunyuanVideo 的训练中，DanceGRPO 对 VQ 和 MQ 分别计算优势，然后用加权系数组合：

```python
# 分别计算 VQ 和 MQ 的组内优势
vq_advantages = group_normalize(vq_rewards)  # 视觉质量优势
mq_advantages = group_normalize(mq_rewards)  # 运动质量优势

# 加权组合为总损失
total_loss = vq_coef * ppo_loss(vq_advantages) + mq_coef * ppo_loss(mq_advantages)
```

这样可以独立调节模型对"画面清晰度"和"动作流畅度"的学习力度。

---

# DanceGRPO 的训练流程

## 完整流程（以 HunyuanVideo T2V 为例）

```
1. 采样阶段（eval 模式, no_grad）
   ├─ 对每个 Prompt 生成 G 个视频
   │   ├─ 从纯噪声 z_T 开始
   │   ├─ 运行 T 步反向 SDE（收集每步的 z_t, log_prob_t）
   │   └─ VAE 解码得到视频 → 保存为 MP4
   ├─ 用 VideoAlign 奖励模型对 G 个视频打分
   └─ 计算组内相对优势 Â_i

2. 训练阶段（train 模式）
   ├─ 从 T 个时间步中随机采样 train_timesteps 个（timestep_fraction）
   ├─ 对每个采样的时间步：
   │   ├─ 用当前策略重新计算 log π_θ(z_{t-1}|z_t, c)（使用存储的 z_t 和 z_{t-1}）
   │   ├─ 计算重要性比率 ρ = exp(log π_new - log π_old)
   │   └─ 计算裁剪策略梯度损失
   └─ 梯度累积 + 参数更新
```

## GRPO 裁剪损失（与 LLM 版本相同的数学结构）

$$
\mathcal{J}_{\text{DanceGRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T}\sum_{t=1}^{T} \min\left(\rho_{i,t}\hat{A}_i,\, \text{clip}(\rho_{i,t},\, 1-\varepsilon,\, 1+\varepsilon)\hat{A}_i\right)\right]
$$

其中 $\rho_{i,t} = \frac{\pi_\theta(z_{t-1}^{(i)} | z_t^{(i)}, c)}{\pi_{\theta_{\text{old}}}(z_{t-1}^{(i)} | z_t^{(i)}, c)}$，$\hat{A}_i = \frac{r_i - \mu_R}{\sigma_R}$ 是组内相对优势。

**注意**：优势 $\hat{A}_i$ 对轨迹 $i$ 的所有时间步共享——这是"终端奖励 + 稀疏信用分配"的直接体现。

## 核心代码

```python
# 训练循环核心（来自 DanceGRPO/fastvideo/train_grpo_flux.py）

# 1. 组内优势计算
if args.use_group:
    for i in range(num_prompts):
        start = i * args.num_generations
        end = (i + 1) * args.num_generations
        group_rewards = samples["rewards"][start:end]
        group_mean = group_rewards.mean()
        group_std = group_rewards.std() + 1e-8
        advantages[start:end] = (group_rewards - group_mean) / group_std

# 2. 裁剪策略梯度损失
train_timesteps = int(len(samples["timesteps"][0]) * args.timestep_fraction)
for sample in samples_batched:
    for step in range(train_timesteps):
        new_log_probs = grpo_one_step(transformer, sample, step)
        
        advantages_clipped = torch.clamp(
            sample["advantages"], -args.adv_clip_max, args.adv_clip_max
        )
        ratio = torch.exp(new_log_probs - sample["log_probs"][:, step])
        
        unclipped_loss = -advantages_clipped * ratio
        clipped_loss = -advantages_clipped * torch.clamp(
            ratio, 1.0 - args.clip_range, 1.0 + args.clip_range
        )
        loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
        loss.backward()
```

---

# 跨模态统一：T2I / T2V / I2V

DanceGRPO 的统一性体现在**同一套 GRPO 框架**适用于三种模态，仅在**条件输入**和**奖励模型**上有所区别：

| 模态 | 条件 $c$ | 去噪对象 $z_t$ | 奖励模型 | 代表模型 |
|:---:|:---|:---|:---|:---:|
| T2I | 文本嵌入 | 2D 隐变量 $(B,C,H,W)$ | HPS v2.1 / PickScore | FLUX, SD |
| T2V | 文本嵌入 | 3D 隐变量 $(B,C,F,H,W)$ | VideoAlign (VQ+MQ+TA) | HunyuanVideo |
| I2V | 文本 + 首帧图像 | 3D 隐变量（拼接首帧） | VideoAlign (MQ) | SkyReels |

**I2V 的特殊处理**：图生视频中，首帧图像通过 VAE 编码为隐变量，与视频噪声隐变量沿通道维度拼接，作为条件注入去噪网络。奖励通常只关注运动质量（MQ），因为视觉质量已经由首帧锚定。

---

# 实验结果

DanceGRPO 在多个基准上超越此前的视觉 RL 方法（如 DDPO、DPOK）：

| 任务 | 基线方法 | DanceGRPO 提升 |
|:---:|:---|:---:|
| T2I (HPS v2.1) | DDPO, DPOK | **+181%** |
| T2I (GenEval) | 基础模型 | 显著提升 |
| T2V (VideoAlign) | 无 RL 基线 | 首次实现稳定的视频 RL |
| I2V (Motion) | 无 RL 基线 | 运动质量提升 |

**关键发现**：

1. 之前的 DDPO/DPOK 在大规模 Prompt 集上训练容易崩溃，而 DanceGRPO 的组内相对归一化有效稳定了优化。
2. DanceGRPO 训练的模型在 Best-of-N 推理扩展中表现更好——生成 $N$ 个候选并选最佳的策略，RL 训练过的模型能生成更多样的高质量候选。
3. 即使在**稀疏二值奖励**（对/错）下也能有效学习，这对规则判题的数学场景尤其重要。

---

# 视频 GRPO 的前沿进展

DanceGRPO 之后，视频生成 RL 仍在快速发展：

**Self-Paced GRPO**（2025.11）：解决静态奖励模型在训练过程中饱和的问题。随着生成质量提升，自动将奖励重心从视觉保真度转移到时间一致性和文本对齐。

**InfLVG**（2025.05）：使用 GRPO 优化一个可学习的上下文选择策略，实现**长视频**（9× 更长）的一致性生成。不累积完整生成历史，而是动态排序选择最相关的上下文帧。

**TAGRPO**（2026.01）：将 GRPO 与对比学习结合，通过直接轨迹对齐实现更快的收敛。在 Wan 2.2 和 HunyuanVideo-1.5 上取得了优于标准 GRPO 的结果。

**TeleBoost**（2026.02）：提出三阶段后训练流水线——监督策略塑形 → 奖励驱动 RL → 偏好优化，系统性地处理视频后训练中的高 rollout 成本、时间维度的级联失败和异构反馈信号。

---

# 视频 GRPO vs 图像 GRPO：关键差异总结

| 维度 | 图像 GRPO（Flow-GRPO / SuperFlow） | 视频 GRPO（DanceGRPO） |
|:---|:---|:---|
| 隐变量 | 2D $(B,C,H,W)$ | 3D $(B,C,F,H,W)$，$F$ 为帧数 |
| 奖励维度 | 单一（美学/对齐） | **多维**（VQ + MQ + TA） |
| 计算代价 | $O(H \times W \times T)$ | $O(F \times H \times W \times T)$，多 $F$ 倍 |
| 信用分配 | 终端奖励，所有步共享 | 同上，但**更稀疏**（时间+空间） |
| 关键超参 | $\varepsilon$, $G$, $\eta$ | + $\text{vq\_coef}$, $\text{mq\_coef}$ |
| 稳定性挑战 | 中等 | **更高**（高维动作空间） |

> 参考资料：
>
> 1. [DanceGRPO: Unleashing GRPO on Visual Generation](https://arxiv.org/abs/2505.07818)
> 2. [Self-Paced GRPO for Video Generation](https://arxiv.org/abs/2511.19356)
> 3. [InfLVG: Reinforce Inference-Time Consistent Long Video Generation with GRPO](https://arxiv.org/abs/2505.17574)
> 4. [TAGRPO: Boosting GRPO on Image-to-Video Generation with Direct Trajectory Alignment](https://arxiv.org/abs/2601.05729)
