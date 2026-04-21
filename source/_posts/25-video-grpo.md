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
> ➡️ 下一篇：[笔记｜世界模型（一）：什么是世界模型？从认知科学到深度学习](/chengYi-xun/posts/26-world-model-basics/)
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

# 数学原理与公式基础：从 Rectified Flow 到 SDE 探索

虽然 DanceGRPO 在宏观上继承了 Flow-GRPO 的组相对优势（GRPO）和 PPO 裁剪损失函数（详情可回顾[上一篇笔记](/chengYi-xun/posts/24-superflow/)），但其底层的生成逻辑是基于 **Latent Flow Matching（如 Flux、HunyuanVideo）** 的。

为了让强化学习能够在视频生成中生效，我们需要将确定性的 ODE 轨迹转化为带有随机探索的 SDE 轨迹。以下是这一过程的严谨推导与通俗解释。

## 1. 基础：Rectified Flow 与速度场 (Velocity Field)

在 Rectified Flow 的理论框架中，前向加噪过程被定义为从“干净潜变量 $x_0$”到“纯噪声 $x_1$”的**直线插值（Straight-Line Interpolation）**：

$$x_t = t \cdot x_1 + (1 - t) \cdot x_0 \quad (t \in [0, 1])$$

*(注：在主流代码实现中，时间步 $t$ 通常记为 $\sigma$，且通常从 1 递减到 0。)*

既然轨迹是一条直线，那么根据微积分，这条直线上的**真实速度（导数）** $v(x_t, t)$ 是一个常数向量：

$$v(x_t, t) = \frac{\mathrm{d}x_t}{\mathrm{d}t} = x_1 - x_0$$

神经网络 $v_\theta$ 的训练目标，正是去拟合这个真实速度：$v_\theta \approx x_1 - x_0$。

既然我们知道了当前的位置 $x_t$、已经走过的时间 $t$，以及当前的速度 $v_\theta$，我们就可以利用初中物理的**匀速直线运动公式（起点 = 当前位置 - 速度 × 时间）**，直接反推起点 $x_0$：

$$\hat{x}_0 = x_t - t \cdot v_\theta$$

这就是模型在任意时刻对最终清晰画面的“一步预测”。

## 2. 引入 SDE 与 Score Function 修正偏航

在纯 ODE 采样中，模型就像是沿着一条设定好的轨道平滑地滑向终点。但 DanceGRPO 为了让强化学习能够“试错”和“探索”，引入了 **SDE（随机微分方程）**，也就是在滑行的过程中加入随机的扰动（噪声）。

如果盲目地加入随机噪声，生成的轨迹就会偏离真实视频的流形（Manifold）。因此我们需要一个“指南针”来纠正这种偏离，这个指南针就是 **Score Function（分数函数 $\nabla_{x_t} \log p_t(x_t)$）**。

*(注：关于 SDE 采样的严谨数学推导（包括特威迪公式、Anderson 定理与 Langevin Dynamics），我们在前一篇图像生成 RL 笔记中已经做了极其详细的推导与代码映射，强烈建议回顾：[笔记｜生成模型（二十）：Flow-GRPO 与图像生成应用](/chengYi-xun/posts/21-flow-grpo/)*。

简单来说，预测 $\hat{x}_0$ 的根本目的，是为了通过特威迪公式（Tweedie's Formula）计算出 Score（重力传感器），进而用它来抵消 SDE 随机探索带来的偏航风险，确保模型在“瞎溜达”探索高分动作的同时，依然能被拉回正确的生成轨道上。

## 3. 延伸：均值回归困境与“一步生成”

既然公式可以“一步到位”算出 $\hat{x}_0$，为什么还要一步步去噪？

答案在于**后验分布 $p(x_0 | x_t)$ 的多峰性**。在早期高噪声阶段（如 $t \approx 1$），一张纯噪声图可能对应无数种清晰图像。神经网络在 MSE 损失的约束下，为了让误差最小化，只能输出所有可能图像的**数学期望（平均值）**：$\hat{x}_0 = \mathbb{E}[x_0 | x_t]$。

把无数张不同的清晰图像叠加平均，得到的一定是**极其模糊、缺乏高频细节**的画面。只有随着 $t$ 逐渐变小（逐步去噪），不确定性降低，后验分布逐渐坍缩到单峰，模型预测的 $\hat{x}_0$ 才会变得锐利。

为了克服上述的“均值回归”问题，学术界提出了多种**蒸馏（Distillation）**技术，试图让模型在 $t=1$ 时也能准确预测出清晰的 $x_0$：

- **Consistency Models (一致性模型)**：强制要求模型在同一条 ODE 轨迹上的任意两点，预测出的 $x_0$ 必须完全一致。通过这种一致性约束，模型学会了直接映射到终点。
- **Rectified Flow 的 Reflow (蒸馏重流)**：原始的 Flow 轨迹可能是弯曲和交叉的。Reflow 技术通过使用第一代模型生成的数据对来重新训练模型，将轨迹彻底“拉直”。当轨迹变成完美的互不交叉的直线时，单步 Euler 积分（即一步生成）的截断误差就会降到最低。

## 4. DanceGRPO 的多维奖励创新

除了上述底层生成机制，DanceGRPO 在强化学习层面的**新贡献**，主要体现在对**多维奖励（Multi-dimensional Rewards）**的处理上。假设我们有画质（VQ）和运动（MQ）两个维度的奖励，DanceGRPO 会**分别**套用 GRPO 公式，算出画质优势 $\hat A_i^{(\mathrm{VQ})}$ 和运动优势 $\hat A_i^{(\mathrm{MQ})}$，然后再分别计算 Loss 并加权组合，而不是简单地把原始分数相加。

# 实验效果与总结

DanceGRPO 在多个榜单上（比如 HPS v2.1, CLIP Score, VideoAlign 等）都取得了巨大的提升（最高相对提升约 181%）。它证明了：**只要把"组内比较"和"多维奖励分离"做得足够好，强化学习完全可以驾驭高维的视频生成任务。**

---

# 附录：DanceGRPO 核心训练代码解析（源码精读）

下面的代码直接提取自 [DanceGRPO 官方仓库](https://github.com/XueZeyue/DanceGRPO)，围绕三个核心函数展开讲解。为突出重点，省略了分布式通信、数据加载等工程代码。

## 步骤 1：`flux_step` —— SDE 单步去噪 + 对数概率

> 源码位置：`fastvideo/train_grpo_hunyuan.py` L57-96（Flux / Wan / Qwen 等变体结构一致）

这是 GRPO 训练的"原子操作"——在标准 Flow Matching ODE 步进基础上，注入 SDE 随机项实现"探索"，并计算该动作的对数概率 $\log\pi$。代码中每一行对应的数学公式如下：

**公式 ①：ODE 步进**（Euler 离散化，注意 $\Delta\sigma < 0$）

$$x_{t-\Delta t} = x_t + \Delta\sigma \cdot v_\theta$$

**公式 ②：Tweedie 反推干净样本**

$$\hat{x}_0 = x_t - t \cdot v_\theta$$

**公式 ③：Score Function**（由 Tweedie 公式推得）

$$\nabla_{x_t}\log p(x_t|x_0) = -\frac{x_t - (1-t)\hat{x}_0}{t^2}$$

**公式 ④：Langevin 修正**（将 Score 纠偏叠加到 ODE 均值上）

$$\mu = x_{t-\Delta t}^{\text{ODE}} + \tfrac{1}{2}\eta^2 \cdot \nabla_{x_t}\log p_t \cdot \Delta t$$

**公式 ⑤：SDE 采样**（在修正后的均值上加噪声）

$$x_{t-\Delta t} = \mu + \eta\sqrt{\Delta t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**公式 ⑥：高斯对数概率**（完整形式，含归一化常数）

$$\log p(x_{t-\Delta t}|x_t) = -\frac{\|x_{t-\Delta t} - \mu\|^2}{2(\eta\sqrt{\Delta t})^2} - \log(\eta\sqrt{\Delta t}) - \tfrac{1}{2}\log(2\pi)$$

```python
# ===== 源码：fastvideo/train_grpo_hunyuan.py =====
import math, torch

def flux_step(model_output, latents, eta, sigmas, index, prev_sample, grpo, sde_solver):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma                   # Δσ < 0（σ 从 1→0 递减）

    # 公式 ①  ODE 步进：x_{t-Δt} = x_t + Δσ·v_θ
    prev_sample_mean = latents + dsigma * model_output

    # 公式 ②  Tweedie：x̂₀ = x_t − t·v_θ
    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]                   # 正的时间增量 Δt
    std_dev_t = eta * math.sqrt(delta_t)                  # 噪声标准差 = η√Δt

    # 公式 ③④  Score Function + Langevin 修正
    if sde_solver:
        # 公式 ③  Score = −(x_t − (1−t)·x̂₀) / t²
        score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
        # 公式 ④  μ = ODE结果 + ½η²·Score·Δt
        #   注意：log_term·dsigma = (−½η²·Score)·(−Δt) = +½η²·Score·Δt
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    # 公式 ⑤  SDE 采样：x_{t-Δt} = μ + η√Δt · ε
    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

    # 公式 ⑥  高斯对数概率：log p = −‖x−μ‖²/(2σ²) − log σ − ½log(2π)
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

DanceGRPO 的核心创新：**VQ（画质）和 MQ（运动）各自独立做组内标准化**，避免某个维度主导训练。代码对应的公式如下：

**公式 ⑦：分维度组内标准化（GRPO 优势）**

$$\hat{A}_i^{(d)} = \frac{r_i^{(d)} - \bar{r}_{\text{group}}^{(d)}}{\sigma_{\text{group}}^{(d)} + \epsilon}, \quad d \in \{\text{VQ}, \text{MQ}\}$$

**公式 ⑧：重要性采样比**

$$\rho = \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} = \exp\left(\log\pi_\theta - \log\pi_{\text{old}}\right)$$

**公式 ⑨：PPO Clip Loss（分维度独立计算）**

$$L^{(d)} = \mathbb{E}\left[\max\left(-\hat{A}^{(d)} \cdot \rho, \;\; -\hat{A}^{(d)} \cdot \text{clip}(\rho, 1{-}\epsilon, 1{+}\epsilon)\right)\right]$$

**公式 ⑩：加权组合总损失**

$$L = \alpha_{\text{VQ}} \cdot L^{(\text{VQ})} + \alpha_{\text{MQ}} \cdot L^{(\text{MQ})}$$

```python
# ===== 源码：fastvideo/train_grpo_hunyuan.py (train_one_step 内部) =====

# ── 公式 ⑦：VQ / MQ 分别组内标准化 ────────────────────────────
#    Â_i^(d) = (r_i^(d) − r̄_group^(d)) / (σ_group^(d) + ε)
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

# ── 公式 ⑧：重要性采样比 ρ = π_θ / π_old ─────────────────────
ratio = torch.exp(new_log_probs - old_log_probs)

# ── 公式 ⑨：PPO clip loss（VQ 维度）──────────────────────────
vq_advantages = torch.clamp(vq_advantages, -adv_clip_max, adv_clip_max)
vq_unclipped_loss = -vq_advantages * ratio
vq_clipped_loss   = -vq_advantages * torch.clamp(ratio, 1-clip_range, 1+clip_range)
vq_loss = torch.mean(torch.maximum(vq_unclipped_loss, vq_clipped_loss))

# ── 公式 ⑨：PPO clip loss（MQ 维度）──────────────────────────
mq_advantages = torch.clamp(mq_advantages, -adv_clip_max, adv_clip_max)
mq_unclipped_loss = -mq_advantages * ratio
mq_clipped_loss   = -mq_advantages * torch.clamp(ratio, 1-clip_range, 1+clip_range)
mq_loss = torch.mean(torch.maximum(mq_unclipped_loss, mq_clipped_loss))

# ── 公式 ⑩：加权组合总损失 ────────────────────────────────────
final_loss = vq_coef * vq_loss + mq_coef * mq_loss
```

为什么不直接 `vq_coef * vq_reward + mq_coef * mq_reward` 合并后再标准化？因为两个维度的**尺度和分布不同**——VQ 分数通常较大，MQ 较小。如果合并后标准化，画质会淹没运动。分别标准化后，每个维度都在"同组中排第几"的公平尺度上竞争。

## 步骤 3：时间步随机抽样训练

> 源码位置：`fastvideo/train_grpo_hunyuan.py` L411-487（`train_one_step` 函数内部）

生成视频时走完全部 T 步去噪，但**训练时只在随机抽样的部分步上反向传播**——这是 DanceGRPO 解决视频显存爆炸的关键。

**背景**：完整的 GRPO 策略梯度需要对轨迹的所有 $T$ 步求和：

$$\nabla_\theta L = \sum_{k=1}^{T} \nabla_\theta \log\pi_\theta(a_k | s_k) \cdot \hat{A}$$

但视频生成中 $T$ 步 × 帧数 × 分辨率的计算图会撑爆显存。DanceGRPO 的做法是**随机抽样 $K \ll T$ 步**，用无偏子集估计替代全部求和：

$$\nabla_\theta L \approx \frac{T}{K}\sum_{k \in \mathcal{S}} \nabla_\theta \log\pi_\theta(a_k | s_k) \cdot \hat{A}, \quad |\mathcal{S}| = K$$

```python
# ===== 源码：fastvideo/train_grpo_hunyuan.py (train_one_step 内部) =====

# ── 随机抽样：从 T 步中无放回抽取 K 步 ───────────────────────
# 技巧：先打乱顺序，再只取前 K 步 = 等效于随机不重复抽样
perms = torch.stack(
    [torch.randperm(len(samples["timesteps"][0])) for _ in range(batch_size)]
).to(device)
# 对 latents、next_latents、log_probs 按打乱顺序重排
for key in ["timesteps", "latents", "next_latents", "log_probs"]:
    samples[key] = samples[key][torch.arange(batch_size).to(device)[:, None], perms]

# K = T × timestep_fraction（如 50步 × 0.2 = 只训练 10 步）
train_timesteps = int(num_total_steps * args.timestep_fraction)

for i, sample in enumerate(samples_batched_list):       # 遍历每个样本
    for t in range(train_timesteps):                     # 只遍历抽中的 K 步
        # 公式 ⑥  用当前模型重算该步的 log π_θ（内部调用 flux_step，见步骤 1）
        new_log_probs = grpo_one_step(
            sample["latents"][:, t],                     # 该步输入 x_t
            sample["next_latents"][:, t],                # 该步输出 x_{t-Δt}（推理时已记录）
            ..., transformer, sigma_schedule,
        )
        # 公式 ⑧⑨⑩  计算 ratio → 分维度 clip loss → 加权组合（详见步骤 2）
        ratio = torch.exp(new_log_probs - sample["log_probs"][:, t])
        final_loss = compute_multidim_clip_loss(ratio, vq_advantages, mq_advantages)
        final_loss.backward()                            # 每步独立反传，立即释放计算图

    if (i + 1) % gradient_accumulation_steps == 0:       # 梯度累积后统一更新
        grad_norm = transformer.clip_grad_norm_(max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
```

**为什么这样有效？**

- `torch.randperm` 为每个样本**独立打乱** T 步顺序，再只取前 K 步 → 等效于无放回随机抽样
- 不同样本抽中的时间步组合不同 → 梯度估计的多样性更高
- 每步单独 `.backward()` 后立即释放计算图 → 显存只需保持 1 步的图，而非 K 步

PyTorch 的关键特性是：**在两次 `optimizer.zero_grad()` 之间，多次 `.backward()` 的梯度会自动累加**。即 `param.grad` 在每次 `.backward()` 后被加上（而非覆盖）。因此：

$$\underbrace{\nabla_\theta L_1 + \nabla_\theta L_2 + \cdots + \nabla_\theta L_K}_{\text{K 次 .backward() 累加}} = \nabla_\theta(L_1 + L_2 + \cdots + L_K)$$


> 参考资料：
>
> 1. Xue, Z., et al. (2025). *DanceGRPO: Unleashing GRPO on Visual Generation*. arXiv:2505.07818.
> 2. Flow-GRPO：Liu, et al. *Flow-GRPO: Training Flow Matching Models via Online RL*，arXiv:2505.05470。
