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

> 本文为 RL 系列第五篇。在完整梳理了从 REINFORCE 到 PPO、DPO，再到最新 GRPO 的演进路线后，我们将目光转向图像生成领域。本文将结合 `flow_grpo` 开源代码库，深入解析如何将 GRPO 算法应用于基于 Flow Matching 的图像生成模型（如 Flux）的微调中。方法学与系统实验见论文 [*Flow-GRPO: Training Flow Matching Models via Online RL*](https://arxiv.org/abs/2505.05470)（文中以 SD3.5 等为主报告；仓库实现覆盖 Flux）。

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

![Flow-GRPO 概览：ODE→SDE 注入随机性、训练期 Denoising Reduction 与组内 GRPO 更新（摘自 Liu et al., arXiv:2505.05470 图 2）](/chengYi-xun/img/flow_grpo_arch.png)

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

## 核心挑战：如何在连续生成过程中定义 $\log \pi_\theta$？

在 LLM 中，动作（Action）是离散的词表 Token，$\log \pi_\theta(a|s)$ 就是 softmax 输出的对数概率——定义清晰、计算简单。然而在 Flow Matching 中，生成过程是一个**连续的常微分方程（ODE）求解过程**，没有天然的"离散动作"概念。

**用例子理解**：LLM 生成文本就像逐字写作——每个字是一个离散的"动作"，概率就是词表上的 softmax。而 Flux 生成图像像是画画——每个时间步的"动作"是在画布上做一次**连续的涂抹**（从噪声图向清晰图的一步变换），这是一个高维连续向量，不存在离散概率。

### 将去噪过程建模为 MDP

Flow-GRPO 的第一个关键设计是：将 Flow Matching 的去噪过程定义为一个 **马尔可夫决策过程**：

| MDP 要素 | LLM (GRPO) | 图像生成 (Flow-GRPO) |
|:---:|:---|:---|
| **状态** $s_t$ | $(x, y_{<t})$ (Prompt + 已生成 token) | $(x_t, t, c)$ (当前噪声图 + 时间步 + 文本条件) |
| **动作** $a_t$ | 下一个 token $y_t \in \mathcal{V}$（离散） | 预测的速度场 $v_\theta(x_t, t, c)$（连续向量） |
| **转移** | 确定性：拼接 $y_t$ 到序列 | 确定性 ODE 步：$x_{t-\Delta t} = x_t - \Delta t \cdot v_\theta$ |
| **奖励** | 只在最后一步（整句完成后打分） | 只在最后一步（整张图生成后打分） |

### 推导 Flow Matching 中的对数概率

在 Flow Matching 框架中，前向过程（加噪）定义为线性插值：

$$
x_t = (1 - t) \cdot x_0 + t \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

其中 $x_0$ 是干净图像，$\epsilon$ 是纯噪声，$t \in [0, 1]$。模型 $v_\theta(x_t, t, c)$ 学习预测速度场（即 $x_0$ 到 $\epsilon$ 方向的向量场）。

在去噪（生成）过程中，每一步的转移可以写成：

$$
x_{t - \Delta t} = x_t - \Delta t \cdot v_\theta(x_t, t, c)
$$

**如何从这个过程中提取对数概率？** 关键观察：确定性 ODE 没有概率可言，但如果我们在每一步都加入微小的高斯扰动（将 ODE 变成 SDE），转移就变成了一个随机过程：

$$
x_{t-\Delta t} = x_t - \Delta t \cdot v_\theta(x_t, t, c) + \sigma_t \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

由于 $\epsilon$ 是标准高斯噪声，$x_{t-\Delta t}$ 的条件分布自然就是以 ODE 预测值为均值、以 $\sigma_t^2 I$ 为协方差的高斯分布：

$$
p_\theta(x_{t-\Delta t} | x_t, c) = \mathcal{N}\left(x_{t-\Delta t}; \; x_t - \Delta t \cdot v_\theta(x_t, t, c), \; \sigma_t^2 I\right)
$$

对应的对数概率为：

$$
\log p_\theta(x_{t-\Delta t} | x_t, c) = -\frac{\| x_{t-\Delta t} - (x_t - \Delta t \cdot v_\theta(x_t, t, c)) \|^2}{2\sigma_t^2} + \text{const}
$$

**公式拆解**：这个结果直接来自多维高斯分布的概率密度取对数。对于 $\mathcal{N}(x;\mu, \sigma_t^2 I)$，密度函数为 $p(x) = (2\pi\sigma_t^2)^{-d/2}\exp\!\bigl(-\frac{\|x-\mu\|^2}{2\sigma_t^2}\bigr)$，取对数后展开：

$$
\log p(x) = -\frac{\|x - \mu\|^2}{2\sigma_t^2} \underbrace{- \frac{d}{2}\log(2\pi) - \frac{d}{2}\log \sigma_t^2}_{\text{const}}
$$

- **分母 $2\sigma_t^2$**：是高斯密度函数中 $-\frac{1}{2}$ 系数与协方差逆 $\Sigma^{-1}=\frac{1}{\sigma_t^2}I$ 相乘的结果。
- **const 项**：包含归一化常数 $-\frac{d}{2}\log(2\pi)$ 和行列式项 $-\frac{d}{2}\log\sigma_t^2$，它们都不含模型参数 $\theta$。在 GRPO 训练中，无论是对 $\theta$ 求梯度还是计算重要性采样比 $\exp(\log\pi_\theta - \log\pi_\text{old})$，这些常数项要么导数为零、要么做差时对消，因此可以安全省略。

**直觉理解**：模型预测的"涂抹方向"是 $v_\theta$，实际走的方向可能略有偏差。偏差越小（$\|x_{t-\Delta t} - \hat{x}_{t-\Delta t}\|^2$ 越小），对数概率越高——模型对自己的生成轨迹越"有信心"。

整条轨迹（$T$ 步去噪）的总对数概率是所有时间步的累加：

$$
\log \pi_\theta(\text{trajectory} | c) = \sum_{k=1}^{T} \log p_\theta(x_{t_k - \Delta t} | x_{t_k}, c)
$$

这就与 LLM 中 token 级对数概率求和的形式完全对应了——至此，GRPO 的整套框架可以无缝迁移到图像生成。

---

# Flow-GRPO 完整实现

## 1. 模型定义与奖励函数

```python
import torch
# PyTorch：张量与自动求导，策略梯度与 PPO 更新依赖计算图
import torch.nn.functional as F
# 函数式接口，便于扩展额外正则或其它可微损失

# --- 模型定义: 与 LLM GRPO 相同，只需 Actor + Reference ---
# Actor π_θ：当前可训练策略，用于采样与重要性采样比
actor_model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
actor_model.enable_lora(rank=32)
# 参考策略 π_ref：与 Actor 同初始化，用于 KL 惩罚稳定训练
ref_model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
# 冻结参考模型，保证 KL 是相对固定锚点而非共同漂移
ref_model.requires_grad_(False)

# 奖励模型: 组合多种评分函数
# 多专家打分合成标量回报 r，供 GRPO 组内归一化得到相对优势
reward_models = {
    # 美学评分（构图、色彩）
    "aesthetic": AestheticScore(),
    # 人类偏好综合评分
    "pick_score": PickScore(),
    # 图文匹配度
    "clip_score": CLIPScore(),
}
# 加权系数，决定 RL 目标更偏重美感、偏好还是对齐
reward_weights = {"aesthetic": 0.3, "pick_score": 0.5, "clip_score": 0.2}


def compute_reward(images, prompts):
    """
    组合多种奖励（加权求和）。

    Args:
        images: 生成图像，供各奖励头打分；形状依奖励实现而定。
        prompts: 与图像对应的文本条件；形状与 images 批次对齐。
    Returns:
        total: 加权标量回报；可与优势逐元素相乘，后续按每组 G 条 reshape。
    """
    total = sum(
        w * reward_models[k](images, prompts)
        for k, w in reward_weights.items()
    )
    return total


# 仅更新 LoRA 参数，与 PPO/GRPO 的策略更新对象一致
optimizer = torch.optim.AdamW(actor_model.lora_parameters(), lr=1e-5)
# 每个 Prompt 生成的图像数
G = 4
# PPO 裁剪阈值
clip_eps = 0.2
# KL 惩罚系数
beta = 0.01
# 去噪步数
num_steps = 50
```

## 2. 计算 Flow Matching 轨迹的对数概率

这是 Flow-GRPO 区别于 LLM GRPO 的**核心函数**：

```python
def compute_trajectory_log_prob(model, trajectory, prompt_embeds, sigma=0.01):
    """
    计算一条去噪轨迹的总对数概率。

    trajectory: 记录的中间状态列表 [(x_T, t_T), (x_{T-1}, t_{T-1}), ..., (x_0, t_0)]
    每一步的对数概率 ∝ -||x_{t-Δt} - (x_t - Δt·v_θ(x_t, t))||² / (2σ²)

    Args:
        model: 策略网络，提供 predict_velocity。
        trajectory: 状态序列，相邻对定义一步转移；x_t 形状如 (1, C, H, W)。
        prompt_embeds: 文本条件嵌入，与 model 接口一致。
        sigma: 高斯转移核标准差（标量）。
    Returns:
        total_log_prob: 标量，log π_θ(trajectory|c)，供 ratio 与 KL。
    """
    # 轨迹对数概率初值；累加后对应 log π_θ(trajectory|c)，供 ratio 与 KL
    total_log_prob = 0.0
    # 遍历相邻时刻对，把连续去噪链拆成逐步转移
    for k in range(len(trajectory) - 1):
        x_t, t_k = trajectory[k]  # (..., C, H, W), 标量 t
        # 采样时真实到达的下一状态（含 SDE 噪声后的 x）
        x_next_actual, t_next = trajectory[k + 1]
        # Δt>0，从 t=1 向 t=0 去噪
        dt = t_k - t_next

        # 策略输出：Flow 意义下的“动作”向量场
        v_pred = model.predict_velocity(x_t, t_k, prompt_embeds)
        # ODE 一步确定性预测均值，作为高斯转移核的中心
        x_next_predicted = x_t - dt * v_pred
        # 实际步与预测均值偏差越小，log p 越大
        log_prob_step = (
            -0.5 * ((x_next_actual - x_next_predicted) ** 2).sum() / (sigma**2)
        )
        # 与 LLM 中 token log prob 求和类比：整条轨迹因子分解
        total_log_prob = total_log_prob + log_prob_step

    # 供 new_logp、old_logp、ref_logp 计算重要性采样比与 KL
    return total_log_prob
```

## 3. 在线采样：生成图像并记录轨迹

```python
def generate_with_trajectory(model, prompt_embeds, num_steps, sigma=0.01):
    """
    生成图像，同时记录完整的去噪轨迹用于计算对数概率。

    Args:
        model: 策略网络。
        prompt_embeds: 文本条件嵌入。
        num_steps: 离散去噪步数。
        sigma: SDE 扰动强度（标量），与 compute_trajectory_log_prob 一致。
    Returns:
        image: 解码后像素图，形状依 VAE 而定。
        trajectory: [(x_t, t), ...]，x_t 如 (1, 16, 64, 64)，t 为标量。
    """
    # 初始纯噪声（VAE 隐空间）
    x_t = torch.randn(1, 16, 64, 64)  # (1, C, H, W)
    # 从 t=1 到 t=0 的离散时间表，Flow 逆过程时间网格
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)  # (num_steps + 1,)
    # 记录起点 (x_T, t_T)，供后续逐步累加 log p
    trajectory = [(x_t.clone(), timesteps[0])]

    # 逐步去噪：每一步是一次 MDP 转移，需记入轨迹算 log π
    for i in range(num_steps):
        # 当前连续时间标量
        t_k = timesteps[i]
        # 下一时刻，决定步长方向（向 t=0）
        t_next = timesteps[i + 1]
        # 本步 Δt>0，与 compute_trajectory_log_prob 一致
        dt = t_k - t_next

        # 采样阶段不反传，旧策略轨迹用于 off-policy 校正
        with torch.no_grad():
            v_pred = model.predict_velocity(x_t, t_k, prompt_embeds)

        # Euler + 微量随机扰动（ODE→SDE，使概率有定义）
        noise = torch.randn_like(x_t)
        # SDE 步：与文中高斯转移核对应
        x_t = x_t - dt * v_pred + sigma * noise * (dt**0.5)
        # 保存真实后继状态，供轨迹对数概率与 ratio 使用
        trajectory.append((x_t.clone(), t_next))

    # 隐空间终点解码为像素图，供奖励模型打分
    image = vae_decode(x_t)
    # 图用于 r，轨迹用于 log π_θ 与 PPO 目标
    return image, trajectory
```

## 4. 完整训练循环

```python
for step in range(total_steps):
    # 外层训练步：每步完成一组 on-policy 数据收集 + 多 epoch 更新
    # 批条件 c：每个 prompt 将独立做 G 路采样
    prompts_batch = sample_prompts(dataset, batch_size=4)
    # 文本编码，作为 Flow 条件输入（与 GRPO 中上下文类比）
    prompt_embeds = encode_text(prompts_batch)

    # --- 阶段 1: 组内采样 (每个 Prompt 生成 G 张图像) ---
    # 缓存本步所有样本供优势与更新
    all_images, all_trajectories, all_old_logps, all_ref_logps = [], [], [], []

    # 采样与旧 log p 不算梯度，避免用当前图错误反传
    with torch.no_grad():
        # 对每个 prompt 的条件嵌入分别展开
        for emb in prompt_embeds:
            # 同一 c 下 G 条轨迹：GRPO 的“组”，用于相对优势
            for _ in range(G):
                # 用当前 Actor rollout
                image, trajectory = generate_with_trajectory(
                    actor_model, emb, num_steps
                )
                # 采样时策略下的 log π_old，作 PPO 分母
                old_logp = compute_trajectory_log_prob(
                    actor_model, trajectory, emb
                )
                # 参考策略在相同轨迹上的 log π_ref
                ref_logp = compute_trajectory_log_prob(
                    ref_model, trajectory, emb
                )
                # 收集生成图，供奖励模型批处理打分
                all_images.append(image)
                # 保留状态序列，更新阶段重算 log π_θ
                all_trajectories.append(trajectory)
                # 采样时刻 Actor 的对数概率，用于 ratio 分母
                all_old_logps.append(old_logp)
                # 同轨迹上 ref 的 log p，用于 KL 正则项
                all_ref_logps.append(ref_logp)

    # --- 阶段 2: 奖励打分 + 组内相对优势 ---
    # 每条轨迹一个终局奖励 r
    rewards = compute_reward(
        all_images, repeat_prompts(prompts_batch, G)
    )  # (batch*G,)
    rewards_grouped = rewards.reshape(-1, G)  # (batch, G)

    # 组内基线：GRPO 用组均值代替 Critic
    mean_r = rewards_grouped.mean(dim=1, keepdim=True)  # (batch, 1)
    # 组内标准差，归一化使不同 prompt 尺度可比
    std_r = rewards_grouped.std(dim=1, keepdim=True)  # (batch, 1)
    # 组内 z-score 优势，与 PPO 裁剪目标逐元素相乘
    advantages = (
        (rewards_grouped - mean_r) / (std_r + 1e-8)
    ).reshape(-1)  # (batch*G,)

    # 旧策略对数概率张量，与 new_logps 逐元素比得 ratio
    old_logps = torch.stack(all_old_logps)  # (batch*G,)
    # 参考策略 log p，用于近似 KL(log π_θ - log π_ref)
    ref_logps = torch.stack(all_ref_logps)  # (batch*G,)

    # --- 阶段 3: 多 epoch 策略更新 ---
    # 同一批数据上多轮 PPO 式更新，提高样本利用率
    for epoch in range(K_epochs):
        # 对同一条轨迹用当前 θ 重算 log π_θ（可导）
        new_logps = torch.stack([
            compute_trajectory_log_prob(actor_model, traj, emb)
            # 轨迹与条件 c 与阶段 1 采样顺序严格对齐
            for traj, emb in zip(
                all_trajectories, repeat_embeds(prompt_embeds, G)
            )
        ])  # (batch*G,)

        # r_t(θ)=exp(logπ_θ-logπ_old)，连续轨迹与离散 token 形式一致
        ratio = torch.exp(new_logps - old_logps)  # (batch*G,)

        # 未裁剪 surrogate：鼓励在正优势方向增大 ratio
        surr1 = ratio * advantages
        # 裁剪后：限制策略单步变化
        surr2 = (
            torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        )
        # 取悲观界并最小化负号即最大化目标，信任域内更新
        policy_loss = -torch.min(surr1, surr2).mean()

        # 与 ref 的 log 差均值作 KL 代理，防止偏离预训练过远
        kl_loss = (new_logps - ref_logps).mean()

        # 总目标：PPO/GRPO 裁剪项 + β·KL
        loss = policy_loss + beta * kl_loss
        optimizer.zero_grad()
        # 梯度经 compute_trajectory_log_prob 回传到速度场/LoRA
        loss.backward()
        # 稳定 RL 训练的大梯度
        torch.nn.utils.clip_grad_norm_(
            actor_model.lora_parameters(), max_norm=1.0
        )
        # 更新 Actor（LoRA）参数
        optimizer.step()
```

> **与 LLM GRPO 的代码结构对比**：整体框架（组采样 → 优势计算 → 裁剪更新 → KL 惩罚）完全一致。唯一的区别在于**对数概率的计算方式**：LLM 用 token 级 softmax 对数概率求和，Flow-GRPO 用高斯转移核的对数密度求和。

---

## Flow-GRPO-Fast：加速采样的工程优化

全量去噪采样是 Flow-GRPO 的计算瓶颈——生成一张 1024×1024 图像，Flux 默认需要 50 步 ODE 求解。每个 Prompt 生成 $G=4$ 张就是 200 步，每个训练 step 有 4 个 Prompt 就是 800 步。Flow-GRPO 提出了两种加速策略：

### 策略 1：部分去噪（Partial Denoising）

不从纯噪声 $t=1$ 开始，而是从中间时间步 $t_{\text{start}}$ 开始（如 $t=0.5$）：

$$
x_{t_{\text{start}}} = (1 - t_{\text{start}}) \cdot x_0^{\text{ref}} + t_{\text{start}} \cdot \epsilon
$$

其中 $x_0^{\text{ref}}$ 是参考模型生成的一张"参考图"。这样只需去噪 $t_{\text{start}} \times T$ 步（比如 25 步而非 50 步），速度翻倍。

**代价**：生成多样性降低（所有 $G$ 张图都从同一个"参考半成品"出发），但对于微调场景通常足够。

### 策略 2：减少采样步数

直接减少 ODE 求解步数（如从 50 步减到 20 步），配合高阶 ODE 求解器（如 DPM-Solver++）。精度略有下降，但速度大幅提升。

---

# 算法对比与开源生态

| 维度 | Diffusion-DPO | DDPO (PPO) | Flow-GRPO |
|:---:|:---:|:---:|:---:|
| **训练方式** | 离线（偏好对） | 在线 RL | 在线 RL |
| **需要 Critic** | 否 | 是 | **否** |
| **基线估计** | 无 | Critic $V_\phi$ | 组内均值 |
| **适用模型** | DDPM / LDM | DDPM / LDM | **Flow Matching (Flux)** |
| **显存** | 低 | 极高 | **低** |
| **探索能力** | 弱 | 强 | 强 |

**开源代码参考：** [flow_grpo](https://github.com/yifan123/flow_grpo) 提供了基于 Flux 的完整实现，支持 LoRA 微调、多 GPU 训练和 Flow-GRPO-Fast 加速。

---

# 系列总结

通过这五篇文章，我们从最基础的强化学习与策略梯度出发，推导了解决步长控制的 PPO 算法，探讨了绕开 RL 的 DPO 路线，最终迎来了解决大模型显存危机的 GRPO 算法，并成功将其落地到了最前沿的 Flow-GRPO 图像生成微调框架中。

强化学习与生成模型的结合，正在开启 AI 领域的新纪元。无论是语言模型中的深度思考（DeepSeek-R1），还是图像生成中的美学对齐（Flow-GRPO），在线强化学习都展现出了无与伦比的潜力。

> 下一篇：[笔记｜生成模型（二十一）：DAPO：从 GRPO 到大规模推理 RL 的工程实践](/chengYi-xun/posts/22-dapo/)
