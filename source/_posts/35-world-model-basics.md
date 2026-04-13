---
title: 笔记｜世界模型（一）：什么是世界模型？从认知科学到深度学习
date: 2026-04-06 00:30:00
categories:
 - Tutorials
tags:
 - World Model
 - Model-based RL
 - Latent Space
 - Generative Model
series: "世界模型"
mathjax: true
---

> **系列说明**：本文是世界模型系列的第一篇。世界模型领域目前技术路线**尚未收敛**，存在五大并行分支。本系列将从基础概念出发，按分支逐一展开。
> **核心论文**：*World Models*（Ha & Schmidhuber, 2018, arXiv:1803.10122）

---

## 0. 闭眼踢球——大脑里的物理模拟器

闭上眼睛，想象你在踢一个足球。球从脚尖飞出，在空中画一条抛物线，弹地后滚动减速。

你不需要看到球，就能"预测"它的轨迹。这种能力来自你大脑中的**世界模型**——一个关于物理世界如何运作的内部模拟器。

认知科学家 Kenneth Craik 在 1943 年就提出了这个概念：

> *"如果生物体能在头脑中构建一个外部现实的微型模型，它就能在行动前先在模型中尝试各种方案，预测哪种最优。"*
> — Kenneth Craik, *The Nature of Explanation*, 1943

在深度学习中，世界模型（World Model）就是这个"内部模拟器"的计算实现：**一个能够预测环境在给定动作下如何变化的神经网络。**

---

## 1. 世界模型的数学定义

### 1.1 核心要素

一个世界模型包含四个基本要素：

| 要素 | 符号 | 含义 |
|------|------|------|
| 状态 | $s_t \in \mathcal{S}$ | 世界在时刻 $t$ 的完整描述 |
| 观测 | $o_t \in \mathcal{O}$ | 智能体能看到/感知到的信息 |
| 动作 | $a_t \in \mathcal{A}$ | 智能体执行的动作 |
| 奖励 | $r_t \in \mathbb{R}$ | 环境给出的即时反馈 |

关键区别：**状态 ≠ 观测**。在 Atari 游戏中，状态包括所有敌人的位置、速度、内部计时器等；而观测只是一帧 $210 \times 160 \times 3$ 的 RGB 图像。

### 1.2 前向动力学模型

世界模型的核心是**前向动力学**（Forward Dynamics）：给定当前状态和动作，预测下一个状态：

$$
s_{t+1} = f(s_t, a_t) + \epsilon_t
$$

其中 $\epsilon_t$ 表示环境的随机性。如果世界是随机的（绝大多数真实场景），则需要建模**条件分布**：

$$
s_{t+1} \sim p_\theta(s_{t+1} \mid s_t, a_t)
$$

### 1.3 观测模型

状态通常是不可直接观测的（POMDP 设定）。观测模型将状态映射到智能体能感知的信息：

$$
o_t \sim p(o_t \mid s_t)
$$

### 1.4 完整的状态空间模型

将以上整合，一个世界模型由三部分组成：

$$
\boxed{
\begin{aligned}
\text{动力学模型（Transition）:} \quad & s_{t+1} \sim p_\theta(s_{t+1} \mid s_t, a_t) \\
\text{观测模型（Observation）:} \quad & o_t \sim p_\theta(o_t \mid s_t) \\
\text{奖励模型（Reward）:} \quad & r_t \sim p_\theta(r_t \mid s_t)
\end{aligned}
}
$$

> **定义（世界模型）**：一个世界模型是一个参数化的状态空间模型 $(p_\theta^{\text{trans}}, p_\theta^{\text{obs}}, p_\theta^{\text{rew}})$，它近似环境的真实动力学，使智能体能在内部"想象"未来轨迹而无需与真实环境交互。

### 1.5 与 MDP / POMDP 的关系

| 框架 | 状态可观测？ | 世界模型的角色 |
|------|------------|-------------|
| **MDP** | 完全观测：$o_t = s_t$ | 学习转移概率 $p(s_{t+1} \mid s_t, a_t)$ |
| **POMDP** | 部分观测：$o_t \neq s_t$ | 同时学习推断状态 $p(s_t \mid o_{\leq t})$ 和预测动力学 |

真实世界几乎都是 POMDP——你看到的只是世界的一个侧面。世界模型需要从有限的观测中**推断**完整状态，然后预测未来。

---

## 2. Ha & Schmidhuber (2018): 经典世界模型

![World Models V-M-C Architecture](/img/world_model_arch.png)

2018 年 Ha 和 Schmidhuber 的论文 *"World Models"* 首次将"世界模型"这个概念在深度学习中系统化。它的架构简洁而深刻，由三个模块组成：

### 2.1 V 模型：压缩感知（VAE）

将高维观测（如 $64 \times 64 \times 3$ 的游戏画面）压缩为低维潜变量：

$$
z_t = \text{Encoder}_\phi(o_t) \in \mathbb{R}^{32}
$$

使用变分自编码器（VAE）训练：

$$
\mathcal{L}_{\text{VAE}} = -\mathbb{E}_{q_\phi(z|o)}[\log p_\psi(o|z)] + \text{KL}[q_\phi(z|o) \| p(z)]
$$

第一项是重建损失（迫使 $z$ 保留足够信息来重建 $o$），第二项是 KL 散度（让 $z$ 的分布接近标准正态）。

### 2.2 M 模型：动力学预测（MDN-RNN）

在潜在空间中预测未来：给定当前潜变量 $z_t$ 和动作 $a_t$，预测下一个潜变量 $z_{t+1}$ 的分布。

使用 LSTM 维护一个隐状态 $h_t$，输出一个**混合密度网络**（MDN）：

$$
\begin{aligned}
h_t &= \text{LSTM}(h_{t-1}, [z_t, a_t]) \\
P(z_{t+1} \mid h_t) &= \sum_{i=1}^{K} \pi_i \cdot \mathcal{N}(z_{t+1} \mid \mu_i(h_t), \sigma_i^2(h_t))
\end{aligned}
$$

其中 $K$ 是高斯混合成分数（论文中 $K = 5$）。

MDN 的训练目标是最大化对数似然：

$$
\mathcal{L}_{\text{MDN}} = -\sum_t \log P(z_{t+1} \mid h_t) = -\sum_t \log \sum_{i=1}^{K} \pi_i \cdot \mathcal{N}(z_{t+1} \mid \mu_i, \sigma_i^2)
$$

### 2.3 C 模型：控制器

一个极简的线性控制器，直接从潜变量和隐状态输出动作：

$$
a_t = W_c [z_t; h_t] + b_c
$$

参数总量极少（仅数百个），通过**进化策略**（CMA-ES）优化——不需要梯度反传。

### 2.4 在"梦境"中训练

这是 World Models 论文最引人注目的实验：**完全在 M 模型生成的"梦境"中训练控制器**，然后直接迁移到真实环境。

流程：

1. **数据收集**：用随机策略与真实环境交互，收集 $(o_t, a_t, o_{t+1})$ 序列

2. **训练 V 模型**：学习将 $o_t$ 压缩为 $z_t$

3. **训练 M 模型**：学习在潜空间中预测 $z_{t+1}$

4. **在梦境中训练 C**：不再与真实环境交互，完全用 M 模型生成未来轨迹，在想象中优化控制器

```python
import torch
import torch.nn as nn


class VAE(nn.Module):
    """V 模型：将观测压缩为潜变量。

    Args:
        latent_dim: 潜变量维度（标量，默认 32）。
    """

    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.Unflatten(1, (256, 2, 2)),
            nn.ConvTranspose2d(256, 128, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, stride=2),
            nn.Sigmoid(),
        )

    def encode(self, x):
        """将图像编码为高斯参数。

        Args:
            x: 输入帧，形状 ``(B, 3, H, W)``。

        Returns:
            ``mu``, ``logvar``，各为 ``(B, latent_dim)``。
        """
        h = self.encoder(x)  # (B, 1024)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """重参数采样 z = mu + std * eps。

        Args:
            mu: 均值，``(B, latent_dim)``。
            logvar: 对数方差，``(B, latent_dim)``。

        Returns:
            采样潜变量 ``z``，``(B, latent_dim)``。
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        """重建输入并返回 VAE 参数。

        Args:
            x: 输入帧，``(B, 3, H, W)``。

        Returns:
            ``(recon, mu, logvar)``，``recon`` 与 ``x`` 同形。
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)  # (B, latent_dim)
        return self.decoder(z), mu, logvar


class MDNRNN(nn.Module):
    """M 模型：在潜空间中用 MDN 预测下一步潜变量。

    Args:
        latent_dim: 潜变量维度。
        action_dim: 动作维度。
        hidden_dim: LSTM 隐状态维度。
        n_gaussians: 高斯混合成分数 K。
    """

    def __init__(self, latent_dim=32, action_dim=3, hidden_dim=256, n_gaussians=5):
        super().__init__()
        in_dim = latent_dim + action_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True)
        self.fc_pi = nn.Linear(hidden_dim, n_gaussians * latent_dim)
        self.fc_mu = nn.Linear(hidden_dim, n_gaussians * latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim, n_gaussians * latent_dim)
        self.n_gaussians = n_gaussians
        self.latent_dim = latent_dim

    def forward(self, z, a, h=None):
        """单步或序列前向：输出 MDN 参数与新的 LSTM 状态。

        Args:
            z: 潜变量，``(B, L, latent_dim)``。
            a: 动作，``(B, L, action_dim)``，与 ``z`` 时间维对齐。
            h: 可选的 ``(h_n, c_n)`` LSTM 状态元组。

        Returns:
            ``(pi, mu, sigma, h_new)``；``pi, mu, sigma`` 为
            ``(..., K, latent_dim)``，``h_new`` 为 LSTM 输出状态。
        """
        x = torch.cat([z, a], dim=-1)  # (B, L, latent_dim + action_dim)
        out, h_new = self.lstm(x, h)  # (B, L, hidden_dim)
        pi = torch.softmax(
            self.fc_pi(out).view(-1, self.n_gaussians, self.latent_dim),
            dim=1,
        )
        mu = self.fc_mu(out).view(-1, self.n_gaussians, self.latent_dim)
        sigma = torch.exp(
            self.fc_sigma(out).view(-1, self.n_gaussians, self.latent_dim)
        )
        return pi, mu, sigma, h_new


class Controller(nn.Module):
    """C 模型：极简线性策略 a = tanh(W [z; h] + b)。

    Args:
        latent_dim: 潜变量维度。
        hidden_dim: RNN 隐状态维度。
        action_dim: 动作维度。
    """

    def __init__(self, latent_dim=32, hidden_dim=256, action_dim=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim + hidden_dim, action_dim)

    def forward(self, z, h):
        """线性映射后经 tanh 限幅。

        Args:
            z: ``(B, latent_dim)`` 或与 ``h`` 对齐的批次维。
            h: ``(B, hidden_dim)``。

        Returns:
            动作张量，``(B, action_dim)``。
        """
        zh = torch.cat([z, h], dim=-1)
        return torch.tanh(self.fc(zh))
```

### 2.5 "在梦境中训练"的伪代码

```python
def train_in_dream(controller, vae, mdnrnn, n_episodes=1000):
    """在世界模型生成的轨迹上训练控制器（示意伪代码）。

    Args:
        controller: 策略模块，输入 ``(z_t, h_t)``，输出动作。
        vae: V 模型，用于编码初始观测。
        mdnrnn: M 模型（MDN-RNN），用于想象潜空间转移。
        n_episodes: 外层 CMA-ES 迭代轮数。

    Returns:
        None；原地更新 ``controller`` 参数。
    """
    for episode in range(n_episodes):
        # 从训练集中采样一个初始观测
        o_0 = sample_initial_observation()
        z_t = vae.encode(o_0)
        h_t = mdnrnn.initial_hidden()
        total_reward = 0.0

        for t in range(max_steps):
            # 控制器决定动作
            a_t = controller(z_t, h_t)

            # 世界模型想象下一步；不与真实环境交互
            pi, mu, sigma, h_new = mdnrnn(z_t, a_t, h_t)
            z_next = sample_from_mdn(pi, mu, sigma)
            r_t = mdnrnn.predict_reward(h_new)

            total_reward = total_reward + r_t
            z_t, h_t = z_next, h_new

        # 用进化策略更新控制器参数
        update_controller_with_cmaes(controller, total_reward)
```

### 2.6 实验结果

Ha & Schmidhuber (2018) 在 CarRacing-v0 上的主要定量结果见论文 Table 1；下表为原文汇报的**真实环境**评估（100 次随机 rollout 的均值 ± 标准差）：

| 方法 | CarRacing-v0 |
|------|--------------|
| V 模型（控制器仅见 $z_t$） | 632 ± 251 |
| V 模型 + 控制器隐层 | 788 ± 141 |
| **完整世界模型 V+M+C**（控制器同时见 $z_t$ 与 $h_t$） | **906 ± 21** |

完整配置相对仅用 V 表示的情形提升明显，说明 MDN-RNN 给出的隐状态 $h_t$ 对控制至关重要。论文 §3.4 还给出了在 MDN-RNN 生成的幻觉轨迹中驾驶的可视化（Figure 13）；而定量对比以 Table 1 为准。

---

## 3. 潜在空间：为什么不在像素空间建模？

这是世界模型设计中的核心选择。为什么不直接预测 $o_{t+1} = f(o_t, a_t)$？

### 3.1 维度灾难

一帧 $64 \times 64 \times 3$ 的图像有 12,288 维。预测下一帧就是在 12,288 维空间中建模条件分布——这在统计上极其困难。

潜在空间 $z \in \mathbb{R}^{32}$ 将维度降低了 **384 倍**。

### 3.2 信息论视角

> **定理（率失真理论, Shannon 1959）**：对于信源 $X$ 和失真函数 $d(x, \hat{x})$，在码率不超过 $R$ 比特的约束下，最小可达失真为：
>
> $$D(R) = \min_{p(\hat{x}|x): I(X;\hat{X}) \leq R} \mathbb{E}[d(X, \hat{X})]$$
>
> 随 $R$ 增加，$D(R)$ 单调递减。

VAE 的潜空间就是在做**有损压缩**。KL 项控制"码率"（信息量），重建项控制"失真"。一个好的世界模型应该压缩掉**与预测无关的高熵细节**（如纹理噪声），保留**与动力学相关的语义信息**（如物体位置、速度）。

### 3.3 误差累积问题

在像素空间做多步预测时，误差累积是致命的：

$$
o_{t+k} = f^{(k)}(o_t, a_t, \ldots, a_{t+k-1})
$$

每步预测的微小误差（如一个像素的偏移）会在后续步骤中放大，导致生成的图像快速退化为模糊噪声。

在低维潜空间中，误差累积更可控——因为维度低，每个维度的误差对整体状态的影响更大，模型被迫学到更鲁棒的动力学。

---

## 4. 五大技术路线概览

Ha & Schmidhuber 2018 开创了"世界模型"的现代研究，之后领域迅速分化为五大技术路线：

### 路线图

![世界模型五大技术路线](/img/wm_five_routes.png)

```
Ha & Schmidhuber (2018)
"World Models"
         │
    ┌────┴────────────────────────────────────┐
    │                                         │
    ▼                                         ▼
Model-based RL                        LeCun (2022)
    │                               "JEPA 提案"
    ▼                                    │
Dreamer v1 (2019)                        ▼
Dreamer v2 (2021)                   I-JEPA (2023)
Dreamer v3 (2023)                   V-JEPA (2024)
IRIS (2023)                         V-JEPA 2 (2025)
TD-MPC2 (2023)                          │
DayDreamer (2022)                        │
    │                                    │
    │         ┌──────────────────────────┐│
    │         │                          ││
    ▼         ▼                          ▼▼
  视频生成世界模型              物理化世界模型
  Sora (2024)                PhysDreamer (2024)
  Genie (2024)               PhysGen (2024)
  Cosmos (2025)              NewtonGen (2025)
  UniSim (2024)              NewtonRewards (2025)
    │                              │
    │    ┌────────────┐            │
    └────┤ 自动驾驶   ├────────────┘
         │ 世界模型   │
         └────────────┘
         GAIA-1 (2023)
         DriveDreamer (2023)
         Vista (2024)
         OccWorld (2024)
```

### 五大路线对比

| 路线 | 核心思想 | 预测空间 | 代表作 | 是否生成像素 |
|------|---------|---------|--------|------------|
| **Model-based RL** | 在潜空间想象+训练策略 | 低维潜变量 | Dreamer v3 | 否（可选解码） |
| **JEPA** | 在嵌入空间预测语义 | 嵌入向量 | V-JEPA 2 | 否（设计如此） |
| **视频生成** | 生成高保真未来视频 | 像素/视频 token | Sora, Cosmos | 是 |
| **物理化生成** | 嵌入物理定律到生成过程 | 像素+物理量 | PhysDreamer | 是（物理约束） |
| **自动驾驶** | 领域特化的驾驶场景预测 | 像素/占用栅格 | GAIA-1, OccWorld | 是/否 |

### 核心分歧：生成 vs 预测

这五条路线之间最根本的分歧在于一个问题：**世界模型是否需要"看得见"？**

- **生成派**（Sora, Cosmos, 视频生成）：世界模型应该能生成逼真的未来视频/图像，因为视觉细节包含重要信息

- **预测派**（JEPA, Dreamer）：世界模型只需预测未来的**抽象表征**，无需重建每个像素——就像人闭眼也能踢球

LeCun 在 2022 年的白皮书中明确站在预测派：

> *"生成模型试图预测世界中的每一个细节，但许多细节是本质上不可预测的（如树叶的随机飘动）。一个好的世界模型应该学会在抽象层面预测。"*
> — Yann LeCun, 2022

这个辩论至今没有定论。接下来的文章将逐一展开每条路线。

---

## 5. 世界模型的训练范式

无论哪条技术路线，世界模型的训练通常遵循类似的范式：

### 5.1 两阶段训练

**第一阶段：学习世界模型**

从环境交互数据 $\mathcal{D} = \{(o_t, a_t, r_t, o_{t+1})\}$ 中学习动力学：

$$
\theta^* = \arg\min_\theta \sum_{(o_t, a_t, o_{t+1}) \in \mathcal{D}} \mathcal{L}_{\text{dynamics}}(o_{t+1}, \hat{o}_{t+1}(\theta))
$$

**第二阶段：利用世界模型**

在学到的世界模型中"想象"未来，训练策略或做规划：

$$
\pi^* = \arg\max_\pi \mathbb{E}_{\tau \sim p_\theta(\tau | \pi)} \left[\sum_{t=0}^{H} \gamma^t r_t\right]
$$

其中 $\tau = (s_0, a_0, s_1, a_1, \ldots)$ 是在世界模型中 rollout 的轨迹。

### 5.2 与 Model-free RL 的对比

| 维度 | Model-free (PPO/SAC) | Model-based (World Model) |
|------|---------------------|--------------------------|
| 样本效率 | 低（需要大量交互） | 高（可以在想象中反复训练） |
| 渐近性能 | 高（直接优化真实奖励） | 可能受限于模型误差 |
| 复合误差 | 无（直接交互） | 有（多步想象误差累积） |
| 迁移能力 | 弱 | 强（世界模型可复用） |

---

## 6. 总结与下一篇预告

| 概念 | 数学表达 | 含义 |
|------|---------|------|
| 前向动力学 | $s_{t+1} \sim p_\theta(s_{t+1} \mid s_t, a_t)$ | 预测世界如何变化 |
| 观测模型 | $o_t \sim p_\theta(o_t \mid s_t)$ | 从状态生成观测 |
| 潜空间 | $z_t = \text{Enc}(o_t)$ | 压缩高维观测 |
| 想象训练 | $\pi^* = \arg\max_\pi \mathbb{E}_{p_\theta}[\sum r_t]$ | 在模型中优化策略 |

下一篇将深入 **Dreamer 系列**——目前 Model-based RL 方向最成功的世界模型家族。从 RSSM 的数学推导开始，一路讲到 DreamerV3 如何用**固定超参**横扫 150+ 个任务。

> **下一篇**：[笔记｜世界模型（二）：Dreamer 系列——在想象中学习控制](posts/36-dreamer/)

---

**参考文献**

1. Ha, D. & Schmidhuber, J. (2018). *World Models*. arXiv:1803.10122.

2. Craik, K. (1943). *The Nature of Explanation*. Cambridge University Press.

3. Shannon, C. E. (1959). *Coding Theorems for a Discrete Source with a Fidelity Criterion*. IRE National Convention Record.
