---
title: 笔记｜世界模型（二）：Dreamer 系列——在想象中学习控制
date: 2026-04-06 00:35:00
categories:
 - Tutorials
tags:
 - Dreamer
 - RSSM
 - Model-based RL
 - World Model
series: "世界模型"
mathjax: true
---

> **核心论文**：Dreamer v1 (arXiv:1912.01603, ICLR 2020)、DreamerV2 (arXiv:2010.02193, ICLR 2021)、DreamerV3 (arXiv:2301.04104, Nature 2025)
>
> **代码**：[danijar/dreamerv3](https://github.com/danijar/dreamerv3) | **前置知识**：[上一篇：世界模型基础](/chengYi-xun/posts/26-world-model-basics/)
>
> ⬅️ 上一篇：[笔记｜世界模型（一）：什么是世界模型？从认知科学到深度学习](/chengYi-xun/posts/26-world-model-basics/)
>
> ➡️ 下一篇：[笔记｜世界模型（三）：JEPA——在嵌入空间预测世界](/chengYi-xun/posts/28-jepa/)

## 0. 在 Minecraft 中不靠人类示范采到钻石

Minecraft 的钻石任务是 RL 界的"登月挑战"：从零开始，需要砍树 → 合成工作台 → 挖石头 → 合成石镐 → 找到并挖掘钻石——每局最长约 **36,000** 步（论文设定约 30 分钟）的长序列，奖励极其稀疏。

DreamerV3 是首个在**不使用人类示范数据、也不依赖自适应课程**的前提下，从零开始在该基准上稳定采到钻石的算法；且与 Atari、DeepMind Control Suite 等域共用**同一套默认超参数**（Hafner et al., Nature 2025 / arXiv:2301.04104）。

核心秘密：**在世界模型的"想象"中大量练习**，然后把学到的策略部署到真实环境。

## 1. RSSM：Dreamer 的心脏

### 1.1 为什么需要 RSSM？

上一篇介绍的 MDN-RNN 有一个限制：它的隐状态 $h_t$ 是纯确定性的。但真实世界是随机的——同样的状态和动作，可能导致不同的结果（如抛硬币）。

**Recurrent State-Space Model (RSSM)** 将隐状态分为**确定性**和**随机性**两部分：

$$
\text{完整状态} = (\underbrace{h_t}_{\text{确定性}}, \underbrace{z_t}_{\text{随机性}})
$$

![RSSM Architecture](/img/rssm_architecture.png)

### 1.2 RSSM 的四个方程

$$
\boxed{
\begin{aligned}
\text{序列模型:} \quad & h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1}) \\
\text{编码器（后验）:} \quad & z_t \sim q_\theta(z_t \mid h_t, o_t) \\
\text{先验:} \quad & \hat{z}_t \sim p_\theta(\hat{z}_t \mid h_t) \\
\text{解码器:} \quad & \hat{o}_t \sim p_\theta(o_t \mid h_t, z_t)
\end{aligned}
}
$$

**序列模型**（确定性路径）：GRU 网络，将历史信息编码到 $h_t$。

**编码器**（后验分布）：在训练时可以看到真实观测 $o_t$，因此能推断出更准确的 $z_t$。

**先验**（无观测时的预测）：在"想象"阶段，无法访问 $o_t$，只能靠 $h_t$ 预测 $z_t$。

**核心思想**：训练时让先验逼近后验（通过 KL 散度），这样在想象时先验就足够准确。

### 1.3 ELBO 推导

RSSM 的训练目标是最大化观测在已知动作下的条件似然 $\log p(o_{1:T} \mid a_{1:T})$ 的**变分下界**（ELBO）：

$$
\log p(o_{1:T} \mid a_{1:T}) \geq \sum_{t=1}^{T} \left[\underbrace{\mathbb{E}_{q_\theta(z_t|h_t, o_t)}[\log p_\theta(o_t \mid h_t, z_t)]}_{\text{重建项}} - \underbrace{\text{KL}[q_\theta(z_t \mid h_t, o_t) \| p_\theta(z_t \mid h_t)]}_{\text{KL 项}}\right]
$$

**重建项**推动模型学习好的状态表征（能重建观测）；**KL 项**推动先验逼近后验（使想象中的预测可靠）。

加上奖励预测，完整损失为：

$$
\mathcal{L} = \sum_t \left[-\log p_\theta(o_t \mid h_t, z_t) - \log p_\theta(r_t \mid h_t, z_t) + \beta \cdot \text{KL}[q \| p]\right]
$$

## 2. Dreamer v1：想象中的策略梯度

![Dreamer：从经验学习动力学，并在想象中学习行为（Figure 3, Hafner et al., ICLR 2020）](/img/dreamer_imagination.png)

### 2.1 想象轨迹

在世界模型学好后，Dreamer v1 完全在**想象**中训练 Actor 和 Critic：

1. 从真实数据采样一个初始状态 $(h_0, z_0)$

2. 用先验 $p_\theta(\hat{z}_t \mid h_t)$ 和策略 $\pi_\psi(a_t \mid h_t, z_t)$ 逐步展开想象轨迹

3. 在想象轨迹上计算价值和策略梯度

$$
\text{想象轨迹}: \quad (h_1, z_1, a_1, r_1) \to (h_2, z_2, a_2, r_2) \to \cdots \to (h_H, z_H)
$$

### 2.2 价值梯度（Value Gradient）

Dreamer v1 使用**反传穿过动力学**的策略梯度（不同于 PPO 的 REINFORCE 估计）：

$$
\max_\psi \sum_{t=1}^{H} V_\xi(h_t, z_t)
$$

其中 $V_\xi$ 是在想象轨迹上训练的价值网络，采用 $\lambda$-return：

$$
V_t^\lambda = r_t + \gamma \left[(1-\lambda) V_\xi(s_{t+1}) + \lambda V_{t+1}^\lambda\right]
$$

## 3. DreamerV2：离散潜变量

### 3.1 从连续到离散

Dreamer v1 使用连续高斯分布 $z_t \sim \mathcal{N}(\mu, \sigma^2)$。DreamerV2 将其改为**离散分类分布**：

$$
z_t \sim \text{Categorical}(\text{logits}_\theta(h_t, o_t))
$$

具体地，$z_t$ 由 32 个独立的分类变量组成，每个有 32 个类别：

$$
z_t = (z_t^1, z_t^2, \ldots, z_t^{32}), \quad z_t^i \sim \text{Cat}(p_1^i, \ldots, p_{32}^i)
$$

总共 $32 \times 32 = 1024$ 种组合。

### 3.2 Straight-Through Gradient

离散采样不可微。DreamerV2 使用 **Straight-Through** 技巧：

**前向传播**：采样离散的 one-hot 向量 $z_t^{\text{hard}}$

**反向传播**：使用连续的 softmax 概率 $z_t^{\text{soft}} = \text{softmax}(\text{logits})$

$$
z_t = z_t^{\text{hard}} - \text{sg}(z_t^{\text{soft}}) + z_t^{\text{soft}}
$$

其中 $\text{sg}$ 是 stop-gradient。前向时值为 $z_t^{\text{hard}}$（离散），反向时梯度流过 $z_t^{\text{soft}}$（连续）。

### 3.3 为什么离散更好？

| 维度 | 连续高斯 | 离散分类 |
|------|---------|---------|
| 模式覆盖 | 单峰 | 天然多模态 |
| KL 优化 | 容易陷入 posterior collapse | 更稳定 |
| Atari 200M | 112% 人类中位数 | **198%** 人类中位数 |

## 4. DreamerV3：固定超参横扫一切

### 4.1 核心问题

DreamerV2 在不同领域需要调整超参数。DreamerV3 的目标是**一套超参打天下**——从 Atari 到机器人控制到 Minecraft，无需任何调整。

### 4.2 三大技术创新

**symlog 预测**：对奖励和价值使用对称对数变换，处理不同尺度：

$$
\text{symlog}(x) = \text{sign}(x) \cdot \ln(|x| + 1)
$$

$$
\text{symexp}(x) = \text{sign}(x) \cdot (\exp(|x|) - 1)
$$

模型在 symlog 空间预测，用 symexp 还原。这使模型对奖励尺度不敏感——Atari 的奖励是整数 0/1/10，而 DMC 的奖励是连续的 [0, 1000]。

**KL 平衡（Free bits）**：防止 KL 项过度压制先验或后验：

$$
\mathcal{L}_{\text{KL}} = \alpha \cdot \text{KL}[\text{sg}(q) \| p] + (1 - \alpha) \cdot \text{KL}[q \| \text{sg}(p)]
$$

$\alpha = 0.8$ 意味着 80% 的 KL 梯度推动先验 $p$ 去逼近后验 $q$（训练世界模型），20% 推动后验 $q$ 去逼近先验 $p$（正则化编码器）。

**百分位归一化（Return Normalization）**：用 running percentile 将回报归一化到 [0, 1]：

$$
R_{\text{norm}} = \frac{R - \text{Perc}_5(R)}{\text{Perc}_{95}(R) - \text{Perc}_5(R)}
$$

### 4.3 实验结果

| 领域 | 任务数 | DreamerV3 vs 领域 SOTA |
|------|--------|----------------------|
| Atari 100K | 26 | **超越** EfficientZero |
| Atari 200M | 55 | **超越** 人类水平 |
| DMC Vision | 20 | **持平** DrQ-v2 |
| DMC Proprio | 10 | **持平** SAC |
| **Minecraft 钻石** | 1 | **首次从零完成** |

## 5. IRIS 与 TD-MPC2：其他世界模型范式

### 5.1 IRIS：Transformer 作为世界模型

IRIS（Micheli et al., 2023）用**离散 VQ 编码 + 自回归 Transformer**替代 RSSM：

1. **VQ 编码器**：将观测量化为离散 token 序列

2. **Transformer**：自回归预测下一个 token（类似 GPT 预测下一个词）

3. **策略学习**：在 Transformer 想象的 token 序列上训练

$$
P(z_{t+1}^{1:K} \mid z_{\leq t}^{1:K}, a_{\leq t}) = \prod_{k=1}^{K} P(z_{t+1}^k \mid z_{t+1}^{<k}, z_{\leq t}^{1:K}, a_{\leq t})
$$

IRIS 在 Atari 100K 上以极少的交互样本（2 小时游戏时间）达到了 DreamerV2 级别的性能。

### 5.2 TD-MPC2 与无解码器（Decoder-Free）的隐式世界模型

TD-MPC2（Hansen et al., 2023）走了一条截然不同的路：**不做观测重建（Decoder-Free）**。

世界模型只在潜空间运作：

$$
\begin{aligned}
z_t &= h_\theta(o_t) \quad &\text{(编码器)} \\
z_{t+1} &= d_\theta(z_t, a_t) \quad &\text{(潜动力学)} \\
r_t &= R_\theta(z_t, a_t) \quad &\text{(奖励预测)} \\
Q &= Q_\theta(z_t, a_t) \quad &\text{(Q 值预测)}
\end{aligned}
$$

没有解码器——模型不需要能重建图像，只需要潜空间对**规划有用**。

规划用 **Model Predictive Control (MPC)**：在每步决策时，在世界模型中短程 rollout 多条轨迹，选择 Q 值最高的动作序列。

**无解码器模型的理论挑战与 2025 年的进展**：

放弃解码器虽然大幅提升了训练速度（无需渲染像素），但也带来了一个致命的理论问题：**表征崩塌（Representation Collapse）**。如果没有重建损失的约束，编码器很容易将所有观测映射到同一个常数向量，从而完美但无意义地最小化动力学预测误差。

早期的无解码器方法严重依赖于数据增强（Data Augmentation）或对比学习来防止崩塌。而在 2025 年的最新研究（如 R2-Dreamer，ICLR 2025）中，研究者引入了受 Barlow Twins 启发的**冗余减少（Redundancy-Reduction）目标**：

$$
\mathcal{L}_{\text{R2}} = \sum_i (1 - \mathcal{C}_{ii})^2 + \lambda \sum_i \sum_{j \neq i} \mathcal{C}_{ij}^2
$$

其中 $\mathcal{C}$ 是特征的互相关矩阵。这迫使潜变量的不同维度捕捉独立的信息，在不使用解码器和数据增强的情况下，成功防止了表征崩塌，训练速度比 DreamerV3 快 1.59 倍，同时保持了相当的性能。

| 维度 | Dreamer | IRIS | TD-MPC2 |
|------|---------|------|---------|
| 潜空间 | RSSM（确定+随机） | 离散 VQ token | 连续向量（无解码器） |
| 序列模型 | GRU | Transformer | MLP |
| 策略学习 | Actor-Critic | Actor-Critic | MPC |
| 解码器 | 需要 | 需要 | 不需要 |
| 多任务能力 | 单任务调超参 | 单任务 | **多任务（单一模型）** |

### 5.3 DayDreamer：走向真实机器人

DayDreamer（Wu et al., 2022）将 Dreamer 思路部署到四种真实机器人平台，证明世界模型不只在模拟器中有效。

关键挑战：

- 真实传感器噪声远大于模拟器

- 执行动作有延迟

- 不能"重置"环境

DayDreamer 在 A1 四足机器人上仅用 **1 小时**真实世界数据就学会了稳定行走——而 model-free RL 在模拟器中需要数百万步。

## 6. 代码实现：简化版 RSSM

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl_divergence


class RSSM(nn.Module):
    """简化版 RSSM（DreamerV1 风格）。

    Args:
        obs_dim: 观测维度。
        act_dim: 动作维度。
        stoch_dim: 随机潜变量维度。
        deter_dim: GRU 确定性状态维度。
    """

    def __init__(self, obs_dim, act_dim, stoch_dim=30, deter_dim=200):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim

        # 序列模型 (GRU)
        self.gru = nn.GRUCell(stoch_dim + act_dim, deter_dim)

        # 先验 p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, 200),
            nn.ELU(),
            nn.Linear(200, 2 * stoch_dim),  # mu + logvar
        )

        # 后验 q(z_t | h_t, o_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + obs_dim, 200),
            nn.ELU(),
            nn.Linear(200, 2 * stoch_dim),
        )

        # 观测嵌入：obs -> 与观测同维的向量
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 200),
            nn.ELU(),
            nn.Linear(200, obs_dim),
        )

        # 解码器：拼接 (h_t, z_t) 重建观测
        self.decoder = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, obs_dim),
        )

        # 奖励预测头
        self.reward_head = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, 200),
            nn.ELU(),
            nn.Linear(200, 1),
        )

    def _get_dist(self, stats):
        """将 ``(mu, logscale)`` 拼接向量拆成对角高斯。

        Args:
            stats: ``(..., 2 * stoch_dim)``，前半为均值、后半为 log-scale。

        Returns:
            ``Independent(Normal, 1)`` 分布对象。
        """
        mu, logvar = stats.chunk(2, dim=-1)
        std = F.softplus(logvar) + 0.1
        return Independent(Normal(mu, std), 1)

    def prior(self, h):
        """先验 p(z_t | h_t)。

        Args:
            h: 确定性状态，``(B, deter_dim)``。

        Returns:
            随机潜变量上的分布。
        """
        return self._get_dist(self.prior_net(h))

    def posterior(self, h, obs_embed):
        """后验 q(z_t | h_t, o_t)。

        Args:
            h: ``(B, deter_dim)``。
            obs_embed: 观测嵌入，``(B, obs_dim)``。

        Returns:
            随机潜变量上的分布。
        """
        x = torch.cat([h, obs_embed], dim=-1)
        return self._get_dist(self.posterior_net(x))

    def observe(self, obs_seq, act_seq, h_prev, z_prev):
        """训练时沿时间展开：用真实观测计算后验轨迹。

        Args:
            obs_seq: 观测序列，``(B, T, obs_dim)``。
            act_seq: 动作序列，``(B, T, act_dim)``。
            h_prev: 初始确定性状态，``(B, deter_dim)``。
            z_prev: 初始随机状态，``(B, stoch_dim)``。

        Returns:
            ``(prior_mean, post_mean, h_seq, z_seq, priors, posteriors)``；
            ``h_seq, z_seq`` 为 ``(B, T, *)``。
        """
        T = obs_seq.shape[1]
        priors, posteriors, h_list, z_list = [], [], [], []
        h, z = h_prev, z_prev

        for t in range(T):
            # 序列更新
            x = torch.cat([z, act_seq[:, t]], dim=-1)  # (B, stoch+act)
            h = self.gru(x, h)

            prior_dist = self.prior(h)
            obs_embed = self.obs_encoder(obs_seq[:, t])
            post_dist = self.posterior(h, obs_embed)
            z = post_dist.rsample()

            priors.append(prior_dist)
            posteriors.append(post_dist)
            h_list.append(h)
            z_list.append(z)

        prior_mean = torch.stack([p.mean for p in priors], dim=1)
        post_mean = torch.stack([p.mean for p in posteriors], dim=1)
        h_seq = torch.stack(h_list, dim=1)
        z_seq = torch.stack(z_list, dim=1)
        return prior_mean, post_mean, h_seq, z_seq, priors, posteriors

    def imagine(self, policy, h, z, horizon):
        """想象 rollout：仅用转移先验与策略生成轨迹。

        Args:
            policy: 映射 ``concat(h,z) -> a`` 的可调用对象。
            h: 初始确定性状态，``(B, deter_dim)``。
            z: 初始随机状态，``(B, stoch_dim)``。
            horizon: 想象步数。

        Returns:
            ``(h_traj, z_traj, a_traj)``；其中 ``h_traj, z_traj`` 为
            ``(B, horizon+1, *)``，``a_traj`` 为 ``(B, horizon, act_dim)``。
        """
        h_list, z_list, a_list = [h], [z], []
        for _ in range(horizon):
            hz = torch.cat([h, z], dim=-1)
            a = policy(hz)
            x = torch.cat([z, a], dim=-1)
            h = self.gru(x, h)
            z = self.prior(h).rsample()
            h_list.append(h)
            z_list.append(z)
            a_list.append(a)
        h_traj = torch.stack(h_list, dim=1)
        z_traj = torch.stack(z_list, dim=1)
        a_traj = torch.stack(a_list, dim=1)
        return h_traj, z_traj, a_traj

    def compute_loss(self, obs_seq, act_seq, rew_seq):
        """简化的重建 + 奖励 + KL 目标。

        Args:
            obs_seq: ``(B, T, obs_dim)``。
            act_seq: ``(B, T, act_dim)``。
            rew_seq: ``(B, T)``。

        Returns:
            标量损失。
        """
        B, T = obs_seq.shape[:2]
        h_0 = torch.zeros(B, self.deter_dim, device=obs_seq.device)
        z_0 = torch.zeros(B, self.stoch_dim, device=obs_seq.device)

        _, _, h_seq, z_seq, priors, posteriors = self.observe(
            obs_seq, act_seq, h_0, z_0
        )

        # 重建
        features = torch.cat([h_seq, z_seq], dim=-1)  # (B, T, deter+stoch)
        obs_pred = self.decoder(features)
        recon_loss = F.mse_loss(obs_pred, obs_seq, reduction="mean")

        # 奖励
        rew_pred = self.reward_head(features).squeeze(-1)
        reward_loss = F.mse_loss(rew_pred, rew_seq, reduction="mean")

        # KL：时间步平均
        kl_terms = [
            kl_divergence(post, prior).mean()
            for prior, post in zip(priors, posteriors)
        ]
        kl_loss = sum(kl_terms) / T

        return recon_loss + reward_loss + 0.1 * kl_loss
```

## 7. 总结

| 模型 | 年份 | 潜变量 | 关键创新 | 标志性成就 |
|------|------|--------|---------|-----------|
| World Models | 2018 | 连续 (VAE) | 在梦境中训练 | CarRacing |
| Dreamer v1 | 2019 | 连续 (RSSM) | 想象中的价值梯度 | DMC 控制 |
| DreamerV2 | 2021 | 离散 (Categorical) | Straight-Through | Atari 200M |
| DreamerV3 | 2023 | 离散 | symlog, KL 平衡 | Minecraft 钻石 |
| IRIS | 2023 | 离散 (VQ) | Transformer 世界模型 | Atari 100K |
| TD-MPC2 | 2023 | 连续（无解码器） | 隐式模型 + MPC | 多任务控制 |
| DayDreamer | 2022 | 连续 (RSSM) | 真实机器人部署 | 1小时学行走 |

Dreamer 系列代表了"在潜空间想象+训练策略"的技术路线。但 LeCun 认为，Dreamer 仍然在做"生成式"预测——用解码器重建观测。下一篇将介绍他提出的替代方案：**JEPA**，一种完全不重建像素的预测架构。

> 参考资料：
>
> 1. Hafner, D., ... & Ba, J. (2019). *Dream to Control: Learning Behaviors by Latent Imagination*. ICLR 2020.
> 2. Hafner, D., ... & Ba, J. (2021). *Mastering Atari with Discrete World Models*. ICLR 2021.
> 3. Hafner, D., ... & Ba, J. (2023). *Mastering Diverse Domains through World Models*. Nature 2025.
> 4. Micheli, V., ... & Fleuret, F. (2023). *Transformers are Sample-Efficient World Models*. ICLR 2023.
> 5. Hansen, N., ... & Abbeel, P. (2023). *TD-MPC2: Scalable, Robust World Models for Continuous Control*. ICLR 2024.
> 6. Wu, P., ... & Abbeel, P. (2022). *DayDreamer: World Models for Physical Robot Learning*. CoRL 2022.
> 7. (2025). *R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation*. ICLR 2025.

> 下一篇：[笔记｜世界模型（三）：JEPA——在嵌入空间预测世界](/chengYi-xun/posts/28-jepa/)
