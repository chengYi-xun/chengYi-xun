---
title: 笔记｜生成模型（二十二）：GRPO 的三重面孔——从 2-GRPO 到 f-GRPO 与 GIFT
date: 2026-04-05 12:00:00
cover: false
mathjax: true
categories:
 - Notes
tags:
 - Deep learning
 - Generative models theory
 - Reinforcement Learning
series: Diffusion Models theory
---

> 本文为 RL 系列第七篇。上一篇介绍了 DAPO 的四大工程改进。本文从理论角度出发，剖析 GRPO 的数学本质：为什么 GRPO 其实是在做 DPO？为什么 2 个 rollout 就够了？如何从 KL 散度推广到任意 f-散度？最后介绍融合了 GRPO 和 DPO 优势的 GIFT 算法。
>
> 论文：
> - [It Takes Two: Your GRPO Is Secretly DPO](https://arxiv.org/abs/2510.00977)（2025.10）
> - [f-GRPO and Beyond: Divergence-Based RL for General LLM Alignment](https://arxiv.org/abs/2602.05946)（2026.02）
> - [GIFT: Group-relative Implicit Fine Tuning](https://arxiv.org/abs/2510.23868)（2025.10）

# 从一个令人意外的实验结果说起

在前几篇中，我们花了大量篇幅推导 GRPO 的组内相对优势计算，强调"组越大（$G = 16$ 或 $64$），优势估计越准确，训练效果越好"。但 2025 年末的一篇论文 *"It Takes Two"* 给出了一个反直觉的实验结果：

**仅用 $G = 2$（两个 rollout）的 GRPO，就能保留 $G = 16$ 版本 98.1% 的性能——但只需要 12.5% 的生成量和 21% 的训练时间。**

| 方法 | 每 Prompt 的 rollout 数 | 总生成量 | 训练时间 | 相对性能 |
|:---:|:---:|:---:|:---:|:---:|
| 16-GRPO | 16 | 1.2M | 100% | 100% |
| **2-GRPO** | **2** | **0.15M** | **21%** | **98.1%** |

这说明 **GRPO 的核心力量不在于"大组 → 精确基线估计"**，而在于别的什么东西。那到底是什么？

---

# 第一重面孔：GRPO 即在线 DPO（2-GRPO 的视角）

## 用例子重新理解 GRPO 的梯度

还是用数学积分题的例子。让模型用 $G = 4$ 个方式解 $\int_0^1 x^2 dx$，得到 2 个正确（$r = +1$）和 2 个错误（$r = -1$）。GRPO 计算组内优势后：

- 正确回答：$\hat{A}_i = +1$（鼓励）
- 错误回答：$\hat{A}_i = -1$（抑制）

**梯度信号的本质是什么？** 把 GRPO 的梯度展开：

$$
\hat{g}_{\text{GRPO}} = \frac{1}{G}\sum_{i=1}^{G} \hat{A}_i \nabla_\theta \log \pi_\theta(o_i | q)
$$

将回答分为正确组 $\mathcal{G}^+$（$G^+$ 个）和错误组 $\mathcal{G}^-$（$G^-$ 个），上式可以改写为：

$$
\hat{g}_{\text{GRPO}} \propto \underbrace{\frac{1}{G^+}\sum_{o \in \mathcal{G}^+} \nabla_\theta \log \pi_\theta(o | q)}_{\text{增加正确回答的概率}} - \underbrace{\frac{1}{G^-}\sum_{o \in \mathcal{G}^-} \nabla_\theta \log \pi_\theta(o | q)}_{\text{减少错误回答的概率}}
$$

**这不就是 DPO 的对比学习结构吗？** DPO 的梯度也是"增加 $y_w$ 的概率、减少 $y_l$ 的概率"的对比形式。区别仅在于：

| | DPO | GRPO |
|---|---|---|
| 正负对来源 | 人类标注的固定偏好对 $(y_w, y_l)$ | 模型在线采样 + 奖励判题动态划分 |
| 对比结构 | 固定 1-vs-1 | 动态 $N$-vs-$M$（$N = G^+$，$M = G^-$） |
| 学习方式 | 离线 | **在线** |

## 控制变量与方差缩减

论文进一步证明了 GRPO 组内配对的第二个关键作用：**方差缩减**。

考虑两个回答 $o^+$（正确）和 $o^-$（错误）来自**同一 Prompt、同一策略**。它们的策略梯度 $g^+ = \nabla_\theta \log \pi_\theta(o^+)$ 和 $g^- = \nabla_\theta \log \pi_\theta(o^-)$ 之间存在正相关性 $\rho > 0$（因为共享了 Prompt 的上下文信息）。

**定理（控制变量方差缩减）**：对于配对梯度估计 $g^+ - c \cdot g^-$，最优系数 $c^* = \frac{\text{Cov}(g^+, g^-)}{\text{Var}(g^-)}$ 下的方差为：

$$
\text{Var}(g^+ - c^* g^-) = (1 - \rho^2) \cdot \text{Var}(g^+)
$$

当 $\rho > 0$ 时，配对估计的方差**严格小于**单独估计。这就是为什么 GRPO 要在**同一 Prompt 的同一组内**做对比——不是为了更精确的均值估计，而是利用**同源配对的相关性**来降低梯度方差。

## 2-GRPO：最小对比单元

既然 GRPO 的核心是对比，那对比的最小单元就是 **2 个 rollout**。当 $G = 2$ 且 $r_1 \neq r_2$ 时（一对一错），GRPO 的优势退化为：

$$
\hat{A}_1 = +1, \quad \hat{A}_2 = -1 \quad \text{（归一化后的符号翻转）}
$$

这完全等价于一个**在线版的 DPO 更新**：增加正确回答的概率，减少错误回答的概率。

**2-GRPO 的训练目标**：

$$
J_{\text{2-GRPO}}(\theta) = \mathbb{E}_{q,\, (o_1, o_2) \sim \pi_{\theta_{\text{old}}}}\left[\mathbf{1}(r_1 \neq r_2) \cdot \frac{1}{2}\left(\frac{1}{|o^+|}\sum_t C_\varepsilon^+(\rho_t^+) - \frac{1}{|o^-|}\sum_t C_\varepsilon^-(\rho_t^-)\right)\right]
$$

其中 $C_\varepsilon^{\pm}$ 是 PPO 风格的裁剪，$o^+$ 和 $o^-$ 是根据奖励划分的正确/错误回答。

**代码对比**：

```python
# 标准 16-GRPO
prompts = sample(32)                        # 32 个 Prompt
responses = model.generate(prompts, G=16)   # 每个 16 个回答 → 512 rollouts
advantages = group_normalize(rewards)       # 组内归一化

# 2-GRPO（相同总 rollout 数 = 512）
prompts = sample(256)                       # 256 个 Prompt
responses = model.generate(prompts, G=2)    # 每个 2 个回答 → 512 rollouts
advantages = group_normalize(rewards)       # 退化为 ±1 对比
```

**总结**：GRPO 的第一重面孔——它是**在线版的 DPO**，核心机制是**对比学习**，不是精确的基线估计。

---

# 第二重面孔：从 KL 到任意 f-散度（f-GRPO）

## 为什么要超越 KL？

标准 GRPO 使用 KL 散度作为正则项来约束策略不要偏离参考模型太远：

$$
\max_\theta \mathbb{E}[r(x, y)] - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
$$

但 KL 散度只是众多散度中的一种。不同的散度对分布差异的度量方式不同：

| 散度 | $f(t)$ | 特性 |
|:---:|:---:|:---|
| KL（前向） | $t \ln t$ | 对 $\pi_\theta$ 的"模式覆盖"敏感 |
| 逆 KL | $-\ln t$ | 对"模式搜索"敏感，倾向集中 |
| Pearson $\chi^2$ | $(t-1)^2$ | 对尾部差异二次敏感 |
| Hellinger | $(\sqrt{t}-1)^2$ | 对称，对中等差异敏感 |
| JS | $t\ln t - (t+1)\ln\frac{t+1}{2}$ | 对称且有界 |
| 全变差 TV | $\frac{1}{2}|t-1|$ | 最强的拓扑性度量 |

**用例子理解**：模型解数学题有 3 种"模式"（直接计算、换元法、部分分式）。KL 散度倾向于让 $\pi_\theta$ 覆盖所有模式（即使参考模型只擅长一种），而逆 KL 倾向于让 $\pi_\theta$ 集中到最好的那个模式上。选择不同的散度，意味着选择不同的"探索 vs 集中"策略。

## f-GRPO 的核心思想

f-GRPO 论文的关键洞察是：GRPO 的组内正确/错误划分，可以自然地映射到 **f-散度的变分表示**中的"正侧"和"负侧"分布。

给定 f-散度的变分表示：

$$
D_f(P \| Q) = \sup_T \left(\mathbb{E}_P[T(v)] - \mathbb{E}_Q[f^*(T(v))]\right)
$$

其中 $f^*$ 是 $f$ 的 Fenchel 共轭。f-GRPO 将高于平均奖励的回答视为"正侧"样本，低于平均的视为"负侧"样本：

$$
\hat{I}_i^+ = \mathbf{1}\{\hat{A}_i > 0\}, \quad \hat{I}_i^- = \mathbf{1}\{\hat{A}_i \leq 0\}
$$

然后通过选择不同的**链接函数** $g$ 和对应的 $f^* \circ g$，构造 f-GRPO 的损失：

$$
\psi(r_{\theta,i}, a_i) = \begin{cases}
w_i^+ \cdot g(r_{\theta,i}) & \text{if } a_i > 0 \\
w_i^- \cdot f^* \circ g(r_{\theta,i}) & \text{if } a_i \leq 0
\end{cases}
$$

$$
\mathcal{L}_{\text{f-GRPO}}^{(f,g)}(\theta) = \mathbb{E}_x \sum_{i=1}^G \frac{-a_i}{1+\beta^{-1}} \cdot \frac{1}{G} \cdot \psi(r_{\theta,i}, a_i)
$$

其中 $r_{\theta,i} = \beta \ln \frac{\pi_\theta(y_i | x)}{\pi_{\text{ref}}(y_i | x)}$ 是隐式奖励，$w_i^{\pm}$ 是基于 softmax 的归一化权重。

## f-GRPO 的理论保证

**定理 4.3（f-HAL & f-GRPO 的收敛性）**：在 $G \to \infty$ 的渐近极限下：

1. **散度估计**：f-GRPO 的负损失值与奖励诱导的正/负分布之间的 f-散度成正比：
$$
-\mathcal{L}_{\text{f-GRPO}}(\theta^{(t+1)}) \propto D_f(D^+_{(r,\theta^{(t)})} \| D^-_{(r,\theta^{(t)})})
$$

2. **平均奖励单调改进**：在弱奖励-密度对应假设下（正面分布关于奖励单调），每次迭代的平均奖励严格递增，直到收敛到最大奖励。

**对比标准 GRPO**：

**定理 4.4（GRPO 的不动点特征）**：标准 GRPO 的隐式更新等价于按标准化奖励对参考策略做指数重加权：

$$
\pi_{\theta_{\text{GRPO}}^{(t+1)}}(y|x) \propto \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x,y) - \mu_r^{(t)}}{\beta \cdot \sigma_r^{(t)}}\right)
$$

这意味着 GRPO 会给低于平均奖励的回答**非零但小的权重**（指数衰减但不归零），而 f-GRPO 使用满足 $g^{-1}(f'_\infty) = \infty$ 的典范链接时，可以实现更激进的集中——将概率完全集中到高奖励回答上。

## 实验结果

f-GRPO 在 Qwen2.5-Math 1.5B/7B 上的数学推理实验中，**所有测试的 f-散度变体都优于标准 GRPO**：

| f-散度 | Relative Overall（1.5B） | 与 GRPO 对比 |
|:---:|:---:|:---:|
| GRPO（KL baseline） | 74.26 | — |
| Pearson $\chi^2$ | **86.49** | +12.23 |
| Hellinger | 82.15 | +7.89 |
| JS | 80.93 | +6.67 |
| 逆 KL | 79.41 | +5.15 |

Pearson $\chi^2$ 散度在数学推理任务上表现最优，可能是因为其对奖励分布尾部的二次敏感性更适合稀疏二值奖励场景。

---

# 第三重面孔：隐式奖励回归（GIFT）

## GRPO 和 DPO 各自的不足

| | GRPO | DPO |
|---|---|---|
| 优势 | 在线采样，能探索新回答 | 稳定，MSE 式损失 |
| 不足 | 裁剪超参数敏感，易过拟合 | 离线，无法探索 |

有没有办法**结合两者的优点**？GIFT 给出了一个优雅的方案。

## GIFT 的核心洞察

回忆 DPO 的隐式奖励公式（从第 19 篇文章推导）：

$$
r_\theta(x, y) = \beta \log \frac{\pi_\theta(y | x)}{\pi_{\text{ref}}(y | x)} + \beta \log Z(x)
$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(\frac{r(x,y)}{\beta})$ 是配分函数——这是一个关于 $x$ 的常数（不随 $y$ 变化），但不可计算。它是 DPO 无法做"单点"奖励匹配（只能做配对比较）的根本原因。

**GIFT 的关键发现**：如果我们对同一 Prompt 的**一组**回答做**均值归一化**，配分函数 $Z(x)$ 就会被消掉！

给定 $N$ 个回答 $\{y_1, \ldots, y_N\}$ 来自同一 Prompt $x$：

**隐式奖励的组内均值**：
$$
\mu_\theta = \frac{1}{N}\sum_{i=1}^N r_\theta(x, y_i) = \frac{1}{N}\sum_{i=1}^N \left(\beta \log \frac{\pi_\theta(y_i|x)}{\pi_{\text{ref}}(y_i|x)} + \beta \log Z(x)\right)
$$

**去中心化**（减去均值）：
$$
r_\theta(x, y_i) - \mu_\theta = \beta\left(\log \frac{\pi_\theta(y_i|x)}{\pi_{\text{ref}}(y_i|x)} - \frac{1}{N}\sum_{j=1}^N \log \frac{\pi_\theta(y_j|x)}{\pi_{\text{ref}}(y_j|x)}\right)
$$

**$\beta \log Z(x)$ 在相减过程中完全消除了！** 进一步做方差归一化后：

$$
\hat{r}'_\theta(x, y_i) = \frac{\log \frac{\pi_\theta(y_i|x)}{\pi_{\text{ref}}(y_i|x)} - \hat{\mu}_\theta}{\hat{\sigma}_\theta}
$$

这个归一化后的隐式奖励**不含** $Z(x)$，也**不含** $\beta$——全部被归一化吸收了。

## GIFT 的损失函数

有了这个发现，GIFT 的训练变得异常简单：**让归一化后的隐式奖励等于归一化后的显式奖励**。

$$
\mathcal{L}_{\text{GIFT}}(\pi_\theta) = \mathbb{E}_{(x, y) \sim \text{on-policy}}\left[\left(r'_\phi(x, y) - \hat{r}'_\theta(x, y)\right)^2\right]
$$

其中 $r'_\phi$ 是外部奖励模型打分经组内归一化后的值。

**用例子理解**：

| 回答 | 外部奖励 $r_\phi$ | 归一化后 $r'_\phi$ | 隐式奖励 $\hat{r}_\theta$ | 归一化后 $\hat{r}'_\theta$ | MSE 损失 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $y_1$（正确简洁） | 1.0 | +1.22 | 0.8 | +1.15 | $(1.22-1.15)^2$ |
| $y_2$（正确冗长） | 0.8 | +0.41 | 0.5 | +0.23 | $(0.41-0.23)^2$ |
| $y_3$（错误） | 0.0 | -0.82 | -0.3 | -0.69 | $(-0.82+0.69)^2$ |
| $y_4$（错误离谱） | -0.5 | -0.82 | -1.0 | -0.69 | $\ldots$ |

**GIFT 的训练就是在做一个简单的 MSE 回归**，让模型的隐式奖励分布与外部奖励分布对齐。

## GIFT vs GRPO vs DPO

| 特性 | GRPO | DPO | **GIFT** |
|:---|:---:|:---:|:---:|
| 在线采样 | ✓ | ✗ | **✓** |
| 需要裁剪超参数 | ✓（$\varepsilon$） | ✗ | **✗** |
| 需要参考模型 | ✓ | ✓ | **✓**（但 $\beta$ 和 $Z(x)$ 被消除） |
| 损失函数类型 | 裁剪策略梯度 | 对数 sigmoid | **MSE** |
| 过拟合风险 | 较高 | 中等 | **较低**（凸损失） |

实验中 GIFT 在 7B 模型上的 AlpacaEval LC Win Rate 为 48.77（vs GRPO 的 35.33），Arena-Hard Win Rate 为 72.43（vs GRPO 的 38.25）。

---

# 三重面孔的统一视角

| 视角 | 核心思想 | 代表方法 |
|:---:|:---|:---:|
| **对比学习** | GRPO ≈ 在线版 DPO，核心是正负对比 | 2-GRPO |
| **f-散度优化** | GRPO 的正负划分 ≈ f-散度变分表示的两侧 | f-GRPO |
| **隐式奖励回归** | 组内归一化消除配分函数，变为 MSE 回归 | GIFT |

这三种视角不是互斥的——它们揭示了 **GRPO 同一枚硬币的三个侧面**。理解这些联系，可以帮助我们：

1. **设计更高效的算法**（2-GRPO：减少 rollout 数量）
2. **选择更好的散度**（f-GRPO：用 Pearson $\chi^2$ 替代 KL）
3. **构建更稳定的训练**（GIFT：MSE 替代裁剪策略梯度）

```python
# GIFT 核心实现
def gift_loss(log_probs, ref_log_probs, rewards):
    """
    log_probs: 当前策略 log π_θ(y|x), shape [batch, N]
    ref_log_probs: 参考策略 log π_ref(y|x), shape [batch, N]
    rewards: 外部奖励 r_φ(x,y), shape [batch, N]
    """
    # 隐式奖励（不含 β 和 Z(x)）
    implicit_rewards = log_probs - ref_log_probs  # shape [batch, N]
    
    # 组内归一化（消除 Z(x)）
    impl_mean = implicit_rewards.mean(dim=1, keepdim=True)
    impl_std = implicit_rewards.std(dim=1, keepdim=True) + 1e-8
    norm_implicit = (implicit_rewards - impl_mean) / impl_std
    
    # 外部奖励同样组内归一化
    rew_mean = rewards.mean(dim=1, keepdim=True)
    rew_std = rewards.std(dim=1, keepdim=True) + 1e-8
    norm_rewards = (rewards - rew_mean) / rew_std
    
    # MSE 损失
    loss = F.mse_loss(norm_implicit, norm_rewards)
    return loss
```

> 下一篇：[笔记｜生成模型（二十三）：SuperFlow 与图像生成 RL 的统一框架](/chengYi-xun/posts/24-superflow/)

> 参考资料：
>
> 1. [It Takes Two: Your GRPO Is Secretly DPO](https://arxiv.org/abs/2510.00977)
> 2. [f-GRPO and Beyond: Divergence-Based RL for General LLM Alignment](https://arxiv.org/abs/2602.05946)
> 3. [GIFT: Group-relative Implicit Fine Tuning](https://arxiv.org/abs/2510.23868)
