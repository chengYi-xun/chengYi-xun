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
>
> - [It Takes Two: Your GRPO Is Secretly DPO](https://arxiv.org/abs/2510.00977)（2025.10）
> - [f-GRPO and Beyond: Divergence-Based RL for General LLM Alignment](https://arxiv.org/abs/2602.05946)（2026.02）
> - [GIFT: Group-relative Implicit Fine Tuning](https://arxiv.org/abs/2510.23868)（2025.10）

# 从一个令人意外的实验结果说起

在前几篇中，我们花了大量篇幅推导 GRPO 的组内相对优势计算，强调"组越大（$G = 16$ 或 $64$），优势估计越准确，训练效果越好"。但 2025 年 10 月的一篇论文 *"It Takes Two"*（arXiv:2510.00977）给出了一个反直觉的实验结果：

- **性能方面**：仅用 $G = 2$（两个 rollout）的 GRPO（记为 2-GRPO），在多项数学推理基准上**保留了 16-GRPO 约 98.1% 的平均性能**（论文 Table 1 中多数指标差距在 1–2 个百分点内，部分设置下 2-GRPO 反而略高）。

- **效率方面**：2-GRPO 的总 rollout 生成量**仅为 16-GRPO 的 $2/16 = 12.5\%$**（约 0.15M vs 1.2M rollouts）；由于训练还包含梯度计算等固定开销，实际墙钟时间（wall-clock time）平均降至 16-GRPO 的约 21%（所有实验均值），即**训练时间缩短约 79%**。

![DPO / GRPO / GIFT 等对齐范式对比（摘自 GIFT 论文图 1，arXiv:2510.23868）](/chengYi-xun/img/grpo_variants.png)

| 方法 | 每 Prompt 的 rollout 数 $G$ | 总 rollout 生成量 | 墙钟时间（Qwen-1.5B / MATH） | 平均性能保留率 |
|:---:|:---:|:---:|:---:|:---:|
| 16-GRPO | 16 | 100%（约 1.2M） | 100%（8.53 h） | 基准 |
| **2-GRPO** | **2** | **12.5%**（约 0.15M，即 $2/16$） | **24.0%**（2.05 h） | **98.1%** |

> **注**：总 rollout 生成量 12.5% 指 2-GRPO 每个 Prompt 仅采样 2 条回答（vs 16 条），因此总生成量为 $2/16 = 12.5\%$。墙钟时间 24.0% 对应 Qwen-1.5B / MATH 单组实验；所有实验的平均墙钟时间比约为 21%（论文 Abstract）。两个比值不同是因为训练的计算开销不仅包含 rollout 生成，还包含梯度计算与参数更新等固定成本。

这说明 **GRPO 的核心力量不在于"大组 → 精确基线估计"**，而在于别的什么东西。那到底是什么？

---

# 第一重面孔：GRPO 即在线 DPO（2-GRPO 的视角）

## 用例子重新理解 GRPO 的梯度

还是用数学积分题的例子。让模型用 $G = 4$ 个方式解 $\int_0^1 x^2 dx$，得到 2 个正确（$r = 1$）和 2 个错误（$r = 0$）。在 RLVR 的二值奖励场景中，组内均值 $\bar{r} = 0.5$，标准差 $\sigma = 0.5$，经 z-score 标准化后的优势为：

- 正确回答：$\hat{A}_i = \frac{1 - 0.5}{0.5} = +1$（鼓励）

- 错误回答：$\hat{A}_i = \frac{0 - 0.5}{0.5} = -1$（抑制）

**梯度信号的本质是什么？** 先回顾 GRPO 的完整目标函数（token 级推导见[第二十篇](/chengYi-xun/posts/20-grpo/)；DAPO 对归一化方式的改进见[第二十二篇](/chengYi-xun/posts/22-dapo/)）。为简洁起见，此处将 token 级操作收缩为序列级记号：

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{Q},\, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min \left( \rho_{i,t}(\theta)\, \hat{A}_i,\; \text{clip}(\rho_{i,t}(\theta),\, 1-\varepsilon,\, 1+\varepsilon)\, \hat{A}_i \right) - \beta\, \hat{D}_{\text{KL}}^{(i,t)} \right) \right]
$$

其中各符号含义如下：

- $\rho_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}$　——第 $i$ 个回答第 $t$ 个 token 的重要性采样比率
- $\text{clip}(\rho, 1-\varepsilon, 1+\varepsilon)$　——PPO 风格的裁剪（如 $\varepsilon = 0.2$），在 **token 级** 逐位执行
- $\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r}$　——组内 z-score 标准化优势（序列级标量，对 $t$ 为常数）
- $\hat{D}_{\text{KL}}^{(i,t)}$　——token 级 KL 散度近似估计
- $\frac{1}{|o_i|}$　——按回答长度归一化，使每条回答权重均为 $\frac{1}{G}$

> **关于 GRPO 与 DAPO 的归一化差异**：GRPO 的损失聚合为 $\frac{1}{G}\sum_i \frac{1}{|o_i|}\sum_t L_{i,t}$——**先对每条回答的 token 求平均，再对 $G$ 条回答求平均**，每条回答无论长短权重都是 $\frac{1}{G}$。DAPO 将此改为 $\frac{1}{\sum_i |o_i|}\sum_i \sum_t L_{i,t}$——**直接对全组所有 token 取平均**，使每个 token 而非每条回答等权贡献梯度。这一归一化方式的改变是 DAPO 的四大改进之一，详见[第二十二篇](/chengYi-xun/posts/22-dapo/)。

为便于分析本节的核心论证（对比学习结构），我们暂时忽略裁剪操作（`clip` 和 `min`）和 KL 惩罚项（$- \beta \hat{D}_{\text{KL}}$）。在这些简化下，目标函数退化为最基础的重要度采样（Importance Sampling）形式：

$$
\mathcal{J}_{\text{simplified}}(\theta) = \mathbb{E}_{q \sim \mathcal{Q},\, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})} \hat{A}_i \right]
$$

为了在序列级别进行分析，我们将 token 级别的累加抽象为整个序列级别的概率比率 $\frac{\pi_\theta(o_i | q)}{\pi_{\theta_{\text{old}}}(o_i | q)}$。此时简化的序列级目标函数为：

$$
\mathcal{J}_{\text{seq}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \frac{\pi_\theta(o_i | q)}{\pi_{\theta_{\text{old}}}(o_i | q)} \hat{A}_i \right]
$$

对该简化目标关于参数 $\theta$ 求梯度：

$$
\nabla_\theta \mathcal{J}_{\text{seq}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \hat{A}_i \frac{\nabla_\theta \pi_\theta(o_i | q)}{\pi_{\theta_{\text{old}}}(o_i | q)} \right]
$$

根据强化学习中经典的**对数求导技巧（Log-Derivative Trick）**，即 $\nabla_\theta \pi_\theta(o_i | q) = \pi_\theta(o_i | q) \nabla_\theta \log \pi_\theta(o_i | q)$，代入上式可得：

$$
\nabla_\theta \mathcal{J}_{\text{seq}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \hat{A}_i \frac{\pi_\theta(o_i | q)}{\pi_{\theta_{\text{old}}}(o_i | q)} \nabla_\theta \log \pi_\theta(o_i | q) \right]
$$

在 PPO/GRPO 的每次更新起点，当前策略 $\pi_\theta$ 和采样策略 $\pi_{\theta_{\text{old}}}$ 是相等的（即 $\theta = \theta_{\text{old}}$），此时重要度比率 $\frac{\pi_\theta(o_i | q)}{\pi_{\theta_{\text{old}}}(o_i | q)} = 1$。

此时真实的策略梯度为：

$$
\nabla_\theta \mathcal{J}_{\text{seq}}(\theta) = \mathbb{E}_{q \sim \mathcal{Q},\, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^G \hat{A}_i \nabla_\theta \log \pi_\theta(o_i | q) \right]
$$

**为什么最终公式中去掉了期望符号 $\mathbb{E}$？**
在实际训练中，我们无法计算遍历所有可能 Prompt 和所有可能回答的真实数学期望。因此，强化学习采用**蒙特卡罗采样（Monte Carlo Sampling）**：通过从当前策略中实际采样一个批次的 Prompt 和对应的 $G$ 个回答，用这些样本的平均值来近似真实的期望。去掉期望符号后得到的 $\hat{g}_{\text{GRPO}}$（带有 $\hat{}$ 符号），正是单次采样的**经验梯度估计（Empirical Gradient Estimate）**：

$$
\hat{g}_{\text{GRPO}} = \frac{1}{G}\sum_{i=1}^{G} \hat{A}_i \nabla_\theta \log \pi_\theta(o_i | q)
$$

> **注**：上式省略了裁剪和 KL 项。完整梯度中，$\min(\rho \hat{A}, \text{clip}(\rho)\hat{A})$ 的效果是：当 $\rho$ 偏离 1 过多时截断梯度，防止策略更新过激进。省略这些项不影响下文关于对比结构的分析，因为裁剪仅改变梯度的幅度，不改变其"正样本向上、负样本向下"的方向性。

将回答分为正确组 $\mathcal{G}^+$（$G^+$ 个）和错误组 $\mathcal{G}^-$（$G^-$ 个），上式可以改写为：

$$
\hat{g}_{\text{GRPO}} \propto \underbrace{\frac{1}{G^+}\sum_{o \in \mathcal{G}^+} \nabla_\theta \log \pi_\theta(o | q)}_{\text{增加正确回答的概率}} - \underbrace{\frac{1}{G^-}\sum_{o \in \mathcal{G}^-} \nabla_\theta \log \pi_\theta(o | q)}_{\text{减少错误回答的概率}}
$$

**这与 DPO 的对比学习结构在形式上是一致的。** DPO 的梯度同样具有"增加偏好回答 $y_w$ 的概率、减少非偏好回答 $y_l$ 的概率"的对比形式。主要区别在于：

| | DPO | GRPO |
|---|---|---|
| 正负对来源 | 人类标注的固定偏好对 $(y_w, y_l)$ | 模型在线采样 + 奖励判题动态划分 |
| 对比结构 | 固定 1-vs-1 | 动态 $N$-vs-$M$（$N = G^+$，$M = G^-$） |
| 学习方式 | 离线 | **在线** |

## 控制变量与方差缩减

上面我们看到 GRPO 的梯度是"正样本梯度 $-$ 负样本梯度"的对比形式。一个自然的问题是：**为什么要减去负样本的梯度？只用正样本梯度不行吗？**

仅使用正样本梯度（即 REINFORCE 的做法）当然可以，但**方差会很大**——不同正样本的梯度方向差异很大，导致训练不稳定。GRPO 减去负样本梯度的做法，本质上是蒙特卡罗估计中经典的**控制变量法**（control variate method）。

**定理（控制变量法，Control Variates）**：假设我们要估计随机变量 $X$ 的期望 $\mathbb{E}[X]$。如果存在另一个随机变量 $Y$，满足：

1. $Y$ 的期望 $\mathbb{E}[Y] = \mu_Y$ 已知（或者在对比中可以被消掉）；
2. $Y$ 与 $X$ 存在相关性，设它们的皮尔逊相关系数为 $\rho = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X)\text{Var}(Y)}}$，且 $\rho \neq 0$。

则构造的新估计量 $Z = X - c^*(Y - \mu_Y)$（其中最优系数 $c^* = \frac{\text{Cov}(X, Y)}{\text{Var}(Y)}$）是 $\mathbb{E}[X]$ 的无偏估计，且其方差为：
$$
\text{Var}(Z) = (1 - \rho^2) \text{Var}(X) < \text{Var}(X)
$$
即：只要 $X$ 和 $Y$ 相关，新估计量的方差就**严格小于**原估计量 $X$ 的方差。相关性越强（$\rho^2$ 越接近 1），方差缩减的效果越好。

> **注（定理证明）**：
> 首先，新估计量是无偏的：
> $$
> \mathbb{E}[Z] = \mathbb{E}[X] - c^*(\mathbb{E}[Y] - \mu_Y) = \mathbb{E}[X] - 0 = \mathbb{E}[X]
> $$
> 其次，计算包含任意系数 $c$ 的方差：
> $$
> \text{Var}(X - cY) = \text{Var}(X) + c^2 \text{Var}(Y) - 2c \text{Cov}(X, Y)
> $$
> 对 $c$ 求导并令其为 0，得到使方差最小化的最优系数 $c^* = \frac{\text{Cov}(X, Y)}{\text{Var}(Y)}$。
> 将 $c^*$ 代回方差公式，并利用相关系数定义 $\rho = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X)\text{Var}(Y)}}$，可得最小方差为：
> $$
> \text{Var}(Z^*) = \text{Var}(X) - \frac{\text{Cov}(X, Y)^2}{\text{Var}(Y)} = \text{Var}(X) - \rho^2 \text{Var}(X) = (1 - \rho^2) \text{Var}(X)
> $$

在 GRPO 中：

- $X$ 对应正样本的策略梯度 $\boldsymbol{g}^+ = \nabla_\theta \log \pi_\theta(o^+ | q)$（我们想要的信号）
- $Y$ 对应负样本的策略梯度 $\boldsymbol{g}^- = \nabla_\theta \log \pi_\theta(o^- | q)$（控制变量）
- 因为 $o^+$ 和 $o^-$ 都是**同一模型对同一 Prompt 生成**的回答，它们共享 Prompt 的上下文信息，所以 $\boldsymbol{g}^+$ 和 $\boldsymbol{g}^-$ 之间通常存在**正相关性**

论文将这一直觉形式化为以下命题：

**命题 4.1（原文 Proposition 4.1）**：设 $\boldsymbol{g}^+$ 和 $\boldsymbol{g}^-$ 的相关系数为 $\rho$。若 $\text{Cov}(\boldsymbol{g}^+, \boldsymbol{g}^-) > 0$（即 $\rho > 0$），则对于配对梯度估计 $\boldsymbol{g}^+ - c \cdot \boldsymbol{g}^-$，存在最优系数：

$$
c^* = \frac{\text{Cov}(\boldsymbol{g}^+, \boldsymbol{g}^-)}{\text{Var}(\boldsymbol{g}^-)}
$$

使得方差达到最小值：

$$
\text{Var}(\boldsymbol{g}^+ - c^* \boldsymbol{g}^-) = (1 - \rho^2) \cdot \text{Var}(\boldsymbol{g}^+)
$$

> 此处 $\text{Var}(\cdot)$ 和 $\text{Cov}(\cdot, \cdot)$ 分别表示梯度向量协方差矩阵的迹（trace），$\rho$ 为对应的相关系数。证明见原文 Appendix B.5。
> **关于符号滥用的补充说明**：在机器学习优化理论中，对于高维随机向量（如参数量达数十亿的策略梯度 $\boldsymbol{g}$），其严格意义上的协方差 $\text{Cov}(\boldsymbol{g}, \boldsymbol{g})$ 是一个巨大的矩阵。但当我们讨论“梯度估计的方差”时，我们关心的是梯度向量偏离期望值的**总幅度**（即所有参数维度方差的总和），这在数学上等于偏差向量 $L_2$ 范数平方的期望：
> $$ \mathbb{E}[\|\boldsymbol{g} - \mathbb{E}[\boldsymbol{g}]\|_2^2] = \mathbb{E}[(\boldsymbol{g} - \mathbb{E}[\boldsymbol{g}])^T (\boldsymbol{g} - \mathbb{E}[\boldsymbol{g}])] = \text{Tr}(\text{Cov}(\boldsymbol{g}, \boldsymbol{g})) $$
> 同理，两个向量的“标量协方差”（内积的期望）为 $\text{Tr}(\text{Cov}(\boldsymbol{g}^+, \boldsymbol{g}^-))$。为了公式简洁，论文作者直接用标量符号 $\text{Var}(\cdot)$ 和 $\text{Cov}(\cdot, \cdot)$ 来表示这些迹（Trace），从而将高维向量的运算转化为标量运算，得出标量系数 $c^*$ 和方差缩减比例 $(1 - \rho^2)$。

**关键含义**：由于 $0 < \rho \leq 1$，因此 $(1 - \rho^2) < 1$，配对估计的方差**严格小于**仅使用正样本梯度 $\boldsymbol{g}^+$ 的方差。相关性 $\rho$ 越大，方差缩减越显著。这解释了 GRPO 为什么要在**同一 Prompt 的同一组内**做对比——目的不是为了更精确的均值估计，而是利用同源配对的正相关性来降低策略梯度估计量的方差。

## 2-GRPO：最小对比单元

既然 GRPO 的核心是对比，那对比的最小单元就是 **2 个 rollout**。当 $G = 2$ 且 $r_1 \neq r_2$ 时（一对一错），经 z-score 标准化后优势退化为：

$$
\hat{A}_1 = +1, \quad \hat{A}_2 = -1 \quad \text{（二值奖励下 $n=2$ 的 z-score 标准化恒为 $\pm 1$）}
$$

其梯度结构与**在线版的 DPO 更新**一致：增加正确回答的概率，减少错误回答的概率。但二者在损失函数形式上仍有区别——2-GRPO 使用 PPO 风格的裁剪代理目标，而 DPO 使用对数 sigmoid 损失。

**2-GRPO 的训练目标**：

$$
J_{\text{2-GRPO}}(\theta) = \mathbb{E}_{q,\, (o_1, o_2) \sim \pi_{\theta_{\text{old}}}}\left[\mathbf{1}(r_1 \neq r_2) \cdot \frac{1}{2}\left(\frac{1}{|o^+|}\sum_t C_\varepsilon^+(\rho_t^+) - \frac{1}{|o^-|}\sum_t C_\varepsilon^-(\rho_t^-)\right)\right]
$$

其中 $C_\varepsilon^{\pm}$ 是 PPO 风格的裁剪，$o^+$ 和 $o^-$ 是根据奖励划分的正确/错误回答。

### 2-GRPO 的完整实现

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

# 模型定义: 与标准 GRPO 完全相同
actor = AutoModelForCausalLM.from_pretrained("sft_checkpoint")
ref_model = AutoModelForCausalLM.from_pretrained("sft_checkpoint")
# 参考模型固定，仅提供 log π_ref 基线
ref_model.requires_grad_(False)
# 只更新 actor，与 GRPO 一致
optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-6)

G = 2  # 核心区别: 只用 2 个 rollout
# PPO 式裁剪区间，限制单步策略变化（与多 epoch 更新配套）
clip_range = 0.2
# 隐式奖励 / KL 惩罚的温度尺度，控制相对 ref 的强度
beta = 0.04

# 外层 RL 迭代：采样 → 算优势 → 多 epoch 更新
for step in range(total_steps):
    # 2-GRPO 用更多 Prompt 补偿 G 的减少 (保持总 rollout 数不变)
    # 标准 GRPO: 32 prompts × 16 rollouts = 512 总量
    # 2-GRPO:   256 prompts × 2 rollouts  = 512 总量 (Prompt 多样性 ↑)
    prompts_batch = sample_prompts(dataset, batch_size=256)

    # 采样阶段不反传，避免把随机性写入梯度
    with torch.no_grad():
        # 展平存放每组 G=2 的轨迹
        all_responses = []
        all_rewards = []
        all_old_logps = []
        all_ref_logps = []
        # 批量 Prompt：2-GRPO 用更多 q 换更大对比覆盖面
        for prompt in prompts_batch:
            # 每个 q 上采 G=2 条回答，形成最小对比对
            for _ in range(G):
                response = actor.generate(prompt, do_sample=True)
                reward = reward_fn(prompt, response)
                # 采样时策略的序列 log 概率，供 PPO 比率
                old_logp = compute_log_probs(actor, prompt, response)
                # ref 下同一回答的 log 概率，用于 KL 项
                ref_logp = compute_log_probs(ref_model, prompt, response)
                all_responses.append(response)
                all_rewards.append(reward)
                all_old_logps.append(old_logp)
                all_ref_logps.append(ref_logp)

    # G=2 时的优势计算: 退化为简单的符号翻转
    rewards = torch.tensor(all_rewards).reshape(-1, 2)  # (B, 2)
    # 只保留一对一错的组
    valid_mask = rewards[:, 0] != rewards[:, 1]
    # 若全同号则无对比信号，跳过本步
    if valid_mask.sum() == 0:
        continue

    mean_r = rewards[valid_mask].mean(dim=1, keepdim=True)  # (B', 1)
    std_r = rewards[valid_mask].std(dim=1, keepdim=True)  # (B', 1)
    advantages = (
        (rewards[valid_mask] - mean_r) / (std_r + 1e-8)
    ).reshape(-1)  # (B'*2,)
    # 对于二值奖励 (1, 0), z-score 标准化后 advantages 恰好是 (+1, -1)

    # 多 epoch 更新 (与标准 GRPO 相同)
    for epoch in range(K_epochs):
        # ... 裁剪损失 + KL 惩罚，代码结构与 GRPO 完全一致
        # 此处接 clipped surrogate + β·KL(π_θ||π_ref)
        pass
```

**2-GRPO 的核心优势**：在相同总 rollout 预算下，2-GRPO 可覆盖 256 个不同的 Prompt（vs 标准 GRPO 的 32 个），**Prompt 多样性提升 8 倍**。论文的核心论点是：GRPO 的有效性源于其隐含的**对比学习机制**（正负样本配对），而非大组带来的优势估计精度；组规模仅影响对比目标的蒙特卡罗估计量的方差，不改变优化方向的无偏性（原文 Section 5, Lemma 5.2）。

**总结**：GRPO 的第一重面孔——其梯度结构与**在线版 DPO** 共享相同的对比学习框架（增加正样本概率、抑制负样本概率），核心有效机制是**对比配对带来的方差缩减**，而非大组带来的优势估计精度。

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

**定理 4.3（f-HAL & f-GRPO 的收敛性，摘自 f-GRPO 论文）**：在 $G \to \infty$ 的渐近极限下：

1. **散度估计**：f-GRPO 的负损失值与奖励诱导的正/负分布之间的 f-散度成正比：

$$
-\mathcal{L}_{\text{f-GRPO}}(\theta^{(t+1)}) \propto D_f(D^+_{(r,\theta^{(t)})} \| D^-_{(r,\theta^{(t)})})
$$

2. **平均奖励单调改进**：在弱奖励-密度对应假设下（正面分布关于奖励单调），每次迭代的平均奖励严格递增，直到收敛到最大奖励。

**对比标准 GRPO**：

**定理 4.4（GRPO 的不动点特征，摘自 f-GRPO 论文）**：标准 GRPO 的隐式更新等价于按标准化奖励对参考策略做指数重加权：

$$
\pi_{\theta_{\text{GRPO}}^{(t+1)}}(y|x) \propto \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x,y) - \mu_r^{(t)}}{\beta \cdot \sigma_r^{(t)}}\right)
$$

这意味着 GRPO 会给低于平均奖励的回答**非零但小的权重**（指数衰减但不归零），而 f-GRPO 使用满足 $g^{-1}(f'_\infty) = \infty$ 的典范链接时，可以实现更激进的集中——将概率完全集中到高奖励回答上。

## 实验结果

f-GRPO 在 Qwen2.5-Math 1.5B/7B 上的数学推理实验中，**所有测试的 f-散度变体都优于标准 GRPO**（数据摘自 f-GRPO 论文 Table 2，LIMR 数据集 + Qwen-1.5B）：

| f-散度 | Relative Overall（1.5B）$\uparrow$ | 与 GRPO 差值 | 平均排名 $\downarrow$ |
|:---:|:---:|:---:|:---:|
| GRPO（KL baseline） | 74.26 | — | 5.2 |
| Pearson $\chi^2$ | **86.49** | **+12.23** | **2.6** |
| Total Variation | 83.11 | +8.85 | 3.8 |
| 逆 KL | 82.47 | +8.21 | 3.0 |
| KL（前向） | 81.83 | +7.57 | 4.4 |
| Hellinger | 81.11 | +6.85 | 4.2 |
| JS | 78.99 | +4.73 | 4.2 |

> **指标说明**（论文 Table 2）：实验基于 Qwen2.5-Math-1.5B 在 LIMR 数据集上训练。Relative Overall 定义为各基准（GSM8K、MATH-500、AMC 2023、AIME 2024、AIME 2025）上 Pass@1 准确率分别做 min-max 归一化至 $[0, 100]$（以 Base 模型为 0、最佳方法为 100）后的算术平均。

Pearson $\chi^2$ 散度在 LIMR 数据集 + Qwen-1.5B 设定下表现最优（Relative Overall 86.49，平均排名 2.6）。但值得注意的是，论文 Table 3 显示不同数据集和模型规模下最优散度并不一致——例如在 GSM8K + 7B 设定下，Jensen–Shannon 以 95.08 领先。论文推测 Pearson $\chi^2$ 在 LIMR/1.5B 场景的优势可能与其对奖励分布尾部差异的二次敏感性有关。

### f-GRPO 的具体实现

不同 f-散度的选择对应不同的链接函数 $g$ 和共轭函数 $f^*$。以下是几种常见散度的实现：

```python
import torch
import torch.nn.functional as F

def f_grpo_loss(
    log_probs,
    ref_log_probs,
    old_log_probs,
    advantages,
    loss_mask,
    f_type="pearson_chi2",
    beta=0.1,
):
    """f-GRPO 损失函数，支持多种 f-散度（正/负侧链接不同；标准 GRPO 两侧裁剪相同）。

    Args:
        log_probs: 当前策略 log π_θ，与 token 对齐，形状 (*, T) 或与 loss_mask 广播一致。
        ref_log_probs: 参考 log π_ref，形状同 log_probs。
        old_log_probs: 采样时 log π_old，可与 PPO 比率联用（本损失核心在 f 侧）。
        advantages: 组内标准化优势 A_i，形状 (*, T) 或与 loss_mask 对齐。
        loss_mask: 有效 token mask，形状 (*, T)，float/bool。
        f_type: f-散度名称，如 ``pearson_chi2``、``hellinger``、
            ``reverse_kl``、``js``。
        beta: 隐式奖励缩放，r_θ = β·log(π_θ/π_ref)。

    Returns:
        标量，平均每有效 token 的 f-GRPO 损失。
    """
    # 隐式奖励 (与 DPO 相同的定义)
    implicit_reward = beta * (log_probs - ref_log_probs)  # (*, T)

    # 将回答分为正侧 (A > 0) 和负侧 (A ≤ 0)
    # 高于组均：f-散度变分“正侧”
    pos_mask = (advantages > 0).float().unsqueeze(-1) * loss_mask
    # 不高于均值：变分“负侧”/ 共轭支
    neg_mask = (advantages <= 0).float().unsqueeze(-1) * loss_mask

    # Pearson χ²：对尾部差异二次放大，利于稀疏二值奖励
    if f_type == "pearson_chi2":
        # f(t) = (t-1)², f*(s) = s + s²/4
        # 正侧: g(r) = r (identity link)
        # 负侧: f*(g(r)) = r + r²/4
        pos_loss = -implicit_reward
        neg_loss = implicit_reward + implicit_reward ** 2 / 4

    # Hellinger：对称、对中等密度差异敏感
    elif f_type == "hellinger":
        # f(t) = (√t - 1)², f*(s) = s/(1-s)
        # 正侧: g(r) = 1 - exp(-r) (饱和链接)
        # 负侧: f*(g(r))
        # 饱和链接，抑制过大隐式奖励导致的梯度爆炸
        g_r = 1.0 - torch.exp(-implicit_reward)
        pos_loss = -g_r
        # 负侧走 f*∘g，鼓励与 ref 的 Hellinger 几何一致
        neg_loss = g_r / (1.0 - g_r + 1e-8)

    # 逆 KL：倾向模式寻求、分布更尖锐
    elif f_type == "reverse_kl":
        # f(t) = -ln t, f*(s) = -1 - ln(-s)
        pos_loss = -implicit_reward
        # 共轭支要求自变量为负，截断防 log(0)
        neg_loss = -1.0 - torch.log(-implicit_reward.clamp(max=-1e-8))

    # JS：有界对称散度，训练较平滑
    elif f_type == "js":
        # Jensen-Shannon: f(t) = t ln t - (t+1) ln((t+1)/2)
        # 恢复 π_θ/π_ref 密度比，代入 JS 变分形式
        ratio = torch.exp(implicit_reward / beta)
        pos_loss = -(
            ratio * implicit_reward / beta
            - (ratio + 1) * torch.log((ratio + 1) / 2)
        )
        # 负侧与正侧对称配对（与论文 ψ 分段对应）
        neg_loss = -pos_loss

    else:
        raise ValueError(f"未知 f-散度类型: {f_type}")

    # 用绝对优势值作为权重 (优势越大/越小，权重越高)
    abs_adv = advantages.abs().unsqueeze(-1)
    loss = pos_loss * pos_mask * abs_adv + neg_loss * neg_mask * abs_adv
    total_tokens = loss_mask.sum()
    return loss.sum() / total_tokens
```

**使用方式与标准 GRPO 完全一致**——只需将 `grpo_loss(...)` 替换为 `f_grpo_loss(..., f_type="pearson_chi2")`：

```python
# 在标准 GRPO 训练循环中，将损失函数替换为（接口与 grpo_loss 对齐，便于 A/B）
loss = f_grpo_loss(
    new_log_probs,
    ref_log_probs,
    old_log_probs,
    advantages,
    loss_mask,
    f_type="pearson_chi2",
    beta=0.1,
)
# 仅换损失：仍用组内优势 A_i 与 mask
# 其余部分（采样、优势计算、优化器）完全不变
# rollout、基线标准化、Adam 等流程与 KL-GRPO 相同
```

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

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(\frac{r(x,y)}{\beta})$ 是配分函数——对于给定的 $x$，$Z(x)$ 不随 $y$ 变化，但需要对整个输出空间求和（或积分），计算上不可行（intractable）。这正是 DPO 无法做"单点"奖励匹配、只能做配对比较的根本原因。

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

GIFT 论文（arXiv:2510.23868, Table 2）报告，在 7B 模型上 GIFT 的 AlpacaEval LC Win Rate 为 48.77（vs GRPO 的 35.33），Arena-Hard Win Rate 为 72.43（vs GRPO 的 38.25）。需要指出，该对比中 GRPO 使用的是论文作者的复现版本，不同实现的超参数选择可能影响结果。

---

# 三重面孔的统一视角

| 视角 | 核心思想 | 代表方法 |
|:---:|:---|:---:|
| **对比学习** | GRPO ≈ 在线版 DPO，核心是正负对比 | 2-GRPO |
| **f-散度优化** | GRPO 的正负划分 ≈ f-散度变分表示的两侧 | f-GRPO |
| **隐式奖励回归** | 组内归一化消除配分函数，变为 MSE 回归 | GIFT |

这三种视角不是互斥的——它们从不同的数学工具出发，揭示了 **GRPO 优化行为的不同侧面**。理解这些联系，对算法设计有直接启示：

1. **设计更高效的算法**（2-GRPO：减少 rollout 数量）

2. **选择更好的散度**（f-GRPO：用 Pearson $\chi^2$ 替代 KL）

3. **构建更稳定的训练**（GIFT：MSE 替代裁剪策略梯度）

### GIFT 的完整实现

**Step 1: 模型定义 — 与 GRPO/DPO 相同**

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

actor = AutoModelForCausalLM.from_pretrained("sft_checkpoint")
ref_model = AutoModelForCausalLM.from_pretrained("sft_checkpoint")
# 参考模型不训练，消除 Z(x) 靠组内归一化而非算 Z
ref_model.requires_grad_(False)
optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-6)

# 每个 Prompt 采 G 条；组内均值/方差归一化用于消去 Z(x)
G = 8
```

**Step 2: GIFT 损失函数**

```python
def gift_loss(log_probs, ref_log_probs, rewards):
    """GIFT 损失：归一化隐式奖励回归归一化外部奖励。

    隐式奖励 = β·log(π_θ/π_ref) + β·log Z(x)；组内归一化后 β·log Z(x) 被消除。

    Args:
        log_probs: log π_θ(y|x)，形状 (B, G)，可带梯度。
        ref_log_probs: log π_ref(y|x)，形状 (B, G)，固定分支。
        rewards: 外部 r_φ(x, y)，形状 (B, G)。

    Returns:
        标量 MSE，默认对元素取 mean。
    """
    # 隐式奖励 (不含 β 和 Z(x)，它们会在归一化中被消除)
    implicit_rewards = log_probs - ref_log_probs  # (B, G)

    # 组内归一化 (在 G 维标准化 → 消除 Z(x))
    impl_mean = implicit_rewards.mean(dim=1, keepdim=True)  # (B, 1)
    impl_std = implicit_rewards.std(dim=1, keepdim=True) + 1e-8  # (B, 1)
    norm_implicit = (implicit_rewards - impl_mean) / impl_std  # (B, G)

    rew_mean = rewards.mean(dim=1, keepdim=True)  # (B, 1)
    rew_std = rewards.std(dim=1, keepdim=True) + 1e-8  # (B, 1)
    norm_rewards = (rewards - rew_mean) / rew_std  # (B, G)

    # detach 外部奖励：不把梯度回传到 RM
    loss = F.mse_loss(norm_implicit, norm_rewards.detach())
    return loss
```

**Step 3: 完整训练循环**

```python
# 外层迭代：每步先 rollout 再逐 prompt GIFT 更新
for step in range(total_steps):
    # on-policy 批次大小可与 GRPO 对齐
    prompts_batch = sample_prompts(dataset, batch_size=32)

    # --- 阶段 1: 在线采样 (与 GRPO 完全相同) ---
    # 预留与 GRPO 一致的缓存结构（可存 response）
    all_log_probs, all_ref_log_probs, all_rewards = [], [], []

    # 生成与 ref 概率在采样时无需对 θ 求导
    with torch.no_grad():
        for prompt in prompts_batch:
            group_logps, group_ref_logps, group_rewards = [], [], []
            for _ in range(G):
                response = actor.generate(prompt, do_sample=True)
                reward = reward_fn(prompt, response)
                ref_logp = compute_seq_log_prob(ref_model, prompt, response)
                group_rewards.append(reward)
                group_ref_logps.append(ref_logp)
            all_rewards.append(torch.tensor(group_rewards))
            all_ref_log_probs.append(torch.stack(group_ref_logps))

    rewards = torch.stack(all_rewards)  # (B, G)
    ref_logps = torch.stack(all_ref_log_probs)  # (B, G)

    # 过滤零方差组 (与 DAPO 类似)
    valid = rewards.std(dim=1) > 0
    if valid.sum() == 0:
        continue

    # --- 阶段 2: GIFT 更新 ---
    # GIFT 不需要"多 epoch 更新"和"重要性采样比率"!
    # 它直接对当前策略做前向传播计算隐式奖励

    for prompt_idx in valid.nonzero(as_tuple=True)[0]:
        prompt = prompts_batch[prompt_idx]
        # G 条解码结果；须与阶段 1 写入的 all_responses 对齐（示意伪代码）
        responses = all_responses[prompt_idx]

        # 用当前策略重新计算对数概率 (需要梯度)
        log_probs = torch.stack(
            [
                compute_seq_log_prob(actor, prompt, resp)
                for resp in responses
            ]
        ).unsqueeze(0)  # (1, G)

        ref_lps = ref_logps[prompt_idx].unsqueeze(0)  # (1, G)
        rews = rewards[prompt_idx].unsqueeze(0)  # (1, G)

        loss = gift_loss(log_probs, ref_lps, rews)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        optimizer.step()
```

> **GIFT vs GRPO 训练的关键区别**：
>
> - **没有裁剪**：GIFT 用 MSE 替代了 PPO 的裁剪机制，不需要 $\varepsilon$ 超参数。
> - **没有重要性采样**：GIFT 不需要 `old_log_probs` 和比率 `ratio = exp(new - old)`。
> - **没有 KL 惩罚**：KL 约束被隐含在参考模型的对数概率中。
> - **凸损失函数**：MSE 是凸函数，比裁剪策略梯度（分段线性）更稳定。
> - 代价是 GIFT 需要每步都重新前向传播（不能像 PPO/GRPO 那样多 epoch 复用旧数据）。

> 参考资料：
>
> 1. [It Takes Two: Your GRPO Is Secretly DPO](https://arxiv.org/abs/2510.00977)
> 2. [f-GRPO and Beyond: Divergence-Based RL for General LLM Alignment](https://arxiv.org/abs/2602.05946)
> 3. [GIFT: Group-relative Implicit Fine Tuning](https://arxiv.org/abs/2510.23868)

> 下一篇：[笔记｜生成模型（二十三）：SuperFlow 与图像生成 RL 的统一框架](/chengYi-xun/posts/24-superflow/)
