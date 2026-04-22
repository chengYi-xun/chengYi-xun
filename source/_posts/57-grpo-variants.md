---
title: 笔记｜强化学习（七）：GRPO 的三重面孔——从 2-GRPO 到 f-GRPO 与 GIFT
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
> ⬅️ 上一篇：[笔记｜强化学习（六）：DAPO：从 GRPO 到大规模推理 RL 的工程实践](/chengYi-xun/posts/56-dapo/)
>
> ➡️ 下一篇：[笔记｜强化学习（八）：SuperFlow 与图像生成 RL 前沿（2026）](/chengYi-xun/posts/58-superflow/)

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

**梯度信号的本质是什么？** 先回顾 GRPO 的完整目标函数（token 级推导见[第四篇](/chengYi-xun/posts/54-grpo/)；DAPO 对归一化方式的改进见[第六篇](/chengYi-xun/posts/56-dapo/)）。为简洁起见，此处将 token 级操作收缩为序列级记号：

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{Q},\, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min \left( \rho_{i,t}(\theta)\, \hat{A}_i,\; \text{clip}(\rho_{i,t}(\theta),\, 1-\varepsilon,\, 1+\varepsilon)\, \hat{A}_i \right) - \beta\, \hat{D}_{\text{KL}}^{(i,t)} \right) \right]
$$

其中各符号含义如下：

- $\rho_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}$　——第 $i$ 个回答第 $t$ 个 token 的重要性采样比率
- $\text{clip}(\rho, 1-\varepsilon, 1+\varepsilon)$　——PPO 风格的裁剪（如 $\varepsilon = 0.2$），在 **token 级** 逐位执行
- $\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r}$　——组内 z-score 标准化优势（序列级标量，对 $t$ 为常数）
- $\hat{D}_{\text{KL}}^{(i,t)}$　——token 级 KL 散度近似估计
- $\frac{1}{|o_i|}$　——按回答长度归一化，使每条回答权重均为 $\frac{1}{G}$

> **关于 GRPO 与 DAPO 的归一化差异**：GRPO 的损失聚合为 $\frac{1}{G}\sum_i \frac{1}{|o_i|}\sum_t L_{i,t}$——**先对每条回答的 token 求平均，再对 $G$ 条回答求平均**，每条回答无论长短权重都是 $\frac{1}{G}$。DAPO 将此改为 $\frac{1}{\sum_i |o_i|}\sum_i \sum_t L_{i,t}$——**直接对全组所有 token 取平均**，使每个 token 而非每条回答等权贡献梯度。这一归一化方式的改变是 DAPO 的四大改进之一，详见[第六篇](/chengYi-xun/posts/56-dapo/)。

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

**2-GRPO 的核心优势**：在相同总 rollout 预算下，2-GRPO 可覆盖 256 个不同的 Prompt（vs 标准 GRPO 的 32 个），**Prompt 多样性提升 8 倍**。论文的核心论点是：GRPO 的有效性源于其隐含的**对比学习机制**（正负样本配对），而非大组带来的优势估计精度；增大组规模 $G$ 虽然能进一步降低梯度估计量的蒙特卡罗方差（按 $O(1/G)$ 衰减），但不改变优化方向的无偏性——$G = 2$ 已经捕获了对比配对这一最核心的方差缩减机制（原文 Section 5, Lemma 5.2；另参 arXiv:2603.01162 关于 GRPO 梯度作为 U-统计量的分析）。

**总结**：GRPO 的第一重面孔——其梯度结构与**在线版 DPO** 共享相同的对比学习框架（增加正样本概率、抑制负样本概率），核心有效机制是**对比配对带来的方差缩减**，而非大组带来的优势估计精度。

## 2-GRPO 的局限性：奖励偏差的符号放大

上一节展示了 2-GRPO 在效率上的巨大优势。但"硬币的另一面"是：**当奖励函数存在系统性偏差（bias）时，$G = 2$ 的 pairwise 结构会将偏差放大为确定性的错误训练信号**。这一问题在 $G$ 较大时被统计平均所缓解，但在 $G = 2$ 时几乎没有缓冲余地。

### 区分噪声与偏差

首先需要区分奖励函数的两类不完美：

- **随机噪声（noise）**：奖励对同一回答有时打高、有时打低，但**期望值正确**。在蒙特卡罗意义下，噪声会随采样次数增加而被平均掉。
- **系统偏差（bias）**：奖励**持续且稳定地**偏向某类特征（如更长的回答、更礼貌的语气、特定格式），即使这些特征与真实质量无关。偏差不会随采样增加而消失。

Shen et al.（2026）在 *How RLHF Amplifies Sycophancy*（arXiv:2602.01002）中给出了形式化分析：RLHF 中行为漂移的方向由**学习到的奖励与偏差信号之间的协方差**决定。当该协方差非零时，策略优化会系统性地放大偏差——即便偏差很小。

### $G = 2$ 中的符号放大机制

回顾 2-GRPO 在二值奖励下的优势计算——当 $r_1 \neq r_2$ 时，z-score 标准化恒为 $\pm 1$。但当奖励是连续值（来自 RM 打分）时，情况更加微妙。考虑这样的场景：

| 回答 | 实际质量 | RM 打分 | 2-GRPO 优势 |
|:---:|:---:|:---:|:---:|
| $o_1$（正确简洁） | 高 | 0.6 | $-1$ |
| $o_2$（冗长啰嗦） | 低 | 0.8 | $+1$ |

RM 因长度偏差将质量更低的 $o_2$ 打了高分。在 2-GRPO 中，$\text{sign}(r_2 - r_1) > 0$ 直接决定了 $o_2$ 是正样本、$o_1$ 是负样本。梯度更新的方向是：

$$
\hat{g} \propto \nabla_\theta \log \pi_\theta(o_2 | q) - \nabla_\theta \log \pi_\theta(o_1 | q)
$$

**错误的排序直接变成了确定性的梯度方向**，增加低质量回答的概率、抑制高质量回答的概率。

这里的核心问题是 **符号放大（sign amplification）**：RM 的连续评分误差（$0.8$ vs $0.6$，差异仅 $0.2$）通过 z-score 标准化被映射为离散的 $\pm 1$ 决策，微小的偏差被放大为最大强度的训练信号。形式化地：

$$
\hat{A}_i = \text{sign}(r_i - \bar{r}) \quad (\text{当 } G = 2 \text{ 且 } r_1 \neq r_2)
$$

设 RM 在单次 pairwise 比较中给出错误排序的概率为 $p$（由系统偏差导致），则每一步优化都有 $p$ 的概率沿**完全相反的方向**更新策略——且更新幅度与正确方向完全相同（都是 $\pm 1$）。

### 为什么 $G = 16$ 能缓解

当 $G = 16$ 时，同一 Prompt 的 16 个回答提供了更丰富的统计信息：

1. **排序鲁棒性**：即使 RM 对部分回答的评分有偏，组内排序仍然大概率保留正确趋势。RM 需要在**多数** pairwise 比较上犯错才能完全颠覆排序。
2. **优势的连续性**：回顾 z-score 标准化公式 $\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r}$，其中 $\bar{r} = \frac{1}{G}\sum_{j=1}^G r_j$，$\sigma_r = \sqrt{\frac{1}{G}\sum_{j=1}^G (r_j - \bar{r})^2}$ 为组内奖励的总体标准差（population standard deviation），度量 $G$ 个奖励围绕均值 $\bar{r}$ 的离散程度。当 $G = 2$ 时，分子 $r_i - \bar{r} = \pm\frac{r_1 - r_2}{2}$，分母 $\sigma_r = \frac{|r_1 - r_2|}{2}$，二者的绝对值恒等，因此 $\hat{A}_i = \pm 1$——**奖励差异的幅度信息被完全消除**，无论 $|r_1 - r_2| = 0.01$ 还是 $0.9$，结果都是相同的 $\pm 1$。而当 $G \geq 3$ 时，分母 $\sigma_r$ 由**全组所有 $G$ 个奖励共同决定**，不再与单个分子 $r_i - \bar{r}$ 成比例，因此不同回答的优势值保持**连续且不等**（如 $G = 16$ 时可能得到 $+1.5, +0.8, -0.3, -1.2, \ldots$）。这意味着梯度权重与回答的相对质量成比例，单个评分错误只会略微扭曲优势分布，而非完全翻转信号。
3. **稀释效应**：一个被错误高估的回答只占 $1/16$ 的梯度权重，其影响被其余 15 个样本的正确信号所稀释。

Pref-GRPO 论文（arXiv:2508.20751）将这一现象命名为**"虚幻优势"（illusory advantage）**：当组内奖励差异很小时，组内标准化会不成比例地放大这些微小差异，产生虚假的对比信号。在 $G = 2$ 时，这一问题最为严重——任何非零的奖励差异都会被放大为最大强度的 $\pm 1$ 信号。

### 缓解策略

若在非 RLVR（二值奖励）场景中使用 2-GRPO，以下策略可降低偏差放大的风险：

**策略一：置信度阈值过滤（Margin Filtering）**。丢弃奖励差异过小的 pair——这些 pair 最容易受 RM 偏差影响：

$$
\text{valid}_i = \mathbf{1}\{|r_1 - r_2| > \delta\}
$$

其中 $\delta$ 是人为设定的最小置信度阈值（如 $\delta = 0.2$）。

**策略二：软优势替代硬符号**。用连续函数替代 $\pm 1$ 的硬判定，保留奖励差异的幅度信息：

$$
\hat{A}_i = \tanh\left(\frac{r_i - \bar{r}}{\tau}\right)
$$

其中 $\tau$ 是温度参数。当 $\tau \to 0$ 退化为 $\text{sign}$ 函数（硬判定），$\tau$ 较大时接近线性（软判定）。

**策略三：KL 正则化**。2-GRPO 本身已包含的 $\beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ 项提供了一定的安全网——即使 RM 持续给出偏差信号，KL 约束也会阻止策略过度偏离参考模型。

**策略四：使用可验证奖励（Verifiable Rewards）**。这也是 "It Takes Two" 论文实验设定的关键前提——其所有实验均使用**数学推理的程序化判题器**（$r \in \{0, 1\}$），而非学习到的 RM。在二值可验证奖励下，不存在系统偏差问题（答案要么对要么错），2-GRPO 的符号放大反而是一种优势（将正确/错误的二分转化为最大对比信号）。

> **要点**：2-GRPO 的效率优势在 RLVR（Reinforcement Learning with Verifiable Rewards）场景下最为显著且安全，因为可验证奖励消除了 RM 偏差的风险。在依赖学习型 RM 的场景（如开放式对话、创意写作），需要格外注意偏差放大问题，并考虑上述缓解策略或改用更大的 $G$。

---

# 第二重面孔：从 KL 到任意 f-散度（f-GRPO）

## 为什么要超越 KL 散度？

在标准的 GRPO 中，为了防止模型在追求高分的路上“走火入魔”（比如输出乱码但碰巧得分高），我们会用一个“紧箍咒”——**KL 散度**，来限制模型：**“你可以去追求高分，但你的回答风格不能偏离原来的参考模型（老模型）太远。”**

数学公式长这样：
$$
\max_\theta \mathbb{E}[r(x, y)] - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
$$

但问题是，**KL 散度只是众多“紧箍咒”中的一种**。不同的散度（如 KL、逆 KL、JS、Pearson 等）衡量“两个模型差多远”的尺子是不一样的。换一把尺子，往往能带来意想不到的训练效果。

### 插曲：KL 散度的“偏心”惩罚（覆盖 vs 专精）

要理解换尺子的好处，我们先来看看 KL 散度这把尺子有多“偏心”。

从数学公式上看，KL 散度 $D_{\text{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$ 的惩罚是**极度不对称**的。它对两种“缺质量”的情况态度截然不同：

- **情况 1**：当 $P(x) > 0$ 但 $Q(x) \to 0$ 时，$P(x) \log \frac{P(x)}{Q(x)} \to +\infty$（**惩罚爆炸**）。
- **情况 2**：当 $P(x) = 0$ 但 $Q(x) > 0$ 时，$0 \cdot \log \frac{0}{Q(x)} = 0$（**完全不管**）。

一句话概括：**KL 散度只在乎 $P$（第一个参数）有值的地方，在那里 $Q$（第二个参数）绝不能缺席；而 $P$ 没值的地方，$Q$ 想怎样都行。**

这种不对称性，在变分推断和实际训练中会导致两种截然不同的行为。假设参考模型（老模型 $p$）会用**三种不同的方法**解一道数学题，现在我们要训练新模型 $q$。

1. **前向 KL（$\min_q D_{\text{KL}}(p \| q)$）——“全能型学霸”要求（Mode-Covering / 模式覆盖）**
   - **数学视角**：新模型 $q$ 放在了第二个参数位置。根据上面的“情况 1”，凡是老模型 $p$ 有值的地方（老模型会的解法），$q$ 绝不能为 0，否则惩罚爆炸。
   - **通俗理解**：老模型会的 3 种解法，新模型**必须都要会**。新模型被迫“摊大饼”，努力覆盖所有的解法。代价是，为了兼顾所有解法，它可能每种解法都学得不够精，甚至把中间的错误解法也学进去了。

2. **反向 KL（$\min_q D_{\text{KL}}(q \| p)$）——“偏科型天才”要求（Mode-Seeking / 模式专精）**
   - **数学视角**：新模型 $q$ 放在了第一个参数位置。根据上面的“情况 2”，如果 $q=0$，惩罚直接为 0。这意味着 $q$ 可以随意放弃 $p$ 的某些部分而不受惩罚。
   - **通俗理解**：新模型只要精通**其中一种解法**就行。对于老模型会的其他 2 种解法，新模型完全放弃也不会受到任何惩罚。结果是，新模型会迅速锁定得分最高的那一种解法，彻底抛弃其他解法。

![前向 KL vs 反向 KL 的拟合行为（图源：Le, 2017）。实线为双峰目标（老模型）；虚线为前向 KL（全都要，但不精确）；点线为反向 KL（只抓一个峰，彻底忽略另一个）。从左到右双峰间距递增，差异越发显著。](/chengYi-xun/img/forward_reverse_kl.png)

### 回到 GRPO：标准 GRPO 培养的是“偏科型天才”

在标准 GRPO 的公式 $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ 中，正在训练的新模型 $\pi_\theta$ 放在了前面，这在数学上对应的是**反向 KL**。

因此，标准 GRPO 的天性是 **Mode-Seeking（模式专精）** 的：
- 训练结束后，模型很可能只会用**最容易拿高分的那一种方法**来解题，完全忘记了其他解法。

**这会带来什么问题？**
在实际训练中，这种“只盯准一个峰”的特性，虽然能快速拿高分，但有时会导致模型过早地“塌缩”到某一种回答模式上，丧失了多样性，或者在探索过程中容易陷入死胡同。

> **前沿研究补充**：Chen et al. (2025) 发现，在强化学习中，不管你用前向还是反向 KL，只要追求奖励，模型最终都会变成“偏科型天才”（只用得分最高的方法）。所以，换散度的真正意义不在于改变最终结果，而在于**改变训练过程中的梯度方向**。不同的散度能让模型在寻找最优解的路上走得更稳、更快。

这也解释了为什么 f-GRPO 换用不同散度能在实验中带来改进——不同散度让训练的每一步走不同的梯度方向，有些方向更高效、更稳定。

| 散度（紧箍咒类型） | 在 GRPO 中的通俗表现 |
|:---:|:---|
| **KL（反向）** | **专精型**（可丢弃其他解法），标准 GRPO 的默认选择 |
| **逆 KL（前向）** | **全能型**（必须覆盖老模型的所有解法） |
| **Pearson $\chi^2$** | **严打型**（对离谱的错误回答给予二次方暴击惩罚），实验中表现最优 |
| **Hellinger** | **温和型**（惩罚有上限，防止梯度爆炸，训练极稳） |
| **Jensen–Shannon (JS)** | **对称型**（有界且平滑，不偏心） |
| **全变差 (TV)** | 最严格的概率差异度量 |

## f-GRPO 的核心思想：用不同的规则对待“好学生”和“差学生”

在标准的 GRPO 中，我们对好回答（正优势值）和坏回答（负优势值）的惩罚机制是对称的：好回答就鼓励，坏回答就按同样的方式抑制（只是符号相反，且都经过一个裁剪函数 clip）。

这意味着更有潜力的改进方向不是简单地“把 KL 换成逆 KL”，而是**系统地寻找让训练过程更高效的散度**——这正是 f-GRPO 的出发点。而且 f-GRPO 更进一步：它不仅换了散度，还利用散度的数学结构**彻底重写了损失函数**。它提出一个核心洞察：**鼓励好回答和抑制坏回答，应该使用两套完全不同的数学规则。**

### f-GRPO 的损失函数（大白话解析）

f-GRPO 的核心改动是：**对正组（好回答）和负组（坏回答）施加不同形式的损失函数**。完整的损失函数为：

$$
\mathcal{L}_{\text{f-GRPO}}^{(f,g)}(\theta) = \mathbb{E}_x \left[ \frac{1}{\sum_{i=1}^{G} |y_i|} \sum_{i=1}^G \sum_{t=1}^{|y_i|} \frac{-a_i}{1+\beta^{-1}} \cdot \psi(r_{\theta,i,t}, a_i) \right]
$$

**这个公式在干什么？（物理意义）**

1. **$\mathbb{E}_x$ 和 $\sum_{i=1}^G$**：这部分和标准 GRPO 一样。
   - **$\mathbb{E}_x$**：表示**求数学期望**。在实际训练中，它代表“对训练集里的每一个问题（prompt）$x$ 取平均”。
   - **$\sum_{i=1}^G$**：表示对于**同一个问题 $x$**，我们让模型生成 $G$ 个不同的回答（比如 $G=4$ 或 $G=8$），把这 $G$ 个回答的损失加起来。
2. **$\frac{1}{\sum |y_i|} \sum_{t=1}^{|y_i|}$**：这是**Token 级的全局平均**。因为大语言模型是逐字（token）生成的，所以我们需要把所有回答里所有 token 的损失加起来，再除以总的有效 token 数。这与代码里的 `loss.sum() / total_tokens` 完全对应。
3. **$\frac{-a_i}{1+\beta^{-1}}$**：这是一个**优势加权与缩放系数**。
   - $a_i$ 是优势值（成绩单），成绩越好，绝对值越大，说明这个回答对最终损失的贡献（权重）越大。前面的负号是为了把最大化奖励变成最小化损失。
   - $\frac{1}{1+\beta^{-1}}$ 是论文中为了将梯度量级与标准 PPO 对齐而引入的全局缩放常数。由于 $\beta$ 是一个固定的超参数（如 0.1），这个系数对于所有 prompt、所有回答、所有 token 都是**完全一样**的常数。因此在实际代码实现中，它通常会被优化器的学习率（Learning Rate）直接吸收，代码里也就省略了这一项。
4. **$\psi(r_{\theta,i,t}, a_i)$**：这是**真正的核心引擎**。它根据回答是好是坏（$a_i$ 的正负），决定用什么规则来计算当前 token $t$ 的损失。

为了在代码中更方便地实现，我们把上面的公式做一步等价代换。
首先，忽略掉常数分母 $1+\beta^{-1}$。
然后，注意到 $\psi$ 函数本身就是分段的（好回答和坏回答分别用不同的公式），我们可以把外面的 $-a_i$ 拆成两部分：**符号（正负号）** 和 **绝对值（$|a_i|$）**。
- 如果 $a_i > 0$（好回答），$-a_i = -|a_i|$。我们把负号扔进 $\psi$ 里面，外面就只剩下 $|a_i|$。
- 如果 $a_i \le 0$（坏回答），$-a_i = +|a_i|$。我们把正号扔进 $\psi$ 里面，外面也只剩下 $|a_i|$。

经过这样拆解后，公式就变成了代码里实际实现的样子：

$$
\mathcal{L}_{\text{f-GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{\sum_{i=1}^{G} |y_i|} \sum_{i=1}^G \sum_{t=1}^{|y_i|} |a_i| \cdot \begin{cases}
g(r_{\theta,i,t}) & \text{if } a_i > 0 \quad \text{(正侧/好回答)} \\[4pt]
f^* \circ g(r_{\theta,i,t}) & \text{if } a_i \le 0 \quad \text{(负侧/坏回答)}
\end{cases} \right]
$$

在这个等价公式中：

- 外面的 $|a_i|$ 对应代码里的 `abs_adv = advantages.abs()`。
- 里面的 $g(r_{\theta,i,t})$ 对应代码里的正组损失 `pos_loss`（注意，这里的 $g$ 已经吸收了刚才扔进来的负号，所以代码里 `pos_loss = -implicit_reward`）。
- 里面的 $f^* \circ g(r_{\theta,i,t})$ 对应代码里的负组损失 `neg_loss`（这里的 $f^*$ 吸收了正号，所以代码里 `neg_loss = implicit_reward + ...`）。

**与标准 GRPO 的对照**：
在标准 GRPO 中，损失函数的核心部分是 $\min\!\big(\rho_i\,\hat{A}_i,\;\text{clip}(\rho_i,\,1\!-\!\varepsilon,\,1\!+\!\varepsilon)\,\hat{A}_i\big)$。

- 标准 GRPO 就像一个**死板的老师**：无论你是好学生（$\hat{A}_i > 0$）还是差学生（$\hat{A}_i < 0$），都用同一套 `clip` 规则来防止你骄傲或气馁。
- 而 f-GRPO 则是**因材施教**：它把标准 GRPO 里那套死板的 `clip` 机制整个扔掉了，换成了 $\psi$ 函数。

这个 $\psi$ 函数对正负组采取了**分道扬镳**的处理方式：

$$
\psi(r_{\theta,i}, a_i) = \begin{cases}
w_i^+ \cdot g(r_{\theta,i}) & \text{if } a_i > 0 \quad \text{（正组：只过一道“转换门” } g \text{）} \\[4pt]
w_i^- \cdot f^* \circ g(r_{\theta,i}) & \text{if } a_i \leq 0 \quad \text{（负组：过完 } g \text{ 还要再受 } f^* \text{ 的“惩罚”）}
\end{cases}
$$

**通俗逐项解释**：

- $r_{\theta,i}$　——**隐式奖励（模型的自信度）**。它等于 $\beta \ln \frac{\pi_\theta}{\pi_{\text{ref}}}$。简单来说，模型越倾向于输出这个回答，隐式奖励就越高。

- $a_i = \hat{A}_i$　——**考试成绩（优势值）**。大于 0 说明比平均水平好（归入正组），小于等于 0 说明比平均差（归入负组）。

- $g(\cdot)$　——**转换门（链接函数）**。

  - **通俗理解**：模型的隐式奖励 $r_\theta$ 是个没有上下限的实数。但后面的惩罚机制（$f^*$）可能比较挑剔，不接受某些数字（比如不接受正数）。$g$ 的作用就是一个“转换门”，把模型的奖励安全、单调地转换成惩罚机制能接受的格式。
  - **数学约束**：$g$ 的输出必须落在 $f^*$ 能接受的定义域内（$\text{im}(g) \subseteq \text{dom}(f^*)$）。
  - **举例**：如果惩罚机制 $f^*$ 啥数字都能吃（比如前向 KL 散度），那 $g$ 就可以是个“旋转门”，原样输出（$g(x) = x$）。如果 $f^*$ 只吃负数（比如逆 KL 散度），那 $g$ 就必须把所有数字都转换成负数（比如 $g(x) = -e^{-x}$）。

- $f^*(\cdot)$　——**差生惩罚规则（Fenchel 共轭）**。
  - **通俗理解**：这是 f-GRPO 的灵魂。不同的 f-散度（如 KL、Pearson、JS）会推导出不同的 $f^*$。它决定了**我们到底要怎么惩罚坏回答**。选不同的散度，就是在选不同的惩罚规则。

- $w_i^{\pm}$　——**注意力权重**。好回答中，模型越不自信的，越要重点鼓励；坏回答中，模型越自信的，越要重点打压。

### 实例化：不同散度的“惩罚规则”长什么样？

上面的 $g$ 和 $f^*$ 看起来抽象，但具体到某个散度时非常直观。我们来看看几种常见散度对应的正/负组规则（省略权重 $w_i^{\pm}$）：

| f-散度 | $f(t)$ | 差生惩罚规则 $f^*(s)$ | 正组（好回答）怎么处理 | 负组（坏回答）怎么惩罚 | 通俗特点 |
|:---:|:---:|:---:|:---:|:---:|:---|
| Pearson $\chi^2$ | $(t-1)^2$ | $s + s^2/4$ | $-r_\theta$ | $r_\theta + r_\theta^2/4$ | **二次暴击**：坏回答越自信，惩罚呈平方级爆炸 |
| Hellinger | $(\sqrt{t}-1)^2$ | $s/(1-s)$ | $-(1 - e^{-r_\theta})$ | $\frac{1-e^{-r_\theta}}{e^{-r_\theta}}$ | **见好就收**：正侧奖励有上限，防止模型过度自信导致崩溃 |
| 逆 KL | $-\ln t$ | $-1-\ln(-s)$ | $-r_\theta$ | $-1 - \ln(-r_\theta)$ | **对数惩罚**：比较温和的常规惩罚 |

以在 1.5B 模型实验中表现最好的 **Pearson $\chi^2$** 为例：

- **对于好回答（正组）**：损失是 $-r_\theta$。模型越自信（$r_\theta$ 越大），损失越小，这很常规。
- **对于坏回答（负组）**：损失是 $r_\theta + \frac{r_\theta^2}{4}$。注意那个平方项！如果模型对一个坏回答非常自信（$r_\theta$ 很大），它不仅会受到线性惩罚，还会受到**二次方的暴击惩罚**。这种对“尾部错误”的零容忍，就是它效果好的原因。

**与标准 GRPO 的对比**：标准 GRPO 就像一个死板的老师，不管是对好学生还是差学生，都用同一套 `clip` 规则。而 f-GRPO 则是因材施教：正组用一套规则鼓励，负组用另一套规则（$f^*$）严厉打击。

### 为什么要这样设计？（理论溯源）

这种“正负组分别处理”的灵感，其实来自数学上的 **f-散度变分表示**（Nguyen et al., 2010）。
在数学上，衡量两个分布（好回答分布 $D^+$ 和坏回答分布 $D^-$）的差异时，公式天然就分成了两部分：

- 一部分是对好回答求期望：$\mathbb{E}_{D^+}[g(r_\theta)]$
- 另一部分是对坏回答求期望：$\mathbb{E}_{D^-}[f^* \circ g(r_\theta)]$

f-GRPO 只是极其巧妙地把这个纯数学公式，直接“翻译”成了深度学习的损失函数。它证明了：**我们不需要拘泥于 PPO 的裁剪（clip）机制，只要顺着数学公式的指引，就能找到更优雅、更高效的对齐方法。**

### f-GRPO 与 GRPO 的关系

名字中虽然带 "GRPO"，但 f-GRPO 并不是对 GRPO 损失函数的微调——它**重写了整个优化框架**，只保留了 GRPO 的采样与分组策略：

| | 标准 GRPO | f-GRPO |
|:---:|:---|:---|
| **继承** | 每个 Prompt 采 $G$ 个回答，按优势分正/负组 | 相同 |
| 策略比率 | PPO 风格 $\rho = \pi_\theta / \pi_{\theta_\text{old}}$ | 隐式奖励 $r_\theta = \beta \log(\pi_\theta / \pi_{\text{ref}})$ |
| 裁剪 | $\text{clip}(\rho, 1\!-\!\varepsilon, 1\!+\!\varepsilon)$ | 无 |
| 正/负组处理 | 相同函数，仅由 $\hat{A}_i$ 符号区分 | **不同函数**：正组 $g$，负组 $f^* \circ g$ |
| 重要性采样 | 需要 $\pi_{\theta_\text{old}}$ | 不需要 |
| KL 惩罚 | 显式 $\beta D_{\text{KL}}$ 项 | 隐含在 $r_\theta$ 的定义中 |
| 理论根基 | 策略梯度 + PPO 代理目标 | f-散度变分表示 |

因此更准确的说法是：**f-GRPO 继承了 GRPO 的"每 Prompt 采 $G$ 个回答并按奖励分组"的在线采样框架，但用 f-散度变分估计完全替换了 PPO 风格的裁剪策略梯度损失**。名字中的 "GRPO" 指的是采样/分组策略，而非损失函数。

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

> **注意表格中的"KL（前向）"行**：它的 Relative Overall 为 81.83，**比标准 GRPO（74.26）高出 7.57 分**。同样是 KL 散度，为什么效果不同？因为两者用 KL 的方式完全不同。标准 GRPO 将 KL 用作 PPO 裁剪损失后的**惩罚项**（$-\beta D_{\text{KL}}$），损失主体仍是 $\min(\rho\hat{A},\;\text{clip}(\rho)\hat{A})$。而 f-GRPO with KL 用 KL 散度的变分表示来**设计正/负组的损失函数本身**，替换了整个 PPO 裁剪框架（详见上文"f-GRPO 与 GRPO 的关系"对比表）。即使选择相同的散度，两种"用法"产生的损失函数和优化行为也完全不同。

### f-GRPO 的具体实现

在代码实现中，我们要把前面提到的理论公式转化为具体的张量计算。回顾一下，我们最原始的核心损失函数是：

$$
\mathcal{L}_{\text{f-GRPO}}^{(f,g)}(\theta) = \mathbb{E}_x \left[ \frac{1}{\sum_{i=1}^{G} |y_i|} \sum_{i=1}^G \sum_{t=1}^{|y_i|} \frac{-a_i}{1+\beta^{-1}} \cdot \psi(r_{\theta,i,t}, a_i) \right]
$$

为了在代码中更方便地实现，我们需要对这个公式做一步**等价变形**。

1. **忽略常数缩放**：分母 $1+\beta^{-1}$ 是一个全局常数，它只影响梯度的整体大小，不影响方向。在代码中，它会被优化器的学习率直接吸收，因此我们可以安全地将其省略。
2. **拆解优势值 $a_i$**：注意到 $\psi(r_{\theta,i,t}, a_i)$ 函数本身就是分段的（好回答和坏回答分别用不同的公式），我们可以把外面的 $-a_i$ 拆成两部分：**符号（正负号）** 和 **绝对值（$|a_i|$）**。
   - 如果 $a_i > 0$（好回答），$-a_i = -|a_i|$。我们把负号扔进 $\psi$ 里面，外面就只剩下 $|a_i|$。
   - 如果 $a_i \le 0$（坏回答），$-a_i = +|a_i|$。我们把正号扔进 $\psi$ 里面，外面也只剩下 $|a_i|$。

经过这样拆解后，公式就变成了代码里实际实现的样子：

$$
\mathcal{L}_{\text{f-GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{\sum_{i=1}^{G} |y_i|} \sum_{i=1}^G \sum_{t=1}^{|y_i|} |a_i| \cdot \begin{cases}
-g(r_{\theta,i,t}) & \text{if } a_i > 0 \quad \text{(正侧/好回答)} \\[4pt]
+f^* \circ g(r_{\theta,i,t}) & \text{if } a_i \le 0 \quad \text{(负侧/坏回答)}
\end{cases} \right]
$$

在这个等价公式中，我们就能和下方的代码实现**完美地一一对应**了：

- 外面的 $|a_i|$ 对应代码里的 `abs_adv = advantages.abs()`。
- 里面的 $-g(r_{\theta,i,t})$ 对应代码里的正组损失 `pos_loss`（注意，这里的负号是从外面的 $-a_i$ 拿进来的，所以代码里 `pos_loss = -implicit_reward`）。
- 里面的 $+f^* \circ g(r_{\theta,i,t})$ 对应代码里的负组损失 `neg_loss`（这里的正号是从外面的 $-a_i$ 拿进来的，所以代码里 `neg_loss = implicit_reward + ...`）。

不同 f-散度的选择对应不同的链接函数 $g$ 和共轭函数 $f^*$。以下是几种常见散度的完整实现：

```python
import torch
import torch.nn.functional as F

def f_grpo_loss(
    log_probs,       # 当前策略模型生成的 token 概率对数 (log π_θ)
    ref_log_probs,   # 冻结的参考模型生成的 token 概率对数 (log π_ref)
    old_log_probs,   # 采样时旧策略生成的 token 概率对数 (log π_old，这里保留是为了接口兼容，f-GRPO 核心计算其实不用它)
    advantages,      # 组内标准化后的优势值 A_i (正数代表好回答，负数代表坏回答)
    loss_mask,       # 掩码 (用于过滤掉 padding 等无效 token，1 为有效，0 为无效)
    f_type="pearson_chi2", # 选择使用哪种 f-散度 (默认 Pearson χ²)
    beta=0.1,        # 隐式奖励的缩放系数 (控制 KL 惩罚的强度)
):
    """f-GRPO 损失函数，支持多种 f-散度（正/负侧链接不同；标准 GRPO 两侧裁剪相同）。
    
    Returns:
        标量，平均每有效 token 的 f-GRPO 损失。
    """
    # 1. 计算隐式奖励 (Implicit Reward)
    # 公式：r_θ = β * (log π_θ - log π_ref)
    # 物理意义：模型当前对这个 token 的“自信度”相对参考模型的偏移量。
    implicit_reward = beta * (log_probs - ref_log_probs)  # (*, T)

    # 2. 划分正负样本组
    # pos_mask: 找出所有优势值大于 0 的 token（好回答），并与 loss_mask 相乘确保是有效 token
    pos_mask = (advantages > 0).float().unsqueeze(-1) * loss_mask
    # neg_mask: 找出所有优势值小于等于 0 的 token（坏回答），同样确保是有效 token
    neg_mask = (advantages <= 0).float().unsqueeze(-1) * loss_mask

    # 3. 根据不同的 f-散度类型，分别计算正组和负组的损失
    if f_type == "pearson_chi2":
        # Pearson χ² 散度：对坏回答的过度自信给予“二次方暴击”惩罚
        # 正侧 (好回答): 损失 = -r_θ。模型越自信 (r_θ 越大)，损失越小。
        pos_loss = -implicit_reward
        # 负侧 (坏回答): 损失 = r_θ + (r_θ^2)/4。如果模型对坏回答很自信 (r_θ 很大)，会受到平方级的巨大惩罚。
        neg_loss = implicit_reward + implicit_reward ** 2 / 4

    elif f_type == "hellinger":
        # Hellinger 散度：温和型，惩罚有上限，防止梯度爆炸
        # g_r 是一个“饱和链接”函数：1 - exp(-r_θ)。当 r_θ 很大时，g_r 趋近于 1，不会无限变大。
        g_r = 1.0 - torch.exp(-implicit_reward)
        # 正侧 (好回答): 损失 = -g_r。鼓励 r_θ 变大，但收益有上限 (-1)。
        pos_loss = -g_r
        # 负侧 (坏回答): 损失 = g_r / (1 - g_r)。惩罚坏回答的自信度。
        neg_loss = g_r / (1.0 - g_r + 1e-8)  # +1e-8 是为了防止分母为 0 导致除零错误

    elif f_type == "reverse_kl":
        # 逆 KL 散度：标准 GRPO 默认的散度类型，倾向于“模式寻求”(偏科型)
        # 正侧 (好回答): 损失 = -r_θ。
        pos_loss = -implicit_reward
        # 负侧 (坏回答): 损失 = -1 - ln(-r_θ)。
        # 注意：clamp(max=-1e-8) 是为了确保 -r_θ 严格大于 0，防止 log(0) 或对负数求对数导致 NaN 报错。
        neg_loss = -1.0 - torch.log(-implicit_reward.clamp(max=-1e-8))

    elif f_type == "js":
        # JS 散度 (Jensen-Shannon)：对称且有界，训练过程非常平滑
        # ratio: 还原出概率比值 π_θ / π_ref = exp(r_θ / β)
        ratio = torch.exp(implicit_reward / beta)
        # 正侧 (好回答): 代入 JS 散度的变分公式推导出的复杂项
        pos_loss = -(
            ratio * implicit_reward / beta
            - (ratio + 1) * torch.log((ratio + 1) / 2)
        )
        # 负侧 (坏回答): 在 JS 散度下，负侧损失刚好与正侧完全对称 (互为相反数)
        neg_loss = -pos_loss

    else:
        # 如果传入了不支持的散度类型，抛出异常
        raise ValueError(f"未知 f-散度类型: {f_type}")

    # 4. 优势加权与掩码过滤
    # abs_adv: 取优势值的绝对值 |A_i|。优势的绝对值越大，说明这个回答“特别好”或“特别差”，应该给予更大的更新权重。
    # unsqueeze(-1) 是为了在最后增加一个维度，以便与 token 级别的 loss 张量形状对齐 (广播机制)。
    abs_adv = advantages.abs().unsqueeze(-1)
    
    # 最终的 token 级 loss:
    # (正侧损失 * 正侧掩码 * 优势绝对值) + (负侧损失 * 负侧掩码 * 优势绝对值)
    loss = pos_loss * pos_mask * abs_adv + neg_loss * neg_mask * abs_adv
    
    # 5. 求平均并返回
    # 计算当前 batch 中有效的 token 总数
    total_tokens = loss_mask.sum()
    # 将所有有效 token 的 loss 求和，然后除以有效 token 总数，得到平均 loss
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

在理解了 GRPO 的“对比”本质（2-GRPO）和“散度”本质（f-GRPO）之后，我们再来看它的第三副面孔：**回归本质**。

这部分的核心思想来自论文 **GIFT** (Generative Implicit Feedback Training，**生成式隐式反馈训练**)。它指出，如果我们换一个角度看问题，大模型的对齐训练其实可以变得非常简单：**它不过是在做一个初中生都会的“均方误差（MSE）回归”而已。**

## GIFT 的核心洞察：消灭“拦路虎” $Z(x)$

要理解 GIFT，我们得先复习一下 DPO（直接偏好优化）中的一个经典公式。DPO 证明了，任何一个语言模型，其实内部都隐藏着一个“隐式奖励模型”。它的公式长这样：

$$
r_\theta(x, y) = \beta \log \frac{\pi_\theta(y | x)}{\pi_{\text{ref}}(y | x)} + \beta \log Z(x)
$$

**通俗解释**：

- $r_\theta(x, y)$：模型自己心里对回答 $y$ 打的分数（隐式奖励）。
- $\beta \log \frac{\pi_\theta}{\pi_{\text{ref}}}$：模型现在的输出概率和老模型（参考模型）的差异。
- $Z(x)$：**配分函数（Partition Function）**。这是个超级大麻烦！它要求我们把模型对这个问题 $x$ 可能生成的所有成千上万种回答的概率都加起来。这在计算上是**绝对不可能完成的任务**（intractable）。

正是因为这个“拦路虎” $Z(x)$ 的存在，DPO 没法直接让模型的隐式奖励去拟合外部的真实奖励，只能退而求其次，让两个回答去“打擂台”（因为相减的时候 $Z(x)$ 会被抵消掉）。

**GIFT 的天才发现**：
我们不需要让两个回答去打擂台！只要我们对同一个问题生成的**一组**回答做一下**“组内均值归一化”**（也就是减去这组回答的平均分），那个讨厌的 $Z(x)$ 就会奇迹般地消失！

> **深度辨析：GIFT 的组内归一化和 GRPO 的组内归一化有什么不同？**
> 
> 很多读者可能会敏锐地发现：**GRPO 不也是在做组内归一化吗？** 没错，但它们归一化的**对象**和**目的**截然不同，这导致了两者在数学本质上的天壤之别：
> 
> 1. **GRPO 归一化的是“外部奖励”**：
>    - **做法**：GRPO 拿到裁判给的 $G$ 个分数，算出组内均值，然后用 $r_i - \mu_R$ 算出**优势值（Advantage）**。
>    - **目的**：为了**取代 Critic 网络（Baseline）**。它只是想知道一个回答是“好于平均”还是“差于平均”，从而决定梯度是鼓励还是抑制。
>    - **本质**：它依然是**策略梯度（Policy Gradient）**，需要 PPO 的 `clip` 裁剪来防止模型更新崩溃。
> 
> 2. **GIFT 归一化的是“隐式奖励”和“外部奖励”双方**：
>    - **做法**：GIFT 不仅对裁判给的分数做归一化，它还让模型自己算出一个“内心打分”（即上面的 $r_\theta$），然后对这 $G$ 个“内心打分”也做组内归一化。
>    - **目的**：为了**消灭配分函数 $Z(x)$**！因为 $Z(x)$ 对同一个 prompt 是一个常数，一减去组内均值，它就彻底消失了。
>    - **本质**：它把强化学习直接降维打击成了一个**监督回归（MSE Regression）**任务。不需要算优势，不需要 `clip` 裁剪，天生极其稳定。

**数学推导非常简单**：
假设我们对同一个问题 $x$ 生成了 $N$ 个回答 $\{y_1, y_2, \dots, y_N\}$。

1. **算平均分**：根据前面的公式，每个回答的隐式奖励是 $r_\theta(x, y_i) = \beta \log \frac{\pi_\theta(y_i | x)}{\pi_{\text{ref}}(y_i | x)} + \beta \log Z(x)$。当我们把这 $N$ 个回答的奖励加起来求平均 $\mu_\theta$ 时：
   $$
   \mu_\theta = \frac{1}{N}\sum_{i=1}^N r_\theta(x, y_i) = \frac{1}{N}\sum_{i=1}^N \left(\beta \log \frac{\pi_\theta(y_i|x)}{\pi_{\text{ref}}(y_i|x)}\right) + \beta \log Z(x)
   $$
   你看，因为对于同一个问题 $x$，$Z(x)$ 是一个固定的常数（不随回答 $y_i$ 变化），所以求平均后，它依然原封不动地保留在均值 $\mu_\theta$ 里。

2. **减去平均分（去中心化）**：当我们用每个回答的隐式奖励减去均值时：
   $$ r_\theta(x, y_i) - \mu_\theta $$
   因为 $r_\theta(x, y_i)$ 和 $\mu_\theta$ 里面都包含了一个完全相同的常数项 $\beta \log Z(x)$，**它在相减的瞬间就被彻底抵消了！**

如果我们再进一步，除以这组回答的标准差（做一次完整的标准化），我们就会得到一个极其干净、纯粹的隐式奖励 $\hat{r}'_\theta(x, y_i)$。它不仅没有 $Z(x)$，连超参数 $\beta$ 都被消掉了！

## GIFT 的损失函数：大道至简的 MSE

既然“拦路虎”被消灭了，我们现在可以直接拿到模型心里真实的、干净的打分了。

那训练模型就变得无比简单：**外部裁判（奖励模型）给这个回答打多少分，你就强迫模型心里的打分也变成多少。**

怎么强迫？用最基础的均方误差（MSE）损失函数：

$$
\mathcal{L}_{\text{GIFT}}(\pi_\theta) = \mathbb{E}_{(x, y) \sim \text{on-policy}}\left[\left(r'_\phi(x, y) - \hat{r}'_\theta(x, y)\right)^2\right]
$$

**通俗解释**：

- $r'_\phi$：外部裁判（奖励模型）给出的分数，经过组内标准化后的结果。
- $\hat{r}'_\theta$：模型自己心里的分数（隐式奖励），经过组内标准化后的结果。
- 整个公式的意思就是：**让模型心里的分数，无限逼近外部裁判的分数。**

**用一个具体的例子来理解**：

假设针对同一个问题，模型生成了 4 个回答。裁判打完分并标准化后，模型也算出自己的隐式奖励并标准化，然后直接算差值的平方：

| 回答 | 裁判打分（标准化后 $r'_\phi$） | 模型心里的打分（标准化后 $\hat{r}'_\theta$） | 损失（MSE） | 模型的内心活动 |
|:---:|:---:|:---:|:---:|:---|
| $y_1$（正确简洁） | **+1.22** | **+1.15** | $(1.22-1.15)^2 \approx 0$ | “裁判觉得好，我也觉得好，不用改。” |
| $y_2$（正确冗长） | **+0.41** | **+0.23** | $(0.41-0.23)^2$ | “裁判觉得还行，我给低了，我要提高这个回答的概率。” |
| $y_3$（错误） | **-0.82** | **-0.69** | $(-0.82 - (-0.69))^2$ | “裁判觉得差，我给高了，我要降低这个回答的概率。” |

你看，**GIFT 的训练本质上就是在做一个极其简单的回归任务**。它抛弃了强化学习里那些复杂的策略梯度、裁剪（clip）、优势函数，直接用回归来解决对齐问题。

## GIFT vs GRPO vs DPO：三足鼎立

| 特性 | GRPO（强化学习派） | DPO（偏好对比派） | **GIFT（回归派）** |
|:---|:---:|:---:|:---:|
| **核心动作** | 算优势值，做策略梯度 | 拿两个回答打擂台对比 | **算分数差，做 MSE 回归** |
| **在线生成新回答？** | ✓ | ✗（只用现成数据） | **✓** |
| **需要裁剪（clip）防崩溃？** | ✓（需要设 $\varepsilon$） | ✗ | **✗**（MSE 天然稳定） |
| **需要参考模型？** | ✓ | ✓ | **✓**（但 $\beta$ 和 $Z(x)$ 被巧妙消除） |
| **数学稳定性** | 容易剧烈震荡 | 中等 | **极高**（MSE 是凸函数，非常好优化） |

> **深度辨析：GRPO 和 GIFT 都要算参考模型概率（`ref_logp`），它们的作用一样吗？**
> 
> 很多读者在看代码时会发现：**GRPO 和 GIFT 都在采样阶段计算了参考模型的对数概率 `ref_logp`**。但它们拿这个概率去做的事情，却有着本质的区别：
> 
> 1. **在 GRPO 中，`ref_logp` 是一个“外挂的惩罚项”（紧箍咒）**：
>    - GRPO 的核心驱动力是**策略梯度**（用优势值 $\hat{A}_i$ 乘以重要性比率）。
>    - `ref_logp` 仅仅出现在最后的 KL 散度惩罚项中：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + \beta (\log \pi_\theta - \log \pi_{\text{ref}})$。
>    - 它的作用是：你可以去追求高分，但你的输出概率不能偏离老模型太远，偏离了就要扣分，防止你“走火入魔”（Reward Hacking）。
> 
> 2. **在 GIFT 中，`ref_logp` 是“核心损失函数的基础部件”（打分器）**：
>    - GIFT 完全抛弃了策略梯度和优势值计算。
>    - 它直接把 $\log \pi_\theta - \log \pi_{\text{ref}}$ 组合起来，定义为模型内心的**隐式奖励**（即模型自己给这个回答打的分数）。
>    - 然后，直接拿这个内心打分去和外部裁判的打分算 MSE 均方误差。
>    - 它的作用是：和 $\log \pi_\theta$ 深度绑定，共同构成模型内心的“真实想法”，是回归目标不可或缺的一半。

> **实验效果**：GIFT 论文指出，在 7B 模型上，由于 MSE 损失函数极其稳定，GIFT 在多个基准测试（如 AlpacaEval 和 Arena-Hard）上的胜率甚至大幅超越了标准 GRPO。这证明了“大道至简”的回归方法在 LLM 对齐中蕴含着巨大的潜力。

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
    """GIFT 损失：让模型内部的打分（隐式奖励）去逼近外部裁判的打分（外部奖励）。
    注意：两边的打分都需要先做组内归一化（减均值除标准差），才能在同一个尺度下比较。

    隐式奖励 = β·log(π_θ/π_ref) + β·log Z(x)；组内归一化后 β·log Z(x) 被消除。

    Args:
        log_probs: log π_θ(y|x)，当前策略的对数概率，形状 (B, G)，**带有梯度**。
        ref_log_probs: log π_ref(y|x)，参考策略的对数概率，形状 (B, G)，**固定无梯度**。
        rewards: 外部裁判打分 r_φ(x, y)，形状 (B, G)，**固定无梯度**。

    Returns:
        标量 MSE，默认对元素取 mean。
    """
    # 1. 计算内部奖励 (Implicit Reward)
    # 公式：r_θ = log π_θ - log π_ref (这里省略了超参数 β，因为归一化后它也会被消掉)
    # 物理意义：模型自己心里对这个回答打的分数。
    implicit_rewards = log_probs - ref_log_probs  # (B, G)

    # 2. 内部奖励的组内归一化 (在 G 维标准化)
    # 这一步是 GIFT 的灵魂！减去均值后，那个无法计算的配分函数 Z(x) 就被彻底消除了。
    impl_mean = implicit_rewards.mean(dim=1, keepdim=True)  # (B, 1)
    impl_std = implicit_rewards.std(dim=1, keepdim=True) + 1e-8  # (B, 1)
    norm_implicit = (implicit_rewards - impl_mean) / impl_std  # (B, G)

    # 3. 外部奖励的组内归一化
    # 裁判给的分数可能在 [0, 1] 之间，也可能在 [-100, 100] 之间。
    # 归一化后，外部奖励和内部奖励就处于同一个尺度（均值为0，方差为1）了。
    rew_mean = rewards.mean(dim=1, keepdim=True)  # (B, 1)
    rew_std = rewards.std(dim=1, keepdim=True) + 1e-8  # (B, 1)
    norm_rewards = (rewards - rew_mean) / rew_std  # (B, G)

    # 4. MSE 回归损失
    # 强迫模型心里的打分 (norm_implicit) 去逼近裁判的打分 (norm_rewards)。
    # detach 外部奖励：确保梯度只流向 actor 模型，不回传到外部奖励。
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

    # 采样阶段不需要对 θ 求导，纯粹为了生成回答和收集外部裁判分数
    with torch.no_grad():
        for prompt in prompts_batch:
            group_logps, group_ref_logps, group_rewards = [], [], []
            for _ in range(G):
                # 1. 生成回答
                response = actor.generate(prompt, do_sample=True)
                
                # 2. 收集外部奖励 (External Reward)
                # 调用裁判模型或规则判题器给当前回答打分。
                reward = reward_fn(prompt, response)
                
                # 3. 收集内部奖励的基础部件 (Reference Log Prob)
                # 计算老模型 (参考模型) 生成这个回答的概率对数 log π_ref。
                # 为什么这里不算 log π_θ？因为我们要用它算梯度更新模型！
                # 所以 log π_θ 必须留到阶段 2 (有梯度环境) 去算。
                ref_logp = compute_seq_log_prob(ref_model, prompt, response)
                
                group_rewards.append(reward)
                group_ref_logps.append(ref_logp)
            
            all_rewards.append(torch.tensor(group_rewards))
            all_ref_log_probs.append(torch.stack(group_ref_logps))

    rewards = torch.stack(all_rewards)  # (B, G)
    ref_logps = torch.stack(all_ref_log_probs)  # (B, G)

    # 过滤零方差组 (与 DAPO 类似)
    # 如果一组内所有回答得分都一样，归一化时分母会是 0，而且也无法提供对比信号，直接跳过。
    valid = rewards.std(dim=1) > 0
    if valid.sum() == 0:
        continue

    # --- 阶段 2: GIFT 更新 ---
    # GIFT 不需要"多 epoch 更新"和"重要性采样比率"!
    # 它直接对当前策略做前向传播计算隐式奖励，并用 MSE 损失更新。

    for prompt_idx in valid.nonzero(as_tuple=True)[0]:
        prompt = prompts_batch[prompt_idx]
        # G 条解码结果；须与阶段 1 写入的 all_responses 对齐（示意伪代码）
        responses = all_responses[prompt_idx]

        # 4. 计算内部奖励的另一半部件 (Actor Log Prob)
        # 注意：这里没有 with torch.no_grad()！
        # 用当前策略 (actor) 重新计算这 G 个回答的对数概率 log π_θ。
        # 这样算出来的 log_probs 是带有计算图的，梯度可以顺着它流回模型参数 θ。
        log_probs = torch.stack(
            [
                compute_seq_log_prob(actor, prompt, resp)
                for resp in responses
            ]
        ).unsqueeze(0)  # (1, G)

        ref_lps = ref_logps[prompt_idx].unsqueeze(0)  # (1, G)
        rews = rewards[prompt_idx].unsqueeze(0)  # (1, G)

        # 5. 计算 GIFT 损失 (MSE)
        # 内部会把 log_probs - ref_lps 拼成内部奖励，并与外部奖励 rews 做 MSE。
        loss = gift_loss(log_probs, ref_lps, rews)

        # 6. 反向传播与梯度更新
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
> 4. [How RLHF Amplifies Sycophancy](https://arxiv.org/abs/2602.01002)
> 5. [Pref-GRPO: Pairwise Preference Reward-based GRPO for Stable Text-to-Image Reinforcement Learning](https://arxiv.org/abs/2508.20751)
> 6. [KL-Regularized Reinforcement Learning is Designed to Mode Collapse](https://arxiv.org/abs/2510.20817)
> 7. [Demystifying GRPO: Its Policy Gradient is a U-Statistic](https://arxiv.org/abs/2603.01162)
> 8. [f-PO: Generalizing Preference Optimization with f-divergence Minimization](https://arxiv.org/abs/2410.21662)
> 9. [f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization](https://arxiv.org/abs/1606.00709)

> 下一篇：[笔记｜强化学习（八）：SuperFlow 与图像生成 RL 的统一框架](/chengYi-xun/posts/58-superflow/)
