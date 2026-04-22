---
title: 笔记｜强化学习（四）：大模型在线 RL 破局者：GRPO 算法详解
date: 2025-08-19 10:00:00
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

> 本文为系列第四篇。在了解了 PPO 的显存痛点和 DPO 的离线局限性后，我们终于迎来了目前大模型在线 RL 的最前沿破局者——GRPO（Group Relative Policy Optimization）。本文将详细推导 GRPO 的核心思想，看它是如何优雅地丢弃 Critic 网络，实现高效的在线强化学习的。
>
> ⬅️ 上一篇：[笔记｜强化学习（三）：大模型对齐的另一条路：DPO (Direct Preference Optimization)](/chengYi-xun/posts/53-dpo/)
>
> ➡️ 下一篇：[笔记｜强化学习（五）：Flow-GRPO 与图像生成应用（基于 Flux 的代码解析）](/chengYi-xun/posts/55-flow-grpo/)


# 在线 RL 的不可替代性与 Critic 的累赘

正如上一篇所言，DPO 虽然简单省显存，但它只能"死记硬背"人类给出的标准答案（离线学习）。为了让模型产生"顿悟"和自我进化，我们必须回归**在线强化学习（Online RL）**。

然而，PPO 算法中的 Critic 网络（价值网络）成为了最大的绊脚石。对于百亿参数的大模型，多维护一个 Critic 意味着显存开销直接翻倍。

**核心思考出发点**：既然 Critic 只是为了给出一个"及格线"（基准值 $V(s)$），我们能不能**彻底去掉 Critic 模型**，用一种更简单的方法来估计这个"及格线"？

---

# GRPO 的核心思想：矮子里拔高个

GRPO 的思路极简：**对同一个 Prompt 采样 $G$ 个回答，用组内奖励的均值和标准差做标准化，得到每个回答的相对优势——高于均值的强化，低于均值的抑制。**

$$
\hat{A}_i = \frac{r_i - \mu_R}{\sigma_R + \varepsilon}, \quad \mu_R = \frac{1}{G}\sum_{j=1}^G r_j, \quad \sigma_R = \text{std}(r_1, \dots, r_G)
$$

这就是"矮子里拔高个"：即使绝对水平不高，只要能分出高低，模型就有学习信号。注意分母的 $\varepsilon$（通常取 $10^{-8}$）：当所有回答奖励相同时 $\sigma_R = 0$，此时分子 $r_i - \mu_R$ 也恰好为零，$\hat{A}_i = 0/(0 + \varepsilon) = 0$，模型不更新——避免了无区分信号时的噪声梯度。

---

# GRPO 的理论根源：从 REINFORCE 到组内相对优势

在深入数学推导之前，先理清 GRPO 的理论脉络——它并不是凭空发明的，而是 **REINFORCE with Baseline** 的一个聪明的工程变体。

## 经典 RL 中的 Baseline

回顾第一篇（RL 基础）中的 REINFORCE with Baseline：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \big(G_t - b(s_t)\big) \right]
$$

**各符号含义：**

| 符号 | 含义 |
|:---|:---|
| $\theta$ | 策略网络的参数 |
| $J(\theta)$ | 策略的目标函数（期望总回报），我们要最大化它 |
| $\tau \sim \pi_\theta$ | 轨迹 $\tau$ 按策略 $\pi_\theta$ 采样（$\sim$ 读作"服从/采样自"） |
| $\tau$ | 一条完整轨迹（trajectory）：$s_0, a_0, r_0, s_1, a_1, r_1, \dots$ |
| $\pi_\theta(a_t \mid s_t)$ | 策略在状态 $s_t$ 下选择动作 $a_t$ 的概率 |
| $G_t = \sum_{k=t}^{T} r_k$ | 从时刻 $t$ 到终止的**累积回报**（未来总收益） |
| $b(s_t)$ | **基线**（baseline）：只依赖状态、不依赖动作的一个标量 |

**公式的物理意义**：$\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ 是"让动作 $a_t$ 更可能"的方向，$(G_t - b)$ 决定沿这个方向走多远——如果实际回报 $G_t$ 高于基线 $b$，就**强化**这个动作；低于基线就**抑制**。

**经典 RL 中的基线选择**：最自然的基线是**状态价值函数** $V(s_t) = \mathbb{E}[G_t \mid s_t]$，即"从当前状态出发、按当前策略行动，**未来累积回报的期望值**"。经典做法是训练一个价值网络 $V_\phi(s_t)$ 来逼近它——这就是 Actor-Critic / PPO 路线（需要额外的 Critic 模型，显存翻倍）。

## 语言模型场景的关键简化

在经典 RL 中，智能体在环境中走很多步（$s_0 \to a_0 \to s_1 \to a_1 \to \cdots$），每步都可能获得奖励，基线需要估计"从当前步到未来的累积回报"。

但在语言模型的 RLHF 场景中，**整个回答是一条完整轨迹**（prompt $s$ → 生成完整回答 $o$ → 得到一个总分 $r(o)$），奖励只在最后一步给出。这意味着：

$$
V(s) = \mathbb{E}_{o \sim \pi_\theta(\cdot|s)}[r(o)]
$$

| 符号 | 含义 |
|:---|:---|
| $s$ | 用户的 prompt |
| $o$ | 模型生成的一条完整回答 |
| $\pi_\theta(\cdot \mid s)$ | 模型在 prompt $s$ 下所有可能回答的概率分布 |
| $r(o)$ | 奖励函数对回答 $o$ 的打分 |
| $V(s)$ | 这个 prompt 下**所有可能回答的平均奖励** |

**物理意义**：因为只有一步决策（生成整个回答），"未来累积回报的期望"退化为"当前这个 prompt 下所有可能回答的平均得分"。

## GRPO 的关键洞察：用采样均值替代价值网络

$V(s) = \mathbb{E}_{o \sim \pi_\theta}[r(o)]$ 是理论上最优的基线（使策略梯度方差最小），但精确计算需要遍历所有可能回答——这不可能。

**经典做法**（PPO）：训练一个 Critic 网络 $V_\phi(s) \approx V(s)$，代价是多一个与策略模型同等规模的网络，显存翻倍。

**GRPO 的做法**：对同一个 prompt 采样 $G$ 个回答 $o_1, \dots, o_G$，直接用经验均值近似：

$$
\mu_R = \frac{1}{G}\sum_{i=1}^G r(o_i) \approx V(s) = \mathbb{E}_{o \sim \pi_\theta}[r(o)]
$$

| 符号 | 含义 |
|:---|:---|
| $G$ | 每个 prompt 的采样数量（通常 8~16） |
| $o_i$ | 第 $i$ 个采样回答 |
| $r(o_i)$ | 第 $i$ 个回答的奖励 |
| $\mu_R$ | 组内均值，$V(s)$ 的蒙特卡洛估计 |

**为什么这行得通？** 三条理论保障：

**1. 无偏性：采样均值的期望 = 真实期望**

$$
\mathbb{E}[\mu_R] = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G r(o_i)\right]
$$

逐步拆解等号：

- **第 1 步**：期望是线性运算，$\frac{1}{G}$ 是常数可以提出来，$\sum$ 可以拆成逐项期望：

$$\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G r(o_i)\right] = \frac{1}{G}\sum_{i=1}^G \mathbb{E}[r(o_i)]$$

- **第 2 步**：每个 $o_i$ 都是从**同一个分布** $\pi_\theta(\cdot|s)$ 独立采样的，所以每项的期望相同：

$$\mathbb{E}[r(o_1)] = \mathbb{E}[r(o_2)] = \cdots = \mathbb{E}[r(o_G)] = \mathbb{E}_{o \sim \pi_\theta}[r(o)]$$

- **第 3 步**：$G$ 个相同的值求和再除以 $G$，结果就是它本身：

$$\frac{1}{G} \cdot G \cdot \mathbb{E}[r(o)] = \mathbb{E}[r(o)] = V(s)$$

**结论**：$\mathbb{E}[\mu_R] = V(s)$——采样均值不会系统性地偏高或偏低，这就是"无偏"的含义。

**2. 收敛性：采样越多，估计越准**

方差衡量的是"每次估计偏离真实值多远"。对于 $G$ 个独立同分布样本的均值：

$$\text{Var}(\mu_R) = \text{Var}\left(\frac{1}{G}\sum_{i=1}^G r(o_i)\right)$$

- **独立性**：$o_1, \dots, o_G$ 相互独立，独立随机变量之和的方差 = 各自方差之和：

$$= \frac{1}{G^2} \sum_{i=1}^G \text{Var}(r(o_i)) = \frac{1}{G^2} \cdot G \cdot \text{Var}(r) = \frac{\text{Var}(r)}{G}$$

**物理意义**：方差随 $G$ 线性下降。$G=1$ 时方差最大（只看一个回答的分数来猜平均分，很不靠谱）；$G=16$ 时方差缩小到 $\frac{1}{16}$（16 个回答取平均，估计精度大幅提升）。当 $G \to \infty$，方差 $\to 0$，大数定律保证 $\mu_R$ 精确收敛到 $V(s)$。

实践中 $G = 8 \sim 16$ 是一个好的折中——采样成本可控，估计精度足够。

**3. 标准化稳定梯度**

除以 $\sigma_R$ 后得到 $\hat{A}_i = \frac{r_i - \mu_R}{\sigma_R + \varepsilon}$。这一步的作用是**消除奖励尺度的影响**。

考虑两个不同的奖励函数：任务 A 的奖励在 $[0, 1]$ 范围（如准确率），任务 B 的奖励在 $[-100, 100]$ 范围（如 BLEU 分数乘以 100）。如果不标准化，任务 B 的梯度会比任务 A 大 100 倍，学习率需要针对每个任务单独调整。

标准化后，无论原始奖励的范围如何，优势值 $\hat{A}_i$ 都近似服从均值为 0、标准差为 1 的分布（即 $\hat{A}_i \in [-3, 3]$ 左右），梯度尺度统一，**同一套超参数可以跨任务复用**。

> **一句话总结**：经典 RL 的 baseline 是"未来累积回报的期望"，在语言模型的单步场景中退化为"平均奖励"，GRPO 用采样均值来近似它——**不需要 Critic 网络，只需要多采几个样本**。

**与 RLOO 的对比**

RLOO（REINFORCE Leave-One-Out）是另一种去 Critic 的基线方案。两种方法的核心区别在于：**计算第 $i$ 个回答的基线时，是否包含 $r_i$ 本身**。

| | GRPO | RLOO |
|:---|:---|:---|
| 基线 | $\mu_R = \frac{1}{G}\sum_{j=1}^{G} r_j$ | $b_i = \frac{1}{G-1}\sum_{j \neq i} r_j$ |
| $r_i$ 是否参与基线计算 | **是**（$r_i$ 在求和里） | **否**（排除了 $r_i$） |
| $r_i$ 与基线的关系 | 正相关（$\text{Cov} > 0$） | 独立（$\text{Cov} = 0$） |

**GRPO 的"自我包含"效应**：由于 $r_i$ 同时出现在被减数和减数中，GRPO 的优势估计会被系统性压缩。展开 $\mu_R$：

$$\mu_R = \frac{1}{G}\bigl(r_i + (G{-}1)\bar{r}_{-i}\bigr), \quad \bar{r}_{-i} = \frac{1}{G{-}1}\sum_{j \neq i} r_j \;\text{（除第 } i \text{ 个之外的均值，即 RLOO 的基线）}$$

$$r_i - \mu_R = r_i - \frac{r_i + (G{-}1)\bar{r}_{-i}}{G} = \frac{G{-}1}{G}\bigl(r_i - \bar{r}_{-i}\bigr)$$

相比 RLOO 直接算出的 $r_i - \bar{r}_{-i}$，GRPO 的结果多了一个 $\frac{G-1}{G}$ 的缩放因子（$G=8$ 时为 $87.5\%$）。这意味着好坏回答之间的区分度被削弱了约 $\frac{1}{G}$，梯度信号略弱。

**从方差公式看**：$\text{Var}(r_i - b_i) = \text{Var}(r_i) + \text{Var}(b_i) - 2\text{Cov}(r_i, b_i)$。GRPO 的正协方差确实让 $r_i - \mu_R$ 的数值波动更小，但这种"稳定"来自信号被压缩而非噪声被消除——**数值方差小不等于估计质量高**。RLOO 的基线与 $r_i$ 独立，梯度信号保持原始尺度，信噪比更优。

**实践中差距不大**：$G=8$ 时压缩仅 $12.5\%$，且 GRPO 的除以 $\sigma_R$ 标准化会部分补偿这个缩放。GRPO 因实现更简单（一次求均值即可）而被广泛使用。

---

# GRPO 的数学推导与损失函数构建

## 1. 组内相对优势计算

给定一个输入 Prompt $s$，策略网络 $\pi_\theta$ 采样出 $G$ 个输出（通常 $G=4 \sim 16$）：
$$
o_1, o_2, \dots, o_G \sim \pi_\theta(\cdot|s)
$$

奖励模型（或规则判题器）对每个输出打分，得到奖励集合 $R = \{r_1, r_2, \dots, r_G\}$。

计算组内均值和标准差：
$$
\mu_R = \frac{1}{G} \sum_{i=1}^G r_i, \quad \sigma_R = \sqrt{\frac{1}{G} \sum_{i=1}^G (r_i - \mu_R)^2}
$$

对于第 $i$ 个输出 $o_i$，其**相对优势**估计为：
$$
\hat{A}_i = \frac{r_i - \mu_R}{\sigma_R + \epsilon}
$$
其中 $\epsilon$ 是极小常数，防止除以零（当所有回答奖励相同时 $\sigma_R = 0$）。

**极端情况分析**：

- 全对 $r = [1,1,1,1]$：$\sigma_R = 0$，$\hat{A}_i = 0$ → 不更新（都对了，没什么可学的）。

- 全错 $r = [0,0,0,0]$：$\sigma_R = 0$，$\hat{A}_i = 0$ → 不更新（都错了，没有正样本可以学习）。

- 一对三错 $r = [1,0,0,0]$：$\hat{A}_1 = +1.73$，$\hat{A}_{2,3,4} = -0.58$ → 大力强化唯一的正确回答。

这种"全对/全错时不更新"的行为避免了在没有区分信号时引入噪声梯度。

## 2. KL 散度正则化：为什么用这个特殊形式？

为了防止策略"钻空子"（Reward Hacking）或丧失语言连贯性，需要约束 $\pi_\theta$ 不偏离参考策略 $\pi_{\text{ref}}$ 太远。

标准的 KL 散度定义为：

$$
D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_{o \sim \pi_\theta}\left[\log \frac{\pi_\theta(o|s)}{\pi_{\text{ref}}(o|s)}\right]
$$

它需要对 $\pi_\theta$ 的完整分布求期望，但我们手头只有来自 $\pi_{\theta_{\text{old}}}$ 的有限采样。直接用采样估计 $D_{\text{KL}}$ 会引入较大方差。GRPO 转而使用一种基于 **$f$-散度** 的替代量。

### 什么是 $f$-散度（$f$-divergence）？

$f$-散度是一族衡量"两个概率分布有多不同"的度量，统一公式为：

$$D_f(P \| Q) = \mathbb{E}_{x \sim Q}\left[f\!\left(\frac{P(x)}{Q(x)}\right)\right]$$

| 符号 | 含义 |
|:---|:---|
| $P, Q$ | 两个概率分布 |
| $\frac{P(x)}{Q(x)}$ | **似然比**：$P$ 和 $Q$ 在同一个 $x$ 上的概率之比 |
| $f(\cdot)$ | 一个凸函数，满足 $f(1) = 0$（当 $P = Q$ 时散度为零） |

不同的 $f$ 对应不同的经典散度：

| 选择的 $f(u)$ | 对应的散度 |
|:---|:---|
| $u \log u$ | 正向 KL 散度 $D_{\text{KL}}(P \| Q)$ |
| $-\log u$ | 反向 KL 散度 $D_{\text{KL}}(Q \| P)$ |
| $(\sqrt{u} - 1)^2$ | Hellinger 距离 |
| $(u-1)^2$ | $\chi^2$ 散度 |
| $u - \log u - 1$ | **GRPO 使用的形式**（下面推导） |

**关键优势**：$f$-散度的期望是在 $Q$（即 $\pi_\theta$）下取的——而我们**正好有来自 $\pi_\theta$（或 $\pi_{\theta_{\text{old}}}$）的采样**，可以直接用样本估计，不需要对整个分布积分。

### GRPO 的 $f$-散度选择

取 $f(u) = u - \log u - 1$，令 $P = \pi_{\text{ref}}$、$Q = \pi_\theta$、$u = \frac{\pi_{\text{ref}}(o_i|s)}{\pi_\theta(o_i|s)}$，单样本估计量为：

$$
\hat{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \underbrace{\frac{\pi_{\text{ref}}(o_i|s)}{\pi_\theta(o_i|s)}}_{u} - \underbrace{\log \frac{\pi_{\text{ref}}(o_i|s)}{\pi_\theta(o_i|s)}}_{\log u} - 1
$$

**为什么选 $f(u) = u - \log u - 1$？**

- $f(1) = 1 - 0 - 1 = 0$：当 $\pi_\theta = \pi_{\text{ref}}$ 时（$u = 1$），惩罚为零。

- $f''(u) = 1/u^2 > 0$：严格凸，$u = 1$ 是唯一最小值点。

- **双侧惩罚**：当 $\pi_\theta \ll \pi_{\text{ref}}$（$u \gg 1$）时 $f(u) \approx u$（线性增长），当 $\pi_\theta \gg \pi_{\text{ref}}$（$u \to 0$）时 $f(u) \approx -\log u$（对数增长）。两个方向的偏离都会被惩罚，防止概率塌缩或异常膨胀。

> **与 PPO 的 KL 惩罚对比**：PPO 使用 $\beta \cdot (\log \pi_\theta - \log \pi_{\text{ref}})$ 逐 token 加到奖励上（见上一篇的 `kl_penalty`）。GRPO 将 KL 惩罚直接作为损失的一部分，并使用 $f$-散度形式，对概率塌缩更敏感。

### Token 级别的操作：从序列到单 token

上面 $\hat{D}_{\text{KL}}$ 中的 $u = \frac{\pi_{\text{ref}}(o_i|s)}{\pi_\theta(o_i|s)}$ 和下面最终目标函数中的重要性比率 $\rho_i = \frac{\pi_\theta(o_i|s)}{\pi_{\theta_{\text{old}}}(o_i|s)}$ 都涉及**整条回答的概率**。但语言模型是自回归的——逐 token 生成。所以需要说明这些"序列级"的量如何从"token 级"构造出来。

**序列概率 = 各 token 条件概率的乘积**（链式法则）：

$$
\pi_\theta(o_i|s) = \prod_{t=1}^{T} \pi_\theta(o_i^t | s, o_i^{<t})
$$

取对数后乘积变为求和（计算更稳定）：

$$
\log \pi_\theta(o_i|s) = \sum_{t=1}^{T} \log \pi_\theta(o_i^t | s, o_i^{<t})
$$

**重要性比率**（用于 PPO 裁剪）利用 $\log$ 相减等价于概率相除：

$$
\rho_i(\theta) = \frac{\pi_\theta(o_i|s)}{\pi_{\theta_{\text{old}}}(o_i|s)} = \exp\left(\sum_{t=1}^{T} \left[\log \pi_\theta(o_i^t | s, o_i^{<t}) - \log \pi_{\theta_{\text{old}}}(o_i^t | s, o_i^{<t})\right]\right)
$$

KL 散度中的 $u = \frac{\pi_{\text{ref}}}{\pi_\theta}$ 同理构造。上式给出的是**序列级**似然比 $\rho_i = \prod_t \rho_{i,t}$，它等于各 token 比率的乘积。但实际实现中**不**计算这个乘积（长序列会导致数值溢出或下溢），而是在每个 token 位置 $t$ 独立计算 $\rho_{i,t}$ 并直接用于 PPO 的逐 token 裁剪代理目标（即对每个 $t$ 分别做 $\min(\rho_{i,t} \hat{A}_i, \text{clip}(\rho_{i,t}) \hat{A}_i)$），然后对 $t$ 求平均。这意味着裁剪在 **token 级** 逐位执行，而非对序列级 $\rho_i$ 做单次裁剪。

## 3. GRPO 最终目标函数

结合 PPO 的裁剪机制和组内相对优势，GRPO 的最终目标函数（需要最大化）定义为：

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q),\, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min \left( \rho_{i,t}(\theta)\, \hat{A}_i,\; \text{clip}(\rho_{i,t}(\theta),\, 1-\varepsilon,\, 1+\varepsilon)\, \hat{A}_i \right) - \beta\, \hat{D}_{\text{KL}}^{(i,t)} \right) \right]
$$

其中：

- $\rho_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}$ 是第 $i$ 个回答第 $t$ 个 token 的重要性采样比率。

- $\varepsilon$ 是裁剪阈值（如 0.2），防止单步更新过大。

- $\beta$ 是 KL 惩罚系数，控制偏离参考策略的代价。

- $\hat{A}_i$ 是组内归一化优势（序列级标量，对 $t$ 为常数，广播到每个 token）。

- $\hat{D}_{\text{KL}}^{(i,t)}$ 是 token 级 KL 散度近似估计（采用 $e^{\log u} - \log u - 1$ 形式，其中 $u = \pi_{\text{ref}} / \pi_\theta$，详见下文），并非严格的 $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ 积分定义。

- $\frac{1}{|o_i|}$ 对每条回答按长度归一化（per-response normalization）：先对 token 求平均，再对 $G$ 条回答平均。这意味着不论回答长短，每条回答的权重都是 $\frac{1}{G}$。后续 DAPO 论文将此聚合方式改为按 token 归一化（$\frac{1}{\sum_i |o_i|}\sum_i\sum_t$，使每个 token 等权），详见[第六篇](/chengYi-xun/posts/56-dapo/)。

> **注**：上式使用 token 级记号 $\rho_{i,t}$，与 DeepSeekMath 原论文（arXiv:2402.03300）从 PPO 继承的 $\frac{1}{|o|}\sum_t$ 结构一致。**GRPO 的所有计算**（IS ratio、裁剪、KL）**均在 token 级逐位执行**——此处"序列级"仅指**聚合方式**（每条回答等权），不是计算粒度。DAPO 改变的是聚合方式（从 per-response 到 per-token），而非引入 token 级计算。


---

# GRPO 的完整实现

以下是 GRPO 的完整 PyTorch 实现伪代码，包括数据准备、模型定义、采样、优势计算和训练循环。

**Step 1: 模型与数据定义**

与 DPO 不同，GRPO 是**在线**算法：不需要预先收集偏好对，只需要 Prompt 集合和一个能打分的奖励函数。模型方面与 DPO 一样只需要两个（策略 + 参考），但保留了在线 RL 的探索能力。

```python
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    AutoModelForSequenceClassification,
)

# 数据: 只需要 Prompt 集合，无需预标注偏好对
prompts = load_dataset("math_problems")

# 模型 1: 待训练的策略模型 (π_θ)
actor = AutoModelForCausalLM.from_pretrained("sft_checkpoint")
# 模型 2: 冻结的参考模型 (π_ref), KL 正则锚点
ref_model = AutoModelForCausalLM.from_pretrained("sft_checkpoint")
ref_model.requires_grad_(False)

tokenizer = AutoTokenizer.from_pretrained("sft_checkpoint")
# decoder-only 模型 batch 生成时必须左填充,
# 确保所有序列的最后一个真实 token 右对齐在同一列,
# 否则 padding 会插在序列末尾, 破坏自回归生成的连续性
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-6)

# 奖励函数: 可以是规则判题器（数学题判对错）或训练好的奖励模型
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "reward_model_checkpoint"
)
reward_model.requires_grad_(False)

def reward_fn(prompt, response_text):
    """用奖励模型给回答打分，返回标量奖励"""
    inputs = tokenizer(
        prompt + response_text,
        return_tensors="pt", truncation=True
    ).to(reward_model.device)
    with torch.no_grad():
        score = reward_model(**inputs).logits.squeeze()
    return score.item()

# 超参数
G = 8             # 每个 Prompt 采样的回答数量
clip_range = 0.2  # PPO 裁剪阈值 ε
beta = 0.04       # KL 惩罚系数
K_epochs = 2      # 每批数据的更新轮数
```

**Step 2: 在线采样 + 奖励收集**

这是 GRPO 与 DPO 的核心区别——GRPO 用当前策略**在线生成**回答并即时评分：

```python
def collect_group_rollouts(actor, prompts_batch, G, reward_fn):
    """
    对每个 Prompt 采样 G 个回答并打分。

    Returns:
        all_prompt_ids:   (B×G, L_p)   prompt token 序列
        all_response_ids: (B×G, L_r)   回答 token 序列（不含 prompt）
        all_rewards:      List[float]  标量奖励, 长度 B×G
        old_log_probs:    (B×G, L_r)   采样时各 token 的 log π_old
        ref_log_probs:    (B×G, L_r)   参考策略各 token 的 log π_ref
    """
    actor.eval()
    with torch.no_grad():
        # Step 2a: 编码所有 prompt 并复制 G 份
        encoded = tokenizer(
            prompts_batch,
            return_tensors="pt", padding=True
        ).to(actor.device)
        # input_ids: (B, L_p), attention_mask: (B, L_p)

        # repeat_interleave 让相邻 G 行属于同一 prompt: (B, L_p) → (B×G, L_p)
        all_prompt_ids = encoded.input_ids.repeat_interleave(G, dim=0)
        all_attn_mask = encoded.attention_mask.repeat_interleave(G, dim=0)

        # Step 2b: batch 生成所有回答
        # 必须传 attention_mask, 否则模型把 padding 当真实 token 处理
        full_ids = actor.generate(
            all_prompt_ids,
            attention_mask=all_attn_mask,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7
        )  # (B×G, L_p + L_r)

        # 分离 response: generate() 返回完整序列, 截掉 prompt 前缀
        prompt_len = all_prompt_ids.shape[1]
        all_response_ids = full_ids[:, prompt_len:]  # (B×G, L_r)

        # Step 2c: 只解码 response 部分, 避免 prompt 文本被重复送入奖励函数
        all_response_texts = tokenizer.batch_decode(
            all_response_ids, skip_special_tokens=True
        )  # List[str], 长度 B×G

        prompts_repeated = [p for p in prompts_batch for _ in range(G)]
        all_rewards = [
            reward_fn(p, t)
            for p, t in zip(prompts_repeated, all_response_texts)
        ]  # List[float], 长度 B×G

        # Step 2d: 计算 token 级 log 概率 (后面算 ρ 和 KL 要用)
        old_log_probs = compute_token_log_probs(
            actor, all_prompt_ids, all_response_ids
        )  # (B×G, L_r): π_old 下每个 token 的 log 概率
        ref_log_probs = compute_token_log_probs(
            ref_model, all_prompt_ids, all_response_ids
        )  # (B×G, L_r): π_ref 下每个 token 的 log 概率

    actor.train()
    return (all_prompt_ids, all_response_ids,
            all_rewards, old_log_probs, ref_log_probs)
```

**Step 3: 组内相对优势计算**

```python
def compute_group_advantages(rewards, G):
    """
    组内相对优势计算: 用组内均值替代 Critic 网络。

    Args:
        rewards: List[float], 长度 = batch_size × G
        G:       int, 每个 prompt 的采样数
    Returns:
        Tensor (batch_size × G,), 标准化后的相对优势 Â_i
    """
    # 显式指定 float32, 避免整数奖励 (如 0/1 判对错) 被推断为 int 导致除法截断
    rewards = torch.tensor(rewards, dtype=torch.float32).reshape(-1, G)
    # rewards: (batch_size, G)

    mean_r = rewards.mean(dim=1, keepdim=True)   # (batch_size, 1)

    # correction=0 → 总体标准差 σ = √(1/G Σ(r-μ)²), 与文中公式一致
    # PyTorch 默认 correction=1 是样本标准差 (除以 G-1), G 较小时差异明显
    std_r = rewards.std(dim=1, keepdim=True, correction=0)  # (batch_size, 1)

    # 全对/全错时 σ=0, 分子也为 0, ε 防除零, Â_i=0 → 不更新
    advantages = (rewards - mean_r) / (std_r + 1e-8)
    return advantages.reshape(-1)   # (batch_size × G,)
```

**Step 4: 完整训练循环**

```python
for step in range(total_steps):

    # ================================================================
    # 阶段 1: 在线采样 (GRPO 独有, DPO 没有这一步)
    # ================================================================
    prompts_batch = sample_prompts(prompts, batch_size=8)

    (prompt_ids, response_ids, rewards,
     old_log_probs, ref_log_probs) = collect_group_rollouts(
        actor, prompts_batch, G, reward_fn
    )
    # old_log_probs, ref_log_probs: (B×G, L_r) token 级

    # ================================================================
    # 阶段 2: 计算组内相对优势 Â_i
    # ================================================================
    advantages = compute_group_advantages(rewards, G).to(actor.device)
    # advantages: (B×G,)

    # ================================================================
    # 阶段 3: 多 epoch 更新
    # 同一批采样数据复用 K 轮, 靠 PPO 裁剪防止更新过大
    # ================================================================
    for epoch in range(K_epochs):
        for idx in minibatch_indices(len(response_ids), batch_size=16):

            # --- 3a: 重新计算当前 π_θ 的 token 级 log 概率 ---
            # π_θ 每个 minibatch 后都在更新, 必须用最新 θ 重算
            new_log_probs = compute_token_log_probs(
                actor, prompt_ids[idx], response_ids[idx]
            )  # (M, L_r): 每个 token 的 log π_θ

            # response 中的 padding token 不应参与损失计算
            response_mask = (response_ids[idx] != tokenizer.pad_token_id)
            # response_mask: (M, L_r), True 为真实 token

            # --- 3b: token 级重要性采样比率 ---
            # 在 token 级别算比率, 避免序列级乘积导致数值爆炸
            log_ratio = new_log_probs - old_log_probs[idx]  # (M, L_r)
            ratio = torch.exp(log_ratio)                     # (M, L_r)

            # --- 3c: PPO 裁剪 (token 级) ---
            # 优势是序列级标量, 广播到每个 token
            adv = advantages[idx].unsqueeze(-1)  # (M,) → (M, 1)
            surr1 = ratio * adv                              # (M, L_r)
            surr2 = torch.clamp(
                ratio, 1.0 - clip_range, 1.0 + clip_range
            ) * adv                                          # (M, L_r)
            # 只对真实 token 求均值, 忽略 padding
            clipped_obj = torch.min(surr1, surr2) * response_mask
            policy_loss = -clipped_obj.sum() / response_mask.sum()

            # --- 3d: KL 散度惩罚 (token 级 f-散度) ---
            log_u = ref_log_probs[idx] - new_log_probs  # log(π_ref/π_θ)
            kl_per_token = torch.exp(log_u) - log_u - 1.0
            kl_penalty = (kl_per_token * response_mask).sum() / response_mask.sum()

            # --- 3e: 总损失 ---
            loss = policy_loss + beta * kl_penalty

            # --- 3f: 梯度更新 ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            optimizer.step()
```

> **与 DPO 训练循环的关键对比**：
>
> - DPO 只有"前向传播 + 损失计算 + 反向传播"，与标准 SFT 训练几乎一样。
> - GRPO 多了"在线采样"阶段（调用 `actor.generate()`），这是计算开销的主要来源，但也是在线 RL 探索能力的来源。
> - DPO 的 batch 是固定的偏好对；GRPO 的 batch 是模型自己实时生成的，每步训练都能看到新的探索结果。

---

# GRPO 与 PPO / DPO 的全景对比

| 维度 | PPO (RLHF) | DPO | GRPO |
| :---: | :---: | :---: | :---: |
| **模型数量** | 4 (Actor+Critic+Ref+RM) | 2 (Actor+Ref) | 2 (Actor+Ref) + 外部奖励函数 |
| **训练方式** | 在线 RL | 离线监督学习 | 在线 RL |
| **基线估计** | Critic 网络 $V_\phi(s)$ | 无需基线 | 组内经验均值 $\mu_R$ |
| **显存开销** | 极高 (4 个大模型) | 低 (2 个大模型) | 低 (2 个大模型) |
| **计算开销** | 中等 (每 Prompt 采样 1 次) | 最低 (纯前向传播) | 较高 (每 Prompt 采样 G 次) |
| **探索能力** | 强 | 弱 (离线数据) | 强 |
| **核心优势** | 经典稳定 | 极简高效 | 省显存 + 在线探索 |


**开源代码参考：** GRPO 随 DeepSeek 开源而爆火，Hugging Face **TRL** 库 ([`trl.GRPOTrainer`](https://huggingface.co/docs/trl/grpo_trainer)) 提供了生产级实现。

GRPO 证明了在生成式大模型时代，简单的经验统计（组内均值）往往比复杂的神经网络预测（Critic）更加鲁棒和高效。

> 参考资料：
>
> 1. Shao, Z., Wang, P., Zhu, Q., Hao, K., Bugliarello, B., ... & Liu, Y. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300.
> 2. Ahmadian, A., Cremer, C., Gallé, M., Fadaee, S., ... & Vulić, A. (2024). *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs*. arXiv:2402.14740.

> 下一篇：[笔记｜强化学习（五）：Flow-GRPO 与图像生成应用（基于 Flux 的代码解析）](/chengYi-xun/posts/55-flow-grpo/)
