---
title: 笔记｜强化学习（十二）：奖励模型进阶——Reward Hacking、生成式奖励模型与可验证奖励
date: 2026-04-05 19:00:00
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

> 本篇是 [上一篇（奖励模型基础）](/chengYi-xun/posts/61-reward-model/) 的续篇。上一篇完成了代理裁判 $\hat{r}_\theta$ 的构建，本篇追问三个递进的问题：**(1)** 拼命优化一个不完美的裁判会怎样？**(2)** 如何让裁判更健壮？**(3)** 能否绕过裁判？
>
> ⬅️ 上一篇：[笔记｜强化学习（十一）：奖励模型基础——从传统 RL 到大模型与视觉生成](/chengYi-xun/posts/61-reward-model/)

## 1 Reward Hacking：当优化器攻击裁判

### 1.1 问题定义与优化动力学

> **阶段澄清**：RLHF 分两步——阶段一用 BT 损失训练奖励模型 $\hat{r}_\theta$（训裁判），阶段二冻结 $\hat{r}_\theta$ 用 PPO 等算法训练策略 $\pi$（考选手）。**Reward Hacking 发生在阶段二**。

上一篇完成了阶段一。现在进入阶段二：我们冻结 $\hat{r}_\theta$，让策略 $\pi$ 最大化期望得分。由于 $\hat{r}_\theta$ 只是真实偏好 $r^*$ 的近似，将其输出分解为

$$
\hat{r}_\theta(x,y) = r^*(x,y) + \epsilon(x,y)
$$

其中 $r^*$ 是不可观测的真实人类偏好，$\epsilon$ 为 RM 的系统性拟合误差。代入策略优化目标：

$$
\max_\pi \; \mathbb{E}_{y \sim \pi}[\hat{r}_\theta(x,y)] \;=\; \underbrace{\mathbb{E}[\,r^*(x,y)\,]}_{\text{目标 1：提升真实质量}} \;+\; \underbrace{\mathbb{E}[\,\epsilon(x,y)\,]}_{\text{目标 2：利用 RM 误差}}
$$

优化器**无法区分**这两项，它只看到一个总分。RL 是一个极强的主动探索器，会像水流一样寻找提升总分最容易的路径。这导致了三阶段的动力学：

1. **初期（分布内优化）**：策略生成的内容仍在 RM 的训练数据分布内，$\epsilon \approx 0$。此时提升总分最有效的路径就是提升 $r^*$——模型在真实地变好。
2. **瓶颈期**：$r^*$ 越来越高，继续提升的边际成本急剧上升（例如从 90 分到 95 分需要更强的推理能力）。
3. **Hacking 期（分布外漂移）**：策略被推至 RM 训练分布之外（OOD）。RM 作为神经网络，在未见过的输入区域必然存在外推失控的"假山峰"——这些点上 $r^*$ 很低但 $\epsilon$ 极大。优化器发现：**提升 $\epsilon$ 的收益远大于提升 $r^*$，且难度极低**。于是概率质量被集中到这些假山峰上，RM 给出的分数（即 Proxy 分数，因为 RM 只是真实偏好的"代理"）飙升，而真实质量崩塌。

这就是 **Reward Hacking**（奖励过度优化）。其理论根源是 **古德哈特定律**（Goodhart's Law）：当代理指标本身变成优化目标时，它不再是好的度量指标（Goodhart, 1975; Manheim & Garrabrant, 2019）。

**实践中常见的三类 Hacking 形态**（Weng, 2024）：

| 类型 | 机制 | 典型表现 |
|:---|:---|:---|
| **冗长（Verbosity）** | RM 隐含"长 = 详尽 = 好"的偏见 | 一句话能说清楚的事写成五段废话 |
| **谄媚（Sycophancy）** | RM 对"同意用户"给高分 | 即使用户说"1+1=3"也附和 |
| **格式套路** | RM 对特定格式特征过拟合 | 堆砌粗体、列表、"作为 AI 语言模型"等安全声明 |

它们的共同特征是 $\hat{r}_\theta$ 高分而 $r^*$ 低分——在代理裁判眼里满分，在人类眼里毫无价值。

### 1.2 过度优化的缩放定律

上面的三阶段分析是定性的。Gao et al.（*Scaling Laws for Reward Model Overoptimization*, ICML 2023）用一个精心设计的实验将其**定量化**了。

**实验设计**：由于真实人类偏好 $r^*$ 不可观测，他们构造了一个合成代理框架来模拟整个过程——

- **Gold RM**（黄金裁判）：一个参数量很大、拟合能力很强的奖励模型，**充当实验中的"真实人类"**。我们把它的评分视为真实偏好 $r^*$ 的最佳近似。
- **Proxy RM**（代理裁判）：用 Gold RM 的少量标注数据训练出的**较小模型**。这就是实际 RLHF 中那个不完美的裁判 $\hat{r}_\theta$。

然后让策略模型拼命优化 Proxy 的分数，同时用 Gold 的分数来衡量策略是否真的在变好。如果两条曲线同步上升，说明 Proxy 可靠；如果 Proxy 一路飙升而 Gold 掉头向下，就证明 Reward Hacking 正在发生。

**核心发现——倒 U 型曲线**：

![Proxy 与 Gold 分数在优化过程中的变化趋势（示意图，基于 Gao et al. 2023 的实验规律）](/chengYi-xun/img/scaling-law-flowchart.png)

如图所示，用策略与初始策略的 KL 散度度量"策略跑了多远"（横轴），Gold 分数（真实质量，蓝色实线）呈现一条**先升后降的倒 U 型曲线**——而 Proxy 分数（红色虚线）始终单调上升。峰值点（Peak）就是优化的"甜蜜点"；越过它之后，继续优化只是在钻 Proxy 的漏洞，真实质量不断崩塌。

这一规律在 Best-of-$N$（拒绝采样）和 RL（PPO）下都被观察到，且曲线的关键参数随 Proxy RM 参数量**平滑缩放**（近似对数趋势），因此可以用小规模实验预测大规模训练的行为。

**工程启示——如何找到最优停止点？**

Gao et al. 的缩放律之所以重要，是因为曲线参数随 Proxy RM 的参数量呈**对数平滑变化**。这意味着你可以在小规模上（小 Proxy RM + 短训练）拟合出 $\alpha, \beta$ 的趋势线，然后**外推**到大规模训练的预期行为——包括峰值出现在哪个 KL 值（Gao et al., 2023, Fig. 12）。

实际工程中，常用的做法包括：

1. **KL 预算法**：在 PPO 训练前预设一个 KL 上限（KL budget），策略偏离超过阈值就强制停止。这等价于在倒 U 型曲线上"提前下车"。
2. **Hold-out 监控法**：保留一组 Gold 标注数据（或用更强的模型评分），定期评估。当 Proxy 分数仍在涨但 Hold-out 分数开始掉时，说明已进入 Hacking 阶段，应立即停止。
3. **迭代 RLHF（Iterated RLHF）**：不跑一次到底。每训练一段时间后暂停 PPO，让当前策略模型生成一批新回答，请人类标注员对这些新回答做偏好排序，然后用这些新标注数据重新训练 RM。这样 RM 见过的数据就不再局限于初始阶段的回答，而是覆盖了策略模型"现在正在说的话"，策略更难跑到 RM 的盲区去钻空子（Gao et al., §4.3）。
4. **RM 集成**：同时训练多个不同初始化 / 不同数据分割的 RM，取平均分。单个 RM 的漏洞不太可能被所有 RM 共享，集成可以有效压缩 $\epsilon$（Lambert, *RLHF Book*, Ch. 14）。

### 1.3 KL 正则化为何不是万能药

既然过度优化会导致崩溃，一个自然的想法是：**限制策略不要跑太远**。标准 PPO 正是这么做的——在奖励目标上加一个 KL 惩罚项，强迫策略 $\pi$ 不要偏离参考策略 $\pi_{\mathrm{ref}}$ 太多。直觉上，如果策略被拴在参考策略附近，它就没法跑到 RM 的分布外去钻空子了。

但 Kwa et al.（*Catastrophic Goodhart*（灾难性古德哈特）, NeurIPS 2024）从理论上证明了这种保护并不可靠。他们的核心发现取决于 RM 误差 $\epsilon$ 的**尾部性质**：

![轻尾 vs 重尾误差下 KL 惩罚的有效性对比（示意图，基于 Kwa et al., NeurIPS 2024）](/chengYi-xun/img/kl-tail-comparison.png)

**轻尾误差（如高斯分布，左图）**：参考策略的输出分布（蓝色）在远离中心后概率急剧衰减为零，尾部几乎没有概率质量。优化后的策略（橙色虚线）即使想搬运质量到高 $\epsilon$ 区域，也找不到多少"原料"可搬——因为参考分布在那里本来就接近零。要到达那些区域，策略需要做大幅度的分布变动，KL 代价会急剧上升，远超收益。所以 KL 惩罚有效。

**重尾误差（如帕累托分布、柯西分布，右图）**：参考策略的输出分布（蓝色）在极端区域仍有缓慢衰减的概率尾巴——这就是重尾的特征。如图所示，优化后的策略（红色虚线）只需将极端尾部的概率**稍微放大**（红色阴影区），就完成了 Hacking。由于参考分布在那里本来就有概率质量（虽然小），这个放大操作的 KL 代价极低；但那些位置上的 $\epsilon$ 值却异常巨大，收益远超代价。

**关键机制**：你可能会问——如果误差很大，策略岂不是要"跑很远"才能碰到那些极端漏洞？那 KL 不也会很大吗？

答案是：**不需要跑很远**。重尾分布的特性是，在参考策略 $\pi_{\mathrm{ref}}$ 的输出空间里，就已经存在一些**极其罕见但 $\epsilon$ 巨大**的输出 $(x, y)$。优化器不需要创造新的输出模式，它只需要把这些本来就存在的罕见输出的概率**稍微调高一点点**——比如从 0.001% 调到 0.01%。

这个微小的概率调整在 KL 散度看来几乎无关紧要（绝大多数输出的概率没变，KL 几乎为零）。但由于那些点上的 $\epsilon$ 异常巨大（重尾极值），这一点点概率质量的转移就能带来巨大的代理奖励收益。**KL 惩罚感受到的是"概率搬了多少"（对数级），而奖励收益取决于"搬到的地方 $\epsilon$ 有多大"（可以任意大）——两者增长速度不匹配。**

Kwa et al. 将这种现象命名为**灾难性古德哈特**（Catastrophic Goodhart）：存在某些策略，RM 给它的分数（代理奖励）无限高、KL 散度却趋近于零，但真实效用不比初始策略好——KL 惩罚形同虚设。

> **实证补充**：Kwa et al. 用 Pythia-1.4B 和 Starling-7B 等开源 RM 做了实验，发现当前模型的误差分布主要呈**轻尾**特征，所以现阶段 KL 正则仍然管用。但作者警告：随着 RM 覆盖的任务更广、评估维度更复杂，重尾误差出现的可能性会增加。灾难性古德哈特不是当下的问题，而是一个**理论红线**——它告诉我们 KL 惩罚的保护不是无条件的。

**阶段性结论**：

1. **缩放定律**（Gao et al.）表明：给定一个 RM，真实的优化收益存在**天花板**——倒 U 型曲线的峰值就是极限，超过它就是在毁灭真实质量。
2. **灾难性古德哈特**（Kwa et al.）表明：光靠调大 KL 惩罚系数 $\beta$ 是不够的——在重尾误差下，KL 惩罚的对数增长永远追不上极端漏洞的收益。

因此，必须从 RM 架构本身入手——要么**造更好的裁判**（GenRM、ArmoRM），要么在适用场景下**绕过裁判**（DeepSeek-R1 的可验证奖励）。

## 2 造更好的裁判：从 AI 评委到生成式奖励模型

### 2.1 AI 反馈的三条路线

传统 RM 的瓶颈在于人类标注成本。为此，学术界发展出三条技术路线（Lee et al., 2023; Bai et al., 2022）：

| 路线 | 核心思路 | 代表工作 | 局限 |
|:---|:---|:---|:---|
| **LLM-as-a-Judge** | 直接 Prompt GPT-4 输出偏好判决 | Zheng et al., 2023 | 依赖闭源 API；存在系统性偏差 |
| **RLAIF** | AI 批量标注 → 蒸馏为轻量 BT-RM | Constitutional AI (Bai et al., 2022) | 蒸馏后信息有损 |
| **GenRM** | 微调开源 LLM 为生成式评委 | Mahan et al., 2024 | 推理成本较高 |

三者并非互斥：RLAIF 可用 LLM-as-Judge 的输出做标注源，GenRM 可在 RLAIF 数据上训练。

### 2.2 GenRM：让评委先推理再判决

Mahan et al.（*Generative Reward Models*, 2024）提出将 RM 从标量黑盒转变为**先推理、后判决**的生成式模型。

**基础 GenRM** 把 LLM 视为分类器，给定 prompt $x$ 和候选对 $(y_1, y_2)$，预测胜出标记 $I$：

$$
\mathcal{L}_{\mathrm{GenRM}} = \mathbb{E}_{(x, y_1, y_2, I)} \bigl[-\log \pi_\phi(I \mid x, y_1, y_2)\bigr]
$$

**CoT-GenRM** 要求模型在判决前先生成推理链 $c$，训练目标变为联合似然：

$$
\mathcal{L}_{\mathrm{CoT}} = \mathbb{E} \bigl[-\log \pi_\phi(c \mid x, y_1, y_2) - \log \pi_\phi(I \mid x, y_1, y_2, c)\bigr]
$$

CoT 的关键价值在于：$\log \pi_\phi(c \mid \cdot)$ 项将中间推理结构暴露为可训练的监督信号，直接惩罚了黑盒 RM 容易学到的捷径（如"长回答=好回答"）。实验显示 CoT-GenRM 在 OOD 泛化上较 BT-RM 提升 10%–45%（Mahan et al., 2024, Fig. 3）。

后续工作进一步发展了 GenRM 范式：**Think-RM**（NeurIPS 2025）引入长程推理链训练；**Rationale Consistency** 指标（2026）用于检测推理链是否为事后合理化（post-hoc rationalization），而非因果性推理。

### 2.3 AI 评委的系统性偏差

无论采用何种 AI 评委，以下三类偏差均需关注（Zheng et al., 2023; Weng, 2024）：

- **位置偏差（Position Bias）**：倾向偏好先出现的回答。缓解：交换 A/B 顺序做两次评判，不一致则判平。
- **冗长偏差（Verbosity Bias）**：将冗长误判为详尽。缓解：Prompt 约束或在训练数据中加入"短精 > 长空"样本。
- **自我偏好（Self-preference Bias）**：对同家族模型的输出给出更高评分（低困惑度效应）。缓解：跨家族交叉评判。

## 3 绕过单一标量：多目标 RM 与可验证奖励

### 3.1 ArmoRM：多目标评分 + 门控聚合

Wang et al.（*Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts*, EMNLP Findings 2024）指出，单一标量 RM 无法区分"太啰嗦"与"太冒险"——策略 exploit 时难以诊断。ArmoRM 将奖励分解为多个可解释维度：

**阶段一：多目标回归。** 线性头将 LLM 末层表征映射为 $k$ 维评分向量：

$$
\min_{\theta, w}\; \mathbb{E}_{(x,y,r) \in \mathcal{D}}\; \bigl\| w^\top f_\theta(x \oplus y) - r \bigr\|_2^2, \quad r \in \mathbb{R}^k
$$

其中各维度对应诚实、安全、简洁等绝对评分，缺失维度不参与损失计算。

**阶段二：门控 MoE 聚合。** 冻结主干与回归头，门控网络 $g_\phi$ 仅读取 prompt 特征 $f_\theta(x)$，经 softmax 输出非负权重向量。对除冗长度外的目标做去相关修正得 $r'$，最终标量奖励为

$$
R = g_\phi\bigl(f_\theta(x)\bigr)^\top r'
$$

门控参数用成对偏好数据做 BT 优化（带可学习温度 $\beta$）。该设计将"各维度表现如何"与"本轮对话更看重什么"解耦，使诊断和纠偏成为可能。ArmoRM-Llama3-8B-v0.1 在 RewardBench 上取得了同规模 SOTA（Wang et al., 2024）。

![ArmoRM 架构：单次前向传播、多目标评分与门控合成](/chengYi-xun/img/armorm-architecture.png)

### 3.2 可验证奖励：DeepSeek-R1 的双轨范式

DeepSeek-AI（*DeepSeek-R1*, Nature 2025; arXiv:2501.12948）提出了更激进的方案：在**对错可判定**的任务上，直接用规则替代神经网络裁判。

DeepSeek-R1-Zero 在数学 / 代码 / 逻辑任务上采用纯规则奖励：

$$
R_{\mathrm{rule}} = R_{\mathrm{acc}} + R_{\mathrm{format}}
$$

其中 $R_{\mathrm{acc}}$ 由编译器或正则匹配判定答案正确性，$R_{\mathrm{format}}$ 检查 `<think>...</think>` 等格式约束。团队显式避免使用神经 RM，理由是大规模 RL 下其 Hacking 风险过高（DeepSeek-AI, 2025, §2.3）。

配合 **GRPO**（Group Relative Policy Optimization）算法——以组内采样的均值/标准差归一化优势函数 $A_i = (r_i - \mu_G) / \sigma_G$，免去显式 Critic 模型——R1-Zero 仅凭纯 RL 便涌现出了自我反思、回溯验证等推理策略（所谓"aha moment"）。

但 R1 **并非全局抛弃学习式 RM**。在通用对齐阶段（helpfulness、safety），论文仍使用学习式 RM 配合多源奖励 $R_{\mathrm{General}}$。准确画像是**双轨范式**：

| 任务类型 | 奖励来源 | 优势 |
|:---|:---|:---|
| 可验证（数学、代码、格式） | 规则硬奖励 | 零 Hacking 空间，零训练成本 |
| 开放对话与价值判断 | BT-RM / GenRM | 覆盖主观偏好 |

这构成了 2024–2026 年最具代表性的路线之一：**推理用可验证，通用用 RM**（亦见 Promptfoo, 2025; Mitra, 2026 对 RLVR 范式的讨论）。

## 4 视觉生成中的奖励模型

图像 / 视频的输出 $y$ 是高维连续信号，既缺少可验证真值，又面临维度分裂（对齐 vs 美感 vs 时序一致性）。

### 4.1 ImageReward

Xu et al.（*ImageReward*, NeurIPS 2023）在约 13.7 万对专家偏好数据上，训练了基于 BLIP 编码器 + MLP 头的跨模态偏好模型：

$$
s(x, y_{\mathrm{img}}) = \mathrm{MLP}\bigl(f_{\mathrm{BLIP}}(x, y_{\mathrm{img}})\bigr), \quad \mathcal{L} = -\log \sigma\bigl(s(x, y_w) - s(x, y_l)\bigr)
$$

该模型联合刻画对齐、美感与细粒度质量，在与人类排序的一致性上显著优于同期的 CLIP Score 和纯美学模型。

### 4.2 Flow-GRPO：视觉 RL 的两大适配

Flow-GRPO（2025）将 GRPO 接入流匹配（Flow Matching）文生图模型，解决两个核心痛点：

**探索不足**：流匹配的默认 ODE 采样是确定性的，缺乏 RL 所需的试错空间。Flow-GRPO 将 ODE 改写为等价 SDE，在保持边际分布不变的前提下注入探索噪声：

$$
d\mathbf{x}_t = \Bigl[\mathbf{v}_\theta(\mathbf{x}_t, t) + 2t\sigma_t^2 \bigl(\mathbf{x}_t + (1-t)\mathbf{v}_\theta(\mathbf{x}_t, t)\bigr)\Bigr]\,dt + \sigma_t\,d\mathbf{W}_t
$$

每步转移变为各向同性高斯，从而可计算 GRPO 所需的概率比。

**算力开销**：采用 Denoising Reduction 技术，以更少步数近似推理轨迹。

与 LLM RLHF 的关键差异在于：(1) 动作空间为连续向量场而非离散 token；(2) 奖励几乎全是多目标加权；(3) 缓解 Hacking 需多奖励帕累托优化或条件化生成。

## 5 评测与选型

### 5.1 RewardBench

Lambert et al.（2024）构建了首个 RM 标准化评测基准 **RewardBench**，包含 (prompt, chosen, rejected) 三元组，覆盖 Chat / Chat-Hard / Safety / Reasoning 四个子集。核心指标为胜率：

$$
\mathrm{WinRate} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}\bigl[\hat{r}_\theta(x_i, y_i^{+}) > \hat{r}_\theta(x_i, y_i^{-})\bigr]
$$

**RewardBench 2**（Malik et al., 2025）进一步提升难度（平均得分下降约 20 分），并与下游 RLHF / Best-of-$N$ 性能的相关性更高。

### 5.2 选型决策树

| 判据 | 推荐方案 |
|:---|:---|
| 任务客观可验证（数学/代码/格式） | **规则奖励**（DeepSeek-R1 路线） |
| 开放对话、创意写作 | **BT-RM / GenRM / LLM 评委** |
| 需要可解释的维度权衡 | **多目标 RM**（ArmoRM） |
| 严重 OOD 挑战 | **CoT-GenRM** 或持续在线标注刷新 |
| 推理成本敏感 | 标量 BT-RM（一次前传） > GenRM（需生成 CoT） > GPT-4 评委（API 调用） |

## 6 结语

本文沿"问题 → 量化 → 解法"的主线展开：

1. **诊断**：Gao et al. 的缩放定律量化了过度优化的倒 U 型曲线；Kwa et al. 证明了 KL 惩罚在重尾误差下的理论失效。
2. **治标**：GenRM 用 CoT 显式化评审推理，OOD 泛化提升显著；ArmoRM 将黑盒标量拆解为多目标门控聚合，增强可诊断性。
3. **治本**：DeepSeek-R1 在可验证任务上直接用规则奖励替代神经 RM，从根源消除 Hacking 空间。
4. **视觉侧**：ImageReward 与 Flow-GRPO 分别在裁判构建与 RL 适配上推进了高维连续空间的偏好对齐。

奖励模型是 RLHF 的隐藏引擎。理解它的脆弱性与演进方向，才算真正看懂大模型对齐的底层逻辑。

> 参考文献：
>
> 1. Gao, L., Schulman, J. & Hilton, J. (2023). *Scaling Laws for Reward Model Overoptimization*. ICML. [arXiv:2210.10760](https://arxiv.org/abs/2210.10760).
> 2. Kwa, T., Thomas, D. & Garriga-Alonso, A. (2024). *Catastrophic Goodhart: Regularizing RLHF with KL Divergence Does Not Mitigate Heavy-Tailed Reward Misspecification*. NeurIPS. [arXiv:2407.14503](https://arxiv.org/abs/2407.14503).
> 3. Mahan, D., et al. (2024). *Generative Reward Models*. [arXiv:2410.12832](https://arxiv.org/abs/2410.12832).
> 4. Wang, H., et al. (2024). *Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts*. EMNLP Findings. [arXiv:2406.12845](https://arxiv.org/abs/2406.12845).
> 5. DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. Nature. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948).
> 6. Xu, J., et al. (2023). *ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation*. NeurIPS. [arXiv:2304.05977](https://arxiv.org/abs/2304.05977).
> 7. *Flow-GRPO: Training Flow Matching Models via Online RL* (2025). [arXiv:2505.05470](https://arxiv.org/abs/2505.05470).
> 8. Lambert, N., et al. (2024). *RewardBench: Evaluating Reward Models for Language Modeling*. [arXiv:2403.13787](https://arxiv.org/abs/2403.13787).
> 9. Malik, S., et al. (2025). *RewardBench 2: Advancing Reward Model Evaluation*. [arXiv:2506.01937](https://arxiv.org/abs/2506.01937).
> 10. Weng, L. (2024). *Reward Hacking in Reinforcement Learning*. [Blog](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/).
> 11. Bai, Y., et al. (2022). *Constitutional AI: Harmlessness from AI Feedback*. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073).
> 12. Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685).

> ⬅️ 上一篇：[笔记｜强化学习（十一）：奖励模型基础——从传统 RL 到大模型与视觉生成](/chengYi-xun/posts/61-reward-model/)
