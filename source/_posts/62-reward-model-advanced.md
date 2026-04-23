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

1. **大模型直接当评委（LLM-as-a-Judge）**：写一段 Prompt 让 GPT-4 直接输出"A 更好"或"B 更好"。最简单灵活，但依赖闭源 API，且存在系统性偏差（Zheng et al., 2023）。

2. **AI 反馈强化学习（RLAIF）**：让 AI 批量生成偏好标注数据，再用这些数据蒸馏出一个轻量的 BT-RM。Anthropic 的 Constitutional AI（Bai et al., 2022）是先驱。局限在于蒸馏过程中信息有损。

3. **生成式奖励模型（GenRM）**：把评委能力直接微调进开源 LLM，让它像人类审稿人一样先推理再判决（Mahan et al., 2024）。效果好但推理成本较高（需要生成 CoT，即 Chain-of-Thought 思维链——让模型在给出判决前先写出一段推理过程）。

三者并非互斥：RLAIF 可用大模型评委的输出做标注源，GenRM 可在 RLAIF 数据上训练。

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

**奖励信号如何传递给策略模型？** GenRM 的推理链 $c$ 是评委内部的"思考过程"，**策略模型看不到它**。在 RL 训练循环中，流程是：策略模型生成回答 $y$ → GenRM 内部先生成推理链 $c$，再输出判决 $I$（或从 $\pi_\phi(I \mid x, y_1, y_2, c)$ 中提取一个标量概率作为分数）→ 策略模型只收到这个**最终的标量奖励信号**，用于更新梯度。CoT 的作用不是展示给策略模型看，而是提升评委自身的判断质量——就像论文审稿人写详细审稿意见是为了让自己的判断更准确，但作者拿到的只是"接受/拒绝"的最终决定。

后续工作进一步发展了 GenRM 范式：**Think-RM**（NeurIPS 2025）引入长程推理链训练；**Rationale Consistency** 指标（2026）用于检测推理链是否为事后合理化（post-hoc rationalization），而非因果性推理。

### 2.3 AI 评委的系统性偏差

无论采用何种 AI 评委，以下三类偏差均需关注（Zheng et al., 2023; Weng, 2024）：

- **位置偏差（Position Bias）**：倾向偏好先出现的回答。缓解：交换 A/B 顺序做两次评判，不一致则判平。
- **冗长偏差（Verbosity Bias）**：将冗长误判为详尽。缓解：Prompt 约束或在训练数据中加入"短精 > 长空"样本。
- **自我偏好（Self-preference Bias）**：对同家族模型的输出给出更高评分（低困惑度效应）。缓解：跨家族交叉评判。

## 3 绕过单一标量：多目标 RM 与可验证奖励

### 3.1 ArmoRM：把黑盒标量拆成多个可解释维度

传统 BT-RM 把所有偏好压进一个标量——当策略 Hacking 时，你只知道"分数降了"，但分不清是"太啰嗦"还是"太冒险"。

Wang et al.（EMNLP Findings 2024）提出 **ArmoRM**，核心思路很简单：**先让模型像各科老师一样分别给"诚实"、"安全"、"简洁"等维度打出独立分数，然后根据当前问题的类型，动态决定各维度的权重。** 具体地，第一阶段训练一个多目标回归头，输出 $k$ 维评分向量；第二阶段用一个门控网络（MoE 结构）读取 prompt 特征来决定权重，加权求和得到最终标量奖励。

这样做的好处是：当模型出问题时，你可以直接看是哪个维度崩了（比如"安全分很高但简洁分暴跌"），而不是对着一个黑盒数字猜原因。ArmoRM-Llama3-8B-v0.1 在 RewardBench 上取得了同规模 SOTA（Wang et al., 2024）。

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

文本生成的输出是一串离散的 token，有些任务甚至有客观对错（数学题、代码题）。但图像和视频的输出是**高维的连续像素信号**——"好不好看"、"是否忠实于描述"往往高度主观，也没有类似数学题那样的 0/1 正确答案。因此，视觉生成侧的奖励模型面临独特的挑战。

### 4.1 ImageReward：训一个同时懂"文"和"图"的裁判

早期的文生图评估主要依赖 **CLIP Score**（衡量图文匹配度）和**美学分数**（衡量画面好看程度），但这两个指标各管一头，经常冲突——比如 CLIP Score 极高但画面构图一塌糊涂，或者美学分极高但完全跑题。

Xu et al.（*ImageReward*, NeurIPS 2023）的思路是：**不再拆成两个独立分数，而是用大规模人类偏好数据（约 13.7 万对专家标注）训练一个跨模态模型，让它像人类一样综合考虑"图片是否符合描述"、"是否好看"、"是否有缺陷"。** 架构上基于 BLIP 跨模态编码器 + MLP 打分头，训练同样使用 BT 配对损失。实验表明，它比单独的 CLIP Score 或美学模型更贴近人类的真实排序。

### 4.2 Flow-GRPO：让文生图模型也能做强化学习

有了 ImageReward 这样的裁判之后，下一步自然是：能不能像文本的 RLHF 一样，用 RL 来优化图像生成模型？

问题在于，当前主流的文生图模型（如基于 Flow Matching 的模型）的生成过程是**确定性的**——给定一个初始噪声，生成的图片就注定了。而强化学习依赖"试错"：模型需要能探索不同的生成可能性，才能发现哪些方向能拿高分。

**Flow-GRPO**（2025）解决了这个问题：它在数学上将确定性的生成过程改写为一个等价的**随机过程**，在不改变最终生成效果的前提下注入了随机性，让模型有了"探索空间"。同时用 Denoising Reduction 技术压缩计算量，使在线 RL 的开销变得可接受。

与文本 RLHF 相比，视觉 RL 有三个关键差异：

1. **动作空间不同**：文本模型每步选一个 token（离散的），图像模型每步更新的是连续的像素向量。
2. **奖励几乎全是多目标的**：需要同时兼顾对齐、美感、构图、时序一致性等多个维度，很少有单一的绝对对错。
3. **Hacking 形态不同**：比如"CLIP 分极高但画面崩坏"、"美学分极高但语义跑题"，缓解方式通常需要多奖励帕累托优化。

## 5 评测与选型

### 5.1 RewardBench：给奖励模型做"统考"

在 2024 年之前，各家论文都用自己的私有数据评测 RM，没有统一标准。Lambert et al.（2024）构建了第一个公开的 RM 评测基准 **RewardBench**。

它的测试方式很直观：给 RM 一道题（prompt）、一个好回答和一个坏回答，看 RM 能否正确识别出哪个更好。覆盖四个场景：日常对话、困难对话、安全拒绝、逻辑推理。核心指标是**胜率**——RM 判断正确的比例。

**RewardBench 2**（Malik et al., 2025）进一步提升了难度（各家模型的平均分下降约 20 分），并且与实际 RLHF 训练效果的相关性更高——在 RewardBench 2 上得分高的 RM，用它训练出来的模型也确实更好。

### 5.2 选型指南

面对这么多 RM 方案，实际工程中怎么选？以下是一个简单的决策流程：

1. **任务有客观对错吗？**（数学题、代码题、格式检查）→ 优先用**规则奖励**（DeepSeek-R1 路线），零 Hacking 风险，零训练成本。
2. **任务是开放式的？**（对话、创意写作）→ 用**传统 BT-RM、GenRM 或大模型评委**。
3. **需要知道"坏在哪里"？**（比如区分"太啰嗦"和"不安全"）→ 用**多目标 RM**（ArmoRM），可以看到各维度分数。
4. **训练数据少、部署场景多变？** → 用带 CoT 推理的 **GenRM**，分布外泛化能力更强；或建立持续标注刷新机制。
5. **推理成本敏感？** → 成本从低到高：标量 BT-RM（一次前向传播）< GenRM（需要生成推理链）< GPT-4 评委（每次都要调 API）。

## 6 结语

本文沿"**发现问题 → 量化问题 → 解决问题**"的主线展开：

1. **问题**：Reward Hacking 是 RLHF 的核心风险——策略模型会钻奖励模型的漏洞，代理分数飙升但真实质量崩塌。
2. **量化**：Gao et al. 的缩放定律给出了那条标志性的倒 U 型曲线；Kwa et al. 证明了 KL 惩罚在重尾误差下理论上会失效（灾难性古德哈特）。
3. **造更好的裁判**：GenRM 让评委先推理再打分，大幅提升了面对新场景时的判断准确率；ArmoRM 把黑盒标量拆成多个可解释维度，让问题诊断成为可能。
4. **绕过裁判**：DeepSeek-R1 在数学 / 代码等有客观答案的任务上，直接用规则奖励替代神经网络裁判，从根源消除了 Hacking 空间。
5. **视觉侧**：ImageReward 和 Flow-GRPO 把"训裁判 + 考选手"的范式搬到了图像和视频生成领域。

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
