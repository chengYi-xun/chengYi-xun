---
title: 笔记｜强化学习（十）：LLM 对齐中的 RL 方法全景对比——从 PPO 到 SuperFlow
date: 2026-04-05 17:00:00
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

> 本文是强化学习系列的总结篇。经过前面九篇文章的旅程——从策略梯度基础、PPO、DPO、GRPO，到 DAPO、Flow-GRPO、SuperFlow 和 DanceGRPO——我们已经接触了十余种方法。本文将它们放在同一张桌上做横向对比，帮助读者在实际选型中快速定位最适合自己场景的算法。
>
> ⬅️ 上一篇：[笔记｜强化学习（九）：DanceGRPO——视频生成的统一强化学习框架](/chengYi-xun/posts/59-video-grpo/)
>
> ➡️ 下一篇：[笔记｜强化学习（十一）：奖励模型——RLHF 的隐藏引擎](/chengYi-xun/posts/61-reward-model/)

## 回顾：从 REINFORCE 到 SuperFlow 的技术脉络

| 篇章 | 主题 | 核心贡献 | 解决的问题 |
|:---:|:---|:---|:---|
| 一 | REINFORCE → Actor-Critic | 策略梯度基础 | 如何用梯度优化策略 |
| 二 | TRPO → PPO | 裁剪 surrogate 目标 | 策略更新步长控制 |
| 三 | DPO | 隐式奖励 + 偏好对 | 绕过奖励模型和 RL |
| 四 | GRPO | 组内相对优势 | 去除 Critic，适配大模型 |
| 五 | Flow-GRPO | 连续动作空间的 GRPO | 将 RL 引入图像生成 |
| 六 | DAPO | 4 项工程改进 | 大规模推理 RL 的稳定性 |
| 七 | 2-GRPO / f-GRPO / GIFT | GRPO 的三重理论视角 | 效率、散度选择、训练稳定性 |
| 八 | SuperFlow | 步级优势 + 动态采样 | 图像 RL 的信用分配和效率 |
| 九 | DanceGRPO | 多维度奖励 + 视频 RL | 视频生成的质量-运动平衡 |

**一条清晰的演进主线**：

$$
\underbrace{\text{REINFORCE}}_{\text{高方差}} \xrightarrow{\text{基线}} \underbrace{\text{Actor-Critic}}_{\text{需要 Critic}} \xrightarrow{\text{步长控制}} \underbrace{\text{PPO}}_{\text{显存翻倍}}
$$

$$
\xrightarrow{\text{去 Critic}} \underbrace{\text{GRPO}}_{\text{显存友好}} \xrightarrow{\text{工程优化}} \underbrace{\text{DAPO}}_{\text{大规模可用}} \xrightarrow{\text{连续动作}} \underbrace{\text{Flow-GRPO / SuperFlow}}_{\text{图像生成 RL}}
$$

$$
\xrightarrow[\text{离线旁支}]{\text{绕过 RL}} \underbrace{\text{DPO}}_{\text{稳定但有限}} \xrightarrow{\text{融合在线}} \underbrace{\text{GIFT}}_{\text{两全其美}}
$$

## 全方法对比：按家族理解差异

### PPO 家族：经典在线 RL

**PPO** 是 RLHF 的"标准答案"（InstructGPT 路线）。它的完整训练需要同时加载 4 个模型——Actor（策略）、Critic（价值网络）、RM（奖励模型）和 Ref（参考策略），通过 GAE（广义优势估计）提供低偏差的优势信号。

**优点**：理论成熟、探索能力强、通用性最好。
**缺点**：4 个大模型同时在显存里，工程复杂度极高。

### GRPO 家族：去 Critic，为大模型量身定制

**GRPO** 的核心洞察是：LLM 对齐通常是单步问答（bandit 设置），不需要 Critic 来估计多步回报——用同一 prompt 生成一组回答，取组内均值和标准差做 z-score 归一化就能得到足够好的优势估计。这一下省掉了与 Actor 同等规模的 Critic，显存压力骤降。

在此基础上，衍生出一系列变体，每个变体瞄准一个具体痛点：

- **DAPO**：进一步去掉 KL 惩罚和 Ref 模型，只剩 Actor + 打分器，将显存压到极致。配合非对称裁剪和动态采样，在 DeepSeek-R1 级别的大规模推理 RL 中表现出色。
- **2-GRPO**：将组大小压到 $G=2$（只生成两个回答做对比），保留约 98% 的性能但墙钟时间降至原来的 1/5。适合快速迭代的可验证奖励场景。
- **f-GRPO**：把 GRPO 的 KL 散度推广到一般 $f$-散度（如 Pearson $\chi^2$），提供更灵活的正则化选项来抑制过拟合。
- **GIFT**：用 MSE 凸损失替代裁剪目标，并融合在线采样与 DPO 式隐式奖励，训练稳定性最好。

### DPO 家族：离线偏好优化

**DPO** 走了一条完全不同的路：绕过 RM 和 RL，直接用偏好对数据（chosen / rejected）通过隐式奖励训练策略。只需 2 个模型（Policy + Ref），训练最简单、超参最少。

**核心限制**：离线训练意味着模型只能在已有的好坏答案之间做选择，无法探索全新的回答方式。在数学推理等需要"越训越聪明"的任务上，DPO 往往不如在线方法。

### 视觉生成 RL 家族

- **Flow-GRPO**：将 GRPO 扩展到 Flow Matching 文生图模型。通过 ODE→SDE 变换注入探索噪声，使每步转移变为可计算概率比的高斯分布。
- **SuperFlow**：在 Flow-GRPO 基础上引入动态采样（高方差 prompt 多采样、低方差少采样）和步级优势重估计（沿去噪轨迹逐步分配信用），大幅提升训练效率。
- **DanceGRPO**：面向视频生成，引入多维度奖励（视觉质量、运动合理性、文本对齐）的联合优化。

## 在 LLM 对齐中的三个关键分歧

### 分歧一：要不要 Critic？

PPO 的 Critic 在传统多步 RL 中至关重要（通过 GAE 提供低偏差的优势估计），但在 LLM 对齐中成了负担——一个与 Actor 同等规模的价值网络会**让显存翻倍**。GRPO 的核心洞察是：LLM 对齐通常是 bandit 设置（单步 MDP），终端奖励就是全部回报，$V(s)$ 退化为 $\mathbb{E}[r]$，用组内均值即可近似。这使得 Critic 从"必需品"变成了"奢侈品"。

### 分歧二：在线还是离线？

DPO 使用固定的偏好对数据训练，**无法涌现超越标注数据的新能力**——模型只能学会在已有好坏答案之间做选择，不能自行探索更优的回答方式。在线方法（PPO / GRPO / DAPO）则允许模型在训练过程中不断生成新回答并获得奖励反馈，在数学推理等任务上展现出"越训越聪明"的特性（Yang et al., 2025）。代价是每轮都需要生成 + 打分，计算开销远大于 DPO。GIFT 试图兼顾两端：用在线采样保留探索能力，同时用 DPO 式隐式奖励降低对显式 RM 的依赖。

### 分歧三：Token 级还是序列级？

PPO 和 GRPO 的原始版本在 **token 级** 计算重要性比率（importance ratio）和裁剪（clip），这对自回归 LLM 更自然——每个 token 都是一次"动作"。但 token 级比率在 MoE 架构中会导致不稳定：长序列中各 token 的比率噪声会累积，经裁剪放大后引发专家路由波动甚至训练崩溃。**GSPO**（Group Sequence Policy Optimization; Qwen 团队, 2025）改为**序列级**比率——对整个回答计算一个几何平均比率，牺牲部分信号粒度换取显著的训练稳定性。DAPO 的 token 级长度归一化则是介于两者之间的折中。

## 选型决策树

以下决策树覆盖了从"你手上有什么"到"你该用什么"的完整路径：

```
第一步：你在对齐什么模型？
│
├─ 大语言模型（LLM）→ 进入第二步
│
└─ 图像/视频生成模型
    ├─ Flow Matching 架构（Flux / SD3 等）
    │   ├─ 数据量大、追求高效 → SuperFlow（动态采样省 30-50% 算力）
    │   └─ 快速起步 → Flow-GRPO
    └─ 扩散模型架构（DDPM 系）→ DDPO（PPO 式）或改造 GRPO
```

```
第二步（LLM）：你有什么样的奖励信号？
│
├─ 只有偏好对数据（chosen/rejected），没有打分器
│   ├─ 数据量充足（>10K 对）→ DPO
│   │   典型场景：通用对话、安全对齐、风格控制
│   │   理由：无需 RM、无需采样、训练最简单
│   └─ 数据量有限 → 先用 SFT，再考虑其他方法
│
├─ 有可验证的规则奖励（数学题对错、代码通过率、格式检查）
│   ├─ 模型 ≥70B 或 MoE，显存极度紧张
│   │   └─ DAPO（去 KL + 去 Ref，只需 Actor + 打分器）
│   │       典型场景：DeepSeek-R1 级别的数学推理 RL
│   ├─ 追求极致训练速度，愿意牺牲少量性能
│   │   └─ 2-GRPO（G=2，墙钟时间降至 ~1/5）
│   │       典型场景：快速迭代、大量 prompt 的 RLVR
│   └─ 标准配置
│       └─ GRPO（G=8-64，组内 z-score）
│           典型场景：中等规模模型的在线对齐
│
├─ 有学习型 Reward Model（打分连续，0-1 或更细粒度）
│   ├─ 显存充足，可同时加载 4 个大模型
│   │   └─ PPO + GAE
│   │       典型场景：资源充足团队的通用 RLHF（InstructGPT 路线）
│   │       理由：Critic 精确估计优势，探索+利用最均衡
│   ├─ 显存有限（只够 2-3 个模型）
│   │   ├─ 追求训练稳定性（不想调 clip 等超参）
│   │   │   └─ GIFT（MSE 凸损失，融合在线探索+隐式奖励）
│   │   ├─ 担心 mode collapse，想精细控制正则化
│   │   │   └─ f-GRPO（选择 Pearson χ² 等散度抑制过拟合）
│   │   └─ 标准配置 → GRPO
│   └─ RM 质量不确定（可能有偏差）
│       └─ 建议先用 DPO 建基线，再切 GRPO/DAPO 在线微调
│
└─ 没有任何奖励信号
    └─ 先构建奖励信号（按实施成本从低到高）：
        1. 规则奖励（正则匹配、单元测试、数学验证）—— 零标注成本
        2. LLM-as-Judge（用强模型给弱模型打分）—— 无需人工，但有偏差
        3. 训练 Reward Model（从偏好数据学打分器）—— 需要偏好数据集
        4. 人工标注（人类直接评分或排序）—— 最准但最贵
```

## DPO 家族：20+ 变体一览

DPO 的成功催生了大量变体，它们在损失函数、正则化策略和数据需求上各有不同。下表列出主要变体及其核心改动：

| 方法 | 核心改动 | 是否需要 Ref 模型 | 关键特点 |
|:---|:---|:---:|:---|
| **DPO** | 基线：logistic 对比损失 | 是 | 标准偏好对优化 |
| **IPO** | 平方损失正则化 | 是 | 防止过拟合偏好数据 |
| **SimPO** | 长度归一化 + 去 Ref | 否 | 无需参考模型，在大模型上表现优异 |
| **KTO** | 非配对二元反馈 | 是 | 只需"好/坏"标签，不需要偏好对 |
| **ORPO** | SFT + 偏好联合训练 | 否 | 将 SFT 和对齐合并为一步 |
| **CPO** | 对比偏好优化 | 否 | 去 Ref，直接对比 |
| **RDPO** | 鲁棒散度惩罚 | 是 | 对标注噪声更鲁棒 |
| **BetaDPO** | 自适应 $\beta$ 调度 | 是 | 训练过程中动态调整偏好强度 |
| **SPPO** | 自博弈偏好优化 | 是 | 模型与自身博弈生成偏好数据 |
| **GPO** | 广义偏好优化 | 是 | 统一框架，可切换不同散度 |

## 来自大规模受控实验的关键洞察

2026 年的 **oxRL** 研究（Li et al., arXiv:2603.19335）提供了迄今最严格的受控比较：51 种算法在完全相同的基础设施下训练，跨 4 个模型规模（0.5B–7B），总计约 240 次训练运行。其发现颠覆了许多"常识"：

### 洞察一：算法排名随模型规模反转

这是最令人惊讶的发现。在 1.5B 规模下，在线 RL（SGRPO）以 58.0% 排名第一，大幅领先 DPO（49.1%）和 SimPO（38.7%）。但到了 **7B 规模**，排名完全反转：**SimPO（85.8%）成为最佳**，DPO（83.9%）紧随其后，而 SFT（76.4%）几乎退化到基线。

| 算法 | 0.5B | 1.5B | 7B |
|:---|:---:|:---:|:---:|
| SFT | 34.0% | 54.4% | 76.4% |
| DPO | 34.0% | 49.1% | **83.9%** |
| SimPO | 26.1% | 38.7% | **85.8%** |
| SGRPO | 32.5% | **58.0%** | — |

**实践启示**：不要迷信"在线 RL 永远优于离线"——在足够大的模型上，SimPO 这种极简方法可能反而最好。模型规模才是最大的杠杆。

### 洞察二：20 种 DPO 变体无一显著优于原版 DPO

在 1.5B 规模下用 5 个随机种子训练 20 种 DPO 变体（共 100 次运行），经 Bonferroni 校正后：**没有一个变体显著优于原版 DPO**。唯一显著的结果是 SimPO 比 DPO **差** 11.5 个百分点。

这意味着：以往各家论文声称自己的变体"优于 DPO"，很可能只是因为训练代码、超参数、数据预处理等实验条件不同导致的随机波动——一旦把所有变量控制住（相同的代码库、相同的数据、多个随机种子取平均），这些差异就消失了。社区投入大量精力设计的损失函数改进，在严格对照下并未带来真实的性能提升。

这一现象并非首次出现。Lucic et al.（NeurIPS 2018）曾对当时流行的 GAN 变体做过类似的受控实验，结论几乎相同：在公平对比下，大多数 GAN 变体并不优于原版。历史似乎在重演。

### 洞察三：杠杆层级

oxRL 给出了一个清晰的**杠杆层级**，帮助从业者确定优化方向：

$$
\underbrace{\text{模型规模}}_{\sim 50\text{pp}} \gg \underbrace{\text{训练范式（SFT vs RL）}}_{\sim 10\text{pp}} \gg \underbrace{\text{在线 vs 离线}}_{\sim 9\text{pp}} \gg \underbrace{\text{损失函数选择}}_{\sim 1\text{pp}}
$$

| 杠杆 | 影响量 | 例子 |
|:---|:---:|:---|
| 模型规模 | ~50 pp | 1.5B → 7B：DPO 从 49% 到 84% |
| 训练范式 | ~10 pp | SFT vs 后训练方法 |
| 在线 vs 离线 | ~9 pp | SGRPO vs DPO（1.5B 规模） |
| 损失函数 | ~1 pp | DPO vs IPO vs KTO（同规模） |

**注意**：这里的"训练范式（SFT vs RL）"并非"二选一"，而是指"**只做 SFT 就停** vs **SFT 之后再做后训练（RL 或 DPO）**"。当前的主流实践是**先 SFT 冷启动，再 RL 微调**——SFT 教会模型指令格式和基本推理逻辑，RL 在此基础上进一步激发复杂推理能力。DeepSeek-R1-Zero 证明了跳过 SFT 直接做大规模 RL 也能涌现推理能力（如自我反思、回溯验证），但代价是训练不稳定、输出格式粗糙、对模型规模要求极高。对于绝大多数场景，**SFT 仍是不可或缺的冷启动基石**。

**实践含义**：如果你的模型还没到足够大的规模，把预算花在扩大模型上的回报远大于调整算法。只有在模型规模固定后，才值得考虑在线 vs 离线的选择。损失函数的微调几乎不值得投入。

### 洞察四：算法优势不跨任务迁移

在 GSM8K（训练分布内）上 19.3 个百分点的算法差距，到 MATH（更难的数学推理）上压缩至 0.54 pp（36 倍衰减），到通用推理基准上更是只有 0.47 pp（41 倍衰减）。换言之，算法 A 在训练任务上大幅领先算法 B，并不意味着它在分布外的新任务上也更好。

这一发现与 Scaling Laws 领域的常见观察一致——分布内的性能差异往往不能外推到分布外，对齐算法亦不例外。

强化学习已从一个"理论优美但工程复杂"的技术，演变为大模型训练不可或缺的核心环节。每一步都在让 RL 变得更简单、更高效、更可规模化——而这个旅程，远未结束。

> 参考文献：
>
> 1. Zou, H. (2025). *RLHF Algorithms: PPO, GRPO, GSPO — Differences, Trade-offs, and Use Cases*. [Medium](https://medium.com/@hongjianzou/rlhf-algorithms-ppo-grpo-gspo-differences-trade-offs-and-use-cases-241d003d806d)
> 2. Sun, Z. et al. (2025). *A Comprehensive Survey of Reward Models: Taxonomy, Applications, Challenges, and Future*. [arXiv:2504.12328](https://arxiv.org/abs/2504.12328)
> 3. Wolfe, C. R. (2025). *Reward Models*. [Substack](https://cameronrwolfe.substack.com/p/reward-models)
> 4. Yang, S. et al. (2025). *Evaluating GRPO and DPO for Faithful Chain-of-Thought Reasoning in LLMs*. [arXiv:2512.22631](https://arxiv.org/abs/2512.22631)
> 5. Xu, H. et al. (2025). *Unifying PPO, DPO, and GRPO: A Theoretical and Empirical Study on LLM Post-Training*. [ResearchGate](https://www.researchgate.net/publication/397211705)
> 6. Li, X. et al. (2026). *Do Post-Training Algorithms Actually Differ? A Controlled Study Across Model Scales Uncovers Scale-Dependent Ranking Inversions*. [arXiv:2603.19335](https://arxiv.org/abs/2603.19335)
> 7. Wang, Z. et al. (2024). *A Comprehensive Survey of LLM Alignment Techniques: RLHF, RLAIF, PPO, DPO and More*. [arXiv:2407.16216](https://arxiv.org/abs/2407.16216)
> 8. Meng, Y. et al. (2024). *SimPO: Simple Preference Optimization with a Reference-Free Reward*. NeurIPS 2024.
> 9. Ethayarajh, K. et al. (2024). *KTO: Model Alignment as Prospect Theoretic Optimization*. ICML 2024.
> 10. Hong, J. et al. (2024). *ORPO: Monolithic Preference Optimization without Reference Model*. EMNLP 2024.
> 11. Qwen Team (2025). *Group Sequence Policy Optimization (GSPO)*. [arXiv:2507.18071](https://arxiv.org/abs/2507.18071).
> 12. Lucic, M. et al. (2018). *Are GANs Created Equal? A Large-Scale Study*. NeurIPS 2018. [arXiv:1711.10337](https://arxiv.org/abs/1711.10337).

> 下一篇：[笔记｜强化学习（十一）：奖励模型——RLHF 的隐藏引擎](/chengYi-xun/posts/61-reward-model/)
