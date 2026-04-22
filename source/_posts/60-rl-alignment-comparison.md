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

# 回顾：从 REINFORCE 到 SuperFlow 的技术脉络

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

# 全方法大对比

## 核心机制对比

| 方法 | 需要 Reward Model? | 需要 Critic/Value? | 需要 Ref 模型? | On/Off-policy | 训练时模型数 | 核心优势估计 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **PPO** | RM | Critic $V_\phi$ | Ref $\pi_\text{ref}$（KL） | 近似 On | 4（Actor+Critic+RM+Ref） | GAE: $\sum_l(\gamma\lambda)^l\delta_{t+l}$ |
| **DPO** | 不需要（隐式） | 不需要 | Ref $\pi_\text{ref}$ | Off（固定数据） | 2（Policy+Ref） | 隐式：$\beta\log\frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)}$ |
| **GRPO** | 外部打分 | 不需要 | Ref $\pi_\text{ref}$（KL） | On | 2-3（Actor+Ref+外部RM） | 组内 z-score: $\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G}$ |
| **DAPO** | 外部打分 | 不需要 | **不需要** | On | 1-2（Actor+外部RM） | 组内 z-score + 非对称 clip |
| **2-GRPO** | 外部打分 | 不需要 | Ref | On | 2-3 | $G=2$ 的二元对比 |
| **f-GRPO** | 外部打分 | 不需要 | Ref | On | 2-3 | $f$-散度变分表示 |
| **GIFT** | 外部打分 | 不需要 | Ref | On | 2-3 | 组内标准化消 $Z(x)$，MSE 回归 |
| **Flow-GRPO** | 外部打分 | 不需要 | Ref | On | 2-3 | SDE logprob + 组内 z-score |
| **SuperFlow** | 外部打分 | 不需要 | Ref | On | 2-3 | 步级优势 $\hat{A}_t \propto \sigma_t A_\tau$ + 运行基线 |

## 优缺点对比

| 方法 | 核心优点 | 核心缺点 | 最佳场景 |
|:---|:---|:---|:---|
| **PPO** | 理论成熟、通用性强、探索能力好 | 4 个大模型显存爆炸、工程复杂 | 资源充足的通用 RLHF |
| **DPO** | 极简（仅 2 模型）、稳定、超参少 | 离线→探索弱、分布偏移、难超越标注 | 数据丰富、资源有限的偏好对齐 |
| **GRPO** | 去 Critic 省显存、保留在线探索 | 每 prompt $G$ 次生成计算量大、组内全同分无梯度 | 大模型在线对齐（DeepSeek-R1） |
| **DAPO** | 去 KL+Ref 进一步省显存、推理 RL 稳定 | 工程逻辑复杂、需可验证奖励 | 大规模数学/代码推理 RL |
| **2-GRPO** | $G=2$ 即可保留 ~98% 性能、墙钟快 5× | 连续 RM 下有虚幻优势风险 | RLVR（可验证奖励）高效训练 |
| **f-GRPO** | 散度选择灵活（Pearson $\chi^2$ 等可抑制过拟合） | 理论复杂、最优散度需调 | 对安全/稳定性有高要求的对齐 |
| **GIFT** | MSE 凸损失、训练最稳定、融合在线+隐式奖励 | 不兼容标准多 epoch 复用 | 追求训练稳定性的大模型对齐 |
| **Flow-GRPO** | 将 GRPO 扩展到 Flow Matching 图像生成 | 全步去噪采样贵 | Flux/SD3 级图像生成 RL |
| **SuperFlow** | 动态采样 + 步级信用分配，大幅省算力 | 系统复杂度高 | 图像生成 RL 的大规模训练 |

# 在 LLM 对齐中的三个关键分歧

## 分歧一：要不要 Critic？

PPO 的 Critic 在传统 RL 中至关重要（低偏差的优势估计），但在 LLM 对齐中成了负担——一个与 Actor 同等规模的价值网络会**让显存翻倍**。GRPO 的核心洞察是：LLM 对齐是单轮（没有多步 MDP），终端奖励就是全部回报，$V(s)$ 就等于 $\mathbb{E}[r]$，用组均值估计即可。这使得 Critic 从"必需品"变成了"奢侈品"。

## 分歧二：在线还是离线？

DPO 用固定偏好对训练，**无法涌现超越标注数据的新能力**——模型只能学会在已有好坏答案之间选择。在线方法（PPO/GRPO/DAPO）允许模型探索新的回答方式，在数学推理等任务上展现出"越训越聪明"的特性。但在线方法需要每轮生成 + 打分，计算开销远大于 DPO。GIFT 试图两全：在线采样 + DPO 式隐式奖励。

## 分歧三：Token 级还是序列级？

PPO 和 GRPO 的原始版本在 **token 级** 计算 ratio 和 clip，这对自回归 LLM 更自然（每个 token 都是一次"动作"）。但 token 级 ratio 在 MoE 架构中会导致不稳定（长序列的 token ratio 累积偏差）。GSPO 等新方法改为**序列级** ratio（整个回答一个 ratio），牺牲一些信号粒度换取更好的稳定性。DAPO 的 token-level 长度归一化是介于两者之间的折中。

# 选型决策树

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

# DPO 家族：20+ 变体一览

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

# 来自大规模受控实验的关键洞察

2026 年的 **oxRL** 研究（Li et al., arXiv:2603.19335）提供了迄今最严格的受控比较：51 种算法在完全相同的基础设施下训练，跨 4 个模型规模（0.5B–7B），总计约 240 次训练运行。其发现颠覆了许多"常识"：

## 洞察一：算法排名随模型规模反转

这是最令人惊讶的发现。在 1.5B 规模下，在线 RL（SGRPO）以 58.0% 排名第一，大幅领先 DPO（49.1%）和 SimPO（38.7%）。但到了 **7B 规模**，排名完全反转：**SimPO（85.8%）成为最佳**，DPO（83.9%）紧随其后，而 SFT（76.4%）几乎退化到基线。

| 算法 | 0.5B | 1.5B | 7B |
|:---|:---:|:---:|:---:|
| SFT | 34.0% | 54.4% | 76.4% |
| DPO | 34.0% | 49.1% | **83.9%** |
| SimPO | 26.1% | 38.7% | **85.8%** |
| SGRPO | 32.5% | **58.0%** | — |

**实践启示**：不要迷信"在线 RL 永远优于离线"——在足够大的模型上，SimPO 这种极简方法可能反而最好。模型规模才是最大的杠杆。

## 洞察二：20 种 DPO 变体无一显著优于原版 DPO

在 1.5B 规模下用 5 个随机种子训练 20 种 DPO 变体（共 100 次运行），经 Bonferroni 校正后：**没有一个变体显著优于原版 DPO**。唯一显著的结果是 SimPO 比 DPO **差** 11.5 个百分点。

这意味着社区投入大量精力优化的损失函数变体，在受控条件下可能只是噪声。正如 2018 年 Lucic 等人发现大多数 GAN 变体在受控条件下并不优于原版——历史似乎在重演。

## 洞察三：杠杆层级

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

**实践含义**：如果你的模型还没到足够大的规模，把预算花在扩大模型上的回报远大于调整算法。只有在模型规模固定后，才值得考虑在线 vs 离线的选择。损失函数的微调几乎不值得投入。

## 洞察四：算法优势不跨任务迁移

在 GSM8K（训练分布内）上 19.3 个百分点的算法差距，在 MATH（更难的数学）上压缩到 0.54 pp（36 倍压缩），在通用推理基准上更是只有 0.47 pp（41 倍压缩）。没有任何方法的优势能迁移到训练分布外的任务。

这与我们在世界模型系列中讨论的"泛化能力的错觉"一脉相承——模型在训练分布内表现完美，但分布外的泛化极其有限。

---

强化学习已从一个"理论优美但工程复杂"的技术，演变为大模型训练不可或缺的核心环节。每一步都在让 RL 变得更简单、更高效、更可规模化——而这个旅程，远未结束。

---

> 参考资料：
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

> 下一篇：[笔记｜强化学习（十一）：奖励模型——RLHF 的隐藏引擎](/chengYi-xun/posts/61-reward-model/)
