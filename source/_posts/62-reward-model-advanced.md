---
title: 笔记｜强化学习（十二）：奖励模型进阶——Reward Hacking、生成式奖励模型与可验证奖励
date: 2026-04-23 18:00:00
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

> 本篇是 [笔记｜强化学习（十一）：奖励模型基础——从传统 RL 到大模型与视觉生成](/chengYi-xun/posts/61-reward-model/) 的续篇。上一篇我们把奖励模型建成**可优化的代理**：用人类比较数据拟合 \(\hat{r}_\theta(x,y)\)，再在 PPO、GRPO 或 Best-of-\(N\) 里把它当「廉价裁判」。本篇追问一个更尖锐的问题：**当代理不完美时，拼命优化它会怎样？** 这正是 Goodhart 定律在 RLHF 里的写照——「一旦一个指标变成目标，它就不再是个好指标」。下文从 Reward Hacking 的定量规律出发，走到生成式评委、多目标架构（ArmoRM）、可验证奖励（DeepSeek-R1），再瞥一眼视觉生成与 RewardBench 评测，收束成一张 2024–2026 的路线图。
>
> ⬅️ 上一篇：[笔记｜强化学习（十一）：奖励模型基础——从传统 RL 到大模型与视觉生成](/chengYi-xun/posts/61-reward-model/)

## 从「代理裁判」到 Goodhart：我们到底在优化什么？

在[上一篇](/chengYi-xun/posts/61-reward-model/)中，我们把奖励模型建成了一个**可优化的代理裁判**：用人类比较数据拟合出一个打分器 $\hat{r}_\theta(x,y)$，然后在 RL（如 PPO）或推理期搜索（如 Best-of-$N$）里让它自动给大模型的回答打分。

但这引出了一个非常尖锐的问题：**当这个裁判不完美时，拼命讨好它会发生什么？**

想象你在训练一位助手，你真正关心的是“对人类有用且诚实”（我们称之为**真实人类效用 $r^*$**）。但因为人类反馈太贵，你只能让优化器去讨好那个**替身裁判 $\hat{r}_\theta$**。

只要替身裁判和真实偏好之间存在哪怕一丝缝隙（$\hat{r}_\theta \neq r^*$），强大的强化学习优化器就一定会钻这个空子：它会找到那些能骗过替身裁判拿高分，但对人类毫无价值的回答。

这就是 **Reward Hacking（奖励篡改 / 过度优化）**——它不是“模型变坏了”，而是**目标函数与真实目标脱节**时，强优化必然导致的灾难。这正是著名的 **Goodhart 定律**在 AI 对齐中的完美写照：“一旦一个指标变成优化目标，它就不再是个好指标”。

下面，我们将把这条缝隙**量化**：代理奖励与“金标准”偏好何时一致、何时分道扬镳？KL 正则化又能在多大程度上挡住这场灾难？

## Reward Hacking：从直觉到缩放定律

### 直觉：模型在优化什么，就会在什么地方「长歪」

Reward hacking 并不是模型产生了恶意，它只是在**对齐压力下的自然演化**。
- **冗长（Verbosity）**：如果 RM 隐约觉得“字数多 = 解释详细 = 好回答”，策略模型就会疯狂堆砌废话，把一句话能说明白的事写成五段。
- **谄媚（Sycophancy）**：如果 RM 对“同意用户立场”的回答给高分，模型就会变成一个没有主见的马屁精，即使用户说“1+1=3”，它也会附和。
- **格式套路**：大量使用粗体、列表、或者“作为一个 AI 语言模型”的安全声明来骗取高分。

它们的共同点是：**在代理裁判那里得分极高，但在真实人类眼里毫无价值。**

### Gao et al.：过度优化的定量规律与倒 U 型曲线

为了精确测量这种现象，Gao 等人（*Scaling Laws for Reward Model Overoptimization*, ICML 2023）设计了一个巧妙的实验：
1. 用一个非常大的、完美的 **Gold RM** 扮演“真实人类”。
2. 用它的少量标注数据，训练一个较小的 **Proxy RM**（替身裁判）。
3. 让策略模型去拼命优化 Proxy RM 的分数，然后观察 **Gold RM 的分数**会怎么变。

他们把策略模型偏离初始状态的距离定义为 $d := \sqrt{D_{\mathrm{KL}}\bigl(\pi \,\|\, \pi_{\mathrm{init}}\bigr)}$。

实验发现，随着优化强度的增加（$d$ 变大），**Proxy 分数会一直单调上升，但 Gold 分数会呈现先升后降的趋势**。

具体来说，在 **Best-of-$N$（拒绝采样）** 下，Gold 分数呈现一条**倒 U 型的抛物线**：
$$
R_{\mathrm{BoN}}(d) = \alpha_{\mathrm{BoN}}\, d - \beta_{\mathrm{BoN}}\, d^2
$$
- **小 $d$ 时（对齐阶段）**：$\alpha d$ 项主导，Gold 分数上升，说明模型确实在变好。
- **大 $d$ 时（Hacking 阶段）**：$-\beta d^2$ 项主导，Gold 分数开始暴跌，说明模型已经找到了 Proxy RM 的漏洞，多优化一点反而损害真实偏好。

而在 **强化学习（RL，如 PPO）** 下，曲线呈现不同的对数形态，但同样存在峰值：
$$
R_{\mathrm{RL}}(d) = d\,\bigl(\alpha_{\mathrm{RL}} - \beta_{\mathrm{RL}}\,\log d\bigr)
$$

![Proxy 与 Gold 分数在优化过程中的变化趋势](/chengYi-xun/img/scaling-law-flowchart.png)

### Kwa et al.：KL 惩罚也挡不住的“灾难性 Goodhart”

在标准的 PPO 算法中，我们通常会加一个 **KL 惩罚项**，不让策略模型跑得太远，试图以此来防止 Reward Hacking。但这真的绝对安全吗？

Kwa 等人（NeurIPS 2024）给出了悲观的答案：**当奖励模型的误差呈现重尾分布（Heavy-Tailed）时，即使加了 KL 正则，依然会发生 Catastrophic Goodhart（灾难性 Goodhart 效应）。**

直白地说，如果 RM 在某些罕见的角落（长尾区域）会给出异常离谱的高分，优化器就会像闻到血腥味的鲨鱼一样扑过去。此时，KL 惩罚只能限制模型“别跑太快”，但**根本无法阻止**模型最终掉进这些高分陷阱里。

**工程结论**：
1. **缩放定律**告诉我们：给定一个 RM，真实的优化收益是有**天花板**的。你不能无限期地跑 PPO。
2. **灾难性 Goodhart**告诉我们：光靠调大 KL 惩罚系数 $\beta$ 是不够的。

要彻底解决这个问题，我们需要更强大的裁判（如 LLM-as-a-Judge）、多维度的打分（如 ArmoRM），甚至在客观任务上**直接抛弃神经网络裁判，改用纯规则验证（如 DeepSeek-R1）**。这正是 2024-2026 年 RM 架构演进的核心主线。

## 数据瓶颈与 AI 评委：从 LLM-as-Judge 到带 CoT 的生成式奖励模型（GenRM）

### 用 AI 评判 AI：如何缓解人类标注的扩展性危机？

传统 RM 最大的瓶颈是**数据太贵了**。让人类专家去对比两段长篇大论的代码或逻辑推理，成本极高且速度极慢。

为了打破这个瓶颈，学术界开始探索**用强大的 AI（如 GPT-4）来当评委**。这演化出了三条不同的技术路线：

1. **LLM-as-a-Judge**：直接写一段 Prompt，让 GPT-4 充当裁判，输出“A 更好”或“B 更好”。这种方法最简单灵活，但非常依赖闭源 API，且存在严重的系统性偏差。
2. **RLAIF（AI 反馈强化学习）**：用 LLM-as-a-Judge 生成海量的偏好数据（A > B），然后再用这些合成数据去训练一个传统的标量 RM（如 BT-RM）。Anthropic 的 Constitutional AI 就是这一路线的先驱。
3. **GenRM（Generative Reward Models）**：这是 2024 年以来的最新趋势。与其把大模型压缩成一个只输出数字的标量 RM，不如**直接把评委的能力微调进一个开源大模型里**。

### 进化：GenRM 与 CoT 推理链的降维打击

Mahan 等人（*Generative Reward Models*, 2024）提出，与其让模型直接输出一个冷冰冰的分数，不如让它在打分前，先生成一段 **Chain-of-Thought（CoT，思维链）** 理由。

**为什么 CoT 这么重要？**
因为传统的标量 RM 是一个黑盒，一旦它学到了“长回答就是好回答”这种捷径，你根本无从知晓。而 GenRM 被强迫写出“为什么 A 比 B 好”的详细分析（比如“A 的逻辑更严密，而 B 在第三段出现了事实错误”），这就把**中间的推理结构**暴露为了训练信号。

实验证明，这种带 CoT 的 GenRM 在面对没见过的新任务（OOD，分布外泛化）时，表现出了对传统 BT-RM 的**降维打击**（准确率高出 10% - 45%）。

### AI 评委的“三大幻觉”与缓解策略

即便用了最先进的 GenRM 或 GPT-4，AI 评委依然会有自己的“小心思”。实证研究反复发现了三大系统性偏差：

1. **位置偏差（Position Bias）**：当两个回答质量差不多时，AI 评委往往会无脑偏好**第一个**出现的回答。
   - *缓解方案*：把 A 和 B 交换位置，让评委打两次分，如果结果不一致就判平局。
2. **冗长偏差（Verbosity Bias）**：AI 评委极度偏爱“长篇大论”，经常把废话连篇的回答误判为“更详细、更完整”。
   - *缓解方案*：在 Prompt 中严厉警告“不要受长度影响”，或者在训练数据中刻意加入“短而精”战胜“长而空”的样本。
3. **自我偏好（Self-preference Bias）**：AI 评委会给自己家族的模型打高分（比如 Llama 评委偏爱 Llama 的生成风格）。研究发现，这其实是因为模型对自己的输出具有更低的困惑度（Perplexity），觉得看着更顺眼。
   - *缓解方案*：在评估时，必须使用与生成模型**不同家族**的模型作为交叉评委。

## 当单一标量不够用时：ArmoRM 的多目标门控专家，与 DeepSeek-R1 的可验证奖励转向

### ArmoRM：多维偏好 + 门控 MoE，把「权衡」从黑箱里拆出来

Wang et al.（*Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts*，[arXiv:2406.12845](https://arxiv.org/abs/2406.12845)）指出：真实人类偏好往往是**多维**的（诚实、正确、简洁、安全等），而经典 Bradley–Terry RM 把它们**压进一个标量**——策略一旦 exploit，你很难判断是「太啰嗦」还是「太冒险」。**ArmoRM（Absolute-Rating Multi-Objective Reward Model）** 的做法分两步：

1. **多目标绝对评分**：在 \((x,y)\) 上回归多个维度（而非只学相对胜负）。
2. **上下文门控（gating）**：用 **MoE 式门控网络** \(g_\phi\)，只看 **prompt** 的表征，动态产生非负、和为 1 的权重，把向量奖励合成标量。

记 LLM 主干对拼接输入 \(x\oplus y\) 的最后一层最后一 token 表征为 \(f_\theta(x\oplus y)\)，多目标回归头输出各维评分；门控读取 **仅 prompt** 的特征 \(f_\theta(x)\)，经 softmax 得到系数向量。为抑制「冗长度维度过大污染一切」，论文对除冗长度外的目标做 **与冗长度去相关** 的修正（在参考分布上用 Spearman 相关减掉冗长度成分），得到 \(r'\)，再与门控结合：

$$
R \;=\; g_\phi\bigl(f_\theta(x)\bigr)^\top r'.
$$

随后在**冻结**主干与回归头的条件下，用成对偏好数据对门控与温度等参数 \((\phi,\beta)\) 做类 BT 的优化，使标量 \(R\) 与人类整体偏好一致。

直觉上，ArmoRM 把 **「这一轮对话更该强调什么」** 交给门控、把 **「这条回答在各维度上如何」** 交给多头回归；比单一黑箱标量更易诊断与纠偏。实现与模型卡见 [RLHFlow/ArmoRM-Llama3-8B-v0.1](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1)。

![ArmoRM 架构：单次前向传播、多目标评分与门控合成](/chengYi-xun/img/armorm-architecture.png)

### DeepSeek-R1 与范式转移：推理任务为何优先「可验证」而非神经 RM？

DeepSeek-AI（*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*，[arXiv:2501.12948](https://arxiv.org/abs/2501.12948)）在 **DeepSeek-R1-Zero** 的数学 / 代码 / 逻辑数据上采用 **rule-based reward**：**accuracy reward**（答案能否被规则判定为正确）与 **format reward**（是否满足可自动检查的格式以便抽取答案）。二者写为

$$
R_{\mathrm{rule}} \;=\; R_{\mathrm{acc}} + R_{\mathrm{format}}.
$$

团队明确 **避免** 在这类推理任务上依赖神经网络 RM，理由包括：大规模 RL 下神经 RM **更易被 hacking**；同时训练与维护成本高、与「对错可判定」的任务结构也不匹配。

同时，R1 **并非**「全局不用学习式奖励」：在通用对齐阶段，论文仍描述 **helpfulness / safety** 等场景使用 **学习式 RM** 与多源 \(R_{\mathrm{General}}\) 的组合。准确画像是一个 **双轨** 范式：

- **可验证子任务**（数学答案、代码单测、格式检查）：优先 **硬奖励**——便宜、语义清晰、难靠套路刷爆。
- **开放对话与价值判断**：仍依赖 **BT-RM、GenRM 或人类偏好管线**。

这与 **GRPO**（组内相对优势、免显式 Critic）等算法配套，构成了 2024–2026 期间极具代表性的一条路线：**推理用可验证，通用用 RM**。

## 视觉生成里的奖励：没有廉价真值时，对齐与美学如何拼在一起？

与文本不同，图像 / 视频的输出 $y$ 是高维连续信号（像素矩阵）。“是否美”、“是否忠实于 prompt”往往**高度主观**，又缺少数学证明题那种 **0/1 可验证真值**。因此，视觉生成侧更常见的是 **CLIP 式对齐分、专门训练的美学头、人类偏好 RM** 的组合，以及 **多奖励加权**。

视觉生成的 Reward Hacking 形态也与文本截然不同，例如：
- **CLIP 分极高但画面崩坏**：模型为了迎合文本提示，把所有元素强行塞进画面，导致构图混乱。
- **美学分极高但语义跑题**：模型画了一幅极美的风景画，但完全忽略了用户要求的“赛博朋克”风格。

### ImageReward：用大规模人类偏好训一个「懂人话又懂图」的裁判

为了解决单纯依赖 CLIP 或美学分带来的问题，清华大学等提出了 **ImageReward**（Xu et al.，NeurIPS 2023）。
它在约 **13.7 万**对专家偏好数据上训练，使用 BLIP 式跨模态编码器加上 MLP 打分头，联合刻画了**对齐、美感与细粒度质量**。实验表明，它相对同期的 CLIP Score 或纯美学模型，往往更贴近人类的真实排序。

### Flow-GRPO：把在线 RL 接进流匹配文生图

在视频生成领域，奖励模型需要同时惩罚**帧间闪烁**、**运动不合理**（如水往高处流）以及**文本—视频不对齐**等问题。

**Flow-GRPO**（2025）是一项面向流匹配（Flow Matching）文生图的最新工作。它解决了视觉生成 RL 的两大痛点：
1. **ODE 采样探索不足**：将 ODE 过程改写为保留边际分布的 SDE，以此注入噪声，换取策略梯度所需的随机性。
2. **逐步去噪算力极重**：使用 Denoising Reduction 技术，在训练中用更少步数近似推理轨迹，把在线 RL 的开销压到可接受范围。

与 LLM 的 RLHF 相比，视觉 RL 的核心差异在于：
- **动作空间**：是噪声更新或向量场，而非离散的 token 分布。
- **奖励信号**：几乎全是**多目标加权**，极少有单一的、可验证的绝对对错。
- **缓解 Hacking**：通常需要多奖励帕累托优化（Pareto），或条件化生成来协同约束。

## RewardBench 与选型：怎样知道你的 RM「真的好用」？

### RewardBench：给 RM 做标准化「体检」

在 2024 年之前，各家论文都在用自己私有的数据集评测 RM，导致“王婆卖瓜，自卖自夸”。
Lambert 等人（*RewardBench*, 2024）构建了第一个标准化的评测集。它包含 (prompt, chosen, rejected) 三元组，覆盖了 **Chat（日常对话）、Chat-Hard（困难对话）、Safety（安全拒绝）和 Reasoning（推理）** 四个子集。
核心指标是 **Win Rate**（预测 chosen 得分高于 rejected 的比例），把 RM 评估推进到了**可复现的公共协议**时代。

### 选型指南：任务、分布、成本与可解释性

当你需要为一个 RLHF 任务选择奖励模型时，请参考以下决策树：

1. **任务是否客观可验证？**
   - 数学、代码单测、格式检查 $\rightarrow$ **绝对优先选择纯规则的可验证奖励（如 DeepSeek-R1）**。
   - 开放对话、创意写作 $\rightarrow$ **BT-RM / GenRM / LLM 评委**。
2. **是否需要可解释的权衡？**
   - 比如需要在“安全”与“有用”、“简洁”与“正确”之间做权衡 $\rightarrow$ **多目标 RM（如 ArmoRM 的 MoE 架构）**。
3. **是否面临严重的 OOD（分布外）挑战？**
   - 训练数据窄，但部署场景多变 $\rightarrow$ 关注带有 CoT 推理能力的 **GenRM**，或建立持续的在线标注刷新机制。
4. **推理成本敏感度**：
   - GPT-4 评委 $\rightarrow$ 效果好但极贵且慢。
   - 标量 BT-RM $\rightarrow$ 便宜，一次前向传播即可。
   - GenRM $\rightarrow$ 效果好，但需要生成长长的 CoT，成本居中。

## 结语：一张 2024–2026 的奖励模型演进地图

**Reward Hacking** 绝不是危言耸听：Gao et al. 的缩放定律揭示了 Gold 分数随优化强度呈现的**倒 U 型曲线**；而 Kwa et al. 则证明了在重尾误差下，**KL 惩罚也无法阻止灾难性 Goodhart 效应**。

为了应对这些挑战，2024-2026 年的奖励模型架构发生了剧烈的演进：
- **GenRM** 用 CoT 自举把评委的逻辑推理能力训进模型，缩小了分布外泛化的差距。
- **ArmoRM** 抛弃了单一标量黑盒，用**多目标绝对评分 + 门控 MoE** 走向了可解释、可纠偏的奖励合成。
- **DeepSeek-R1** 则在推理任务上彻底掀翻了桌子，示范了**“推理优先可验证奖励、通用阶段仍用学习式 RM”**的双轨范式。

在视觉生成侧，**ImageReward** 这样的跨模态偏好模型与 **Flow-GRPO** 等算法的结合，正在努力驾驭高维像素空间中的多维主观偏好。

奖励模型是 RLHF 的隐藏引擎。理解了它，你才算真正看懂了大模型对齐的底层逻辑。

> 参考资料（节选）：
>
> 1. Gao, L., Schulman, J. & Hilton, J. (2023). *Scaling Laws for Reward Model Overoptimization*. ICML. [arXiv:2210.10760](https://arxiv.org/abs/2210.10760).
> 2. Kwa, T., Thomas, D. & Garriga-Alonso, A. (2024). *Catastrophic Goodhart: Regularizing RLHF with KL Divergence Does Not Mitigate Heavy-Tailed Reward Misspecification*. NeurIPS. [arXiv:2407.14503](https://arxiv.org/abs/2407.14503).
> 3. Mahan, D., et al. (2024). *Generative Reward Models*. [arXiv:2410.12832](https://arxiv.org/abs/2410.12832).
> 4. Wang, H., et al. (2024). *Interpretable Preferences via Multi-Objective Reward Modeling and Mixture-of-Experts*. [arXiv:2406.12845](https://arxiv.org/abs/2406.12845).
> 5. DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948).
> 6. Xu, J., et al. (2023). *ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation*. NeurIPS. [arXiv:2304.05977](https://arxiv.org/abs/2304.05977).
> 7. *Flow-GRPO: Training Flow Matching Models via Online RL*. [arXiv:2505.05470](https://arxiv.org/abs/2505.05470).
> 8. Lambert, N., et al. (2024). *RewardBench: Evaluating Reward Models for Language Modeling*. [arXiv:2403.13787](https://arxiv.org/abs/2403.13787).
> 9. Malik, S., Pyatkin, V., Land, S., Morrison, J., Smith, N. A., Hajishirzi, H. & Lambert, N. (2025). *RewardBench 2: Advancing Reward Model Evaluation*. [arXiv:2506.01937](https://arxiv.org/abs/2506.01937).
