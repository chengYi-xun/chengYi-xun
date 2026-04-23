---
title: 笔记｜强化学习（十一）：奖励模型基础——从传统 RL 到大模型与视觉生成
date: 2026-04-05 18:00:00
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

> 本篇为全文的第一部分：**基础与原理**。我们从「奖励模型是什么」出发，对比传统强化学习、大语言模型 RLHF 与视觉生成的奖励信号差异；给出 \(r_\theta(x,y)\) 的一般数学范式与 Bradley–Terry 配对损失；梳理文本侧「骨干 + 标量头」架构、五步训练管线，以及序列级 / 结果级 / 过程级 / 可验证奖励的粒度谱系；最后落到图像与视频生成里常见的对齐分、美学分与人类偏好模型，并点出视频奖励的多维难点。
>
> ⬅️ 上一篇：[笔记｜强化学习（十）：LLM 对齐中的 RL 方法全景对比——从 PPO 到 SuperFlow](/chengYi-xun/posts/60-rl-alignment-comparison/)
> ➡️ 下一篇：[笔记｜强化学习（十二）：奖励模型的挑战与 SOTA 演进 (2024-2026)](/chengYi-xun/posts/62-reward-model-advanced/)

## 奖励模型是什么：从环境的客观标量到人类偏好的可微代理

**奖励模型（Reward Model, RM）**是在数据上学习得到的标量打分器，用来近似某个「裁判」对生成结果的好坏判断。在数学上，它通常被定义为一个映射函数：

$$
r_\theta:\mathcal{X}\times\mathcal{Y}\to\mathbb{R},\quad (x,y)\mapsto r_\theta(x,y)
$$

这个公式精准概括了奖励模型的核心特征：

1. **输入 $\mathcal{X}\times\mathcal{Y}$**：同时接收提示词 $x$ 和生成的回答 $y$，评估回答是否**对齐**了用户意图。
2. **输出 $\mathbb{R}$**：将高维的文本/图像极度压缩为一个**一维实数标量**，作为强化学习（如 PPO）计算梯度的信号。
3. **参数 $\theta$**：它是一个神经网络，通过数据驱动来拟合人类难以言传的主观偏好，而非写死的规则。

有了这个 $r_\theta(x, y)$，我们就得到了一个**可自动求值的“代理裁判”**。无论是用强化学习（如 PPO）去更新模型参数，还是在推理时生成多个回答选最好的（Best-of-$N$），优化器都不需要每次去打扰人类打分，而是直接向这个裁判要分数，从而引导大模型不断朝着“人类更偏好的方向”进化。

打个通俗的比方：传统的 AI 任务（比如下棋、做数学题）就像是**客观题考试**，有标准答案，对就是对，错就是错；但生成式 AI（写文章、聊天）更像是一场**艺术设计评审**，没有绝对的满分，只有“这篇比那篇写得好”。

因为人类不可能每天坐在电脑前给 AI 的几百万句话逐一打分，所以我们需要用少量的人类打分数据，训练出一个**“稳定、便宜、不知疲倦”的 AI 评委**。这个 AI 评委，就是奖励模型。

### 传统强化学习：环境直接给出绝对回报刻度

在经典 RL（如 Atari 游戏、机器人控制）中，环境会直接返回数值奖励。它的关键性质是：

- **信号客观**：奖励由游戏规则或物理世界定义，比如吃一个金币得 10 分，摔倒扣 100 分。
- **具有绝对意义**：分数越高就是越好，智能体的目标就是最大化这个客观的绝对数值。

### 生成式 AI 的 RL：主观偏好与“代理裁判”的必要性

当任务变成大语言模型（聊天）或视觉生成（画图）时，环境变成了**人类用户**。此时面临三个根本差异：

1. **主观性**：什么叫“有帮助、安全、美观”？这没有绝对的标准答案。
2. **绝对分数难标，相对比较易标**：让人给一句话精确打“7.32分”既贵又不可靠；但让人在两句话里选“哪句更好”要自然得多。这把我们推向了**配对比较（Pairwise Comparisons）**。
3. **人类反馈不可扩展**：RL 训练需要评估百万级样本，不可能让人类实时打分。

因此，在 RLHF 中，我们必须先收集人类的**相对偏好数据**，然后训练一个 $r_\theta(x,y)$ 作为**代理裁判**，最后再用这个裁判去指导 RL 训练。

### 文本与视觉：同一骨架，不同的裁判构造

无论是大语言模型（LLM）还是视觉生成（图像/视频），核心思想都是训练 $r_\theta(x,y)$，但工程实现上差异巨大：

- **LLM RLHF**：评估的是离散的文本序列。裁判通常是一个因果 Transformer 模型。对于数学/代码等客观任务，甚至可以直接用**编译器/求解器**作为绝对客观的裁判（可验证奖励）。
- **视觉生成 RL**：评估的是连续的图像/视频像素。人类更关注美学、构图和动态连贯性。裁判通常是基于 CLIP 的图文对齐模型，或者是专门训练的视觉美学打分器（如 ImageReward）。视频生成还需要引入时间一致性等更复杂的奖励维度。

下文我们将先给出适用于文本与视觉的**通用配对建模骨架**（Bradley–Terry 模型），再深入 LLM 侧的实现细节，最后展开视觉侧的特殊性。

## 一般数学范式：配对偏好下的 $r_\theta(x,y)$ 与 Bradley–Terry 损失

### 把“谁更好”写成概率分布：Bradley–Terry 模型

Bradley–Terry（BT）模型（Bradley & Terry, 1952）最初是用来预测体育比赛胜负的。它假设每个选手 $i$ 都有一个潜在的“实力值” $r_i$。当选手 $i$ 和 $j$ 比赛时，$i$ 获胜的概率为：

$$
P(i \succ j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \sigma(r_i - r_j)
$$

其中 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 是我们熟悉的 Sigmoid 函数。

**平移不变性**：注意看这个公式，如果我们给所有选手的实力值都加上 100 分，概率 $P(i \succ j)$ 完全不变，因为 Sigmoid 里面算的是**差值** $(r_i + 100) - (r_j + 100) = r_i - r_j$。
这完美契合了人类的打分习惯：人类很难给出一个绝对客观的分数，但很容易判断“A 比 B 好”。因此，奖励模型学到的分数本质上只是用来**排序**的，绝对数值没有意义。

### 奖励模型的损失函数推导

现在，我们把“选手”换成“同一个提示词 $x$ 下的两个回答”。假设人类偏好的回答是 $y_w$（winning），不喜欢的回答是 $y_l$（losing）。

我们让神经网络 $r_\theta(x, y)$ 来输出这个“实力值”。那么，模型预测 $y_w$ 战胜 $y_l$ 的概率就是：

$$
P_\theta(y_w \succ y_l \mid x) = \sigma\big(r_\theta(x, y_w) - r_\theta(x, y_l)\big)
$$

在训练时，我们希望最大化这个正确预测的概率。在机器学习中，最大化似然等价于最小化**负对数似然损失（Negative Log-Likelihood）**。因此，我们就得到了大名鼎鼎的 **Bradley-Terry 损失函数**：

$$
\mathcal{L}_{\text{BT}}(\theta) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\Big[\log \sigma\big(r_\theta(x, y_w) - r_\theta(x, y_l)\big)\Big]
$$

### 这个损失函数在优化什么？

我们把差值记为 $\Delta = r_\theta(x,y_w) - r_\theta(x,y_l)$。
- 如果模型预测得很好（$\Delta$ 很大且为正），$\sigma(\Delta)$ 接近 1，损失 $\mathcal{L}$ 接近 0。
- 如果模型预测错了（$\Delta \le 0$），$\sigma(\Delta)$ 会很小，损失 $\mathcal{L}$ 会急剧增大。

此时，反向传播的梯度会做两件事：**拼命拉高 $r_\theta(x,y_w)$ 的分数，同时拼命压低 $r_\theta(x,y_l)$ 的分数**，直到两者的分差足够大。

所以，BT 损失的本质就是**排序学习（Learning to Rank）**：它用一个平滑的 Sigmoid 函数，把离散的“A > B”偏好，转化成了拉扯神经网络参数的连续梯度。

### 与 DPO 的同一骨架（可选阅读）

RLHF 在带 KL 正则的设定下可建立最优策略与隐式奖励的关系；**直接偏好优化（DPO）**把同样的 \(\sigma(\cdot)\) 结构接到策略对上，从而**不必单独维护 \(r_\theta\)**（Rafailov et al., 2023）。要点不是「否定 RM」，而是：**BT 配对似然**是偏好建模的共用数学骨架；差别在于奖励是显式的 \(r_\theta\)，还是折叠进 \(\beta\log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\) 的隐式参数化。

## 大语言模型中的奖励模型：因果表示、标量头与训练流水线

### 架构选择：SFT 骨干替换词表头为线性标量头，为何读取最后一个 token

实践中 \(r_\theta(x,y)\) 几乎总由**因果 Transformer 语言模型**实现：以 **SFT checkpoint** 初始化，**移除原词表投影（unembedding）**，接新的**单维仿射头**（Ouyang et al., 2022）。其前向计算可写为

$$
r_\theta(x,y) \;=\; \mathbf{w}^\top \mathbf{h}_{\text{last}} + b,
$$

其中 \(\mathbf{h}_{\text{last}}\) 是拼接后的完整序列（prompt + response）在**最后一个有效 token**（实现上常按 `attention_mask` 取末位非 padding 索引）上的隐状态。

**为何不是平均池化而是末位 token？** 因果掩码下，位置 \(t\) 只能看见 \(\le t\) 的上下文；只有序列末尾已经**读完全部 \((x,y)\)**。末位隐状态在信息意义上是整段文本的**全局汇聚点**，与 BERT 系用 `[CLS]` 做句向量对称：在 GPT 式架构里，由「读完全文再下判断」的位置输出标量，语义上与「对完整回答打分」一致。对中间位置做简单平均，会混入尚未读完回答的状态，通常与任务定义不匹配。

下图概括 RM 在训练期的数据流与模块关系（实现细节因框架略异，逻辑一致）：

![奖励模型：从 SFT 骨干到标量偏好分的数据流与结构示意](/chengYi-xun/img/rm-architecture.png)

与上文一致的极简实现骨架（省略梯度检查点等工程细节）：

```python
class RewardModel(nn.Module):
    def __init__(self, base_lm):
        super().__init__()
        self.lm = base_lm
        self.head = nn.Linear(base_lm.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.lm(
            input_ids, attention_mask, output_hidden_states=True
        ).hidden_states[-1]
        lengths = attention_mask.sum(dim=1) - 1
        last_hidden = hidden[range(len(hidden)), lengths]
        return self.head(last_hidden).squeeze(-1)
```

### 训练管线五步：从偏好三元组到可部署的 $r_\theta$

把 RM 训练想成一条可追溯的流水线：**数据收集 → 模型初始化 → 前向传播与损失计算 → 反向传播与参数更新 → 验证与评测**。

**步骤 1 — 数据收集：** 给人类标注者展示一个 prompt $x$ 和两个不同的模型回答，让他们判断哪个更好。这会产出一条偏好对：$(x, y_w, y_l)$。

**步骤 2 — 模型初始化：** 从 **SFT（监督微调）checkpoint** 初始化奖励模型的 Backbone，而不是从头预训练。
*原因*：SFT 模型已经学会了指令格式和人类语言规律，RM 只需要在此基础上进一步学习“什么是人类偏好的好回答”。加上线性头后，新增的随机初始化参数极少。

**步骤 3 — 前向传播与损失计算：** 在每个训练 Batch 中，将 chosen 和 rejected 的完整序列分别输入模型，在各自的最后一个 token 处读出隐藏状态，通过线性头产出标量分数 $r_c$ 和 $r_r$。随后，计算 Bradley-Terry 损失：
$$
\mathcal{L}_{\text{BT}} = -\log\sigma(r_c - r_r)
$$
这个损失函数的核心目的就是**拉大 $r_c$ 和 $r_r$ 的差距**。

**步骤 4 — 反向传播与参数更新：** 梯度从 BT 损失出发，一路回传到 Backbone 的每一层。
**注意：整个大语言模型的所有参数都会被更新！** 只有这样，模型才能在底层特征提取阶段，学会把注意力集中在那些决定回答质量的关键特征上（如逻辑、语气、幻觉等）。整个 Backbone 实际上被优化成了一个“偏好编码器”。

**步骤 5 — 验证与评测：** 
- **关键超参数**：学习率通常比 SFT 低一个数量级（避免破坏 SFT 阶段学到的语言能力）。**Epoch 数坚决只训 1 个**（因为 RM 的训练数据量远小于预训练数据，多训会严重过拟合偏好数据中的表面特征，导致后续的 Reward Hacking）。
- **评测指标**：在留出的偏好测试集上计算**配对准确率**——给定一个没见过的 $(x, y_w, y_l)$，RM 能否正确判断 $r(y_w) > r(y_l)$？好的 RM 准确率通常在 65%-75% 之间（人类偏好本身存在主观不一致性，上限远低于 100%）。

### 粒度谱系：序列级、结果级、过程级与可验证奖励

标准的 BT-RM 是对整个回答打一个总分，但在推理密集型任务（如数学、代码）中，这种“一锤子买卖”的粒度太粗了。为此，学术界发展出了不同粒度的奖励范式：

**1. 序列级（Sequence-level, BT-RM）**
对整条 $(x,y)$ 输出一个标量，语义是“整段回答的总分”。这是 InstructGPT 式 RM 的默认粒度，最适合开放式对话、安全性对齐等主观任务。

**2. 结果级（Outcome Reward Model, ORM）**
ORM 只关心**最终答案是否正确**。在训练时，它会给解答中的每个 token 都打上同一个二值标签（对或错）。推断时，通常取最后一个 token 的预测概率作为整条解答的得分。
*局限*：ORM 难以惩罚“过程全错但最后蒙对答案”的情况（False Positives）。

**3. 过程级（Process Reward Model, PRM）**
PRM 对推理链的**每一个中间步骤**进行打分（正 / 负 / 中性）。在 OpenAI 的 *Let's Verify Step by Step* 中，PRM 被证明在复杂数学推理上显著优于 ORM，因为它能精准定位逻辑错误发生在哪一步。
*局限*：需要极高昂的人工标注成本（如 PRM800K 数据集）。后续如 Math-Shepherd 尝试用蒙特卡洛树搜索自动生成步骤级信号来降低成本。

![PRM 逐步打分与归约流程](/chengYi-xun/img/prm-flowchart.png)

**4. 可验证奖励（Verifiable Rewards）**
对于数学答案、代码单测等，**正确性可由程序直接判定**，此时可以直接使用规则化的 0/1 奖励，**完全不需要训练神经网络 RM**。
*代表案例*：DeepSeek-R1-Zero 阶段完全抛弃了神经 RM，仅使用编译器和正则匹配作为奖励信号驱动 GRPO。这彻底消除了 Reward Hacking 的空间，且计算开销极低。

**总结：何时用哪一类？**
- **主观对话、安全、风格**：必须依赖 **序列级 BT-RM**。
- **客观可验证（代码/数学结果）**：首选 **可验证奖励**。
- **复杂多步推理且搜索空间大**：**PRM** 最善于引导搜索和剪枝，但系统复杂度最高。

## 视觉生成中的奖励模型：对齐、美学、人类偏好与视频的多维挑战

### 与文本 RM 的根本差异：缺少廉价真值、评价维度分裂

在图像与视频生成中，奖励模型 $r_\theta(x,y)$ 面对的条件 $x$ 通常是文本提示词，而输出 $y$ 则是高维的像素矩阵。与 LLM 相比，视觉生成领域的奖励模型面临着截然不同的挑战：

1. **没有统一的“正确答案”**：写代码可以跑单元测试，做数学题可以对答案，但“画一幅赛博朋克风格的猫”没有绝对的客观真值。
2. **主观维度的严重分裂**：一张图可能在“语义对齐”（画的确实是猫）上得分很高，但在“主观美感”（画得太丑了）上极差。视频生成更是增加了“时间一致性”（猫不能走着走着变成狗）和“运动合理性”等维度。单一的标量分数很难概括这么多互相冲突的维度。
3. **人类反馈极其昂贵**：让人类看两段 10 秒的视频并给出多维度的偏好评价，成本远高于看两段文本。

因此，视觉领域的 RM 往往不是一个单一的模型，而是**一组不同维度的打分器集合**。

### 常见图像奖励代理：CLIP 对齐分、美学估计与 ImageReward

目前，图像生成（如 Stable Diffusion 的微调）中最常用的奖励信号有三种：

**1. CLIP Score（语义对齐分）**
- **原理**：使用 CLIP 的图文双塔模型，计算图像特征向量和文本特征向量的余弦相似度。
- **优缺点**：计算极其便宜，且完全可微。但它只管“像不像”，不管“好不好看”。高 CLIP 分数的图像可能构图极差，或者充满了扭曲的伪影（Artifacts）。

**2. Aesthetic Score（美学估计分）**
- **原理**：在 CLIP 的视觉编码器之上，加一个简单的线性回归头，使用 LAION 美学数据集（人类对图像好看程度的打分）进行训练。
- **优缺点**：弥补了 CLIP 不管美感的缺陷。但“美”是高度主观的，这个分数会把特定标注群体（或特定摄影风格）的审美偏见直接写入模型。

**3. ImageReward（综合人类偏好 RM）**
- **原理**：这是清华大学在 2023 年提出的专为文生图设计的偏好模型。它收集了 13.7 万对专家级的人类偏好数据（综合考虑了对齐、美感、无害性等），训练了一个基于 BLIP 架构的打分器。
- **地位**：它在视觉领域的地位，就相当于 InstructGPT 里的 BT-RM。它直接拟合人类的综合排序，表现显著优于单纯的 CLIP 或 Aesthetic Score。

### 视频奖励：时间一致性、运动质量与多目标耦合

当从图像升级到视频（即帧序列 $\{y^{(t)}\}$）时，奖励模型需要同时惩罚更多维度的错误：
- **帧间闪烁与身份漂移**：主角走着走着衣服变了颜色。
- **运动不合理**：违背物理直觉的形变（如水往高处流）。
- **文本—视频对齐**：视频的事件发展顺序是否符合提示词的描述。

在工程实践中（例如 Flow-GRPO 等视频 RL 算法），通常会将奖励**分解为多个可计算的子项**，然后进行加权组合：
$$
r_{\text{total}} = w_1 \cdot r_{\text{CLIP}} + w_2 \cdot r_{\text{Aesthetic}} + w_3 \cdot r_{\text{OpticalFlow}} + \dots
$$
这里的难点在于**目标冲突**：比如为了让运动更剧烈（提高运动分），模型可能会生成模糊的画面（降低美学分）。

**总结**：在视觉生成中，理解“哪一个分数在优化什么维度”，比把 $r_\theta$ 当成一个神秘的黑盒数字要重要得多。视觉 RL 本质上是在多个互相牵制的奖励代理之间寻找帕累托最优（Pareto Optimal）。

> **参考文献（第一部分）**
>
> 1. Bradley, R. A. & Terry, M. E. (1952). *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons*. Biometrika.
> 2. Ouyang, L., et al. (2022). *Training language models to follow instructions with human feedback*. NeurIPS.
> 3. Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS.
> 4. Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML.
> 5. Xu, J., et al. (2023). *ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation*. NeurIPS.
> 6. Lightman, H., et al. (2024). *Let's Verify Step by Step*. ICLR. arXiv:2305.20050.
> 7. Cobbe, K., et al. (2021). *Training Verifiers to Solve Math Word Problems*. arXiv:2110.14168.
> 8. Touvron, H., et al. (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models*. arXiv:2307.09288.
> 9. Wang, P., et al. (2024). *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations*. ACL. arXiv:2312.08935.
> 10. Guo, D., et al. (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948.
> 11. Shao, Z., et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300.
