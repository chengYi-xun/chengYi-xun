---
title: 笔记｜强化学习（十一）：奖励模型——RLHF 的隐藏引擎
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

> 在前面的 RL 系列中，我们详细讨论了 PPO、DPO、GRPO 等策略优化算法——它们都在回答"如何根据奖励信号更新策略"。但一个更基本的问题被忽略了：**奖励信号本身从何而来？** 本文专门讨论奖励模型（Reward Model, RM），它是 RLHF 管线中真正将人类偏好转化为可优化信号的核心组件。
>
> ⬅️ 上一篇：[笔记｜强化学习（十）：LLM 对齐中的 RL 方法全景对比——从 PPO 到 SuperFlow](/chengYi-xun/posts/60-rl-alignment-comparison/)

# 1. 为什么需要奖励模型？

在传统强化学习中，环境直接给出奖励——Atari 游戏得分、围棋输赢、机器人是否倒下。奖励函数是明确的、确定性的。但在 LLM 对齐中，"环境"是人类用户，"奖励"是人类对回答质量的**主观判断**——什么叫"有帮助"、"安全"、"诚实"？没有一个公式能精确定义。

更关键的困难在于**规模**。PPO 训练中，策略每生成一条回答就需要一个奖励分数。一次训练可能涉及百万级的生成样本——不可能让人类逐一打分。因此我们需要一个**代理**（proxy）：用有限的人类标注数据训练一个模型，让它**模拟人类的偏好判断**。

这就是奖励模型。它在 RLHF（Reinforcement Learning from Human Feedback）管线中的位置：

$$
\underbrace{\text{预训练}}_{\text{通用语言能力}} \longrightarrow \underbrace{\text{SFT}}_{\text{指令跟随}} \longrightarrow \underbrace{\text{RM 训练}}_{\text{学习偏好}} \longrightarrow \underbrace{\text{RL 优化}}_{\text{PPO/GRPO/...}}
$$

**一句话定义**：奖励模型把人类的主观偏好压缩为一个标量分数 $r_\theta(x, y)$，其中 $x$ 是 prompt，$y$ 是模型生成的回答。

注意这个分数**没有绝对意义**——"这个回答值 3.7 分"本身毫无意义。重要的是**相对排序**：如果人类偏好回答 A 胜过回答 B，那么 $r_\theta(x, A) > r_\theta(x, B)$。"只有差值有意义"——这一洞察直接引出了下一节的 Bradley-Terry 偏好模型。

# 2. Bradley-Terry 偏好模型：从人类偏好到数学公式

## 2.1 模型定义

Bradley-Terry（BT）模型是 1952 年由 Ralph Allan Bradley 和 Milton E. Terry 提出的配对比较模型（Bradley & Terry, 1952, *Biometrika*）。它原本用于体育比赛排名——给每个选手一个"强度"分数，预测配对比赛的胜负概率。

核心假设：每个选项 $i$ 有一个潜在的"强度"参数 $r_i \in \mathbb{R}$，两者比较时选 $i$ 胜出的概率为：

$$
P(i \succ j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \sigma(r_i - r_j)
$$

其中 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 是 sigmoid 函数。

**平移不变性**：给所有分数加一个常数 $c$，$\sigma((r_i + c) - (r_j + c)) = \sigma(r_i - r_j)$，概率不变。这意味着**只有分数之差有意义**，绝对分数没有。

应用到 LLM 对齐中：给定 prompt $x$ 和两个回答 $y_c$（chosen，人类偏好的）和 $y_r$（rejected，人类不偏好的），奖励模型预测 $y_c$ 被偏好的概率为：

$$
P(y_c \succ y_r \mid x) = \sigma\big(r_\theta(y_c \mid x) - r_\theta(y_r \mid x)\big)
$$

## 2.2 从概率到损失函数

给定偏好数据集 $\mathcal{D} = \{(x^{(i)}, y_c^{(i)}, y_r^{(i)})\}_{i=1}^N$，训练目标是最大化正确偏好的对数似然：

$$
\max_\theta \sum_{i=1}^N \log P(y_c^{(i)} \succ y_r^{(i)} \mid x^{(i)}) = \max_\theta \sum_{i=1}^N \log\sigma\big(r_\theta(y_c^{(i)} \mid x^{(i)}) - r_\theta(y_r^{(i)} \mid x^{(i)})\big)
$$

等价地，最小化负对数似然（这就是 BT 损失函数）：

$$
\mathcal{L}_{\text{BT}}(\theta) = -\mathbb{E}_{(x, y_c, y_r) \sim \mathcal{D}}\left[\log\sigma\big(r_\theta(y_c \mid x) - r_\theta(y_r \mid x)\big)\right]
$$

利用 $-\log\sigma(\Delta) = \log(1 + e^{-\Delta})$，等价的 softplus 形式为：

$$
\mathcal{L}_{\text{BT}}(\theta) = \mathbb{E}_{(x, y_c, y_r) \sim \mathcal{D}}\left[\log\big(1 + e^{r_\theta(y_r \mid x) - r_\theta(y_c \mid x)}\big)\right]
$$

**直觉**：当 $r_\theta(y_c) \gg r_\theta(y_r)$（chosen 远高于 rejected）时，$\sigma(\Delta) \to 1$，$\mathcal{L} \to 0$。反之，模型搞反了偏好时，$\sigma(\Delta) \to 0$，$\mathcal{L} \to \infty$。

### Llama 2 的 Margin Loss 变体

当标注者不仅给出偏好方向还给出**偏好强度**（如 Likert 量表 1-5 分）时，可以利用分数差值作为 margin（Touvron et al., 2023）：

$$
\mathcal{L}_{\text{margin}}(\theta) = -\log\sigma\big(r_\theta(y_c \mid x) - r_\theta(y_r \mid x) - m(y_c, y_r)\big)
$$

例如 chosen 被评 5 分、rejected 被评 2 分，则 $m = 5 - 2 = 3$。这迫使 chosen 的分数不仅要高于 rejected，还要**高出至少 3 个单位**。但 Llama 3 团队观察到该技术在规模化后收益递减，最终移除了 margin 项（Llama 3 Team, 2024）。

## 2.3 伪代码实现

BT 损失的实现核心只有一行：

```python
rewards_c = model(prompt + chosen_response)     # 标量分数 r_θ(y_c|x)
rewards_r = model(prompt + rejected_response)    # 标量分数 r_θ(y_r|x)

loss = -F.logsigmoid(rewards_c - rewards_r).mean()  # BT 损失
```

让 chosen 的分数比 rejected 高，差值越大损失越小。

## 2.4 与 DPO 的深层联系

回忆 RLHF 的标准优化目标（Ziegler et al., 2019）：

$$
\max_\pi \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot|x)}\big[r(x, y)\big] - \beta\, D_{\text{KL}}\big(\pi(\cdot|x) \| \pi_\text{ref}(\cdot|x)\big)
$$

这个 KL 约束的优化问题有**解析解**（Rafailov et al., 2023）：

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_\text{ref}(y \mid x)\, \exp\!\Big(\frac{r(x, y)}{\beta}\Big)
$$

其中 $Z(x) = \sum_y \pi_\text{ref}(y|x)\exp(r(x,y)/\beta)$ 是配分函数。对上式取对数并求解 $r$：

$$
r(x, y) = \beta\log\frac{\pi^*(y \mid x)}{\pi_\text{ref}(y \mid x)} + \beta\log Z(x)
$$

这就是**隐式奖励**——用策略的对数概率比来表示奖励。将它代入 BT 模型：

$$
P(y_c \succ y_r | x) = \sigma\!\bigg(\beta\Big[\log\frac{\pi^*(y_c|x)}{\pi_\text{ref}(y_c|x)} - \log\frac{\pi^*(y_r|x)}{\pi_\text{ref}(y_r|x)}\Big]\bigg)
$$

注意 $Z(x)$ 只依赖 $x$ 不依赖 $y$，在做差时被消去了。最终 DPO 损失：

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_c, y_r)}\left[\log\sigma\!\bigg(\beta\Big[\log\frac{\pi_\theta(y_c|x)}{\pi_\text{ref}(y_c|x)} - \log\frac{\pi_\theta(y_r|x)}{\pi_\text{ref}(y_r|x)}\Big]\bigg)\right]
$$

与 BT-RM 损失对比：

| | BT-RM 损失 | DPO 损失 |
|:---|:---|:---|
| **形式** | $-\log\sigma(r_\theta(y_c) - r_\theta(y_r))$ | $-\log\sigma\big(\beta[\log\frac{\pi_\theta(y_c)}{\pi_\text{ref}(y_c)} - \log\frac{\pi_\theta(y_r)}{\pi_\text{ref}(y_r)}]\big)$ |
| **奖励来源** | 显式学习的标量 $r_\theta$ | 隐式的 $\beta\log\frac{\pi_\theta}{\pi_\text{ref}}$ |
| **需要 RM** | 是 | 否 |

**结构完全一致**——DPO 本质上是把策略本身当作 reward model，省去了独立训练 RM 的步骤。正如 Rafailov et al. 的论文标题所言：*Your Language Model is Secretly a Reward Model*。

# 3. RM 的架构与训练流程

## 3.1 架构

奖励模型的架构非常简单——在一个预训练语言模型顶部加一个**线性头**（Linear Head），将最后一个 token 的隐藏状态映射为标量分数：

$$
r_\theta(x, y) = \mathbf{w}^\top \mathbf{h}_{\text{last}} + b
$$

其中 $\mathbf{h}_{\text{last}} \in \mathbb{R}^{d}$ 是序列最后一个非 padding token（通常是 EOS token）的隐藏状态，$\mathbf{w} \in \mathbb{R}^d$、$b \in \mathbb{R}$ 是线性头的参数。整个 backbone（语言模型）的参数也会被更新。

```python
class RewardModel(nn.Module):
    def __init__(self, base_lm):
        self.lm = base_lm                                         # SFT 后的语言模型骨干
        self.head = nn.Linear(base_lm.config.hidden_size, 1)      # hidden_size → 1 的线性头

    def forward(self, input_ids, attention_mask):
        hidden = self.lm(                                          # 前向传播整个序列
            input_ids, attention_mask,
            output_hidden_states=True
        ).hidden_states[-1]                                        # 取最后一层隐藏状态

        lengths = attention_mask.sum(dim=1) - 1                    # 每条序列最后一个非 pad token 的位置
        last_hidden = hidden[range(len(hidden)), lengths]          # (batch, hidden_size)
        return self.head(last_hidden).squeeze(-1)                  # (batch,) —— 每条序列一个标量分数
```

为什么取**最后一个 token**？因为在 causal LM（自回归模型）中，最后一个 token 的隐藏状态已经通过注意力机制聚合了整个序列的信息。它是对完整回答的"压缩摘要"。

## 3.2 训练流程

完整的 RM 训练过程如下：

**第一步：收集偏好数据**

给人类标注者展示一个 prompt 和两个（或更多）不同的模型回答，让他们判断哪个更好。标注格式：

```
Prompt: "解释什么是量子纠缠"
Response A: "量子纠缠是两个粒子之间的一种关联..."（详细、准确）
Response B: "量子纠缠就是两个东西连在一起"（过于简化）
人类判断: A > B
→ 产出一条偏好对: (prompt, chosen=A, rejected=B)
```

InstructGPT（Ouyang et al., 2022）雇佣了 40 名标注者，对每个 prompt 标注 4-9 个回答的排序。每个排序可以产生 $\binom{K}{2}$ 个配对。

**第二步：初始化 RM**

从 **SFT checkpoint**（而非预训练 checkpoint）初始化奖励模型。原因：SFT 模型已经学会了指令格式和任务理解，RM 只需要在此基础上学习"什么是好的回答"。加上线性头后，新增参数量极少（hidden_size + 1）。

**第三步：训练**

- **损失函数**：上述 BT 损失
- **学习率**：通常比 SFT 低一个数量级（$\sim 1 \times 10^{-6}$ 到 $5 \times 10^{-6}$），因为 backbone 已经预训练好了
- **Epoch 数**：**只训 1 个 epoch**——这是最重要的技巧。RM 的训练数据量远小于预训练数据，多个 epoch 会严重过拟合偏好数据中的 spurious pattern（Nathan Lambert, *RLHF Book*, Chapter 5）
- **奖励归一化**：训练后通常对输出分数做归一化（如减去均值），使分数分布稳定在零附近，有利于下游 RL 训练的数值稳定性

**第四步：验证**

在留出的偏好测试集上计算准确率——给定一个 (prompt, chosen, rejected) 三元组，RM 能否正确判断 $r(y_c) > r(y_r)$？好的 RM 准确率通常在 65%-75%（偏好本身存在标注者间的不一致性，上限远低于 100%）。

## 3.3 多响应排序：从配对到 K-wise

InstructGPT 对每个 prompt 采样 $K = 4 \sim 9$ 个回答让标注者排序，产生 $\binom{K}{2}$ 个配对比较。但这些配对**高度相关**（共享同一个 prompt），如果天真地把它们打散到数据集中，RM 会过拟合到个别 prompt。

解决方案：将同一 prompt 的所有 $\binom{K}{2}$ 个配对打包到同一个训练 batch 中，并做平均：

$$
\mathcal{L}(\theta) = -\frac{1}{\binom{K}{2}} \sum_{(y_c, y_r) \in \text{pairs}} \log\sigma\big(r_\theta(y_c|x) - r_\theta(y_r|x)\big)
$$

更一般地，**Plackett-Luce** 模型可以直接建模完整排序的概率。设 $\sigma^i$ 是标注者给出的排列（$\sigma^i(0)$ 是最优回答），则：

$$
P(\sigma^i | x) = \prod_{k=0}^{K-1} \frac{\exp\big(r_\theta(x, y_{\sigma^i(k)})\big)}{\sum_{j=k}^{K-1} \exp\big(r_\theta(x, y_{\sigma^i(j)})\big)}
$$

这是一个"逐步淘汰"的过程——每一步从剩余候选中选出最好的那个。当 $K=2$ 时退化为 Bradley-Terry 模型（Zhu et al., 2023, Starling-RM）。

# 4. 三种奖励范式

标准的 BT-RM 对**整个回答**给一个分数。但在推理密集型任务（数学、代码）中，"整体打分"的粒度太粗——模型可能推理过程全错但蒙对了答案。为此，研究者发展出了不同粒度的奖励范式。

## 4.1 Outcome Reward Model（ORM）：按结果打分

ORM 的核心思想：不管推理过程如何，只看**最终结果是否正确**。

ORM 的架构与标准 RM 类似，但输出粒度不同。它在**每个 token 位置**输出一个正确概率（而非只在 EOS 处输出一个标量），使用二元交叉熵损失训练：

$$
\mathcal{L}_{\text{ORM}}(\theta) = -\mathbb{E}_{(x, y, r) \sim \mathcal{D}}\Big[r\log p_\theta(x, y) + (1-r)\log\big(1-p_\theta(x, y)\big)\Big]
$$

其中 $r \in \{0, 1\}$ 是整个回答的正确性标签（正确为 1，错误为 0）。在实现中，**这个标签被复制到每个 completion token 上**（prompt token 被 mask 为 -100 不参与损失计算）：

```python
class OutcomeRewardModel(nn.Module):
    def __init__(self, base_lm):
        self.lm = base_lm
        self.head = nn.Linear(base_lm.config.hidden_size, 1)  # 每个 token 一个 logit

    def forward(self, input_ids, attention_mask, labels):
        hidden = self.lm(                                       # 前向传播
            input_ids, attention_mask,
            output_hidden_states=True
        ).hidden_states[-1]
        logits = self.head(hidden).squeeze(-1)                  # (batch, seq_len) —— 逐 token logit
        mask = (labels != -100)                                 # prompt token 不参与损失
        loss = F.binary_cross_entropy_with_logits(
            logits[mask], labels[mask].float()                  # 二元交叉熵
        )
        return loss, logits
```

**推理时的聚合**：ORM 输出的是逐 token 的正确概率 $p_t$。聚合方式包括：

- **均值**：$\bar{p} = \frac{1}{T}\sum_t p_t$
- **最小值**（保守策略）：$\min_t p_t$
- **乘积**（等价于对数概率求和）：$\prod_t p_t$

**局限**：ORM 只看结果不看过程——一个推理步骤全错但最终蒙对答案的回答，ORM 仍会给高分。这种"答案正确但过程错误"的问题在数学推理中尤为严重。

## 4.2 Process Reward Model（PRM）：按步骤打分

PRM 对**每个推理步骤**单独打分，精确定位错误发生在哪一步。

### 核心论文：Let's Verify Step by Step

OpenAI 的 Lightman et al.（2023）是 PRM 的奠基性工作。他们在 MATH 数据集上对比了两种监督方式：

- **Outcome Supervision（ORM）**：只给最终答案标签
- **Process Supervision（PRM）**：给每个推理步骤标签（正确/错误/中性）

核心发现：**Process Supervision 在 MATH 数据集上显著优于 Outcome Supervision**。具体地，使用 PRM 做 Best-of-N 选择时，MATH 准确率从 ORM 的 ~53% 提升到 ~78%（在从 MATH 测试集中均匀采样的 500 道题的代表性子集上）。

PRM 的损失函数是逐步骤的交叉熵：

$$
\mathcal{L}_{\text{PRM}}(\theta) = -\mathbb{E}_{(x, s) \sim \mathcal{D}} \left[\sum_{i=1}^{K} y_{s_i}\log r_\theta(s_i \mid x) + (1-y_{s_i})\log\big(1-r_\theta(s_i \mid x)\big)\right]
$$

其中 $s = (s_1, s_2, \ldots, s_K)$ 是包含 $K$ 个推理步骤的链式推理，$y_{s_i} \in \{0, 1\}$ 标注每步是否正确，$r_\theta(s_i | x)$ 是模型在第 $i$ 个步骤分隔符处输出的正确概率。

标签只在**步骤分隔符**（如双换行 `\n\n` 或特殊 token）处提供，其余 token 被 mask。HuggingFace TRL 中的实现：

```python
separator_ids = tokenizer.encode(step_separator, add_special_tokens=False)
completions_ids = [completion + separator_ids for completion in completions_ids]
# 标签：只在每个步骤的最后一个 token 上有值，其余为 -100
labels = [[-100] * (len(c) - 1) + [label] for c, label in zip(completions_ids, labels)]
```

PRM 的架构与 ORM 类似，但输出 3 个类别（正确/中性/错误）而非 2 个：

```python
class ProcessRewardModel(nn.Module):
    def __init__(self, base_lm, num_classes=3):
        self.lm = base_lm
        self.head = nn.Linear(base_lm.config.hidden_size, num_classes)  # 3 类输出

    def forward(self, input_ids, attention_mask, labels):
        hidden = self.lm(
            input_ids, attention_mask,
            output_hidden_states=True
        ).hidden_states[-1]
        logits = self.head(hidden)                  # (batch, seq_len, 3)
        mask = (labels != -100)                     # 只在步骤分隔符处有标签
        loss = F.cross_entropy(logits[mask], labels[mask])
        return loss, logits
```

**推理时**：PRM 在步骤边界处输出分数。聚合方式：

- **最小值**（fail-fast）：$\min_i r_\theta(s_i)$——一旦某步被判为错误，整个链即低分
- **均值**：$\frac{1}{K}\sum_i r_\theta(s_i)$
- **加权均值**（偏重后续步骤）：后面的步骤更关键

### PRM 的标注成本问题与 Math-Shepherd

PRM 的最大瓶颈是**步骤级标注成本极高**。OpenAI 为 PRM800K 数据集请了人工标注者对 75,000 条解答中的 800,000 个步骤逐一标注正确/错误/中性。

**Math-Shepherd**（Wang et al., 2024, ACL）提出了自动化方案：

1. 对于每个中间步骤 $s_i$，从该步骤出发**采样 $M$ 条后续路径**直到得出最终答案
2. 统计这 $M$ 条路径中最终答案正确的比例 $\hat{c}_i$
3. 将 $\hat{c}_i$ 作为该步骤的软标签（蒙特卡洛估计）

这种方法无需人工标注，自动为每个步骤生成质量信号。实验表明（Wang et al., 2024, ACL），以 LLaMA2-70B 为生成器，Math-Shepherd 在 GSM8K 上达到 93.2%（vs. ORM 的 91.8%），在 MATH 上达到 44.5%（vs. ORM 的 40.4%）；以 Mistral-7B 为基础的 step-by-step PPO 则将 GSM8K 从 77.9% 提升到 84.1%，MATH 从 28.6% 提升到 33.0%。与 ORM 相比，PRM 在更具挑战性的 MATH 数据集上优势更为明显。

## 4.3 可验证奖励（Verifiable Reward）：规则直接判定

对于某些任务，根本不需要学习型 RM——规则就能直接给出对错：

| 任务类型 | 验证方式 | 示例 |
|:---|:---|:---|
| 数学计算 | 答案精确匹配 | 模型输出 "42" vs 标准答案 "42" |
| 编程题 | 单元测试通过率 | `assert solution([1,2,3]) == 6` |
| 格式要求 | 正则匹配 | 输出是否为合法 JSON |
| 逻辑推理 | 形式化验证 | 证明是否可被 Lean/Coq 验证 |

可验证奖励与 GRPO/DAPO 天然配合——不需要 RM，直接用规则打分。这也是 DeepSeek-R1 的核心策略：在数学和代码任务上用规则奖励驱动 GRPO，完全绕过了 RM 训练。

可验证奖励的最大优势是**不可被 hack**——通过单元测试就是通过，不存在"骗过 RM"的空间。但它的适用范围有限——开放式对话、创意写作、安全性判断等任务无法用规则验证。

## 4.4 四种模型类型对比

| 模型类型 | 预测什么 | 训练数据 | 输出粒度 | 推理时用法 |
|:---|:---|:---|:---:|:---|
| **BT-RM** | 序列级质量分数 $r(x,y)$ | 偏好对 $(y_c, y_r)$ | 序列级 | 对 $K$ 条回答排序，选最好的（BoN）；或作为 RL 终端奖励 |
| **ORM** | 逐 token 正确概率 | 正确/错误回答对 | Token 级 | 聚合为序列分数（均值/最小值/乘积） |
| **PRM** | 逐步骤正确概率 | 步骤级标注 | 步骤级 | 指导搜索/剪枝低分分支；BoN 选择 |
| **可验证奖励** | 规则判定结果 | 无需训练数据 | 序列级 | 直接作为 RL 奖励信号 |

还需要区分 ORM 和**价值函数**（Value Function）——两者架构相似但语义不同：

- **ORM** 预测"这个 token 正确吗？"——$p(\text{correct}_t)$，标签来自离线数据
- **Value Function** 预测"从当前状态开始还能获得多少奖励？"——$V(s_t) = \mathbb{E}[\sum_{k \geq t} \gamma^{k-t} r_k]$，标签来自在线 rollout

当折扣因子 $\gamma = 1$ 时两者形式接近，但训练来源和更新机制完全不同。

# 5. 视觉生成中的奖励模型

前面讨论的 BT-RM、ORM、PRM 都是为**文本**设计的。但当 RL 被引入图像/视频生成（如 Flow-GRPO、SuperFlow、DanceGRPO），奖励模型的形态完全不同——因为"奖励"不再是"这段文字是否有帮助"，而是"这张图是否好看、是否符合提示词、构图是否合理"。

## 5.1 基础奖励模型

| 模型 | 评估维度 | 架构 | 训练数据 | 代表论文 |
|:---|:---|:---|:---|:---|
| **CLIP Score** | 文本-图像对齐 | CLIP ViT-L/14 余弦相似度 | CLIP 预训练 | Radford et al., 2021 |
| **Aesthetic Score** | 视觉美感 | CLIP 视觉编码器 + 线性头 | LAION 美学评分数据 | Schuhmann et al., 2022 |
| **ImageReward** | 综合偏好 | BLIP 编码器 + MLP 打分头 | 137K 专家偏好对 | Xu et al., NeurIPS 2023 |
| **PickScore** | 人类偏好 | CLIP ViT-H/14 + 偏好头 | Pick-a-Pic 数据集 | Kirstain et al., 2023 |
| **HPSv2** | 人类偏好 | CLIP 微调 | HPD v2 偏好数据 | Wu et al., 2023 |

**CLIP Score** 是最基础的——它直接用 CLIP 模型算文本和图像嵌入的余弦相似度，衡量"图片是否匹配提示词"。但它只关注语义对齐，完全不考虑美感或细节质量。

**Aesthetic Score** 在 CLIP 视觉编码器上加一个线性头，预测图像的视觉吸引力分数。它是**prompt-agnostic** 的——不看提示词，只看图片好不好看。

**ImageReward**（NeurIPS 2023）是第一个专为文本到图像生成设计的通用奖励模型。它在 137K 条专家级偏好数据上训练，同时考虑文本对齐、美感和细节质量。实验表明 ImageReward 与人类判断的一致性显著优于 CLIP Score 和 Aesthetic Score。

## 5.2 视频生成的专项奖励

视频生成引入了文本和图像没有的维度——**时间连贯性**和**运动质量**：

| 奖励维度 | 评估内容 | 实现方式 |
|:---|:---|:---|
| **VideoAlign** | 视频整体与提示词的对齐 | VLM 打分（如 InternVL） |
| **Video Motion Quality** | 运动轨迹是否物理合理 | 物理感知 VLM 分析轨迹和形变 |
| **时间一致性** | 帧间是否连贯无闪烁 | 光流一致性 + VLM 评估 |

DanceGRPO 使用了**五个奖励维度**的加权组合：图像美感、文本对齐、视频美感、运动质量、二值奖励（通过阈值将连续分数离散化为 0/1）。

## 5.3 多奖励融合与 Reward Hacking

单一奖励优化的最大风险是**Reward Hacking**——模型只优化一个维度而牺牲其他维度。例如：

- 只优化 CLIP Score → 生成"语义正确但画面丑陋"的图像
- 只优化 Aesthetic Score → 生成"好看但完全不符合提示词"的图像
- 只优化 ImageReward → 生成"中等质量但安全"的平庸图像

解决方案是**多奖励融合**：

**加权求和**（Flow-GRPO, SuperFlow）：

$$
r_{\text{total}} = \sum_k w_k \cdot r_k
$$

权重 $w_k$ 通常需要手动调节，且不同奖励的尺度可能差异巨大。

**Pareto 选择**（Flow-Multi）：在多个奖励维度上做 Pareto 最优筛选——只保留在所有维度上都不被其他样本"支配"的生成结果，避免单一维度的极端优化。

**MIRO 多奖励条件化**：在预训练阶段将多个奖励信号（ImageReward, HPSv2, PickScore, Aesthetic Score）作为条件注入模型，让模型学会在不同奖励组合下生成——推理时可以通过调节条件来控制生成偏好。

## 5.4 与 LLM 奖励模型的核心差异

| 维度 | LLM 奖励模型 | 视觉生成奖励模型 |
|:---|:---|:---|
| **输入** | 文本序列 | 图像/视频 + 文本提示词 |
| **架构** | LLM + 线性头 | CLIP/BLIP/VLM + 打分头 |
| **训练数据** | 文本偏好对 | 图像偏好对或美学评分 |
| **可验证性** | 数学/代码可验证 | 几乎不可验证（美感主观） |
| **多维性** | 通常单一分数 | 需要多维融合（美感、对齐、一致性） |
| **Reward Hacking** | 冗长、格式操纵 | 单维极端化（美但不对齐） |

图像/视频生成中**几乎没有可验证奖励**——"好看"是主观的，没有"单元测试"可以判定一张图的美感。这使得视觉生成 RL 比 LLM 数学推理 RL 更依赖学习型 RM，也更容易 Reward Hacking。

前面四节讨论了各种类型的 RM——无论是文本还是视觉领域，它们的训练都依赖人类偏好数据。但人工标注既昂贵又缓慢。有没有办法让 AI 自己来当评委？

# 6. LLM-as-Judge：用 AI 替代人类标注

## 6.1 基本思路

随着 GPT-4、Claude 等强大 LLM 的出现，一个自然的想法浮出水面：能否用 LLM 替代人类标注者来生成偏好数据？

**LLM-as-Judge**（Zheng et al., 2023, NeurIPS）就是这一思路的实现。典型的评估 prompt：

```
[System]
请作为公正的评委，评估两个 AI 助手对用户问题的回答质量。
评估标准：有用性、相关性、准确性、深度、创造性和细节程度。
先给出简短分析，再按以下格式输出裁决：
"[[A]]" 表示 A 更好，"[[B]]" 表示 B 更好，"[[C]]" 表示平局。

注意：不要因回答的呈现顺序影响判断。不要因回答的长度影响判断。

[用户问题]
{question}

[助手 A 的回答]
{answer_a}

[助手 B 的回答]
{answer_b}
```

这一思路催生了多个主流评估基准：**MT-Bench**（Zheng et al., 2023）、**AlpacaEval**（Li et al., 2023）、**Arena-Hard**（Li et al., 2024）、**WildBench**（Lin et al., 2024）。

## 6.2 RLAIF：用 AI 反馈训练 RM

**RLAIF**（Reinforcement Learning from AI Feedback）进一步将 LLM-as-Judge 嵌入训练管线——用 AI 生成的偏好数据来训练奖励模型，再用该 RM 做 RL。

Anthropic 的 **Constitutional AI**（Bai et al., 2022）是 RLAIF 的先驱工作。其流程：

1. **Critique & Revision**：让模型根据一组"宪法原则"（如"回答应当诚实"、"不应包含有害内容"）自我批评并修改回答
2. **AI Preference**：用模型根据宪法原则判断哪个回答更好，生成偏好数据
3. **RM + RL**：用 AI 生成的偏好数据训练 RM，再做 PPO

这大幅降低了人工标注成本，但引入了新的偏差来源。

## 6.3 已知偏差：来自实证研究的证据

大规模实证研究（覆盖 15+ LLM judge 和 ~150,000 评估实例）揭示了三种主要的非语义偏差：

### 位置偏差（Position Bias）

LLM 倾向于偏好**放在特定位置**的回答（通常是第一个位置）。这种偏差不是随机的——它受两个回答之间的**质量差距**强烈影响：当两个回答质量接近时，位置偏差尤为显著（Shi et al., 2025, AACL/IJCNLP）。

**缓解**：对每个评估样本，随机交换 A/B 的位置做两次评估，取多数票。

### 冗长偏差（Verbosity Bias）

LLM judge 系统性地偏好更长的回答，将长度作为"彻底性"的启发式代理。

**缓解**：在 prompt 中明确指出"不要因长度影响判断"；对评分做长度回归后取残差。

### 自我偏好偏差（Self-Preference Bias）

LLM 倾向于给**自己生成的文本**更高分。Panickssery et al.（NeurIPS 2024）发现 LLM 具有非平凡的自我识别能力，且自我识别强度与自我偏好偏差呈线性相关。Wataoka et al.（2024）进一步表明，这种偏差的根源是**困惑度**（perplexity）——LLM 对困惑度更低（即更"熟悉"）的文本给出更高评分，无论该文本是否由自己生成。

**缓解**：使用与生成模型不同家族的模型作为 judge。

## 6.4 最佳实践清单

| 实践 | 原理 |
|:---|:---|
| `temperature = 0` | 减少采样随机性，提高评分一致性 |
| 多次评估取平均/多数票 | 降低单次评估的噪声 |
| 交换位置做两次评估 | 消除位置偏差 |
| 使用不同模型交叉评估 | 消除自我偏好偏差 |
| 提供参考答案（如有） | 将评估锚定在客观标准上 |
| 在有人工标注的子集上校准 | 量化 judge 的系统性偏差 |

## 6.5 Generative RM vs. 传统 RM

LLM-as-Judge 也被称为**生成式奖励模型**（Generative RM）——它用自然语言推理来做偏好判断，而非输出一个标量分数。

一个关键事实：在 RewardBench 等评测上，**专门训练的 RM 仍然优于 Generative RM**（Lambert et al., 2024）。这说明：

- LLM-as-Judge 适合做**评估**（evaluation）和**数据标注**
- 但在需要高精度偏好预测的 RL 训练中，专门训练的 RM 仍有不可替代的价值

无论是专门训练的 RM 还是 LLM-as-Judge，它们都是人类偏好的**不完美近似**。当策略模型对着这个不完美的代理疯狂优化时，会发生什么？

# 7. Reward Hacking——奖励模型的阿喀琉斯之踵

## 7.1 什么是 Reward Hacking？

Reward Hacking 指策略模型找到了 RM 的"漏洞"——获得高分但实际上并没有真正提高回答质量。这是 **Goodhart's Law** 在 AI 对齐中的直接体现：

> "当一个度量指标变成优化目标时，它就不再是一个好的度量指标了。"
> —— Marilyn Strathern, 1997（对 Goodhart 定律的通俗转述）
>
> 原版（Goodhart, 1975）："任何被观测到的统计规律性，一旦被施加控制压力，就会趋于崩溃。"

在 RLHF 中，RM 是人类偏好的**代理指标**。当策略对这个代理指标优化到极致时，策略的行为开始偏离人类真正想要的——不是因为 RM 本身是错的，而是因为它只是一个不完美的近似。

## 7.2 Gao et al. 的过优化缩放定律

Gao et al.（2023, ICML）是 Reward Hacking 的系统性实证研究。他们用一个"黄金 RM"（Gold RM）模拟真实人类偏好，用一个较小的"代理 RM"（Proxy RM）来训练策略，然后观察在代理 RM 上优化时黄金 RM 分数的变化。

定义 $d := \sqrt{D_{\text{KL}}(\pi \| \pi_{\text{init}})}$，黄金奖励 $R$ 与 $d$ 的关系被拟合为：

$$
R_{\text{BoN}}(d) = \alpha_{\text{BoN}} \cdot d - \beta_{\text{BoN}} \cdot d^2
$$

$$
R_{\text{RL}}(d) = \alpha_{\text{RL}} \cdot d - \beta_{\text{RL}} \cdot d^2
$$

这是一个**先升后降的抛物线**：

- **初期**（小 $d$）：优化代理 RM 确实提升了黄金 RM 分数（$\alpha d$ 项主导）
- **后期**（大 $d$）：策略开始 hack 代理 RM，黄金分数反而下降（$-\beta d^2$ 项主导）
- **峰值**在 $d^* = \alpha / (2\beta)$ 处

关键发现：

- $\alpha$ 和 $\beta$ 系数随代理 RM 的参数量近似对数增长——更大的 RM 有更高的 $\alpha$（学得更好）和更低的 $\beta$（更难被 hack）
- 更大的**策略**模型整体表现更好，但 overoptimization 的程度（proxy-gold gap）与小策略相近
- KL 惩罚在 RL 中提高了代理分数/KL 的效率，但**并没有改善黄金分数/KL 的前沿**

## 7.3 常见表现

| 症状 | 机制 | 示例 |
|:---|:---|:---|
| **冗长膨胀** | RM 隐含"长 = 好"的偏差 | 本来一段话能说清楚的，扩展到五段 |
| **谄媚（Sycophancy）** | RM 对"同意用户"的回答给高分 | 用户说"1+1=3 对吗？"，模型回答"对的" |
| **格式操纵** | RM 对 Markdown 格式化的回答给高分 | 大量使用 `###`、`**粗体**`、列表 |
| **重复填充** | RM 对某些"安全短语"给高分 | 反复说"这是一个很好的问题..." |
| **编造推理** | RM 对"看起来像推理"的文本给高分 | 写出似是而非的推导过程来支撑错误答案 |

## 7.4 理论解释

设 RM 的分数为 $\hat{r}(x, y)$（proxy），真实人类偏好为 $r^*(x, y)$（gold）。RM 的近似误差为 $\epsilon(x, y) = \hat{r}(x, y) - r^*(x, y)$。

策略在优化 $\hat{r}$ 时，实际上在最大化：

$$
\hat{r}(x, y) = r^*(x, y) + \epsilon(x, y)
$$

如果 $\epsilon$ 在某些区域系统性偏高（即 RM 在这些输出上**过度给分**），策略会被吸引到这些区域——即使 $r^*$ 在那里并不高。优化压力越强（KL 约束越松），策略越容易跑到这些 RM 误差大的区域。

更严重的是，Kwa, Thomas & Garriga-Alonso（NeurIPS 2024）发现：当 RM 误差是**重尾分布**时，即使有 KL 惩罚也无法防止**灾难性 Goodhart 效应**（Catastrophic Goodhart）——策略可以在 proxy reward 上获得任意高分，却不比基础模型提供更多效用。反之，若误差是轻尾的，KL 正则化可以有效引导策略获得更高的真实效用。

## 7.5 缓解策略

| 策略 | 原理 | 代表方法 | 效果 |
|:---|:---|:---|:---|
| **KL 惩罚** | 限制策略偏离参考模型 | PPO 标配 | 基本防线，但不能防止重尾 hacking |
| **RM 集成** | 多个独立 RM 取平均或最小值 | Coste et al., 2023 | 降低单个 RM 的偏差风险 |
| **迭代训练** | 定期用新策略输出更新 RM | Online RLHF | 让 RM 跟上策略的分布漂移 |
| **绕过 RM** | DPO 或可验证奖励 | DPO、GRPO+规则 | 从根源消除 proxy 误差 |
| **长度惩罚** | 显式惩罚过长回答 | DAPO 的 overlong shaping | 针对冗长偏差 |
| **多目标 RM** | 分别建模有用性、安全性等 | Llama 2 的双 RM | 避免单一分数掩盖多维偏好 |

最根本的缓解方案是**可验证奖励**——如果奖励来自不可欺骗的规则（数学精确匹配、代码单元测试），就从根源消除了 hacking 空间。这也解释了为什么 GRPO/DAPO 在数学推理上如此成功，以及为什么 DeepSeek-R1 选择了"先用可验证奖励训练推理能力，再用 RM 做通用对齐"的两阶段策略。

# 8. 如何选择和评测奖励模型

## 8.1 RewardBench：RM 的标准评测

RewardBench（Lambert et al., 2024, Allen AI）是第一个专门评测奖励模型的标准框架。它的评测数据由 (prompt, chosen, rejected) 三元组构成，覆盖四个类别：

| 类别 | 子集 | 测试能力 |
|:---|:---|:---|
| **Chat** | alpacaeval-easy, mt-bench-easy, mt-bench-medium | 通用对话偏好判断 |
| **Chat Hard** | mt-bench-hard, llmbar 系列 | 困难对话——chosen 和 rejected 质量接近 |
| **Safety** | refusals-dangerous, refusals-offensive, xstest | 安全拒绝行为 |
| **Reasoning** | math-prm, hep（代码） | 推理和代码正确性 |

**指标**：Win Rate——RM 是否对 chosen 给出比 rejected 更高的分数。最终分数是各子集分数的加权平均。

**RewardBench 2**（Lambert et al., 2025）扩展为六个领域：Factuality、Precise Instruction Following、Math、Safety、Focus、Ties。采用 best-of-4 格式（1 个 chosen vs. 3 个 rejected），挑战性显著提升——顶级模型的平均分比 v1 低约 20 个百分点。

一个重要发现：**RewardBench 上的高分不等于下游 RL 训练的实际效果**。RM 和策略之间的"血统对齐"（lineage alignment）很重要——用 Llama 训的 RM 可能不适合训 Mistral 的策略。

此外，还有大量专项评测：

| 领域 | 评测框架 |
|:---|:---|
| 数学 | RewardMATH、AceMath-RewardBench |
| 过程奖励 | PRM-Bench、ProcessBench |
| 多模态 | VL-RewardBench、MJ-Bench |
| 多语言 | M-RewardBench |
| Agent | Agent-RewardBench |

## 8.2 选型建议

| 场景 | 推荐方案 | 理由 |
|:---|:---|:---|
| 通用对话对齐 | BT-RM + 高质量人工偏好数据 | 成熟、通用、工具链完善（TRL RewardTrainer） |
| 数学 / 代码推理 | 可验证奖励（首选）或 PRM | 零标注成本、不可被 hack；PRM 可指导搜索 |
| 复杂多步推理 | PRM + Math-Shepherd 自动标注 | 精确定位错误步骤，自动化降低标注成本 |
| 资源有限 | LLM-as-Judge → RLAIF | 无需人工标注，但需警惕系统性偏差 |
| 安全对齐 | BT-RM + 专项安全偏好数据 | 安全判断无法用规则验证，必须用学习型 RM |
| 多维度对齐 | 多个专项 RM（有用性 + 安全性分开训练） | 避免单一 RM 在不同维度间权衡 |

## 8.3 与策略优化算法的对接

回顾前文系列中的各种算法——不同的策略优化方法对奖励信号的需求完全不同：

| 算法 | 需要什么样的奖励？ | 推荐 RM 选型 |
|:---|:---|:---|
| **PPO** | 序列级标量奖励 + 价值函数（Critic） | BT-RM 或 ORM |
| **DPO** | 不需要 RM（隐式奖励） | 直接用偏好对训练策略 |
| **GRPO** | 序列级外部打分（无 Critic） | BT-RM、可验证奖励或 LLM-as-Judge |
| **DAPO** | 序列级外部打分（偏好可验证任务） | 可验证奖励 |
| **Best-of-N** | 序列级排序 | BT-RM 或 ORM（用于重排序） |
| **Tree Search** | 步骤级打分（指导分支剪枝） | PRM |

理解奖励模型的类型、训练方法和局限性，是正确选择和使用策略优化算法的前提。RM 的质量往往是 RLHF 管线中真正的**瓶颈**——再好的策略优化算法，面对一个有偏的 RM，也只会学到有偏的行为。

---

> 参考资料：
>
> 1. Bradley, R. A. & Terry, M. E. (1952). *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons*. Biometrika.
> 2. Lambert, N. (2025). *RLHF Book*, Chapter 5: Reward Models. [rlhfbook.com](https://rlhfbook.com/c/05-reward-models)
> 3. Ouyang, L., et al. (2022). *Training language models to follow instructions with human feedback (InstructGPT)*. NeurIPS.
> 4. Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS.
> 5. Lightman, H., et al. (2023). *Let's Verify Step by Step*. [arXiv:2305.20050](https://arxiv.org/abs/2305.20050).
> 6. Wang, P., et al. (2024). *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations*. ACL 2024. [arXiv:2312.08935](https://arxiv.org/abs/2312.08935).
> 7. Gao, L., Schulman, J. & Hilton, J. (2023). *Scaling Laws for Reward Model Overoptimization*. ICML 2023. [arXiv:2210.10760](https://arxiv.org/abs/2210.10760).
> 8. Lambert, N., et al. (2024). *RewardBench: Evaluating Reward Models for Language Modeling*. [arXiv:2403.13787](https://arxiv.org/abs/2403.13787).
> 9. Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023.
> 10. Bai, Y., et al. (2022). *Constitutional AI: Harmlessness from AI Feedback*. Anthropic.
> 11. Touvron, H., et al. (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models*. [arXiv:2307.09288](https://arxiv.org/abs/2307.09288).
> 12. Zhong, J., et al. (2025). *A Comprehensive Survey of Reward Models: Taxonomy, Applications, Challenges, and Future*. [arXiv:2504.12328](https://arxiv.org/abs/2504.12328).
> 13. Zhu, B., et al. (2023). *Starling-7B: Improving LLM Helpfulness and Harmlessness with RLAIF*. [arXiv:2312.09245](https://arxiv.org/abs/2312.09245).
> 14. Kwa, T., Thomas, D. & Garriga-Alonso, A. (2024). *Catastrophic Goodhart: Regularizing RLHF with KL Divergence Does Not Mitigate Heavy-Tailed Reward Misspecification*. NeurIPS 2024. [arXiv:2407.14503](https://arxiv.org/abs/2407.14503).
> 15. Panickssery, A., et al. (2024). *LLM Evaluators Recognize and Favor Their Own Generations*. NeurIPS 2024.
> 16. Wataoka, K., et al. (2024). *Self-Preference Bias in LLM-as-a-Judge*. [arXiv:2410.21819](https://arxiv.org/abs/2410.21819).
> 17. Wolfe, C. R. (2025). *Reward Models*. [Substack](https://cameronrwolfe.substack.com/p/reward-models).
> 18. Weng, L. (2024). *Reward Hacking in Reinforcement Learning*. [Lil'Log](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/).
> 19. Xu, J., et al. (2023). *ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation*. NeurIPS 2023. [arXiv:2304.05977](https://arxiv.org/abs/2304.05977).
> 20. Kirstain, Y., et al. (2023). *Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation*. [arXiv:2305.01569](https://arxiv.org/abs/2305.01569).
> 21. Wu, X., et al. (2023). *Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis*. [arXiv:2306.09341](https://arxiv.org/abs/2306.09341).
> 22. Xue, Z., et al. (2025). *DanceGRPO: Unleashing GRPO on Visual Generation*. [arXiv:2505.07818](https://arxiv.org/abs/2505.07818).
> 23. Dufour, N., et al. (2025). *MIRO: Multi-Reward Conditioning for Efficient Text-to-Image Generation*.
