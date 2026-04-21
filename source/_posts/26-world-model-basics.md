---
title: 笔记｜世界模型（一）：世界模型全景综述——从认知科学到通用物理智能
date: 2026-04-06 00:30:00
categories:
 - Tutorials
tags:
 - World Model
 - Model-based RL
 - Latent Space
 - Generative Model
 - JEPA
 - Autonomous Driving
 - Video Generation
 - Survey
series: "世界模型"
mathjax: true
---

> 系列说明：本文是世界模型系列的第一篇，也是一篇面向初学者的全景综述。本文从最直觉的类比出发，循序渐进地展开概念、数学、技术路线、核心理论和未来展望。后续文章将对每条路线做深入拆解。
>
> ⬅️ 上一篇：[笔记｜生成模型（二十四）：DanceGRPO——让视频生成模型"跳好舞"的强化学习框架](/chengYi-xun/posts/25-video-grpo/)
>
> ➡️ 下一篇：[笔记｜世界模型（二）：Dreamer 系列——在想象中学习控制](/chengYi-xun/posts/27-dreamer/)



## 闭眼踢球——大脑里的物理模拟器

闭上眼睛，想象你在踢一个足球。球从脚尖飞出，在空中画一条抛物线，弹地后滚动减速。你不需要看到球，就能在脑海中预测它的轨迹。这种能力来自你大脑中的世界模型——一个关于物理世界如何运作的内部模拟器。

认知科学家 Kenneth Craik 在 1943 年就提出了这个概念："如果生物体能在头脑中构建一个外部现实的微型模型，它就能在行动前先在模型中尝试各种方案，预测哪种最优。"

在深度学习中，世界模型（World Model）就是这个内部模拟器的计算实现：一个能够预测环境在给定动作下如何变化的神经网络。这个定义虽然简单，却引出了 2018-2026 年间 AI 领域最激动人心的研究路线之一。从 DeepMind 的 Dreamer 系列在虚拟想象中训练机器人，到 OpenAI 的 Sora 将视频生成重新定义为世界模拟，再到 LeCun 提出的 JEPA 架构试图从根本上改变 AI 理解世界的方式——世界模型正在成为通往通用人工智能的关键拼图。



## 世界模型的数学定义

在展开各种具体模型之前，我们需要一套精确的数学语言。所有技术路线，无论看起来多不同，都是在实例化这同一套框架。

智能体（Agent）是一个能感知环境、做出决策并执行动作的实体。一个世界模型包含四个基本要素：状态 $s_t \in \mathcal{S}$（世界在时刻 $t$ 的完整描述）、观测 $o_t \in \mathcal{O}$（智能体能看到或感知到的信息）、动作 $a_t \in \mathcal{A}$（智能体执行的动作）、奖励 $r_t \in \mathbb{R}$（环境给出的即时反馈）。

关键区别在于状态不等于观测。在 Atari 游戏中，状态包括所有敌人的位置、速度、内部计时器等；而观测只是一帧 $210 \times 160 \times 3$ 的 RGB 图像。你看到的，永远只是世界的一个侧面。

前向动力学模型（Forward Dynamics Model）是给定当前时刻的世界状态 $s_t$ 和智能体执行的动作 $a_t$，预测下一时刻状态 $s_{t+1}$ 的函数或概率分布。世界模型的核心就是前向动力学：

$$
s_{t+1} \sim p_\theta(s_{t+1} \mid s_t, a_t)
$$

在正式定义观测模型之前，我们需要理解两个基本框架。

马尔可夫决策过程（Markov Decision Process, MDP）是一个五元组 $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$。MDP 的核心假设是马尔可夫性：下一个状态只依赖于当前状态和动作，与更早的历史无关。

部分可观测马尔可夫决策过程（Partially Observable MDP, POMDP）在 MDP 的基础上增加观测空间 $\mathcal{O}$ 和观测函数 $O(o_t \mid s_t)$。POMDP 的关键区别是：智能体不能直接看到真实状态 $s_t$，只能看到一个模糊的影子 $o_t$。真实世界几乎都是 POMDP，世界模型的核心挑战就是从不完整的观测中推断出尽可能完整的状态。

将以上要素整合，一个完整的状态空间模型由四个核心组件组成：

$$
\begin{aligned}
\text{推断模型（Inference）:} \quad & s_t \sim q_\theta(s_t \mid o_{\leq t}, a_{<t}) \\
\text{动力学模型（Transition）:} \quad & s_{t+1} \sim p_\theta(s_{t+1} \mid s_t, a_t) \\
\text{观测模型（Observation）:} \quad & o_t \sim p_\theta(o_t \mid s_t) \\
\text{奖励模型（Reward）:} \quad & r_t \sim p_\theta(r_t \mid s_t)
\end{aligned}
$$

这四个组件各自的直觉含义：

- **推断模型**：从"看到的东西"反推"世界真正的样子"。智能体只能获得局部观测（好比用手电筒照一小块区域），它需要结合所有历史观测 $o_{\leq t}$ 和历史动作 $a_{<t}$，在内部拼出完整的世界状态 $s_t$。这里用 $q_\theta$ 而非 $p_\theta$，是因为它是变分推断中的近似后验（编码器）。
- **动力学模型**：在脑中模拟"如果我这样做，世界会怎么变"。给定当前状态和动作，向前推演下一步状态。只依赖 $s_t$ 和 $a_t$、不依赖原始观测 $o_t$，体现了马尔可夫性。这是世界模型的核心引擎，让智能体不用真的执行动作，就能在"想象中"推演未来。
- **观测模型**：从世界状态生成"智能体应该看到什么"，本质是一个解码器/渲染器。训练时它提供重建损失——如果从 $s_t$ 无法还原出 $o_t$，说明学到的隐状态丢失了关键信息。
- **奖励模型**：根据当前状态预测"这个局面对我有多好"。有了它，智能体在脑中推演时可以同时评估每一步收益，从而选出最优策略——这正是 model-based RL 不需要大量真实交互就能做规划的关键。

四者协作：推断模型把碎片观测压缩为完整状态 → 动力学模型在状态上向前推演 → 观测模型验证推演是否合理（训练信号）→ 奖励模型评估每个推演出的未来值不值得追求。

对于潜变量世界模型，所有路线的训练目标都可以统一写为 ELBO（证据下界）。这是序列潜变量模型的标准变分推断结果，具体形式出自 Hafner et al. 的 PlaNet（arXiv:1811.04551）和 Dreamer（arXiv:1912.01603）：

$$
\log p_\theta(o_{1:T} \mid a_{1:T}) \geq \sum_{t=1}^{T} \mathbb{E}_{q_\phi}\left[\log p_\theta(o_t \mid s_t)\right] - \sum_{t=1}^{T} \text{KL}\left[q_\phi(s_t \mid o_{\leq t}) \;\|\; p_\theta(s_t \mid s_{t-1}, a_{t-1})\right]
$$

逐项解读：

**左边** $\log p_\theta(o_{1:T} \mid a_{1:T})$：在动作序列已知的条件下，模型认为出现这串观测的对数概率。动作 $a_{1:T}$ 是智能体实际执行过的（由策略决定，记录在 replay buffer 中），世界模型的任务是回答"做了这些动作之后，会看到什么"。但由于潜状态 $s_t$ 的存在，直接计算需要对所有可能的 $s_{1:T}$ 积分，计算上不可行，因此转而优化其下界（ELBO）。

**重建项** $\sum_{t=1}^{T} \mathbb{E}_{q_\phi}[\log p_\theta(o_t \mid s_t)]$：注意 $\mathbb{E}_{q_\phi}$ 的期望是对**隐状态** $s_t$ 取的。具体来说：推断模型 $q_\phi(s_t \mid o_{\leq t}, a_{<t})$ 根据历史观测和动作输出隐状态的分布，从中采样 $s_t$；然后观测模型 $p_\theta(o_t \mid s_t)$ 评估"如果世界状态是 $s_t$，看到真实观测 $o_t$ 的概率有多大"；对采样求期望、对时间步求和。本质就是 VAE 的编码-解码重建损失——还原不好说明 $s_t$ 丢失了关键信息。

**KL 正则项** $\sum_{t=1}^{T} \text{KL}[q_\phi(s_t \mid o_{\leq t}) \| p_\theta(s_t \mid s_{t-1}, a_{t-1})]$：要求两种途径推断出的状态分布保持一致。$q_\phi(s_t \mid o_{\leq t})$ 是推断模型"开卷"得到的结果——它看了真实观测 $o_t$，知道世界实际变成了什么样；$p_\theta(s_t \mid s_{t-1}, a_{t-1})$ 是动力学模型"闭卷"得到的预测——它只凭上一步状态和动作，靠自身学到的规律来推演。KL 散度衡量两者之间的差距：差距大说明动力学模型的预测不准，差距小说明它仅靠想象就能准确预测状态转移。训练的目标是让"闭卷"的预测逼近"开卷"的答案。

两项之间存在张力。重建项要求 $s_t$ 编码尽可能多的细节——因为少编码任何信息都会导致还原观测时出错。KL 项则要求 $s_t$ 只能包含动力学模型从 $s_{t-1}$ 和 $a_{t-1}$ 就能预测到的信息——因为动力学模型看不到当前观测，任何它预测不到的内容（比如画面中随机飞过的一只鸟、传感器噪声）一旦被编码进 $s_t$，就会拉大两个分布的距离，推高 KL 代价。平衡的结果是：$s_t$ 丢弃不可预测的噪声和偶发细节，只保留从历史可推导、对解释观测又必要的结构性信息。KL 项在这里充当了信息瓶颈，让世界模型学到的不是逐像素记忆，而是世界的运行规律。

实际优化 ELBO 时，需要解决一个核心问题：如何让动力学模型捕捉可预测的物理规律，同时保留环境的随机性？以下三个技巧在后续的 Dreamer 系列中逐步发展出来：

**RSSM：确定性与随机性的拆分。** PlaNet（Hafner et al., 2019, arXiv:1811.04551）提出的 Recurrent State-Space Model 将隐状态拆为 $s_t = (h_t, z_t)$：确定性部分 $h_t$ 由 GRU 递推维护，捕捉长期时序依赖和可预测的物理规律；随机性部分 $z_t$ 从学到的分布中采样，表达环境的不确定性和多模态结果。动力学模型的推演变为 $h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})$（确定性递推）加上 $z_t \sim p_\theta(z_t \mid h_t)$（随机采样）。

**KL Balancing：非对称优化。** Dreamer v2（Hafner et al., 2021, arXiv:2010.02193）发现标准 ELBO 中对称的 KL 梯度容易导致后验坍缩——随机部分 $z_t$ 被压成和先验一样，丢失有用信息。解决方法是将 KL 拆为两个不对称的损失：

$$
\mathcal{L}_{\text{dyn}} = \text{KL}[\text{sg}(q_\phi) \| p_\theta], \quad \mathcal{L}_{\text{rep}} = \text{KL}[q_\phi \| \text{sg}(p_\theta)]
$$

前者主要训练动力学模型的先验去追后验（让"闭卷预测"更准），后者轻微约束后验去靠先验（让表征更平滑），其中 $\text{sg}$ 表示 stop-gradient。通常给动力学损失更大权重（如 0.8），避免推断模型被迫"装傻"。

**Free Bits：保底随机性。** Kingma et al.（2016, arXiv:1606.04934）提出的 free bits 技巧为 KL 设定最低阈值：

$$
\mathcal{L}_{\text{KL}} = \max(\lambda, \text{KL}[\cdot])
$$

低于阈值 $\lambda$ 时不施加梯度。Dreamer v2/v3 中取 $\lambda = 1$ nat，保证模型至少保留一定量的随机信息，不会把 $z_t$ 优化到完全确定。



## 经典世界模型：Ha & Schmidhuber (2018)

![World Models V-M-C Architecture](/chengYi-xun/img/world_model_arch.png)

2018 年，David Ha 和 Jürgen Schmidhuber 发表了论文 World Models（arXiv:1803.10122），首次将世界模型这个概念在深度学习中系统化。核心思路是：人类面对复杂环境时不会直接在原始感官信号上做决策，而是先在脑中建立一个压缩的世界模型，然后在模型内部"想象"不同方案的后果。这篇论文用三个模块复现了这个过程：

**V 模型（Vision，压缩感知）**：对应人眼到视觉皮层的处理。原始观测 $o_t$（如一帧游戏画面）维度太高，无法直接用于预测。V 模型使用变分自编码器（VAE）将其压缩为低维潜变量 $z_t = \text{Encoder}_\phi(o_t) \in \mathbb{R}^{32}$。这一步把 12,288 维的像素降到 32 维，只保留对决策有用的结构性信息。

**M 模型（Memory，动力学预测）**：对应大脑对未来的预测能力。M 模型在潜在空间中建模"如果当前状态是 $z_t$、我执行动作 $a_t$，下一个状态 $z_{t+1}$ 会是什么分布"。它使用 LSTM 维护一个记忆状态 $h_t$，并输出一个混合密度网络（MDN）来表达多模态的未来可能性：

$$
\begin{aligned}
h_t &= \text{LSTM}(h_{t-1}, [z_t, a_t]) \\
P(z_{t+1} \mid h_t) &= \sum_{i=1}^{K} \pi_i \cdot \mathcal{N}(z_{t+1} \mid \mu_i(h_t), \sigma_i^2(h_t))
\end{aligned}
$$

这里 $K$ 个高斯分量允许模型表达"球可能往左飞也可能往右飞"这类多模态不确定性，而不是只给出一个均值预测。

**C 模型（Controller，控制器）**：对应最终的决策输出。一个极简的线性控制器，直接从潜变量和记忆状态输出动作 $a_t = W_c [z_t; h_t] + b_c$。之所以用线性模型而不是深度网络，是因为 V 和 M 已经把复杂性都吸收了——好的表征让简单的策略就够用。

三个模块的数据流为：$o_t \xrightarrow{V} z_t \xrightarrow{M} h_t, \hat{z}_{t+1} \xrightarrow{C} a_t$。V 负责"看"，M 负责"想"，C 负责"做"。

这篇论文最引人注目的实验是**在梦中训练**：先用真实环境数据训练好 V 和 M 模型，然后完全在 M 模型生成的"梦境"轨迹中训练 C 控制器（不再与真实环境交互），最后将训练好的控制器直接迁移到真实环境——在 VizDoom 赛道上成功通关。这证明了世界模型的核心价值：一旦学会了世界的运行规律，就可以在想象中无限练习。



## 潜在空间：为什么不在像素空间建模？

为什么不直接在像素空间预测下一帧？有两个直观的原因。

**维度太高，学不动。** 一帧 $64 \times 64 \times 3$ 的图像有 12,288 个数字。预测下一帧意味着同时预测这 12,288 个数字各自的值——大多数数字描述的是纹理、光影等与"接下来会发生什么"无关的细节。潜在空间 $z \in \mathbb{R}^{32}$ 将维度从 12,288 压缩到 32，只保留与动力学相关的信息（物体位置、速度、朝向等），压缩掉与预测无关的噪声。这就像：你跟朋友描述一个路口的交通状况，不会逐像素描述摄像头画面，而是说"前面有一辆白色 SUV 在左转"——这就是压缩到语义空间。

**误差会滚雪球。** 世界模型需要做多步预测（想象未来 10 步、50 步），每一步的预测误差都会传递到下一步。在像素空间中，哪怕每步只有微小误差，经过几十步累积后画面就会变得模糊甚至崩溃。在低维潜空间中，因为只有几十个有意义的维度，每步的误差更小、累积更慢，长程预测更稳定。



## 领域全景：技术路线总览

截至 2026 年，世界模型研究已形成八大技术路线：

1. Model-based RL：在潜空间想象并训练策略（代表作：Dreamer v3, TD-MPC2）。
2. JEPA：在嵌入空间预测语义，不生成像素（代表作：V-JEPA 2）。
3. 视频生成 WM：生成高保真未来视频（代表作：Sora, Cosmos, Genie 3）。
4. 物理化生成：嵌入物理定律到生成过程（代表作：PhysDreamer, PSIVG）。
5. 自动驾驶 WM：领域特化的驾驶场景预测（代表作：GAIA-1, Vista, OccWorld）。
6. 3D/空间智能：生成和模拟 3D 世界（代表作：HY-World, PointWorld）。
7. LLM-as-WM：语言模型作为世界模拟器（代表作：RAP, LAW 框架）。
8. 中国工业界：大规模工程化世界模型（代表作：混元, WorldVLA）。



## 路线一：Model-based RL——在想象中训练策略

Model-based RL 是世界模型最正统的应用：先学一个环境模型，然后在模型内部做梦来训练策略，从而大幅减少与真实环境的交互次数。

Dreamer 系列（Hafner et al., 2020-2023）是 Model-based RL 中最成功的世界模型家族。其核心架构 RSSM 和 ELBO 优化技巧（KL Balancing、Free Bits）已在前文数学定义部分详细介绍。这里聚焦 Dreamer 系列的演进和 DreamerV3 特有的工程贡献。

RSSM 在 Dreamer 中的具体实现包含三条路径：

- 确定性路径: $h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1})$
- 随机先验: $\hat{z}_t \sim p_\phi(\hat{z}_t \mid h_t)$
- 随机后验: $z_t \sim q_\phi(z_t \mid h_t, o_t)$

DreamerV3（2023）在 RSSM + KL Balancing + Free Bits 的基础上，进一步引入了两个让模型跨任务泛化的关键技巧，使其能使用同一套固定超参数横扫 150 多个任务：

**Symlog 变换**：不同环境的奖励和观测尺度可能相差数个数量级（Atari 游戏得分从 1 到 100,000 不等）。DreamerV3 引入对称对数变换 $\text{symlog}(x) = \text{sign}(x) \ln(|x|+1)$，将大数值压缩到对数尺度，同时在原点附近保持近似线性，使得单一的均方误差损失就能拟合极其多样的目标分布。

**Two-hot 编码**：将连续值离散化为 $K$ 个桶，目标值映射为相邻两个桶的线性组合，将回归问题转化为分类问题。这不仅提供了更丰富的梯度信号，还能自然地表达价值函数的多模态不确定性。



## 路线二：JEPA——不生成像素的世界模型

2022 年，Yann LeCun 提出了 JEPA（Joint-Embedding Predictive Architecture）。他的核心论点是：世界模型不应该生成像素。生成模型试图预测世界的每一个细节，但这些细节大多是本质上不可预测且与决策无关的。

JEPA 的核心架构包含上下文编码器 $s_x = f_\theta(x)$、目标编码器 $\bar{s}_y = \bar{f}_{\bar{\theta}}(y)$ 和预测器 $\hat{s}_y = g_\psi(s_x, c)$。训练目标是最小化嵌入空间中的预测误差 $\mathcal{L}_{\text{JEPA}} = \|\bar{s}_y - \hat{s}_y\|^2$。

JEPA 理论的核心挑战是表示坍塌（Representation Collapse）。如果不对编码器加以限制，网络可以通过将所有输入映射为零向量来轻易地使预测误差降为零。为了防止这种灾难性退化，JEPA 采用了非对称的优化策略：目标编码器的参数 $\bar{\theta}$ 不接受梯度更新，而是作为上下文编码器参数 $\theta$ 的指数移动平均（EMA）进行缓慢更新：$\bar{\theta} \leftarrow \tau \bar{\theta} + (1-\tau)\theta$。这种动量更新机制在理论上打破了对称性，迫使网络学习到具有丰富语义的嵌入空间。

**JEPA 与传统世界模型的关键区别。** 传统世界模型（如 Dreamer）回答"给定动作 $a_t$，会**看到**什么观测 $o_{t+1}$"——需要观测模型（解码器）生成像素，需要动作标签，在 ELBO 框架下训练。JEPA 回答"接下来世界的**语义**会怎么变"——在嵌入空间预测表征，没有解码器，通常也不需要动作标签（从无动作标注的互联网视频中自监督学习）。但本质目标相同：两者都在学习世界的动力学——"世界如何随时间演变"，区别在于工作空间（像素/观测空间 vs 嵌入空间）和是否需要动作条件。当 JEPA 被应用到机器人控制时（如 V-JEPA 2），才会加入动作条件。

V-JEPA 2（Assran et al., 2025）是当前 JEPA 路线的集大成者，拥有 1.2B 参数，在 100 万小时互联网视频上训练，实现了运动理解和人类动作预测的 SOTA，并在机器人应用上实现了零样本抓取。

Var-JEPA（2026）提出了 JEPA 的变分公式，从数学上证明了：标准 JEPA 是耦合潜变量模型上变分推断的确定性特例。

$$
\text{标准 JEPA}: \; \hat{s}_y = g_\psi(s_x) \;\;\Longleftrightarrow\;\; \lim_{\sigma \to 0} \; \text{ELBO}\left[q_\phi(s_y \mid s_x), \; p_\theta(s_y)\right]
$$

这意味着 JEPA（预测派）和 VAE/扩散模型（生成派）在数学结构上并非水火不容——它们是同一框架在不同极限下的特例。当观测噪声的方差 $\sigma$ 趋于零时，变分推断的随机性消失，退化为 JEPA 的确定性嵌入预测。



## 路线三：视频生成世界模型——生成即理解

与 JEPA 的不生成像素相反，视频生成路线的核心信念是：如果模型能完美模拟物理世界的视觉变化，它就必然在内部学到了物理规律。这就是生成即理解假说。

OpenAI 的 Sora 是一个 DiT（Diffusion Transformer），在视频的时空 patch 上工作。其训练目标是标准的扩散去噪损失：

$$
\mathcal{L}_{\text{Sora}} = \mathbb{E}_{x_0, \epsilon, t}\left[\Vert\epsilon - \epsilon_\theta(x_t, t, c_{\text{text}})\Vert^2\right]
$$

Sora 架构的理论突破在于将 Transformer 的缩放定律（Scaling Laws）引入了扩散模型。传统的 U-Net 架构在处理高维时空数据时存在固有的归纳偏置局限。Sora 将视频视为三维时空体，并将其切分为不重叠的时空 Patch（类似于 ViT 中的图像块）。这种 Patch 化表示使得模型能够统一处理任意分辨率、任意长宽比和任意时长的视频数据。在去噪过程中，DiT 通过自适应层归一化（adaLN）将时间步 $t$ 和文本条件 $c_{\text{text}}$ 注入到每一个 Transformer 块中，从而实现了对生成过程的精确控制。

Google DeepMind 的 Genie 3（2025）实现了实时 24fps 的交互世界生成。Genie 的理论创新在于潜在动作模型（Latent Action Model）。它能够在完全没有动作标签的互联网视频上进行无监督学习，通过自回归动力学模型推断出视频帧之间的潜在动作空间。这使得生成的世界不仅是视觉上连贯的，而且是可以通过离散动作进行交互控制的。

NVIDIA 的 Cosmos（2025）则提供了一整套物理 AI 平台，其核心是基于连续时间流匹配（Flow Matching）的世界生成模型，进一步提升了生成物理世界的稳定性和保真度。



## 路线四：物理化世界模型——如何将物理定律注入视频生成？

Sora 等纯数据驱动的视频生成模型虽然能生成惊艳的视觉效果，但仔细观察会发现：物体有时会穿模、重力方向不一致、碰撞后反应不合理。2025 年的一项系统性研究《How Far is Video Generation from World Model》指出：**由于视觉数据的内生歧义性（Visual Ambiguity），纯靠视觉观测是无法让模型自发学会牛顿力学等精确物理定律的**。模型学到的只是视觉上的"相关性"，而非物理上的"因果性"。

为了让生成的视频真正符合物理规律，2025-2026 年的研究集中在**如何将物理定律作为归纳偏置注入到生成过程中**。目前主要有四种核心策略：

## 1. 物理模拟器在环（Simulator-in-the-Loop）

这种策略直接将成熟的物理引擎（如 MuJoCo, Taichi）接入扩散模型的生成管线。代表作 **PSIVG**（Physical Simulator In-the-Loop Video Generation, 2026）的流程如下：

1. **状态重建**：从初始视频帧中重建出 4D 场景网格（Mesh）和物体的初始物理状态（位置、速度、质量）。
2. **正向模拟**：在真实的物理引擎中运行这些状态，计算出严格符合牛顿定律的未来轨迹。
3. **条件引导**：将物理引擎输出的精确轨迹作为条件（Conditioning），注入到视频扩散模型的时间注意力层中，强制生成的像素跟随物理轨迹移动。

## 2. 物理感知的训练目标（Physics-Informed Objectives）

不改变模型架构，而是在训练损失函数中加入物理约束。

- **频域正则化**：2026 年的研究发现，刚体运动（平移、旋转）在频域具有特定的谱特征。通过在扩散模型的训练中加入频域的辅助损失（Physics-Guided Motion Loss），可以显著减少生成视频中的"橡胶般形变"（Rubber-sheet deformations）和穿模现象。
- **大模型规则校验**：**DiffPhy**（2025）利用多模态大语言模型（MLLM）作为"物理裁判"。在扩散去噪的中间步骤，MLLM 会检查当前的潜变量是否违反了常识物理规则（如重力方向错误），并产生一个惩罚梯度（Phenomena Loss）来引导去噪方向。

## 3. 基于强化学习的物理对齐（RL & DPO）

借鉴大语言模型的 RLHF，研究者开始用强化学习来对齐视频模型的物理常识。
**PhysMaster**（2025）引入了一个专门的"物理编码器"（PhysEncoder）来提取图像中的物理表征。为了让这个编码器真正理解物理，作者使用了**直接偏好优化（DPO）**：构建包含"符合物理"和"违反物理"的视频对，强制模型在生成时最大化符合物理轨迹的似然，从而将物理常识内化到模型的权重中。

## 4. 物理参数蒸馏（Distillation via MPM）

**PhysDreamer**（2024）提出了一种优雅的逆向工程方法。它使用**物质点方法（Material Point Method, MPM）**来模拟物体的弹性形变。MPM 的控制方程是连续介质的动量守恒偏微分方程：

$$
\rho \frac{D\mathbf{v}}{Dt} = \nabla \cdot \boldsymbol{\sigma} + \mathbf{f}_{\text{ext}}
$$

PhysDreamer 将材料的本构关系（即应力 $\boldsymbol{\sigma}$ 与形变梯度的映射）参数化，并利用视频扩散模型（如 Zero-1-to-3）提供的分数蒸馏采样（SDS）梯度来监督这些物理参数的优化。这样，模型不仅生成了视觉上合理的视频，而且其内部的运动完全遵循牛顿力学定律。



## 路线五：自动驾驶世界模型

自动驾驶是世界模型最重要的应用领域之一。真实驾驶数据昂贵且危险，而世界模型可以在虚拟中生成无限的驾驶场景，特别是难以在现实中采集的长尾场景。

GAIA-1（Wayve, 2023）将驾驶场景预测建模为下一个 token 预测问题。OccWorld（2024）提出了一种独特的预测空间：3D 语义占用栅格。不预测 RGB 图像，而是预测这个空间点在未来会被什么物体占据：$\hat{\mathcal{O}}_{t+1} \in \{0, 1, \ldots, C\}^{X \times Y \times Z}$。这种表示天然适合自动驾驶的规划：不需要理解像素，只需要知道哪里有障碍物。

最新的 Drive-JEPA（2026）将 JEPA 架构引入自动驾驶，结合多模态轨迹蒸馏，实现端到端驾驶。



## 路线六：3D 世界与空间智能

2024 年，斯坦福大学的李飞飞发表了关于空间智能的框架性文章，并创立了 World Labs 公司。她指出，当前的 LLM 就像黑暗中的文字匠——它们理解语言，但不理解空间。

World Labs 推出的 Marble 平台能从文本、图像、视频或粗糙 3D 布局生成语义、物理和几何一致的 3D 世界。PointWorld（2025）将状态和动作统一在共享 3D 空间中，使用 3D 点流表示动力学。

腾讯混元团队开发了 HY-World 系列 3D 世界模型。HY-World 2.0（2026.04）是多模态 3D 生成模型，并且开源，在 Stanford WorldScore 排行榜位列第一。它包含全景生成、轨迹规划、世界扩展和 3D 重建四个模块。



## 路线七：LLM 作为世界模型

大语言模型在预测下一个 token 的过程中，是否隐式地构建了某种世界模型？一方面，LLM 在训练中接触了海量的关于世界运作方式的文本描述；另一方面，LLM 缺乏交互式预测能力。LLM 学习的是语言的结构，而世界模型学习的是因果的结构。二者互补但不同。

RAP（2023）提出让 LLM 同时扮演世界模型和推理代理。LLM 作为世界模型预测下一个状态，作为推理代理评估各状态的价值，并结合蒙特卡洛树搜索在 LLM 的想象中探索推理路径。

Emu3（2025）将语言、图像和动作统一为 token 序列，通过自回归预测实现多模态世界交互——本质上是将下一 token 预测扩展为下一世界状态预测。



## 路线八：中国工业界的世界模型

2025-2026 年，中国 AI 产业在世界模型领域的竞争进入白热化。

阿里巴巴推出了 WorldVLA（2025.06），统一 Vision-Language-Action 与世界模型，通过预测未来状态学习物理规律。WorldVLA 的核心理论洞察是世界模型和动作模型是互益的。阿里还推出了 Happy Oyster（2026.04，游戏/影视交互世界模型）和 RynnBrain（2026.02，开源具身基础模型）。

腾讯混元推出了 HY-World 2.0（2026.04，开源 3D 世界模型）和 Hunyuan-GameCraft-2。



## 评估与辩论：生成 vs 预测

评估世界模型是一个尚未解决的难题。传统的图像/视频质量指标无法衡量物理理解。WorldScore 基准（2025）是目前最全面的世界生成评估基准，从可控性、质量、动态性三个维度评估。Meta 的 V-JEPA 2 配套了 IntPhys 2、MVPBench 和 CausalVQA 等物理推理基准。

世界模型领域最根本的哲学分歧是：世界模型是否需要看得见（重建像素）？

生成派（Sora, Cosmos）认为生成即理解——能完美重建就意味着完美理解。其优势是输出直观可解释，可以直接用于内容创作。预测派（JEPA, Dreamer）认为无需重建每个像素——决策只需要抽象表示。其优势是计算效率高，避免预测无关细节。

2026 年的 Var-JEPA 从数学上证明：标准 JEPA 可以被视为应用于耦合潜变量模型的变分推断的确定性特例。这意味着生成派和预测派在数学上是同一框架的不同特例。



## 训练范式与开放问题

无论哪条技术路线，世界模型的训练通常遵循类似的两阶段范式。第一阶段从环境交互数据中学习动力学；第二阶段在学到的世界模型中想象未来，训练策略或做规划。与 Model-free RL 相比，Model-based RL 样本效率高，迁移能力强，但可能受限于模型误差和多步想象的误差累积。

未来的开放问题包括：
1. 长期一致性与记忆：当前的世界模型在长时间跨度上仍然表现不佳。
2. 因果推理 vs 相关性模拟：当前的世界模型可能只是在做相关性匹配而非真正的因果推理（ACM CSUR 2025 综述指出）。
3. 评估标准的统一：不同路线使用完全不同的评估体系，难以跨路线比较。
4. Sim2Real Gap：从虚拟到现实的迁移仍然是瓶颈。
5. 世界模型的缩放定律：更大的模型是否一定能学到更好的物理，系统性的研究仍然缺乏。



## 总结

本文从最直觉的类比出发，梳理了世界模型的数学定义（前向动力学、观测模型、推断模型、ELBO），回顾了经典架构（VAE + MDN-RNN），并详细拆解了 2026 年的八大技术路线：Model-based RL、JEPA、视频生成、物理化、自动驾驶、3D 世界、LLM-as-WM 和中国工业界。我们探讨了生成派与预测派的哲学之争，并展望了未来的开放问题。

下一篇将深入 Dreamer 系列——目前 Model-based RL 方向最成功的世界模型家族。从 RSSM 的数学推导开始，一路讲到 DreamerV3 如何用固定超参横扫 150+ 个任务。



> 参考文献：
>
> 经典与基础
> 1. Craik, K. (1943). The Nature of Explanation. Cambridge University Press.
> 2. Shannon, C. E. (1959). Coding Theorems for a Discrete Source with a Fidelity Criterion. IRE National Convention Record.
> 3. Ha, D. & Schmidhuber, J. (2018). World Models. arXiv:1803.10122.
>
> 综述论文
> 4. Ding, Z. et al. (2025). Understanding World or Predicting Future? A Comprehensive Survey of World Models. ACM Computing Surveys. arXiv:2411.14499.
> 5. Li, X. et al. (2025). A Comprehensive Survey on World Models for Embodied AI. arXiv:2510.16732.
> 6. Shang, J. et al. (2025). A Survey of Embodied World Models. ResearchGate.
> 7. Yue, Y. et al. (2025). Simulating the Visual World with Artificial Intelligence: A Roadmap. arXiv:2511.08585.
>
> Model-based RL
> 8. Hafner, D. et al. (2020). Dream to Control: Learning Behaviors by Latent Imagination. ICLR 2020.
> 9. Hafner, D. et al. (2021). Mastering Atari with Discrete World Models. ICLR 2021.
> 10. Hafner, D. et al. (2023). Mastering Diverse Domains through World Models. arXiv:2301.04104.
> 11. Wu, P. et al. (2022). DayDreamer: World Models for Physical Robot Learning. arXiv:2206.14176.
> 12. Micheli, V. et al. (2023). Transformers are Sample-Efficient World Models. ICLR 2023. arXiv:2209.00588.
> 13. Hansen, N. et al. (2024). TD-MPC2: Scalable, Robust World Models for Continuous Control. ICLR 2024.
> 14. Chen, C. et al. (2022). TransDreamer: Reinforcement Learning with Transformer World Models. arXiv:2202.09481.
>
> JEPA 系列
> 15. LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. OpenReview.
> 16. Assran, M. et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. CVPR 2023. arXiv:2301.08243.
> 17. Bardes, A. et al. (2024). V-JEPA: Video Joint Embedding Predictive Architecture. Meta AI.
> 18. Assran, M. et al. (2025). V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. arXiv:2506.09985.
> 19. (2025). LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels. arXiv:2603.19312.
> 20. (2026). Var-JEPA: A Variational Formulation of the Joint-Embedding Predictive Architecture. arXiv:2603.20111.
>
> 视频生成世界模型
> 21. OpenAI (2024). Video Generation Models as World Simulators. Technical Report.
> 22. Bruce, J. et al. (2024). Genie: Generative Interactive Environments. DeepMind.
> 23. DeepMind (2024). Genie 2: A Large-Scale Foundation World Model. Blog post.
> 24. NVIDIA (2025). World Simulation with Video Foundation Models for Physical AI. arXiv:2511.00062.
> 25. Du, Y. et al. (2023). Learning Universal Policies via Text-Guided Video Generation. NeurIPS 2023.
>
> 物理化世界模型
> 26. Zhang, T. et al. (2024). PhysDreamer: Physics-Based Interaction with 3D Objects via Video Generation. ECCV 2024. arXiv:2404.13026.
> 27. (2025). AI-Newton: A Concept-Driven Physical Law Discovery System without Prior Physical Knowledge. arXiv:2504.01538.
> 28. (2025). Think Before You Diffuse: Infusing Physical Rules into Video Diffusion (DiffPhy). arXiv:2505.21653.
> 29. (2026). Physical Simulator In-the-Loop Video Generation (PSIVG). arXiv:2603.06408.
> 30. (2025). PhysMaster: Mastering Physical Representation for Video Generation via Reinforcement Learning.
> 31. (2025). How Far is Video Generation from World Model: A Physical Law Perspective.
>
> 自动驾驶世界模型
> 32. Hu, A. et al. (2023). GAIA-1: A Generative World Model for Autonomous Driving. arXiv:2309.17080.
> 33. Wang, X. et al. (2023). DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving. arXiv:2309.09777.
> 34. Gao, S. et al. (2024). Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability. NeurIPS 2024. arXiv:2405.17398.
> 35. Zheng, W. et al. (2024). OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving. ECCV 2024.
>
> 3D/空间智能
> 36. Li, F.-F. (2024). From Words to Worlds: Spatial Intelligence is AI's Next Frontier. Substack.
> 37. (2025). PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation. arXiv:2601.03782.
> 38. Tencent (2026). HY-World 2.0: A Multi-Modal World Model for Reconstructing, Generating, and Simulating 3D Worlds.
> 39. Tencent (2025). Hunyuan-GameCraft-2: Instruction-following Interactive Game World Model.
>
> LLM 与世界模型
> 40. Hao, S. et al. (2023). Reasoning with Language Model is Planning with World Model. EMNLP 2023. arXiv:2305.14992.
> 41. Guan, L. et al. (2023). Language Models, Agent Models, and World Models: The LAW for Machine Reasoning and Planning. arXiv:2312.05230.
>
> 中国工业界
> 42. (2025). WorldVLA: Towards Autoregressive Action World Model. arXiv:2506.21539.
> 43. Alibaba (2026). RynnBrain: Open-sourced Embodied Foundation Model for Robotics.
>
> 评估
> 44. Duan, H. et al. (2025). WorldScore: A Unified Evaluation Benchmark for World Generation. Stanford / ICCV 2025.
>
> 哲学与反思
> 45. (2024). Sora and V-JEPA Have Not Learned The Complete Real World Model — A Philosophical Analysis of Video AIs. arXiv.

> 下一篇：[笔记｜世界模型（二）：Dreamer 系列——在想象中学习控制](/chengYi-xun/posts/27-dreamer/)
