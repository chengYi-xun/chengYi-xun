---
title: 笔记｜世界模型（二）：Dreamer 系列——在想象中学习控制
date: 2026-04-06 00:35:00
categories:
 - Tutorials
tags:
 - Dreamer
 - RSSM
 - Model-based RL
 - World Model
series: "世界模型"
mathjax: true
---

> **核心论文**：Dreamer v1 (arXiv:1912.01603, ICLR 2020)、DreamerV2 (arXiv:2010.02193, ICLR 2021)、DreamerV3 (arXiv:2301.04104, Nature 2025)
>
> **代码**：[danijar/dreamerv3](https://github.com/danijar/dreamerv3) | **前置知识**：[上一篇：世界模型基础](/chengYi-xun/posts/26-world-model-basics/)
>
> ⬅️ 上一篇：[笔记｜世界模型（一）：什么是世界模型？从认知科学到深度学习](/chengYi-xun/posts/26-world-model-basics/)
>
> ➡️ 下一篇：[笔记｜世界模型（三）：JEPA——在嵌入空间预测世界](/chengYi-xun/posts/28-jepa/)

## 0. 在 Minecraft 中不靠人类示范采到钻石

想象你在打 Minecraft——需要砍树 → 合成工作台 → 挖石头 → 合成石镐 → 找到并挖掘钻石。整个过程长达 **36,000 步**（约 30 分钟），中间没有任何奖励反馈，只有最终挖到钻石时才得一分。这是 RL 界公认的"登月挑战"。

如果用传统的 model-free RL（如 PPO），智能体必须在**真实环境**中反复试错。由于奖励极其稀疏，绝大多数尝试一无所获，需要几亿步交互才有可能学到有意义的策略。

DreamerV3 换了一种思路：**先学一个环境的"脑内模型"，然后在脑子里大量练习**。它是首个在**不使用人类示范、不依赖自适应课程**的前提下，从零开始在 Minecraft 中稳定采到钻石的算法；更惊人的是，它与 Atari、DeepMind Control Suite 等完全不同的域共用**同一套默认超参数**（Hafner et al., Nature 2025 / arXiv:2301.04104）。

本文将拆解 Dreamer 系列的三代演进，解释每一代解决了什么问题、为什么这样设计、核心代码如何实现。如果你读过[上一篇的强化学习基础](/chengYi-xun/posts/17-rl-basics/)，你已经知道 Actor-Critic 和 $\lambda$-return；Dreamer 的核心贡献在于**Actor-Critic 不在真实环境中训练，而是在学到的世界模型中"想象"着训练**。

## 1. RSSM：Dreamer 的心脏

### 1.1 从"录像带回放"到"脑内模拟器"

上一篇介绍了世界模型的基本思路：学一个 $s_{t+1} = f(s_t, a_t)$ 的环境动力学。但最朴素的方法——用一个 RNN 记住历史并预测下一帧——有一个致命的限制：**它的隐状态 $h_t$ 是纯确定性的**。

为什么这是个问题？考虑一个简单的场景：你在十字路口，前方的行人可能**左转**也可能**右转**。这两种未来同样合理，但确定性模型只能输出一个预测——它会输出两种可能性的"平均"，得到一个模糊的、哪边都不像的预测（比如行人"同时"往两边走的重影）。

**真实世界是随机的**——同样的状态和动作，可能导致不同的结果。我们需要一个**既能记住确定性的历史规律、又能表达未来不确定性**的状态表示。

这就是 **RSSM（Recurrent State-Space Model）** 的核心思想：将隐状态拆成**确定性**和**随机性**两部分。

$$
\text{完整状态} = (\underbrace{h_t}_{\text{确定性：记住历史规律}}, \underbrace{z_t}_{\text{随机性：表达未来不确定性}})
$$

- **确定性部分 $h_t$**（用 GRU 实现）：像一个"笔记本"，记录到目前为止发生的所有确定性信息——我在哪里、做过什么动作、看到过什么。这部分不会因为随机性而改变。
- **随机部分 $z_t$**（从分布中采样）：像一个"骰子"，表示当前时刻的随机因素——行人选择了左转还是右转、硬币落在了正面还是反面。

这种设计的妙处在于：确定性路径提供了稳定的长程记忆（GRU 擅长此事），随机路径允许模型表达多模态的未来（同一个 $h_t$ 可以采样出不同的 $z_t$，对应不同的可能未来）。

![RSSM 架构：确定性路径与随机路径并行建模状态转移（Hafner et al., ICML 2019）](/chengYi-xun/img/rssm_architecture.png)

### 1.2 RSSM 的四个方程

RSSM 由四个紧密配合的组件构成：

$$
\boxed{
\begin{aligned}
\text{序列模型:} \quad & h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1}) \\
\text{编码器（后验）:} \quad & z_t \sim q_\theta(z_t \mid h_t, o_t) \\
\text{先验:} \quad & \hat{z}_t \sim p_\theta(\hat{z}_t \mid h_t) \\
\text{解码器:} \quad & \hat{o}_t \sim p_\theta(o_t \mid h_t, z_t)
\end{aligned}
}
$$

逐项拆解：

**序列模型** $h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})$（确定性路径）：一个 GRU 网络，输入是上一时刻的完整状态 $(h_{t-1}, z_{t-1})$ 和执行的动作 $a_{t-1}$，输出新的确定性状态 $h_t$。它的作用类似于你大脑中的"经验记忆"——你知道自己之前做了什么、看到了什么，这些确定性的历史信息被压缩到 $h_t$ 中。

**编码器（后验）** $z_t \sim q_\theta(z_t \mid h_t, o_t)$：在**训练时**，模型可以"偷看"真实观测 $o_t$，所以能推断出更准确的随机状态 $z_t$。这相当于你在**事后**才知道行人选择了左转——有了这个信息，你对当前状态的判断就非常准确。之所以叫"后验"，正是因为它利用了观测 $o_t$ 这个"事后证据"。

**先验** $\hat{z}_t \sim p_\theta(\hat{z}_t \mid h_t)$：在**想象阶段**（脑内模拟），模型无法访问真实观测 $o_t$，只能根据确定性状态 $h_t$ 来"猜"随机状态。这相当于你**事前**预测行人会往哪边走——没有观测信息，只能靠过往经验推测。

**解码器** $\hat{o}_t \sim p_\theta(o_t \mid h_t, z_t)$：从完整状态 $(h_t, z_t)$ 重建观测。它的作用是**训练信号**——如果重建出来的图像和真实观测相差甚远，说明状态表示没有捕获足够的信息。

**四个组件如何配合？** 训练时，编码器利用真实观测给出准确的后验 $q(z_t)$；同时先验 $p(z_t)$ 也在尝试预测。通过 KL 散度迫使先验逼近后验——当先验学得足够准时，即使在想象阶段看不到真实观测，先验的预测也足够可靠。这就是**"先开卷考试学会知识，然后闭卷也能答对"**的思路。

### 1.3 训练目标：ELBO

**核心问题**：我们有一堆从真实环境收集的轨迹数据 $(o_1, a_1, o_2, a_2, \ldots)$，如何训练 RSSM 的四个组件？

直觉上，一个好的世界模型应该能**解释**看到的观测序列：给定执行的动作 $a_{1:T}$，观测 $o_{1:T}$ 出现的概率应该尽可能大。数学上就是最大化条件似然 $\log p(o_{1:T} \mid a_{1:T})$。

但直接最大化这个似然是不可行的——因为它需要对隐变量 $z_{1:T}$ 求积分（把所有可能的隐状态都考虑一遍），计算量是指数级的。变分推断的标准做法是优化它的一个**可计算的下界**——ELBO（Evidence Lower Bound，证据下界）。关于 ELBO 的严格推导，可参考 [VAE 原始论文 (Kingma & Welling, 2014)](https://arxiv.org/abs/1312.6114) 和 [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)。

$$
\log p(o_{1:T} \mid a_{1:T}) \geq \sum_{t=1}^{T} \left[\underbrace{\mathbb{E}_{q_\theta(z_t|h_t, o_t)}[\log p_\theta(o_t \mid h_t, z_t)]}_{\text{重建项}} - \underbrace{\text{KL}[q_\theta(z_t \mid h_t, o_t) \| p_\theta(z_t \mid h_t)]}_{\text{KL 项}}\right]
$$

**逐项拆解**：

| 组成部分 | 写法 | 含义 |
|:---|:---|:---|
| 重建项 | $\mathbb{E}_{q}[\log p_\theta(o_t \mid h_t, z_t)]$ | 从后验采样出的状态 $(h_t, z_t)$，能多好地重建观测 $o_t$？越大越好——说明状态表示保留了足够的信息 |
| KL 项 | $\text{KL}[q_\theta(z_t \mid h_t, o_t) \| p_\theta(z_t \mid h_t)]$ | 后验（看到观测后的状态估计）和先验（没看到观测时的预测）差多远？越小越好——说明先验已经学会了准确预测 |

**用面试的类比理解两个项的张力**：重建项就像面试时要求你**尽可能详细地复述看到的材料**——逼着你记住所有细节。KL 项则要求你的**"猜测"和"看过答案后的判断"尽量一致**——逼着你在看答案之前就能猜对。这两者之间存在张力：为了完美重建，你想记住每一个像素；但 KL 约束说"你的猜测能力要跟上"，迫使状态表示**只保留可预测的、有规律的信息**，丢弃随机噪声。这就是信息瓶颈（Information Bottleneck）的效果。

加上**奖励预测**（世界模型还需要预测每一步能获得多少奖励），完整的训练损失为：

$$
\mathcal{L} = \sum_t \left[\underbrace{-\log p_\theta(o_t \mid h_t, z_t)}_{\text{重建观测}} \underbrace{- \log p_\theta(r_t \mid h_t, z_t)}_{\text{预测奖励}} + \underbrace{\beta \cdot \text{KL}[q \| p]}_{\text{先验逼近后验}}\right]
$$

其中 $\beta$ 控制 KL 约束的强度——后续 DreamerV3 对这个 $\beta$ 做了精细的分解（KL 平衡），这是 V3 的核心创新之一。

## 2. Dreamer v1：想象中的策略梯度

### 2.1 核心思想：为什么不在真实环境中训练策略？

在[上一篇 RL 基础](/chengYi-xun/posts/17-rl-basics/)中，我们介绍了 Actor-Critic 架构——Actor 选动作，Critic 评价好坏，两者交替改进。但传统的 Actor-Critic（包括 PPO）有一个根本性的瓶颈：**每一步策略梯度都需要与真实环境交互**。

在 Atari 游戏中，一次交互就是渲染一帧画面 + 执行一步动作 + 返回奖励，速度还能接受。但在 Minecraft 或真实机器人上，一次交互可能耗时数秒甚至数分钟。训练 PPO 需要几百万步交互——在真实世界中这意味着几千小时的机器人运行时间。

Dreamer v1 的核心创新：**先学一个世界模型（RSSM），然后把 Actor-Critic 搬到世界模型内部训练**。世界模型跑在 GPU 上，一步"想象"只需要矩阵乘法，比真实环境快几个数量级。

![Dreamer：从经验学习动力学，并在想象中学习行为（Figure 3, Hafner et al., ICLR 2020）](/chengYi-xun/img/dreamer_imagination.png)

Dreamer 的训练流程分为三个交替的阶段（对应上图）：

1. **经验收集**（Environment）：用当前策略在真实环境中执行动作，收集 $(o_t, a_t, r_t)$ 存入经验缓冲区
2. **世界模型学习**（Dynamics Learning）：从缓冲区中采样数据，训练 RSSM（最小化上一节的 ELBO 损失）
3. **策略学习**（Behavior Learning）：在世界模型内部"想象"出大量轨迹，用这些想象轨迹训练 Actor 和 Critic

### 2.2 想象轨迹

在世界模型学好后，Dreamer 从真实数据中采样一个初始状态 $(h_0, z_0)$，然后**完全在想象中**展开轨迹：

$$
\text{想象轨迹}: \quad (h_1, z_1, a_1, \hat{r}_1) \to (h_2, z_2, a_2, \hat{r}_2) \to \cdots \to (h_H, z_H)
$$

每一步：Actor 根据当前状态 $(h_t, z_t)$ 输出动作 $a_t$ → 序列模型更新 $h_{t+1}$ → 先验采样 $z_{t+1}$ → 奖励头预测 $\hat{r}_t$。整个过程**不与真实环境交互**，所有操作都是神经网络的前向传播。

### 2.3 策略优化：反传穿过动力学

传统 PPO 使用 REINFORCE 风格的策略梯度——把奖励当作一个"标量信号"，告诉 Actor"这个动作好/坏"，但不告诉"好在哪里/该怎么调"。这就像考试只给总分、不给每题的批改。

Dreamer v1 利用了世界模型的一个关键优势：**想象轨迹中的每一步都是可微的**（因为都是神经网络的前向传播）。这意味着可以直接计算 $\nabla_\psi V$——"策略参数 $\psi$ 如何影响想象轨迹上每个状态的价值"——梯度沿着动力学路径一路反传回来。这就是论文中所说的 **value gradient**，比 REINFORCE 的方差低得多。

$$
\max_\psi \sum_{t=1}^{H} V_\xi(h_t, z_t)
$$

其中 $V_\xi$ 是在想象轨迹上训练的 Critic 网络，采用 $\lambda$-return（与[上一篇 GAE](/chengYi-xun/posts/17-rl-basics/) 的思想一致，$\lambda$ 在 TD 偏差和蒙特卡洛方差之间权衡）：

$$
V_t^\lambda = r_t + \gamma \left[(1-\lambda) V_\xi(s_{t+1}) + \lambda V_{t+1}^\lambda\right]
$$

**与 PPO 的关键区别**：PPO 的梯度不穿过环境（环境是黑盒，不可微），只能用 REINFORCE + 裁剪。Dreamer 的梯度穿过世界模型（世界模型是可微的神经网络），因此能用更高效的反向传播梯度。这是 model-based RL 的核心优势。

## 3. DreamerV2：离散潜变量

### 3.1 Dreamer v1 的瓶颈：连续高斯分布的局限

Dreamer v1 使用连续高斯分布来建模随机状态：$z_t \sim \mathcal{N}(\mu, \sigma^2)$。这有两个问题：

**问题一：单峰困境**。高斯分布是**单峰**的——它只能表达"一种可能性的不确定程度"。但回到十字路口的例子：行人可能左转（模式 A）也可能右转（模式 B），这是一个**双峰**分布。单个高斯只能把均值放在中间（不左不右），无法忠实表达"两种截然不同的可能"。

**问题二：后验坍缩（Posterior Collapse）**。训练 ELBO 时，KL 项惩罚后验偏离先验。如果 KL 惩罚太强，后验会被"压扁"到先验上——$z_t$ 变得和先验一样无信息，所有有用的信息都被迫塞进确定性路径 $h_t$。结果：随机路径名存实亡，模型退化为纯确定性 RNN。

### 3.2 解决方案：离散分类分布

DreamerV2 把连续高斯替换为**离散分类分布**：

$$
z_t \sim \text{Categorical}(\text{logits}_\theta(h_t, o_t))
$$

具体地，$z_t$ 由 32 个**独立的**分类变量组成，每个有 32 个类别：

$$
z_t = (z_t^1, z_t^2, \ldots, z_t^{32}), \quad z_t^i \sim \text{Cat}(p_1^i, \ldots, p_{32}^i)
$$

总共 $32^{32}$ 种组合（远超连续高斯的表达力）。离散表示天然是多模态的：每个分类变量可以独立地"选择"不同的模式（行人左转 vs 右转、硬币正面 vs 反面），无需混合多个高斯。

### 3.3 Straight-Through Gradient：让离散采样可微

离散采样有一个致命的技术问题：**argmax 和采样操作不可微**——无法对"选第 3 类"这个操作求梯度。但 Dreamer 的策略训练需要梯度穿过世界模型反传，如果 $z_t$ 这一步断了，整个 value gradient 就用不了。

DreamerV2 使用 **Straight-Through** 技巧——前向传播和反向传播走不同的路：

- **前向传播**：采样离散的 one-hot 向量 $z_t^{\text{hard}}$（保证下游看到的是离散值）
- **反向传播**：用连续的 softmax 概率 $z_t^{\text{soft}} = \text{softmax}(\text{logits})$ 传递梯度

实现上用一行代码就能搞定：

$$
z_t = z_t^{\text{hard}} - \text{sg}(z_t^{\text{soft}}) + z_t^{\text{soft}}
$$

其中 $\text{sg}$ 是 stop-gradient。数学上，前向时 $z_t = z_t^{\text{hard}}$（因为 $\text{sg}$ 在前向时什么都不做，而 $z_t^{\text{hard}} - z_t^{\text{soft}} + z_t^{\text{soft}} = z_t^{\text{hard}}$）；反向时梯度绕过 $z_t^{\text{hard}}$，流过 $z_t^{\text{soft}}$（因为 $z_t^{\text{hard}}$ 和 $\text{sg}(z_t^{\text{soft}})$ 的梯度都被截断了，只剩最后一个 $z_t^{\text{soft}}$ 提供梯度）。

### 3.4 连续 vs 离散对比

| 维度 | 连续高斯 (V1) | 离散分类 (V2) |
|------|---------|---------|
| 模式覆盖 | 单峰，难以表达多模态未来 | 天然多模态（$32^{32}$ 种组合） |
| KL 优化 | 容易后验坍缩 | 更稳定（离散 KL 有上界） |
| 梯度传递 | 天然可微 | 需要 Straight-Through 技巧 |
| Atari 200M | 112% 人类中位数 | **164%** 人类中位数 |

## 4. DreamerV3：固定超参横扫一切

### 4.1 DreamerV2 的痛点：换个域就得调超参

DreamerV2 在 Atari 上表现惊艳，但换到别的领域就需要重新调参。根本原因是**不同领域的数值尺度差异巨大**：

- **奖励尺度**：Atari 的奖励是整数 0/1/10，而 DMC（机器人控制）的奖励是连续的 [0, 1000]
- **KL 尺度**：某些领域的 KL 项天然很大（观测信息丰富），另一些很小
- **回报尺度**：Minecraft 的总回报可能在 [0, 1]，而某些 Atari 游戏的总回报在 [0, 100000]

DreamerV3 的目标是**一套超参打天下**——从 Atari 到机器人控制到 Minecraft，无需任何调整。它通过三个精心设计的归一化技术实现了这一点。

### 4.2 创新一：symlog 变换——抹平奖励尺度差异

**问题**：如果直接用 MSE 损失训练奖励预测头，当奖励值为 1000 时，梯度是奖励值为 1 时的 1000 倍。不同领域的奖励尺度不同，就需要不同的学习率——这正是需要调参的根源。

**解决方案**：在预测奖励和价值之前，先对目标值做对称对数变换：

$$
\text{symlog}(x) = \text{sign}(x) \cdot \ln(|x| + 1)
$$

$$
\text{symexp}(x) = \text{sign}(x) \cdot (\exp(|x|) - 1) \quad \text{（逆变换，用于还原预测值）}
$$

| 原始值 $x$ | $\text{symlog}(x)$ | 压缩效果 |
|:---:|:---:|:---|
| 1 | 0.69 | 几乎不变 |
| 100 | 4.62 | 压缩到 ~5 |
| 10000 | 9.21 | 压缩到 ~9 |
| -50 | -3.93 | 负值也能处理 |

模型在 symlog 空间预测，预测值通过 symexp 还原。效果：无论奖励是 1 还是 10000，经过 symlog 后都在 [0, 10] 量级——梯度尺度统一了，同一个学习率到处都能用。

### 4.3 创新二：KL 平衡——防止先验或后验被"压死"

**问题**：在 1.3 节的 ELBO 损失中，KL 项 $\text{KL}[q \| p]$ 同时对后验 $q$ 和先验 $p$ 产生梯度。如果 KL 梯度主要推动后验 $q$ 去靠拢先验 $p$，后验就会丧失信息（后验坍缩）；如果主要推动先验 $p$，先验就会变成"事后诸葛亮"而在想象阶段表现不佳。

**解决方案**：将 KL 拆成两部分，用 stop-gradient（$\text{sg}$）控制梯度流向，并引入 **free bits**（自由比特）裁剪：

$$
\mathcal{L}_{\text{dyn}} = \beta_{\text{dyn}} \cdot \max(1, \text{KL}[\text{sg}(q) \| p])
$$

$$
\mathcal{L}_{\text{rep}} = \beta_{\text{rep}} \cdot \max(1, \text{KL}[q \| \text{sg}(p)])
$$

DreamerV3 默认 $\beta_{\text{dyn}} = 0.5$，$\beta_{\text{rep}} = 0.1$（比值约 5:1）。

- **动力学损失 $\mathcal{L}_{\text{dyn}}$**：冻结后验 $q$，只更新先验 $p$——**大部分 KL 梯度推动先验学得更准**
- **表示损失 $\mathcal{L}_{\text{rep}}$**：冻结先验 $p$，只更新后验 $q$——**温和地正则化后验，防止过度自由**
- **$\max(1, \cdot)$（free bits）**：当 KL 已经降到 1 nat 以下时，停止优化它——避免模型把全部精力放在让动力学变平凡上，而是转向提升预测损失

设计直觉：先验是想象阶段的命根——如果先验不准，在脑内模拟的轨迹全是胡扯。所以把大部分"学习压力"给先验，让它拼命逼近后验。后验只需要轻度正则化，防止它跑得太偏离先验即可。free bits 则防止"过度正则化"——当 KL 已经足够小时，不再强迫它进一步缩小。

> **注**：KL 平衡的思想最早在 DreamerV2 中以 $\alpha=0.8$ 的混合权重形式提出（$\alpha \cdot \text{KL}[\text{sg}(q) \| p] + (1-\alpha) \cdot \text{KL}[q \| \text{sg}(p)]$）。DreamerV3 在此基础上引入了 free bits 裁剪和分离的 $\beta$ 系数，使训练更稳定。

### 4.4 创新三：百分位归一化——不同领域的回报自动对齐

**问题**：Actor 的目标是最大化想象轨迹上的回报 $\sum_t \hat{r}_t$。但不同领域的回报范围天差地别：Minecraft 可能只有 [0, 1]，某些 Atari 游戏可能是 [0, 100000]。如果不归一化，同一个学习率在高回报领域更新太猛、在低回报领域更新太弱。

**解决方案**：用 running percentile 将回报自动归一化到 [0, 1]：

$$
R_{\text{norm}} = \frac{R - \text{Perc}_5(R)}{\text{Perc}_{95}(R) - \text{Perc}_5(R)}
$$

用第 5 和第 95 百分位数（而非 min/max）作为归一化边界，对离群值鲁棒。这样无论回报的绝对尺度如何，Actor 看到的梯度信号总在 [0, 1] 量级。

### 4.5 三个创新的协同效应与实验结果

| 组件 | 解决的问题 | 受益环节 |
|------|-----------|---------|
| symlog | 奖励/价值尺度不一致 | 世界模型训练、Critic 训练 |
| KL 平衡 | 先验/后验的学习速率不对称 | 世界模型训练 |
| 百分位归一化 | 回报尺度不一致 | Actor 训练 |

三个组件分别解决了训练流程中三个不同环节的尺度问题。它们叠加在一起，使得**同一套超参**在以下所有领域都达到或超越领域 SOTA：

| 领域 | 任务数 | DreamerV3 vs 领域 SOTA |
|------|--------|----------------------|
| Atari 100K | 26 | **接近** EfficientZero（无需树搜索） |
| Atari 200M | 55 | **超越** 人类水平 |
| DMC Vision | 20 | **持平** DrQ-v2 |
| DMC Proprio | 10 | **持平** SAC |
| **Minecraft 钻石** | 1 | **首次从零完成** |

## 5. 其他世界模型范式：RSSM 不是唯一的路

Dreamer 系列证明了"潜空间想象 + Actor-Critic"的有效性，但 RSSM 并非唯一的世界模型架构。下面介绍两种有代表性的替代方案——一种用 Transformer 替代 GRU，另一种完全放弃解码器。

### 5.1 IRIS：Transformer 作为世界模型

IRIS（Micheli et al., 2023）用**离散 VQ 编码 + 自回归 Transformer**替代 RSSM：

1. **VQ 编码器**：将观测量化为离散 token 序列

2. **Transformer**：自回归预测下一个 token（类似 GPT 预测下一个词）

3. **策略学习**：在 Transformer 想象的 token 序列上训练

$$
P(z_{t+1}^{1:K} \mid z_{\leq t}^{1:K}, a_{\leq t}) = \prod_{k=1}^{K} P(z_{t+1}^k \mid z_{t+1}^{<k}, z_{\leq t}^{1:K}, a_{\leq t})
$$

IRIS 在 Atari 100K 上以极少的交互样本（2 小时游戏时间）达到了 DreamerV2 级别的性能。

### 5.2 TD-MPC2：完全放弃解码器

回顾 Dreamer 的 ELBO 损失：重建项 $-\log p(o_t \mid h_t, z_t)$ 要求解码器能从潜空间重建像素级观测。这带来两个成本：(1) 解码器本身的参数和计算量；(2) 为了让重建损失有效，潜空间被迫保留大量与决策无关的视觉细节。

TD-MPC2（Hansen et al., 2023）走了一条截然不同的路：**完全不做观测重建（Decoder-Free）**。

世界模型只在潜空间运作：

$$
\begin{aligned}
z_t &= h_\theta(o_t) \quad &\text{(编码器)} \\
z_{t+1} &= d_\theta(z_t, a_t) \quad &\text{(潜动力学)} \\
r_t &= R_\theta(z_t, a_t) \quad &\text{(奖励预测)} \\
Q &= Q_\theta(z_t, a_t) \quad &\text{(Q 值预测)}
\end{aligned}
$$

没有解码器——模型不需要能重建图像，只需要潜空间对**规划有用**。

规划用 **Model Predictive Control (MPC)**：在每步决策时，在世界模型中短程 rollout 多条轨迹，选择 Q 值最高的动作序列。

**无解码器模型的理论挑战与 2025 年的进展**：

放弃解码器虽然大幅提升了训练速度（无需渲染像素），但也带来了一个致命的理论问题：**表征崩塌（Representation Collapse）**。如果没有重建损失的约束，编码器很容易将所有观测映射到同一个常数向量，从而完美但无意义地最小化动力学预测误差。

早期的无解码器方法严重依赖于数据增强（Data Augmentation）或对比学习来防止崩塌。而在最新研究（如 R2-Dreamer，ICLR 2026）中，研究者引入了受 Barlow Twins 启发的**冗余减少（Redundancy-Reduction）目标**：

$$
\mathcal{L}_{\text{R2}} = \sum_i (1 - \mathcal{C}_{ii})^2 + \lambda \sum_i \sum_{j \neq i} \mathcal{C}_{ij}^2
$$

其中 $\mathcal{C}$ 是特征的互相关矩阵。这迫使潜变量的不同维度捕捉独立的信息，在不使用解码器和数据增强的情况下，成功防止了表征崩塌，训练速度比 DreamerV3 快 1.59 倍，同时保持了相当的性能。

| 维度 | Dreamer | IRIS | TD-MPC2 |
|------|---------|------|---------|
| 潜空间 | RSSM（确定+随机） | 离散 VQ token | 连续向量（无解码器） |
| 序列模型 | GRU | Transformer | MLP |
| 策略学习 | Actor-Critic | Actor-Critic | MPC |
| 解码器 | 需要 | 需要 | 不需要 |
| 多任务能力 | 单任务调超参 | 单任务 | **多任务（单一模型）** |

### 5.3 DayDreamer：走向真实机器人

DayDreamer（Wu et al., 2022）将 Dreamer 思路部署到四种真实机器人平台，证明世界模型不只在模拟器中有效。

关键挑战：

- 真实传感器噪声远大于模拟器

- 执行动作有延迟

- 不能"重置"环境

DayDreamer 在 A1 四足机器人上仅用 **1 小时**真实世界数据就学会了稳定行走——而 model-free RL 在模拟器中需要数百万步。

## 6. 代码实现：简化版 RSSM

下面用 PyTorch 实现一个简化版的 RSSM（DreamerV1 风格），包括四个核心方程（序列模型、先验、后验、解码器）以及训练和想象的完整流程。代码与上文的公式一一对应，重点关注**数据流**而非工程优化。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl_divergence


class RSSM(nn.Module):
    """简化版 RSSM（DreamerV1 风格）。

    Args:
        obs_dim: 观测维度。
        act_dim: 动作维度。
        stoch_dim: 随机潜变量维度。
        deter_dim: GRU 确定性状态维度。
    """

    def __init__(self, obs_dim, act_dim, stoch_dim=30, deter_dim=200):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim

        # 序列模型 (GRU)
        self.gru = nn.GRUCell(stoch_dim + act_dim, deter_dim)

        # 先验 p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, 200),
            nn.ELU(),
            nn.Linear(200, 2 * stoch_dim),  # mu + logvar
        )

        # 后验 q(z_t | h_t, o_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + obs_dim, 200),
            nn.ELU(),
            nn.Linear(200, 2 * stoch_dim),
        )

        # 观测嵌入：obs -> 与观测同维的向量
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 200),
            nn.ELU(),
            nn.Linear(200, obs_dim),
        )

        # 解码器：拼接 (h_t, z_t) 重建观测
        self.decoder = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, obs_dim),
        )

        # 奖励预测头
        self.reward_head = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, 200),
            nn.ELU(),
            nn.Linear(200, 1),
        )

    def _get_dist(self, stats):
        """将 ``(mu, logscale)`` 拼接向量拆成对角高斯。

        Args:
            stats: ``(..., 2 * stoch_dim)``，前半为均值、后半为 log-scale。

        Returns:
            ``Independent(Normal, 1)`` 分布对象。
        """
        mu, logvar = stats.chunk(2, dim=-1)
        std = F.softplus(logvar) + 0.1
        return Independent(Normal(mu, std), 1)

    def prior(self, h):
        """先验 p(z_t | h_t)。

        Args:
            h: 确定性状态，``(B, deter_dim)``。

        Returns:
            随机潜变量上的分布。
        """
        return self._get_dist(self.prior_net(h))

    def posterior(self, h, obs_embed):
        """后验 q(z_t | h_t, o_t)。

        Args:
            h: ``(B, deter_dim)``。
            obs_embed: 观测嵌入，``(B, obs_dim)``。

        Returns:
            随机潜变量上的分布。
        """
        x = torch.cat([h, obs_embed], dim=-1)
        return self._get_dist(self.posterior_net(x))

    def observe(self, obs_seq, act_seq, h_prev, z_prev):
        """训练时沿时间展开：用真实观测计算后验轨迹。

        Args:
            obs_seq: 观测序列，``(B, T, obs_dim)``。
            act_seq: 动作序列，``(B, T, act_dim)``。
            h_prev: 初始确定性状态，``(B, deter_dim)``。
            z_prev: 初始随机状态，``(B, stoch_dim)``。

        Returns:
            ``(prior_mean, post_mean, h_seq, z_seq, priors, posteriors)``；
            ``h_seq, z_seq`` 为 ``(B, T, *)``。
        """
        T = obs_seq.shape[1]
        priors, posteriors, h_list, z_list = [], [], [], []
        h, z = h_prev, z_prev

        for t in range(T):
            # 序列更新
            x = torch.cat([z, act_seq[:, t]], dim=-1)  # (B, stoch+act)
            h = self.gru(x, h)

            prior_dist = self.prior(h)
            obs_embed = self.obs_encoder(obs_seq[:, t])
            post_dist = self.posterior(h, obs_embed)
            z = post_dist.rsample()

            priors.append(prior_dist)
            posteriors.append(post_dist)
            h_list.append(h)
            z_list.append(z)

        prior_mean = torch.stack([p.mean for p in priors], dim=1)
        post_mean = torch.stack([p.mean for p in posteriors], dim=1)
        h_seq = torch.stack(h_list, dim=1)
        z_seq = torch.stack(z_list, dim=1)
        return prior_mean, post_mean, h_seq, z_seq, priors, posteriors

    def imagine(self, policy, h, z, horizon):
        """想象 rollout：仅用转移先验与策略生成轨迹。

        Args:
            policy: 映射 ``concat(h,z) -> a`` 的可调用对象。
            h: 初始确定性状态，``(B, deter_dim)``。
            z: 初始随机状态，``(B, stoch_dim)``。
            horizon: 想象步数。

        Returns:
            ``(h_traj, z_traj, a_traj)``；其中 ``h_traj, z_traj`` 为
            ``(B, horizon+1, *)``，``a_traj`` 为 ``(B, horizon, act_dim)``。
        """
        h_list, z_list, a_list = [h], [z], []
        for _ in range(horizon):
            hz = torch.cat([h, z], dim=-1)
            a = policy(hz)
            x = torch.cat([z, a], dim=-1)
            h = self.gru(x, h)
            z = self.prior(h).rsample()
            h_list.append(h)
            z_list.append(z)
            a_list.append(a)
        h_traj = torch.stack(h_list, dim=1)
        z_traj = torch.stack(z_list, dim=1)
        a_traj = torch.stack(a_list, dim=1)
        return h_traj, z_traj, a_traj

    def compute_loss(self, obs_seq, act_seq, rew_seq):
        """简化的重建 + 奖励 + KL 目标。

        Args:
            obs_seq: ``(B, T, obs_dim)``。
            act_seq: ``(B, T, act_dim)``。
            rew_seq: ``(B, T)``。

        Returns:
            标量损失。
        """
        B, T = obs_seq.shape[:2]
        h_0 = torch.zeros(B, self.deter_dim, device=obs_seq.device)
        z_0 = torch.zeros(B, self.stoch_dim, device=obs_seq.device)

        _, _, h_seq, z_seq, priors, posteriors = self.observe(
            obs_seq, act_seq, h_0, z_0
        )

        # 重建
        features = torch.cat([h_seq, z_seq], dim=-1)  # (B, T, deter+stoch)
        obs_pred = self.decoder(features)
        recon_loss = F.mse_loss(obs_pred, obs_seq, reduction="mean")

        # 奖励
        rew_pred = self.reward_head(features).squeeze(-1)
        reward_loss = F.mse_loss(rew_pred, rew_seq, reduction="mean")

        # KL：时间步平均
        kl_terms = [
            kl_divergence(post, prior).mean()
            for prior, post in zip(priors, posteriors)
        ]
        kl_loss = sum(kl_terms) / T

        return recon_loss + reward_loss + 0.1 * kl_loss
```

## 7. 总结：从 V1 到 V3，一条清晰的演进脉络

Dreamer 系列的演进可以概括为一句话：**让"想象训练"这件事变得更稳定、更通用、更省心**。

| 模型 | 年份 | 潜变量 | 核心痛点 → 创新 | 标志性成就 |
|------|------|--------|----------------|-----------|
| World Models | 2018 | 连续 (VAE) | 首次证明"在梦境中训练"可行 | CarRacing |
| Dreamer v1 | 2019 | 连续 (RSSM) | 如何在想象中学策略 → 价值梯度 | DMC 控制 |
| DreamerV2 | 2021 | 离散 (Categorical) | 连续高斯单峰 → 离散多模态 + ST 梯度 | Atari 200M |
| DreamerV3 | 2023 | 离散 | 超参不通用 → symlog + KL 平衡 + 百分位归一化 | Minecraft 钻石 |
| IRIS | 2023 | 离散 (VQ) | 用 Transformer 替代 GRU | Atari 100K |
| TD-MPC2 | 2023 | 连续（无解码器） | 不重建像素，纯潜空间 + MPC | 多任务控制 |
| DayDreamer | 2022 | 连续 (RSSM) | 从仿真迁移到真实机器人 | 1小时学行走 |

**一个关键观察**：Dreamer 系列都在做"生成式"预测——用解码器重建观测。LeCun 认为这条路线的根本问题在于，像素级预测会把计算资源浪费在与决策无关的视觉细节上（比如树叶的纹理）。他提出了一种完全不重建像素的替代方案：**JEPA**，在嵌入空间而非像素空间做预测。这正是下一篇的主题。

> 参考资料：
>
> 1. Hafner, D., ... & Ba, J. (2019). *Dream to Control: Learning Behaviors by Latent Imagination*. ICLR 2020.
> 2. Hafner, D., ... & Ba, J. (2021). *Mastering Atari with Discrete World Models*. ICLR 2021.
> 3. Hafner, D., ... & Ba, J. (2023). *Mastering Diverse Domains through World Models*. Nature 2025.
> 4. Micheli, V., ... & Fleuret, F. (2023). *Transformers are Sample-Efficient World Models*. ICLR 2023.
> 5. Hansen, N., ... & Abbeel, P. (2023). *TD-MPC2: Scalable, Robust World Models for Continuous Control*. ICLR 2024.
> 6. Wu, P., ... & Abbeel, P. (2022). *DayDreamer: World Models for Physical Robot Learning*. CoRL 2022.
> 7. Nakano, M., ... & Harada, T. (2026). *R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation*. ICLR 2026.

> 下一篇：[笔记｜世界模型（三）：JEPA——在嵌入空间预测世界](/chengYi-xun/posts/28-jepa/)
