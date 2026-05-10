---
title: 笔记｜强化学习（五）：Flow-GRPO 与图像生成应用（基于 Flux 的代码解析）
date: 2025-08-20 10:00:00
cover: false
mathjax: true
categories:

 - Notes
tags:

 - Deep learning

 - Generative models theory

 - Reinforcement Learning

 - Flow Matching
series: Diffusion Models theory
---

> 本文为 RL 系列第五篇。在完整梳理了从 REINFORCE 到 PPO、DPO，再到最新 GRPO 的演进路线后，我们将目光转向图像生成领域。本文将结合 `flow_grpo` 开源代码库，深入解析如何将 GRPO 算法应用于基于 Flow Matching 的图像生成模型（如 Flux）的微调中。方法学与系统实验见论文 [*Flow-GRPO: Training Flow Matching Models via Online RL*](https://arxiv.org/abs/2505.05470)（文中以 SD3.5 等为主报告；仓库实现覆盖 Flux）。
>
> ⬅️ 上一篇：[笔记｜强化学习（四）：大模型在线 RL 破局者：GRPO 算法详解](/chengYi-xun/posts/54-grpo/)
>
> ➡️ 下一篇：[笔记｜强化学习（六）：DAPO：从 GRPO 到大规模推理 RL 的工程实践](/chengYi-xun/posts/56-dapo/)


# 图像生成中的强化学习

**先用一个例子理解为什么需要 RL。**

假设你用一个 Flux 模型生成图像，给定 Prompt："一只橘猫坐在蓝色沙发上"。模型可能生成以下几种结果：

| 生成结果 | 问题 |
|:---|:---|
| 一只白色猫坐在蓝色沙发上 | 颜色不对（应该是橘猫） |
| 一只橘猫站在蓝色沙发旁边 | 动作不对（应该是"坐在"） |
| 一只橘猫坐在蓝色沙发上，画面清晰 | 符合预期 |
| 一只橘猫坐在蓝色沙发上，但画面模糊 | 质量差 |

传统的训练方式（Flow Matching 损失）只是让模型学会"生成看起来像训练集的图像"。但训练集里可能有模糊的、构图差的、与 Prompt 不一致的图像——模型无法区分好坏。

**RL 的价值**：我们训练一个“代理奖励模型”（Proxy Reward Model, RM，如 PickScore 或 ImageReward）来给图像打分。模型自己生成图像 → RM 打分 → 模型根据分数调整自己。这就是 RLHF 在图像生成中的应用。

![Flow-GRPO 概览：ODE→SDE 注入随机性、训练期 Denoising Reduction 与组内 GRPO 更新（摘自 Liu et al., arXiv:2505.05470 图 2）](/chengYi-xun/img/flow_grpo_arch.png)

---

# Flow-GRPO 框架解析：基于组内相对优势的策略优化

**先看例子**：对于 Prompt "一只橘猫坐在蓝色沙发上"，我们让 Flux 模型生成 $G = 4$ 张图像，RM 分别打分：

| 图像 | 描述 | 奖励 $r_i$ | 相对优势 $\hat{A}_i$ |
|:---:|:---|:---:|:---:|
| 图 1 | 橘猫坐沙发，画面清晰 | $r_1 = 0.9$ | $+1.27$ |
| 图 2 | 橘猫坐沙发，稍微模糊 | $r_2 = 0.6$ | $-0.12$ |
| 图 3 | 白猫坐沙发（颜色错） | $r_3 = 0.3$ | $-1.50$ |
| 图 4 | 橘猫坐沙发，普通水平 | $r_4 = 0.7$ | $+0.35$ |

（均值 $\mu_R = 0.625$，标准差 $\sigma_R \approx 0.22$）

跟上一篇 GRPO 的做法完全一样：图 1 和图 4 高于平均（正优势），模型学习生成更像它们的图；图 3 远低于平均（负优势），模型学习远离这种生成方式。**不需要 Critic 网络，只需要多生成几张图做对比。**

**核心思考出发点**：由于像 Flux 这样的图像生成模型参数量达到百亿级别，传统的 PPO 算法由于需要额外的 Critic 网络，显存开销极大。因此，Flow-GRPO 采用了 GRPO 算法——移除了 Critic，用"组内相对评分"来实现高效的在线强化学习。

## 核心挑战：如何在连续生成过程中定义 $\log \pi_\theta$？

在 LLM 中，动作（Action）是离散的词表 Token，$\log \pi_\theta(a|s)$ 就是 softmax 输出的对数概率——定义清晰、计算简单。然而在 Flow Matching 中，生成过程是一个**连续的常微分方程（ODE）求解过程**，没有天然的"离散动作"概念。

**用例子理解**：LLM 生成文本就像逐字写作——每个字是一个离散的"动作"，概率就是词表上的 softmax。而 Flux 生成图像像是画画——每个时间步的"动作"是在画布上做一次**连续的涂抹**（从噪声图向清晰图的一步变换），这是一个高维连续向量，不存在离散概率。

### 将去噪过程建模为 MDP

Flow-GRPO 的第一个关键设计是：将 Flow Matching 的去噪过程定义为一个 **马尔可夫决策过程**：

| MDP 要素 | LLM (GRPO) | 图像生成 (Flow-GRPO) |
|:---:|:---|:---|
| **状态** $s_t$ | $(x, y_{<t})$ (Prompt + 已生成 token) | $(x_t, t, c)$ (当前噪声图 + 时间步 + 文本条件) |
| **动作** $a_t$ | 下一个 token $y_t \in \mathcal{V}$（离散） | 预测的速度场 $v_\theta(x_t, t, c)$（连续向量） |
| **转移** | 确定性：拼接 $y_t$ 到序列 | 确定性 ODE 步：$x_{t-\Delta t} = x_t - \Delta t \cdot v_\theta$ |
| **奖励** | 稀疏奖励（仅在整句完成后获得） | 稀疏奖励（仅在 $t=0$ 生成完整图像后获得） |

由于这是一个典型的**稀疏奖励（Sparse Reward）**设定——中间去噪步的即时奖励均为 0，只有在轨迹终点才能获得 RM 的打分。这在数学上构成了长视野的信用分配（Credit Assignment）问题，因此我们需要计算整条轨迹的累积对数概率来进行策略更新。

### 推导 Flow Matching 中的对数概率

在 Flow Matching 框架中，前向过程（加噪）定义为线性插值：

$$
x_t = (1 - t) \cdot x_0 + t \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

其中 $x_0$ 是干净图像，$\epsilon$ 是纯噪声，$t \in [0, 1]$。模型 $v_\theta(x_t, t, c)$ 学习预测速度场（即 $x_0$ 到 $\epsilon$ 方向的向量场）。

在去噪（生成）过程中，每一步的转移可以写成：

$$
x_{t - \Delta t} = x_t - \Delta t \cdot v_\theta(x_t, t, c)
$$

**如何从这个过程中提取对数概率？** 确定性 ODE 没有概率可言——给定初始噪声 $x_T$，每步转移是唯一确定的，不存在"选择 A 而非 B"的随机性，自然也就没有 $\log\pi_\theta$ 可以计算。

**解决思路（三步走）**：

1. **引入随机性**：在 ODE 的每一步注入高斯噪声，将确定性 ODE 改为随机的 SDE。这样每步转移就变成了一个高斯分布，$\log\pi_\theta$ 就有了。
2. **保持分布不变**：光加噪声会破坏生成质量。我们需要同时加入 Score Function 纠偏项，使改造后的 SDE 在统计分布上与原始 ODE 完全等价（即生成的图像质量不变）。
3. **提取对数概率**：从 SDE 的高斯转移核中直接计算每步的 $\log p(x_{t-\Delta t} | x_t)$，累加得到整条轨迹的 $\log\pi_\theta$。

以下我们按这三步展开。

### 1. 为什么要引入 SDE 与 Score Function？（第一步：引入随机性）

在纯 ODE 采样中，模型就像是沿着一条设定好的轨道平滑地滑向终点（只需速度 $v_\theta$ 即可更新 $x_t$）。但 Flow-GRPO 为了让强化学习能够“试错”和“探索”，引入了 **SDE（随机微分方程）**，也就是在滑行的过程中加入随机的扰动（噪声）。

**问题来了**：如果盲目地加入随机噪声，生成的轨迹就会偏离真实图像的流形（Manifold），最终生成崩坏的画面。

**解决方案**：我们需要一个“指南针”来纠正这种偏离，这个指南针就是 **Score Function（分数函数 $\nabla_{x_t} \log p_t(x_t)$）**。它在数学上指向数据分布密度增加（更接近真实图像分布）的方向。一旦随机探索导致偏航，Score 修正项会提供一个指向高密度区域的拉力。

**Score 在 SDE 中的数学角色**：将去噪 ODE 转换为逆向 SDE 后（完整推导见下文 Section 2），每一步去噪更新变为（注意时间方向：$t \to t - \Delta t$，即**生成/去噪方向**）：

$$x_{t-\Delta t} = \underbrace{(x_t - \Delta t \cdot v_\theta)}_{\text{逆向 ODE 漂移}} + \underbrace{\tfrac{1}{2}g^2 \cdot \nabla_{x_t}\log p_t(x_t) \cdot \Delta t}_{\text{Score 纠偏}} + \underbrace{g\sqrt{\Delta t}\cdot\epsilon}_{\text{随机探索}}$$

第一项是原始 ODE 的确定性去噪；第三项是为 RL 注入的随机噪声；第二项就是 Score 纠偏——它的方向指向数据高密度区域，恰好抵消噪声带来的分布偏移。

{% note warning no-icon %}
**关键问题：Score 纠偏和随机噪声之间是什么关系？**

直觉上会产生一个疑问：噪声 $\epsilon$ 是随机的（每个样本不同），那纠偏是否也应该依赖于具体加了什么噪声？答案是：**Score 不纠正某个具体的 $\epsilon$，而是提供一个位置相关的"恢复力场"。**

类比：想象一群粒子在山谷中随机游走。噪声 = 每个粒子随机晃动的方向（每人不同）；Score = 山谷壁给的重力（只取决于你站在哪里，不关心你怎么晃过来的）。山谷壁不需要知道你具体往哪个方向晃了——它只需要把所有偏离谷底的人往回拉。

但"山谷的形状"必须正确——Score $\nabla\log p_t(x_t)$ 依赖于当前步的正确分布 $p_t$。如果中间分布偏了（山谷形状变了），Score 给出的纠偏方向也会错，后续所有步骤的错误就会累积。所以我们需要证明 SDE 在**每一步**都保持和 ODE 相同的中间分布 $p_t(x)$，而不仅仅是最终输出一致。

数学上，通过 **Fokker-Planck 方程（福克-普朗克方程，又称 Kolmogorov Forward Equation）**可以严格证明这一点。

**预备知识：Fokker-Planck 方程（福克-普朗克方程）是什么？**

一句话：**SDE 描述的是单个粒子怎么走，FP 方程描述的是一大群粒子的概率密度怎么变。** 它把"随机轨迹"的问题翻译成了"确定性的偏微分方程"问题，从而可以用分析工具来研究。

想象你同时释放 100 万个粒子，每个都独立遵循相同的 SDE $dx = \mu\,d\tau + \sigma\,dW$。在任意时刻 $\tau$，这些粒子散布在空间中形成一团"概率云"。$p(x, \tau)$ 就是这个云的密度——描述在位置 $x$ 附近找到粒子的概率。

Fokker-Planck 方程告诉我们这个密度如何随时间演化，它由两个物理效应叠加而成：

$$\frac{\partial p}{\partial \tau} = \underbrace{-\nabla\cdot(\mu\, p)}_{\text{漂移搬运概率（连续性方程）}} + \underbrace{\frac{1}{2}\sigma^2\nabla^2 p}_{\text{噪声摊平概率（热方程）}}$$

- **第一项**来自确定性漂移 $\mu$：如同流体力学中的质量守恒，$\mu \cdot p$ 是"概率流"，$\nabla\cdot(\mu p)$ 是某处的净流出量。流出多于流入 → 密度下降 → 需要负号。
- **第二项**来自随机噪声 $\sigma\,dW$：噪声让粒子扩散（从密集处散开），其效果等价于热传导——热量从高温（高密度）流向低温（低密度）。$\nabla^2 p$ 度量局部密度的"凹凸程度"。系数中的 $\frac{1}{2}$ 来自**伊藤引理（Itô's Lemma）中 Taylor 展开的二阶项系数** $\frac{1}{2!}$：普通微积分中 $(dx)^2$ 是高阶无穷小可以丢弃，但维纳过程满足 $(dW)^2 = d\tau$（不为零），因此二阶项 $\frac{1}{2}\sigma^2\frac{\partial^2}{\partial x^2}$ 必须保留，$\frac{1}{2}$ 便随之进入 FP 方程。

**若没有噪声**（$\sigma = 0$），FP 方程退化为纯漂移的 Liouville 方程 $\partial_\tau p = -\nabla\cdot(\mu\,p)$。

以下是利用 FP 方程的完整证明。**为什么要证这个？** 因为我们刚刚把 ODE 改成了 SDE（加了噪声 + score 纠偏）。读者的合理疑问是：你这么一改，生成出来的图还正常吗？以下证明回答：**正常，改造前后每一步的概率密度 $p_t(x)$ 完全一致。** （第二步：保持分布不变）

**目标**：证明 SDE 采样与 ODE 采样产生相同的边缘分布 $p_t(x)$。

**Step 1. 写出两者的分布演化方程。** 设反向时间变量 $\tau$（$\tau$ 递增时实际时间 $t$ 递减）。

纯 ODE（$x_{\text{new}} = x - v_\theta \Delta t$）的 Liouville 方程（FP 方程中 $\sigma=0$ 的特例）：

$$\frac{\partial p}{\partial \tau} = \nabla \cdot (v_\theta \cdot p)$$

SDE（漂移 $\mu = -v_\theta + \frac{1}{2}g^2\nabla\log p_t$，扩散 $\sigma = g$）的 Fokker-Planck 方程：

$$\frac{\partial p}{\partial \tau} = -\nabla\cdot(\mu\, p) + \frac{1}{2}\sigma^2\nabla^2 p$$

**Step 2. 展开 SDE 的 Fokker-Planck 方程：**

$$= -\nabla\cdot\left[\left(-v_\theta + \tfrac{1}{2}g^2\nabla\log p_t\right)p\right] + \tfrac{1}{2}g^2\nabla^2 p$$

$$= \underbrace{\nabla\cdot(v_\theta\, p)}_{\text{ODE 贡献}} \;-\; \underbrace{\tfrac{1}{2}g^2\nabla\cdot\left[(\nabla\log p_t)\,p\right]}_{\text{Score 项贡献}} \;+\; \underbrace{\tfrac{1}{2}g^2\nabla^2 p}_{\text{噪声扩散贡献}}$$

**Step 3. 利用恒等式 $(\nabla\log p_t)\cdot p_t = \nabla p_t$**（因为 $\nabla\log p = \nabla p / p$）：

$$\tfrac{1}{2}g^2\nabla\cdot[(\nabla\log p_t)\,p_t] = \tfrac{1}{2}g^2\nabla\cdot(\nabla p_t) = \tfrac{1}{2}g^2\nabla^2 p_t$$

**Step 4. Score 项与噪声项精确抵消：**

$$\frac{\partial p}{\partial \tau} = \nabla\cdot(v_\theta p) - \cancel{\tfrac{1}{2}g^2\nabla^2 p_t} + \cancel{\tfrac{1}{2}g^2\nabla^2 p_t} = \nabla\cdot(v_\theta\, p)$$

这与纯 ODE 的 Liouville 方程完全相同。$\blacksquare$

**结论**：Score 纠偏项（$\frac{1}{2}g^2 \nabla\log p_t$）与噪声扩散效应（方差 $g^2\Delta t$）在 Fokker-Planck 方程中逐项对消，使 SDE 的分布演化等价于纯 ODE。这就是为什么 Score 系数恰好是 $\frac{1}{2}g^2$：它精确匹配噪声的统计效应。

这个等价是**逐步成立**的——FP 方程在任意时刻 $t$ 都一样，所以不只是最终生成图 $p_0(x)$ 一致，而是**每一步中间分布 $p_t(x)$ 都一致**。这一点至关重要，因为 score 纠偏项本身依赖当前步的 $\nabla\log p_t$：如果某一步的中间分布 $p_t$ 偏了，那该步的 score 就会算错，错误会向后续步骤累积——就像导航基于错误位置给出的指引只会让你越走越偏。逐步等价保证了这条因果链不会断裂。

- 个体样本：轨迹因 $\epsilon$ 不同而各异（这正是 RL "探索"的意义）
- 统计分布：所有样本构成的整体分布**在每一步**都与 ODE 保持一致（"边缘分布不变"）

{% endnote %}

**回到主线（第三步：提取对数概率）**：现在我们已经证明了 SDE 不会破坏分布。这意味着我们可以放心地使用 SDE 的高斯转移核来计算 $\log\pi_\theta$——因为每一步 $x_{t-\Delta t} | x_t$ 都是一个高斯分布，其对数概率可以直接写出来。我们将在后续代码解析中看到这一步的具体实现。

但要使用这个 SDE，还需要解决一个实际问题：公式中的 Score $\nabla_{x_t}\log p_t(x_t)$ 怎么算？这就用到了高斯特威迪公式（Gaussian Tweedie's Formula, Efron 2011）：

Tweedie 公式是一个适用于指数族分布的广义定理，在各向同性高斯扰动核的特例下，它证明了一个深刻的结论：**Score 可以通过贝叶斯后验均值 $\mathbb{E}[x_0 \mid x_t]$（即模型预测的 $\hat{x}_0$）来反向精确表达。** 这使得神经网络无需直接拟合 Score，而是可以通过预测 $x_0$ 间接得到。

具体推导如下：

**第一步：确定 $x_t$ 的条件分布**
在 Rectified Flow 中，前向加噪过程是干净图像 $x_0$ 和纯噪声 $x_1 \sim \mathcal{N}(0, I)$ 的直线插值：
$$x_t = (1-\sigma)x_0 + \sigma x_1$$
由于 $x_1$ 是标准高斯噪声，所以给定 $x_0$ 时，$x_t$ 的条件分布 $p(x_t | x_0)$ 自然也是一个高斯分布：

- 均值 $\mu_t = (1-\sigma)x_0$
- 方差 $\sigma_t^2 = \sigma^2 I$

**第二步：写出高斯分布的概率密度函数（PDF）并取对数**
多维高斯分布的概率密度函数为：
$$p(x_t | x_0) = \frac{1}{(2\pi \sigma^2)^{d/2}} \exp\left( -\frac{\|x_t - \mu_t\|^2}{2\sigma^2} \right)$$
我们在两边同时取自然对数 $\log$：
$$\log p(x_t | x_0) = -\frac{\|x_t - \mu_t\|^2}{2\sigma^2} - \frac{d}{2}\log(2\pi\sigma^2)$$

**第三步：对 $x_t$ 求梯度（即计算 Score）**
Score Function 的定义就是对数概率密度对 $x_t$ 的偏导数（梯度）。
因为后面的 $-\frac{d}{2}\log(2\pi\sigma^2)$ 是常数，求导后为 0。我们只需要对前面的二次项求导：
$$\nabla_{x_t} \log p(x_t | x_0) = \nabla_{x_t} \left( -\frac{\|x_t - \mu_t\|^2}{2\sigma^2} \right)$$
根据向量求导法则 $\nabla_x \|x - \mu\|^2 = 2(x - \mu)$，代入上式得到：
$$\nabla_{x_t} \log p(x_t | x_0) = -\frac{2(x_t - \mu_t)}{2\sigma^2} = -\frac{x_t - \mu_t}{\sigma^2}$$

**第四步：代入均值 $\mu_t$ 与模型预测**
将 $\mu_t = (1-\sigma)x_0$ 代入，得到：
$$\nabla_{x_t} \log p(x_t | x_0) = -\frac{x_t - (1-\sigma)x_0}{\sigma^2}$$
在实际生成时，我们并不知道真实的 $x_0$ 是什么，但根据特威迪公式，我们可以用模型当前预测的 $\hat{x}_0$ 来近似替代真实的 $x_0$。
而根据直线运动公式（起点 = 当前位置 - 速度 × 时间），我们可以用模型预测的速度场 $v_\theta$ 反推 $\hat{x}_0$：
$$\hat{x}_0 = x_t - \sigma \cdot v_\theta$$

把模型预测的 $\hat{x}_0$ 代入 Score 公式，就能在代码中直接计算出纠偏所需的“指南针”了。

### 2. SDE 离散化公式链

下面将上述连续 SDE 逐步离散化。为保证学术严谨性，本文将 $t \in [0,1]$ 严格作为连续时间变量（$t=1$ 为纯噪声），而将 $\sigma$ 定义为离散化采样时的调度节点（Noise Schedule）。设当前时间步为 $\sigma$，模型预测速度场为 $v_\theta$，扩散系数为 $g(\sigma)$，离散步长为 $\Delta\sigma = \sigma_{\text{next}} - \sigma < 0$（去噪方向），对应正向时间增量 $\Delta t = -\Delta\sigma > 0$。

{% note warning no-icon %}
**离散化带来的截断误差（Truncation Error）**：
在将连续的 SDE 转化为离散的代码实现时（通常使用 Euler-Maruyama 方法），我们隐含了一个极其强烈的假设：**在 $\Delta \sigma$ 这一大步内，Score 的方向是恒定不变的。** 在高维非线性空间中，沿着一个恒定方向走一大步必然会产生截断误差，导致样本偏离真实的数据流形。这正是后续产生“高频颗粒感伪影”的数学根源，也是后续工作（如 Flow-CPS 提出的系数保持采样）致力于解决的核心痛点。
{% endnote %}

**公式 ①：Tweedie 反推干净样本**

$$\hat{x}_0 = x_t - \sigma \cdot v_\theta \tag{①}$$

利用 Rectified Flow 的直线插值 $x_t = (1-\sigma)x_0 + \sigma x_1$（其中 $x_1$ 为纯噪声），速度场 $v_\theta$ 训练目标为预测 $x_1 - x_0$。因此 $x_t = x_0 + \sigma(x_1 - x_0) = x_0 + \sigma v_\theta$，可直接推导出 $\hat{x}_0 = x_t - \sigma v_\theta$。

**公式 ②：Score Function（Tweedie 估计）**

将 ① 代入 Score 定义 $\nabla_{x_t}\log p_t = -\frac{x_t - (1-\sigma)\hat{x}_0}{\sigma^2}$：

$$\nabla_{x_t}\log p_t(x_t) = -\frac{x_t - (1-\sigma)\hat{x}_0}{\sigma^2} = -\frac{x_t + (1-\sigma)v_\theta}{\sigma} \tag{②}$$

**公式 ③：SDE 转移均值（ODE 漂移 + Score 纠偏）**

$$\mu = \underbrace{(x_t + v_\theta \Delta\sigma)}_{\text{ODE 漂移}} + \underbrace{\tfrac{1}{2} \frac{g^2}{\sigma} \cdot (x_t + (1-\sigma)v_\theta) \cdot \Delta\sigma}_{\text{Score 纠偏}} \tag{③}$$

**为什么需要 Score 纠偏？** 如果仅在 ODE 落点上叠加噪声（跳过修正项），噪声会使 $x_{t-\Delta t}$ 的分布相对于真实 $p_{t-\Delta t}$ 发生额外膨胀。随步数累积，分布偏移导致图像崩坏。Score 纠偏沿 Score 方向预补偿噪声引起的分布膨胀，确保加噪后的采样仍落在正确的边缘分布内。若 $g=0$，修正项为零，退化为纯 ODE。

**公式 ④：SDE 采样**

$$x_{t-\Delta t} = \mu + g\sqrt{\Delta t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) \tag{④}$$

公式 ③④ 合起来构成 SDE 一步：先计算修正均值 $\mu$，再以 $\mu$ 为中心重采样。条件分布为 $p(x_{t-\Delta t} \mid x_t) = \mathcal{N}(\mu,\; g^2 \Delta t \cdot I)$。

**边缘分布不变性**：③④ 的设计保证 SDE 采样与纯 ODE 采样在**统计意义**上遵循相同的边缘分布——图像质量和多样性的总体特征一致。但单条轨迹变为随机的：同一初始噪声下，ODE 每次产出相同图像，SDE 每次产出不同图像。此性质使 GRPO 能在同一 Prompt 下生成多张不同图像做组内对比，同时不因探索而降低生成质量。

**公式 ⑤：单步对数概率（单维度）**

严格的多元高斯对数似然应对所有维度 $d$ 求和。为与代码对应，这里给出**单维度（或单像素）**的对数似然公式：

$$\log p_\theta(x_{t-\Delta t} \mid x_t, c) = -\frac{(x_{t-\Delta t} - \mu)^2}{2\,g^2\,\Delta t} - \log(g\sqrt{\Delta t}) - \tfrac{1}{2}\log(2\pi) \tag{⑤}$$

后两项（归一化常数）仅依赖 $g$ 和 $\Delta t$，不含策略参数 $\theta$。在 GRPO 的 importance ratio $\exp(\log\pi_\theta^{\text{new}} - \log\pi_\theta^{\text{old}})$ 中分子分母相消，对梯度无贡献，但实现时保留以便数值调试。

{% note info no-icon %}
**理论与代码的缩放关系**：在真实的 $d$ 维空间中（$d \sim 65536$），严谨的对数概率应是各维度之和。而官方代码中使用了 `log_prob.mean(dim=...)`，即在空间维度上取了**均值**而非求和。在数学上，这等价于将 Importance Ratio 从 $r_t = \exp(\Delta \log \pi_\text{sum})$ 变为了 $\hat{r}_t = \exp(\frac{1}{d} \Delta \log \pi_\text{sum}) = (r_t)^{1/d}$。
这并非简单的 Loss 缩放！在图像的极高维空间中，严谨的 $\Delta \log \pi_\text{sum}$ 的绝对值会非常大，直接取指数会导致 $r_t$ 数值溢出或下溢为 0。通过取均值（即对概率比开 $d$ 次方），使得 $\hat{r}_t$ 能够保持在合理的数值范围内，从而让 PPO 的梯度能够正常反传。这是高维连续空间 RL 中不可或缺的工程处理。

**值得注意的是**：高斯对数概率的固有数学性质导致 importance ratio 的分布**系统性左偏**（均值 < 1）。如果不做 mean（直接 sum），这个偏移会被放大到"数值崩溃级"（ratio ≈ 0，训练完全无法进行）。mean 操作将其压缩到了"可观察可修复级"（ratio ≈ 0.9x），使训练成为可能。但残留的偏移仍会导致 PPO clipping 机制对正样本失效，引发隐式过优化——proxy reward 上升而图像质量下降。这个残留问题正是后文 GRPO-Guard 章节要解决的核心问题。
{% endnote %}

**公式 ⑥：整条轨迹对数概率**

$$\log \pi_\theta(\text{trajectory} \mid c) = \sum_{k=1}^{T} \log p_\theta(x_{t_k - \Delta t} \mid x_{t_k}, c) \tag{⑥}$$

与 LLM 中 token 级对数概率求和形式完全对应，至此 GRPO 框架可无缝迁移到图像生成。

### 3. 漂移均值（Drift Mean）更新公式推导

本节详细推导两种实现中“统一 SDE 框架”下单步去噪漂移均值（Drift Mean）更新公式的演变过程。我们的目标是从基础的 SDE 转移均值公式（公式 ③）出发，推导出代码中融合了 $x_t$ 和 $v_\theta$ 的多项式形式。

**Step 1：转换时间步长与符号**

公式 ③ 是以正向时间步 $\Delta t$ 编写的，包含 ODE 漂移项和 Score 纠偏项（即 Langevin 修正）：
$$ \mu = \underbrace{(x_t - \Delta t \cdot v_\theta)}_{\text{ODE 漂移}} + \underbrace{\frac{1}{2}g^2 \cdot \nabla_{x_t}\log p_t \cdot \Delta t}_{\text{Score 纠偏}} $$

在实际代码中，去噪过程的 $\sigma$ 是逐渐减小的，即 $\Delta\sigma = \sigma_{\text{next}} - \sigma < 0$。它们的关系是：
$$ \Delta t = -\Delta\sigma $$

将代码符号替换进去，ODE 漂移项变成加号，Score 纠偏项因为 $\Delta t$ 的替换变成减号：
$$ \mu = (x_t + v_\theta \Delta\sigma) - \frac{1}{2}g^2 \cdot \nabla_{x_t}\log p_t \cdot \Delta\sigma $$

**Step 2：展开 Score 函数 $\nabla_{x_t}\log p_t$**

在 Flow Matching 的设定下，通过预测的干净图像 $\hat{x}_0$ 反推 Score 的简化公式为：
$$ \nabla_{x_t}\log p_t = - \frac{x_t - (1-\sigma)\hat{x}_0}{\sigma} $$

根据代码中的直线轨迹定义 $\hat{x}_0 = x_t - \sigma v_\theta$，我们将其代入：
$$
\begin{aligned}
\nabla_{x_t}\log p_t &= - \frac{x_t - (1-\sigma)(x_t - \sigma v_\theta)}{\sigma} \\
&= - \frac{x_t - (x_t - \sigma x_t - \sigma v_\theta + \sigma^2 v_\theta)}{\sigma} \\
&= - \frac{\sigma x_t + \sigma(1-\sigma)v_\theta}{\sigma}
\end{aligned}
$$

约掉分子分母的 $\sigma$，得到极其简化的 Score 形式：
$$ \nabla_{x_t}\log p_t = - (x_t + (1-\sigma)v_\theta) $$

**Step 3：将 Score 代回原公式**

把化简后的 Score 代入 Step 1 的 $\mu$ 公式（注意负负得正）：
$$ \mu = x_t + v_\theta \Delta\sigma + \frac{1}{2\sigma} g^2 (x_t + (1-\sigma)v_\theta) \Delta\sigma $$

*(注：为匹配特定 SDE 设定的方差缩放，这里系数显式提取了一个 $\frac{1}{\sigma}$)*

**Step 4：代入 Flow-GRPO 特有的自适应噪声系数 $g^2$**

值得重点明确的是，**“自适应噪声”是 Flow-GRPO 针对此框架做出的核心设计与更改**（相比之下，同期的 DanceGRPO 采用的是恒定噪声）。虽然**无论采用何种噪声调度，我们都可以在代数上对均值进行同类项合并**，但 Flow-GRPO 设计的自适应有效噪声方差 $g^2 = \frac{\sigma \eta^2}{1-\sigma}$ 在数学化简上展现出了极佳的优雅性与数值稳定性。

将其代入前面的系数 $\frac{g^2}{2\sigma}$ 中，分子里的 $\sigma$ 刚好与分母消掉：
$$ \frac{g^2}{2\sigma} = \frac{1}{2\sigma} \cdot \frac{\sigma \eta^2}{1-\sigma} = \frac{\eta^2}{2(1-\sigma)} $$

试想，如果像 DanceGRPO 那样采用恒定噪声，分母中的 $2\sigma$ 将会被保留；当生成末期 $\sigma \to 0$ 时，该系数将趋于无穷大，带来严重的数值崩溃风险。而 Flow-GRPO 的自适应设计完美避开了这一除以极小值的问题。

现在公式变成了：
$$ \mu = x_t + v_\theta \Delta\sigma + \frac{\eta^2}{2(1-\sigma)} (x_t + (1-\sigma)v_\theta) \Delta\sigma $$

**Step 5：展开并合并同类项**

我们把大括号完全拆开，分为含有 $x_t$ 的项和含有 $v_\theta$ 的项：
$$
\begin{aligned}
\mu &= x_t + v_\theta \Delta\sigma + \left( \frac{\eta^2}{2(1-\sigma)} x_t \right) \Delta\sigma + \left( \frac{\eta^2(1-\sigma)}{2(1-\sigma)} v_\theta \right) \Delta\sigma \\
&= x_t + v_\theta \Delta\sigma + \frac{\eta^2}{2(1-\sigma)} x_t \Delta\sigma + \frac{\eta^2}{2} v_\theta \Delta\sigma
\end{aligned}
$$

最后，提取 $x_t$ 和 $v_\theta$ 的公因式：

对于 $x_t$ 的系数：
$$ x_t + \frac{\eta^2}{2(1-\sigma)} x_t \Delta\sigma = x_t \left( 1 + \frac{\eta^2}{2(1-\sigma)} \Delta\sigma \right) $$

对于 $v_\theta$ 的系数：
$$ v_\theta \Delta\sigma + \frac{\eta^2}{2} v_\theta \Delta\sigma = v_\theta \left( 1 + \frac{\eta^2}{2} \right) \Delta\sigma $$

把这两部分加起来，就完美得到了 Flow-GRPO 代码中所写的最终合并公式（即公式 ③'）：
$$ \mu = x_t \left( 1 + \frac{\eta^2}{2(1-\sigma)} \Delta\sigma \right) + v_\theta \left( 1 + \frac{\eta^2}{2} \right) \Delta\sigma \tag{③'} $$

### 4. Flow-GRPO：算子融合与自适应噪声（核心实现）

在代码库中，真正采用了自适应噪声 $g^2 = \frac{\sigma \eta^2}{1-\sigma}$ 并应用了我们上述最终推导结果（公式 ③'）的，是 Flow-GRPO 的实现 `flow_grpo_step`：

```python
import math
import torch
from typing import Optional

# 假设环境中有 randn_tensor 函数，这里提供一个简单的替身以便代码完整
def randn_tensor(shape, generator=None, device=None, dtype=None):
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)

# 从1->0  是噪声到清晰图像
def flow_grpo_step(
    model_output: torch.Tensor,  # Transformer 输出的速度 v_θ (B, seq, hidden)
    latents: torch.Tensor,  # 当前 x_t (float32)
    eta: float,  # SDE 噪声强度系数
    sigmas: torch.Tensor,  # 完整 σ 调度表
    index: int,  # 当前步索引 i
    prev_sample: torch.Tensor,  # 若外部已采样则传入，否则 None
    generator: Optional[torch.Generator] = None,  # 随机数生成器
    determistic: bool = False,  # True=ODE（覆盖噪声采样）
    sde_type: str = "sde",  # "sde" 或 "cps"
    noise_level: Optional[float] = None,  # 覆盖 eta 的显式噪声水平
):
    """MixGRPO 单步更新：由速度场 v_θ 更新 latent，计算 SDE 转移的 log_prob。

    做什么：给定当前 x_t 和模型预测 v，执行一步 ODE 或 SDE 转移到 x_{t-Δ}。
    怎么做：
      1. 从 σ 调度读取 σ_i → σ_{i+1}，算步长 dt = σ_{i+1} - σ_i (< 0)
      2. 预测 x̂_0 = x_t - σ·v
      3. ODE：x_{next} = x + dt·v（确定性欧拉步）
         SDE：构造均值 μ 和标准差 σ_eff，采样 x_{next} ~ N(μ, σ_eff²)
      4. 计算 log N(x_{next}; μ, σ_eff²) 作为策略 log_prob
    返回：(x_{next}, x̂_0, log_prob, μ, σ_eff)
    """
    device = model_output.device
    # ── 读取相邻 σ 节点 ──
    sigma = sigmas[index].to(device)
    sigma_prev = sigmas[index + 1].to(device)
    sigma_max = sigmas[1].item()  # 用于 σ=1 时的数值稳定替换
    dt = sigma_prev - sigma  # 负值 = 沿去噪方向前进

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    # ════════════════════════════════════════════════════════════════════
    # sde_type 分支：决定噪声注入公式。ODE 不是单独分支，而是在每个分支内
    # 通过 determistic=True 覆盖采样结果为纯 Euler 步 x + dt·v。
    # ════════════════════════════════════════════════════════════════════

    if sde_type == "sde":
        # ── 标准 SDE：噪声量自适应 σ_eff = √(σ/(1-σ)) · η ──
        # 当 σ→1 时 σ_eff→∞（充分探索），σ→0 时 σ_eff→0（保护细节）
        _noise_level = eta if noise_level is None else noise_level
        # σ==1 时分母会为 0，用 sigma_max 替代防止除零
        std_dev_t = (
            torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))
            * _noise_level
        )

        # x̂_0 = x_t - σ·v_θ （从当前含噪状态和速度反推干净图）
        pred_original_sample = latents - sigma * model_output

        # SDE 漂移均值 μ：
        #
        #   μ = z_t · (1 + η²/(2(1-σ)) · Δσ) + v_θ · (1 + η²/2) · Δσ
        #
        # 其中 std_dev_t² = σ·η²/(1-σ)，代入可验证：
        #   std_dev_t²/(2σ)          = η²/(2(1-σ))     → 第 1 项系数
        #   std_dev_t²·(1-σ)/(2σ)    = η²/2            → 第 2 项系数
        # Δσ = dt = σ_next - σ < 0（去噪方向）
        prev_sample_mean = (
            latents * (1 + std_dev_t**2 / (2 * sigma) * dt)
            + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        )

        # SDE 采样：x_{next} = μ + σ_eff · √|dt| · ε
        # 注：dt < 0（去噪方向），所以 √(-dt) 保证正数
        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=device,
                dtype=model_output.dtype,
            )
            prev_sample = (
                prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise
            )

        # ODE 覆盖：determistic=True 时直接用 Euler 步，丢弃上面的 SDE 采样
        # x_{next} = x + dt · v（纯确定性，无随机性）
        if determistic:
            prev_sample = latents + dt * model_output

        # ── 计算 log_prob = log N(x_next; μ, σ_eff²) ──
        # σ_eff = std_dev_t × √|dt|  (连续扩散系数 × 离散步长→实际标准差)
        # dt < 0（去噪方向），所以 √(-dt) 保证正数
        effective_std = std_dev_t * torch.sqrt(-1 * dt)
        # 完整高斯 log-likelihood 三项：
        #   -(x-μ)²/(2σ²)   马氏距离：采样离均值多远（梯度主信号）
        #   -log(σ)          方差惩罚：σ_eff 越大概率密度越低
        #   -0.5·log(2π)     归一化常数（不影响梯度，仅保数值完整）
        # detach(prev_sample)：固定采样值，梯度只通过 μ 传到模型参数（REINFORCE）
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2)
            / (2 * (effective_std ** 2))
            - torch.log(effective_std)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi, device=device)))
        )
        # (B, C, H, W) → (B,)：对所有像素的 log_prob 取均值
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return (
            prev_sample,
            pred_original_sample,
            log_prob,
            prev_sample_mean,
            effective_std,
        )

    elif sde_type == "cps":
        # ── CPS (Coefficient-Preserving Sampling)：有界噪声，σ→1 时不爆炸 ──
        # 与标准 SDE 的区别：噪声量 = σ_prev · sin(η·π/2)，永远 ≤ σ_prev
        _noise_level = 0.8 if noise_level is None else noise_level
        # 噪声尺度：sin 映射保证 std ∈ [0, σ_prev]（有上界，不爆方差）
        std_dev_t = sigma_prev * math.sin(_noise_level * math.pi / 2)

        # x̂_0 和噪声估计 ε̂（两个方向的"锚点"）
        pred_original_sample = latents - sigma * model_output
        noise_estimate = latents + model_output * (1 - sigma)  # ε̂ = x + v·(1-σ)

        # CPS 均值：在干净图和噪声之间做"系数保持"插值
        # μ = (1-σ_prev)·x̂_0 + √(σ_prev² - std²)·ε̂
        # 保证 μ 的范数不会因噪声注入而膨胀
        prev_sample_mean = pred_original_sample * (
            1 - sigma_prev
        ) + noise_estimate * torch.sqrt(sigma_prev**2 - std_dev_t**2)

        # CPS 采样：x_{next} = μ + std · ε
        if prev_sample is None:
            variance_noise = randn_tensor(
                model_output.shape,
                generator=generator,
                device=device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + std_dev_t * variance_noise

        # ODE 覆盖（与 sde 分支相同：纯 Euler 步）
        if determistic:
            prev_sample = latents + dt * model_output

        # CPS log_prob（简化形式：省略常数项，只保留 -(x-μ)² 信号）
        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob, prev_sample_mean, std_dev_t

    else:
        raise ValueError(f"Unsupported sde_type: {sde_type}. Must be 'sde' or 'cps'.")
```

**工程优势**：

1. **算子融合（Operator Fusion）**：完全摒弃了显式计算 Score 和 $\hat{x}_0$，直接对 $x_t$ (`latents`) 和 $v_\theta$ (`model_output`) 乘以标量系数相加，极大提升了 CUDA 上的计算吞吐量。
2. **数值稳定性（Robustness）**：通过 `torch.where` 防护罩，消除了 $\sigma \to 1$ 时的分母溢出问题。
3. **策略梯度完备性**：其产生的 `log_prob` 包含了完整的马氏距离和方差惩罚项，确保了 REINFORCE 优化的梯度严谨性。

### 5. DanceGRPO：显式分解法（早期/基础实现）

相比之下，DanceGRPO 的 `dance_grpo_step` 保留了更加“原始但粗糙”的数学结构，忠实还原了未化简的推导**Step 1 到 Step 3**的分解形态（即直接使用公式 ③），且未采用自适应噪声设定：

```python
def dance_grpo_step(
    model_output: torch.Tensor,  # 模型速度输出 v_θ
    latents: torch.Tensor,  # 当前 x_t
    eta: float,  # 噪声强度
    sigmas: torch.Tensor,  # σ 调度表
    index: int,  # 当前步索引
    prev_sample: torch.Tensor,  # 外部已采样则传入，否则 None
    grpo: bool,  # True=返回 log_prob；False=仅返回均值
    sde_solver: bool,  # True=SDE（加噪）；False=ODE（确定性）
    sde_type: str = "sde",  # "sde" 或 "cps"
    noise_level: Optional[float] = None,  # 覆盖 eta
):
    """DanceGRPO 单步更新：与 flow_grpo_step 类似但噪声注入公式不同。

    做什么：用模型速度 v 更新 latent，SDE 模式下含 score 修正项。
    怎么做：
      1. 基础 ODE 均值：μ = x + dsigma·v（dsigma < 0）
      2. SDE 模式额外加 score 修正：μ += -0.5·η²·score·dsigma
      3. 采样 x_{next} ~ N(μ, η²·Δt)，计算 log_prob
    返回：grpo=True → (x_{next}, x̂_0, log_prob)；grpo=False → (μ, x̂_0)
    """
    device = latents.device
    sigma = (
        sigmas[index].to(device) if sigmas[index].device != device else sigmas[index]
    )
    sigma_prev = (
        sigmas[index + 1].to(device)
        if sigmas[index + 1].device != device
        else sigmas[index + 1]
    )
    dsigma = sigma_prev - sigma  # < 0，去噪方向步长
    # ── 预测干净图 x̂_0 = x - σ·v ──
    pred_original_sample = latents - sigma * model_output

    if sde_type == "sde":
        # ── ODE 均值：简单欧拉步 ──
        prev_sample_mean = latents + dsigma * model_output
        delta_t = sigma - sigma_prev  # > 0，用于噪声尺度
        _noise_level = eta if noise_level is None else noise_level
        std_dev_t = _noise_level * torch.sqrt(delta_t)

        if sde_solver:
            # ── SDE 修正：加入 score 项使分布更准确 ──
            score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
            log_term = -0.5 * _noise_level**2 * score_estimate
            prev_sample_mean = prev_sample_mean + log_term * dsigma

        # ── 采样或确定性更新 ──
        if grpo and prev_sample is None:
            if sde_solver:
                prev_sample = (
                    prev_sample_mean
                    + torch.randn_like(prev_sample_mean, device=device) * std_dev_t
                )
            else:
                prev_sample = prev_sample_mean

        # ── 计算 log_prob ──
        if grpo:
            log_prob = -(
                (
                    prev_sample.detach().to(torch.float32)
                    - prev_sample_mean.to(torch.float32)
                )
                ** 2
            ) / (2 * (std_dev_t**2))
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
            return prev_sample, pred_original_sample, log_prob
        else:
            return prev_sample_mean, pred_original_sample

    elif sde_type == "cps":
        # ── CPS 模式：系数保持采样 ──
        _noise_level = 0.8 if noise_level is None else noise_level
        std_dev_t = sigma_prev * math.sin(_noise_level * math.pi / 2)
        noise_estimate = latents + model_output * (1 - sigma)
        # 均值由 x̂_0 和 noise_estimate 插值得到
        prev_sample_mean = pred_original_sample * (
            1 - sigma_prev
        ) + noise_estimate * torch.sqrt(sigma_prev**2 - std_dev_t**2)

        if grpo and prev_sample is None:
            if sde_solver:
                prev_sample = prev_sample_mean + std_dev_t * torch.randn_like(
                    prev_sample_mean, device=device, dtype=prev_sample_mean.dtype
                )
            else:
                prev_sample = prev_sample_mean

        if grpo:
            log_prob = -(
                (
                    prev_sample.detach().to(torch.float32)
                    - prev_sample_mean.to(torch.float32)
                )
                ** 2
            )
            log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
            return prev_sample, pred_original_sample, log_prob
        else:
            return prev_sample_mean, pred_original_sample

    else:
        raise ValueError(f"Unsupported sde_type: {sde_type}. Must be 'sde' or 'cps'.")

```

**工程缺陷**：
1. **极值崩溃风险**：直接除以 `sigma**2`。当生成到达末期（$\sigma \to 0$）时，由于缺乏极小值截断，极易导致数值不稳定（NaN）。
2. **中间变量内存开销**：需要分配 `pred_original_sample` 和 `score_estimate` 等多个中间 tensor，降低了 GPU 效率。

### 6. 两种代码实现的对比总结

代码实现与理论往往存在一定的“错位”：在这个框架下，DanceGRPO 的代码结构保留了理论推导的原始形态，更像是一个未经深度数值优化的对照基线（Baseline）；而 Flow-GRPO 则优雅地完成了算子融合，落实了数值上更稳定的 SDE 自适应噪声与漂移均值计算。

### 7. 整体框架回顾：与 LLM GRPO 的异同

梳理完完整的 SDE 改造与对数概率推导后，我们可以清晰地看到：Flow-GRPO **与 LLM GRPO 的宏观算法结构（组采样 → 优势计算 → PPO 裁剪更新 → KL 惩罚）是完全一致的**。

两者的**核心差异仅仅在于“对数概率 $\log \pi_\theta$”的获取方式**：

- **LLM（离散空间）**：直接从模型最后一层的分类头中，按生成的 token 提取 softmax 的对数概率并求和。
- **Flow-GRPO（连续空间）**：基于 SDE 改造，利用每一去噪步的高斯转移核对数密度，按前文的公式 ⑥ 沿着整条轨迹逐步累加，最终得到 $\log\pi_\theta(\text{trajectory}\mid c)$。

---

## Flow-GRPO-Fast：加速采样的工程优化

全量去噪采样是 Flow-GRPO 的计算瓶颈——生成一张 1024×1024 图像，Flux 默认需要 50 步 ODE 求解。每个 Prompt 生成 $G=4$ 张就是 200 步，每个训练 step 有 4 个 Prompt 就是 800 步。Flow-GRPO 提出了两种加速策略：

### 策略 1：部分去噪（Partial Denoising）

不从纯噪声 $t=1$ 开始，而是从中间时间步 $t_{\text{start}}$ 开始（如 $t=0.5$）：

$$
x_{t_{\text{start}}} = (1 - t_{\text{start}}) \cdot x_0^{\text{ref}} + t_{\text{start}} \cdot \epsilon
$$

其中 $x_0^{\text{ref}}$ 是参考模型生成的一张"参考图"。这样只需去噪 $t_{\text{start}} \times T$ 步（比如 25 步而非 50 步），速度翻倍。

**代价**：生成多样性降低（所有 $G$ 张图都从同一个"参考半成品"出发），但对于微调场景通常足够。

### 策略 2：减少采样步数

直接减少 ODE 求解步数（如从 50 步减到 20 步），配合高阶 ODE 求解器（如 DPM-Solver++）。精度略有下降，但速度大幅提升。

---

# GRPO-Guard：缓解隐式过优化

> 论文：[GRPO-Guard: Mitigating Implicit Over-Optimization in Flow Matching via Regulated Clipping](https://arxiv.org/abs/2510.22319)（同为 Flow-GRPO 团队，2025）
> 代码：已集成在 [flow_grpo](https://github.com/yifan123/flow_grpo) 和 [Flow-Factory](https://github.com/X-GenGroup/Flow-Factory) 中

## 问题：Importance Ratio 的固有偏差

Flow-GRPO 和 DanceGRPO 在训练中使用 PPO-style clipping 来约束策略更新。PPO 的 clipping 机制假设 importance ratio $r_t = \pi_\theta(a_t|s_t) / \pi_{\theta_\text{old}}(a_t|s_t)$ 的分布**以 1 为中心**。但在 Flow Matching 模型中，importance ratio 的分布存在**系统性的负向偏差**：

1. **均值始终低于 1**，在低噪声步（如 SD3.5-M 的 step 8）偏差尤为显著。
2. **方差在不同去噪步之间差异极大**，对于对数比率 $\log r_t$，低噪声步的方差远大于高噪声步。

{% note info no-icon %}
**深度解析：为什么偏差是"负向"的？为什么方差随时间步变化？**

在 Flow Matching SDE 中，每一步的"策略"是一个高斯分布：$\pi_\theta(x_{t+1}|x_t) = \mathcal{N}(\mu_\theta, \sigma_t^2 I)$。设 $\delta = \mu_\theta - \mu_{\theta_\text{old}}$ 为策略更新导致的均值偏移，$e = x_{t+1} - \mu_{\theta_\text{old}} \sim \mathcal{N}(0, \sigma_t^2 I)$ 为采样噪声，则 log-ratio 可以精确展开为：

$$\log r_t = \frac{e^T \delta}{\sigma_t^2} - \frac{\|\delta\|^2}{2\sigma_t^2}$$

第一项 $\frac{e^T \delta}{\sigma_t^2}$ 是零均值的随机项（因为 $\mathbb{E}[e] = 0$），第二项 $-\frac{\|\delta\|^2}{2\sigma_t^2}$ 是一个**恒为负的常数偏置**。因此：

$$\mathbb{E}[\log r_t] = -\frac{\|\delta\|^2}{2\sigma_t^2} < 0$$

**1. 为什么偏差是负向的？**
由于 $\|\delta\|^2 \geq 0$，这个偏置项恒为非正——这是高斯分布的固有数学性质，与策略更新的方向无关。只要 $\theta \neq \theta_\text{old}$（即策略发生了更新），$\log r_t$ 的均值就通常小于 0。虽然理论上 $\mathbb{E}[r_t] = 1$（对数正态分布的性质），但在图像 latent 的高维空间中（维度 $d \sim 16 \times 64 \times 64 = 65536$），$\|\delta\|^2$ 随维度 $d$ 缩放，导致 $\log r_t$ 的分布极度右偏：绝大多数样本的 $r_t \ll 1$，仅有极少数样本 $r_t \gg 1$ 来拉平均值。在有限样本下，这些极端值很难被观测到，因此经验均值通常远低于 1。

**2. 为什么低噪声步的对数比率方差大、高噪声步方差小？**
根据公式，$\log r_t$ 的方差为 $\text{Var}[\log r_t] = \frac{\|\delta\|^2}{\sigma_t^2}$。在低噪声步（$\sigma_t$ 小），由于分母较小，导致 $\log r_t$ 的方差极大。直观上，低噪声步时参考分布的高斯噪声 $\mathcal{N}(0, \sigma_t^2 I)$ 非常"尖锐"，因此策略模型的预测均值稍有偏移 $\|\delta\|$，对概率密度的影响就会被剧烈放大，使得 log-ratio 波动极大。相反，在高噪声步（$\sigma_t$ 大），分布宽而平坦，相同的偏移引起的概率变化较小，$\log r_t$ 的方差也相对较小。这种方差的不一致，进一步破坏了 PPO clipping 机制在各个时间步上的对称性和有效性。
{% endnote %}

这种偏差使得 PPO 的 clipping 区间 $[1-\varepsilon, 1+\varepsilon]$ 变得**不对称**。由于绝大多数样本的 $r_t$ 远小于 1，正样本（高奖励的好图）的 ratio 反而更容易落在 clipping 区间内部（不被截断），导致正样本的梯度更新不受约束，策略模型不断向这些样本偏移。

{% note warning no-icon %}
**为什么向"好图"方向偏移，图像质量反而会下降？**

这是经典的**古德哈特定律（Goodhart's Law）**："当一个度量成为了目标，它就不再是一个好的度量。"

GRPO 训练中的"正样本"（好图）是由**代理奖励模型**（如 PickScore）定义的，而非真实的人类偏好。PickScore 本质上是一个有限能力的神经网络，它对"好图"的判断存在系统性的盲点和偏见（例如对某些纹理模式的虚假偏好）。当策略模型被过度优化后，它学会了**精准利用奖励模型的这些弱点**——生成那些"PickScore 认为很好，但人类觉得不对"的图像。

因此，**代理分数（proxy score）持续上升**（模型越来越擅长"取悦" PickScore），但**真实质量（gold score）不断下降**（人类评估者认为图像质量在退化）。这就是隐式过优化（Implicit Over-Optimization）的本质。
{% endnote %}

## 解决方案：RatioNorm + Gradient Reweight

GRPO-Guard 提出了两个互补的机制：

### 1. RatioNorm（比率归一化）

RatioNorm 的目标是纠正 importance ratio 的分布偏差，使其均值回归到 1、方差在不同步之间保持一致。

具体做法是引入一个时间步相关的缩放因子 $c_t = \sqrt{\Delta t} \cdot \sigma_t$（其中 $\Delta t$ 是步长，$\sigma_t$ 是前向加噪的噪声标准差，对应代码中的 `sigma`，而非 SDE 探索噪声 $g$），并用它来重新缩放 log-ratio：

$$\hat{r}_t = \exp\left[({\log \pi_\theta - \log \pi_{\theta_\text{old}}}) \cdot c_t + \frac{\|\mu_\theta - \mu_{\theta_\text{old}}\|^2}{2 c_t}\right]$$

这个公式在数学上完美呼应了前文发现的两个缺陷：
1. **乘法因子 $c_t$**：用于将不同时间步的方差缩放对齐，解决“低噪声步方差极大、高噪声步方差极小”的问题。
2. **加法补偿项 $\frac{\|\mu_\theta - \mu_{\theta_\text{old}}\|^2}{2 c_t}$**：这里的分子 $\|\mu_\theta - \mu_{\theta_\text{old}}\|^2$ 正是前文推导中的 $\|\delta\|^2$。该项被精确设计用来抵消高斯分布带来的负向常数偏置 $\mathbb{E}[\log r_t] = -\frac{\|\delta\|^2}{2\sigma_t^2}$，从而解决“均值始终低于 1”的问题。

经过 RatioNorm 校正后，importance ratio 的分布在所有时间步上都以 1 为中心，PPO 的 clipping 机制重新恢复了对称性。

### 2. Gradient Reweight（梯度重加权）

即使 RatioNorm 校正了 ratio 的分布，不同时间步对总 loss 的梯度贡献仍然不均衡。Gradient Reweight 对最终的 policy loss 进行时间步相关的重加权：

$$\mathcal{L}_\text{Guard} = \frac{\mathcal{L}_\text{PPO}(\hat{r}_t)}{(\sqrt{\Delta t})^2}$$

这使得每个时间步对总梯度的贡献大致相等，防止某些特定噪声水平下的过度优化。

## 与 MixGRPO 的对比：哪个更好？

GRPO-Guard 与 MixGRPO 都试图解决 Flow-GRPO 的过优化/Reward Hacking 问题，但它们解决的是**不同层面**的问题，严格来说不构成"谁更好"的竞争关系，而是互补关系。

**切入点完全不同：**

- **GRPO-Guard** 从**梯度端**入手：认为问题的根源是 SDE 使 importance ratio 的统计特性产生了偏差，导致 PPO clipping 失效。解决方法是 RatioNorm + Gradient Reweight。
- **Flow-CPS 与 MixGRPO** 从**采样端**入手：解决 SDE 带来的副作用。Flow-CPS 提出了系数保持采样（消除高频伪影）；而 MixGRPO 则引入了滑动窗口（Mixed ODE-SDE）来提升训练效率，并结合原始模型推理限制伪影传播。

**各自的优势领域：**

- **抗过优化能力**：GRPO-Guard 是专门为此设计的。GRPO-Guard 论文的实验表明，在 SD3.5-M 上以 GenEval 为 proxy reward 训练 1860 步后，Flow-GRPO 的 Gold Score（三项真实指标 HPS-v2、ImageReward、UnifiedReward 的归一化均值）跌至 0.84（基线 = 1.00），而 GRPO-Guard 维持在 0.89（提升 +0.05）。在 Flux.1-dev 上，DanceGRPO 的 Gold Score 跌至 0.88，GRPO-Guard 则恢复到 1.02（甚至超过原始模型）。视觉上，Flow-GRPO 和 DanceGRPO 在训练后期会出现严重的水平/垂直条纹伪影、面部同质化和人体比例失调，而 GRPO-Guard 保持了正常的图像质量和多样性。
- **训练效率**：MixGRPO 在这方面优势明显，通过 Mixed ODE-SDE + 滑动窗口机制将训练开销削减了约 50%，同时在 ImageReward 和 HPS-v2.1 等指标上超越了 Flow-GRPO 和 DanceGRPO。

**GRPO-Guard 的局限**（论文自己承认的）：RatioNorm 只能修复 clipping 机制的失效问题，**无法消除奖励模型本身的固有缺陷**（proxy score 与 gold score 之间的 gap）。如果奖励模型本身就有系统性偏见，单纯修复 clipping 也无法完全阻止 reward hacking。更根本的解决方案是提升奖励模型本身的能力（如 RewardDance），但这会引入大量计算开销。

**实际使用建议**：这两种方法并不互斥，可以组合使用。[Flow-Factory](https://github.com/X-GenGroup/Flow-Factory) 已经同时支持了两者，用户可以选择 `trainer_type: 'grpo-guard'` + `dynamics_type: 'CPS'`，将梯度端的 ratio 修正与采样端的伪影消除同时启用，理论上能获得最佳的抗过优化效果。

---

# 算法对比与开源生态

| 维度 | Diffusion-DPO | DDPO (PPO) | Flow-GRPO |
|:---:|:---:|:---:|:---:|
| **训练方式** | 离线（偏好对） | 在线 RL | 在线 RL |
| **探索与优化机制的数学本质** | 基于轨迹 KL 散度的闭式解 | 基于单步高斯转移的 REINFORCE | 基于 SDE 轨迹对数似然的 PPO/GRPO |
| **需要 Critic** | 否 | 是 | **否** |
| **基线估计** | 无 | Critic $V_\phi$ | 组内均值 |
| **适用模型** | DDPM / LDM | DDPM / LDM | **Flow Matching (Flux)** |
| **显存** | 低 | 极高 | **低** |
| **探索能力** | 弱 | 强 | 强 |

**开源代码参考：** [flow_grpo](https://github.com/yifan123/flow_grpo) 提供了基于 Flux 的完整实现，支持 LoRA 微调、多 GPU 训练和 Flow-GRPO-Fast 加速。

---

# 系列总结

通过这五篇文章，我们从最基础的强化学习与策略梯度出发，推导了解决步长控制的 PPO 算法，探讨了绕开 RL 的 DPO 路线，最终迎来了解决大模型显存危机的 GRPO 算法，并成功将其落地到了最前沿的 Flow-GRPO 图像生成微调框架中。

强化学习与生成模型的结合，正在开启 AI 领域的新纪元。无论是语言模型中的深度思考（DeepSeek-R1），还是图像生成中的美学对齐（Flow-GRPO），在线强化学习都展现出了无与伦比的潜力。

> 参考资料：
>
> 1. Liu, Y., Wang, P., Shao, Z., ... & Hao, K. (2025). *Flow-GRPO: Training Flow Matching Models via Online RL*. arXiv:2505.05470.
> 2. Black Forest Labs. (2024). *Flux.1 [dev]*. https://blackforestlabs.ai/
> 3. [flow_grpo](https://github.com/yifan123/flow_grpo)
> 4. Wang, J., et al. (2025). *GRPO-Guard: Mitigating Implicit Over-Optimization in Flow Matching via Regulated Clipping*. arXiv:2510.22319.

> 下一篇：[笔记｜强化学习（六）：DAPO：从 GRPO 到大规模推理 RL 的工程实践](/chengYi-xun/posts/56-dapo/)
