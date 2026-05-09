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

**如何从这个过程中提取对数概率？** 关键观察：确定性 ODE 没有概率可言，但如果我们在每一步都加入微小的高斯扰动（将 ODE 变成 SDE），转移就变成了一个随机过程。

为了保证加入噪声后轨迹依然收敛到真实数据分布，必须引入 **Score Function** 进行 Langevin 纠偏。

### 1. 为什么要引入 SDE 与 Score Function？

在纯 ODE 采样中，模型就像是沿着一条设定好的轨道平滑地滑向终点（只需速度 $v_\theta$ 即可更新 $x_t$）。但 Flow-GRPO 为了让强化学习能够“试错”和“探索”，引入了 **SDE（随机微分方程）**，也就是在滑行的过程中加入随机的扰动（噪声）。

**问题来了**：如果盲目地加入随机噪声，生成的轨迹就会偏离真实图像的流形（Manifold），最终生成崩坏的画面。

**解决方案**：我们需要一个“指南针”来纠正这种偏离，这个指南针就是 **Score Function（分数函数 $\nabla_{x_t} \log p_t(x_t)$）**。它在数学上指向数据分布密度增加（更接近真实图像分布）的方向。一旦随机探索导致偏航，Score 修正项会提供一个指向高密度区域的拉力。

**Score 在 SDE 中的数学角色**：将 ODE 转换为 SDE 后（完整推导见下文 Section 2），每一步去噪更新变为：

$$x_{t-\Delta t} = \underbrace{(x_t - \Delta t \cdot v_\theta)}_{\text{ODE 漂移}} + \underbrace{\tfrac{1}{2}g^2 \cdot \nabla_{x_t}\log p_t(x_t) \cdot \Delta t}_{\text{Score 纠偏}} + \underbrace{g\sqrt{\Delta t}\cdot\epsilon}_{\text{随机探索}}$$

第一项是原始 ODE 的确定性去噪；第三项是为 RL 注入的随机噪声；第二项就是 Score 纠偏——它的方向指向数据高密度区域，恰好抵消噪声带来的分布偏移。

{% note warning no-icon %}
**关键问题：Score 纠偏和随机噪声之间是什么关系？**

直觉上会产生一个疑问：噪声 $\epsilon$ 是随机的（每个样本不同），那纠偏是否也应该依赖于具体加了什么噪声？答案是：**Score 不纠正某个具体的 $\epsilon$，而是提供一个位置相关的"恢复力场"。**

类比：想象一群粒子在山谷中随机游走。噪声 = 每个粒子随机晃动的方向（每人不同）；Score = 山谷壁给的重力（只取决于你站在哪里，不关心你怎么晃过来的）。山谷壁不需要知道你具体往哪个方向晃了——它只需要把所有偏离谷底的人往回拉。

数学上，通过 **Fokker-Planck 方程（Kolmogorov Forward Equation）**可以严格证明这一点。

**预备知识：Fokker-Planck 方程是什么？**

想象你同时释放 100 万个粒子，每个都独立遵循相同的 SDE $dx = \mu\,d\tau + \sigma\,dW$。在任意时刻 $\tau$，这些粒子散布在空间中形成一团"概率云"。$p(x, \tau)$ 就是这个云的密度——描述在位置 $x$ 附近找到粒子的概率。

Fokker-Planck 方程告诉我们这个密度如何随时间演化，它由两个物理效应叠加而成：

$$\frac{\partial p}{\partial \tau} = \underbrace{-\nabla\cdot(\mu\, p)}_{\text{漂移搬运概率（连续性方程）}} + \underbrace{\frac{1}{2}\sigma^2\nabla^2 p}_{\text{噪声摊平概率（热方程）}}$$

- **第一项**来自确定性漂移 $\mu$：如同流体力学中的质量守恒，$\mu \cdot p$ 是"概率流"，$\nabla\cdot(\mu p)$ 是某处的净流出量。流出多于流入 → 密度下降 → 需要负号。
- **第二项**来自随机噪声 $\sigma\,dW$：噪声让粒子扩散（从密集处散开），其效果等价于热传导——热量从高温（高密度）流向低温（低密度）。$\nabla^2 p$ 度量局部密度的"凹凸程度"，系数 $\frac{1}{2}\sigma^2$ 来自维纳过程的方差性质 $\text{Var}(\sigma\,dW) = \sigma^2\,d\tau$。

**若没有噪声**（$\sigma = 0$），FP 方程退化为纯漂移的 Liouville 方程 $\partial_\tau p = -\nabla\cdot(\mu\,p)$。

以下是利用 FP 方程的完整证明：

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

- 个体样本：轨迹因 $\epsilon$ 不同而各异（这正是 RL "探索"的意义）
- 统计分布：所有样本构成的整体分布始终与 ODE 保持一致（"边缘分布不变"）

{% endnote %}

**那么，核心问题变成了：如何高效计算这个 $\nabla_{x_t}\log p_t(x_t)$？这就用到了高斯特威迪公式（Gaussian Tweedie's Formula, Efron 2011）：**

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

### 2. 从前向 SDE 推导逆向 SDE（Anderson 定理）

在 SDE 理论中，一个连续时间的随机微分方程（如扩散模型的前向加噪过程）通常写为：
$$\mathrm{d}x_t = f(x_t, t) \mathrm{d}t + g(t) \mathrm{d}w_t$$

其中 $f(x_t, t)$ 是漂移项（决定运动的宏观方向），$g(t) \mathrm{d}w_t$ 是扩散项（注入高斯噪声，导致分布发散）。

**如何推导它的逆向过程？**
我们可以通过离散化的时间步结合贝叶斯公式来进行直观推导。

假设时间步长为 $\Delta t > 0$，前向过程从 $t$ 走到 $t + \Delta t$：
$$x_{t+\Delta t} = x_t + f(x_t, t)\Delta t + g(t)\sqrt{\Delta t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

这说明，给定 $x_t$ 时，$x_{t+\Delta t}$ 服从高斯分布：
$$p(x_{t+\Delta t} | x_t) \approx \mathcal{N}(x_t + f(x_t, t)\Delta t, \; g(t)^2 \Delta t I)$$

现在我们要推导逆向过程，即给定 $x_{t+\Delta t}$ 时，求 $x_t$ 的分布 $p(x_t | x_{t+\Delta t})$。根据贝叶斯公式：
$$p(x_t | x_{t+\Delta t}) = \frac{p(x_{t+\Delta t} | x_t) p(x_t)}{p(x_{t+\Delta t})}$$

我们在两边取对数：
$$\log p(x_t | x_{t+\Delta t}) = \log p(x_{t+\Delta t} | x_t) + \log p(x_t) - \log p(x_{t+\Delta t})$$

利用一阶泰勒展开，我们可以将 $\log p(x_t)$ 在 $x_{t+\Delta t}$ 处近似展开：
$$\log p(x_t) \approx \log p(x_{t+\Delta t}) + \nabla_{x_{t+\Delta t}} \log p(x_{t+\Delta t}) \cdot (x_t - x_{t+\Delta t})$$

将这个展开式和前面前向转移概率 $p(x_{t+\Delta t} | x_t)$ 的高斯对数密度代入贝叶斯公式。经过代数化简和配方（合并关于 $x_t$ 的二次项），我们可以发现逆向分布 $p(x_t | x_{t+\Delta t})$ 依然是一个高斯分布，其均值为：
$$\mathbb{E}[x_t | x_{t+\Delta t}] \approx x_{t+\Delta t} - \left[ f(x_{t+\Delta t}, t+\Delta t) - g(t+\Delta t)^2 \nabla_{x} \log p(x_{t+\Delta t}) \right] \Delta t$$

当 $\Delta t \to 0$ 时，我们将时间倒流记为逆向微元 $\mathrm{d}\bar{t}$，上面的离散更新公式就化为了连续的**逆向 SDE（Reverse SDE，由 Anderson, 1982 证明）**：
$$\mathrm{d}x_t = \left[ f(x_t, t) - g(t)^2 \nabla_{x_t} \log p_t(x_t) \right] \mathrm{d}t + g(t) \mathrm{d}\bar{w}_t$$

**公式中各项的物理意义：**

- **$f(x_t, t)$**：**漂移项**——系统固有的确定性演化速度场，即神经网络预测的去噪方向（Probability Flow ODE 的速度）。
- **$g(t)$**：**扩散系数**——为探索而注入的随机噪声的标准差，控制随机性强度。
- **$\nabla_{x_t} \log p_t(x_t)$**：**得分函数**——概率密度场在状态空间的梯度，指向数据流形高密度方向。
- **$- g(t)^2 \nabla_{x_t} \log p_t(x_t)$**：**得分修正项**——抵消噪声注入带来的分布发散。注入 $g(t)\mathrm{d}\bar{w}_t$ 会使样本偏离数据流形，此修正项沿 Score 方向将其拉回。
- **$g(t) \mathrm{d}\bar{w}_t$**：**逆向布朗运动**——纯随机扰动，为 RL 微调提供探索不同生成结果的随机性来源。

**核心结论：**

在图像生成与强化学习微调（如 Flow-GRPO）的结合中，该公式展示了“探索与保真”的数学平衡。为了在生成过程中探索不同的高奖励输出（如更好的美学评分或更准确的 prompt 遵循），必须通过 $g(t) \mathrm{d}\bar{w}_t$ **注入随机噪声**；但纯粹的噪声会破坏图像的真实性（导致伪影或崩坏），因此必须同步引入 $- g(t)^2 \nabla_{x_t} \log p_t(x_t)$ 作为**数学上的强制纠偏力**，确保探索轨迹始终贴合真实图像的数据流形。

**Flow Matching 中的特殊性（ODE 转 SDE）：**
在 Flow Matching 中，我们原本拥有的是一个确定性的常微分方程（ODE），即 $\mathrm{d}x_t = f(x_t, t) \mathrm{d}t$。根据 Song et al. (2020) 的经典理论，对于任意一个扩散 SDE，都存在一个具有相同边缘分布的确定性常微分方程，即 Probability Flow ODE。它能够在噪声和图像分布之间建立确定性的映射。
现在，为了在强化学习中进行“探索”，我们想在这个 ODE 中**强行注入噪声** $g(t) \mathrm{d}\bar{w}_t$，但又**不想改变每个时刻的边缘分布 $p_t(x_t)$**。
根据随机过程理论，如果我们向一个保持分布的 ODE 中注入方差为 $g(t)^2$ 的噪声，为了抵消噪声带来的发散效应，必须在漂移项中加入 $-\frac{1}{2} g(t)^2 \nabla_{x_t} \log p_t(x_t)$ 的修正。

因此，Flow-GRPO 中实际使用的 SDE 漂移项修正系数是 $\frac{1}{2}$：

1. $f(x_t, t)$ 是原本的 ODE 漂移项（基础更新）。
2. $-\frac{1}{2} g(t)^2 \nabla_{x_t} \log p_t(x_t)$ 就是**为了抵消注入噪声带来的分布发散，而必须强制加入的 Score 修正项**。

### 3. SDE 离散化公式链

下面将上述连续 SDE 逐步离散化。为保证学术严谨性，本文将 $t \in [0,1]$ 严格作为连续时间变量（$t=1$ 为纯噪声），而将 $\sigma$ 定义为离散化采样时的调度节点（Noise Schedule）。设当前时间步为 $\sigma$，模型预测速度场为 $v_\theta$，扩散系数为 $g(\sigma)$，离散步长为 $\Delta\sigma = \sigma_{\text{next}} - \sigma < 0$（去噪方向），对应正向时间增量 $\Delta t = -\Delta\sigma > 0$。

{% note warning no-icon %}
**离散化带来的截断误差（Truncation Error）**：
在将连续的 SDE 转化为离散的代码实现时（通常使用 Euler-Maruyama 方法），我们隐含了一个极其强烈的假设：**在 $\Delta \sigma$ 这一大步内，Score 的方向是恒定不变的。** 在高维非线性空间中，沿着一个恒定方向走一大步必然会产生截断误差，导致样本偏离真实的数据流形。这正是后续产生“高频颗粒感伪影”的数学根源，也是后续工作（如 MixGRPO 中的 CPS 采样）致力于解决的核心痛点。
{% endnote %}

**公式 ①：Tweedie 反推干净样本**

$$\hat{x}_0 = x_t - \sigma \cdot v_\theta \tag{①}$$

利用 Rectified Flow 的直线插值 $x_t = (1-\sigma)x_0 + \sigma x_1$（其中 $x_1$ 为纯噪声），速度场 $v_\theta$ 训练目标为预测 $x_1 - x_0$。因此 $x_t = x_0 + \sigma(x_1 - x_0) = x_0 + \sigma v_\theta$，可直接推导出 $\hat{x}_0 = x_t - \sigma v_\theta$。

**公式 ②：Score Function（Tweedie 估计）**

将 ① 代入 Score 定义 $\nabla_{x_t}\log p_t = -\frac{x_t - (1-\sigma)\hat{x}_0}{\sigma^2}$：

$$\nabla_{x_t}\log p_t(x_t) = -\frac{x_t - (1-\sigma)\hat{x}_0}{\sigma^2} = -\frac{x_t + (1-\sigma)v_\theta}{\sigma} \tag{②}$$

**公式 ③：SDE 转移均值（ODE 漂移 + Langevin 修正）**

$$\mu = \underbrace{(x_t + v_\theta \Delta\sigma)}_{\text{ODE 漂移}} + \underbrace{\tfrac{1}{2} \frac{g^2}{\sigma} \cdot (x_t + (1-\sigma)v_\theta) \cdot \Delta\sigma}_{\text{Langevin 修正}} \tag{③}$$

**为什么需要 Langevin 修正？** 如果仅在 ODE 落点上叠加噪声（跳过修正项），噪声会使 $x_{t-\Delta t}$ 的分布相对于真实 $p_{t-\Delta t}$ 发生额外膨胀。随步数累积，分布偏移导致图像崩坏。Langevin 修正沿 Score 方向预补偿噪声引起的分布膨胀，确保加噪后的采样仍落在正确的边缘分布内。若 $g=0$，修正项为零，退化为纯 ODE。

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
{% endnote %}

**公式 ⑥：整条轨迹对数概率**

$$\log \pi_\theta(\text{trajectory} \mid c) = \sum_{k=1}^{T} \log p_\theta(x_{t_k - \Delta t} \mid x_{t_k}, c) \tag{⑥}$$

与 LLM 中 token 级对数概率求和形式完全对应，至此 GRPO 框架可无缝迁移到图像生成。

### 4. 算子融合：从显式 Score 到合并多项式

上述公式 ②③ 要求显式计算 Score，分母含 $\sigma^2$，当 $\sigma \to 0$ 时数值不稳定。本节将公式 ③ 代数化简为关于 $x_t$ 和 $v_\theta$ 的线性多项式，消除显式 Score 计算。

{% note info no-icon %}
**前提条件**：
1. **插值形式**：Rectified Flow 直线插值 $x_t = (1-\sigma)x_0 + \sigma\epsilon$
2. **自适应噪声设计**：$g^2(\sigma) = \frac{\sigma\,\eta^2}{1-\sigma}$（其中 $\eta$ 为超参数，控制探索强度）。**物理直觉**：在生成初期（$\sigma \to 1$），噪声极大，鼓励模型大幅度探索；在生成末期（$\sigma \to 0$），噪声趋于 0，让模型收敛到清晰图像。
3. **Score 近似**：使用 Tweedie 公式通过 $\hat{x}_0 = x_t - \sigma v_\theta$ 估计 Score

三个前提同时成立时，公式 ③ 可化简为无 Score 的封闭形式。
{% endnote %}

**Step 1**：将 ② 代入 ③，注意 $\Delta t = -\Delta\sigma$：

$$\mu = (x_t + v_\theta\Delta\sigma) + \frac{g^2}{2\sigma}(x_t + (1-\sigma)v_\theta)\Delta\sigma$$

**Step 2**：代入 $g^2 = \frac{\sigma\eta^2}{1-\sigma}$，得 $\frac{g^2}{2\sigma} = \frac{\eta^2}{2(1-\sigma)}$：

$$\mu = x_t + v_\theta\Delta\sigma + \frac{\eta^2}{2(1-\sigma)}(x_t + (1-\sigma)v_\theta)\Delta\sigma$$

**Step 3**：展开并按 $x_t$、$v_\theta$ 归类：

$$
\begin{aligned}
\mu &= x_t\left(1 + \frac{\eta^2}{2(1-\sigma)}\Delta\sigma\right) + v_\theta\left(1 + \frac{\eta^2}{2}\right)\Delta\sigma
\end{aligned}
$$

**Step 4**：用代码变量 $\text{std\_dev\_t}^2 = g^2 = \frac{\sigma\eta^2}{1-\sigma}$ 回代，验证系数恒等：

$$\boxed{\mu = x_t\left(1 + \frac{\text{std\_dev\_t}^2}{2\sigma}\,dt\right) + v_\theta\left(1 + \frac{\text{std\_dev\_t}^2(1-\sigma)}{2\sigma}\right)dt} \tag{③'}$$

其中 $dt = \Delta\sigma < 0$。该变换将奇异点从 $\sigma \to 0$（Score 分母 $\sigma^2$）转移至 $\sigma \to 1$（融合公式分母 $1-\sigma$），后者可通过条件替换消除。

---

# Flow-GRPO 实现

本节给出 SDE 单步更新的两种代码实现——标准写法（显式 Score 分解）和合并写法（算子融合），并标注与上述理论公式的对应关系。

Flow-GRPO（arXiv:2505.05470，2025.5.8）与 DanceGRPO（arXiv:2505.07818，2025.5.12）为同期工作，SDE 理论框架一致，代码实现风格不同。

## 1. 标准写法：显式 Score 分解（DanceGRPO `flux_step`）

来自 [DanceGRPO](https://github.com/ByteDance/DanceGRPO) 仓库 `fastvideo/train_grpo_flux.py`。逐步对应公式 ①→②→③→④→⑤：

```python
def flux_step(model_output, latents, eta, sigmas, index, prev_sample, grpo, sde_solver):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma           # Δσ < 0（去噪方向）
    delta_t = sigma - sigmas[index + 1]          # Δt > 0

    # 公式 ①：x̂₀ = x_t − σ·v_θ
    pred_original_sample = latents - sigma * model_output

    # ODE 步进：μ_ODE = x_t + Δσ·v_θ
    prev_sample_mean = latents + dsigma * model_output

    # 噪声标准差：g·√Δt = η·√Δt（常数噪声设计）
    std_dev_t = eta * math.sqrt(delta_t)

    if sde_solver:
        # 公式 ②：Score = −(x_t − x̂₀·(1−σ)) / σ²
        score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2

        # 公式 ③：μ = μ_ODE + (−½η²·Score)·Δσ
        prev_sample_mean = prev_sample_mean + (-0.5 * eta**2 * score_estimate) * dsigma

    # 公式 ④：x_{next} = μ + g√Δt · ε
    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

    # 公式 ⑤：log p = −‖x−μ‖²/(2g²Δt) − log(g√Δt) − ½log(2π)
    if grpo:
        log_prob = (
            -((prev_sample.detach().float() - prev_sample_mean.float()) ** 2)
                / (2 * std_dev_t**2)
            - math.log(std_dev_t)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean, pred_original_sample
```

**局限**：Score 分母 `sigma**2` 在 $\sigma \to 0$ 时数值不稳定；需分配 `pred_original_sample` 和 `score_estimate` 两个中间 tensor。

## 2. 合并写法：算子融合（Flow-GRPO `sde_step_with_logprob`）

来自 [flow_grpo](https://github.com/yifan123/flow_grpo) 仓库 `flow_grpo/diffusers_patch/sd3_sde_with_logprob.py`。对应公式 ③'→④→⑤：

```python
def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
):
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step + 1 for step in step_index]
    sigma = self.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_prev = self.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_max = self.sigmas[1].item()
    dt = sigma_prev - sigma  # < 0

    # 自适应噪声：g(σ) = √(σ/(1-σ)) · η
    std_dev_t = torch.sqrt(
        sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))
    ) * noise_level

    # 公式 ③'：算子融合均值
    prev_sample_mean = (
        sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
        + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
    )

    # 采样：x_{next} = μ + g·√|dt|·ε
    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape, generator=generator,
            device=model_output.device, dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

    # 完整高斯 log-likelihood
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2)
            / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
        - torch.log(std_dev_t * torch.sqrt(-1 * dt))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return prev_sample, log_prob, prev_sample_mean, std_dev_t
```

### 两种写法的对比

| 维度 | 标准写法（`flux_step`） | 合并写法（`sde_step_with_logprob`） |
|:---|:---|:---|
| **对应公式** | ①→②→③→④→⑤ 逐步展开 | ③'→④→⑤ 直接计算 |
| **均值计算** | 分步：$\hat{x}_0$ → Score → $\mu$ | 一步：标量系数 × ($x_t$ + $v_\theta$) |
| **噪声设计** | 常数 $g = \eta$（代码 `std_dev_t` $= g\sqrt{\Delta t}$） | 自适应 $g = \sqrt{\sigma/(1-\sigma)}\cdot\eta$（代码 `std_dev_t` $= g$） |
| **数值奇异点** | $\sigma \to 0$（Score 分母 $\sigma^2$） | $\sigma \to 1$（`torch.where` 防护） |
| **GPU 效率** | 多个中间 tensor 分配 | 两次标量乘加，可融合为单 kernel |
| **log_prob** | 完整三项 | 完整三项 |

**与 LLM GRPO 的结构对比**：整体框架（组采样 → 优势计算 → 裁剪更新 → KL 惩罚）一致。核心差异仅在于对数概率的计算——LLM 用 token 级 softmax 对数概率求和；Flow-GRPO 用高斯转移核的对数密度，按公式 ⑥ 逐步累加得到 $\log\pi_\theta(\text{trajectory}\mid c)$。

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
2. **方差在不同去噪步之间差异极大**，低噪声步方差远小于高噪声步。

{% note info no-icon %}
**深度解析：为什么偏差是"负向"的？为什么方差随时间步变化？**

在 Flow Matching SDE 中，每一步的"策略"是一个高斯分布：$\pi_\theta(x_{t+1}|x_t) = \mathcal{N}(\mu_\theta, \sigma_t^2 I)$。设 $\delta = \mu_\theta - \mu_{\theta_\text{old}}$ 为策略更新导致的均值偏移，$e = x_{t+1} - \mu_{\theta_\text{old}} \sim \mathcal{N}(0, \sigma_t^2 I)$ 为采样噪声，则 log-ratio 可以精确展开为：

$$\log r_t = \frac{e^T \delta}{\sigma_t^2} - \frac{\|\delta\|^2}{2\sigma_t^2}$$

第一项 $\frac{e^T \delta}{\sigma_t^2}$ 是零均值的随机项（因为 $\mathbb{E}[e] = 0$），第二项 $-\frac{\|\delta\|^2}{2\sigma_t^2}$ 是一个**恒为负的常数偏置**。因此：

$$\mathbb{E}[\log r_t] = -\frac{\|\delta\|^2}{2\sigma_t^2} < 0$$

**1. 为什么偏差是负向的？**
由于 $\|\delta\|^2 \geq 0$，这个偏置项恒为非正——这是高斯分布的固有数学性质，与策略更新的方向无关。只要 $\theta \neq \theta_\text{old}$（即策略发生了更新），$\log r_t$ 的均值就通常小于 0。虽然理论上 $\mathbb{E}[r_t] = 1$（对数正态分布的性质），但在图像 latent 的高维空间中（维度 $d \sim 16 \times 64 \times 64 = 65536$），$\|\delta\|^2$ 随维度 $d$ 缩放，导致 $\log r_t$ 的分布极度右偏：绝大多数样本的 $r_t \ll 1$，仅有极少数样本 $r_t \gg 1$ 来拉平均值。在有限样本下，这些极端值很难被观测到，因此经验均值通常远低于 1。

**2. 为什么低噪声步方差小、高噪声步方差大？**
$\log r_t$ 的方差为 $\text{Var}[\log r_t] = \frac{\|\delta\|^2}{\sigma_t^2}$。在低噪声步（$\sigma_t$ 小），高斯分布 $\mathcal{N}(\mu_{\theta_\text{old}}, \sigma_t^2 I)$ 非常"尖锐"，样本 $x_{t+1}$ 紧紧聚拢在均值附近。虽然 log-ratio 的波动范围大，但由于偏置项 $-\|\delta\|^2/(2\sigma_t^2)$ 也极大，几乎所有样本都被压到了 $r_t \approx 0$ 的位置——方差自然极小（大家都挤在零附近）。相反，在高噪声步（$\sigma_t$ 大），高斯分布宽而平坦，偏置项较小，样本的 $r_t$ 在 1 附近分散得更开，方差更大。
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
1. **乘法因子 $c_t$**：用于将不同时间步的方差缩放对齐，解决“低噪声步方差极小、高噪声步方差极大”的问题。
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
- **MixGRPO** 从**采样端**入手：认为问题的根源是标准 SDE 的高频伪影欺骗了奖励模型。解决方法是 CPS 采样（消除伪影）+ Hybrid Inference（推理时混合原始模型限制传播）+ 滑动窗口提升训练效率。

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
