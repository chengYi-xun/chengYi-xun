---
title: 笔记｜生成模型（二十）：Flow-GRPO 与图像生成应用（基于 Flux 的代码解析）
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
> ⬅️ 上一篇：[笔记｜生成模型（十九）：大模型在线 RL 破局者：GRPO 算法详解](/chengYi-xun/posts/20-grpo/)
>
> ➡️ 下一篇：[笔记｜生成模型（二十一）：DAPO：从 GRPO 到大规模推理 RL 的工程实践](/chengYi-xun/posts/22-dapo/)


# 图像生成中的强化学习

**先用一个例子理解为什么需要 RL。**

假设你用一个 Flux 模型生成图像，给定 Prompt："一只橘猫坐在蓝色沙发上"。模型可能生成以下几种结果：

| 生成结果 | 问题 |
|:---|:---|
| 一只白色猫坐在蓝色沙发上 | 颜色不对（应该是橘猫） |
| 一只橘猫站在蓝色沙发旁边 | 动作不对（应该是"坐在"） |
| 一只橘猫坐在蓝色沙发上，画面清晰 | 完美 |
| 一只橘猫坐在蓝色沙发上，但画面模糊 | 质量差 |

传统的训练方式（Flow Matching 损失）只是让模型学会"生成看起来像训练集的图像"。但训练集里可能有模糊的、构图差的、与 Prompt 不一致的图像——模型无法区分好坏。

**RL 的价值**：我们训练一个"美术老师"（奖励模型，如 PickScore 或 ImageReward）来给图像打分。模型自己生成图像 → 美术老师打分 → 模型根据分数调整自己。这就是 RLHF 在图像生成中的应用。

![Flow-GRPO 概览：ODE→SDE 注入随机性、训练期 Denoising Reduction 与组内 GRPO 更新（摘自 Liu et al., arXiv:2505.05470 图 2）](/chengYi-xun/img/flow_grpo_arch.png)

---

# Flow-GRPO 框架解析：图像版"矮子里拔高个"

**先看例子**：对于 Prompt "一只橘猫坐在蓝色沙发上"，我们让 Flux 模型生成 $G = 4$ 张图像，美术老师分别打分：

| 图像 | 描述 | 奖励 $r_i$ | 相对优势 $\hat{A}_i$ |
|:---:|:---|:---:|:---:|
| 图 1 | 橘猫坐沙发，画面清晰 | $r_1 = 0.9$ | $+1.27$ |
| 图 2 | 橘猫坐沙发，稍微模糊 | $r_2 = 0.6$ | $-0.12$ |
| 图 3 | 白猫坐沙发（颜色错） | $r_3 = 0.3$ | $-1.50$ |
| 图 4 | 橘猫坐沙发，普通水平 | $r_4 = 0.7$ | $+0.35$ |

（均值 $\mu_R = 0.625$，标准差 $\sigma_R \approx 0.22$）

跟上一篇 GRPO 的做法完全一样：图 1 和图 4 高于平均（正优势），模型学习生成更像它们的图；图 3 远低于平均（负优势），模型学习远离这种生成方式。**不需要 Critic 网络，只需要多生成几张图做对比。**

**核心思考出发点**：由于像 Flux 这样的图像生成模型参数量达到百亿级别，传统的 PPO 算法由于需要额外的 Critic 网络，显存根本无法承受。因此，Flow-GRPO 采用了 GRPO 算法——彻底抛弃 Critic，用"组内相对评分"来实现高效的在线强化学习。

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
| **奖励** | 只在最后一步（整句完成后打分） | 只在最后一步（整张图生成后打分） |

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

**解决方案**：我们需要一个“指南针”来纠正这种偏离，这个指南针就是 **Score Function（分数函数 $\nabla_{x_t} \log p_t(x_t)$）**。它在数学上永远指向数据分布最密集（最像真实图像）的方向。一旦随机探索让你偏航，Score 就会把你拉回来。

**那么，怎么计算这个 Score 呢？这就用到了统计学中神奇的特威迪公式（Tweedie's Formula）：**

特威迪公式（Tweedie's Formula）证明了一个深刻的结论：**只要你能预测出当前的干净图像 $\hat{x}_0$，就能直接算出当前的 Score。**

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
$$\text{Score} = -\frac{2(x_t - \mu_t)}{2\sigma^2} = -\frac{x_t - \mu_t}{\sigma^2}$$

**第四步：代入均值 $\mu_t$ 与模型预测**
将 $\mu_t = (1-\sigma)x_0$ 代入，得到：
$$\text{Score} = -\frac{x_t - (1-\sigma)x_0}{\sigma^2}$$
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

当 $\Delta t \to 0$ 时，我们将时间倒流记为逆向微元 $\mathrm{d}\bar{t}$，上面的离散更新公式就化为了连续的**逆向 SDE（Reverse SDE，即 Anderson 定理）**：
$$\mathrm{d}x_t = \left[ f(x_t, t) - g(t)^2 \nabla_{x_t} \log p_t(x_t) \right] \mathrm{d}t + g(t) \mathrm{d}\bar{w}_t$$

**公式中各变量的物理意义与代码映射：**

在图像生成（如 Diffusion/Flow Matching）的上下文中，这个逆向 SDE 公式描述了如何从纯噪声逐步去噪生成清晰图像的过程：

- **$t$ 与 $\mathrm{d}t$**：时间步与时间微元。在图像生成中，$t \in [0, 1]$，$t=1$ 对应纯噪声，$t=0$ 对应清晰图像。代码中通常对应离散的时间步长 `dt = t_prev - t_curr`。
- **$x_t$ 与 $\mathrm{d}x_t$**：时刻 $t$ 的状态及其微小变化量。在图像生成中，$x_t$ 代表**正在被去噪的中间潜变量（Latent）或图像**。$\mathrm{d}x_t$ 则是这一步去噪后像素值的变化量。代码中对应 `sample`（当前潜变量）和 `prev_sample - sample`。
- **$f(x_t, t)$**：**漂移项（Drift term）**。物理上代表系统固有的确定性演化速度场。在图像生成中，它代表**神经网络预测的去噪主方向**（即 Probability Flow ODE 的速度）。代码中对应模型输出 `model_output`（如预测的 velocity 或 noise）。
- **$g(t)$**：**扩散系数（Diffusion coefficient）**。物理上代表注入系统的热噪声强度。在图像生成中，它代表**为了探索（Exploration）而刻意注入的随机噪声的标准差**。代码中对应随时间变化的噪声调度参数，如 `std_dev_t`。
- **$\nabla_{x_t} \log p_t(x_t)$**：**得分函数（Score function）**。物理上代表概率密度场在状态空间的梯度。在图像生成中，它是一个向量场，**指向真实图像数据流形的最短路径**（即告诉每个像素如何修改才能更像真实图像）。在基于 Score 的模型中，这通常由神经网络隐式或显式拟合。
- **$- g(t)^2 \nabla_{x_t} \log p_t(x_t)$**：**得分修正项**。物理上是为了抵消布朗运动带来的发散效应而施加的恢复力。在图像生成中，它代表**纠偏项**。因为注入了随机噪声 $g(t) \mathrm{d}\bar{w}_t$，图像可能会偏离真实的数据流形；这个修正项利用 Score function 强行将偏离的像素拉回真实图像的概率分布内。
- **$g(t) \mathrm{d}\bar{w}_t$**：**逆向标准布朗运动微元**。物理上代表纯粹的随机热运动。在图像生成（尤其是 RL 微调）中，它代表**纯随机的像素扰动（高斯白噪声）**，用于在生成轨迹中引入随机性以探索不同的生成结果。代码中对应 `noise = torch.randn_like(sample)`。

**核心结论：**

在图像生成与强化学习微调（如 Flow-GRPO）的结合中，该公式展示了“探索与保真”的数学平衡。为了在生成过程中探索不同的高奖励输出（如更好的美学评分或更准确的 prompt 遵循），必须通过 $g(t) \mathrm{d}\bar{w}_t$ **注入随机噪声**；但纯粹的噪声会破坏图像的真实性（导致伪影或崩坏），因此必须同步引入 $- g(t)^2 \nabla_{x_t} \log p_t(x_t)$ 作为**数学上的强制纠偏力**，确保探索轨迹始终贴合真实图像的数据流形。

**Flow Matching 中的特殊性（ODE 转 SDE）：**
在 Flow Matching 中，我们原本拥有的是一个确定性的常微分方程（ODE），即 $\mathrm{d}x_t = f(x_t, t) \mathrm{d}t$。这个 ODE 已经能够完美地在噪声和图像分布之间映射（它被称为 Probability Flow ODE）。
现在，为了在强化学习中进行“探索”，我们想在这个 ODE 中**强行注入噪声** $g(t) \mathrm{d}\bar{w}_t$，但又**不想改变每个时刻的边缘分布 $p_t(x_t)$**。
根据随机过程理论，如果我们向一个保持分布的 ODE 中注入方差为 $g(t)^2$ 的噪声，为了抵消噪声带来的发散效应，必须在漂移项中加入 $-\frac{1}{2} g(t)^2 \nabla_{x_t} \log p_t(x_t)$ 的修正。

因此，Flow-GRPO 中实际使用的 SDE 漂移项修正系数是 $\frac{1}{2}$：

1. $f(x_t, t)$ 是原本的 ODE 漂移项（基础更新）。
2. $-\frac{1}{2} g(t)^2 \nabla_{x_t} \log p_t(x_t)$ 就是**为了抵消注入噪声带来的分布发散，而必须强制加入的 Score 修正项**。

### 3. 实现公式链：从 SDE 理论到代码

下面将上述 SDE 理论逐步具体化为离散实现公式。设 $t$ 为当前时间步，$v_\theta = v_\theta(x_t, t, c)$ 为模型预测速度场，$g$ 为 SDE 扩散系数（控制探索噪声强度），$\Delta t = t - t_{\text{next}} > 0$。

> **符号对照**：官方代码 `sd3_sde_with_logprob.py` 中，`sigma` = 时间 $t$，`std_dev_t` = 扩散系数 $g(t) = \sqrt{t/(1-t)} \cdot \text{noise\_level}$（随时间自适应），`model_output` = 速度 $v_\theta$，`dt` = $-\Delta t$（符号相反）。下文伪代码中统一使用 `t_now` = $t$，`g` = $g$（简化为常数），`velocity` = $v_\theta$，`dt` = $\Delta t > 0$。其中 `noise_level` 是控制探索强度的超参数（默认 0.7）。

**公式 ①：Tweedie 反推干净样本**

$$\hat{x}_0 = x_t - t \cdot v_\theta \tag{①}$$

> 代码：`pred_x0 = noisy_img - t_now * velocity`

**公式 ②：Score Function（Tweedie 估计）**

将 ① 代入前文推导的 Score 公式 $\nabla_{x_t}\log p_t = -\frac{x_t - (1-t)x_0}{t^2}$：

$$\nabla_{x_t}\log p_t(x_t) \approx -\frac{x_t - (1-t)\hat{x}_0}{t^2} \tag{②}$$

展开化简：$= -\frac{x_t + (1-t)v_\theta}{t}$。

> 代码：`score = -(noisy_img - (1 - t_now) * pred_x0) / (t_now ** 2)`

**公式 ③：SDE 转移均值（ODE 漂移 + Langevin 修正）**

$$\mu = \underbrace{(x_t - \Delta t \cdot v_\theta)}_{\text{ODE 漂移}} + \underbrace{\tfrac{1}{2}g^2 \cdot \nabla_{x_t}\log p_t \cdot \Delta t}_{\text{Langevin 修正}} \tag{③}$$

> 代码：`mu = (noisy_img - dt * velocity) + 0.5 * g**2 * score * dt`
>
> 官方合并写法：`prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt) + model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt`

**为什么不直接用 ODE 均值？** 如果我们只在 ODE 落点 $(x_t - \Delta t \cdot v_\theta)$ 上直接加噪声（跳过 Langevin 修正），噪声会把 $x_{t-\Delta t}$ 的分布"撑大"——相比真实分布 $p_{t-\Delta t}$ 多出了一个额外的高斯扩散。随着步数累积，这种分布偏移会让图像逐渐崩坏。Langevin 修正项沿 Score 方向（即数据高密度区域）对均值做微调，恰好**预先补偿**即将注入的噪声带来的分布膨胀——确保加噪后的采样仍然落在正确的分布内。如果不注入噪声（$g=0$），该修正项为零，公式退化为纯 ODE。

**公式 ④：SDE 采样（添加随机探索噪声）**

$$x_{t-\Delta t} = \mu + g\sqrt{\Delta t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) \tag{④}$$

> 代码：`noisy_img = mu + g * noise * (dt ** 0.5)`

公式 ③④ 合起来构成 SDE 一步的核心操作：**先估计修正均值 $\mu$，再以 $\mu$ 为中心重采样**。此时 $x_{t-\Delta t}$ 的条件分布为 $\mathcal{N}(\mu, \; g^2 \Delta t \cdot I)$。

**边缘分布不变的含义**：③④ 的设计保证了——在**统计意义**上，SDE 采样与纯 ODE 采样生成的图像集合遵循同一分布（质量、多样性、风格等总体特征一致）。但单条轨迹变为随机的：同一个初始噪声，纯 ODE 每次产出相同的图，而 SDE 每次产出不同的图。这种"统计等效但轨迹随机"的性质，正是 GRPO 所需要的——同一 Prompt 下能生成多张不同的图做组内对比（"矮子里拔高个"），且不会因为探索而降低图像质量。

**公式 ⑤：单步对数概率**

由于 $x_{t-\Delta t} \sim \mathcal{N}(\mu, g^2\Delta t \cdot I)$，实际采样到的 $x_{t-\Delta t}$ 离修正均值 $\mu$ 越近，对数概率越高——模型对这条轨迹越"有信心"：

$$\log p_\theta(x_{t-\Delta t} | x_t, c) = -\frac{\|x_{t-\Delta t} - \mu\|^2}{2\,g^2\,\Delta t} \underbrace{- \log(g\sqrt{\Delta t}) - \tfrac{1}{2}\log(2\pi)}_{\text{高斯归一化常数}} \tag{⑤}$$

> 代码：`log_prob_step = -0.5 * ((actual_next - mu) ** 2).sum() / (g ** 2 * dt)`

归一化常数只依赖 $g$ 和 $\Delta t$，不依赖策略参数 $\theta$，因此在 GRPO 的 importance ratio $\exp(\log\pi_\theta^{\text{new}} - \log\pi_\theta^{\text{old}})$ 中分子分母相消。伪代码省略了这些常数项；官方实现保留了完整的 `- log(std_dev_t * sqrt(-dt)) - log(sqrt(2π))` 以便调试。

**公式 ⑥：整条轨迹对数概率**

$$\log \pi_\theta(\text{trajectory} | c) = \sum_{k=1}^{T} \log p_\theta(x_{t_k - \Delta t} | x_{t_k}, c) \tag{⑥}$$

> 代码：`total_log_prob = total_log_prob + log_prob_step`（逐步累加）

这与 LLM 中 token 级对数概率求和的形式完全对应——至此，GRPO 的整套框架可以无缝迁移到图像生成。

---

# Flow-GRPO 完整实现

## 1. 模型定义与奖励函数

```python
import torch
import torch.nn.functional as F

# ── 模型定义：与 LLM GRPO 相同，只需 Actor + Reference ──
# Actor π_θ：当前可训练策略（只训练 LoRA 参数）
actor_model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
actor_model.enable_lora(rank=32)
# Reference π_ref：冻结的参考策略，用于 KL 惩罚（锚定初始分布）
ref_model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
ref_model.requires_grad_(False)

# ── 奖励模型：多个"美术老师"加权打分 ──
reward_models = {
    "aesthetic": AestheticScore(),     # 美学评分（构图、色彩）
    "pick_score": PickScore(),         # 人类偏好综合评分
    "clip_score": CLIPScore(),         # 图文匹配度
}
reward_weights = {"aesthetic": 0.3, "pick_score": 0.5, "clip_score": 0.2}


def compute_reward(images, prompts):
    """多奖励加权求和 → 单个标量回报 r。"""
    return sum(
        w * reward_models[k](images, prompts)
        for k, w in reward_weights.items()
    )


# ── 训练超参 ──
optimizer = torch.optim.AdamW(actor_model.lora_parameters(), lr=1e-5)
G = 4             # 每个 Prompt 生成的图像数（GRPO 组大小）
clip_eps = 0.2    # PPO 裁剪阈值
beta = 0.01       # KL 惩罚系数
num_steps = 50    # 去噪步数
```

## 2. 计算 Flow Matching 轨迹的对数概率

这是 Flow-GRPO 区别于 LLM GRPO 的**核心函数**：

```python
def compute_trajectory_log_prob(model, trajectory, prompt_embeds, g=0.01):
    """
    计算一条去噪轨迹的总对数概率 log π_θ(trajectory | c)。
    依次实现公式 ①→②→③→⑤，按公式 ⑥ 逐步累加。
    """
    total_log_prob = 0.0

    for k in range(len(trajectory) - 1):
        noisy_img, t_now = trajectory[k]
        actual_next, t_next = trajectory[k + 1]
        dt = t_now - t_next                       # Δt > 0

        velocity = model.predict_velocity(noisy_img, t_now, prompt_embeds)

        # 公式 ①  x̂₀ = x_t − t·v_θ
        pred_x0 = noisy_img - t_now * velocity

        # 公式 ②  Score = −(x_t − (1−t)·x̂₀) / t²
        score = -(noisy_img - (1 - t_now) * pred_x0) / (t_now ** 2)

        # 公式 ③  μ = (x_t − Δt·v_θ) + ½·g²·Score·Δt
        mu = (noisy_img - dt * velocity) + 0.5 * g**2 * score * dt

        # 公式 ⑤  log p = −‖x_{t−Δt} − μ‖² / (2·g²·Δt)
        variance = g ** 2 * dt
        log_prob_step = -0.5 * ((actual_next - mu) ** 2).sum() / variance

        # 公式 ⑥  逐步累加
        total_log_prob = total_log_prob + log_prob_step

    return total_log_prob
```

## 3. 在线采样：生成图像并记录轨迹

```python
def generate_with_trajectory(model, prompt_embeds, num_steps, g=0.01):
    """
    生成一张图像，并记录完整去噪轨迹供后续计算 log π_θ。
    依次实现公式 ①→②→③→④。
    """
    noisy_img = torch.randn(1, 16, 64, 64)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
    trajectory = [(noisy_img.clone(), timesteps[0])]

    for i in range(num_steps):
        t_now  = timesteps[i]
        t_next = timesteps[i + 1]
        dt = t_now - t_next                                     # Δt > 0

        with torch.no_grad():
            velocity = model.predict_velocity(noisy_img, t_now, prompt_embeds)

        # 公式 ①  x̂₀ = x_t − t·v_θ
        pred_x0 = noisy_img - t_now * velocity

        # 公式 ②  Score = −(x_t − (1−t)·x̂₀) / t²
        score = -(noisy_img - (1 - t_now) * pred_x0) / (t_now ** 2)

        # 公式 ③  μ = (x_t − Δt·v_θ) + ½·g²·Score·Δt
        mu = (noisy_img - dt * velocity) + 0.5 * g**2 * score * dt

        # 公式 ④  x_{t−Δt} = μ + g·√Δt·ε
        noise = torch.randn_like(noisy_img)
        noisy_img = mu + g * noise * (dt ** 0.5)

        trajectory.append((noisy_img.clone(), t_next))

    image = vae_decode(noisy_img)
    return image, trajectory
```

## 4. 完整训练循环

```python
for step in range(total_steps):
    prompts_batch = sample_prompts(dataset, batch_size=4)
    prompt_embeds = encode_text(prompts_batch)

    # ══════════ 阶段 1：组内采样（每个 Prompt 生成 G 张图像）══════════
    all_images, all_trajectories = [], []
    all_old_logps, all_ref_logps = [], []

    with torch.no_grad():  # 采样阶段不需要梯度
        for emb in prompt_embeds:
            for _ in range(G):  # 同一 Prompt 下生成 G 条轨迹（GRPO 的"组"）
                image, traj = generate_with_trajectory(actor_model, emb, num_steps)

                # 记录采样时 Actor 和 Reference 在同一轨迹上的 log π
                old_logp = compute_trajectory_log_prob(actor_model, traj, emb)
                ref_logp = compute_trajectory_log_prob(ref_model, traj, emb)

                all_images.append(image)
                all_trajectories.append(traj)
                all_old_logps.append(old_logp)      # log π_old（ratio 的分母）
                all_ref_logps.append(ref_logp)       # log π_ref（KL 惩罚项）

    # ══════════ 阶段 2：奖励打分 → 组内相对优势 ══════════
    rewards = compute_reward(all_images, repeat_prompts(prompts_batch, G))  # (batch*G,)
    rewards_grouped = rewards.reshape(-1, G)  # (batch, G)

    # GRPO 核心：用组内均值/标准差代替 Critic
    mean_r = rewards_grouped.mean(dim=1, keepdim=True)
    std_r  = rewards_grouped.std(dim=1, keepdim=True)
    advantages = ((rewards_grouped - mean_r) / (std_r + 1e-8)).reshape(-1)  # z-score

    old_logps = torch.stack(all_old_logps)  # (batch*G,)
    ref_logps = torch.stack(all_ref_logps)  # (batch*G,)

    # ══════════ 阶段 3：多 epoch PPO 式策略更新 ══════════
    for epoch in range(K_epochs):
        # 用当前 θ 重算 log π_θ（这次需要梯度，用于反传）
        new_logps = torch.stack([
            compute_trajectory_log_prob(actor_model, traj, emb)
            for traj, emb in zip(all_trajectories, repeat_embeds(prompt_embeds, G))
        ])  # (batch*G,)

        # 重要性采样比 r(θ) = π_θ / π_old
        ratio = torch.exp(new_logps - old_logps)

        # PPO 裁剪目标：min(r·A, clip(r)·A)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL 惩罚：防止策略偏离参考模型过远
        kl_loss = (new_logps - ref_logps).mean()

        loss = policy_loss + beta * kl_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_model.lora_parameters(), max_norm=1.0)
        optimizer.step()
```

> **与 LLM GRPO 的代码结构对比**：整体框架（组采样 → 优势计算 → 裁剪更新 → KL 惩罚）完全一致。唯一的区别在于**对数概率的计算方式**：LLM 用 token 级 softmax 对数概率求和，Flow-GRPO 用高斯转移核的对数密度求和。

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

# 算法对比与开源生态

| 维度 | Diffusion-DPO | DDPO (PPO) | Flow-GRPO |
|:---:|:---:|:---:|:---:|
| **训练方式** | 离线（偏好对） | 在线 RL | 在线 RL |
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

> 下一篇：[笔记｜生成模型（二十一）：DAPO：从 GRPO 到大规模推理 RL 的工程实践](/chengYi-xun/posts/22-dapo/)
