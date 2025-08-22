---
title: 笔记｜生成模型（八）：SDE统一DDPM和SMLD
date: 2025-08-21 23:08:30
cover: false
mathjax: true
categories:
 - Notes
tags:
 - Deep learning
 - Generative models theory
series: Score Base Models theory
---


# 随机微分方程简介

首先介绍常微分方程和随机微分方程的基本知识，为后续讨论奠定基础。我们先从一个常微分方程（ODE）例子开始：

$$
\frac{\mathrm{d}x}{\mathrm{d}t}=f(x,t)\quad\mathrm{or}\quad\mathrm{d}x=f(x,t)\mathrm{d}t
$$

其中 $f(x,t)$ 是关于 $x$ 和 $t$ 的函数，描述了 $x$ 随时间的变化趋势，如下图左侧所示。

![常微分方程与随机微分方程](/chengYi-xun/img/ode.jpeg)

直观地讲，$f(x,t)$ 对应于图中的青色箭头，确定某一时刻的 $x(t)$ 后，沿着箭头方向可以找到下一时刻的 $x(t+\Delta t)$。这个常微分方程的解析解为：

$$
x(t)=x(0)+\int_0^t f(x,\tau)\mathrm{d}\tau
$$

然而，实际应用中我们使用的 $f(x,t)$ 通常是比较复杂的函数（如神经网络），求解析解往往不切实际。因此，通常采用迭代法求数值解：

$$
x(t+\Delta t)\approx x(t)+f(x(t),t)\Delta t
$$

在迭代过程中，每次沿箭头方向线性移动一小段距离，经多次迭代得到解析解的近似，如上图左侧绿色曲线所示。

从上述描述可知，常微分方程描述了一个确定性过程。对于非确定性过程（如从分布中采样），则需要使用随机微分方程（SDE）描述。SDE 相比 ODE 形式上增加了一个高斯噪声项：

$$
\frac{\mathrm{d}x}{\mathrm{d}t}=\underbrace{f(x,t)}_{漂移系数}+\underbrace{\sigma(x,t)}_{扩散系数}\omega_t\quad\mathrm{or}\quad\mathrm{d}x=f(x,t)\mathrm{d}t+\sigma(x,t)\mathrm{d}\omega_t
$$

采样时类似 ODE，可进行迭代采样：

$$
x(t+\Delta t)\approx x(t)+f(x(t),t)\Delta t+\sigma(x(t),t)\sqrt{\Delta t}\mathcal{N}(0,I)
$$

由于采样过程中存在高斯噪声，多次采样会得到不同轨迹，如上图右侧的一系列绿色折线所示。

# 基于 SDE 的 Score-based Models

在上一篇文章中我们介绍了基于分数的生成模型的基本概念。添加多个噪声尺度对于基于分数的生成模型至关重要。将噪声尺度数量推广到无穷大，不仅可获得更高质量的样本，还可以精确计算对数似然并实现可控的逆问题求解生成。

## 使用 SDE 描述扰动过程

当噪声尺度数量趋于无穷大时，扰动过程可视为连续时间内的随机过程，如下图所示，这与扩散模型的加噪过程有相似之处。

![连续的加噪过程](/chengYi-xun/img/ode.gif)

为表示上述随机过程，可用随机微分方程描述：

$$
\mathrm{d}x=f(x,t)\mathrm{d}t+g(t)\mathrm{d}w
$$

用 $p_t(x)$ 表示 $x(t)$ 的概率密度函数，可知 $p_0(x)=p(x)$ 是未加噪时的真实数据分布，经过足够多时间步 $T$ 的扰动后，$p_T(x)$ 接近先验分布 $\pi(x)$。从这一角度看，扰动过程与扩散模型的扩散过程本质上是一致的。

扰动随机过程可采用多种形式的 SDE，例如：

$$
\mathrm{d}x=e^t\mathrm{d}w
$$

这表示用均值为 0、方差呈指数增长的高斯噪声对分布进行扰动。

## 使用反向 SDE 进行采样

在上一章节中，我们推导了Score Matching 的离散过程（离散形式的SDE），可用朗之万动力学采样进行生成新样本。然而当正向过程改为连续形态的 SDE 描述后，逆向过程也需相应变化。对于给定 SDE，其逆向过程同样是一个 SDE，表示为（推导过程见[这个链接](https://kexue.fm/archives/9209)）：

$$
\mathrm{d}x=\left[f(x,t)-g^2(t)\textcolor{green}{\nabla_x\log p_t(x)}\right]\mathrm{d}t+g(t)\mathrm{d}w
$$

此处 $\mathrm{d}t$ 表示反向时间梯度，即从 $t=T$ 到 $t=0$ 的方向。上式中绿色部分正是我们在上一篇文章中介绍的 score function $s_\theta(x,t)$。这表明，虽从离散形式变为连续形式，但学习目标仍然一致的——即用网络学习分布的 score function。

获得 score function 后，可从反向 SDE 中采样，最简单的方法是欧拉–丸山方法（Euler-Maruyama） 方法：
$$
\begin{aligned}
\Delta x &\leftarrow[f(x,t)-g^2(t)s_\theta(x,t)]\Delta t+g(t)\sqrt{|\Delta t|}z_t\\
x &\leftarrow x+\Delta x\\
t &\leftarrow t+\Delta t
\end{aligned}
$$

其中 $z\sim\mathcal{N}(0,I)$，可通过直接对高斯噪声采样得到。上式中 $f(x,t)$ 和 $g(t)$ 均有解析形式，$\Delta t$ 可选取较小值，仅 $s_\theta(x,t)$ 为参数模型。采样过程可通过下图直观感受：

![通过反向扰动过程进行采样](/chengYi-xun/img/rever_ode.gif)

{% note info no-icon %}

Euler-Maruyama 方法是求解随机微分方程 (SDE) 的一种数值方法，是常规 Euler 方法在处理随机过程时的扩展。对于一个标准形式的 SDE：

$$\mathrm{d}x = f(x,t)\mathrm{d}t + g(t)\mathrm{d}w_t$$

其中 $f(x,t)$ 是漂移项，$g(t)$ 是扩散系数，$w_t$ 是维纳过程（布朗运动）。

Euler-Maruyama 的离散近似为：

$$x_{t+\Delta t} = x_t + f(x_t,t)\Delta t + g(t)\sqrt{\Delta t}z_t$$

其中 $z_t \sim \mathcal{N}(0,I)$ 是从标准正态分布采样的随机变量，$\Delta t$ 是时间步长。

**性质：**

1. **弱收敛性**：当 $\Delta t \to 0$ 时，数值解收敛到真实解的分布
2. **强收敛性**：对于光滑系数的 SDE，收敛阶为 $\mathcal{O}(\sqrt{\Delta t})$
3. **简单性**：计算简单，易于实现，是求解 SDE 最基本的方法
4. **稳定性**：对于步长 $\Delta t$ 较大时可能不稳定，需要选择足够小的步长
5. **确定性+随机性**：包含确定性部分 $f(x_t,t)\Delta t$ 和随机部分 $g(t)\sqrt{\Delta t}z_t$

**与经典 Euler 方法的区别：**

- 经典 Euler 方法只处理确定性部分，适用于 ODE
- Euler-Maruyama 方法增加了随机项，适用于 SDE
- 随机项的缩放因子为 $\sqrt{\Delta t}$ 而非 $\Delta t$（这是由维纳过程的性质决定的）


**维纳过程与 $\sqrt{\Delta t}$ 缩放因子的关系：**

维纳过程（Wiener process）是布朗运动的数学模型，具有如下性质：
$$\text{Var}(w_{t+\Delta t} - w_t) = \Delta t$$
即维纳过程 $w_t$ 在时间间隔 $\Delta t$ 内的变化量的方差与 $\Delta t$ 成正比，这意味着增量服从均值为 0，方差为 $\Delta t$ 的正态分布：
$$w_{t+\Delta t} - w_t \sim \mathcal{N}(0, \Delta t)$$

将标准正态随机变量 $z \sim \mathcal{N}(0,1)$ 转换为维纳过程的增量，需要：
$$w_{t+\Delta t} - w_t = \sqrt{\Delta t} \cdot z$$
（根据正态分布的性质，如果 $z \sim \mathcal{N}(0,1)$，那么 $\sqrt{\Delta t} \cdot z \sim \mathcal{N}(0, \Delta t)$ ）
因此在 Euler-Maruyama 方法中，随机项系数为：
$$g(t) \mathrm{d}w_t \approx g(t)\sqrt{\Delta t}z_t$$



**这与常规 Euler 方法的区别**：

- ODE 中的变化率与时间成正比（一阶关系）：$\Delta x \propto \Delta t$
- 而维纳过程的变化与时间的平方根成正比（二次变差关系）：$\Delta w \propto \sqrt{\Delta t}$

这是随机过程与确定性过程的本质区别，也是为什么 SDE 需要特殊的数值求解方法。

{% endnote %}


## 使用 score matching 进行训练

在反向 SDE 采样过程中，需要学习 score function $s_\theta(x,t)\approx\nabla_x\log p_t(x)$。为对其进行估计，同样可使用 score matching 方式训练。优化目标为：

$$
\mathbb{E}_{t\in\mathcal{U}(0,T)}\mathbb{E}_{p_t(x)}\left[\lambda(t)||\nabla_x\log p_t(x)-s_\theta(x,t)||_2^2\right]
$$

可见仍使用 L2 损失优化，但不再简单对所有噪声求和，而是计算均匀时间分布 $[0,T]$ 范围内损失的期望。另一不同点是权重选取变为 $\lambda(t)\propto 1/\mathbb{E}[||\nabla_{x(t)}\log p(x(t)|x(0))||_2^2]$。

值得讨论的是，在离散情况下，$\lambda(t)$ 的选取为 $\lambda(t)=\sigma_t^2$。若此处也使用类似形式 $\lambda(t)=g^2(t)$，可推导出 $p_0(x)$ 和 $p_\theta(x)$ 之间的 KL 散度与上述损失间的关系：

$$
\mathrm{KL}(p_0(x)||p_\theta(x))\le\frac{T}{2}\mathbb{E}_{t\in\mathcal{U}(0,T)}\mathbb{E}_{p_t(x)}\left[\lambda(t)||\nabla_x\log p_t(x)-s_\theta(x,t)||_2^2\right]+\mathrm{KL}(p_T||\pi)
$$

这里的 $\lambda(t)=g^2(t)$ 被称为 likelihood weighting function，通过使用该加权函数，可学习到良好的分布。这表明连续表示方式和离散表示方式在本质上是统一的。（反向训练关系这里过于抽象，看不懂，抄来的，hhh）

# 讨论

建立完整的基于 SDE 的 score-based modeling 框架后，还有三个方面值得讨论。

## 和 DDPM 的联系

通过上文介绍，我们发现用 SDE 描述的 score-based model 与扩散模型有许多相似之处。在 DDPM 中，前向过程描述为：

$$
x_{t}=\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\epsilon_{t-1},\quad\epsilon_{t-1}\sim\mathcal{N}(0,I)
$$

这是一个离散过程，$t\in\{0,1,\cdots,T\}$。为将 DDPM 转变为连续形式，可将所有时间步除以 $T$，即 $t\in\{0,\frac{1}{T},\cdots,\frac{T-1}{T},1\}$。当 $T\rightarrow\infty$，DDPM 变为连续过程。代入上式：

$$
x(t+\Delta t)=\sqrt{1-\beta(t+\Delta t)\Delta t}~x(t)+\sqrt{\beta(t+\Delta t)\Delta t}~\epsilon(t)
$$

泰勒展开后近似得到：

$$
\begin{aligned}
x(t+\Delta t)&=\sqrt{1-\beta(t+\Delta t)\Delta t}~x(t)+\sqrt{\beta(t+\Delta t)\Delta t}~\epsilon(t)\\
&\approx x(t)-\frac{1}{2}\beta(t+\Delta t)\Delta t~x(t)+\sqrt{\beta(t+\Delta t)\Delta t}~\epsilon(t)\\
&\approx x(t)-\frac{1}{2}\beta(t)\Delta t x(t)+\sqrt{\beta(t)\Delta t}\epsilon(t)
\end{aligned}
$$

当 $T\rightarrow\infty$，即 $\Delta t\rightarrow0$ 时，有：

$$
\mathrm{d}x=-\frac{\beta(t)x}{2}\mathrm{d}t+\sqrt{\beta(t)}\mathrm{d}w
$$

推导表明，从 DDPM 前向过程出发，可得到与 score-based model 形式相符的 SDE 方程，因此也可使用 score matching、Langevin MCMC 等策略进行学习和采样。详细推导可参见 *[Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)* 附录 B。

## 将 SDE 转化为 ODE 概率流

使用 Langevin MCMC 和 SDE 虽可获得较好的采样效果，但这两种方式仅能对 log-likelihood 进行估计，无法精确计算。通过将 SDE 转化为 ODE，可精确计算 log-likelihood（SDE 和 ODE 的关系类似于 DDPM 和 DDIM 的关系）。

可以在不改变 $p_t(x)$ 概率分布的前提下将 SDE 转化为 ODE：

$$
\mathrm{d}x=\left[f(x,t)-\frac{1}{2}g^2(t)\nabla_x\log p_t(x)\right]\mathrm{d}t
$$

两者关系如下图所示，ODE 概率流比 SDE 更平滑，且最终得到的分布与 SDE 相同。由于 ODE 是确定性的，前向和反向过程都可逆，因此 ODE 概率流与 normalizing flow 有相似之处。

![SDE 和 ODE 比较](/chengYi-xun/img/ode2.jpeg)

## 条件生成

DDPM 难以推导出条件概率的形式，使用 DDPM 进行条件生成较难显式实现（尽管可通过 classifier guidance 等隐式方式实现）。而 SDE 不存在此问题，可显式解决条件生成问题。

形式化表述：给定随机变量 $y$ 和 $x$，已知前向过程概率分布 $p(y|x)$，以 $y$ 为条件生成 $x$ 可表示为：

$$
p(x|y)=\frac{p(x)p(y|x)}{\int p(x)p(y|x)\mathrm{d}x}
$$

两侧求梯度，得到：

$$
\nabla_x\log p(x|y)=\nabla_x\log p(x)+\nabla_x\log p(y|x)
$$

由于 $\nabla_x\log p(x)$ 可通过 score matching 建模，且已知 $p(y|x)$，先验分布 $\nabla_x\log p(y|x)$ 也较易求得。因此可求得后验分布梯度 $\nabla_x\log p(x|y)$，再使用 Langevin MCMC 采样实现条件生成。

# 总结

本文介绍了基于 SDE 进行 score-based 建模的方法。相比上一篇文章，使用 SDE 主要将离散形式的扰动过程转变为连续形式，而训练方式、采样方式与离散形式大同小异。通过指定特定形式的 $f(x,t)$ 和 $g(t)$，可获得与 DDPM 相同的性质；通过将 SDE 转化为 ODE，则与 normalizing flow 相似。可见 SDE 是一个通用的描述框架，统一了多种生成模型的视角。

> 参考资料：
>
> 1. [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/#score-based-generative-modeling-with-multiple-noise-perturbations)
> 2. [CVPR 2022 Tutorial: Denoising Diffusion-based Generative Modeling: Foundations and Applications](https://cvpr2022-tutorial-diffusion-models.github.io/)
> 3. [一文解释 Diffusion Model (二) Score-based SDE 理论推导](https://zhuanlan.zhihu.com/p/589106222)
> 4. [基于 SDE 的模型](https://littlenyima.github.io/posts/17-score-based-modeling-with-sde/)