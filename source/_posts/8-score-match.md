---
title: 笔记｜生成模型（七）：Score-Base理论
date: 2025-08-16 23:08:30
cover: false
mathjax: true
categories:
 - Notes
tags:
 - Deep learning
 - Generative models theory
series: Score Base Models theory
---

# Score-based 模型理论基础

生成模型通常从某个已有的概率分布中进行采样以生成样本。Score-based 模型的关键在于对概率分布的对数梯度，即 **Score Function** 的建模。为了学习这个对象，我们需要使用一种称为 **score matching** 的技术，这是 Score-based 模型名称的由来。

## Score Function 和 Score-based Models

考虑一个数据集 $\{x_1, x_2, \ldots, x_N\}$ 的概率分布 $p(x)$，我们可以使用一种通用的方式——能量模型（Energy-based model）的形式表示这个概率分布：

$$
p_\theta(x) = \frac{\exp(-f_\theta(x))}{Z_\theta}
$$

{% note info no-icon %}

**能量模型定义：**

对于数据 $x \in \mathbb{R}^D$，能量模型定义概率分布为：

$$p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z_\theta}$$

其中：
- $E_\theta(x)$ 为能量函数，由参数 $\theta$ 参数化
- $Z_\theta = \int \exp(-E_\theta(x)) \mathrm{d}x$ 为配分函数，确保归一化条件 $\int p_\theta(x) \mathrm{d}x = 1$ 成立

**性质：**

- 低能量对应高概率：$E_\theta(x)$ 越小，$p_\theta(x)$ 越大
- 高能量对应低概率：$E_\theta(x)$ 越大，$p_\theta(x)$ 越小
- 连续性：能量函数通常是连续可微的

{% endnote %}



$Z_\theta$ 是归一化因子，定义为：

$$
Z_\theta = \int \exp(-f_\theta(x)) \mathrm{d} x
$$

因此，有了上式通用的概率模型定义方式，我们一般可以通过最大化对数似然的方式来训练参数，使得网络模型的分布逼近真实数据的分布：

$$
\max_{\theta} \sum_{i=1}^N \log p_\theta(x_i)
$$
但是因为
$$\log p_\theta(x) = -f_\theta(x) - \log Z_\theta$$
$Z_\theta$ 是难处理的，我们无法求出 $\log p_\theta(x)$，自然也就无法优化参数 $\theta$。这个问题已经有了几种不同的解决方案，例如正则化流通过保证网络可逆来使得每一步的 $Z_\theta$ 恒定为1；VAE 改用变分推断逼近并学习距离的变分下界等。

在Score-based Models中，为了解决归一化项无法计算的问题，引入了 Score Function。定义如下：

$$
{s}_\theta(x) = \nabla_x \log p_\theta(x)
$$

因为 $Z_\theta$ 是常数，所以它的梯度为零，从而有：

$$
{s}_\theta(x) = \nabla_x \log p_\theta(x) = -\nabla_x f_\theta(x)
$$

这表明 Score Function 实际上对应于神经网络的梯度。因此我们可以构建优化目标为：

$$
\theta = \arg\min_{\theta} \mathbb{E}_{p(x)}\left[ \left\| \nabla_x \log p(x) - {s}_\theta(x) \right\|_2^2 \right]
$$

使得模型预测的 Score Function 与真实分布的 Score Function 之间的均方误差最小。然而真实分布的log梯度 $\nabla_x \log p(x)$ 是未知的，我们需要利用 **Score Matching** 方法，Score Matching可以让我们在不知道真实分布的 $p(x)$ 的情况下最小化这个Loss。

## Score Matching

我们把上述的损失函数记为 $\mathcal{L}$，并进行展开：

$$
\begin{aligned}
\mathcal{L} &= \mathbb{E}_{p(x)}\left[ \left\| \nabla_x \log p(x) - s_\theta(x) \right\|^2 \right] \\
&= \int p(x) \left\| \nabla_x \log p(x) - s_\theta(x) \right\|^2 \mathrm{d}x
\end{aligned}
$$


进一步展开平方项：

$$
= \int p(x) \left[ \left\| \nabla_x \log p(x) \right\|^2 + \left\| s_\theta(x) \right\|^2 - 2 (\nabla_x \log p(x))^T s_\theta(x) \right] \mathrm{d}x
$$

第一项是常数项，与参数 $\theta$ 无关，可以忽略。第二项可以通过数据样本直接计算。现在我们关注第三项：

$$
-2 \int p(x) (\nabla_x \log p(x))^T s_\theta(x) \mathrm{d}x
$$

对于 $x$ 的维度 $N$，我们可以将其展开为：

$$
= -2 \int p(x) \sum_{i=1}^{N} \frac{\partial \log p(x)}{\partial x_i} s_{\theta i}(x) \mathrm{d}x
$$

利用对数导数的链式法则 $\frac{\partial \log p(x)}{\partial x_i} = \frac{1}{p(x)} \frac{\partial p(x)}{\partial x_i}$：

$$
= -2 \int p(x) \sum_{i=1}^{N} \frac{1}{p(x)} \frac{\partial p(x)}{\partial x_i} s_{\theta i}(x) \mathrm{d}x
$$

$$
= -2 \sum_{i=1}^{N} \int \frac{\partial p(x)}{\partial x_i} s_{\theta i}(x) \mathrm{d}x
$$

对每个分量应用分部积分法，令 $u = s_{\theta i}(x)$，$\mathrm{d}v = \frac{\partial p(x)}{\partial x_i} \mathrm{d}x$：

$$
= -2 \sum_{i=1}^{N} \left[ \left. p(x) s_{\theta i}(x) \right|_{-\infty}^{\infty} - \int p(x) \frac{\partial s_{\theta i}(x)}{\partial x_i} \mathrm{d}x \right]
$$

假设当 $|x| \rightarrow \infty$ 时，$p(x) \rightarrow 0$，则边界项为零：

$$
\begin{aligned}
&= -2 \sum_{i=1}^{N} \left[ - \int p(x) \frac{\partial s_{\theta i}(x)}{\partial x_i} \mathrm{d}x \right] \\
&= 2 \sum_{i=1}^{N} \int p(x) \frac{\partial s_{\theta i}(x)}{\partial x_i} \mathrm{d}x \\
&= 2 \int p(x) \sum_{i=1}^{N} \frac{\partial s_{\theta i}(x)}{\partial x_i} \mathrm{d}x
\end{aligned}
$$


{% note info no-icon %}

**雅可比矩阵与迹的关系**：对于 $s_\theta(x)$，其雅可比矩阵 $\nabla_x s_\theta(x)$ 定义为：

$$\nabla_x s_\theta(x) = 
\begin{bmatrix} 
\frac{\partial s_{\theta 1}}{\partial x_1} & \frac{\partial s_{\theta 1}}{\partial x_2} & \cdots & \frac{\partial s_{\theta 1}}{\partial x_N} \\
\frac{\partial s_{\theta 2}}{\partial x_1} & \frac{\partial s_{\theta 2}}{\partial x_2} & \cdots & \frac{\partial s_{\theta 2}}{\partial x_N} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial s_{\theta N}}{\partial x_1} & \frac{\partial s_{\theta N}}{\partial x_2} & \cdots & \frac{\partial s_{\theta N}}{\partial x_N}
\end{bmatrix}$$

矩阵的迹（trace）是对角线元素的和：
$$\mathrm{tr}(\nabla_x s_\theta(x)) = \sum_{i=1}^{N} \frac{\partial s_{\theta i}}{\partial x_i}$$

因此：
$$\nabla \cdot s_\theta(x) = \mathrm{tr}(\nabla_x s_\theta(x))$$

{% endnote %}

所以我们有：
$$\sum_{i=1}^{N} \frac{\partial s_{\theta i}(x)}{\partial x_i} = \mathrm{tr}(\nabla_x s_\theta(x))$$

最终表达式为：
$$= 2 \int p(x) \mathrm{tr}(\nabla_x s_\theta(x)) \mathrm{d}x$$

因此，最终的优化目标为：

$$
\mathcal{L} = \int p(x) \left\| s_\theta(x) \right\|^2 \mathrm{d}x + 2 \int p(x) \mathrm{tr}(\nabla_x s_\theta(x)) \mathrm{d}x
$$

写成期望形式：

$$
\mathcal{L} = \mathbb{E}_{p(x)} \left[ \left\| s_\theta(x) \right\|^2 + 2 \mathrm{tr}(\nabla_x s_\theta(x)) \right] 
$$

这个优化目标的重要意义在于：

1. 它完全避开了对 $Z_\theta$ 的计算
2. 不再依赖于真实分布 $p(x)$ 的 Score Function
3. 只需要通过数据样本就可以直接优化



## 从分布采样：郎之万动力学采样（Langevin Dynamics）

现在我们已经通过神经网络学习到了数据分布的Score Function，那么如何用Score Function从这个数据分布中得到样本呢？答案就是朗之万动力学采样：

1. 从任意先验分布中采样初始状态 $x_0 \sim \pi(x)$。
2. 迭代更新：

$$
x_{i+1} \leftarrow x_i + \epsilon \nabla_{x} \log p(x) + \sqrt{2\epsilon} {z}_i, \quad i = 0, 1, \ldots, K
$$

其中 $ {z}_i \sim \mathcal{N}(0, I)$，当 $\epsilon \rightarrow 0$ 且 $K \rightarrow \infty$ 时，$x_K$ 会收敛到从 $p(x)$ 直接采样的结果。

这样我们其实就得到了一个生成模型。我们可以先训练一个网络用来估计Score Function，然后用Langevin Dynamics和网络估计的Score Function采样，就可以得到原分布的样本。因为整个方法由Score Matching和Langevin Dynamics两部分组成，所以叫SMLD。


## Pitfall陷阱

现在我们得到了SMLD生成模型，但实际上这个模型由很大的问题。
$$
\begin{aligned}
\mathcal{L} = \int p(x) \left\| \nabla_x \log p(x) - s_\theta(x) \right\|^2 \mathrm{d}x
\end{aligned}
$$
观察我们用来训练神经网络的损失函数，我们可以发现这个L2项其实是被 $p(x)$ 加权了。所以对于低概率的区域，估计出来的Score Function就很不准确。

![Pitfall陷阱](/chengYi-xun/img/pillfall.jpg)

对于上面这张图来说，只有在高概率的红色区域，loss才高，Score Function可以被准确地估计出来。但如果我们采样的初始点在低概率区域的话，因为估计出的 Score Function 不准确，很有可能生成不出真实分布的样本。

那怎么样才能解决上面的问题呢？其实可以通过给数据增加噪声扰动的方式扩大高概率区域的面积。给原始分布加上高斯噪声，原始分布的方差会变大。这样相当于高概率区域的面积就增大了，更多区域的Score Function可以被准确地估计出来。

![fix](/chengYi-xun/img/pillow-fix.jpg)

但是噪声扰动的强度如何控制是个问题：

1. 强度太小起不到效果，高概率区域的面积还是太小
2. 强度太大会破坏数据的原始分布，估计出来的Score Function就和原分布关系不大了

这里作者给出的解决方法是加不同程度的噪声，让网络可以学到加了不同噪声的原始分布的Score Function。

对于使用高斯噪声进行扰动的情况，可以使用 $L$ 个带有不同方差 $\sigma_1<\sigma_2<\cdots<\sigma_L$ 的高斯分布 $\mathcal{N}(0,\sigma_i^2I),i=1,2,\cdots,L$ 分别对原始分布 $p(\mathbf{x})$ 进行扰动：
$$
p_{\sigma_i}(\mathbf{x})=\int p(\mathbf{y})\mathcal{N}(\mathbf{x};\mathbf{y},\sigma_i^2I)\mathrm{d}\mathbf{y}
$$
从 $p_{\sigma_i}(\mathbf{x})$ 中采样是比较容易的，和 diffusion 中的重参数化技巧类似：先采样 $\mathbf{x}\sim p(\mathbf{x})$，再计算 $\mathbf{x}+\sigma_i\mathbf{z}$，其中 $\mathbf{z}\sim\mathcal{N}(0,I)$。

获得一系列用噪声进行扰动过的分布后，依然是对每一个分布进行 score matching，对于 $\nabla_\mathbf{x}\log p_{\sigma_i}(\mathbf{x})$ 得到一个与噪声有关的 Score Function $\mathbf{s}_\theta(\mathbf{x},i)$。总体上的优化目标是对所有的这些分布 score matching 优化目标的加权：
$$
\sum_{i=1}^L\lambda(i)\mathbb{E}_{p_{\sigma_i}(\mathbf{x})}\left[||\nabla_\mathbf{x}\log p_{\sigma_i}(\mathbf{x})-\mathbf{s}_\theta(\mathbf{x},i)||_2^2\right]
$$
对于加权权重的选择，通常直接指定 $\lambda(i)=\sigma_i^2$。这样我们就获得了一系列用不同的高斯噪声扰动过的分布，直观地看，扰动程度比较小的分布更接近真实分布，能在高概率密度的区域提供比较好的估计；扰动程度比较大的分布则能在低概率密度的区域提供比较好的估计，带有不同扰动程度的分布形成了一种比较互补的关系，有利于提高概率建模质量。

采样的过程依然是进行一系列迭代，不过因为有多个分布，所以需要依次对每个分布迭代一遍，相当于一共迭代 $L\times T$ 轮，得到最终的结果。这种采样方法叫做 Annealed Langevin Dynamics，具体的采样算法可以参考[这个链接](https://uvadl2c.github.io/lectures/Advanced%20Generative%20&%20Energy-based%20Models/modern-based-models/lecture%204.2.pdf)的内容。


# 总结

作为生成模型的一种，score-based model 也遵循学习+采样的范式，其学习过程使用 score matching 来间接学习分布，采样过程使用 Langevin dynamics 通过迭代过程进行采样（和 diffusion models 的采样过程有点类似）。在训练时由于低概率密度区域会有比较低的权重，所以这部分区域无法准确学习，为了解决这个问题，又使用 multiple noise pertubation 和 annealed Langevin dynamics 进行了改进。

> 参考资料：
>
> 1. [Score-based Generative Models](https://littlenyima.github.io/posts/16-score-based-generative-models/)
> 2. [Score-based Generative Models总结](https://www.zhihu.com/search?type=content&q=score%20matching)

