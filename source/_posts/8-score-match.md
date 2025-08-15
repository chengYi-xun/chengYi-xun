---
title: 笔记｜生成模型（六）：Score-Base理论
date: 2025-08-15 23:08:30
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

生成模型通常从某个已有的概率分布中进行采样以生成样本。Score-based 模型的关键在于对概率分布的对数梯度，即 **score function** 的建模。为了学习这个对象，我们需要使用一种称为 **score matching** 的技术，这是 Score-based 模型名称的由来。

## Score Function 和 Score-based Models

考虑一个数据集 $\{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N\}$ 的概率分布 $p(\mathbf{x})$，我们旨在建模此分布。我们可以使用以下形式表示这个概率分布，该形式来源于能量模型（Energy-based model）：

$$
p_\theta(\mathbf{x}) = \frac{\exp(-f_\theta(\mathbf{x}))}{Z_\theta}
$$
{% note info no-icon %}
能量模型是一种基于能量的概率分布建模方法，其核心思想是通过能量函数来定义概率分布，而不是直接参数化概率密度函数。
对于数据 $\mathbf{x} \in \mathbb{R}^D$，能量模型将其概率分布定义为：
$$p_\theta(\mathbf{x}) = \frac{e^{-E_\theta(\mathbf{x})}}{Z_\theta}$$
其中：
$E_\theta(\mathbf{x})$ 是能量函数（Energy Function），由参数 $\theta$ 参数化
$Z_\theta = \int e^{-E_\theta(\mathbf{x})} d\mathbf{x}$ 是配分函数（Partition Function），用于归一化

**能量函数的性质:**
 - 低能量对应高概率：$E_\theta(\mathbf{x})$ 越小，$p_\theta(\mathbf{x})$ 越大
 - 高能量对应低概率：$E_\theta(\mathbf{x})$ 越大，$p_\theta(\mathbf{x})$ 越小
 - 连续性：能量函数通常是连续可微的

其中，$f_\theta(\mathbf{x})$ 是具有可学习参数 $\theta$ 的函数，通常表示为神经网络。$Z_\theta$ 是归一化因子，定义为：
{% endnote %}


$$
Z_\theta = \int \exp(-f_\theta(\mathbf{x})) \mathrm{d} \mathbf{x}
$$

由于 $p_\theta(\mathbf{x})$ 是一个概率分布，因此必须满足归一化条件 $\int p_\theta(\mathbf{x}) \mathrm{d} \mathbf{x} = 1$。接下来，我们可以对 $\theta$ 进行训练以进行极大似然估计，目标为：

$$
\max_{\theta} \sum_{i=1}^N \log p_\theta(\mathbf{x}_i)
$$

然而，$Z_\theta$ 的确切值通常是未知的，尤其是对于任意分布。这一问题的一种解法是通过学习 score function 来规避 $Z_\theta$ 的问题。score function 定义为：

$$
\mathbf{s}_\theta(\mathbf{x}) = \nabla_\mathbf{x} \log p_\theta(\mathbf{x})
$$

因为 $Z_\theta$ 是常数，所以它的梯度为零，从而有：

$$
\mathbf{s}_\theta(\mathbf{x}) = \nabla_\mathbf{x} \log p_\theta(\mathbf{x}) = -\nabla_\mathbf{x} f_\theta(\mathbf{x})
$$

这表明 score function 实际上对应于神经网络的梯度。我们可以构建优化目标：

$$
\theta = \arg\min_{\theta} \mathbb{E}_{p(\mathbf{x})}\left[ \left\| \nabla_\mathbf{x} \log p(\mathbf{x}) - \mathbf{s}_\theta(\mathbf{x}) \right\|_2^2 \right]
$$

真实分布的 log 梯度 $\nabla_\mathbf{x} \log p(\mathbf{x})$ 是未知的，因此我们需要利用 **score matching** 方法。

## Score Matching

为了避免使用真实分布 $p(\mathbf{x})$，我们对优化目标进行分析。首先展开 L2 的平方：

$$
\begin{aligned}
\left\| \nabla_x \log p(x) - \mathbf{s}_\theta(x) \right\|_2^2 &= \left\| \nabla_x \log p(x) - \nabla_x \log p_\theta(x) \right\|_2^2 \\
&= \underbrace{(\nabla_x \log p(x))^2}_{\text{const}} - 2 \nabla_x \log p(x) \nabla_x \log p_\theta(x) + \left(\nabla_x \log p_\theta(x)\right)^2
\end{aligned}
$$

其中第一项是常量，可以忽略。最后一项可以通过数据集的样本估计。现在只需关注第二项：

$$
\mathbb{E}_{p(x)}[-\nabla_x \log p(x) \nabla_x \log p_\theta(x)] 
$$

使用分部积分法，这一项可以进一步推导为：

$$
\begin{aligned}
&\mathbb{E}_{p(x)}[-\nabla_x \log p(x) \nabla_x \log p_\theta(x)] \\
&= -\int_{-\infty}^{\infty} \nabla_x \log p(x) \nabla_x \log p_\theta(x) p(x) \, \mathrm{d} x \\
&= -\int_{-\infty}^{\infty} \frac{\nabla_x p(x)}{p(x)} \nabla_x \log p_\theta(x) p(x) \, \mathrm{d} x \\
&= -\int_{-\infty}^{\infty} \nabla_x p(x) \nabla_x \log p_\theta(x) \, \mathrm{d} x \\
&= -p(x)\nabla_x \log p_\theta(x) \bigg|_{-\infty}^\infty + \int_{-\infty}^{\infty} p(x) \nabla_x^2 \log p_\theta(x) \, \mathrm{d} x
\end{aligned}
$$

假设对于真实的数据分布，当 $|x| \rightarrow \infty$，有 $p(x) \rightarrow 0$，所以第一项为 0，因此得到：

$$
\mathbb{E}_{p(x)}[-\nabla_x \log p(x) \nabla_x \log p_\theta(x)] = \mathbb{E}_{p(x)}[\nabla_x^2 \log p_\theta(x)]
$$

最终的优化目标为：

$$
\begin{aligned}
\mathbb{E}_{p(x)}\left[ \left\| \nabla_x \log p(x) - \mathbf{s}_\theta(x) \right\|_2^2 \right] &= 2\mathbb{E}_{p(x)}\left[ \nabla_x^2 \log p_\theta(x) \right] + \mathbb{E}_{p(x)}\left[ \left( \nabla_x \log p_\theta(x) \right)^2 \right] + \text{const}
\end{aligned}
$$

对于多元情况，优化目标变为：

$$
\mathbb{E}_{p(\mathbf{x})} \left[ 2 \mathrm{tr}(\nabla_{\mathbf{x}}^2 \log p_\theta(\mathbf{x})) + \left\| \nabla_{\mathbf{x}} \log p_\theta(\mathbf{x}) \right\|_2^2 \right] + \text{const}
$$

这使得优化目标不再依赖于真实分布 $p(x)$，可以直接用于优化。

## 从分布采样：Langevin Dynamics

通过上述步骤，我们得到了 $\mathbf{s}_\theta(\mathbf{x}) \approx \nabla_\mathbf{x} \log p(\mathbf{x})$。要从这样的分布进行采样，可以使用 Langevin Dynamics（朗之万动力学）过程。其基本步骤为：

1. 从任意先验分布中采样初始状态 $\mathbf{x}_0 \sim \pi(\mathbf{x})$。
2. 迭代更新：

$$
\mathbf{x}_{i+1} \leftarrow \mathbf{x}_i + \epsilon \nabla_{\mathbf{x}} \log p(\mathbf{x}) + \sqrt{2\epsilon} \mathbf{z}_i, \quad i = 0, 1, \ldots, K
$$

其中 $ \mathbf{z}_i \sim \mathcal{N}(0, I)$，当 $\epsilon \rightarrow 0$ 且 $K \rightarrow \infty$ 时，$\mathbf{x}_K$ 会收敛到从 $p(\mathbf{x})$ 直接采样的结果。

## 存在的问题与改进方案

尽管 Score-based 模型解决了许多问题，但依然存在一些挑战，如低概率密度区域建模不准确的问题。为了解决这一问题，可以使用各向同性高斯噪声的扰动。这意味着使用不同方差的高斯噪声以平衡概率密度区域，从而提高模型的准确性。对于每个分布 $p_{\sigma_i}(\mathbf{x})$，我们有：

$$
p_{\sigma_i}(\mathbf{x}) = \int p(\mathbf{y}) \mathcal{N}(\mathbf{x}; \mathbf{y}, \sigma_i^2 I) \, \mathrm{d} \mathbf{y}
$$

通过对使用高斯噪声扰动的分布进行 score matching，可以定义加权目标：

$$
\sum_{i=1}^L \lambda(i) \mathbb{E}_{p_{\sigma_i}(\mathbf{x})} \left[ \left\| \nabla_{\mathbf{x}} \log p_{\sigma_i}(\mathbf{x}) - \mathbf{s}_\theta(\mathbf{x}, i) \right\|_2^2 \right]
$$

通常选择 $\lambda(i) = \sigma_i^2$，这样能够组合不同噪声水平的信息，增强模型的表现。

## 总结

如上所述，Score-based 模型遵循学习与采样的范式。学习过程通过 score matching 来间接学习概率分布，采样过程则采用 Langevin dynamics。在优化过程中，解决了低概率密度区域学习不足的问题，同时通过多种高斯噪声扰动和退火 Langevin 动力学的组合来提高采样效果。

