---
title: 笔记｜扩散模型（一）：一些概率论的基础概念和理论
date: 2025-07-31 01:37:31
cover: false
mathjax: true
categories:
 - Notes
tags:
 - Deep learning
 - Generative models theory
series: Diffusion Models theory
---

## 概率 vs 似然

概率：已知某种分布和其参数 $\theta$ 的情况下，某件事情发生的概率。

似然：已知一种分布形态（可能是高斯分布，泊松分布等）和一组观测数据的情况下，不同的参数 $\theta$ 产生这组观测数据的可能性。

简单来说，概率是已知 $P(x)$ 的具体形式，求 $x = x_0$ 时候的值。
而似然是已知 $P(x)$ 的形态，比如二次函数，一次函数之类的，但是其具体参数未知。假设这里是二次函数 $P(x) = ax^2+bx+c$，同时我们又已知一组观测数据 $x_1, x_2, x_3, \ldots, x_n$，则任意一组参数 $a, b, c$ 产生上述观测数据的可能性即为似然。

最大似然（MLE）：假设存在独立同分布的观测数据 $x_1, x_2, x_3, \ldots, x_n$，某个概率分布函数是 $P_\theta(x)$，$\theta$ 为该函数的参数，则这组观测数据的联合概率有如下形式：

$$P_\theta(x_1, \ldots, x_n) = \prod_{i = 1}^N P_\theta(x_i)$$

最大似然就是求出上述公式最大值时候的函数参数 $\theta$：

$$\hat{\theta} = \arg\max_\theta \left( \prod_{i = 1}^N P_\theta(x_i) \right)$$

由于最大似然有一个连乘符号，会给计算引入一些困难，因此一般会把最大似然转化成对数似然，避免了数值不稳定（如下溢出）的问题。由于对数函数是单调递增的，因此不会改变极值点，因此有对数似然公式如下：

$$\mathcal{L}(\theta) = \log \left( \prod_{i=1}^N P_\theta(x_i) \right) = \sum_{i=1}^N \log P_\theta(x_i)$$

然后，MLE 的参数估计就变成了最大化这个对数似然：

$$\hat{\theta} = \arg\max_\theta \mathcal{L}(\theta) = \arg\max_\theta \sum_{i=1}^N \log P_\theta(x_i)$$



## 变分推断 vs 证据下界

证据下界是一个英译词，英文名是 Evidence Lower Bound（ELBO），证据下界是变分推断中引入的一个概念。

### 变分推断：

问题引出：如果想要求 $x$ 已知的情况下，潜变量 $z$ 的概率分布 $P(z|x)$，我们要如何求呢。由于 $z$ 是潜变量的原因，直接求是很难求的。（为了更加形象，你可以将 $x$ 理解成已知的数据分布，即神经网络的输入，潜变量可以理解成隐藏变量，比如网络中间层这种难以直接定义形态和参数的变量，直接写出网络学习过程中某层的概率分布函数几乎是不可能的）。所以我们有另一个办法，那就是近似。

我们可以定义一个分布形态 $Q$ (最简单的可以看做高斯分布)。然后在这个分布形态去寻找一组参数构造出一个概率分布 $q^* \in Q$ ，使得我们选定的这种分布和潜分布具有相近最大似然。

$${q^ * }(z) = \mathop {\arg \min }\limits_{q(z) \in Q} \{ L[q(z),P(z|x)]\} $$

其中 $L$ 表示两种概率分布的距离度量。以上就是变分推断的概念，通俗的来说，即通过引入一个参数化的近似分布来代替真实的分布。通过这种方式，变分推断可以在不直接计算真实后验分布的情况下，估计出一个较好的近似后验分布。

这里插个题外话，变分推断这个名字怎么这么奇怪，他是怎么来的呢。其实变分推断中的变分，来源于数学中的变分法（Variational Methods）。变分法最初来自于数学中的变分原理，它通常用于最优化问题，特别是求解某些特定函数的极值。变分法通过引入一个试探函数（通常是一个可调的函数家族）来近似某个目标函数的最优解。在机器学习和统计学中，变分法被用来近似一些复杂的概率分布或推断问题的解。

而推断是统计学中的一个基本任务，它指的是从观测数据出发，推导出对模型参数或潜在变量的合理估计（可以理解成参数估计）。在贝叶斯推断中，我们通常想要通过计算后验分布来推断潜在变量或模型参数。而在变分推断中，由于直接计算后验分布通常是非常困难的，变分方法通过寻找一个易于计算的近似分布来实现推断。

因此，“变分推断”这个名字的由来就是：

1. “变分”表示使用变分法（变分法是通过优化一个下界来寻找近似解的技术）；

2. “推断”表示这个过程是用于推断概率模型中的潜在变量或参数的。

### 证据下界 ELBO

回到正题，如前所述，变分推断的核心思想在于：通过引入一个参数化的已知分布 $q(z)$，来近似原本难以直接求解的后验分布 $P(z|x)$。此时，一个关键问题是：如何衡量这两个分布之间的差异性，即如何定量评估这种近似的优劣。

在概率论与信息论领域，已经提出了多种度量概率分布之间差异的指标，包括 JS 散度（Jensen–Shannon divergence）、全变差距离（Total Variation Distance）、Bhattacharyya 距离、华盛顿距离（Wasserstein Distanc） 以及 KL 散度（Kullback–Leibler divergence）等。其中，KL 散度是最常用的一种衡量两个分布近似程度的指标。
KL 散度的定义如下：

$$
\mathrm{KL}(q(z) \,\|\, P(z|x)) = \int q(z) \log \left( \frac{q(z)}{P(z|x)} \right) \, dz
$$

KL 散度的一个重要性质是非负性，即：

$$
\mathrm{KL}(q(z) \,\|\, P(z|x)) \geq 0
$$

当且仅当 $q(z) = P(z|x)$ 几乎处处成立时取等号。

我们可以对非负性做一个简单的证明，由初中知识可得 (=`ω´=)：

$$
\log x \leq x - 1 \quad \text{对所有 } x > 0，\text{且当且仅当 } x = 1 \text{ 时等号成立}
$$

对该不等式两边取负，得到：$-\log x \geq 1 - x$ 令：$x = \frac{P(z|x)}{q(z)}$ 代入不等式得：

$$
- \log \left( \frac{P(z|x)}{q(z)} \right) \geq 1 - \frac{P(z|x)}{q(z)}
\Rightarrow \log \left( \frac{q(z)}{P(z|x)} \right) \geq \frac{q(z) - P(z|x)}{q(z)}
$$

对两边乘以 $q(z)$ 有：

$$
\mathrm{KL}(q(z) \,\|\, P(z|x)) = \int q(z) \log \left( \frac{q(z)}{P(z|x)} \right) dz \geq \int (q(z) - P(z|x)) \, dz
$$

但由于 $q(z)$ 与 $P(z|x)$ 都是概率密度函数，其积分为 1，因此：

$$
\int (q(z) - P(z|x)) dz = \int q(z) dz - \int P(z|x) dz = 1 - 1 = 0
$$

从而得出：

$$
\mathrm{KL}(q(z) \,\|\, P(z|x)) \geq 0
$$

当且仅当 $q(z) = P(z|x)$ 几乎处处成立时取等号。

同时KL散度也可以展开成信息熵的形式：

$$
\mathrm{KL}(q(z) \| P(z|x)) = \sum_z q(z) \log \left( \frac{q(z)}{P(z|x)} \right)
= \underbrace{ - \sum_z q(z) \log P(z|x) }_{\text{交叉熵}}+\underbrace{\sum_z q(z) \log q(z) }_{\text{负熵}}
$$

即，KL 散度等于 $H(q,P)-H(q)$，我相信做分类模型的同学应该非常熟悉，多分类任务损失函数中经常看到他的身影。

兜了一圈，终于要讲什么是证据下界了。由上述可知，我们定义出了变分推断的公式，其中用KL散度替换掉距离度量的 $L$ 有如下形式：

$${q^*}(z) = \mathop {\arg \min }\limits_{q(z) \in Q} \{ KL(q(z)||P(z|x))\}$$

从之前的推断可知，当${q^*}(z)=P(z|x)$的时候，即为我们所要求的最优解。但$P(z|x)$是未知而且基本不可求的，因此我们无法计算该公式，所以，引入了证据下界。

$$
\begin{align*}
  \mathrm{KL}(q(z) \| P(z|x)) 
  &= - \int q(z) \log \left( \frac{P(z|x)}{q(z)} \right) \, dz \\
  &= \int q(z) \log q(z) \, dz - \int q(z) \log P(z|x) \, dz
\end{align*}
$$

观察上式可发现，我们可以将其写成概率函数的期望形式：

$$
\begin{align*}
\mathrm{KL}(q(z) \parallel P(z \mid x)) 
&= \int q(z) \log q(z) \, dz - \int q(z) \log P(z \mid x) \, dz \\
&= E_q[\log q(z)] - E_q[\log P(z \mid x)]
\end{align*}
$$

结合贝叶斯公式，可以写成：

$$
\begin{align*}
\mathrm{KL}(q(z)\parallel P(z\mid x)) 
&= \int q(z)\log q(z)\,dz - \int q(z)\log P(z\mid x)\,dz \\
&= E_q[\log q(z)] - E_q[\log P(z\mid x)] \\
&= E_q\left[\log q(z)\right] - E_q\left[\log \left(\frac{P(x,z)}{P(x)}\right)\right] \\
&= E_q[\log q(z)] - E_q[\log P(x,z)] + E_q[\log P(x)]
\end{align*}
$$

又由于对 $q$ 的期望与P无关，因此可以写成

$$\mathrm{KL}(q(z)\parallel P(z\mid x)) = E_q[\log q(z)] - E_q[\log P(x,z)] + \log P(x)$$

这里我们定义

$$ELBO =E_q[\log P(x,z)] - E_q[\log q(z)]$$

因此就有

$$\log P(x)=ELBO+\mathrm{KL}(q(z)\parallel P(z\mid x))$$

又因为KL散度的非负性，因此我们得出了生成领域非常著名公式：

$$\log(p(x)) \ge ELBO_q$$

这里$P(x)$就是贝叶斯统计所说的"证据"，也是模型的边际似然。表示我们输入观测数据$x$模型输出概率是怎么样的，也就是观测数据的"证据"。证据的下界，即为ELBO。

贝叶斯推断的核心：

$$
\text{posterior} = \frac{\text{likelihood} \times \text{prior}}{\text{evidence}}
$$

用数学符号表示为：

$$
P(z \mid x) = \frac{P(x,z)}{P(x)}
$$

**所以，如果想要求某个分布的最大似然，可以转化成求最小化负对数，再转化为求最大化ELBO，即为求该似然的最大下界**。

$${q^*}(z) = \mathop {\arg \min }\limits_{q(z) \in Q} \{ KL(q(z)||P(z|x))\} = \arg \max (ELBO_q)$$


如果你读到这里，恭喜你！你被我恭喜到了！