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

证据下界是一个英译词，英文名是 Evidence Lower Bound，缩写是 ELBO，是变分推断中引入的一个概念。
