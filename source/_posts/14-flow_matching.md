---
title: 笔记｜生成模型（十三）：Flow Matching理论与实现
date: 2025-09-12 11:16:52
cover: false
mathjax: true
categories:
 - Notes
tags:
 - Deep learning
 - Generative models theory
series: Diffusion Models theory
---

> 论文链接：*[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)*

在 Stable Diffusion 3 中，模型是通过 Flow Matching 的方法训练的。从这个方法的名字来看，就知道它和 Flow-based Model 有比较强的关联，因此在正式开始介绍这个方法之前先交代一些 Flow-based Model 相关的背景知识。

# Flow-based Models

## Normalizing Flow

Normalizing Flow 是一种基于**变换**对概率分布进行建模的模型，其通过一系列**离散且可逆的变换**实现任意分布与先验分布（例如标准高斯分布）之间的相互转换。在 Normalizing Flow 训练完成后，就可以直接从高斯分布中进行采样，并通过逆变换得到原始分布中的样本，实现生成的过程。

从这个角度看，Normalizing Flow 和 Diffusion Model 是有一些相通的，其做法的对比如下表所示。从表中可以看到，两者大致的过程是非常类似的，尽管依然有些地方不一样，但这两者应该可以通过一定的方法得到一个比较统一的表示。

| 模型             | 前向过程                                                     | 反向过程                                                     |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Normalizing Flow | 通过显式的可学习变换将样本分布变换为标准高斯分布             | 从标准高斯分布采样，并通过上述变换的逆变换得到生成的样本     |
| Diffusion Model  | 通过不可学习的 schedule 对样本进行加噪，多次加噪变换为标准高斯分布 | 从标准高斯分布采样，通过模型隐式地学习反向过程的噪声，去噪得到生成样本 |

## Continuous Normalizing Flow

Continuous Normalizing Flow（CNF），也就是连续标准化流，可以看作 Normalizing Flow 的一般形式。CNF 将原本 Normalizing Flow 中离散的变换替换为连续的变换，并用常微分方程（ODE）来表示，可以写成以下的形式：
$$
\frac{\mathrm{d}\mathbf{z}_t}{\mathrm{d}t}=v(\mathbf{z}_t,t)
$$
其中 $t\in[0,1]$，$\mathbf{z}_t$ 可以看作时间 $t$ 下的数据点，$v(\mathbf{z}_t,t)$ 是一个向量场，定义了数据点在每个时间下的变化大小与方向，这个向量场通常由神经网络来学习。当这个向量场完成了学习后，就可以用迭代法来求解：
$$
\mathbf{z}_{t+\Delta t}=\mathbf{z}_t+\Delta t\cdot v(\mathbf{z}_t,t)
$$
也就是说，一旦我们得知从标准高斯分布到目标分布的变换向量场，就可以从标准高斯分布采样，然后通过上述迭代过程得到目标分布中的一个近似解，完成生成的过程。这和离散的 Normalizing Flow 是一致的。

在 Normalizing Flow 中存在 Change of Variable Theory，这个定理是用来保证概率分布在进行变化时，概率密度在全体分布上的积分始终为 1 的一个式子，其形式为：
$$
p(\mathbf{x})=\pi(\mathbf{z})\left|\mathrm{det}\ \frac{\mathrm{d}\mathbf{z}}{\mathrm{d}\mathbf{x}}\right|=\pi(\mathbf{f}^{-1}(\mathbf{x}))\left|\mathrm{det}\ \mathbf{J}(\mathbf{f}^{-1}(\mathbf{x}))\right|
$$
在 Flow Matching 的论文中，也给出了形式类似的公式，称为 push-forward equation，定义为：
$$
p_t=[\phi_t]_*p_0
$$
其中的 push-forward 运算符，也就是星号，定义为：
$$
[\phi_t]_*p_0(x)=p_0(\phi_t^{-1}(x))\mathrm{det}\left[\frac{\partial\phi_t^{-1}}{\partial x}(x)\right]
$$
可以看出形式也是类似的。

# Flow Matching
流匹配的核心思想是在已知源分布 $ p $ 和目标分布 $ q $ 之间设计一条概率路径 $ p_t $，并学习一个速度场 $ u_t $，使该速度场能够生成该概率路径。
具体的实现步骤包括：

设计概率路径 $ p_t $：从源分布 $ p_0 = p $ 到目标分布 $ p_1 = q $ 的插值。在实际操作中，概率路径 $ p_t $ 通常通过条件概率路径 $ p_{t|1}(x|x_1) $ 来建设，后者又依赖于来自训练集的样本 $ x_1 $。

学习速度场 $ u_t $：速度场模型通常是一个参数化的神经网络，目标是尽量匹配真实的速度场 $ u_t $，该速度场生成与设计的概率路径 $ p_t $ 相符合的样本。

优化损失函数：流匹配最小化的损失函数 $ L_{FM} $ 是通过对比实际速度 $ u_t(X_t) $ 和预测的速度 $ u_\theta(X_t) $ 之间的差异来训练模型的。损失函数定义为 $ L_{FM}(\theta) = E_{X_t \sim p_t} D(u_t(X_t), u_\theta(X_t)) $，其中 $ D $ 是一个度量两者之间差异的方法，例如均方误差。


通过以上步骤，流匹配能够在不依赖于常微分方程的仿真情况下，较为高效地训练出模型，并且具备相对灵活的扩展性。
![Flow Matching](/chengYi-xun/img/flow_matching.png)
![Flow Matching And Diffusion](/chengYi-xun/img/flow_and_diffusion.png)



> 参考资料：
>
> 1. [深入解析Flow Matching技术](https://zhuanlan.zhihu.com/p/685921518)
> 2. [【AI知识分享】你一定能听懂的扩散模型Flow Matching基本原理深度解析](https://www.bilibili.com/video/BV1Wv3xeNEds/)
> 3. [flow_matching](https://littlenyima.github.io/posts/51-flow-matching-for-diffusion-models/)
> 4. [Normalizing Flow](https://littlenyima.github.io/posts/12-basic-concepts-of-normalizing-flow/)