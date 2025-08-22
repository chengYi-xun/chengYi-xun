---
title: 笔记｜生成模型（五）：DDPM理论
date: 2025-08-10 23:08:30
cover: false
mathjax: true
categories:
 - Notes
tags:
 - Deep learning
 - Generative models theory
series: Diffusion Models theory
---


> 论文链接：*[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)*

之前我们讨论过，生成模型的目的是：给定从真实分布 $P(x)$ 中采样的观测数据 $x$，训练得到一个由参数 $\theta$ 控制、能够逼近真实分布的模型 $p_\theta(x)$，这个任务太难了，所以使用变分推断去逼近，即为从一个标准高斯分布出发，经过某种映射或推导得到真实分布。而这个“某种映射”，并不要求一步到位，可以分多步执行。其中一个影响最大的实现方式，就是**扩散模型。**

扩散模型的名字来源于物理中的扩散过程，对于一张图像来说，类比扩散过程，向这张图像逐渐加入高斯噪声，当加入的次数足够多的时候，图像中像素的分布也会逐渐变成一个高斯分布。当然这个过程也可以反过来，如果我们设计一个神经网络，每次能够从图像中去掉一个高斯噪声，那么最后就能从一个高斯噪声得到一张图像。虽然一张有意义的图像不容易获得，但高斯噪声很容易采样，如果能实现这个逆过程，就能实现图像的生成。

![DDPM 示意图](/chengYi-xun/img/illustration-of-ddpm.jpg)

这个过程可以形象地用上图表示，扩散模型中有两个过程，分别是前向过程（从图像加噪得到噪音）和反向过程（从噪音去噪得到图像）。在上图中，向图像 $\mathbf{x}_0$ 逐渐添加噪声可以得到一系列的 $\mathbf{x}_1,\mathbf{x}_2,...,\mathbf{x}_T$，最后的 $\mathbf{x}_T$ 即接近完全的高斯噪声，这个过程显然是比较容易的。而从 $\mathbf{x}_T$ 逐渐去噪得到 $\mathbf{x}_0$​ 并不容易，扩散模型学习的就是这个去噪的过程。

# 前向过程


我们从比较简单的前向过程开始。首先回顾一下线性组合的方差公式：对于常数 $a, b$ 和**独立随机变量** $X, Y$，有

$$
\mathrm{Var}(aX+bY) = a^2\mathrm{Var}(X) + b^2\mathrm{Var}(Y)。
$$

如果 $X, Y$ 不独立，还需要加上 $2ab\,\mathrm{Cov}(X,Y)$ 的交叉协方差项。

另外，一个常用的概念是**马尔科夫链**：设 $\{X_t\}_{t=0}^\infty$ 是一个随机过程，如果对于任意时刻 $t$ 和状态 $i_0, i_1, \ldots, i_{t+1}$，都有

$$
P(X_{t+1} \mid X_0, X_1, \ldots, X_t) = P(X_{t+1} \mid X_t)，
$$

那么称 $\{X_t\}$ 为马尔科夫链，即未来状态只依赖于当前状态，与过去状态无关。

在前向加噪过程中，我们希望对图像逐步添加高斯噪声，并且在设定条件下保持总体方差不变。令初始图像为 $x_0$，标准高斯噪声为 $\epsilon_t \sim \mathcal{N}(0,\mathbf{I})$，且假设 $\epsilon_t$ 与 $x_{t-1}$ 独立。我们构造递推：

$$
x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon_t，
$$

此时协方差满足：

$$
\mathrm{Cov}(x_t) = (1-\beta_t)\,\mathrm{Cov}(x_{t-1}) + \beta_t\,\mathbf{I}。
$$



将上式写成条件概率分布的形式，可以得到：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\big(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t\,\mathbf{I} \big)，
$$

其中均值是 $\sqrt{1-\beta_t}\,x_{t-1}$，协方差是 $\beta_t$，每个维度的标准差为 $\sqrt{\beta_t}$。

在实际上进行加噪时，起始时使用的方差比较小，随着加噪步骤增加，方差会逐渐增大。例如在 DDPM 的原文中，使用的方差是从 $\beta_1=10^{-4}$ 随加噪时间步线性增大到 $\beta_T=0.02$。这样设置主要是为了方便模型进行学习，如果在最开始就加入很大的噪声，对图像信息的破坏会比较严重，不利于模型学习图像的信息。这个过程也可以从反向进行理解，即去噪时先去掉比较大的噪音得到图像的雏形，再去掉小噪音进行细节的微调。

上边等号的右边表示的就是当前的变量 $\mathbf{x}_t$ 满足一个 $\mathcal{N}(\sqrt{1-\beta_t}\mathbf{x}_{t-1},\beta_t\mathbf{I})$ 的概率分布。通过上边的公式我们可以看到，每一个时间步的 $\mathbf{x}_t$ 都只和 $\mathbf{x}_{t-1}$ 有关，因此这个扩散过程是一个马尔可夫过程。在前向过程中，每一步的 $\beta$ 都是固定的，真正的变量只有 $\mathbf{x}_{t-1}$，那么我们可以将公式中的 $\mathbf{x}_{t-1}$ 进一步展开：
$$
\begin{aligned}
\mathbf{x}_t&=\sqrt{1-\beta_t}\mathbf{x}_{t-1}+\sqrt{\beta_t}\epsilon_{t-1}\\
&=\sqrt{1-\beta_t}(\sqrt{1-\beta_{t-1}}\mathbf{x}_{t-2}+\sqrt{\beta_{t-1}}\epsilon_{t-2})+\sqrt{\beta_t}\epsilon_{t-1}\\
&=\sqrt{(1-\beta_t)(1-\beta_{t-1})}\mathbf{x}_{t-2}+\sqrt{(1-\beta_t)\beta_{t-1}}\epsilon_{t-2}+\sqrt{\beta_t}\epsilon_{t-1}
\end{aligned}
$$
在上边的公式里，实际上 $\epsilon_{t-2}$ 和 $\epsilon_{t-1}$ 是同分布的，都是 $\mathcal{N}(0,1)$，因此可以进行合并（**两个高斯分布的线性加权公式**）：


{% note info no-icon %}
**两个高斯分布线性加权的公式**

**一般情况：**

设 $X \sim \mathcal{N}(\mu_X, \sigma_X^2)$，$Y \sim \mathcal{N}(\mu_Y, \sigma_Y^2)$，且 $X \perp Y$（独立），则：

$$aX + bY \ \sim\  \mathcal{N}\big(a\mu_X + b\mu_Y,\ a^2\sigma_X^2 + b^2\sigma_Y^2 \big)$$

**零均值同分布的特例：**

若 $X, Y \stackrel{i.i.d.}{\sim} \mathcal{N}(0, 1)$，则：

$$aX + bY \ \sim\  \mathcal{N}\big(0,\ a^2 + b^2 \big)$$

或者写作：

$$aX + bY \ \stackrel{d}{=}\ \sqrt{a^2 + b^2}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$
{% endnote %}
因此有
$$
\begin{aligned}
\mathbf{x}_t&=\sqrt{(1-\beta_t)(1-\beta_{t-1})}\mathbf{x}_{t-2}+\sqrt{(\sqrt{(1-\beta_t)\beta_{t-1}})^2+(\sqrt{\beta_t})^2}\bar{\epsilon}_{t-2}\\
&=\sqrt{(1-\beta_t)(1-\beta_{t-1})}\mathbf{x}_{t-2}+\sqrt{1-(1-\beta_t)(1-\beta_{t-1})}\bar{\epsilon}_{t-2}
\end{aligned}
$$

令 $\alpha_t=1-\beta_t$，$\bar{\alpha}_t=\prod_{i=1}^t\alpha_i$，继续推导，可以得到：
$$
\begin{aligned}
\mathbf{x}_t&=\sqrt{\alpha_t\alpha_{t-1}}\mathbf{x}_{t-2}+\sqrt{1-\alpha_t\alpha_{t-1}}\bar{\epsilon}_{t-2}\\
&=\cdots\\
&=\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\epsilon
\end{aligned}
$$
通过上述的推导，我们发现给定 $\mathbf{x}_0$ 和加噪的时间步，可以直接用一步就得到 $\mathbf{x}_t$，而并不需要一步步地重复最开始的加权求和。和上述同理，这个关系也可以写成：
$$
q(\mathbf{x}_t|\mathbf{x}_0)=\mathcal{N}(\mathbf{x}_t;\sqrt{\bar{\alpha}_t}\mathbf{x}_0,(1-\bar{\alpha}_t)\mathbf{I})
$$
从这个式子里我们可以看出，加噪过程中的 $\mathbf{x}_t$ 可以看作原始图像 $\mathbf{x}_0$ 和高斯噪声 $\epsilon$ 的线性组合，且两个组合系数的平方和为 1。在实现加噪过程时，加噪的 scheduler 也是根据 $\bar{\alpha}_t$ 设计的，这样更加直接，且为了保证最后得到的足够接近噪声，可以将 $\bar\alpha_t$ 直接设置为一个接近 0 的数。

# 反向过程

正如文章开始所说的，反向过程就是从 $\mathbf{x}_T$ 逐渐去噪得到 $\mathbf{x}_0$ 的过程，也就是求 $q(\mathbf{x}_{t-1}|\mathbf{x}_t)$。根据贝叶斯公式：
$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t)=\frac{q(\mathbf{x}_t|\mathbf{x}_{t-1})q(\mathbf{x}_{t-1})}{q(\mathbf{x}_t)}
$$
在上边的公式里，在前文中我们已经给出了 $q(\mathbf{x}|\mathbf{x}_{t-1})$，但 $q(\mathbf{x}_{t-1})$ 和 $q(\mathbf{x}_t)$ 依然是未知的。虽然这两个分布目前未知，但是在上一节的最后，我们已经推导出了 $q(\mathbf{x}_t|\mathbf{x}_0)$ 这个分布，那么我们可以给上面的贝叶斯公式加上 $\mathbf{x}_0$ 作为条件，将等号右侧的两个未知分布转化为已知分布：
$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)=\frac{q(\mathbf{x}_t|\mathbf{x}_{t-1},\mathbf{x}_0)q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)}
$$
而且因为先验分布 $q(\mathbf{x}_t|\mathbf{x}_{t-1})$ 是马尔可夫过程，$\mathbf{x}_t$ 只与 $\mathbf{x}_{t-1}$ 有关，而与 $\mathbf{x}_0$ 无关，所以上边式子里的 $q(\mathbf{x}_t|\mathbf{x}_{t-1},\mathbf{x}_0)=q(\mathbf{x}_t|\mathbf{x}_{t-1})$。但推导到这里还有问题，我们把 $\mathbf{x}_0$ 加入到了条件概率分布的条件中，但 $\mathbf{x}_0$ 依然是未知的，因此我们需要继续推导出一个与 $\mathbf{x}_0$ 无关的式子。

上面的公式右侧的几个条件概率分布全都是高斯分布：
$$
\begin{aligned}
q(\mathbf{x}_t|\mathbf{x}_{t-1})&=\mathcal{N}(\mathbf{x}_t;\sqrt{\alpha_t}\mathbf{x}_{t-1},1-\alpha_t)\\
q(\mathbf{x}_{t-1}|\mathbf{x}_0)&=\mathcal{N}(\mathbf{x}_{t-1};\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0,1-\bar{\alpha}_{t-1})\\
q(\mathbf{x}_t|\mathbf{x}_0)&=\mathcal{N}(\mathbf{x}_t;\sqrt{\bar{\alpha}_t}\mathbf{x}_0,1-\bar\alpha_t)
\end{aligned}
$$
用概率密度函数把这个公式展开，如果不看前边的常数项，可以得到：
$$
\begin{aligned}
q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)&\propto\exp\left(-\frac{1}{2}\left[\frac{(\mathbf{x}_t-\sqrt{\alpha_t}\mathbf{x}_{t-1})^2}{\beta_t}+\frac{(\mathbf{x}_{t-1}-\sqrt{\bar\alpha_{t-1}}\mathbf{x}_0)^2}{1-\bar\alpha_{t-1}}+\frac{(\mathbf{x}_t-\sqrt{\bar{\alpha}_t}\mathbf{x}_0)}{1-\bar\alpha_t}\right]\right)\\
\end{aligned}
$$
因为我们在这一步去噪的时候想求得的是 $\mathbf{x}_{t-1}$ 的分布，所以我们把上式展开并整理成一个关于 $\mathbf{x}_{t-1}$ 的多项式：
$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)\propto\exp\left(-\frac{1}{2}\left[\left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}\right)\mathbf{x}_{t-1}^2-\left(\frac{2\sqrt{\alpha_t}}{\beta_t}\mathbf{x}_t+\frac{2\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}\mathbf{x}_0\right)\mathbf{x}_{t-1}+C(\mathbf{x}_t,\mathbf{x}_0)\right]\right)
$$
上边的式子里常数项不重要（因为可以直接变成常数从指数部分挪走），所以可以暂时不管。对比高斯分布（可以证明反向过程的分布也是高斯分布）的指数部分 $\exp\left(-\frac{1}{2}\left(\frac{1}{\sigma^2}x^2-\frac{2\mu}{\sigma^2}x+\frac{\mu^2}{\sigma^2}\right)\right)$：
$$
\begin{cases}
\begin{aligned}
\frac{1}{\sigma^2}&=\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}\\
\frac{2\mu}{\sigma^2}&=\frac{2\sqrt{\alpha_t}}{\beta_t}\mathbf{x}_t+\frac{2\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}\mathbf{x}_0
\end{aligned}
\end{cases}
$$
可以发现 $\sigma$ 的表达式里都是我们 scheduler 里的定值，而求解出均值 $\mu$：
$$
\mu=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t+\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\mathbf{x}_0
$$
代入上一章最后的 $\mathbf{x}_t=\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\epsilon$，得到：
$$
\mu=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\tilde{\epsilon}\right)
$$
注意在反向过程中我们并不知道在前向过程中加入的噪声 $\epsilon$ 是 $\mathcal{N}(0,1)$ 中的具体哪一个噪声，而噪声也没有办法继续转换成其他的形式。因此我们使用神经网络在反向过程中估计的目标就是 $\tilde{\epsilon}$。在这个网络中，输入除了 $\mathbf{x}_t$ 之外还需要 $t$，可以简单理解为：加噪过程中 $\mathbf{x}_t$ 的噪声含量是由 $t$ 决定的，因此在预测噪声时也需要知道时间步 $t$​ 作为参考，以降低预测噪声的难度。

注：关于反向过程为什么要这样做，Lilian Weng 基于变分推断给出了[一个复杂的证明](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)，因为过于难以理解，这里暂且把它跳过。

# 具体的训练过程

我们已经知道了去噪网络的参数和预测目标，下一个问题就是如何去训练这个去噪网络。原始论文中给出了如下的训练过程：

![DDPM 训练](/chengYi-xun/img/ddpm-training.jpg)

该算法的核心步骤如下：

1. **数据采样**：从训练数据分布 $q(\mathbf{x}_0)$ 中随机采样一个样本 $\mathbf{x}_0$
2. **时间步采样**：从均匀分布中随机采样时间步 $t$
3. **噪声采样**：从标准高斯分布 $\mathcal{N}(0, \mathbf{I})$ 中采样噪声 $\epsilon$
4. **前向扩散**：利用重参数化技巧，通过公式 $\mathbf{x}_t=\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\epsilon$ 直接计算第 $t$ 步的噪声图像
5. **噪声预测**：将 $\mathbf{x}_t$ 和时间步 $t$ 输入去噪网络 $\epsilon_\theta(\mathbf{x}_t, t)$，预测添加的噪声
6. **损失计算**：计算预测噪声与真实噪声之间的 L2 损失：$\mathcal{L} = \|\epsilon - \epsilon_\theta(\mathbf{x}_t, t)\|^2$

**关键理解**：这里需要澄清一个重要概念——为什么要预测"特定的"噪声 $\epsilon$ 而不是任意的高斯噪声？

原因在于：虽然 $\epsilon$ 确实是从标准高斯分布中采样的，但在给定 $\mathbf{x}_0$ 和 $\mathbf{x}_t$ 的情况下，将 $\mathbf{x}_0$ 变换到 $\mathbf{x}_t$ 的噪声 $\epsilon$ 是唯一确定的。网络学习的是这种从噪声图像 $\mathbf{x}_t$ 到其对应的"噪声成分" $\epsilon$ 的映射关系。在反向去噪过程中，准确预测这个特定的噪声至关重要，因为：

- 它决定了反向过程中每一步的去噪方向
- 任何预测误差都会在多步去噪过程中累积，导致生成质量下降
- 这种噪声预测实际上隐含地学习了数据分布的梯度信息（score function）

# 具体的采样过程

论文中同样也给出了采样过程：

![DDPM 采样](/chengYi-xun/img/ddpm-sampling.jpg)


采样算法通过迭代反向去噪过程生成新样本，具体步骤如下：

**算法输入**：训练好的噪声预测网络 $\epsilon_\theta(\mathbf{x}_t, t)$

**算法步骤**：

1. **初始化**：从标准高斯分布采样初始噪声 $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$

2. **迭代去噪**：对于 $t = T, T-1, ..., 1$，执行以下步骤：
   
   a) **噪声预测**：使用神经网络预测当前步的噪声
      $$\hat{\epsilon} = \epsilon_\theta(\mathbf{x}_t, t)$$
   
   b) **计算去噪均值**：根据后验分布公式计算
      $$\mu_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\hat{\epsilon}\right)$$
   
   c) **采样下一步**：
      - 当 $t > 1$ 时，添加随机噪声：
        $$\mathbf{x}_{t-1} = \mu_\theta(\mathbf{x}_t, t) + \sigma_t \mathbf{z}, \quad \text{其中} \quad \mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$$
      - 当 $t = 1$ 时，直接输出均值：
        $$\mathbf{x}_0 = \mu_\theta(\mathbf{x}_1, 1)$$

其中，方差 $\sigma_t^2$ 可以设置为：
- $\sigma_t^2 = \beta_t$（DDPM原论文选择）
- $\sigma_t^2 = \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$（理论最优后验方差）

**数学原理**：该算法基于重参数化技巧，将从 $\mathcal{N}(\mu, \sigma^2)$ 的采样分解为确定性的均值计算和随机的噪声添加两部分。最后一步不添加噪声是为了获得确定性的输出，避免在生成的最终图像中引入不必要的随机性。

# DDPM 的代码实现

现有的主流方法使用 UNet 来实现去噪网络，如下图所示。

![去噪网络的结构](/chengYi-xun/img/denoising-unet.jpg)

在此我们不关心网络的架构，感兴趣的同学可以自己去阅读源码。这个网络接收一个噪声图 $\mathbf{x}_t$ 和一个时间步 $t$ 作为参数，并输出一个噪声的预测结果 $\epsilon_\theta(\mathbf{x}_t,t)$。



## DDPM 核心算法

首先我们需要先定义 $\beta$、$\alpha$，以及 $\bar\alpha$ 等最基本的常量，这里我们保持 DDPM 原论文的配置，也就是 $\beta$ 初始为 $1\times10^{-4}$，最终为 $0.02$，且共有 $1000$ 个时间步：

```python
import torch

class DDPM:
    def __init__(
        self,
        num_train_timesteps:int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)
```

然后是比较简单的前向过程，只需要实现加噪即可，按照 $\mathbf{x}_t=\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\epsilon$ 这个公式实现即可。注意需要将系数的维度数量都与输入样本对齐：

```python
class DDPM:
    ...

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device ,dtype=original_samples.dtype)
        noise = noise.to(original_samples.device)
        timesteps = timesteps.to(original_samples.device)

        # \sqrt{\bar\alpha_t}
        sqrt_alpha_prod = alphas_cumprod[timesteps].flatten() ** 0.5
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # \sqrt{1 - \bar\alpha_t}
        sqrt_one_minus_alpha_prod = (1.0 - alphas_cumprod[timesteps]).flatten() ** 0.5
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
```

反向过程相对来说比较复杂，不过因为我们已经完成了公式的推导，只需要按照公式实现即可。我们也再把公式贴到这里，对着公式实现具体的代码：
$$
\begin{aligned}
\sigma&=\left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}\right)^{-1/2}\\
\mu&=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\tilde{\epsilon}_t\right)
\end{aligned}
$$

```python
class DDPM:
    ...

    @torch.no_grad()
    def sample(
        self,
        unet: UNet2DModel,
        batch_size: int,
        in_channels: int,
        sample_size: int,
    ):
        betas = self.betas.to(unet.device)
        alphas = self.alphas.to(unet.device)
        alphas_cumprod = self.alphas_cumprod.to(unet.device)
        timesteps = self.timesteps.to(unet.device)
        images = torch.randn((batch_size, in_channels, sample_size, sample_size), device=unet.device)
        for timestep in tqdm(timesteps, desc='Sampling'):
            pred_noise: torch.Tensor = unet(images, timestep).sample

            # mean of q(x_{t-1}|x_t)
            alpha_t = alphas[timestep]
            alpha_cumprod_t = alphas_cumprod[timestep]
            sqrt_alpha_t = alpha_t ** 0.5
            one_minus_alpha_t = 1.0 - alpha_t
            sqrt_one_minus_alpha_cumprod_t = (1 - alpha_cumprod_t) ** 0.5
            mean = (images - one_minus_alpha_t / sqrt_one_minus_alpha_cumprod_t * pred_noise) / sqrt_alpha_t

            # variance of q(x_{t-1}|x_t)
            if timestep > 0:
                beta_t = betas[timestep]
                one_minus_alpha_cumprod_t_minus_one = 1.0 - alphas_cumprod[timestep - 1]
                one_divided_by_sigma_square = alpha_t / beta_t + 1.0 / one_minus_alpha_cumprod_t_minus_one
                variance = (1.0 / one_divided_by_sigma_square) ** 0.5
            else:
                variance = torch.zeros_like(timestep)

            epsilon = torch.randn_like(images)
            images = mean + variance * epsilon
        images = (images / 2.0 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        return images
```


# 总结

本文总结了 DDPM 的理论和实现方式，在代码部分我们是完全根据推导出的公式实现的采样过程。实际上在很多代码库中，采样过程并没有严格按照论文中的公式实现，而是先从 $\mathbf{x}_t$、$t$ 和预测的噪声反向计算出 $\mathbf{x}_0$，再基于 $\mu=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t+\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\mathbf{x}_0$ 计算均值，这样的好处在于可以对 $\mathbf{x}_0$ 进一步规范化，控制输出的范围。

可以看出 DDPM 虽然理论比较复杂，但实现起来还是比较简单直接的。

最后提一下Open AI的Improved DDPM。虽然 DDPM 在生成任务上取得了不错的效果，但如果使用一些 metric 对 DDPM 进行评价，就会发现其虽然能在 FID 和 Inception Score 上获得不错的效果，但在负对数似然（Negative Log-likelihood，NLL）这个指标上表现不够好。根据 VQ-VAE2 文章中的观点，NLL 上的表现体现的是模型捕捉数据整体分布的能力。而且有工作表明即使在 NLL 指标上仅有微小的提升，就会在生成效果和特征表征能力上有很大的提升。

Improved DDPM 主要是针对 DDPM 的训练过程进行改进，主要从两个方面进行改进：

1. 不使用 DDPM 原有的固定方差，而是使用可学习的方差；
2. 改进了加噪过程，使用余弦形式的 Scheduler，而不是线性 Scheduler。

作者发现线性的 $\beta_t$ 对于高分辨率图像效果不错，但对于低分辨率的图像表现不佳。在之前的文章中我们提到过，在 DDPM 加噪的时候 $\beta_t$ 是从一个比较小的数值逐渐增加到比较大的数值的，因为如果最开始的时候加入很大的噪声，会严重破坏图像信息，不利于图像的学习。在这里应该也是相同的道理，因为低分辨率图像包含的信息本身就不多，虽然一开始使用了比较小的 $\beta_t$，但线性的 schedule 对于这些低分辨率图像来说还是加噪比较快。

作者把方差用一种 cosine 的形式定义，不过并不是直接定义 $\beta_t$，而是定义 $\bar{\alpha}_t$：
$$
\bar{\alpha}_t=\frac{f(t)}{f(0)},\quad f(t)=\cos\left(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\right)^2
$$
这个 schedule 和线性 schedule 的比较如下图所示：

![iddpm](/chengYi-xun/img/iddpm.png)

这个 schedule 在 $t=0$ 和 $t=T$ 附近都变化比较小，而在中间有一个接近于线性的下降过程，同时可以发现 cosine schedule 比 linear schedule 对信息的破坏更慢。这也印证了我们在前边提到的理论：在扩散开始的时候更加缓慢地加噪，可以得到更好的训练效果。除此之外设计这个 schedule 的时候作者也有一些比较细节的考虑，比如选取一个比较小的偏移量 $s=8\times10^{-3}$，防止 $\beta_t$ 在 $t=0$ 附近过小，并且将 $\beta_t$ 裁剪到 $0.999$ 来防止 $t=T$ 附近出现奇异点。


> 参考资料：
>
> 1. [DDPM 理论与实现](https://littlenyima.github.io/posts/13-denoising-diffusion-probabilistic-models/)
> 2. [简单基础入门理解Denoising Diffusion Probabilistic Model，DDPM扩散模型](https://blog.csdn.net/qq_40714949/article/details/126643111)
> 3. [扩散模型之DDPM](https://zhuanlan.zhihu.com/p/563661713)
> 4. [Denoising Diffusion-based Generative Modeling: Foundations and Applications](https://drive.google.com/file/d/1DYHDbt1tSl9oqm3O333biRYzSCOtdtmn/view)
> 5. [Train a diffusion model](https://huggingface.co/docs/diffusers/tutorials/basic_training)
