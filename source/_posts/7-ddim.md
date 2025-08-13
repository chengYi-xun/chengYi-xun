---
title: 笔记｜生成模型（六）：DDIM理论
date: 2025-08-12 23:08:30
cover: false
mathjax: true
categories:
 - Notes
tags:
 - Deep learning
 - Generative models theory
series: Diffusion Models theory
---


> 论文链接：*[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)*

在[DDPM理论](../6-ddpm)中我们进行了 DDPM 的理论推导以及给出了核心代码。但 DDPM 有一个非常明显的问题：采样过程很慢。因为 DDPM 的反向过程利用了马尔可夫假设，所以每次都必须在相邻的时间步之间进行去噪，而不能跳过中间步骤。原始论文使用了 1000 个时间步，所以我们在采样时也需要循环 1000 次去噪过程，这个过程是非常慢的。

为了加速 DDPM 的采样过程，DDIM 在不利用马尔可夫假设的情况下推导出了 diffusion 的反向过程，最终可以实现仅采样 20～100 步的情况下达到和 DDPM 采样 1000 步相近的生成效果，也就是提速 10～50 倍。这篇文章将对 DDIM 的理论进行讲解，并实现 DDIM 采样的代码。

# DDPM 的反向过程

首先我们回顾一下 DDPM 反向过程的推导，为了推导出 $q(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 这个条件概率分布，DDPM 利用贝叶斯公式将其变成了先验分布的组合，并且通过向条件中加入 $\mathbf{x}_0$ 将所有的分布转换为已知分布：
$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)=\frac{q(\mathbf{x}_t|\mathbf{x}_{t-1},\mathbf{x}_0)q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)}
$$
在上边这个等式的右侧，$q(\mathbf{x}_{t-1}|\mathbf{x}_0)$ 和 $q(\mathbf{x}_t|\mathbf{x}_0)$ 都是已知的，需要求解的只有 $q(\mathbf{x}_t|\mathbf{x}_{t-1},\mathbf{x}_0)$。在这里 DDPM 引入**马尔可夫假设**，认为 $\mathbf{x}_t$ 只与 $\mathbf{x}_{t-1}$ 有关，将其转化成了 $q(\mathbf{x}_t|\mathbf{x}_{t-1})$。最后经过推导，得出条件概率分布：
$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t)=\mathcal{N}(\mathbf{x}_{t-1};\mu_\theta(\mathbf{x}_t,t),\sigma_t^2\mathbf{I})
$$
我们可以看到之所以 DDPM 很慢，就是因为在推导 $q(\mathbf{x}_t|\mathbf{x}_{t-1},\mathbf{x}_0)$ 的时候引入了马尔可夫假设，使得去噪只能在相邻时间步之间进行。如果我们可以在不依赖马尔可夫假设的情况下推导出 $q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)$，就可以将上面式子里的 $t-1$ 替换为任意的中间时间步 $\tau$，从而实现采样加速。总结来说，DDIM 主要有两个出发点：

1. 保持前向过程的分布 $q(\mathbf{x}_t|\mathbf{x}_{t-1})=\mathcal{N}\left(\mathbf{x}_t;\sqrt{\bar{\alpha}_t}\mathbf{x}_0,(1-\bar{\alpha}_t)\mathbf{I}\right)$ 不变；
2. 构建一个不依赖于马尔可夫假设的 $q(\mathbf{x}_\tau|\mathbf{x}_t,\mathbf{x}_0)$ 分布。

## $q(\mathbf{x}_\tau|\mathbf{x}_t,\mathbf{x}_0)$ 的推导

开始推导之前简单说明一下，这个 $q(\mathbf{x}_\tau|\mathbf{x}_t,\mathbf{x}_0)$ 实际上就是上一章中提到的 $q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)$，只不过是因为我们的推导不再依赖马尔可夫假设，所以 $t-1$ 可以替换为任意的 $\tau\in(0,t)$。为了避免混淆，我们在这里使用一个通用的符号 $\tau\in(0,t)$ 表示中间的时间步。

另一点需要说明的是，在 DDIM 的论文中，$\alpha$ 表示的含义和 DDPM 论文中的 $\bar{\alpha}$ 相同。为了保证前后一致，我们在这里依然使用 DDPM 的符号约定，令 $\alpha_t=1-\beta_t$，$\bar{\alpha}_t=\prod_{i=1}^t\alpha_i$。

我们在 DDPM 里已经推导出了 $q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)$ 是一个高斯分布，均值和方差为：
$$
\begin{aligned}
\mu&=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t+\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\mathbf{x}_0\\
\sigma&=\left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}\right)^{-1/2}
\end{aligned}
$$
可以看到均值是 $\mathbf{x}_0$ 与 $\mathbf{x}_t$ 的线性组合，方差是时间步的函数。DDIM 基于这样的规律，使用待定系数法：
$$
q(\mathbf{x}_\tau|\mathbf{x}_t,\mathbf{x}_0)=\mathcal{N}(\mathbf{x}_\tau;\lambda\mathbf{x}_0+k\mathbf{x}_t,\sigma_t^2\mathbf{I})
$$
也就是 $\mathbf{x}_\tau=\lambda\mathbf{x}_0+k\mathbf{x}_t+\sigma_t\epsilon_\tau$。又因为前向过程满足 $\mathbf{x}_t=\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\epsilon_t$，代入可以得到：
$$
\begin{aligned}
\mathbf{x}_\tau&=\lambda\mathbf{x}_0+k\mathbf{x}_t+\sigma_t\epsilon_\tau\\
&=\lambda\mathbf{x}_0+k(\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\epsilon_t)+\sigma_t\epsilon_\tau\\
&=(\lambda+k\sqrt{\bar{\alpha}_t})\mathbf{x}_0+(k\sqrt{1-\bar{\alpha}_t}\epsilon_t+\sigma_t\epsilon_\tau)\\
&=(\lambda+k\sqrt{\bar{\alpha}_t})\mathbf{x}_0+\sqrt{k^2(1-\bar{\alpha}_t)+\sigma_t^2}\epsilon
\end{aligned}
$$
在上面的推导过程中，由于 $\epsilon_t$ 和 $\epsilon_\tau$ 都满足标准正态分布，因此两项可以合并。又因为根据前向过程，有 $\mathbf{x}_\tau=\sqrt{\bar{\alpha}_\tau}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_\tau}\epsilon_\tau$，将两个式子的系数对比，可以得到方程组：
$$
\begin{cases}
\begin{aligned}
\lambda+k\sqrt{\bar{\alpha}_t}&=\sqrt{\bar{\alpha}_\tau}\\
\sqrt{k^2(1-\bar{\alpha}_t)+\sigma_t^2}&=\sqrt{1-\bar{\alpha}_\tau}
\end{aligned}
\end{cases}
$$
解方程组得到 $\lambda$ 和 $k$：
$$
\begin{cases}
\begin{aligned}
\lambda&=\sqrt{\bar{\alpha}_\tau}-\sqrt{\frac{(1-\bar{\alpha}_\tau-\sigma_t^2)\bar{\alpha}_t}{1-\bar{\alpha}_t}}\\
k&=\sqrt{\frac{1-\bar{\alpha}_\tau-\sigma_t^2}{1-\bar{\alpha}_t}}
\end{aligned}
\end{cases}
$$
在上边的结果中，我们得到了 $q(\mathbf{x}_\tau|\mathbf{x}_t,\mathbf{x}_0)$ 均值中的两个参数，而方差 $\sigma_t^2$ 并没有唯一定值，因此这个结果对应于一组解，通过规定不同的方差，可以得到不同的采样过程。我们把 $\mathbf{x}_0$ 用 $\mathbf{x}_t$ 替换，可以得到均值的表达式：
$$
\begin{aligned}
\mu&=\lambda\mathbf{x}_0+k\mathbf{x}_t\\
&=\left(\sqrt{\bar{\alpha}_\tau}-\sqrt{\frac{(1-\bar{\alpha}_\tau-\sigma_t^2)\bar{\alpha}_t}{1-\bar{\alpha}_t}}\right)\mathbf{x}_0+\sqrt{\frac{1-\bar{\alpha}_\tau-\sigma_t^2}{1-\bar{\alpha}_t}}\mathbf{x}_t\\
&=\left(\sqrt{\bar{\alpha}_\tau}-\sqrt{\frac{(1-\bar{\alpha}_\tau-\sigma_t^2)\bar{\alpha}_t}{1-\bar{\alpha}_t}}\right)\left(\frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(\mathbf{x}_t,t)}{\sqrt{\bar{\alpha}_t}}\right)+\sqrt{\frac{1-\bar{\alpha}_\tau-\sigma_t^2}{1-\bar{\alpha}_t}}\mathbf{x}_t\\
&=\sqrt{\bar{\alpha}_\tau}\frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(\mathbf{x}_t,t)}{\sqrt{\bar{\alpha}_t}}+\sqrt{1-\bar{\alpha}_\tau-\sigma_t^2}\epsilon_\theta(\mathbf{x}_t,t)
\end{aligned}
$$
因此我们可以得到最终的 $\mathbf{x}_\tau$ 的表达式：
$$
\begin{aligned}
\mathbf{x}_\tau&=\mu+\sigma_t\epsilon\\
&=\sqrt{\bar{\alpha}_\tau}\underbrace{\frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(\mathbf{x}_t,t)}{\sqrt{\bar{\alpha}_t}}}_{预测的\mathbf{x}_0}+\underbrace{\sqrt{1-\bar{\alpha}_\tau-\sigma_t^2}\epsilon_\theta(\mathbf{x}_t,t)}_{指向\mathbf{x}_t的方向}+\underbrace{\sigma_t\epsilon}_{随机的噪声}
\end{aligned}
$$

## 方差参数化与模型变体

由于方差 $\sigma_t^2$ 在推导过程中未被唯一确定，DDIM论文提出了一个参数化形式：
$$
\sigma_t=\eta\sqrt{\frac{1-\bar{\alpha}_\tau}{1-\bar{\alpha}_t}}\sqrt{1-\alpha_t}
$$

其中 $\eta \in [0, 1]$ 是一个控制随机性的超参数：

1. **$\eta = 1$（DDPM等价）**：此时方差与DDPM的后验方差一致，采样过程完全等同于DDPM的马尔可夫过程。

2. **$\eta = 0$（确定性DDIM）**：此时 $\sigma_t = 0$，采样过程变为确定性映射：
   $$\mathbf{x}_\tau = \sqrt{\bar{\alpha}_\tau}\hat{\mathbf{x}}_0 + \sqrt{1-\bar{\alpha}_\tau-\sigma_t^2}\epsilon_\theta(\mathbf{x}_t,t)$$
   
   其中 $\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(\mathbf{x}_t,t)}{\sqrt{\bar{\alpha}_t}}$ 是网络预测的原始图像。

   这种确定性映射使得每个初始噪声 $\mathbf{x}_T$ 都对应唯一的生成结果 $\mathbf{x}_0$，这是DDIM的核心特性。

## 非马尔可夫采样与加速

DDIM的关键优势在于其反向过程不依赖于马尔可夫假设，这意味着采样可以在任意时间步序列上进行，而不必严格按照相邻时间步的顺序。

**理论基础**：由于DDIM的采样公式不依赖于前一时间步的具体值，我们可以从完整的时间步序列 $[T, T-1, ..., 2, 1]$ 中选择任意子序列 $[\tau_S, \tau_{S-1}, ..., \tau_2, \tau_1]$ 进行采样，其中 $S \ll T$。

**子序列选择策略**：论文提出了两种时间步子序列的构造方法：

1. **线性采样**：$\tau_i = \lfloor ci \rfloor$，其中 $c = \frac{T}{S}$
2. **二次采样**：$\tau_i = \lfloor ci^2 \rfloor$，其中 $c = \frac{T}{S^2}$

**加速效果**：通过选择合适的子序列，可以将采样步数从 $T$（通常为1000）减少到 $S$（通常为10-50），实现10-100倍的采样加速，同时保持生成质量。


# DDIM 的核心特性

## 1. 采样一致性（Sampling Consistency）

DDIM的确定性采样过程（$\eta = 0$）赋予了模型一个独特性质：对于给定的初始噪声 $\mathbf{x}_T$，无论使用何种时间步子序列进行采样，最终生成的图像 $\mathbf{x}_0$ 都高度一致。

**数学表述**：设 $\mathcal{S}_1$ 和 $\mathcal{S}_2$ 为两个不同的时间步子序列，则：
$$\|\mathbf{x}_0(\mathbf{x}_T, \mathcal{S}_1) - \mathbf{x}_0(\mathbf{x}_T, \mathcal{S}_2)\| \ll \|\mathbf{x}_0(\mathbf{x}_T, \mathcal{S}_1)\|$$

**实际应用**：这一特性使得 $\mathbf{x}_T$ 可以被视为 $\mathbf{x}_0$ 的隐空间表示，类似于VAE中的隐变量。在生成过程中，可以先使用较少时间步快速生成草图预览，确认大致方向后再使用更多时间步进行精细生成。

## 2. 语义插值（Semantic Interpolation）

基于采样一致性，DDIM支持在隐空间中进行语义插值。给定两个隐变量 $\mathbf{x}_T^{(0)}$ 和 $\mathbf{x}_T^{(1)}$，可以通过球面线性插值（Spherical Linear Interpolation, SLERP）构造中间隐变量：

$$\mathbf{x}_T^{(\alpha)} = \frac{\sin((1-\alpha)\theta)}{\sin\theta}\mathbf{x}_T^{(0)} + \frac{\sin(\alpha\theta)}{\sin\theta}\mathbf{x}_T^{(1)}$$

其中 $\alpha \in [0, 1]$ 是插值参数，$\theta$ 是两向量间的夹角：
$$\theta = \arccos\left(\frac{\langle\mathbf{x}_T^{(0)}, \mathbf{x}_T^{(1)}\rangle}{\|\mathbf{x}_T^{(0)}\| \cdot \|\mathbf{x}_T^{(1)}\|}\right)$$

**插值效果**：通过这种方式生成的中间图像序列展现出平滑的语义过渡，而不是简单的像素级插值，这证明了DDIM隐空间具有良好的语义结构。

# DDIM 的代码实现

从上面的推导过程可以发现，DDIM 假设的前向过程和 DDPM 相同，只有采样过程不同。因此想把 DDPM 改成 DDIM 并不需要重新训练，只要修改采样过程就可以了。

## 核心代码实现

### 1. 初始化与参数设置

```python
import torch
import math
from tqdm import tqdm

class DDIM:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        sample_steps: int = 20,
    ):
        """
        DDIM采样器初始化
        
        Args:
            num_train_timesteps: 训练时的时间步数（通常为1000）
            beta_start: β调度的起始值
            beta_end: β调度的结束值
            sample_steps: 采样时的时间步数（通常为10-50）
        """
        self.num_train_timesteps = num_train_timesteps
        
        # 定义β调度：β_t从beta_start线性增长到beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        
        # 计算α_t = 1 - β_t
        self.alphas = 1.0 - self.betas
        
        # 计算累积α：ᾱ_t = ∏_{i=1}^t α_i
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 定义采样时间步序列（线性采样策略）
        # 从T-1到0，均匀选择sample_steps个时间步
        self.timesteps = torch.linspace(num_train_timesteps - 1, 0, sample_steps).long()
```

### 2. DDIM采样算法

DDIM的核心采样公式为：
$$\mathbf{x}_\tau = \sqrt{\bar{\alpha}_\tau}\hat{\mathbf{x}}_0 + \sqrt{1-\bar{\alpha}_\tau-\sigma_t^2}\epsilon_\theta(\mathbf{x}_t,t) + \sigma_t\epsilon$$

其中方差参数化：
$$\sigma_t = \eta\sqrt{\frac{1-\bar{\alpha}_\tau}{1-\bar{\alpha}_t}}\sqrt{1-\alpha_t}$$

```python
    @torch.no_grad()
    def sample(
        self,
        unet,  # 预训练的噪声预测网络
        batch_size: int,
        in_channels: int,
        sample_size: int,
        eta: float = 0.0,  # 控制随机性的参数，η=0为确定性DDIM
    ):
        """
        DDIM采样过程
        
        Args:
            unet: 预训练的UNet模型，用于预测噪声
            batch_size: 批次大小
            in_channels: 输入通道数
            sample_size: 图像尺寸
            eta: 随机性控制参数，η∈[0,1]
        """
        # 将参数转移到设备上
        alphas = self.alphas.to(unet.device)
        alphas_cumprod = self.alphas_cumprod.to(unet.device)
        timesteps = self.timesteps.to(unet.device)
        
        # 初始化：从标准高斯分布采样 x_T ~ N(0, I)
        images = torch.randn(
            (batch_size, in_channels, sample_size, sample_size), 
            device=unet.device
        )
        
        # 迭代去噪：从T到1，使用预定义的时间步子序列
        for t, tau in tqdm(list(zip(timesteps[:-1], timesteps[1:])), desc='DDIM Sampling'):
            # 步骤1：使用网络预测噪声 ε_θ(x_t, t)
            pred_noise = unet(images, t).sample
            
            # 步骤2：计算方差 σ_t
            # σ_t = η * √[(1-ᾱ_τ)/(1-ᾱ_t)] * √(1-α_t)
            if not math.isclose(eta, 0.0):
                # 计算方差参数化的各个组成部分
                one_minus_alpha_prod_tau = 1.0 - alphas_cumprod[tau]  # 1 - ᾱ_τ
                one_minus_alpha_prod_t = 1.0 - alphas_cumprod[t]      # 1 - ᾱ_t
                one_minus_alpha_t = 1.0 - alphas[t]                   # 1 - α_t
                
                sigma_t = eta * torch.sqrt(
                    (one_minus_alpha_prod_tau * one_minus_alpha_t) / one_minus_alpha_prod_t
                )
            else:
                # η = 0 时，σ_t = 0，实现确定性采样
                sigma_t = torch.zeros_like(alphas[0])
            
            # 步骤3：计算预测的原始图像 x̂_0
            # x̂_0 = (x_t - √(1-ᾱ_t) * ε_θ(x_t,t)) / √ᾱ_t
            sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])           # √ᾱ_t
            sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod[t])  # √(1-ᾱ_t)
            
            predicted_x0 = (images - sqrt_one_minus_alphas_cumprod_t * pred_noise) / sqrt_alphas_cumprod_t
            
            # 步骤4：计算DDIM采样公式的三个组成部分
            
            # 第一项：√ᾱ_τ * x̂_0
            sqrt_alphas_cumprod_tau = torch.sqrt(alphas_cumprod[tau])  # √ᾱ_τ
            first_term = sqrt_alphas_cumprod_tau * predicted_x0
            
            # 第二项：√(1-ᾱ_τ-σ_t²) * ε_θ(x_t,t)
            coeff = torch.sqrt(1.0 - alphas_cumprod[tau] - sigma_t ** 2)
            second_term = coeff * pred_noise
            
            # 第三项：σ_t * ε（随机噪声项）
            if not math.isclose(eta, 0.0):
                epsilon = torch.randn_like(images)
                third_term = sigma_t * epsilon
            else:
                third_term = 0.0
            
            # 步骤5：更新图像 x_τ = 第一项 + 第二项 + 第三项
            images = first_term + second_term + third_term
        
        # 后处理：将图像从[-1,1]范围转换到[0,1]范围
        images = (images / 2.0 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        return images
```

### 3. 关键实现要点

1. **时间步子序列**：使用线性采样策略，从1000个训练时间步中选择20个进行采样
2. **方差控制**：通过η参数控制随机性，η=0实现确定性采样
3. **公式对应**：代码中的每个步骤都严格对应DDIM的数学公式
4. **设备管理**：确保所有张量都在同一设备上计算
5. **数值稳定性**：使用`math.isclose()`进行浮点数比较，避免精度问题

> 参考资料：
>
> 1. [DDIM 理论与实现](https://littlenyima.github.io/posts/14-denoising-diffusion-implicit-models/)
> 2. [diffusion model(二)：DDIM技术小结 (denoising diffusion implicit model)](http://www.myhz0606.com/article/ddim)
> 3. [扩散模型（一）| DDPM & DDIM](https://lichtung612.github.io/posts/1-diffusion-models/)
