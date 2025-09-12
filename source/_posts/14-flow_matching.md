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

图像生成任务的核心在于建模样本如何从简单的先验分布（如标准高斯分布 $\mathcal{N}(0, \mathbf{I})$）转换到复杂的数据分布 $p_{\text{data}}(x)$。与基于去噪的扩散模型通过学习逐步去噪过程不同，Flow Matching 直接建模样本在两个分布之间的连续时间传输过程。从数学角度来看，Flow Matching 学习一个时间相关的向量场 $v_t(x)$，该向量场定义了从先验分布 $p_0$ 到目标分布 $p_1$ 的概率路径 $\{p_t\}_{t \in [0,1]}$。这种方法将图像生成问题转化为最优传输问题：如何找到一个连续的概率流，使得 $p_0 = \mathcal{N}(0, \mathbf{I})$ 和 $p_1 = p_{\text{data}}(x)$。因此，Flow Matching 的核心思想是在已知源分布 $p_0$ 和目标分布 $p_1$ 之间构造一条时间参数化的概率路径 $\{p_t\}_{t \in [0,1]}$，并学习生成该概率路径的速度场 $v_t(x)$。

![Flow Matching通过建模时间相关速度场学习从噪声到数据的平滑路径。](/chengYi-xun/img/flow_matching.png)

下图展示了 Flow Matching 中关键变量的定义：$x_0$ 表示初始噪声，$x_1$ 表示目标数据，$x_t$ 表示时间 $t$ 处的中间状态。
$x_t = (1-t)x_0 + tx_1 = x_0 + t(x_1-x_0)$，其中t一般定义为采样时间步的方差，在代码中一般用sigma表示，sigma的范围是从0到1。（一般用倒数表示，随机采样正整数时间步，则sigma = 1/time，这里的sigma就是公式里面的t）。$x_0 + t(x_1-x_0)$可以理解成从$x_0$的起点开始，朝着$x_{0->1}$的方向步进了t的长度。

![Flow Matching 变量定义示意图](/chengYi-xun/img/flow.jpeg)

具体的实现步骤包括：

建模过程首先从每个分布中采样一个点，然后用路径连接它们。最常用的连接方式是直线路径：$x(t) = (1-t)x_0 + tx_1$。这种线性插值方法在$x_0$和$x_1$之间建立了最直接的连接路径。虽然也可以使用其他插值方法如球面插值，但线性插值因其简单性和良好的实际效果而被广泛采用。

目标是学习一个时间相关的速度场$v(x,t)$，该速度场描述轨迹上每个点的瞬时速度。由于我们已经从路径定义中知道了真实速度，神经网络的任务是近似$v(x,t) = x_1 - x_0$。

训练过程就是使模型的预测速度与真实速度匹配。

$$\mathcal{L} = \mathbb{E}_{x_0,x_1,t}\left[\left\|f_\theta(x(t),t) - \frac{d}{dt}x(t)\right\|^2\right]$$

• $x(t) = (1-t)x_0 + tx_1$: interpolated point between noise and data

• $f_\theta(x(t),t)$: predicted velocity by the neural network  

• $\frac{d}{dt}x(t) = x_1 - x_0$: ground truth velocity

• $\mathbb{E}_{x_0,x_1,t}$: expectation over random noise/data pairs and interpolation time



# Flow Matching采样过程

在训练阶段，网络学习了将点从噪声移动到数据的速度场$v(x,t)$。训练结束后同时访问$x_0$和$x_1$，但在生成阶段只有$x_0$。采样的目标是取一个噪声点并将其推进到$p_1$分布，最终获得类似自然图像的结果。

采样过程从样本$x_0 \sim p_0$(例如标准高斯噪声)开始，然后定义从$t = 0$到$t = 1$的时间网格，将其均分割成一系列步骤。在每个时间步，我们向前求解ODE来更新样本，采样过程一般使用欧拉积分进行迭代，欧拉积分是一阶离散的近似数值积分，为了更加精确，也会有人使用4阶龙格库塔积分进行更精细的数值积分。

$$x_{t+\Delta t} = x_t + \Delta t \cdot f_\theta(x_t, t)$$

• $x_t$: current sample at time $t$

• $\Delta t$: step size (e.g. $\frac{1}{N}$ for $N$ steps)

• $f_\theta(x_t, t)$: learned velocity field — tells how to update the sample at this point in time

• $x_{t+\Delta t}$: next sample along the trajectory, after one step forward using the predicted velocity

**方程4.** Flow Matching采样的更新规则

速度场$v(x,t)$在每个步骤中与当前的$x$和$t$一起使用，以获得$v(t)$的估计。一旦到达$t = 1$，就得到了一个完整看起来像自然图像的样本。这个过程类似于跟随流场，我们沿着学习的速度路径将样本"推向"数据方向。

上述的不管是欧拉数值积分，还是龙哥库塔积分，他们都是一种确定性积分，所以是一条确定性路径。最终的生成样本的随机性只来源于0时刻的随机采样的高斯白噪声。
所以SD3的实现中其实还有另一种形式，即随机路径。



这里我们直观的展现出Flow Matching和其他方法在不同分布寻找路径的区别。

![Flow Matching And Diffusion](/chengYi-xun/img/flow_and_diffusion.png)

### 训练代码
```python

# FlowMatchEulerDiscreteScheduler中的time stamp的创建：
# 创建线性时间步序列并反转（从大到小）
timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
# 转换为PyTorch张量
timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

# 将时间步归一化为sigma值（0到1之间）
sigmas = timesteps / num_train_timesteps
# 如果不使用动态偏移，应用静态偏移变换
if not use_dynamic_shifting:
    # when use_dynamic_shifting is True, we apply the timestep shifting on the fly based on the image resolution
    # 应用偏移公式：shift * sigmas / (1 + (shift - 1) * sigmas)
    #其实是 对 [0,1] 区间内的 sigmas 做了一种非线性映射，通过参数 shift 改变时间步（timestep）对应的分布，让采样的重心往前或往后偏移。
    #相当于 在时间轴上重新分配采样密度
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

# 将sigma转换回时间步
self.timesteps = sigmas * num_train_timesteps
```

```python
def compute_density_for_timestep_sampling(  # 函数定义：计算时间步采样的密度
    weighting_scheme: str,  # 权重方案类型：决定使用哪种采样策略
    batch_size: int,  # 批次大小：需要生成多少个时间步样本
    logit_mean: float = None,  # logit正态分布的均值参数（仅logit_normal方案使用）
    logit_std: float = None,  # logit正态分布的标准差参数（仅logit_normal方案使用）
    mode_scale: float = None,  # 模式缩放参数（仅mode方案使用）
    device: Union[torch.device, str] = "cpu",  # 计算设备：CPU或GPU
    generator: Optional[torch.Generator] = None,  # 随机数生成器：用于可重现的随机性
):
    
    # 第一种采样策略：logit正态分布，从正态分布中采样，使用指定的均值和标准差，最后通过sigmoid函数将正态分布的值映射到(0,1)区间
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device, generator=generator)
        u = torch.nn.functional.sigmoid(u)
        
    # 第二种采样策略：模式采样（偏向某些时间步）
    # 首先生成均匀分布的随机数 [0,1)，然后应用复杂的变换函数来改变分布形状u' = 1 - u - mode_scale * (cos²(πu/2) - 1 + u)
    ## 这个变换会使某些时间步被更频繁地采样
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
        
    # 第三种采样策略：均匀分布（默认情况）。生成标准的均匀分布随机数 [0,1)
    else:
        u = torch.rand(size=(batch_size,), device=device, generator=generator)
    return u

noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
transformer = FlowMatching2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )

def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
    # 获取调度器的sigma值
    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()  # 获取对应的sigma值并展平
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma    

for step, batch in enumerate(train_dataloader):
        # 将图像转换到潜在空间
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype)  # 获取像素值
        model_input = vae.encode(pixel_values).latent_dist.sample()  
        #vae偏移和缩放，使得任务图像的输入latent都能趋近标准正态分布
        model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
        model_input = model_input.to(dtype=weight_dtype) 

        noise = torch.randn_like(model_input)  # 生成随机噪声
        bsz = model_input.shape[0]  
        # 为每个图像采样随机时间步
        # 用于非均匀时间步采样的加权方案
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,  # 加权方案
            batch_size=bsz,  # 批次大小
            logit_mean=args.logit_mean,  # logit均值
            logit_std=args.logit_std,  # logit标准差
            mode_scale=args.mode_scale,  # 模式缩放
        )
        # 计算时间步索引
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)

        # 根据Flow Matching添加噪声
        # zt = (1 - texp) * x + texp * z1
        sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise  # 添加噪声

        # 获取用于条件的文本编码
        prompt_embeds = batch["prompt_embeds"].to(dtype=weight_dtype)  # 提示嵌入
        # 用DIT预测速度场
        model_pred = transformer(
            hidden_states=noisy_model_input,  
            timestep=timesteps,  
            encoder_hidden_states=prompt_embeds,  
            return_dict=False,  
        )[0]
        target = noise - model_input  # 否则目标是噪声与输入的差
        velocity_loss = args.weighting* (model_pred - target) ** 2).reshape(target.shape[0], -1)
        loss = torch.mean(velocity_loss,1)
        loss = loss.mean()  
        accelerator.backward(loss)
        optimizer.step()  
        lr_scheduler.step()  
        optimizer.zero_grad(set_to_none=args.set_grads_to_none)  # 清零梯度
```

### 推理代码

```python
shape = (
  batch_size,
  num_channels_latents,
  int(height) // self.vae_scale_factor,
  int(width) // self.vae_scale_factor,
)
latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
transformer = FlowMatching2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )
timesteps = scheduler.timesteps
num_inference_steps = len(timesteps)
prompt_embeds = encode_prompt(prompt=prompt)
for i, t in enumerate(timesteps):  # 遍历时间步
    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents 
    timestep = t.expand(latent_model_input.shape[0])  # 将时间步扩展到批次维度
    noise_pred = transformer(  
        hidden_states=latent_model_input,  
        timestep=timestep, 
        encoder_hidden_states=prompt_embeds,  
        return_dict=False,
    )[0]  
    if self.do_classifier_free_guidance:  
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)  
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)  # 计算引导后的速度预测

    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]  # 使用调度器执行去噪步骤
#scheduler 的setp
def step(
    self,  # 调度器实例
    model_output: torch.FloatTensor,  # 模型输出（预测的噪声或速度场）
    timestep: Union[float, torch.FloatTensor],  # 当前时间步
    sample: torch.FloatTensor,  # 当前样本状态
    s_churn: float = 0.0,  # 随机性参数，控制噪声注入
    s_tmin: float = 0.0,  # 最小时间阈值
    s_tmax: float = float("inf"),  # 最大时间阈值
    s_noise: float = 1.0,  # 噪声缩放因子
    generator: Optional[torch.Generator] = None,  # 随机数生成器
    per_token_timesteps: Optional[torch.Tensor] = None,  # 每个token的时间步（可选）
    return_dict: bool = True,  # 是否返回字典格式结果
) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:  # 返回类型：输出对象或元组


    if self.step_index is None:
        self._init_step_index(timestep)  # 初始化步骤索引

    sample = sample.to(torch.float32)


    if per_token_timesteps is not None:
        per_token_sigmas = per_token_timesteps / self.config.num_train_timesteps
        sigmas = self.sigmas[:, None, None]
        lower_mask = sigmas < per_token_sigmas[None] - 1e-6
        # 应用掩码获取较小的sigma值
        lower_sigmas = lower_mask * sigmas
        # 获取每个位置的最大较小sigma值
        lower_sigmas, _ = lower_sigmas.max(dim=0)

        # 设置当前和下一个sigma值
        current_sigma = per_token_sigmas[..., None]  # 当前sigma
        next_sigma = lower_sigmas[..., None]  # 下一个sigma
        dt = current_sigma - next_sigma  # 时间步长
    else:
        # 标准情况：使用全局时间步
        sigma_idx = self.step_index  # 获取当前sigma索引
        sigma = self.sigmas[sigma_idx]  # 当前sigma值
        sigma_next = self.sigmas[sigma_idx + 1]  # 下一个sigma值

        current_sigma = sigma  # 设置当前sigma
        next_sigma = sigma_next  # 设置下一个sigma
        dt = sigma_next - sigma  # 计算时间步长（通常为负值）

    # 根据采样模式选择不同的更新策略
    #在标准的Flow Matching理论中，x_0 代表纯噪声，x_1 代表目标数据，在 训练或采样轨迹推理 里，我们不是单独看噪声，而是要看「噪声和某个真实数据点的配对」
    if self.config.stochastic_sampling:
        #不执着于原来的终点，而是随机选择新的噪声终点。样就能得到一条新的轨迹，从而在推理时引入随机性。
        #x_t = (1-t)x_0 + t * x_1 = x_0+t(x_1-x_0) sample=x_t  (x_1-x_0) = model_output
        #sample = x_0+t(x_1-x_0) = x_0+t*model_output
        #x_0 = sample - t * model_output
        x0 = sample - current_sigma * model_output  # 估计x0，这里指的是纯噪声,即估计起点噪声
        #这里的 x0 是估计的“初始噪声样本”（即训练时的 x₀）
        #它是「数据点对应的噪声起点」，而不是“纯粹意义上的 N(0,I)”噪声。也是 Flow Matching / Diffusion 理论里容易混淆的点。
        #对于Diffusion 理论，所有图像的起点都是相同的，因为他们的起点都是纯噪声。（高斯白噪声）
        #在 Flow Matching（或者 Diffusion 训练）里，我们假设有一个映射：每个图像都对应一种高斯白噪声的采样样本，不同的采样样本代表不同的图像。
        #数据点对应的噪声起点” 指的是在训练时，与某个特定 𝑥1 搭配在一起的那个噪声点 𝑥0
        #这里重新生成了一个与 x_0 形状完全相同的全新高斯噪声。
        noise = torch.randn_like(sample)  # 2. 生成新的随机噪声
        # prev_sample = (1 - t_next) * x0 + t_next * noise
        #这一步非常巧妙。它没有直接使用欧拉法沿着旧路径回退一步。
        #相反，它构建了一条新的路径。即从数据起点x_0开始，随机朝着一个方向走一个步长next_sigma，得到prev_sample
        prev_sample = (1.0 - next_sigma) * x0 + next_sigma * noise
    else:
        # 确定性采样模式：使用Euler方法，如果只做 Euler，就会沿着这条方向推进，得到一条固定的轨迹（确定性）
        # 使用Euler方法进行数值积分：x_{t+dt} = x_t + dt * v_t
        prev_sample = sample + dt * model_output

    # 完成后将步骤索引增加1
    self._step_index += 1
    # 如果不是per-token模式，将样本转换回模型兼容的数据类型
    if per_token_timesteps is None:
        # 将样本转换回模型输出的数据类型
        prev_sample = prev_sample.to(model_output.dtype)

    # 根据return_dict参数决定返回格式
    if not return_dict:
        return (prev_sample,)  # 返回元组格式

    # 返回结构化输出对象
    return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
```




> 参考资料：
>
> 1. [深入解析Flow Matching技术](https://zhuanlan.zhihu.com/p/685921518)
> 2. [【AI知识分享】你一定能听懂的扩散模型Flow Matching基本原理深度解析](https://www.bilibili.com/video/BV1Wv3xeNEds/)
> 3. [flow_matching](https://littlenyima.github.io/posts/51-flow-matching-for-diffusion-models/)
> 4. [Normalizing Flow](https://littlenyima.github.io/posts/12-basic-concepts-of-normalizing-flow/)