# -*- coding: utf-8 -*-
import io
with io.open("source/_posts/14-flow_matching.md", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace(
    "> 4. [Normalizing Flow](https://littlenyima.github.io/posts/12-basic-concepts-of-normalizing-flow/)",
    "> 4. [Normalizing Flow](https://littlenyima.github.io/posts/12-basic-concepts-of-normalizing-flow/)\n\n# 总结与算法对比\n\n**算法对比：Flow Matching vs DDPM**\n- **路径**：DDPM 的加噪路径是弯曲且充满随机性的（马尔可夫链），而 Flow Matching（特别是 Rectified Flow）的路径是两点之间的直线。\n- **速度**：因为直线路径更平滑，ODE 求解器可以用更大的步长（更少的步数）进行采样，这使得 Flow Matching 的生成速度远快于传统的 DDPM。\n- **实现**：Flow Matching 的 Loss 计算极其简单（预测速度与真实速度的 MSE），不需要像 DDPM 那样推导复杂的变分下界（ELBO）和各时间步的权重。\n\n**开源代码参考**：\n目前最前沿的开源图像大模型，如 **Stable Diffusion 3** 和 **Flux**，都已经全面抛弃了 DDPM，转而采用 Flow Matching（Rectified Flow）作为其核心的生成机制。你可以在 `diffusers` 库的 `FlowMatchEulerDiscreteScheduler` 中看到其采样过程的实现。\n\n> 下一篇：[笔记｜生成模型（十四）：Stable Diffusion 3 架构解析 (MMDiT)](/chengYi-xun/2026/04/03/15-sd3/)"
)

content = content.replace(
    "• $\\mathbb{E}_{x_0,x_1,t}$: 随机噪声/数据对和插值时间的期望",
    "• $\\mathbb{E}_{x_0,x_1,t}$: 随机噪声/数据对和插值时间的期望\n\n**开源代码参考：**\n在实际的 PyTorch 实现中（例如 Stable Diffusion 3 或 Flux 的训练代码），Flow Matching 的 Loss 计算极其简单直观：\n\n```python\n# x_1: 真实图片 (目标)\n# x_0: 纯高斯噪声 (起点)\n# t: 随机采样的时间步 [0, 1]\n\n# 1. 线性插值得到当前时刻的 x_t\nx_t = (1 - t) * x_0 + t * x_1\n\n# 2. 计算真实的速度场 (目标速度)\ntarget_velocity = x_1 - x_0\n\n# 3. 神经网络预测速度场\npredicted_velocity = model(x_t, t)\n\n# 4. 计算 MSE Loss\nloss = F.mse_loss(predicted_velocity, target_velocity)\n```\n"
)

with io.open("source/_posts/14-flow_matching.md", "w", encoding="utf-8") as f:
    f.write(content)
