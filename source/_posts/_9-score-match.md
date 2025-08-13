---
title: 笔记｜生成模型（七）：分数匹配理论（Score Matching）
date: 2025-08-14 23:08:30
cover: false
mathjax: true
categories:
 - Notes
tags:
 - Deep learning
 - Generative models theory
series: Diffusion Models theory
---

> 论文链接：*[Estimation of Non-Normalized Statistical Models by Score Matching](https://www.jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf)*

分数匹配（Score Matching）是一种用于估计非归一化概率密度函数的强大方法。与传统的最大似然估计不同，分数匹配不需要计算归一化常数，这使得它在处理复杂的高维分布时具有显著优势。本文将详细介绍分数匹配的理论基础、数学推导以及实际应用。

# 分数匹配的基本概念

## 什么是分数函数（Score Function）

给定一个概率密度函数 $p(\mathbf{x})$，其分数函数定义为对数概率密度的梯度：

$$\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})$$

**物理意义**：分数函数表示在给定点 $\mathbf{x}$ 处，概率密度增长最快的方向。它描述了数据分布在该点的"梯度信息"。

**重要性质**：
1. 分数函数与归一化常数无关：如果 $p(\mathbf{x}) = \frac{1}{Z} \tilde{p}(\mathbf{x})$，则 $\nabla_{\mathbf{x}} \log p(\mathbf{x}) = \nabla_{\mathbf{x}} \log \tilde{p}(\mathbf{x})$
2. 分数函数在数据分布的高密度区域较小，在低密度区域较大

## 分数匹配的核心思想

分数匹配的目标是学习一个参数化的分数函数 $\mathbf{s}_\theta(\mathbf{x})$，使其与真实数据分布的分数函数 $\mathbf{s}(\mathbf{x})$ 尽可能接近。

**关键优势**：
- 不需要计算归一化常数 $Z$
- 适用于高维数据
- 理论基础扎实

# 分数匹配的数学推导

## 目标函数

分数匹配的目标函数定义为真实分数函数与模型分数函数之间的平方差：

$$J(\theta) = \frac{1}{2} \mathbb{E}_{p(\mathbf{x})} \left[ \|\mathbf{s}_\theta(\mathbf{x}) - \mathbf{s}(\mathbf{x})\|^2 \right]$$

其中 $\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})$ 是真实分数函数。

## 关键推导

由于我们无法直接访问真实分数函数 $\mathbf{s}(\mathbf{x})$，需要将目标函数转换为可计算的形式。

**步骤1**：展开平方项
$$J(\theta) = \frac{1}{2} \mathbb{E}_{p(\mathbf{x})} \left[ \|\mathbf{s}_\theta(\mathbf{x})\|^2 - 2\mathbf{s}_\theta(\mathbf{x})^T \mathbf{s}(\mathbf{x}) + \|\mathbf{s}(\mathbf{x})\|^2 \right]$$

**步骤2**：利用积分恒等式
对于任意函数 $\mathbf{f}(\mathbf{x})$，有：
$$\mathbb{E}_{p(\mathbf{x})} \left[ \mathbf{f}(\mathbf{x})^T \mathbf{s}(\mathbf{x}) \right] = -\mathbb{E}_{p(\mathbf{x})} \left[ \nabla_{\mathbf{x}} \cdot \mathbf{f}(\mathbf{x}) \right]$$

其中 $\nabla_{\mathbf{x}} \cdot \mathbf{f}(\mathbf{x})$ 是向量场 $\mathbf{f}(\mathbf{x})$ 的散度。

**步骤3**：应用积分恒等式
将 $\mathbf{f}(\mathbf{x}) = \mathbf{s}_\theta(\mathbf{x})$ 代入，得到：
$$\mathbb{E}_{p(\mathbf{x})} \left[ \mathbf{s}_\theta(\mathbf{x})^T \mathbf{s}(\mathbf{x}) \right] = -\mathbb{E}_{p(\mathbf{x})} \left[ \nabla_{\mathbf{x}} \cdot \mathbf{s}_\theta(\mathbf{x}) \right]$$

**步骤4**：最终目标函数
$$J(\theta) = \mathbb{E}_{p(\mathbf{x})} \left[ \frac{1}{2} \|\mathbf{s}_\theta(\mathbf{x})\|^2 + \nabla_{\mathbf{x}} \cdot \mathbf{s}_\theta(\mathbf{x}) \right] + \text{const}$$

其中常数项 $\|\mathbf{s}(\mathbf{x})\|^2$ 与参数 $\theta$ 无关，可以忽略。

## 散度计算

对于向量值函数 $\mathbf{s}_\theta(\mathbf{x}) = [s_1(\mathbf{x}), s_2(\mathbf{x}), ..., s_d(\mathbf{x})]^T$，其散度为：

$$\nabla_{\mathbf{x}} \cdot \mathbf{s}_\theta(\mathbf{x}) = \sum_{i=1}^d \frac{\partial s_i(\mathbf{x})}{\partial x_i}$$

# 分数匹配的算法实现

## 基本算法

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ScoreNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def score_matching_loss(score_net, data_samples):
    """
    计算分数匹配损失
    
    Args:
        score_net: 分数网络
        data_samples: 数据样本 [batch_size, dim]
    """
    data_samples.requires_grad_(True)
    
    # 计算模型分数函数
    score_pred = score_net(data_samples)  # [batch_size, dim]
    
    # 计算散度
    divergence = 0.0
    for i in range(data_samples.shape[1]):
        # 对第i个维度求偏导
        grad_i = torch.autograd.grad(
            score_pred[:, i].sum(), 
            data_samples, 
            create_graph=True
        )[0][:, i]
        divergence += grad_i
    
    # 计算损失
    loss = 0.5 * torch.sum(score_pred**2, dim=1) + divergence
    
    return loss.mean()

# 训练循环
def train_score_network(score_net, data_loader, num_epochs=1000):
    optimizer = optim.Adam(score_net.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_data in data_loader:
            optimizer.zero_grad()
            
            loss = score_matching_loss(score_net, batch_data)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(data_loader):.4f}")
```

## 改进版本：切片分数匹配（Sliced Score Matching）

基本分数匹配需要计算散度，这在实践中可能不稳定。切片分数匹配通过随机投影来避免散度计算：

```python
def sliced_score_matching_loss(score_net, data_samples, num_slices=1):
    """
    切片分数匹配损失
    
    Args:
        score_net: 分数网络
        data_samples: 数据样本 [batch_size, dim]
        num_slices: 切片数量
    """
    data_samples.requires_grad_(True)
    
    # 生成随机向量
    random_vectors = torch.randn_like(data_samples)  # [batch_size, dim]
    
    # 计算投影后的分数
    score_pred = score_net(data_samples)  # [batch_size, dim]
    projected_score = torch.sum(score_pred * random_vectors, dim=1)  # [batch_size]
    
    # 计算投影后的梯度
    projected_grad = torch.autograd.grad(
        projected_score.sum(), 
        data_samples, 
        create_graph=True
    )[0]
    
    # 计算损失
    loss = torch.sum(score_pred * random_vectors, dim=1) + \
           0.5 * torch.sum(projected_grad * random_vectors, dim=1)
    
    return loss.mean()
```

# 分数匹配与扩散模型的关系

## 理论联系

分数匹配与扩散模型之间存在深刻的联系：

1. **扩散模型的分数函数**：在扩散过程中，分数函数 $\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)$ 描述了噪声图像 $\mathbf{x}_t$ 的去噪方向

2. **去噪分数匹配**：通过向数据添加噪声，可以学习条件分数函数 $\nabla_{\mathbf{x}} \log p(\mathbf{x} | \sigma)$

3. **朗之万动力学采样**：学习到分数函数后，可以使用朗之万动力学进行采样：
   $$\mathbf{x}_{t+1} = \mathbf{x}_t + \epsilon \nabla_{\mathbf{x}} \log p(\mathbf{x}_t) + \sqrt{2\epsilon} \mathbf{z}_t$$

## 实际应用

```python
def langevin_sampling(score_net, initial_samples, num_steps=1000, step_size=1e-4):
    """
    使用朗之万动力学进行采样
    
    Args:
        score_net: 训练好的分数网络
        initial_samples: 初始样本
        num_steps: 采样步数
        step_size: 步长
    """
    samples = initial_samples.clone()
    
    for step in range(num_steps):
        # 计算分数
        with torch.no_grad():
            score = score_net(samples)
        
        # 朗之万更新
        noise = torch.randn_like(samples)
        samples = samples + step_size * score + torch.sqrt(2 * step_size) * noise
        
        # 可选：添加噪声退火
        if step % 100 == 0:
            print(f"Step {step}, Sample norm: {torch.norm(samples).item():.4f}")
    
    return samples
```

# 分数匹配的优势与挑战

## 优势

1. **无需归一化常数**：直接处理非归一化概率密度
2. **理论基础扎实**：基于信息几何学
3. **适用于高维数据**：计算复杂度与维度线性相关
4. **与扩散模型兼容**：为现代生成模型奠定基础

## 挑战

1. **散度计算困难**：需要二阶导数，计算成本高
2. **训练不稳定**：梯度可能爆炸或消失
3. **采样效率低**：朗之万动力学需要大量步骤
4. **模式崩塌**：可能无法覆盖所有数据模式

# 总结

分数匹配为生成模型提供了一个强大的理论基础，它通过直接学习概率分布的梯度信息来避免归一化常数的计算。虽然存在一些实践挑战，但分数匹配的思想为后来的扩散模型、能量模型等奠定了基础。

在现代深度学习中，分数匹配已经演化为更高效的形式，如去噪分数匹配、切片分数匹配等，这些方法在图像生成、音频合成等领域取得了显著成功。

> 参考资料：
>
> 1. [Estimation of Non-Normalized Statistical Models by Score Matching](https://www.jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf)
> 2. [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600)
> 3. [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
