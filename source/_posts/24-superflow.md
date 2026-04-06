---
title: 笔记｜生成模型（二十三）：SuperFlow 与图像生成 RL 前沿（2026）
date: 2026-04-05 14:00:00
cover: false
mathjax: true
categories:
 - Notes
tags:
 - Deep learning
 - Generative models theory
 - Reinforcement Learning
 - Flow Matching
series: Diffusion Models theory
---

> 本文为 RL 系列的最终篇。在第 21 篇中我们介绍了 Flow-GRPO，将 GRPO 应用于基于 Flow Matching 的图像生成。本文将介绍其后续改进 SuperFlow，以及 2026 年图像/视频生成 RL 的统一框架生态，最后回顾整个系列的完整技术脉络。
>
> 论文：[SuperFlow: Training Flow Matching Models with RL on the Fly](https://arxiv.org/abs/2512.17951)（2025.12, revised 2026.01）

# Flow-GRPO 的三个遗留问题

**延续之前"橘猫坐在蓝色沙发上"的例子。** 回忆 Flow-GRPO 的做法：对每个 Prompt 生成 $G$ 张图像，用奖励模型打分，计算组内相对优势，然后用裁剪策略梯度更新模型。

但在大规模实践中，这个看似完美的流程暴露出三个问题：

## 问题一：固定组大小导致计算浪费

Flow-GRPO 对**所有** Prompt 使用相同的组大小（比如 $G = 16$）。但不同 Prompt 的"不确定性"差异巨大：

| Prompt | 模型当前水平 | 生成 16 张图的结果 | 有效性 |
|:---|:---:|:---|:---:|
| "一只猫"（简单） | 已掌握 | 16 张都高分，$\sigma_R \approx 0$ | 浪费 |
| "橘猫坐蓝色沙发"（中等） | 部分掌握 | 分数差异大，$\sigma_R > 0$ | **有效** |
| "穿西装的猫弹钢琴"（困难） | 完全不会 | 16 张都低分，$\sigma_R \approx 0$ | 浪费 |

对"已掌握"和"完全不会"的 Prompt，16 个 rollout 几乎没有产生学习信号——但每张 1024×1024 的图像需要 50 步 ODE 求解，计算代价极高。

## 问题二：轨迹级优势的信用分配偏差

Flow-GRPO 对整条去噪轨迹（$t = T \to t = 0$）使用**相同的优势值** $\hat{A}_i$。但去噪过程的不同阶段角色截然不同：

- **前期**（$t$ 接近 1）：确定大致构图和主体，决策空间大，"动作噪声"高
- **后期**（$t$ 接近 0）：细化纹理和细节，决策空间小，"动作噪声"低

用统一的 $\hat{A}_i$ 加权所有时间步，等于假设"选择构图方向"和"微调毛发细节"的重要性完全一样——这显然不合理。

## 问题三：训练后期的稳定性

Flow-GRPO 使用纯组内相对基线。在训练后期，模型质量提升导致组内奖励方差下降，优势信号趋近于零，训练可能"停滞"或因噪声主导而不稳定。

---

# SuperFlow 的三项改进

## 改进一：方差感知的动态采样（Dynamic Sampling）

SuperFlow 借鉴了 SPO（Self-Play Optimization）的思想，为每个 Prompt 维护一个**在线奖励追踪器** $\hat{v}(c)$，用 Beta 分布建模：

$$
\hat{v}(c) = \frac{\alpha(c)}{\alpha(c) + \beta(c)}
$$

每次观测到奖励 $r(c, y)$ 后，用遗忘因子 $\rho(c)$ 更新 Beta 参数（$\rho$ 与策略漂移程度相关，漂移大则遗忘快）。

**Prompt 不确定性**定义为 Bernoulli 方差的开根号：

$$
w(c) \propto \sqrt{\hat{v}(c)(1 - \hat{v}(c)) + \epsilon}
$$

当 $\hat{v}(c) \approx 0.5$ 时不确定性最大（模型"半会半不会"），$\hat{v}(c)$ 接近 0 或 1 时不确定性最小。

**动态组大小**：将 Prompt 按不确定性分为 $K$ 个区间（bin），每个 Prompt 的组大小根据其所在区间分配：

$$
m(c) = M_{\max} - b(c) + 1
$$

其中 $b(c) \in \{1, \ldots, K\}$ 是 Prompt $c$ 的不确定性区间编号。

**用例子理解**（$M_{\max} = 24$，$K = 4$）：

| Prompt | $\hat{v}(c)$ | 不确定性 $w(c)$ | bin | 组大小 $m(c)$ |
|:---|:---:|:---:|:---:|:---:|
| "一只猫" | 0.95 | 低 | 4 | $24 - 4 + 1 = 21$ |
| "橘猫坐蓝色沙发" | 0.55 | **高** | 1 | $24 - 1 + 1 = 24$ |
| "穿西装猫弹钢琴" | 0.05 | 低 | 4 | $24 - 4 + 1 = 21$ |

同时，高不确定性的 Prompt 在 batch 采样中的权重 $w(c)$ 也更大——**更频繁地被训练到**。

## 改进二：步级优势重估计（Step-Level Advantage Re-estimation）

SuperFlow 不再对所有时间步使用统一的轨迹级优势，而是根据每一步的"动作不确定性"（即反向 SDE 的扩散系数 $\sigma_t$）重新加权：

$$
\hat{A}_t = \eta \cdot \sigma_t \cdot A_\tau
$$

其中 $A_\tau$ 是轨迹级优势（来自奖励减去 Beta 追踪器基线），$\sigma_t$ 是第 $t$ 步的 SDE 噪声标准差，$\eta$ 是缩放超参数。

**物理直觉**：$\sigma_t$ 大的时间步（前期），模型的"自由度"高，策略的选择空间大，应该获得更大的优势加权——因为这些步骤对最终图像质量的影响更大。$\sigma_t$ 小的时间步（后期），微调细节，策略已经高度确定，优势加权应该更小。

**用例子理解**（假设 10 步去噪，轨迹级优势 $A_\tau = +1.5$）：

| 时间步 $t$ | $\sigma_t$ | $\hat{A}_t = \eta \sigma_t \cdot 1.5$ | 含义 |
|:---:|:---:|:---:|:---|
| $t = 10$（从纯噪声开始） | 0.8 | $1.2\eta$ | 选择构图方向→高权重 |
| $t = 5$（中间步） | 0.3 | $0.45\eta$ | 细化结构→中权重 |
| $t = 1$（接近成品） | 0.05 | $0.075\eta$ | 调整细节→低权重 |

## 改进三：运行基线替代纯组相对

SuperFlow 用 Beta 追踪器 $\hat{v}(c)$ 作为**每 Prompt 的运行基线**，而非纯粹依赖组内均值：

$$
A(c, y) = r(c, y) - \hat{v}(c)
$$

然后在 batch 内做标准化。这比纯组相对更稳定，因为：

1. 追踪器 $\hat{v}(c)$ 综合了历史信息，不会因当前组内样本的偶然性而剧烈波动
2. 即使当前组内所有样本的奖励都很高（$\sigma_R \approx 0$），只要它们高于历史追踪值，仍然能产生正向梯度

---

# SuperFlow 的训练效果

在 SD3.5-Medium 骨干上使用 LoRA（$r = 32$，$\alpha = 64$）进行 RL 训练：

| 方法 | GenEval 综合 | OCR 准确率 | PickScore |
|:---:|:---:|:---:|:---:|
| SD3.5-M（基线） | 0.5814 | 0.5717 | 0.8304 |
| Flow-GRPO | 0.7829 | 0.7252 | 0.8536 |
| Flow-SPO | 0.7540 | 0.7652 | 0.8516 |
| **SuperFlow** | **0.8045** | **0.8413** | **0.8685** |

SuperFlow 在 OCR（图像中的文字渲染准确度）上提升最为显著——相比 SD3.5-M 提升了 **47.2%**，相比 Flow-GRPO 提升了 **16.0%**。这验证了步级优势重估计对细粒度任务的重要性。

训练效率方面，SuperFlow 比 Flow-GRPO 减少了 **5.4%-56.3%** 的训练步数和 **5.2%-16.7%** 的训练时间。

消融实验显示，步级优势重估计（Adv-Est）对 OCR 的贡献最大（移除后下降 16.4%），动态采样（Dyn-Samp）对训练稳定性贡献最大（移除后后期出现性能崩溃）。

---

# 2026 年图像/视频生成 RL 的统一框架

SuperFlow 之外，两个开源框架正在将图像/视频生成的 RL 训练推向工程化：

## Flow-Factory

[Flow-Factory](https://github.com/X-GenGroup/Flow-Factory) 是一个模块化的统一框架，支持 GRPO、DiffusionNFT、AWM 等算法跨多种模型进行 RL 训练：

| 维度 | 支持范围 |
|:---|:---|
| 模型 | Flux, Qwen-Image, WAN Video |
| 算法 | FlowGRPO, MixGRPO, CPS, AWM |
| 奖励 | PickScore, ImageReward, CLIP, 美学评分 |
| 模态 | T2I（文生图）、T2V（文生视频）、I2V（图生视频） |

其核心设计思想是**解耦算法、模型和奖励**——更换奖励函数不需要修改算法代码，更换模型不需要修改奖励逻辑。

## GenRL

[GenRL](https://github.com/ModelTC/GenRL) 覆盖 T2I、T2V、I2V 三种模态，支持 diffusion 和 rectified flow 两种范式。特点是提供了约 20 万条经过过滤的 RL 训练 Prompt，以及多节点 FSDP 分布式训练支持。

## TRL v1.0（Hugging Face, 2026.04）

Hugging Face 在 2026 年 4 月正式发布 TRL v1.0，将原来的研究工具升级为**生产级框架**：

```
# TRL v1.0 统一流水线
trl sft      --model Qwen/Qwen2.5-7B --dataset my_data   # 阶段一：SFT
trl reward   --model Qwen/Qwen2.5-7B --dataset pref_data # 阶段二：奖励建模
trl grpo     --model sft_checkpoint   --reward rm_model   # 阶段三：GRPO 对齐
```

TRL v1.0 的三阶段流水线（SFT → Reward Modeling → Alignment）支持 DPO、GRPO、KTO 等主流对齐算法，提供统一的 CLI 和配置系统，可在各种硬件上扩展。

---

# 系列终章：从 REINFORCE 到 SuperFlow 的完整技术脉络

通过这八篇文章，我们走过了强化学习从基础理论到前沿应用的完整旅程：

| 篇章 | 主题 | 核心贡献 | 解决的问题 |
|:---:|:---|:---|:---|
| 十六 | REINFORCE → Actor-Critic | 策略梯度基础 | 如何用梯度优化策略 |
| 十七 | TRPO → PPO | 裁剪 surrogate 目标 | 策略更新步长控制 |
| 十八 | DPO | 隐式奖励 + 偏好对 | 绕过奖励模型和 RL |
| 十九 | GRPO | 组内相对优势 | 去除 Critic，适配大模型 |
| 二十 | Flow-GRPO | 连续动作空间的 GRPO | 将 RL 引入图像生成 |
| 二十一 | DAPO | 4 项工程改进 | 大规模推理 RL 的稳定性 |
| 二十二 | 2-GRPO / f-GRPO / GIFT | GRPO 的三重理论视角 | 效率、散度选择、训练稳定性 |
| **二十三** | **SuperFlow** | **步级优势 + 动态采样** | **图像 RL 的信用分配和效率** |

**一条清晰的演进主线**：

$$
\underbrace{\text{REINFORCE}}_{\text{高方差}} \xrightarrow{\text{基线}} \underbrace{\text{Actor-Critic}}_{\text{需要 Critic}} \xrightarrow{\text{步长控制}} \underbrace{\text{PPO}}_{\text{显存翻倍}}
$$

$$
\xrightarrow{\text{去 Critic}} \underbrace{\text{GRPO}}_{\text{显存友好}} \xrightarrow{\text{工程优化}} \underbrace{\text{DAPO}}_{\text{大规模可用}} \xrightarrow{\text{连续动作}} \underbrace{\text{Flow-GRPO / SuperFlow}}_{\text{图像生成 RL}}
$$

$$
\xrightarrow[\text{离线旁支}]{\text{绕过 RL}} \underbrace{\text{DPO}}_{\text{稳定但有限}} \xrightarrow{\text{融合在线}} \underbrace{\text{GIFT}}_{\text{两全其美}}
$$

强化学习已从一个"理论优美但工程复杂"的技术，演变为大模型训练不可或缺的核心环节。每一步都在让 RL 变得更简单、更高效、更可规模化——而这个旅程，远未结束。

---

下一篇：[笔记｜生成模型（二十四）：DanceGRPO——视频生成的统一强化学习框架](/posts/25-video-grpo/)

> 参考资料：
>
> 1. [SuperFlow: Training Flow Matching Models with RL on the Fly](https://arxiv.org/abs/2512.17951)
> 2. [Flow-Factory: A Unified Framework for RL in Flow-Matching Models](https://arxiv.org/html/2602.12529v3)
> 3. [GenRL: Scalable RL for Generative Models](https://github.com/ModelTC/GenRL)
> 4. [Hugging Face TRL v1.0](https://www.marktechpost.com/2026/04/01/hugging-face-releases-trl-v1-0-a-unified-post-training-stack-for-sft-reward-modeling-dpo-and-grpo-workflows/)
