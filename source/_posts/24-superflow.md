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

> 本文为 RL 系列的图像生成篇。在第 20 篇中我们介绍了 Flow-GRPO，将 GRPO 应用于基于 Flow Matching 的图像生成。本文将介绍其后续改进 SuperFlow，以及 2026 年图像/视频生成 RL 的统一框架生态，最后回顾整个系列的完整技术脉络。
>
> 论文：[SuperFlow: Training Flow Matching Models with RL on the Fly](https://arxiv.org/abs/2512.17951)（2025.12, revised 2026.01）

# Flow-GRPO 的三个遗留问题

**延续之前"橘猫坐在蓝色沙发上"的例子。** 回忆 Flow-GRPO 的做法：对每个 Prompt（提示词）生成 $G$ 张图像，用奖励模型（裁判）打分，算出这 $G$ 张图谁比谁好（组内相对优势），然后用策略梯度更新模型。

这个流程在理论上很完美，但在大规模训练时，暴露出三个致命的“浪费”和“不合理”：

## 问题一：固定组大小导致计算浪费（吃大锅饭）

Flow-GRPO 规定，**所有** Prompt 都必须生成固定数量的图片（比如 $G = 16$）。但问题是，不同 Prompt 对模型来说，难度和“学习价值”是完全不同的：

| Prompt | 模型的真实水平 | 生成 16 张图的结果 | 学习价值（有效性） |
|:---|:---:|:---|:---:|
| "一只猫"（太简单） | 闭着眼睛都会画 | 16 张全拿满分，分不出高低（方差 $\sigma_R \approx 0$） | **纯属浪费算力** |
| "橘猫坐蓝色沙发"（中等） | 半会半不会 | 有的画得好，有的画得差（方差 $\sigma_R > 0$） | **极具学习价值** |
| "穿西装的猫弹钢琴"（太难） | 完全不会画 | 16 张全是不及格，同样分不出高低（方差 $\sigma_R \approx 0$） | **纯属浪费算力** |

**数学视角的痛点（为什么学不到东西？）**：
在 GRPO 的公式中，优势值 $\hat{A}_i$ 是通过**组内标准化**算出来的：$\hat{A}_i = \frac{r_i - \text{均值}}{\text{标准差} \sigma_R}$。

- 如果题目太简单，16 张图都拿了 99 分。均值是 99，标准差 $\sigma_R$ 接近 0。算出来的优势值 $\hat{A}_i$ 全是 0。
- 如果题目太难，16 张图都拿了 10 分。均值是 10，标准差 $\sigma_R$ 还是接近 0。算出来的优势值 $\hat{A}_i$ 也全是 0。
优势值为 0，意味着**梯度为 0，模型在这一步完全没有更新**。

**物理代价（为什么说在烧钱？）**：
大语言模型生成 16 段文字可能只需要几秒钟。但在图像生成（Flow Matching）中，生成一张 1024×1024 的高分辨率图像，需要进行 50 步繁重的 ODE（常微分方程）求解。
让模型在“太简单”或“太难”的题目上强行生成 16 张图，不仅梯度为 0（毫无学习效果），还会白白浪费极其昂贵的 GPU 算力。

## 问题二：轨迹级优势的“一刀切”（信用分配偏差）

在图像生成（Diffusion/Flow Matching）中，生成一张图是一个**多步去噪**的过程（比如从 $t=T$ 的纯噪声，一步步降噪到 $t=0$ 的清晰图像）。

Flow-GRPO 的做法是：如果最终生成的图片得了高分，就把这个高分（优势值 $\hat{A}_i$）**平均分配给去噪过程中的每一个时间步**。

**为什么这不合理？**

- **前期（$t$ 接近 $T$）**：模型在“打草稿”，决定猫在哪、沙发在哪。这时候的决策空间极大，动作的随机性（噪声）很高。
- **后期（$t$ 接近 0）**：模型在“画毛发”，大局已定，只能微调细节。这时候的决策空间极小。

**数学视角的痛点**：在强化学习中，这叫“信用分配（Credit Assignment）”问题。前期决定了生死，后期只是锦上添花。用同一个常数 $\hat{A}_i$ 去更新所有时间步的梯度，等于假设“画毛发”和“定构图”对最终高分的贡献是一样的，这会导致模型学不到重点。

## 问题三：训练后期的“躺平”现象（稳定性问题）

Flow-GRPO 只看**组内相对表现**（比同学考得好就行）。
到了训练后期，模型能力整体提升，对同一个 Prompt 生成的 16 张图都很完美，得分都很高。
**数学视角的痛点**：此时组内方差再次趋近于 0，优势信号消失。模型发现“反正随便画画都能和组里其他人差不多”，于是梯度更新停滞，甚至被随机噪声主导，导致训练崩溃。

---

# SuperFlow 的三项改进：精打细算与因材施教

为了解决上述问题，SuperFlow 提出了三项极具针对性的改进。

![SuperFlow 概览：方差感知组采样（左）与沿去噪路径的步级优势重估计（右）（摘自 *SuperFlow*, arXiv:2512.17951 图 1）](/chengYi-xun/img/superflow_overview.png)

## 改进一：方差感知的动态采样（把算力花在刀刃上）

既然“太简单”和“太难”的题目学不到东西，SuperFlow 决定：**根据模型对这道题的“不确定性”，动态决定生成多少张图。**

**数学建模与追踪**：
SuperFlow 为每个 Prompt 维护了一个**在线成绩单**（奖励追踪器 $\hat{v}(c)$），用 Beta 分布来建模模型在这个 Prompt 上的历史得分率：
$$
\hat{v}(c) = \frac{\alpha(c)}{\alpha(c) + \beta(c)}
$$
每次生成图片并拿到奖励 $r(c, y)$ 后，都会用一个遗忘因子 $\rho(c)$ 来更新这个成绩单（模型进步越快，旧成绩遗忘越快）。

**不确定性（方差）的计算**：
模型对这个 Prompt 的“不确定性”，在数学上定义为伯努利分布方差的开根号：
$$
w(c) \propto \sqrt{\hat{v}(c)(1 - \hat{v}(c)) + \epsilon}
$$
- 当 $\hat{v}(c) \approx 0.5$ 时（半会半不会），方差最大，$w(c)$ 最高。
- 当 $\hat{v}(c) \approx 1$（太简单）或 $0$（太难）时，方差极小，$w(c)$ 最低。

**动态分配算力**：
SuperFlow 将所有 Prompt 按不确定性 $w(c)$ 分为 $K$ 个等级（bin）。不确定性越高的 Prompt，分配的组大小 $m(c)$ 越大，并且在训练中被抽中的概率也越高。

**通俗例子**（假设最大生成 24 张，分 4 个等级）：

| Prompt | 历史得分率 $\hat{v}(c)$ | 不确定性 $w(c)$ | 待遇 | 组大小 $m(c)$ |
|:---|:---:|:---:|:---:|:---:|
| "一只猫" | 0.95（太简单） | 低 | 偶尔抽查 | 21 张（省算力） |
| "橘猫坐蓝色沙发" | 0.55（半会半不会） | **最高** | **重点训练** | **24 张（满配算力）** |
| "穿西装猫弹钢琴" | 0.05（太难） | 低 | 偶尔抽查 | 21 张（省算力） |

## 改进二：步级优势重估计（按劳分配）

为了解决“一刀切”的信用分配问题，SuperFlow 不再对所有时间步使用同一个优势值，而是**根据每一步的“动作噪声大小”来重新分配功劳**。

**数学推导与公式**：
在 Flow Matching（或扩散模型）的反向 SDE（随机微分方程）中，每一步的随机性由扩散系数 $\sigma_t$ 决定。$\sigma_t$ 越大，模型在这一步的“自由发挥空间”越大。
因此，SuperFlow 将第 $t$ 步的实际优势值 $\hat{A}_t$ 定义为：
$$
\hat{A}_t = \eta \cdot \sigma_t \cdot A_\tau
$$
其中：
- $A_\tau$ 是整张图的最终优势值（总奖金）。
- $\sigma_t$ 是第 $t$ 步的 SDE 噪声标准差（这一步的贡献比例）。
- $\eta$ 是缩放系数。

**物理直觉与通俗理解**：
- **前期（$t$ 接近 $T$，打草稿）**：$\sigma_t$ 很大，模型在做极其重要的构图决定。如果最终图画得好，这几步居功至伟，分到的优势权重 $\hat{A}_t$ 最大。
- **后期（$t$ 接近 0，修细节）**：$\sigma_t$ 很小，大局已定。这几步只是锦上添花，分到的优势权重 $\hat{A}_t$ 极小。

这种“按劳分配”的梯度更新，让模型能极其精准地学到**在什么时间点该做什么正确的决定**。

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

## TRL v1.0（Hugging Face, 2026.03）

Hugging Face 在 2026 年 3 月 31 日发布 TRL [v1.0.0](https://github.com/huggingface/trl/releases/tag/v1.0.0)（官方博文与社区多简称 **TRL v1.0**），将原来的研究工具升级为**生产级框架**：

```
# TRL v1.0 统一流水线
# 阶段一：SFT
trl sft --model Qwen/Qwen2.5-7B --dataset my_data
# 阶段二：奖励建模
trl reward --model Qwen/Qwen2.5-7B --dataset pref_data
# 阶段三：GRPO 对齐
trl grpo --model sft_checkpoint --reward rm_model
```

TRL v1.0 的三阶段流水线（SFT → Reward Modeling → Alignment）支持 DPO、GRPO、KTO 等主流对齐算法，提供统一的 CLI 和配置系统，可在各种硬件上扩展。

---

# 阶段性总结：从 REINFORCE 到 SuperFlow 的技术脉络

通过这几篇文章，我们走过了强化学习在生成模型中从基础理论到前沿应用的旅程：

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

> 参考资料：
>
> 1. [SuperFlow: Training Flow Matching Models with RL on the Fly](https://arxiv.org/abs/2512.17951)
> 2. [Flow-Factory: A Unified Framework for RL in Flow-Matching Models](https://arxiv.org/html/2602.12529v3)
> 3. [GenRL: Scalable RL for Generative Models](https://github.com/ModelTC/GenRL)
> 4. [Hugging Face TRL v1.0](https://www.marktechpost.com/2026/04/01/hugging-face-releases-trl-v1-0-a-unified-post-training-stack-for-sft-reward-modeling-dpo-and-grpo-workflows/)

> 下一篇：[笔记｜生成模型（二十四）：DanceGRPO——视频生成的统一强化学习框架](/chengYi-xun/posts/25-video-grpo/)
