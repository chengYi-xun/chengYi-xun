# chengYi-xun 技术博客

基于 [Hexo](https://hexo.io/) + [Butterfly](https://github.com/jerryc127/hexo-theme-butterfly) 主题的个人技术博客，系统记录生成式 AI、多模态融合与世界模型领域的学习笔记。

**在线地址**：[https://chengyi-xun.github.io/chengYi-xun/](https://chengyi-xun.github.io/chengYi-xun/)

---

## 内容目录

### 生成式人工智能（24 篇）

从概率论基础出发，完整覆盖生成模型的理论演进与代码实现：

| 模块 | 内容 | 篇数 |
|------|------|------|
| 基础理论 | 概率论、GAN、VAE | 4 |
| 扩散模型 | DDPM → DDIM → Score-Based → SDE 统一框架 | 4 |
| 条件生成 | Classifier Guidance / Classifier-Free Guidance | 2 |
| 架构演进 | UiT/DiT、Normalizing Flow、Flow Matching、SD3 (MMDiT)、Flux | 5 |
| 强化学习与对齐 | REINFORCE → PPO → DPO → GRPO → DAPO → Flow-GRPO → DanceGRPO | 9 |

### 多模态融合（6 篇）

从早期特征拼接到 2026 前沿，系统讲解多模态学习：

- 三级融合框架（早期 / 中期 / 晚期）的数学推导与代码实现
- CLIP、BLIP-2 (Q-Former)、LLaVA、Flamingo、Chameleon
- MMDiT、InternVL、Qwen-VL、Mamba 多模态

### 世界模型（7 篇）

五大技术路线全覆盖，从认知科学起源讲到 2026 前沿：

- **Model-based RL**：RSSM、DreamerV1/V2/V3、IRIS、TD-MPC2
- **JEPA**：I-JEPA、V-JEPA、V-JEPA 2
- **视频生成世界模型**：Sora、Genie、Cosmos、UniSim
- **物理化世界模型**：PhysDreamer、PhysGen、NewtonGen、NewtonRewards
- **自动驾驶世界模型**：GAIA-1、DriveDreamer、Vista、OccWorld

### 杂谈

- VGGT 3D 视觉统一 Transformer
- Vision Transformers Need Registers

---

## 环境搭建与本地运行

### 1. 安装 Node.js

推荐使用 [nvm](https://github.com/nvm-sh/nvm) 管理 Node 版本：

```bash
# 安装 nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash

# 重启终端后安装 Node.js（建议 v18+）
nvm install 18
nvm use 18
```

或直接从 [Node.js 官网](https://nodejs.org/) 下载安装。

### 2. 安装 Pandoc

本项目使用 `hexo-renderer-pandoc` 渲染 Markdown，需要系统安装 Pandoc：

```bash
# Ubuntu / Debian
sudo apt install pandoc

# macOS
brew install pandoc
```

### 3. 安装项目依赖

```bash
cd chengYi-xun
npm install
```

### 4. 本地预览

```bash
# 清理缓存 + 启动本地服务器（默认 http://localhost:4000）
npx hexo clean && npx hexo server
```

### 5. 生成静态文件

```bash
npx hexo clean && npx hexo generate
```

生成的静态文件在 `public/` 目录下，可直接部署到 GitHub Pages 等静态托管服务。

### 6. 部署到 GitHub Pages

```bash
npx hexo deploy
```

部署配置在 `_config.yml` 中的 `deploy` 字段。

## 目录结构

```
source/_posts/          # Markdown 文章
source/img/             # 图片资源
information/            # 参考论文与代码仓库
themes/butterfly/       # Butterfly 主题
_config.yml             # Hexo 全局配置
_config.butterfly.yml   # 主题配置
```

## 联系方式

- GitHub: [chengYi-xun](https://github.com/chengYi-xun)
- 邮箱: ldq4399@163.com
