---
title: 笔记｜MemoryBank：用艾宾浩斯遗忘曲线赋予 LLM 长期记忆
date: 2026-04-05 18:00:00
cover: false
mathjax: true
categories:
 - Notes
tags:
 - Deep learning
 - LLM
 - Memory mechanism
 - AI Companion
---

> 论文：[MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://arxiv.org/abs/2305.10250)（2023.05, Sun Yat-Sen University）
>
> 代码：[github.com/zhongwanjun/MemoryBank-SiliconFriend](https://github.com/zhongwanjun/MemoryBank-SiliconFriend)

# 为什么 LLM 需要长期记忆？

**先从一个场景理解问题。** 假设你和一个 AI 助手连续聊天 10 天：

| 第 1 天 | 你：我最近分手了，心情很低落 |
|:---|:---|
| **第 3 天** | 你：我开始学 Python 了，推荐本书吧 |
| **第 5 天** | 你：你之前推荐的那本书不错，快看完了 |
| **第 8 天** | 你：我在考虑学数据分析还是机器学习方向 |
| **第 10 天** | 你：你觉得以我的情况，应该走哪个方向？ |

一个**没有长期记忆**的 ChatGPT，在第 10 天面对这个问题时：
- 不记得你第 1 天的分手经历（无法给出有温度的回复）
- 不记得你第 3 天才开始学 Python（无法评估你的技术水平）
- 不知道你读了哪本书、进度如何（无法给出针对性建议）

它只能给出一个**通用的、模板化的**回答："这取决于你的兴趣和背景……"

而一个**有长期记忆**的 AI 可以说：

> "还记得你第 3 天才开始学 Python，到现在已经看完了《Automate the Boring Stuff with Python》——学习速度很快！考虑到你刚经历分手，或许数据分析方向上手更快，做出可视化作品的成就感能帮你调节心情。等 Python 基础更扎实了再学机器学习也不迟。"

**这就是 MemoryBank 要解决的问题**：让 LLM 在长期交互中记住过去、理解用户、像人一样有选择地遗忘。

---

# MemoryBank 的三大支柱

MemoryBank 的架构围绕三个核心组件展开：

$$
\text{MemoryBank} = \underbrace{\text{Memory Storage}}_{\text{记什么}} + \underbrace{\text{Memory Retrieval}}_{\text{怎么找}} + \underbrace{\text{Memory Updating}}_{\text{怎么忘}}
$$

## 支柱一：Memory Storage（记忆仓库）

记忆仓库存储三个层次的信息，构成一个**层次化记忆结构**：

| 层次 | 内容 | 类比人类记忆 |
|:---:|:---|:---|
| **详细对话记录** | 每天的完整多轮对话（带时间戳） | 日记本 |
| **事件摘要** | 每天对话的关键事件 → 合并为全局摘要 | 回忆中的"大事记" |
| **用户画像** | 从对话中推断的性格特征 → 持续更新 | 对一个人的"印象" |

### 记忆仓库的数据结构

在[开源实现](https://github.com/zhongwanjun/MemoryBank-SiliconFriend)中，记忆仓库以 JSON 文件持久化存储，结构如下：

```json
{
  "Emily": {
    "name": "Emily",
    "history": {
      "2023-04-27": [
        {"query": "Hello, my name is Emily.", "response": "Hello, Emily. I'm your AI companion."},
        {"query": "I want to learn painting.", "response": "That's great! ..."}
      ],
      "2023-04-28": [...]
    },
    "summary": {
      "2023-04-27": {"content": "Emily introduced herself and discussed..."},
      "2023-04-28": {"content": "..."}
    },
    "personality": {
      "2023-04-27": "Emily is open-minded and curious...",
      "2023-04-28": "..."
    },
    "overall_history": "Emily discussed topics including painting, travel...",
    "overall_personality": "Emily is an open-minded, curious girl with diverse interests..."
  }
}
```

**摘要和画像的生成**由 `summarize_memory.py` 自动完成。生成链条如下：

```python
# 每日事件摘要
prompt = "请总结以下的对话内容，尽可能精炼，提取对话的主题和关键信息。"
for dialog in content:
    prompt += f"\n{user_name}：{dialog['query'].strip()}"
    prompt += f"\nAI：{dialog['response'].strip()}"
daily_summary = llm_client.generate_text_simple(prompt)

# 全局事件摘要（将所有每日摘要再压缩一次）
prompt = "请高度概括以下的事件，概括并保留核心关键信息。"
for date, summary in daily_summaries:
    prompt += f"\n时间{date}发生的事件为{summary}"
overall_summary = llm_client.generate_text_simple(prompt)

# 用户画像
prompt = f"请根据以下的对话推测总结{user_name}的性格特点和心情，并制定回复策略。"
daily_personality = llm_client.generate_text_simple(prompt)

# 全局画像
prompt = "以下是用户在多段对话中展现出来的人格特质和心情..."
overall_personality = llm_client.generate_text_simple(prompt)
```

**用例子说明**：经过 10 天对话后，MemoryBank 可能生成如下用户画像——

> "Linda 是一个内向但有决心的女孩，重视个人成长，喜欢探索新文化和新爱好，乐于接受建议。"

当 Linda 问 "周末推荐点活动吧？"时，AI 可以基于画像给出**个性化建议**——推荐烹饪课或博物馆参观，而不是泛泛地推荐"去公园散步"。

## 支柱二：Memory Retrieval（记忆检索）

当用户发送新消息时，如何从海量历史对话中找到**相关记忆**？MemoryBank 使用的是 **Dense Passage Retrieval**（稠密段落检索）方法：

1. **离线编码**：每段对话和事件摘要都被视为一个记忆片段 $m$，由编码器 $E(\cdot)$ 编码为向量 $h_m = E(m)$
2. **索引**：所有 $h_m$ 存入 FAISS 向量索引中
3. **在线检索**：当前对话上下文 $c$ 编码为 $h_c = E(c)$，在索引中搜索最相似的记忆

$$
m^* = \arg\max_{m \in M} \text{sim}(h_c, h_m)
$$

### 检索流程的代码实现

开源实现提供了两种检索后端，分别用于 ChatGLM/BELLE（`local_doc_qa.py`）和 ChatGPT（`build_memory_index.py` + LlamaIndex）：

```python
# local_doc_qa.py — 基于 LangChain + FAISS 的检索
class LocalMemoryRetrieval:
    def init_cfg(self, embedding_model, embedding_device, top_k, language):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_dict[embedding_model],
            model_kwargs={'device': embedding_device}
        )
        self.top_k = top_k

    def init_memory_vector_store(self, filepath, vs_path, user_name, cur_date):
        docs = load_memory_file(filepath, user_name, language)
        vector_store = FAISS.from_documents(docs, self.embeddings)
        vector_store.save_local(vs_path)

    def search_memory(self, query, vector_store):
        related_docs_with_score = vector_store.similarity_search_with_score(
            query, k=self.top_k  # 默认 top_k = 6
        )
        # 按日期分组，返回检索到的记忆片段和日期
        related_docs = sorted(related_docs, key=lambda x: x.metadata["source"])
        return date_docs, dates
```

英文用 MiniLM，中文用 Text2vec 作编码器。检索返回 top-6 最相关的记忆片段。

### 检索后如何强化记忆

检索不只是"读取"——被检索到的记忆会自动**强化**（`forget_memory.py`）：

```python
def update_memory_when_searched(self, recalled_memos, user, cur_date):
    for recalled in recalled_memos:
        recalled_id = recalled.metadata['memory_id']
        recalled_date = recalled_id.split('_')[1]
        for i, memory in enumerate(self.memory_bank[user]['history'][recalled_date]):
            if memory['memory_id'] == recalled_id:
                # 记忆强度 +1，遗忘速率变慢
                self.memory_bank[user]['history'][recalled_date][i]['memory_strength'] += 1
                # 重置上次回忆日期
                self.memory_bank[user]['history'][recalled_date][i]['last_recall_date'] = cur_date
                break
```

每次检索完成后，立即保存更新后的记忆状态到 JSON 文件，确保持久化。

## 支柱三：Memory Updating（记忆更新——艾宾浩斯遗忘曲线）

这是 MemoryBank 最独特的设计。人类不会记住所有事——重要的事记得牢，不重要的事逐渐遗忘。MemoryBank 用**艾宾浩斯遗忘曲线**来模拟这个过程。

### 艾宾浩斯遗忘曲线

1885 年，德国心理学家赫尔曼·艾宾浩斯通过实验发现，记忆的保持量随时间呈指数衰减：

$$
R = e^{-t/S}
$$

其中：
- $R \in [0, 1]$：**记忆保持率**（retention），即还能记住多少
- $t$：距离上次学习/回忆经过的**时间**
- $S$：**记忆强度**（strength），取决于学习深度和复习次数

**用数字理解**：假设初始记忆强度 $S = 1$（天）：

| 经过时间 $t$ | 保持率 $R = e^{-t/S}$ | 含义 |
|:---:|:---:|:---|
| 0 天 | $e^0 = 1.00$（100%） | 刚学完，记得很清楚 |
| 1 天 | $e^{-1} \approx 0.37$（37%） | 一天后只记得 37% |
| 2 天 | $e^{-2} \approx 0.14$（14%） | 两天后只记得 14% |
| 5 天 | $e^{-5} \approx 0.007$（0.7%） | 五天后几乎遗忘 |

### MemoryBank 如何使用遗忘曲线

MemoryBank 将 $S$ 建模为**离散整数**：

1. **初始化**：每个记忆片段首次出现时，$S = 1$
2. **衰减**：随着时间 $t$ 增长，$R = e^{-t/S}$ 下降
3. **强化**：当记忆在对话中被**回忆**（即被检索到并使用），则 $S \leftarrow S + 1$，同时 $t$ 重置为 $0$

用表格演示一个记忆片段 "用户推荐了一本书" 的命运：

| 事件 | $S$ | $t$ | $R = e^{-t/S}$ | 状态 |
|:---|:---:|:---:|:---:|:---|
| 第 1 天：首次提到 | 1 | 0 | 1.00 | 新鲜记忆 |
| 第 3 天：未被回忆 | 1 | 2 | 0.14 | 快要遗忘 |
| 第 3 天：被检索并引用 | **2** | **0** | **1.00** | 强化！重新鲜活 |
| 第 6 天：未被回忆 | 2 | 3 | 0.22 | 衰减变慢了（S=2） |
| 第 10 天：未被回忆 | 2 | 7 | 0.03 | 接近遗忘 |

**关键洞察**：被反复回忆的记忆越来越难遗忘（$S$ 越大，衰减越慢），而从未被提及的记忆会逐渐消失。这与人类记忆的"间隔效应"（Spacing Effect）一致——反复复习会降低遗忘速率。

### 遗忘机制的代码实现

`forget_memory.py` 中的遗忘曲线函数：

```python
def forgetting_curve(t, S):
    return math.exp(-t / 5*S)
```

> **注意**：根据 Python 运算优先级，`-t / 5*S` 等价于 `(-t / 5) * S = -tS/5`，即实际公式为 $R = e^{-tS/5}$。这与论文中的 $R = e^{-t/S}$ 有出入——论文中 $S$ 增大时遗忘变慢，而代码中 $S$ 增大反而遗忘更快。推测作者意图是 `math.exp(-t / (5*S))`，即 $R = e^{-t/(5S)}$，多了一个系数 5 来减缓整体遗忘速率。这是一个值得注意的代码细节。

遗忘的触发在系统每次加载记忆时执行——通过**概率性遗忘**决定哪些记忆被保留：

```python
def initial_load_forget_and_save(self, name, now_date):
    docs = []
    for user_name, user_memory in memories.items():
        for date, content in user_memory['history'].items():
            forget_ids = []
            for i, dialog in enumerate(content):
                memory_strength = dialog.get('memory_strength', 1)
                last_recall_date = dialog.get('last_recall_date', date)
                
                # 计算距上次回忆的天数
                days_diff = self._get_date_difference(last_recall_date, now_date)
                # 计算记忆保持率
                retention_probability = forgetting_curve(days_diff, memory_strength)
                
                # 掷骰子：随机数 > 保持率 → 遗忘
                if random.random() > retention_probability:
                    forget_ids.append(i)
                else:
                    docs.append(Document(page_content=..., metadata=...))
            
            # 从记忆仓库中移除被遗忘的记忆
            for idd in sorted(forget_ids, reverse=True):
                self.memory_bank[user_name]['history'][date].pop(idd)
    
    self.write_memories(self.filepath)
    return docs
```

**概率性遗忘**是一个巧妙的设计：不是 $R$ 低于阈值就立刻删除，而是以 $R$ 为概率保留。这意味着即使 $R$ 很低（如 0.05），仍有 5% 的概率被保留——模拟了人类偶尔突然想起某个久远记忆的现象。

---

# SiliconFriend：基于 MemoryBank 的 AI 陪伴助手

MemoryBank 是一个**通用机制**，可以嵌入任何 LLM。论文通过 **SiliconFriend** 这个 AI 陪伴聊天机器人来展示其效果。

## 两阶段构建

**第一阶段：心理对话微调**（仅开源模型）

使用 3.8 万条心理咨询对话数据，通过 LoRA 对开源 LLM（ChatGLM、BELLE）进行微调：

$$
y = Wx + BAx, \quad B \in \mathbb{R}^{d \times r},\, A \in \mathbb{R}^{r \times k},\, r \ll \min(d, k)
$$

LoRA rank $r = 16$，在 A100 GPU 上训练 3 个 epoch。微调后的模型在情感对话中表现出更强的共情能力。

**第二阶段：集成 MemoryBank**

将 MemoryBank 的记忆存储、检索、更新机制集成到聊天流程中。

### Memory-Augmented Prompt 的构建

这是整个系统的核心粘合剂——如何把检索到的记忆注入 LLM 的 Prompt 中。`prompt_utils.py` 中定义了 Meta Prompt 模板：

```python
# ChatGLM/BELLE 的 Meta Prompt（来自 prompt_utils.py）
meta_prompt = """
现在你将扮演用户{user_name}的专属AI伴侣，你的名字是{boot_actual_name}。
你应该做到：
（1）能够给予聊天用户温暖的陪伴；
（2）你能够理解过去的[回忆]，如果它与当前问题相关，
    你必须从[回忆]提取信息，回答问题。
（3）你还是一名优秀的心理咨询师，当用户向你倾诉困难、寻求帮助时，
    你可以给予他温暖、有帮助的回答。

用户{user_name}的性格以及AI伴侣的回复策略为：{personality}

根据当前用户的问题，你开始回忆你们二人过去的对话，
你想起与问题最相关的[回忆]是：
"{related_memory_content}
记忆中这段[回忆]的日期为{memo_dates}。"

{history_text}
"""
```

**Prompt 组装的完整流程**：

```python
def build_prompt_with_search_memory(history, text, user_memory,
                                     user_name, user_memory_index, ...):
    # 1. 用当前消息检索相关记忆
    related_memos, memo_dates = local_memory_qa.search_memory(
        text, user_memory_index
    )
    related_memory_content = '\n'.join(related_memos)
    
    # 2. 获取全局用户画像
    personality = user_memory.get('overall_personality', '')
    
    # 3. 拼接对话历史
    history_text = ''
    for dialog in history:
        history_text += f"\n [|用户|]: {dialog['query']}"
        history_text += f"\n [|AI伴侣|]: {dialog['response']}"
    history_text += f"\n [|用户|]: {text} \n [|AI伴侣|]: "
    
    # 4. 填充 Meta Prompt 模板
    prompt = meta_prompt.format(
        user_name=user_name,
        related_memory_content=related_memory_content,
        personality=personality,
        boot_actual_name=boot_actual_name,
        history_text=history_text,
        memo_dates=memo_dates
    )
    return prompt
```

**新用户处理**：如果用户是第一次使用（没有历史记忆），系统会切换到一个简化版 Prompt，不包含记忆和画像：

```python
if history_summary and related_memory_content and personality:
    prompt = meta_prompt.format(...)     # 包含记忆的完整 Prompt
else:
    prompt = new_user_meta_prompt.format(...)  # 只有基本陪伴能力
```

## 完整对话生命周期

`cli_demo.py` 展示了一次完整对话的全生命周期：

```
1. 用户登录
   └─ enter_name() → 加载记忆 JSON → 构建 FAISS 索引
       └─ 如果启用遗忘机制，先执行概率性遗忘

2. 用户发送消息
   ├─ build_prompt_with_search_memory()
   │   ├─ FAISS 语义检索 Top-K 相关记忆
   │   ├─ 强化被检索到的记忆（S += 1, t 重置）
   │   └─ 组装 Memory-Augmented Prompt
   ├─ 送入 LLM 生成回复
   └─ save_local_memory() → 将新对话存入 JSON

3. 会话结束（或定期触发）
   └─ summarize_memory() → 生成/更新每日摘要和用户画像
```

## 三种 LLM 后端

SiliconFriend 支持三种后端，展示了 MemoryBank 的通用性：

| 后端 | 类型 | 记忆检索方案 | 特点 |
|:---:|:---:|:---|:---|
| ChatGPT | 闭源 | LlamaIndex + GPTSimpleVectorIndex | 综合能力最强 |
| ChatGLM (6.2B) | 开源 | LangChain + FAISS + Text2vec | 中文优化 |
| BELLE (7B) | 开源 | LangChain + FAISS + MiniLM/Text2vec | 基于 LLaMA |

ChatGPT 使用 LlamaIndex 构建索引（借助 GPT-3.5 的嵌入能力），开源模型则用 HuggingFace 本地嵌入模型 + FAISS。

---

# 实验评估

## 定性分析

论文通过真实用户对话展示了三个关键能力：

**1. 共情心理陪伴**

当用户表达"我最近分手了"时，SiliconFriend 与原始 ChatGLM 的对比：

| | SiliconFriend | 原始 ChatGLM |
|:---|:---|:---|
| 第一轮 | 提供情感支持 + 建设性建议 | 标准化安慰 |
| 第二轮 | 捕捉"分手其实是解脱"的情绪转变 | 仍在模板化地表达同情 |
| 第三轮 | 鼓励展望未来 + 具体交友建议 | 给出通用列表式建议 |

**2. 记忆回忆**

用户 Linda 在第 1 天和 SiliconFriend 讨论了 Python 学习和快速排序。几天后问：

- "你之前推荐了什么书？" → 正确回忆：《Automate the Boring Stuff with Python》
- "我之前让你写什么代码？" → 正确回忆：快速排序
- "我们一起写过堆排序吗？" → 正确否认：没有

**3. 个性化交互**

面对不同用户画像，SiliconFriend 给出了差异化的周末活动建议：

| 用户 | 画像 | 推荐 |
|:---|:---|:---|
| Linda | 内向、重视成长、喜欢探索文化 | 烹饪课、博物馆 |
| Emily | 开放、好奇、有时自我怀疑 | 户外运动、学乐器 |

## 定量分析

使用 ChatGPT 扮演 15 个不同性格的虚拟用户，生成 10 天的对话记录，再设计 194 个记忆探测问题（英文 97 + 中文 97）进行评估：

| 指标 | SiliconFriend ChatGPT | SiliconFriend BELLE | SiliconFriend ChatGLM |
|:---|:---:|:---:|:---:|
| **记忆检索准确率** | 0.763 / 0.711 | 0.814 / 0.856 | 0.809 / 0.840 |
| **回复正确性** | **0.716** / **0.655** | 0.479 / 0.603 | 0.438 / 0.418 |
| **上下文连贯性** | **0.912** / **0.675** | 0.582 / 0.562 | 0.680 / 0.428 |
| **模型排名得分** | **0.818** / **0.758** | 0.517 / 0.565 | 0.498 / 0.510 |

（格式：英文 / 中文）

**关键发现**：
1. **MemoryBank 对所有 LLM 都有效**：三个后端的记忆检索准确率都超过了 70%
2. **基座模型能力决定上限**：ChatGPT 在回复正确性和连贯性上远超开源模型，说明 MemoryBank 是"锦上添花"而非"雪中送炭"——基座能力越强，记忆增强的效果越好
3. **语言差异**：ChatGLM 和 ChatGPT 在英文上表现更好，BELLE 在中文上更优

---

# 彩蛋：源码中一个有趣的 Bug

值得一提的是，`forget_memory.py` 中的遗忘曲线实现与论文描述存在不一致：

```python
def forgetting_curve(t, S):
    return math.exp(-t / 5*S)
```

根据 Python 运算优先级，`-t / 5*S` 等价于 `((-t) / 5) * S`，即：

$$
R_{\text{code}} = e^{-tS/5}
$$

而论文中的公式是 $R = e^{-t/S}$。两者的行为**截然相反**：

| | 论文：$R = e^{-t/S}$ | 代码：$R = e^{-tS/5}$ |
|:---|:---|:---|
| $S$ 增大时 | 遗忘变**慢** ✓ | 遗忘变**快** ✗ |
| 效果 | 被多次回忆的记忆更持久 | 被多次回忆的记忆反而更容易遗忘 |

推测作者意图应该是 `math.exp(-t / (5*S))`（少了一对括号），对应 $R = e^{-t/(5S)}$，其中系数 5 用于让遗忘速率不至于太快（避免 1 天后只剩 37%）。

这种论文与代码之间微妙但关键的差异，在开源项目中其实并不少见——所以复现论文时，永远不要跳过读源码这一步。

---

# 技术局限与思考

**1. 遗忘曲线模型过于简化**

真实的人类记忆远比 $R = e^{-t/S}$ 复杂——情感关联、上下文重要性、个体差异都会影响遗忘速率。论文使用离散整数 $S$ 和简单的 +1 强化规则，只是一个粗略近似。

**2. 检索是瓶颈**

如果检索失败（未能找到相关记忆），即使记忆仓库中存储了正确信息，LLM 也无法利用。从实验中 ChatGPT 的检索准确率（76.3%）可以看出，约四分之一的情况下记忆检索失败。

**3. 可扩展性挑战**

随着对话天数增长到数月甚至数年，记忆仓库和 FAISS 索引的规模会持续膨胀。虽然遗忘机制可以清除部分记忆，但层次化摘要的质量可能随着信息量增大而下降。

**4. 隐私问题**

长期存储用户的详细对话和性格画像引发了严肃的隐私问题——这些数据如何保护？用户能否要求删除？在实际部署中需要仔细权衡。

---

# 总结

MemoryBank 的核心贡献在于将心理学中的**艾宾浩斯遗忘曲线**引入 LLM 的记忆管理，构建了一个"记忆—检索—遗忘"的完整闭环：

$$
\text{对话} \xrightarrow{\text{存储}} \text{记忆仓库} \xrightarrow{\text{编码+索引}} \text{FAISS} \xrightarrow{\text{检索}} \text{增强 Prompt} \xrightarrow{\text{生成回复}}
$$
$$
\text{记忆仓库} \xrightarrow{R = e^{-t/S}} \text{遗忘/强化} \xrightarrow{\text{更新}} \text{记忆仓库}
$$

它不需要修改 LLM 的参数，作为一个**外挂模块**即可为任何 LLM 赋予长期记忆能力。这种"即插即用"的设计理念，在 2023 年 RAG（Retrieval-Augmented Generation）快速发展的背景下，展示了记忆增强范式的潜力。

从工程角度看，MemoryBank 的技术栈非常清晰：

| 组件 | 技术选型 |
|:---|:---|
| 记忆存储 | JSON 文件 |
| 记忆编码 | HuggingFace Embeddings（MiniLM / Text2vec） |
| 向量索引 | FAISS |
| 检索框架 | LangChain（开源模型）/ LlamaIndex（ChatGPT） |
| 摘要/画像生成 | GPT-3.5-turbo |
| 微调方案 | LoRA (rank=16) |
| 交互界面 | Gradio Web UI / CLI |

这些都是 2023 年 LLM 生态中的主流组件，降低了复现和扩展的门槛。

> 参考资料：
>
> 1. [MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://arxiv.org/abs/2305.10250)
> 2. [Ebbinghaus, H. Memory: A Contribution to Experimental Psychology, 1885/1964](https://en.wikipedia.org/wiki/Forgetting_curve)
> 3. [Dense Passage Retrieval (DPR)](https://arxiv.org/abs/2004.04906)
> 4. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
