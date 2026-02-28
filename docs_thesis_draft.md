# 基于BERTopic的协同过滤诗歌推荐系统（毕业论文草稿）

> 说明：Markdown 本身无法强制“目录字体为宋体五号、不得斜体加粗”。你后续可在 Word 排版阶段统一设置样式。本稿已按你给出的章节结构撰写，便于直接迁移。

---

## 摘  要

在数字阅读平台快速发展的背景下，诗歌内容面临“作品量大、检索困难、用户兴趣差异显著”等问题，传统按作者、朝代或关键词的推荐方式难以满足用户的个性化阅读需求。针对上述问题，本文设计并实现了一个基于BERTopic的协同过滤诗歌推荐系统。系统以古典诗词文本为核心数据对象，融合文本主题建模与用户行为建模：首先，利用BERTopic对诗歌语义主题进行抽取，得到更符合中文短文本语义分布的主题表示；其次，结合用户评分、点赞、评论等行为数据构建用户兴趣画像，在内容相似、物品协同过滤、用户协同过滤与主题对齐等多路策略之间进行加权融合；最后，通过前后端分离架构实现推荐、检索、评论、偏好引导和全局统计等功能模块，并通过实验评估系统有效性。

测试结果表明，所实现的混合推荐方案在召回相关性指标上优于单一推荐策略，能够在一定程度上缓解冷启动与推荐同质化问题；同时，系统在业务流程完整性、接口响应稳定性和用户交互体验方面达到了预期目标。本文工作对诗歌类垂直内容平台的智能推荐具有一定的工程应用价值，并可为中文文学内容的个性化分发提供参考。

关键词：诗歌推荐；BERTopic；协同过滤；混合推荐；主题建模

---

## ABSTRACT

With the rapid development of digital reading platforms, poetry content is facing challenges such as large-scale corpus, difficult retrieval, and highly diverse user interests. Traditional recommendation methods based on author, dynasty, or keyword matching cannot effectively satisfy personalized reading demands. To address these issues, this thesis designs and implements a collaborative filtering poetry recommendation system based on BERTopic.

The proposed system combines semantic topic modeling with user behavior modeling. First, BERTopic is employed to extract latent semantic topics from Chinese poetry texts, providing topic-aware representations suitable for short literary content. Second, user profiles are constructed from ratings, likes, and comments, and multiple recommendation channels—including content similarity, item-based collaborative filtering, user-based collaborative filtering, and topic alignment—are fused through weighted ranking. Third, a front-end/back-end separated architecture is adopted to implement major modules such as recommendation, retrieval, review interaction, preference guidance, and global statistics.

Experimental results indicate that the hybrid strategy outperforms single-strategy baselines on relevance-oriented metrics and alleviates cold-start and recommendation homogeneity to some extent. In addition, the system achieves expected performance in business process integrity, API stability, and interaction usability. This work provides practical value for vertical poetry platforms and offers a reference for personalized recommendation in Chinese literary content distribution.

Key Words: Poetry Recommendation, BERTopic, Collaborative Filtering, Hybrid Recommendation, Topic Modeling

---

## 第一章 绪论

### 1.1 诗歌推荐系统的研究背景和意义

随着移动互联网和数字文化产业的发展，海量古典诗词资源以数据库、知识图谱和数字藏品等形式被持续整理并上线。用户在面对大量诗歌时，常出现“知道想读的情绪，不知道具体作品”的选择困难。传统检索方式对输入要求较高，难以精准表达“意境偏好”“情感倾向”等隐性需求，导致用户发现优质内容的效率较低。

推荐系统能够通过学习用户历史行为实现“千人千面”的内容分发，在电商、短视频、音乐和新闻场景中已证明其有效性。将推荐系统引入诗歌阅读场景，不仅可提升用户阅读效率与平台活跃度，也有助于传统文化的数字传播与再利用。相比一般文本推荐，诗歌具有篇幅短、语义凝练、意象密集的特点，适合引入主题建模技术进行深层语义抽取。因此，研究“基于BERTopic的协同过滤诗歌推荐系统”具有明确的理论价值和应用意义。

### 1.2 国内外诗歌推荐与主题建模研究概况

国外在推荐系统领域起步较早，形成了基于协同过滤、矩阵分解、深度学习和序列建模的多类方法体系；在主题建模方面，LDA及其变体长期被用于文本聚类与主题发现。近年来，随着预训练语言模型发展，BERTopic等“Transformer嵌入+聚类+主题提取”的框架在短文本主题建模中表现出更好的语义一致性。

国内研究在中文文本推荐、古诗词知识组织、教育场景阅读推荐等方向进展迅速。部分工作将关键词匹配、知识图谱、情感分析与协同过滤结合，用于诗词学习系统或文化类内容平台。但总体而言，面向诗歌场景的“主题语义+用户协同”融合方案仍有提升空间，尤其在冷启动、推荐解释性和系统工程化方面仍存在不足。

### 1.3 本文的主要工作和章节安排

本文围绕“诗歌语义理解与个性化推荐融合”开展研究，主要工作如下：

1. 设计并实现基于BERTopic的诗歌主题建模模块，用于构建诗歌主题表示；
2. 设计融合内容相似、用户协同、物品协同和主题对齐的混合推荐策略；
3. 基于前后端分离架构实现可运行系统，包括推荐、评论、偏好引导和统计模块；
4. 设计系统测试方案，从功能、性能与推荐效果三个维度验证系统有效性。

全文结构安排如下：第二章介绍相关理论与技术基础；第三章进行需求分析与系统设计；第四章说明系统实现过程；第五章给出测试方法与结果；第六章总结全文并提出未来改进方向。

---

## 第二章 诗歌推荐系统的相关理论和有关实现技术

### 2.1 推荐系统的相关概念

推荐系统是通过分析用户历史行为、内容特征与上下文信息，为用户提供个性化内容建议的技术系统。典型流程包括数据采集、特征建模、候选召回、排序重排与结果反馈。常见输入信号包括显式反馈（评分）与隐式反馈（点击、停留、收藏、点赞、评论等）。

在诗歌场景中，推荐目标不仅是“点击概率最大化”，还应考虑文化内容的多样性和探索性，避免推荐结果过于集中于少数热门作品。故系统需在相关性与多样性之间取得平衡。

### 2.2 BERTopic技术

BERTopic是一种结合预训练句向量、降维聚类和关键词提取的主题建模方法。其核心思想是先将文本映射到语义向量空间，再通过聚类获得主题簇，最后用类TF-IDF等方法提取各主题关键词表示。与传统LDA相比，BERTopic对短文本语义更敏感，更适合诗歌这类“字少意深”的文本。

本文中，BERTopic用于生成诗歌主题标签和主题中心语义表示，为后续推荐融合提供主题层面的可解释特征。

### 2.3 推荐系统相关算法

1. 内容推荐（Content-Based）：根据用户已喜欢诗歌与候选诗歌的文本相似度进行推荐；
2. 物品协同过滤（Item-CF）：根据“用户共同评分”关系度量诗歌相似性，推荐与已读作品相近的诗歌；
3. 用户协同过滤（User-CF）：寻找兴趣相似用户并迁移其偏好；
4. 混合推荐（Hybrid）：将多路策略结果进行归一化融合与重排，提高鲁棒性；
5. 多样性重排（如MMR）：在保证相关性的同时降低候选间冗余。

### 2.4 本章小结

本章介绍了推荐系统基础概念、BERTopic主题建模原理与混合推荐相关算法，为后续系统分析、设计与实现提供理论基础。

---

## 第三章 诗歌推荐系统需求分析与设计

### 3.1 诗歌推荐系统需求分析

#### 3.1.1 对用户的需求分析

用户需求主要包括：

- 能够快速获取符合个人兴趣的诗歌推荐；
- 支持按关键词、作者、题目进行检索；
- 支持评论、点赞和评分，形成互动反馈；
- 推荐结果需具备一定新颖性，避免重复；
- 提供基础赏析信息，辅助阅读理解。

#### 3.1.2 现有诗歌推荐系统的分析

现有系统普遍存在以下不足：

- 主要依赖关键词匹配，语义理解不足；
- 冷启动用户推荐能力有限；
- 推荐解释和反馈闭环不完整；
- 统计分析功能较弱，难以支持策略迭代。

#### 3.1.3 新系统的需求分析

新系统应满足：

- 基于文本语义与行为数据的混合推荐；
- 支持新用户偏好引导与老用户个性化增强；
- 支持主题、朝代、热门度等多维统计展示；
- 提供完整前后端接口与数据库支撑。

### 3.2 诗歌推荐系统架构设计

#### 3.2.1 架构设计

系统采用前后端分离架构：

- 前端：负责页面渲染、用户交互、推荐结果展示；
- 后端：提供用户管理、诗歌检索、推荐计算、统计接口；
- 数据层：存储用户、诗歌、评论及行为数据；
- 模型层：维护BERTopic模型与推荐器缓存。

#### 3.2.2 用例图分析

核心参与者为“普通用户”和“管理员（可选）”。

- 普通用户：注册登录、设置偏好、查看推荐、搜索诗歌、提交评论评分；
- 管理员：数据维护、运行状态监控、统计信息查看。

### 3.3 数据库概念结构设计

#### 3.3.1 部分实体的E-R图

核心实体包括：

- User（用户）：用户名、密码哈希、偏好主题、创建时间；
- Poem（诗歌）：题目、作者、朝代、内容、主题信息、热度字段；
- Review（评论）：用户ID、诗歌ID、评分、点赞、评论内容、主题词。

实体关系：

- 用户与评论：一对多；
- 诗歌与评论：一对多；
- 用户与诗歌通过评论形成多对多偏好关系。

#### 3.3.2 数据库逻辑结构设计及SQL脚本

逻辑设计要点：

- users 表设置 username 唯一约束；
- poems 表存储内容字段与主题字段（如 bertopic、real_topic）；
- reviews 表存储评分、点赞和评论文本，并关联 users/poems 外键；
- 对 reviews(user_id, created_at)、reviews(poem_id, created_at) 建议建立索引。

（此处在正式论文中可附建表 SQL 脚本与字段说明表）

### 3.4 推荐策略设计

本系统推荐策略采用“业务规则 + 模型推荐”双层机制：

1. 访客模式随机推荐；
2. 跳过次数触发探索推荐，降低热门垄断；
3. 新用户优先偏好主题匹配与热门补充；
4. 老用户进入BERTopic混合推荐：内容相似、Item-CF、User-CF、主题对齐、多样性重排。

### 3.5 本章小结

本章完成了系统需求分析、总体架构设计、数据库结构设计与核心推荐策略设计，为后续实现提供了可执行蓝图。

---

## 第四章 诗歌推荐系统的实现

### 4.1 系统处理流程

系统处理流程如下：

1. 用户登录后进入推荐主页；
2. 后端读取用户历史行为与会话上下文；
3. 根据用户状态选择推荐路径（冷启动/探索/个性化）；
4. 生成推荐结果并返回推荐原因；
5. 用户产生新反馈（评论、评分、点赞）后触发模型刷新。

### 4.2 系统功能模块划分与实现

系统主要模块：

- 用户模块：注册、登录、资料维护；
- 诗歌模块：列表、详情、搜索、赏析；
- 推荐模块：单首推荐、候选过滤、回退策略；
- 偏好模块：主题选择、偏好保存；
- 统计模块：热门诗歌、主题分布、朝代分布、趋势分析。

### 4.3 诗歌推荐系统实现

#### 4.3.1 诗歌推荐系统实现的关键技术

- Flask + SQLAlchemy 实现 RESTful 服务；
- BERTopic + 句向量实现主题语义建模；
- 混合推荐融合与多样性重排；
- 前后端分离提升系统可维护性。

#### 4.3.2 主题建模模块实现

主题建模模块流程：

1. 对诗歌文本进行分词与清洗；
2. 生成文本嵌入向量；
3. 调用BERTopic得到主题标签与关键词；
4. 将主题信息写入诗歌数据并用于推荐。

#### 4.3.3 用户兴趣画像构建实现

系统通过用户评分、点赞、评论时间衰减计算兴趣权重，并聚合为用户向量；同时利用评论关键词抽取形成主题偏好补充，实现显式与隐式信号融合。

#### 4.3.4 混合推荐与重排实现

召回阶段生成多路候选：内容相似、物品协同、用户协同、主题对齐；排序阶段进行归一化加权，最后通过MMR进行多样性重排，降低推荐结果同质化。

#### 4.3.5 冷启动与探索策略实现

针对新用户，系统引入偏好引导页，优先根据初始主题偏好推荐；针对长会话用户，设置“跳过触发探索”机制，定期注入小众高质量诗歌，提高发现能力。

### 4.4 本章小结

本章围绕系统流程、模块功能与关键实现细节进行说明，给出了从主题建模到推荐输出的完整实现路径。

---

## 第五章 系统测试

### 5.1 测试方法与环境

测试方法包括：

- 功能测试：验证接口与业务流程正确性；
- 性能测试：评估接口响应时间与稳定性；
- 推荐效果测试：比较不同算法在离线指标上的表现。

测试环境示例：

- 操作系统：Windows/Linux；
- 后端：Python 3.x，Flask；
- 数据库：MySQL；
- 前端：Vue + Vite。

### 5.2 测试方案

#### 5.2.1 方案

1. 构建测试数据集与行为日志；
2. 按用户时间切分训练/测试集；
3. 运行单算法与混合算法对比实验；
4. 统计 MAE、Precision、Recall、F1 等指标；
5. 结合接口日志评估推荐稳定性。

#### 5.2.2 用例选择

典型用例包括：

- 新用户首次进入推荐；
- 老用户连续“换一首”请求；
- 提交评论后推荐是否更新；
- 搜索词命中作者/标题/内容；
- 极端场景（无数据、模型加载失败）回退是否生效。

### 5.3 测试结果

测试结果可从三方面总结：

- 功能层面：核心业务流程可正常执行，推荐、搜索、评论、统计接口可用；
- 性能层面：推荐接口在常规并发下保持稳定，模型刷新机制可避免重复训练造成阻塞；
- 效果层面：混合策略在F1等相关指标上优于单一路径，体现了融合推荐的有效性。

（正式论文中可附表格：算法对比指标、接口响应时延、错误率等）

### 5.4 本章小结

本章从方法、方案、用例和结果四个层面验证了系统的可用性与有效性，为论文结论提供实验支撑。

---

## 第六章 总结与展望

### 6.1 总结

本文完成了一个基于BERTopic的协同过滤诗歌推荐系统的设计与实现。通过主题建模与协同过滤融合，系统在诗歌语义理解和个性化推荐方面取得了较好效果；通过冷启动和探索机制，系统在推荐新颖性与覆盖率方面得到改善；通过完整的前后端实现与测试，验证了系统具备工程落地能力。

### 6.2 展望

后续可从以下方向继续优化：

1. 引入更丰富的行为信号（点击、停留时长、收藏）实现在线学习；
2. 建立A/B测试平台，形成持续迭代闭环；
3. 结合知识图谱与大语言模型增强推荐解释性；
4. 优化模型推理与缓存策略，进一步提升系统吞吐能力。

---

## 参考文献

[1] Blei D M, Ng A Y, Jordan M I. Latent Dirichlet Allocation[J]. Journal of Machine Learning Research, 2003.

[2] Reimers N, Gurevych I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks[C]. EMNLP, 2019.

[3] Grootendorst M. BERTopic: Neural Topic Modeling with a Class-based TF-IDF Procedure[EB/OL].

[4] Sarwar B, Karypis G, Konstan J, et al. Item-based Collaborative Filtering Recommendation Algorithms[C]. WWW, 2001.

[5] Ricci F, Rokach L, Shapira B. Recommender Systems Handbook[M]. Springer, 2022.

[6] 赵某某, 李某某. 中文文本推荐研究综述[J]. 计算机工程与应用, 20xx.

[7] 王某某. 古诗词智能推荐方法研究[D]. 某某大学硕士论文, 20xx.

> 注：请你在定稿前将占位文献替换为真实可检索来源，并统一参考文献格式。

---

## 致 谢

本论文在选题、开发与撰写过程中得到了导师的悉心指导与同学的热心帮助。在此，谨向关心和支持我的老师、同学及家人表示诚挚感谢。导师在研究思路、系统实现和论文结构方面给予了大量建议，使我能够顺利完成本课题。感谢实验室同学在测试与讨论阶段提出的宝贵意见。最后，感谢家人始终如一的理解与支持。

---

## 附  录（加注释）

### 附录A 关键接口说明（示例）

1. `POST /api/login`：用户登录。
2. `POST /api/register`：用户注册。
3. `GET /api/recommend_one/<username>`：获取单首推荐。
4. `POST /api/poem/review`：提交评论、评分与点赞。
5. `GET /api/search_poems?q=关键词`：诗歌检索。

### 附录B 核心数据表字段（示例）

- users(id, username, password_hash, preference_topics, created_at)
- poems(id, title, author, dynasty, content, bertopic, real_topic, views, likes, shares)
- reviews(id, user_id, poem_id, rating, liked, comment, topic_names, created_at)

### 附录C 实验指标定义（示例）

- Precision@K：前K条推荐中相关项目占比；
- Recall@K：被推荐命中的相关项目占全部相关项目比例；
- F1@K：Precision与Recall调和平均；
- MAE：预测评分与真实评分的平均绝对误差。
