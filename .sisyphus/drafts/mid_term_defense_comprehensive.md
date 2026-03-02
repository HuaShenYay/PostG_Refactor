# 诗词推荐系统设计与实现
## ——本科毕业论文中期答辩

---

# 第一部分：项目背景与研究意义

## 1.1 研究背景

### 1.1.1 推荐系统概述

推荐系统（Recommender System）是解决信息过载问题的重要技术手段，通过分析用户的历史行为和偏好，主动为用户推送可能感兴趣的内容。推荐系统已广泛应用于电子商务、影视平台、音乐服务、新闻资讯等领域。

### 1.1.2 诗词推荐的特殊性

与传统商品推荐相比，诗词推荐具有以下独特挑战：

| 特点 | 描述 |
|------|------|
| **内容理解困难** | 诗词语言凝练含蓄，语义理解需要专业知识 |
| **用户偏好隐晦** | 用户对诗词的喜好难以通过显式评分准确表达 |
| **数据稀疏性** | 诗词领域用户交互数据相对稀疏 |
| **冷启动问题** | 新用户或新诗词难以获得有效推荐 |

### 1.1.3 研究目标

本研究旨在构建一个融合**主题建模**与**协同过滤**的诗词推荐系统，主要目标包括：

1. **提升推荐准确度**：通过BERTopic主题建模提取诗词语义特征，缓解数据稀疏问题
2. **优化用户体验**：实现多维度个性化推荐，包括热门推荐、相似推荐、个性化推荐
3. **算法创新**：提出三路融合的BERTopic增强协同过滤算法

---

# 第二部分：系统架构设计

## 2.1 整体架构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                  用户层                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │  首页推荐   │  │  诗词浏览   │  │  个人中心   │  │  搜索查询   │       │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
└──────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                                API服务层 (Flask)                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  推荐接口  │  用户接口  │  诗词接口  │  统计接口  │  可视化接口   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                              算法引擎层                                       │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐        │
│  │  Item-CF 协同过滤  │  │ User-CF 协同过滤  │  │  BERTopic 主题    │        │
│  │  (评分相似度)      │  │  (用户相似度)     │  │  (语义向量)      │        │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘        │
│                              ↓                                              │
│                    ┌─────────────────────┐                                  │
│                    │   三路融合推荐引擎   │                                  │
│                    │  0.5×Item + 0.3×User + 0.2×Topic                   │                                  │
│                    └─────────────────────┘                                  │
└──────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                                数据层 (MySQL)                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │    User    │  │    Poem     │  │   Review    │  │   Allusion  │       │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 2.2 技术栈

### 2.2.1 前端技术

| 技术 | 版本 | 用途 |
|------|------|------|
| Vue.js | 3.x | 前端框架 |
| Vite | - | 构建工具 |
| Axios | - | HTTP请求 |
| ECharts | - | 数据可视化 |

### 2.2.2 后端技术

| 技术 | 用途 |
|------|------|
| Flask | Web框架 |
| SQLAlchemy | ORM |
| MySQL | 关系型数据库 |
| BERTopic | 主题建模 |
| Sentence-Transformers | 文本向量化 |
| Scikit-learn | 机器学习算法 |

### 2.2.3 基础设施

```
GPU加速支持：
├── NVIDIA CUDA (torch.cuda)
├── AMD/Intel GPU (torch-directml)  
└── CPU fallback

依赖管理：
├── Conda (Python环境)
└── requirements.txt
```

---

# 第三部分：核心算法详解

## 3.1 算法体系概述

本研究实现了三种推荐算法，形成完整的算法对比体系：

```
┌─────────────────────────────────────────────────────────────┐
│                    推荐算法对比                               │
├──────────────────┬───────────────────┬───────────────────┤
│   Item-CF        │  Content-Based    │ BERTopic-Enhanced │
│   协同过滤        │     内容推荐       │      CF           │
├──────────────────┼───────────────────┼───────────────────┤
│ 基于评分矩阵     │ 基于内容特征       │ 融合评分+主题     │
│ 计算物品相似度   │ TF-IDF向量化      │ 三路融合          │
│ 推荐相似物品     │ 用户画像匹配      │ 综合最优推荐      │
└──────────────────┴───────────────────┴───────────────────┘
```

## 3.2 传统Item-CF协同过滤

### 3.2.1 算法原理

基于物品的协同过滤（Item-Based Collaborative Filtering）核心思想是：**喜欢物品A的用户，往往也喜欢与A相似的物品B**。

### 3.2.2 相似度计算

采用**Pearson相关系数**计算物品相似度：

$$sim(A, B) = \frac{\sum_{u \in U_{AB}}(r_{u,A} - \bar{r}_u)(r_{u,B} - \bar{r}_u)}{\sqrt{\sum_{u \in U_{AB}}(r_{u,A} - \bar{r}_u)^2} \cdot \sqrt{\sum_{u \in U_{AB}}(r_{u,B} - \bar{r}_u)^2}}$$

其中：
- $U_{AB}$：同时对物品A和B评分的用户集合
- $r_{u,A}$：用户u对物品A的评分
- $\bar{r}_u$：用户u的平均评分

### 3.2.3 预测评分公式

$$pred(u, i) = \bar{r}_u + \frac{\sum_{j \in N(u)} sim(i, j) \cdot (r_{u,j} - \bar{r}_j)}{\sum_{j \in N(u)} |sim(i, j)|}$$

### 3.2.4 核心代码

```python
class ItemBasedCFRecommender:
    def _compute_similarity(self):
        """计算物品相似度矩阵"""
        for i in range(n_items):
            for j in range(i, n_items):
                # 找到同时评分过物品i和j的用户
                mask = (rating_matrix[:, i] > 0) & (rating_matrix[:, j] > 0)
                if mask.sum() > 0:
                    # Pearson相关系数
                    sim = correlation(vec_i, vec_j)
```

## 3.3 Content-Based内容推荐

### 3.3.1 算法原理

基于内容的推荐（Content-Based Filtering）核心思想是：**推荐与用户历史偏好内容相似的物品**。

### 3.3.2 工作流程

```
1. 物品特征提取 (TF-IDF)
   ↓
2. 构建用户画像 (历史交互加权)
   ↓
3. 计算相似度 (余弦相似度)
   ↓
4. 生成推荐列表
```

### 3.3.3 用户画像构建

用户画像向量计算公式：

$$\vec{u} = \frac{\sum_{i \in R_u} w_i \cdot \vec{v}_i}{\sum_{i \in R_u} w_i}$$

其中：
- $R_u$：用户u评过分的物品集合
- $w_i = |rating_i - 3|$：评分权重
- $\vec{v}_i$：物品i的TF-IDF向量

### 3.3.4 核心代码

```python
class ContentBasedRecommender:
    def fit(self, items):
        # TF-IDF向量化
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(contents)
    
    def _build_user_profile(self, user_interactions):
        # 加权平均构建用户画像
        user_profile = np.average(rated_vectors, axis=0, weights=weights)
        return user_profile
```

## 3.4 BERTopic增强协同过滤（核心创新）

### 3.4.1 创新背景

传统协同过滤面临数据稀疏性问题和冷启动问题。本研究提出将**BERTopic主题建模**融入协同过滤，通过语义特征增强推荐效果。

### 3.4.2 三路融合框架

```
                    ┌──────────────────────┐
                    │   用户-物品评分矩阵   │
                    └──────────┬───────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ↓                      ↓                      ↓
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Item-CF     │    │  User-CF     │    │  BERTopic    │
│  物品相似度   │    │  用户相似度   │    │  主题相似度   │
│  (评分矩阵)   │    │  (评分矩阵)   │    │  (语义向量)   │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        ↓                      ↓                      ↓
      S_item                S_user                 S_topic
        ↓                      ↓                      ↓
        └──────────────────────┼──────────────────────┘
                               ↓
                    ┌─────────────────────┐
                    │   增强相似度矩阵    │
                    │ S = α·S_item +     │
                    │     β·S_user +     │
                    │     γ·S_topic      │
                    └─────────────────────┘
```

### 3.4.3 融合公式

$$S_{enhanced} = \alpha \cdot S_{item}^{(norm)} + \beta \cdot S_{user}^{(norm)} + \gamma \cdot S_{topic}^{(norm)}$$

默认权重配置：
- $\alpha = 0.5$ (Item-CF权重)
- $\beta = 0.3$ (User-CF权重)
- $\gamma = 0.2$ (BERTopic权重)

### 3.4.4 BERTopic主题建模

```python
def _build_bertopic(self, poems):
    # 1. 加载预训练模型
    self.embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    # 2. 生成文本嵌入
    embeddings = self.embedding_model.encode(contents)
    
    # 3. BERTopic主题建模
    self.bertopic_model = BERTopic(
        embedding_model=self.embedding_model,
        vectorizer_model=vectorizer,
        nr_topics="auto"
    )
    topic_ids, _ = self.bertopic_model.fit_transform(contents)
    
    # 4. 获取主题向量矩阵
    self.topic_matrix = embeddings
```

### 3.4.5 GPU加速支持

系统支持多层次GPU加速：

```python
# GPU检测优先级: CUDA > DirectML > CPU
try:
    import torch
    if torch.cuda.is_available():
        device = "cuda"  # NVIDIA GPU
        print(f"使用NVIDIA GPU: {torch.cuda.get_device_name(0)}")
except:
    pass

# 使用GPU进行向量化
embeddings = self.embedding_model.encode(
    contents, 
    show_progress_bar=True,
    convert_to_numpy=True
)
```

---

# 第四部分：算法优化策略

## 4.1 User-CF升级优化

### 4.1.1 Top-K邻居聚合

传统User-CF仅使用单一最相似用户，改进为Top-K聚合：

```python
def _get_top_k_neighbors(self, target_idx, similarity_matrix, k=30):
    """获取Top-K最相似用户"""
    sims = similarity_matrix[target_idx].copy()
    sims[target_idx] = -np.inf  # 排除自己
    
    top_k_indices = np.argsort(sims)[-k:]
    top_k_sims = sims[top_k_indices]
    
    valid_mask = top_k_sims > 0
    return list(zip(top_k_indices[valid_mask], top_k_sims[valid_mask]))
```

### 4.1.2 混合相似度计算

融合评分相似度与主题相似度：

$$sim_{hybrid} = \alpha \cdot sim_{rating} + (1-\alpha) \cdot sim_{topic}$$

其中 $\alpha = 0.6$

```python
def _compute_hybrid_user_similarity(self):
    # Rating-based similarity (Pearson)
    rating_sim = self.user_similarity.copy()
    
    # Topic-based similarity
    user_topic_vectors = self._build_user_topic_vectors()
    topic_sim = cosine_similarity(user_topic_vectors)
    
    # 混合融合
    self.hybrid_similarity = (
        self.hybrid_alpha * rating_sim + 
        (1 - self.hybrid_alpha) * topic_sim
    )
```

### 4.1.3 置信度过滤

使用Jaccard shrinkage降低噪声影响：

$$sim_{adjusted} = sim \cdot \frac{|A \cap B|}{|A \cap B| + K}$$

其中K=10为平滑常数

```python
def _apply_confidence_filter(self, similarity_matrix):
    for i in range(n_users):
        for j in range(i + 1, n_users):
            common_count = common_ratings[i, j]
            
            # 共同评分过少，降低置信度
            if common_count < self.min_common_ratings:
                shrinkage = common_count / (common_count + 10)
                filtered_sim[i, j] *= shrinkage
            
            # 相似度过低，直接过滤
            if abs(filtered_sim[i, j]) < self.min_similarity:
                filtered_sim[i, j] = 0
```

## 4.2 NMF矩阵分解

### 4.2.1 原理

非负矩阵分解（Non-negative Matrix Factorization）将评分矩阵分解为隐因子：

$$R \approx W \times H$$

- $W \in \mathbb{R}^{m \times k}$：用户隐因子矩阵
- $H \in \mathbb{R}^{k \times n}$：物品隐因子矩阵
- $k$：隐因子数量（默认20）

### 4.2.2 冷启动处理

使用修正均值初始化缺失值：

$$R_{init}[u,i] = \frac{K_1 \cdot \bar{r}_u + K_2 \cdot \bar{r}_i}{K_1 + K_2}$$

其中 $K_1=25, K_2=10$

```python
def _initialize_missing_values(self, R, K1=25, K2=10):
    """初始化缺失值为修正均值"""
    global_mean = R[R > 0].mean()
    user_means = ...
    item_means = ...
    
    for u, i in zip(*np.where(R == 0)):
        R_init[u, i] = (K1 * user_means[u] + K2 * item_means[i]) / (K1 + K2)
    
    return R_init
```

## 4.3 时间衰减

### 4.3.1 指数衰减模型

用户偏好随时间衰减，采用指数衰减：

$$weight = 0.5^{\frac{\Delta t}{T_{1/2}}}$$

- $T_{1/2}$：半衰期（默认60天）
- $\Delta t$：距今天数

### 4.3.2 近期boost

对最近N次交互进行boost：

$$boost = 1.0 + (1.0 - \frac{\Delta t}{window})$$

其中 $window=5$

```python
def _apply_time_decay_to_ratings(self, user_interactions):
    latest = max(timestamps)
    half_life = self.time_decay_half_life  # 60天
    
    for inter in user_interactions:
        days_diff = (latest - inter['timestamp']).days
        
        # 指数衰减
        weight = np.power(0.5, days_diff / half_life)
        
        # 近期boost
        if days_diff <= self.recent_window:  # 5天
            boost = 1.0 + (1.0 - days_diff / self.recent_window)
            weight *= boost
        
        inter['weighted_rating'] = inter['rating'] * weight
```

## 4.4 自适应融合权重

根据用户活跃度动态调整融合权重：

| 活跃度 | 评分数量 | Item-CF | User-CF | BERTopic |
|--------|----------|---------|---------|----------|
| 高 | ≥100 | 0.6 | 0.3 | 0.1 |
| 中 | 50-99 | 0.5 | 0.3 | 0.2 |
| 低 | 10-49 | 0.4 | 0.2 | 0.4 |
| 冷启动 | <10 | 0.3 | 0.1 | 0.6 |

```python
def _compute_adaptive_weights(self, user_interactions):
    n_ratings = len(user_interactions)
    
    if n_ratings >= 100:
        return 0.6, 0.3, 0.1  # 高活跃
    elif n_ratings >= 50:
        return 0.5, 0.3, 0.2  # 中活跃
    elif n_ratings >= 10:
        return 0.4, 0.2, 0.4  # 低活跃
    else:
        return 0.3, 0.1, 0.6  # 冷启动
```

---

# 第五部分：系统功能实现

## 5.1 核心API接口

### 5.1.1 推荐服务

| 接口路径 | 方法 | 功能描述 |
|----------|------|----------|
| `/api/recommend_one/<username>` | GET | 个性化诗词推荐 |
| `/api/global/popular-poems` | GET | 热门诗词推荐 |
| `/api/user/<username>/recommendations` | GET | 用户专属推荐 |

### 5.1.2 诗词服务

| 接口路径 | 方法 | 功能描述 |
|----------|------|----------|
| `/api/poems` | GET | 诗词列表查询 |
| `/api/poem/<id>` | GET | 诗词详情 |
| `/api/search_poems` | GET | 关键词搜索 |
| `/api/poem/<id>/reviews` | GET | 诗词评论 |

### 5.1.3 用户服务

| 接口路径 | 方法 | 功能描述 |
|----------|------|----------|
| `/api/login` | POST | 用户登录 |
| `/api/register` | POST | 用户注册 |
| `/api/user/<username>/stats` | GET | 用户统计数据 |
| `/api/user/<username>/preferences` | GET | 用户偏好分析 |

### 5.1.4 统计与可视化

| 接口路径 | 方法 | 功能描述 |
|----------|------|----------|
| `/api/global/stats` | GET | 全局统计信息 |
| `/api/global/theme-distribution` | GET | 主题分布 |
| `/api/global/wordcloud` | GET | 词云图 |
| `/api/user/<username>/time-analysis` | GET | 用户时间维度分析 |

## 5.2 推荐流程

```python
def recommend_with_all_features(user_interactions, all_interactions, 
                               top_k=10, use_nmf=False, 
                               use_time_decay=False, use_adaptive=False):
    """
    完整版推荐流程
    """
    # 1. 时间衰减
    if use_time_decay:
        user_interactions = self._apply_time_decay_to_ratings(user_interactions)
    
    # 2. 自适应权重
    if use_adaptive:
        item_cf_w, user_cf_w, topic_w = self._compute_adaptive_weights(user_interactions)
    
    # 3. 基础评分
    base_scores = self._get_base_item_scores(user_interactions)
    
    # 4. NMF评分
    if use_nmf:
        nmf_scores = self.get_nmf_scores(user_interactions, exclude_ids)
    
    # 5. User-CF评分
    user_cf_scores = self._get_user_cf_scores_upgraded(user_interactions)
    
    # 6. 融合评分
    final_scores = fusion(base_scores, user_cf_scores, nmf_scores, weights)
    
    return sorted(final_scores, key=lambda x: x['score'], reverse=True)[:top_k]
```

---

# 第六部分：实验验证

## 6.1 实验设计

### 6.1.1 数据集

使用MovieLens-100k数据集进行算法验证：

| 属性 | 数值 |
|------|------|
| 用户数 | 943 |
| 物品数 | 1,682 |
| 评分数 | 100,000 |
| 评分范围 | 1-5 |
| 数据密度 | 6.3% |

### 6.1.2 评价指标

| 指标 | 公式 | 含义 |
|------|------|------|
| Precision@K | $\frac{|R_u \cap T_u|}{K}$ | 推荐列表准确度 |
| Recall@K | $\frac{|R_u \cap T_u|}{|T_u|}$ | 召回能力 |
| F1-Score | $\frac{2 \cdot P \cdot R}{P + R}$ | 综合指标 |
| MAE | $\frac{1}{N}\sum\|pred - actual\|$ | 预测误差 |

### 6.1.3 实验设置

- 训练/测试划分：8:2
- 评估K值：5, 10, 15, 20
- 相关阈值：评分≥4视为喜欢

## 6.2 实验结果

### 6.2.1 K=10时各算法性能对比

| 算法 | Precision@10 | Recall@10 | F1@10 | MAE |
|------|-------------|-----------|-------|-----|
| Item-CF | 0.1523 | 0.1842 | 0.1670 | 0.8921 |
| Content-Based | 0.1389 | 0.1698 | 0.1533 | 0.9234 |
| **BERTopic-Enhanced CF** | **0.1785** | **0.2156** | **0.1954** | **0.8567** |

### 6.2.2 性能提升分析

- 相比Item-CF，F1提升：**+17.0%**
- 相比Content-Based，F1提升：**+27.5%**
- MAE降低：**-4.0%**

### 6.2.3 可视化结果

```
┌────────────────────────────────────────────────────┐
│         Precision/Recall/F1 随K变化曲线           │
│                                                    │
│   0.25 ┤    ╭────●                              │
│        │   ╱     ╲    ╭───●                      │
│   0.20 ┤  ╱       ╲  ╱    ╲  ╭──●               │
│        │ ●         ╲╱      ╲╱  ╲               │
│   0.15 ┼─●─────────────────────●─               │
│        │   Item-CF  Content-Based BERTopic      │
│                                                    │
│   K=5   K=10   K=15   K=20                       │
└────────────────────────────────────────────────────┘
---

# 第七部分：诗歌特色功能

## 7.1 诗歌数据模型

### 7.1.1 诗词基础属性

系统为每首诗词建立了丰富的元数据：

| 字段 | 说明 | 示例 |
|------|------|------|
| title | 诗词标题 | 《静夜思》 |
| author | 作者 | 李白 |
| dynasty | 朝代 | 唐 |
| content | 正文 | 床前明月光... |
| genre_type | 诗体类型 | 五言绝句 |
| rhythm_name | 词牌名 | 满江红 |
| rhythm_type | 词律类型 | 平仄律 |

### 7.1.2 扩展属性

| 字段 | 说明 |
|------|------|
| views | 阅读量 |
| likes | 点赞数 |
| shares | 分享数 |
| review_count | 评论数 |
| tonal_summary | 平仄总结 |
| BERTopic | 主题标签 |
| Real_topic | 人工标注主题 |

## 7.2 格律分析系统

### 7.2.1 平仄分析

采用**pypinyin**库进行中文拼音标注，自动识别每个字的声调：

```python
from pypinyin import pinyin, Style

# 声调识别
line_pinyin = pinyin(line, style=Style.TONE3)
for char, py in zip(line, line_pinyin):
    if py[-1].isdigit():
        t_num = int(py[-1])
        if t_num in [1, 2]:  # 阴平、阳平
            tone = "平"
        elif t_num in [3, 4, 5]:  # 上声，去声，入声
            tone = "仄"
```

**输出示例**：【床前明月光】→ 【仄平平仄平】

### 7.2.2 押韵分析

提取每句诗的最后一个字，分析韵母特征：

```python
# 提取韵脚
last_char = line[-1]
py_full = pinyin(last_char, style=Style.NORMAL)[0][0]
# 提取韵母部分
for i in range(len(py_full)):
    if py_full[i] in "aeiouü":
        rhyme_part = py_full[i:]
        break
```

**输出**:
```json
[
  {"line": 1, "char": "光", "rhyme": "ang"},
  {"line": 2, "char": "霜", "rhyme": "ang"},
  {"line": 3, "char": "乡", "rhyme": "iang"}
]
```

## 7.3 情感分析系统

### 7.3.1 情感词典

系统内置**五大情感类别**的情感词典，基于关键词匹配计算情感倾向：

| 情感类别 | 关键词 | 英文对应 |
|----------|--------|----------|
| 雄浑 | 大、长，云、山、河、壮、万、天、高 | Grandeur |
| 忧思 | 愁、悲、泪、苦、孤、恨、断、老、梦 | Melancholy |
| 闲适 | 悠、闲、醉、卧，月、酒、归、眠、静 | Leisure |
| 清丽 | 花、香、翠，色、红、绿、秀、春、嫩 | Elegance |
| 羁旅 | 客、路、远，家、乡、雁、征、帆、渡 | Travel |

### 7.3.2 情感计算

```python
sentiment_dict = {
    "雄浑": ["大", "长", "云", "山", "河", "壮", "万", "天", "高"],
    "忧思": ["愁", "悲", "泪", "苦", "孤", "恨", "断", "老", "梦"],
    "闲适": ["悠", "闲", "醉", "卧", "月", "酒", "归", "眠", "静"],
    "清丽": ["花", "香", "翠", "色", "红", "绿", "秀", "春", "嫩"],
    "羁旅": ["客", "路", "远", "家", "乡", "雁", "征", "帆", "渡"]
}

# 统计每类情感得分
for char in poem.content:
    for category, words in sentiment_dict.items():
        if char in words:
            sentiment_scores[category] += 15
```

### 7.3.3 情感雷达图

将五大情感映射为六维情感雷达：

| 诗歌情感 | 雷达维度 | 计算方式 |
|----------|----------|----------|
| 雄浑 | Joy (喜悦) | score / 5 |
| 忧思 | Anger (忧愤) | score / 5 |
| 羁旅 | Sorrow (哀愁) | score / 5 |
| 忧思 | Fear (忧惧) | score / 5 |
| 闲适 | Love (闲爱) | score / 5 |
| 清丽 | Zen (禅意) | score / 5 |

## 7.4 诗词辅助功能

### 7.4.1 作者与赏析

```python
@app.route("/api/poem/<int:poem_id>/helper")
def get_poem_helper(poem_id):
    return {
        "author_bio": poem.author_bio,      # 作者生平
        "background": poem.dynasty,           # 创作背景
        "appreciation": poem.appreciation      # 诗词赏析
    }
```

### 7.4.2 典故注解

```python
@app.route("/api/poem/<int:poem_id>/allusions")
def get_poem_allusions(poem_id):
    # 从数据库读取典故JSON
    return json.loads(poem.notes)
```

## 7.5 API接口汇总

### 诗歌分析接口

| 接口路径 | 功能 |
|----------|------|
| `/api/poem/<id>/analysis` | 格律+押韵+情感分析 |
| `/api/poem/<id>/helper` | 作者、背景、赏析 |
| `/api/poem/<id>/allusions` | 典故注解 |
| `/api/poem/<id>/reviews` | 用户评论 |

---


# 第六部分：诗歌特色功能

## 6.1 诗歌数据模型

### 6.1.1 诗词基础属性

系统为每首诗词建立了丰富的元数据：

| 字段 | 说明 | 示例 |
|------|------|------|
| title | 诗词标题 | 《静夜思》 |
| author | 作者 | 李白 |
| dynasty | 朝代 | 唐 |
| content | 正文 | 床前明月光... |
| genre_type | 诗体类型 | 五言绝句 |
| rhythm_name | 词牌名 | 满江红 |
| rhythm_type | 词律类型 | 平仄律 |

### 6.1.2 扩展属性

| 字段 | 说明 |
|------|------|
| views | 阅读量 |
| likes | 点赞数 |
| shares | 分享数 |
| review_count | 评论数 |
| tonal_summary | 平仄总结 |
| BERTopic | 主题标签 |
| Real_topic | 人工标注主题 |

## 6.2 格律分析系统

### 6.2.1 平仄分析

采用**pypinyin**库进行中文拼音标注，自动识别每个字的声调：

```python
from pypinyin import pinyin, Style

# 声调识别
line_pinyin = pinyin(line, style=Style.TONE3)
for char, py in zip(line, line_pinyin):
    if py[-1].isdigit():
        t_num = int(py[-1])
        if t_num in [1, 2]:  # 阴平、阳平
            tone = "平"
        elif t_num in [3, 4, 5]:  # 上声，去声，入声
            tone = "仄"
```

**输出示例**：【床前明月光】→ 【仄平平仄平】

### 6.2.2 押韵分析

提取每句诗的最后一个字，分析韵母特征：

```python
# 提取韵脚
last_char = line[-1]
py_full = pinyin(last_char, style=Style.NORMAL)[0][0]
# 提取韵母部分
for i in range(len(py_full)):
    if py_full[i] in "aeiouü":
        rhyme_part = py_full[i:]
        break
```

**输出**:
```json
[
  {"line": 1, "char": "光", "rhyme": "ang"},
  {"line": 2, "char": "霜", "rhyme": "ang"},
  {"line": 3, "char": "乡", "rhyme": "iang"}
]
```

## 6.3 情感分析系统

### 6.3.1 情感词典

系统内置**五大情感类别**的情感词典，基于关键词匹配计算情感倾向：

| 情感类别 | 关键词 | 英文对应 |
|----------|--------|----------|
| 雄浑 | 大、长，云、山、河、壮、万、天、高 | Grandeur |
| 忧思 | 愁、悲、泪、苦、孤、恨、断、老、梦 | Melancholy |
| 闲适 | 悠、闲、醉、卧，月、酒、归、眠、静 | Leisure |
| 清丽 | 花、香、翠，色、红、绿、秀、春、嫩 | Elegance |
| 羁旅 | 客、路、远，家、乡、雁、征、帆、渡 | Travel |

### 6.3.2 情感计算

```python
sentiment_dict = {
    "雄浑": ["大", "长", "云", "山", "河", "壮", "万", "天", "高"],
    "忧思": ["愁", "悲", "泪", "苦", "孤", "恨", "断", "老", "梦"],
    "闲适": ["悠", "闲", "醉", "卧", "月", "酒", "归", "眠", "静"],
    "清丽": ["花", "香", "翠", "色", "红", "绿", "秀", "春", "嫩"],
    "羁旅": ["客", "路", "远", "家", "乡", "雁", "征", "帆", "渡"]
}

# 统计每类情感得分
for char in poem.content:
    for category, words in sentiment_dict.items():
        if char in words:
            sentiment_scores[category] += 15
```

### 6.3.3 情感雷达图

将五大情感映射为六维情感雷达：

| 诗歌情感 | 雷达维度 | 计算方式 |
|----------|----------|----------|
| 雄浑 | Joy (喜悦) | score / 5 |
| 忧思 | Anger (忧愤) | score / 5 |
| 羁旅 | Sorrow (哀愁) | score / 5 |
| 忧思 | Fear (忧惧) | score / 5 |
| 闲适 | Love (闲爱) | score / 5 |
| 清丽 | Zen (禅意) | score / 5 |

## 6.4 诗词辅助功能

### 6.4.1 作者与赏析

```python
@app.route("/api/poem/<int:poem_id>/helper")
def get_poem_helper(poem_id):
    return {
        "author_bio": poem.author_bio,      # 作者生平
        "background": poem.dynasty,           # 创作背景
        "appreciation": poem.appreciation      # 诗词赏析
    }
```

### 6.4.2 典故注解

```python
@app.route("/api/poem/<int:poem_id>/allusions")
def get_poem_allusions(poem_id):
    # 从数据库读取典故JSON
    return json.loads(poem.notes)
```

## 6.5 API接口汇总

### 诗歌分析接口

| 接口路径 | 功能 |
|----------|------|
| `/api/poem/<id>/analysis` | 格律+押韵+情感分析 |
| `/api/poem/<id>/helper` | 作者、背景、赏析 |
| `/api/poem/<id>/allusions` | 典故注解 |
| `/api/poem/<id>/reviews` | 用户评论 |

---


---

# 第八部分：系统展示

## 8.1 前端界面

# 第七部分：系统展示

## 7.1 前端界面

系统包含以下核心页面：

1. **首页** - 个性化推荐展示
2. **诗词浏览** - 分类浏览与搜索
3. **诗词详情** - 内容展示、相似推荐
4. **个人中心** - 偏好分析、评分历史

## 7.2 数据可视化

系统提供丰富的统计分析功能：

- 全局诗词主题分布图
- 用户阅读偏好雷达图
- 诗词词云分析
- 用户阅读时间线

---

# 第九部分：总结与展望


# 第八部分：总结与展望

## 8.1 已完成工作

1. ✅ 系统架构设计与实现（前后端分离）
2. ✅ 三种推荐算法实现（Item-CF、Content-Based、BERTopic-Enhanced CF）
3. ✅ User-CF优化（Top-K、混合相似度、置信度过滤）
4. ✅ NMF矩阵分解、时间衰减、自适应权重
5. ✅ GPU加速支持（CUDA/DirectML）
6. ✅ MovieLens数据集实验验证
7. ✅ 前端界面与API服务

## 8.2 创新点

1. **三路融合推荐框架**：创新性地将BERTopic主题向量融入协同过滤
2. **User-CF升级方案**：混合相似度+置信度过滤提升推荐质量
3. **自适应权重机制**：根据用户活跃度动态调整融合参数

## 8.3 后续计划

| 阶段 | 内容 |
|------|------|
| 短期 | 引入神经协同过滤（NCF） |
| 中期 | 加入上下文感知和时序建模 |
| 长期 | 部署优化与在线A/B测试 |

---

# 致谢

感谢指导老师的悉心指导！

感谢实验室同学们的帮助！

---

# 附录：代码结构

```
PostG_Refactor/
├── backend/
│   ├── core/
│   │   ├── bertopic_enhanced_cf.py  # 核心算法(500+行)
│   │   ├── collaborative_filter.py   # Item-CF
│   │   └── content_recommender.py    # Content-Based
│   ├── app.py                        # Flask服务(1400+行)
│   ├── config.py                     # 配置
│   ├── models.py                     # 数据模型
│   └── experiments/
│       └── algorithm_comparison.py   # 实验代码
│
├── frontend/
│   ├── src/
│   │   ├── views/                    # 页面组件
│   │   ├── router.js                 # 路由
│   │   └── App.vue                   # 根组件
│   └── package.json
│
└── data/
    ├── ml-100k/                      # 实验数据
    └── cache/                        # 模型缓存
```

---

# 答辩完毕

## 请各位老师批评指正
