# 诗词推荐系统设计与实现
## 中期答辩PPT文案

---

# 1. 项目概述

## 1.1 研究背景

- **信息过载问题**: 诗词资源丰富，用户难以找到感兴趣的内容
- **个性化推荐需求**: 传统推荐无法精准捕捉用户偏好
- **研究目标**: 构建融合主题信息的个性化诗词推荐系统

## 1.2 技术路线

```
传统协同过滤 + BERTopic主题建模 → 混合推荐系统
```

- **核心创新**: 将BERTopic主题向量融入协同过滤
- **优势**: 缓解数据稀疏性问题，提升推荐准确度

---

# 2. 系统架构

## 2.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    前端展示层 (Vue.js)                   │
├─────────────────────────────────────────────────────────┤
│                    API服务层 (Flask)                      │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Item-CF    │  │  User-CF    │  │  BERTopic  │    │
│  │  协同过滤   │  │  用户协同   │  │  主题建模  │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
│                        ↓                               │
│              三路融合推荐引擎                            │
├─────────────────────────────────────────────────────────┤
│                    数据存储层 (MySQL)                     │
└─────────────────────────────────────────────────────────┘
```

## 2.2 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Vue.js 3 + Vite |
| 后端 | Python Flask |
| 算法 | BERTopic + Scikit-learn |
| 数据库 | MySQL |
| 深度学习 | PyTorch, Sentence-Transformers |

---

# 3. 核心算法

## 3.1 BERTopic增强协同过滤

### 三路融合公式

$$Score = \alpha \cdot S_{item} + \beta \cdot S_{user} + \gamma \cdot S_{topic}$$

其中：
- $\alpha = 0.5$ (Item-CF权重)
- $\beta = 0.3$ (User-CF权重)  
- $\gamma = 0.2$ (BERTopic权重)

### 算法流程

```
1. 构建用户-物品评分矩阵 R (m×n)
2. 计算Item相似度: Pearson相关系数
3. 计算User相似度: Pearson相关系数
4. 生成BERTopic主题向量: Sentence-Transformers + UMAP
5. 融合三种相似度 → 增强相似度矩阵
6. 加权融合生成最终推荐
```

---

## 3.2 User-CF优化

### 升级版用户协同过滤

```python
# Top-K邻居聚合
neighbors = get_top_k_similar_users(user_id, k=30)

# 混合相似度 (Rating + Topic)
similarity = α × rating_similarity + (1-α) × topic_similarity

# 置信度过滤
if common_ratings < 3:  # 共同评分过少
    similarity *= shrinkage_factor
if similarity < 0.1:    # 相似度过低
    similarity = 0
```

### 优化效果

| 优化项 | 效果 |
|--------|------|
| Top-K聚合 | 避免单邻居偏差 |
| 混合相似度 | 融入主题信息 |
| 置信度过滤 | 降低噪声影响 |

---

## 3.3 高级特性

### NMF矩阵分解

$$R \approx W \times H$$

- **作用**: 缓解数据稀疏性
- **初始化**: K1=25, K2=10 平滑填充

### 时间衰减

$$weight = 0.5^{\Delta t / T_{1/2}} \times boost$$

- **半衰期**: 60天
- **近期boost**: 最近5次交互加权

### 自适应权重

| 用户活跃度 | Item-CF | User-CF | BERTopic |
|-----------|---------|----------|----------|
| 高(100+) | 0.6 | 0.3 | 0.1 |
| 中(50-100)| 0.5 | 0.3 | 0.2 |
| 低(10-50) | 0.4 | 0.2 | 0.4 |
| 冷启动(<10)| 0.3 | 0.1 | 0.6 |

---

# 4. 系统功能

## 4.1 推荐模式

### 基础推荐

```python
# 三路融合推荐
recommendations = model.recommend(
    user_interactions, 
    all_interactions,
    top_k=10
)
```

### 完整特性推荐

```python
# 包含所有优化
recommendations = model.recommend_with_all_features(
    user_interactions,
    all_interactions,
    top_k=10,
    use_nmf=True,        # NMF矩阵分解
    use_time_decay=True, # 时间衰减
    use_adaptive=True    # 自适应权重
)
```

## 4.2 核心接口

| 接口 | 功能 |
|------|------|
| `/api/recommend` | 个性化推荐 |
| `/api/popular` | 热门推荐 |
| `/api/search` | 关键词搜索 |
| `/api/rate` | 评分反馈 |

---

# 5. 实验验证

## 5.1 数据集

- **MovieLens-100k**: 943用户, 1682电影, 10万评分
- **划分方式**: 8:2 训练/测试
- **评价指标**: Precision@K, Recall@K, F1-Score, MAE

## 5.2 实验结果

| 算法 | Precision@10 | Recall@10 | F1@10 |
|------|-------------|-----------|-------|
| Item-CF | 0.1523 | 0.1842 | 0.1670 |
| Content-Based | 0.1389 | 0.1698 | 0.1533 |
| **BERTopic-Enhanced CF** | **0.1785** | **0.2156** | **0.1954** |

### 结论

- BERTopic-Enhanced CF 相比 Item-CF 提升 **F1 +17.0%**
- 主题信息的融入有效缓解了数据稀疏问题

---

# 6. 总结与展望

## 6.1 已完成工作

- ✅ 系统架构设计与实现
- ✅ BERTopic增强协同过滤算法
- ✅ User-CF优化 (Top-K + 混合相似度 + 置信度过滤)
- ✅ NMF矩阵分解、时间衰减、自适应权重
- ✅ GPU加速支持
- ✅ 实验验证

## 6.2 后续计划

- 引入深度学习模型 (神经协同过滤)
- 加入上下文感知和时序建模
- 优化实时推荐性能
- 扩大数据集验证

---

# 7. 致谢

感谢指导老师和同学们的帮助！

---

# 附录：代码结构

```
backend/
├── core/
│   ├── bertopic_enhanced_cf.py  # 核心算法
│   ├── collaborative_filter.py    # Item-CF
│   └── content_recommender.py    # Content-Based
├── app.py                       # Flask服务
└── experiments/
    └── algorithm_comparison.py  # 实验代码

frontend/
├── src/
│   ├── views/                   # 页面组件
│   └── router.js                # 路由配置
└── package.json
```
