# 重构计划：基于评论语义的推荐系统

## 背景

当前系统使用**诗歌内容**生成语义向量，无法区分用户评论中的深层动机差异。

目标：改为基于**用户评论**的语义分析，区分用户的真实兴趣。

---

## 核心改动

### 当前逻辑
```
poem.content → SentenceTransformer → 语义向量 → 诗歌相似度
```

### 目标逻辑
```
review.comment → SentenceTransformer → 评论语义向量 → 用户画像 → 推荐
```

---

## 详细方案

### 方案：用户评论语义画像（推荐）

#### 核心思想
- 不再为诗歌生成语义向量
- 而是为每个用户生成**评论语义画像**
- 基于用户评论过的诗歌的**评论内容**来编码

#### 数据流
```
用户A的评论们
  ↓
合并评论文本
  ↓
SentenceTransformer 编码
  ↓
用户A的语义画像 (384维)
  ↓
找到相似画像的用户 → 推荐他们喜欢的诗歌
```

#### 冷启动解决
- 新用户（无评论）：使用诗歌内容语义
- 有评论用户：使用评论语义

---

## 待实现任务

### Phase 1: 修改数据构建

- [ ] 1.1 修改 `_build_interactions()` - 添加评论内容
```python
{
    "user_id": r.user_id,
    "poem_id": r.poem_id,
    "rating": r.rating,
    "comment": r.comment or "",  # 新增
}
```

- [ ] 1.2 修改 `_build_poems()` - 保留诗歌内容（用于冷启动）
```python
{
    "id": p.id,
    "content": p.content or "",
    "title": p.title or "",
    "comments": [所有评论文本...]  # 新增：用于生成诗歌评论语义
}
```

---

### Phase 2: 修改算法核心

- [ ] 2.1 新增 `_build_comment_embeddings()` - 为每首诗的评论生成语义向量

```python
def _build_comment_embeddings(self, poems, interactions):
    """为每首诗的评论生成语义向量"""
    # 按poem_id分组收集评论
    poem_comments = defaultdict(list)
    for inter in interactions:
        if inter.get("comment"):
            poem_comments[inter["poem_id"]].append(inter["comment"])
    
    # 为每首诗生成评论语义向量（多评论取平均）
    for poem in poems:
        comments = poem_comments.get(poem["id"], [])
        if comments:
            # 合并评论，用SentenceTransformer编码
            combined = " ".join(comments)
            emb = model.encode(combined)
        else:
            # 无评论则用诗歌内容作为fallback
            emb = model.encode(poem.get("content", ""))
    
    return embeddings
```

- [ ] 2.2 新增 `_build_user_comment_profiles()` - 基于评论生成用户画像

```python
def _build_user_comment_profiles(self):
    """基于用户评论生成语义画像"""
    # 用户画像 = 加权平均(评论语义向量)
    # 权重 = 评分 - 全局均值
```

- [ ] 2.3 修改 `recommend()` - 使用评论语义

```python
# 新用户：无评论，使用诗歌内容语义
# 老用户：有评论，使用评论语义
```

---

### Phase 3: 缓存机制更新

- [ ] 3.1 更新缓存逻辑 - 包含评论数据hash

- [ ] 3.2 添加评论变化检测

---

## 关键文件改动

| 文件 | 改动 |
|-----|------|
| `app.py` | 修改 `_build_poems()`, `_build_interactions()` |
| `core/bertopic_enhanced_cf.py` | 新增评论语义方法，修改推荐逻辑 |

---

## 验收标准

- [ ] 新用户（无评论）仍能获得推荐（使用诗歌内容fallback）
- [ ] 有评论用户的推荐基于评论语义
- [ ] 评论变化后触发模型更新
- [ ] 缓存机制正常工作

---

## 学术价值

此改动将使系统具备：
1. **动机分析能力**：区分"思乡"vs"月亮"等不同兴趣
2. **可解释推荐**：可以说明"因为你的评论提到X"
3. **更精准的协同**：基于评论的相似度比纯行为更准确
