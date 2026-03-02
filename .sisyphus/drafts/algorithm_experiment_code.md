# 推荐系统算法对比实验

## 实验概述
- **数据集**: MovieLens-100k
- **算法**: Item-CF, Content-Based, BERTopic-Enhanced CF
- **指标**: Precision, Recall, F1, MAE

---

## 完整代码 (复制到 Jupyter Notebook)

```python
# ============================================================
# 推荐系统算法对比实验
# BERTopic增强协同过滤 vs 传统协同过滤 vs 内容推荐
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 数据准备
# ============================================================

DATA_DIR = "../../data/ml-100k"

# 加载评分数据
ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
ratings_df = pd.read_csv(f"{DATA_DIR}/u.data", sep='\t', names=ratings_cols, encoding='latin-1')

# 加载电影信息
items_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + [f'genre_{i}' for i in range(19)]
items_df = pd.read_csv(f"{DATA_DIR}/u.item", sep='|', names=items_cols, encoding='latin-1')

# 加载用户信息
users_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users_df = pd.read_csv(f"{DATA_DIR}/u.user", sep='|', names=users_cols, encoding='latin-1')

print(f"评分数据: {ratings_df.shape}")
print(f"电影数据: {items_df.shape}")
print(f"用户数据: {users_df.shape}")

# ============================================================
# 2. 数据预处理
# ============================================================

# 提取电影类型作为内容特征
genre_cols = [f'genre_{i}' for i in range(19)]
genre_names = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
               'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
               'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
               'Thriller', 'War', 'Western']

items_df['genres'] = items_df[genre_cols].apply(
    lambda x: ' '.join([genre_names[i] for i in range(len(x)) if x.iloc[i] == 1]), axis=1)
items_df['content'] = items_df['title'] + ' ' + items_df['genres']

# 构建poems和interactions格式
poems = []
for _, row in items_df.iterrows():
    poems.append({
        'id': row['item_id'],
        'content': row['content'],
        'title': row['title']
    })

interactions = []
for _, row in ratings_df.iterrows():
    interactions.append({
        'user_id': row['user_id'],
        'poem_id': row['item_id'],
        'rating': row['rating'],
        'created_at': pd.to_datetime(row['timestamp'], unit='s')
    })

print(f"电影数: {len(poems)}, 交互数: {len(interactions)}")

# ============================================================
# 3. 数据集划分 (8:2)
# ============================================================

np.random.seed(42)

train_interactions = []
test_interactions = []

user_groups = defaultdict(list)
for inter in interactions:
    user_groups[inter['user_id']].append(inter)

for user_id, user_interactions in user_groups.items():
    user_interactions = sorted(user_interactions, key=lambda x: x['created_at'])
    n = len(user_interactions)
    
    if n >= 5:
        train_size = int(n * 0.8)
        train_interactions.extend(user_interactions[:train_size])
        test_interactions.extend(user_interactions[train_size:])
    else:
        train_interactions.extend(user_interactions)

print(f"训练集: {len(train_interactions)}, 测试集: {len(test_interactions)}")

# ============================================================
# 4. 训练模型
# ============================================================

import sys
sys.path.append('../../backend')

from core.collaborative_filter import ItemBasedCFRecommender
from core.content_recommender import ContentBasedRecommender
from core.bertopic_enhanced_cf import BERTopicEnhancedCF

# 过滤活跃用户和热门电影 (加速)
MIN_RATINGS = 20
user_counts = pd.Series([i['user_id'] for i in train_interactions]).value_counts()
active_users = user_counts[user_counts >= MIN_RATINGS].index.tolist()

item_counts = pd.Series([i['poem_id'] for i in train_interactions]).value_counts()
popular_items = item_counts[item_counts >= MIN_RATINGS].index.tolist()

train_filtered = [i for i in train_interactions 
                  if i['user_id'] in active_users and i['poem_id'] in popular_items]
poems_filtered = [p for p in poems if p['id'] in popular_items]

print(f"过滤后: {len(train_filtered)}条, {len(active_users)}用户, {len(popular_items)}电影")

# 训练 Item-CF
print("\n训练 Item-CF...")
item_cf = ItemBasedCFRecommender()
item_cf.fit(train_interactions, [p['id'] for p in poems])

# 训练 Content-Based
print("训练 Content-Based...")
content_rec = ContentBasedRecommender()
content_rec.fit(poems)

# 训练 BERTopic-Enhanced CF
print("训练 BERTopic-Enhanced CF...")
bertopic_cf = BERTopicEnhancedCF(item_cf_weight=0.5, user_cf_weight=0.3, topic_weight=0.2)
bertopic_cf.fit(poems_filtered, train_filtered)

print("所有模型训练完成!")

# ============================================================
# 5. 评估函数
# ============================================================

def evaluate_recommender(recommender, train_interactions, test_interactions, poems, top_k=10):
    """评估推荐系统性能"""
    user_train = defaultdict(list)
    for inter in train_interactions:
        user_train[inter['user_id']].append(inter)
    
    user_test = defaultdict(set)
    for inter in test_interactions:
        if inter['rating'] >= 4:
            user_test[inter['user_id']].add(inter['poem_id'])
    
    precisions, recalls, f1_scores, maes = [], [], [], []
    
    for user_id, test_items in user_test.items():
        if user_id not in user_train:
            continue
        
        train_items = user_train[user_id]
        
        try:
            if isinstance(recommender, BERTopicEnhancedCF):
                recs = recommender.recommend(train_items, train_interactions, top_k=top_k)
            elif isinstance(recommender, ItemBasedCFRecommender):
                recs = recommender.recommend(train_items, top_k=top_k)
            elif isinstance(recommender, ContentBasedRecommender):
                user_profile = recommender.build_user_profile(train_items)
                recs = recommender.recommend(user_profile, top_k=top_k)
            else:
                continue
        except:
            continue
        
        if isinstance(recs, list) and len(recs) > 0:
            if isinstance(recs[0], dict):
                rec_items = set([r['poem_id'] for r in recs])
            else:
                rec_items = set(recs[:top_k])
        else:
            rec_items = set()
        
        relevant = test_items
        recommended = rec_items
        
        if len(relevant) == 0:
            continue
        
        hits = len(relevant & recommended)
        precision = hits / len(recommended) if len(recommended) > 0 else 0
        recall = hits / len(relevant) if len(relevant) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    return {
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1_scores),
        'n_users': len(precisions)
    }

# ============================================================
# 6. 实验评估
# ============================================================

K_VALUES = [5, 10, 15, 20]
results = {'Item-CF': {}, 'Content-Based': {}, 'BERTopic-Enhanced CF': {}}

print("\n开始评估...")
for k in K_VALUES:
    print(f"K={k}")
    results['Item-CF'][k] = evaluate_recommender(item_cf, train_filtered, test_interactions, poems_filtered, k)
    results['Content-Based'][k] = evaluate_recommender(content_rec, train_filtered, test_interactions, poems_filtered, k)
    results['BERTopic-Enhanced CF'][k] = evaluate_recommender(bertopic_cf, train_filtered, test_interactions, poems_filtered, k)
    print(f"  Item-CF: P={results['Item-CF'][k]['precision']:.4f}, R={results['Item-CF'][k]['recall']:.4f}, F1={results['Item-CF'][k]['f1']:.4f}")
    print(f"  Content: P={results['Content-Based'][k]['precision']:.4f}, R={results['Content-Based'][k]['recall']:.4f}, F1={results['Content-Based'][k]['f1']:.4f}")
    print(f"  BERTopic: P={results['BERTopic-Enhanced CF'][k]['precision']:.4f}, R={results['BERTopic-Enhanced CF'][k]['recall']:.4f}, F1={results['BERTopic-Enhanced CF'][k]['f1']:.4f}")

# ============================================================
# 7. 可视化
# ============================================================

# 图1: 各指标随K变化曲线
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, metric in zip(axes.flat, ['precision', 'recall', 'f1', 'mae']):
    for algo in results.keys():
        vals = [results[algo][k][metric] for k in K_VALUES]
        ax.plot(K_VALUES, vals, marker='o', linewidth=2, label=algo)
    ax.set_xlabel('K')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()}@K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_VALUES)

plt.tight_layout()
plt.savefig('../../experiment_results/metrics_comparison.png', dpi=150)
plt.show()

# 图2: K=10柱状图
K = 10
x = np.arange(3)
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
algorithms = list(results.keys())
metrics = ['precision', 'recall', 'f1']
colors = ['#2ecc71', '#3498db', '#e74c3c']

for i, (metric, color) in enumerate(zip(metrics, colors)):
    vals = [results[algo][K][metric] for algo in algorithms]
    ax.bar(x + i*width, vals, width, label=metric.capitalize(), color=color)

ax.set_xlabel('Algorithm')
ax.set_ylabel('Score')
ax.set_title(f'Metrics Comparison @ K={K}')
ax.set_xticks(x + width)
ax.set_xticklabels(['Item-CF', 'Content-Based', 'BERTopic-Enhanced'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('../../experiment_results/k10_comparison.png', dpi=150)
plt.show()

# 图3: 热力图
heatmap_data = [[results[algo][K][m] for m in ['precision', 'recall', 'f1']] for algo in algorithms]
heatmap_df = pd.DataFrame(heatmap_data, 
                          index=['Item-CF', 'Content-Based', 'BERTopic-Enhanced'],
                          columns=['Precision', 'Recall', 'F1'])

plt.figure(figsize=(8, 5))
sns.heatmap(heatmap_df, annot=True, fmt='.4f', cmap='RdYlGn', linewidths=0.5)
plt.title(f'Metrics Heatmap @ K={K}')
plt.tight_layout()
plt.savefig('../../experiment_results/heatmap.png', dpi=150)
plt.show()

# ============================================================
# 8. 结果汇总
# ============================================================

print("\n" + "="*60)
print("实验结果汇总 @ K=10")
print("="*60)
print(f"{'Algorithm':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-"*60)
for algo in algorithms:
    m = results[algo][K]
    print(f"{algo:<25} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
print("="*60)

# 保存结果
results_df = []
for algo, k_results in results.items():
    for k, metrics in k_results.items():
        results_df.append({
            'Algorithm': algo, 'K': k,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1']
        })

import os
os.makedirs('../../experiment_results', exist_ok=True)
pd.DataFrame(results_df).to_csv('../../experiment_results/results.csv', index=False)
print("\n结果已保存到 experiment_results/")
```

---

## 文件保存位置

请将上述代码保存为：
```
backend/experiments/algorithm_comparison_experiment.ipynb
```

并确保以下文件存在：
- `data/ml-100k/` - MovieLens-100k数据集
- `backend/core/bertopic_enhanced_cf.py` - BERTopic增强CF
- `backend/core/collaborative_filter.py` - Item-CF
- `backend/core/content_recommender.py` - Content-Based

---

## 运行说明

1. 确保已安装依赖：
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install torch sentence-transformers bertopic jieba
```

2. 下载数据集：
```bash
# MovieLens-100k
wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip -d data/
```

3. 运行Jupyter Notebook：
```bash
jupyter notebook backend/experiments/algorithm_comparison_experiment.ipynb
```
