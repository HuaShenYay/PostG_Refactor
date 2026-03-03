"""
冷启动推荐算法对比实验

对比三种算法在冷启动场景下的效果:
1. SentenceTransformerEnhancedCF (本系统核心算法) - 语义+协同过滤混合
2. Item-CF (协同过滤)
3. Content-Based (内容推荐)

冷启动定义:
- 极冷启动: 1条评分
- 冷启动: 2-3条评分
- 轻度冷启动: 4-10条评分
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import time
import json

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ========== 数据生成 ==========

def generate_poems(n=200):
    topics = ["山水", "思乡", "边塞", "离别", "咏史", "田园", "闺怨", "怀古", "哲理", "人生", "送别", "咏物"]
    genres = ["诗", "词", "曲", "赋"]
    dynasties = ["唐", "宋", "元", "明", "清"]
    
    poems = []
    for i in range(n):
        topic = topics[i % len(topics)]
        poems.append({
            "id": i + 1,
            "content": f"这是一首关于{topic}的诗歌，表达了诗人的情感和思想。" * 3,
            "title": f"《{topic}》其一",
            "author": f"作者{i % 10}",
            "dynasty": dynasties[i % len(dynasties)],
            "genre_type": genres[i % len(genres)],
            "topic": topic,
        })
    return poems


def generate_interactions(poems, cold_users=100, low_users=50, active_users=30):
    interactions = []
    user_id = 1
    
    # 极冷启动用户: 1条评分
    for _ in range(cold_users // 3):
        poem = np.random.choice(poems)
        rating = np.random.choice([3, 4, 5], p=[0.2, 0.3, 0.5])
        interactions.append({
            "user_id": user_id,
            "poem_id": poem["id"],
            "rating": rating,
            "created_at": time.time(),
        })
        user_id += 1
    
    # 冷启动用户: 2-3条评分
    for _ in range(cold_users - cold_users // 3):
        n_ratings = np.random.choice([2, 3], p=[0.5, 0.5])
        rated = np.random.choice(poems, size=n_ratings, replace=False)
        for p in rated:
            rating = np.random.choice([3, 4, 5], p=[0.2, 0.3, 0.5])
            interactions.append({
                "user_id": user_id,
                "poem_id": p["id"],
                "rating": rating,
                "created_at": time.time(),
            })
        user_id += 1
    
    # 轻度冷启动用户: 4-10条评分
    for _ in range(low_users):
        n_ratings = np.random.randint(4, 11)
        rated = np.random.choice(poems, size=n_ratings, replace=False)
        for p in rated:
            rating = np.random.choice([3, 4, 5], p=[0.3, 0.4, 0.3])
            interactions.append({
                "user_id": user_id,
                "poem_id": p["id"],
                "rating": rating,
                "created_at": time.time(),
            })
        user_id += 1
    
    # 活跃用户: 15-30条评分
    for _ in range(active_users):
        n_ratings = np.random.randint(15, 31)
        rated = np.random.choice(poems, size=n_ratings, replace=False)
        for p in rated:
            rating = np.random.choice([3, 4, 5], p=[0.3, 0.4, 0.3])
            interactions.append({
                "user_id": user_id,
                "poem_id": p["id"],
                "rating": rating,
                "created_at": time.time(),
            })
        user_id += 1
    
    return interactions


def split_train_test(interactions, test_ratio=0.3, cold_threshold=3):
    user_interactions = defaultdict(list)
    for inter in interactions:
        user_interactions[inter["user_id"]].append(inter)
    
    train, test = [], []
    
    for user_id, its in user_interactions.items():
        sorted_its = sorted(its, key=lambda x: x["created_at"])
        n = len(sorted_its)
        
        if n <= cold_threshold:
            n_train = max(1, n - 1)
        else:
            n_train = max(cold_threshold + 1, int(n * (1 - test_ratio)))
        
        train.extend(sorted_its[:n_train])
        test.extend(sorted_its[n_train:])
    
    train_poem_ids = set(i["poem_id"] for i in train)
    train = [i for i in train if i["poem_id"] in train_poem_ids]
    test = [i for i in test if i["poem_id"] in train_poem_ids]
    
    return train, test


# ========== 算法实现 ==========

class ItemCF:
    def __init__(self):
        self.poem_ids = []
        self.poem_id_map = {}
        self.rating_matrix = None
        self.item_sim = None
        
    def fit(self, poems, interactions):
        self.poem_ids = [p["id"] for p in poems]
        self.poem_id_map = {pid: i for i, pid in enumerate(self.poem_ids)}
        
        users = sorted({i["user_id"] for i in interactions})
        user_map = {uid: i for i, uid in enumerate(users)}
        
        self.rating_matrix = np.zeros((len(users), len(self.poem_ids)))
        
        for inter in interactions:
            u = user_map.get(inter["user_id"])
            p = self.poem_id_map.get(inter["poem_id"])
            if u is not None and p is not None:
                self.rating_matrix[u, p] = inter["rating"]
        
        R = self.rating_matrix
        mask = R > 0
        col_means = R.sum(axis=0) / (mask.sum(axis=0) + 1e-8)
        R_centered = np.where(mask, R - col_means, 0)
        self.item_sim = cosine_similarity(R_centered.T)
        np.fill_diagonal(self.item_sim, 0)
        
    def recommend(self, user_interactions, exclude_ids=None, top_k=10):
        exclude_ids = set(exclude_ids or [])
        
        user_ratings = np.zeros(len(self.poem_ids))
        for inter in user_interactions:
            p = self.poem_id_map.get(inter["poem_id"])
            if p is not None:
                user_ratings[p] = inter["rating"]
        
        rated = np.where(user_ratings > 0)[0]
        if rated.size == 0:
            return [{"poem_id": pid, "score": 0.5} for pid in self.poem_ids[:top_k] if pid not in exclude_ids]
        
        sims = self.item_sim[:, rated]
        ratings = user_ratings[rated]
        
        scores = (sims * ratings).sum(axis=1) / (np.abs(sims).sum(axis=1) + 1e-8)
        
        for pid in exclude_ids:
            p = self.poem_id_map.get(pid)
            if p is not None:
                scores[p] = -np.inf
        scores[rated] = -np.inf
        
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [{"poem_id": self.poem_ids[i], "score": float(scores[i])} for i in top_idx if scores[i] > -np.inf]


class ContentBased:
    def __init__(self):
        self.poem_ids = []
        self.poem_id_map = {}
        self.item_sim = None
        
    def fit(self, poems, interactions=None):
        self.poem_ids = [p["id"] for p in poems]
        self.poem_id_map = {pid: i for i, pid in enumerate(self.poem_ids)}
        
        contents = [p.get("content", "") for p in poems]
        tfidf = TfidfVectorizer(max_features=500)
        tfidf_matrix = tfidf.fit_transform(contents)
        self.item_sim = cosine_similarity(tfidf_matrix)
        
    def recommend(self, user_interactions, exclude_ids=None, top_k=10):
        exclude_ids = set(exclude_ids or [])
        
        if not user_interactions:
            return [{"poem_id": pid, "score": 0.5} for pid in self.poem_ids[:top_k] if pid not in exclude_ids]
        
        user_profile = np.zeros(len(self.poem_ids))
        for inter in user_interactions:
            p = self.poem_id_map.get(inter["poem_id"])
            if p is not None:
                user_profile += inter["rating"] * self.item_sim[p]
        
        rated = set(self.poem_id_map.get(inter["poem_id"]) for inter in user_interactions)
        
        for pid in exclude_ids:
            p = self.poem_id_map.get(pid)
            if p is not None:
                user_profile[p] = -np.inf
        
        for idx in rated:
            if idx is not None:
                user_profile[idx] = -np.inf
        
        top_idx = np.argsort(user_profile)[::-1][:top_k]
        return [{"poem_id": self.poem_ids[i], "score": float(user_profile[i])} for i in top_idx if user_profile[i] > -np.inf]


class SentenceTransformerEnhancedCF:
    def __init__(
        self,
        cf_weight=0.5,
        semantic_weight=0.5,
        cold_weights=(0.80, 0.20),
        low_weights=(0.60, 0.40),
        active_weights=(0.40, 0.60),
    ):
        self.cf_weight = cf_weight
        self.semantic_weight = semantic_weight
        self.cold_weights = cold_weights
        self.low_weights = low_weights
        self.active_weights = active_weights
        
        self.poem_ids = []
        self.poem_id_map = {}
        self.rating_matrix = None
        self.item_embeddings = None
        self.enhanced_sim = None
        
    def fit(self, poems, interactions):
        self.poem_ids = [p["id"] for p in poems]
        self.poem_id_map = {pid: i for i, pid in enumerate(self.poem_ids)}
        
        users = sorted({i["user_id"] for i in interactions})
        user_map = {uid: i for i, uid in enumerate(users)}
        
        self.rating_matrix = np.zeros((len(users), len(self.poem_ids)))
        
        for inter in interactions:
            u = user_map.get(inter["user_id"])
            p = self.poem_id_map.get(inter["poem_id"])
            if u is not None and p is not None:
                self.rating_matrix[u, p] = inter["rating"]
        
        contents = [p.get("content", "") for p in poems]
        tfidf = TfidfVectorizer(max_features=500)
        self.item_embeddings = tfidf.fit_transform(contents).toarray()
        self.item_embeddings = self.item_embeddings / (np.linalg.norm(self.item_embeddings, axis=1, keepdims=True) + 1e-8)
        
        R = self.rating_matrix
        mask = R > 0
        col_means = R.sum(axis=0) / (mask.sum(axis=0) + 1e-8)
        R_centered = np.where(mask, R - col_means, 0)
        cf_sim = cosine_similarity(R_centered.T)
        
        sem_sim = cosine_similarity(self.item_embeddings)
        
        def minmax(m):
            return (m - m.min()) / (m.max() - m.min() + 1e-8)
        
        self.enhanced_sim = self.cf_weight * minmax(cf_sim) + self.semantic_weight * minmax(sem_sim)
        
    def recommend(self, user_interactions, exclude_ids=None, top_k=10):
        exclude_ids = set(exclude_ids or [])
        
        user_ratings = np.zeros(len(self.poem_ids))
        for inter in user_interactions:
            p = self.poem_id_map.get(inter["poem_id"])
            if p is not None:
                user_ratings[p] = inter["rating"]
        
        n_rated = int((user_ratings > 0).sum())
        
        if n_rated <= 1:
            sem_w, cf_w = self.cold_weights
        elif n_rated <= 3:
            sem_w, cf_w = self.cold_weights
        elif n_rated <= 10:
            sem_w, cf_w = self.low_weights
        else:
            sem_w, cf_w = self.active_weights
        
        sem_scores = None
        if self.item_embeddings is not None:
            user_profile = np.zeros(self.item_embeddings.shape[1])
            for inter in user_interactions:
                p = self.poem_id_map.get(inter["poem_id"])
                if p is not None:
                    user_profile += inter["rating"] * self.item_embeddings[p]
            
            norm = np.linalg.norm(user_profile)
            if norm > 1e-8:
                user_profile = user_profile / norm
                sem_scores = self.item_embeddings @ user_profile
        
        cf_scores = None
        rated = np.where(user_ratings > 0)[0]
        if rated.size > 0 and self.enhanced_sim is not None:
            sims = self.enhanced_sim[:, rated]
            ratings = user_ratings[rated]
            cf_scores = (sims * ratings).sum(axis=1) / (np.abs(sims).sum(axis=1) + 1e-8)
        
        def norm_(arr):
            if arr is None:
                return None
            valid = arr[arr > -np.inf]
            if valid.size == 0:
                return None
            return (arr - valid.min()) / (valid.max() - valid.min() + 1e-8)
        
        sem_scores = norm_(sem_scores)
        cf_scores = norm_(cf_scores)
        
        combined = np.zeros(len(self.poem_ids))
        total_w = 0
        
        if sem_scores is not None:
            combined += sem_w * sem_scores
            total_w += sem_w
        if cf_scores is not None:
            combined += cf_w * cf_scores
            total_w += cf_w
        
        if total_w > 0:
            combined = combined / total_w
        
        for pid in exclude_ids:
            p = self.poem_id_map.get(pid)
            if p is not None:
                combined[p] = -np.inf
        combined[rated] = -np.inf
        
        top_idx = np.argsort(combined)[::-1][:top_k]
        return [{"poem_id": self.poem_ids[i], "score": float(combined[i])} for i in top_idx if combined[i] > -np.inf]


# ========== 评估 ==========

def evaluate(recommender, train_interactions, test_interactions, poems, top_k=10, rating_threshold=3):
    user_train = defaultdict(list)
    for inter in train_interactions:
        user_train[inter["user_id"]].append(inter)
    
    user_test = defaultdict(dict)
    for inter in test_interactions:
        if inter["rating"] >= rating_threshold:
            user_test[inter["user_id"]][inter["poem_id"]] = inter["rating"]
    
    test_users = list(user_test.keys())
    
    if not test_users:
        return {"precision": 0, "recall": 0, "f1": 0, "mae": 0}
    
    precisions, recalls, f1s, maes = [], [], [], []
    
    for user_id in test_users:
        seen = set(i["poem_id"] for i in user_train[user_id])
        relevant = set(user_test[user_id].keys())
        
        if not relevant:
            continue
        
        try:
            recs = recommender.recommend(user_train[user_id], exclude_ids=seen, top_k=top_k)
        except:
            continue
        
        rec_items = [r["poem_id"] for r in recs]
        
        hits = sum(1 for pid in rec_items if pid in relevant)
        
        precision = hits / top_k
        recall = hits / len(relevant) if len(relevant) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # MAE
        errors = []
        for rec in recs:
            pid = rec["poem_id"]
            if pid in user_test[user_id]:
                actual_rating = user_test[user_id][pid]
                pred_score = rec.get("score", 0.5) * 5
                errors.append(abs(pred_score - actual_rating))
        
        mae = np.mean(errors) if errors else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        maes.append(mae)
    
    return {
        "precision": np.mean(precisions) if precisions else 0,
        "recall": np.mean(recalls) if recalls else 0,
        "f1": np.mean(f1s) if f1s else 0,
        "mae": np.mean(maes) if maes else 0,
    }


def get_cold_users(train_interactions):
    user_counts = Counter(i["user_id"] for i in train_interactions)
    
    very_cold = [u for u, c in user_counts.items() if c <= 1]
    cold = [u for u, c in user_counts.items() if 1 < c <= 3]
    low = [u for u, c in user_counts.items() if 3 < c <= 10]
    active = [u for u, c in user_counts.items() if c > 10]
    
    return {
        "very_cold": very_cold,
        "cold": cold,
        "low_activity": low,
        "active": active,
    }


# ========== 主实验 ==========

def run_experiment():
    print("=" * 70)
    print("冷启动推荐算法对比实验")
    print("=" * 70)
    
    # 1. 生成数据
    print("\n【1. 生成数据】")
    np.random.seed(42)
    poems = generate_poems(n=200)
    interactions = generate_interactions(poems, cold_users=100, low_users=50, active_users=30)
    
    print(f"  诗歌: {len(poems)}")
    print(f"  用户: {len(set(i['user_id'] for i in interactions))}")
    print(f"  交互: {len(interactions)}")
    
    user_counts = Counter(i["user_id"] for i in interactions)
    print(f"  极冷用户(1条): {sum(1 for c in user_counts.values() if c <= 1)}")
    print(f"  冷启动(2-3条): {sum(1 for c in user_counts.values() if 1 < c <= 3)}")
    print(f"  轻度(4-10条): {sum(1 for c in user_counts.values() if 3 < c <= 10)}")
    print(f"  活跃(>10条): {sum(1 for c in user_counts.values() if c > 10)}")
    
    # 2. 划分数据
    print("\n【2. 划分训练/测试集】")
    train, test = split_train_test(interactions, test_ratio=0.3)
    
    train_poem_ids = set(i["poem_id"] for i in train)
    poems_filtered = [p for p in poems if p["id"] in train_poem_ids]
    
    print(f"  训练集: {len(train)}")
    print(f"  测试集: {len(test)}")
    
    user_cold = get_cold_users(train)
    print(f"  训练集冷用户(≤3条): {len(user_cold['very_cold']) + len(user_cold['cold'])}")
    
    # 3. 训练模型
    print("\n【3. 训练模型】")
    
    print("  训练 Item-CF...")
    item_cf = ItemCF()
    item_cf.fit(poems_filtered, train)
    
    print("  训练 Content-Based...")
    content_cb = ContentBased()
    content_cb.fit(poems_filtered, train)
    
    print("  训练 SentenceTransformerEnhancedCF...")
    sem_cf = SentenceTransformerEnhancedCF(
        cf_weight=0.5,
        semantic_weight=0.5,
        cold_weights=(0.80, 0.20),
        low_weights=(0.60, 0.40),
        active_weights=(0.40, 0.60),
    )
    sem_cf.fit(poems_filtered, train)
    
    # 4. 评估
    print("\n【4. 评估结果】")
    print("-" * 70)
    
    models = {
        "Item-CF": item_cf,
        "Content-Based": content_cb,
        "SentenceTransformerEnhancedCF": sem_cf,
    }
    
    K_VALUES = [5, 10, 15, 20]
    
    cold_levels = [
        ("极冷启动(1条)", "very_cold"),
        ("冷启动(2-3条)", "cold"),
        ("轻度冷启动(4-10条)", "low_activity"),
    ]
    
    all_results = {}
    
    for level_name, level_key in cold_levels:
        print(f"\n--- {level_name} ---")
        user_list = user_cold[level_key]
        
        if not user_list:
            print("  无用户数据")
            continue
        
        test_filtered = [i for i in test if i["user_id"] in user_list and i["rating"] >= 3]
        
        if not test_filtered:
            print("  无测试数据")
            continue
        
        results = {}
        for k in K_VALUES:
            print(f"\n  K = {k}:")
            for name, model in models.items():
                metrics = evaluate(model, train, test_filtered, poems_filtered, top_k=k)
                results.setdefault(name, {})[k] = metrics
                print(f"    {name:30s} P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f} MAE={metrics['mae']:.4f}")
        
        all_results[level_name] = results
    
    # 5. 汇总
    print("\n" + "=" * 70)
    print("【5. 汇总对比】")
    print("-" * 70)
    
    print(f"\n{'算法':<35} {'冷启动等级':<18} {'P@10':>8} {'R@10':>8} {'F1@10':>8} {'MAE@10':>8}")
    print("-" * 80)
    
    for level_name in all_results:
        for name in models:
            if name in all_results[level_name] and 10 in all_results[level_name][name]:
                m = all_results[level_name][name][10]
                print(f"{name:<35} {level_name:<18} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['mae']:>8.4f}")
    
    # 6. 可视化
    print("\n【6. 生成图表】")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics_list = [("precision", "Precision@K"), ("recall", "Recall@K"), ("f1", "F1@K"), ("mae", "MAE@K")]
    
    colors = {"Item-CF": "#e74c3c", "Content-Based": "#3498db", "SentenceTransformerEnhancedCF": "#2ecc71"}
    
    cold_key = "冷启动(2-3条)"
    if cold_key in all_results:
        for idx, (metric, title) in enumerate(metrics_list):
            ax = axes[idx // 2, idx % 2]
            for name in models:
                if name in all_results[cold_key]:
                    vals = [all_results[cold_key][name][k][metric] for k in K_VALUES]
                    ax.plot(K_VALUES, vals, marker="o", linewidth=2, label=name, color=colors[name])
            
            ax.set_xlabel("K")
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f"{title} ({cold_key})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(K_VALUES)
    
    plt.tight_layout()
    plt.savefig("cold_start_comparison.png", dpi=150)
    print("  保存: cold_start_comparison.png")
    
    with open("cold_start_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("  保存: cold_start_results.json")
    
    return all_results


if __name__ == "__main__":
    results = run_experiment()
