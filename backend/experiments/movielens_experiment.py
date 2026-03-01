#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MovieLens实验：使用真实数据集验证BERTopic-Enhanced-CF算法

该实验使用MovieLens-100k数据集，包含：
- 943个用户
- 1682部电影
- 10万条评分
- 电影类型和简介文本

实验对比：
- Content-Based (TF-IDF on 电影简介)
- Item-CF (传统协同过滤)
- BERTopic-Enhanced-CF (0.6×评分 + 0.4×主题)
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import warnings

warnings.filterwarnings("ignore")

notebook_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(notebook_dir, "..", ".."))
sys.path.insert(0, project_root)
os.chdir(project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    dataset_path: str = "backend/experiments/data/ml-100k"
    test_ratio: float = 0.2
    n_seeds: int = 5
    top_k: int = 10
    threshold: float = 3.5


def download_movielens(cfg: Config) -> None:
    """下载MovieLens-100k数据集"""
    if os.path.exists(cfg.dataset_path):
        logger.info("MovieLens数据集已存在")
        return
    
    os.makedirs(os.path.dirname(cfg.dataset_path), exist_ok=True)
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = cfg.dataset_path + ".zip"
    
    logger.info("下载MovieLens-100k数据集...")
    urllib.request.urlretrieve(url, zip_path)
    
    logger.info("解压数据集...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(cfg.dataset_path))
    
    os.rename(os.path.join(os.path.dirname(cfg.dataset_path), "ml-100k"), cfg.dataset_path)
    os.remove(zip_path)
    logger.info("数据集准备完成")


def load_movies(cfg: Config) -> List[dict]:
    """加载电影数据"""
    movies = {}
    
    genres_map = {
        "Action": 1, "Adventure": 2, "Animation": 3, "Children": 4, 
        "Comedy": 5, "Crime": 6, "Documentary": 7, "Drama": 8, 
        "Fantasy": 9, "FilmNoir": 10, "Horror": 11, "Musical": 12, 
        "Mystery": 13, "Romance": 14, "SciFi": 15, "Thriller": 16, 
        "War": 17, "Western": 18
    }
    
    item_path = os.path.join(cfg.dataset_path, "u.item")
    with open(item_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            mid = int(parts[0])
            title = parts[1]
            genres = parts[5:23]
            genre_text = " ".join([g for g, v in zip(genres_map.keys(), genres) if v == "1"])
            
            content = f"{title} {genre_text} movie film {title.lower()}"
            
            movies[mid] = {
                "id": mid,
                "title": title,
                "content": content,
                "genres": [g for g, v in zip(genres_map.keys(), genres) if v == "1"]
            }
    
    logger.info(f"加载电影数量: {len(movies)}")
    return list(movies.values())


def load_ratings(cfg: Config) -> List[dict]:
    """加载评分数据"""
    ratings = []
    rating_path = os.path.join(cfg.dataset_path, "u.data")
    
    with open(rating_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            uid = int(parts[0])
            mid = int(parts[1])
            rating = float(parts[2])
            timestamp = int(parts[3])
            
            ratings.append({
                "user_id": uid,
                "poem_id": mid,
                "rating": rating,
                "liked": rating >= cfg.threshold,
                "created_at": datetime.fromtimestamp(timestamp),
            })
    
    logger.info(f"加载评分数量: {len(ratings)}")
    return ratings


def temporal_split(interactions: List[dict], test_ratio: float) -> Tuple[List[dict], List[dict]]:
    """时序划分训练集和测试集"""
    by_user = defaultdict(list)
    for x in interactions:
        by_user[x["user_id"]].append(x)
    
    train, test = [], []
    for _, xs in by_user.items():
        xs.sort(key=lambda v: v["created_at"])
        n = len(xs)
        n_test = max(1, int(n * test_ratio))
        train.extend(xs[: n - n_test])
        test.extend(xs[n - n_test:])
    return train, test


def dcg_at_k(rels: List[int], k: int) -> float:
    rels = np.array(rels[:k])
    if len(rels) == 0:
        return 0.0
    return float(np.sum((2**rels - 1) / np.log2(np.arange(2, len(rels) + 2))))


def ndcg_at_k(recommended: List[int], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    rels = [1 if pid in relevant else 0 for pid in recommended[:k]]
    idcg = dcg_at_k(sorted(rels, reverse=True), k)
    return dcg_at_k(rels, k) / idcg if idcg > 0 else 0.0


def mrr_at_k(recommended: List[int], relevant: set, k: int) -> float:
    for idx, pid in enumerate(recommended[:k], start=1):
        if pid in relevant:
            return 1.0 / idx
    return 0.0


class EnglishCB:
    """英文内容推荐 - 使用sklearn默认英文分词器"""
    
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2))
        self.tfidf_matrix = None
        self.items = None
        
    def fit(self, items):
        self.items = items
        contents = [item.get("content", "") for item in items]
        self.tfidf_matrix = self.vectorizer.fit_transform(contents)
        
    def get_user_profile(self, rated_items, ratings):
        if not rated_items:
            return None
        rated_contents = [item.get("content", "") for item in rated_items]
        rated_vectors = self.vectorizer.transform(rated_contents)
        ratings_arr = np.array(ratings)
        weights = np.abs((ratings_arr - 3.0) / 2.0)
        if weights.sum() > 0:
            user_profile = np.average(rated_vectors.toarray(), axis=0, weights=weights)
        else:
            user_profile = np.mean(rated_vectors.toarray(), axis=0)
        return user_profile
        
    def recommend(self, user_profile, exclude_ids, top_k):
        from sklearn.metrics.pairwise import cosine_similarity
        if user_profile is None or self.tfidf_matrix is None:
            return []
        exclude_ids = exclude_ids or set()
        similarities = cosine_similarity([user_profile], self.tfidf_matrix.toarray())[0]
        results = []
        for i, item in enumerate(self.items):
            if item["id"] not in exclude_ids:
                results.append({"poem_id": item["id"], "score": float(similarities[i])})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


class EnhancedCF:
    """BERTopic Enhanced CF - 核心算法"""

    def __init__(self, rating_weight=0.6, topic_weight=0.4):
        self.rating_weight = rating_weight
        self.topic_weight = topic_weight
        self.item_cf = None
        self.bertopic = None
        self.poems = None
        self.interactions = None
        self.poem_ids = []
        self.poem_id_map = {}
        self.enhanced_similarity = None

    def fit(self, poems, interactions):
        self.poems = poems
        self.interactions = interactions
        self.poem_ids = [p["id"] for p in poems]
        self.poem_id_map = {pid: idx for idx, pid in enumerate(self.poem_ids)}
        
        logger.info("训练Item-CF模型...")
        from backend.core.collaborative_filter import ItemBasedCFRecommender
        self.item_cf = ItemBasedCFRecommender()
        self.item_cf.fit(interactions, self.poem_ids)
        
        logger.info("训练BERTopic模型...")
        from backend.core.bertopic_recommender import BertopicRecommender
        self.bertopic = BertopicRecommender()
        self.bertopic.fit(poems, interactions)
        
        logger.info("计算增强相似度矩阵...")
        self._compute_enhanced_similarity()

    def _min_max_normalize(self, matrix):
        min_val = matrix.min()
        max_val = matrix.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(matrix)
        return (matrix - min_val) / (max_val - min_val)

    def _compute_enhanced_similarity(self):
        from sklearn.metrics.pairwise import cosine_similarity
        
        n_items = len(self.poem_ids)
        rating_sim = self.item_cf.item_similarity
        
        if self.bertopic.topic_matrix is not None:
            topic_sim = cosine_similarity(self.bertopic.topic_matrix)
        else:
            topic_sim = np.zeros((n_items, n_items))
            np.fill_diagonal(topic_sim, 1.0)
        
        if rating_sim.shape != topic_sim.shape:
            min_dim = min(rating_sim.shape[0], topic_sim.shape[0])
            rating_sim = rating_sim[:min_dim, :min_dim]
            topic_sim = topic_sim[:min_dim, :min_dim]
            n_items = min_dim
        
        rating_norm = self._min_max_normalize(rating_sim[:n_items, :n_items])
        topic_norm = self._min_max_normalize(topic_sim[:n_items, :n_items])
        
        self.enhanced_similarity = (
            self.rating_weight * rating_norm + 
            self.topic_weight * topic_norm
        )
        
        logger.info(f"融合完成: {self.rating_weight}×评分 + {self.topic_weight}×主题")

    def recommend(self, user_interactions, all_interactions, top_k=10):
        if self.enhanced_similarity is None:
            return self._popular_fallback(top_k)
        
        exclude_ids = set(i["poem_id"] for i in user_interactions)
        
        user_ratings = np.zeros(len(self.poem_ids))
        for inter in user_interactions:
            if inter["poem_id"] in self.poem_id_map:
                idx = self.poem_id_map[inter["poem_id"]]
                user_ratings[idx] = inter.get("rating", 3.0)
        
        rated_indices = np.where(user_ratings > 0)[0]
        if len(rated_indices) == 0:
            return self._popular_fallback(top_k, exclude_ids)
        
        scores = np.zeros(len(self.poem_ids))
        for i in range(len(self.poem_ids)):
            if user_ratings[i] > 0:
                continue
            
            neighbors = self.enhanced_similarity[i, rated_indices]
            neighbor_ratings = user_ratings[rated_indices]
            
            pos_mask = neighbors > 0
            if pos_mask.sum() > 0:
                scores[i] = np.dot(neighbors[pos_mask], neighbor_ratings[pos_mask]) / (np.abs(neighbors[pos_mask]).sum() + 1e-8)
        
        results = []
        for idx, score in enumerate(scores):
            poem_id = self.poem_ids[idx]
            if poem_id not in exclude_ids and score > 0:
                results.append({"poem_id": poem_id, "score": float(score)})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _popular_fallback(self, top_k, exclude_ids=None):
        exclude_ids = exclude_ids or set()
        poem_scores = defaultdict(float)
        for inter in self.interactions:
            if inter["poem_id"] not in exclude_ids:
                poem_scores[inter["poem_id"]] += inter.get("rating", 3.0)
        
        sorted_items = sorted(poem_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"poem_id": pid, "score": float(score)} for pid, score in sorted_items[:top_k]]

    def predict_rating(self, user_interactions, poem_id):
        if poem_id not in self.poem_id_map or self.enhanced_similarity is None:
            return 3.0
        
        poem_idx = self.poem_id_map[poem_id]
        
        user_ratings = np.zeros(len(self.poem_ids))
        for inter in user_interactions:
            if inter["poem_id"] in self.poem_id_map:
                idx = self.poem_id_map[inter["poem_id"]]
                user_ratings[idx] = inter.get("rating", 3.0)
        
        rated_indices = np.where(user_ratings > 0)[0]
        if len(rated_indices) == 0:
            return 3.0
        
        neighbors = self.enhanced_similarity[poem_idx, rated_indices]
        neighbor_ratings = user_ratings[rated_indices]
        
        pos_mask = neighbors > 0
        if pos_mask.sum() > 0:
            pred = np.dot(neighbors[pos_mask], neighbor_ratings[pos_mask]) / (np.abs(neighbors[pos_mask]).sum() + 1e-8)
            return float(np.clip(pred, 1.0, 5.0))
        
        return 3.0


class Evaluator:
    def __init__(self, items, top_k, threshold):
        self.items = items
        self.top_k = top_k
        self.threshold = threshold
        self.pid_to_idx = {p["id"]: idx for idx, p in enumerate(items)}

    def evaluate(self, models, train_data, test_data):
        user_train = defaultdict(list)
        user_test = defaultdict(list)
        for x in train_data:
            user_train[x["user_id"]].append(x)
        for x in test_data:
            user_test[x["user_id"]].append(x)

        results = {}

        for name, info in models.items():
            rec = info["recommender"]
            kind = info["type"]
            p_list, r_list, f1_list, ndcg_list, mrr_list = [], [], [], [], []
            all_rec_items = set()

            for inter in test_data:
                uid, pid, actual = inter["user_id"], inter["poem_id"], inter["rating"]
                train_items = user_train.get(uid, [])
                if not train_items:
                    continue

                try:
                    if kind == "cb":
                        rated_items = [self.items[self.pid_to_idx[t["poem_id"]]] for t in train_items if t["poem_id"] in self.pid_to_idx]
                        ratings = [t["rating"] for t in train_items if t["poem_id"] in self.pid_to_idx]
                        profile = rec.get_user_profile(rated_items, ratings) if rated_items else None
                        recs = rec.recommend(profile, set(t["poem_id"] for t in train_items), self.top_k) if profile else []
                    elif kind == "cf":
                        recs = rec.recommend(train_items, set(t["poem_id"] for t in train_items), self.top_k)
                    elif kind == "enhanced_cf":
                        recs = rec.recommend(train_items, [], self.top_k)
                    else:
                        recs = []
                except Exception as e:
                    logger.warning(f"{name} 推荐失败: {e}")
                    recs = []

                exclude = set(t["poem_id"] for t in train_items) | set(t["poem_id"] for t in user_test.get(uid, []))
                recs = [r for r in recs if r["poem_id"] not in exclude]

                predicted = [r["poem_id"] for r in recs[: self.top_k]]
                relevant = set(t["poem_id"] for t in user_test[uid] if t["rating"] >= self.threshold)

                if not relevant:
                    continue

                tp = len(set(predicted) & relevant)
                fp = len(predicted) - tp
                fn = len(relevant) - tp

                precision = tp / len(predicted) if predicted else 0.0
                recall = tp / len(relevant) if relevant else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                p_list.append(precision)
                r_list.append(recall)
                f1_list.append(f1)
                ndcg_list.append(ndcg_at_k(predicted, relevant, self.top_k))
                mrr_list.append(mrr_at_k(predicted, relevant, self.top_k))

                all_rec_items.update(predicted)

            coverage = len(all_rec_items) / len(self.items) if self.items else 0

            results[name] = {
                "precision": {"mean": np.mean(p_list), "std": np.std(p_list)} if p_list else {"mean": 0.0},
                "recall": {"mean": np.mean(r_list), "std": np.std(r_list)} if r_list else {"mean": 0.0},
                "f1": {"mean": np.mean(f1_list), "std": np.std(f1_list)} if f1_list else {"mean": 0.0},
                "ndcg": {"mean": np.mean(ndcg_list), "std": np.std(ndcg_list)} if ndcg_list else {"mean": 0.0},
                "mrr": {"mean": np.mean(mrr_list), "std": np.std(mrr_list)} if mrr_list else {"mean": 0.0},
                "coverage": {"mean": coverage},
            }

        return results


def run_once(cfg: Config, movies: List[dict], ratings: List[dict]) -> dict:
    from backend.core.collaborative_filter import ItemBasedCFRecommender

    train, test = temporal_split(ratings, cfg.test_ratio)
    logger.info(f"训练集: {len(train)}, 测试集: {len(test)}")

    cb = EnglishCB()
    cb.fit(movies)

    cf = ItemBasedCFRecommender()
    cf.fit(train, [m["id"] for m in movies])

    enhanced_cf = EnhancedCF(rating_weight=0.6, topic_weight=0.4)
    enhanced_cf.fit(movies, train)

    models = {
        "Content-Based": {"recommender": cb, "type": "cb"},
        "Item-CF": {"recommender": cf, "type": "cf"},
        "BERTopic-Enhanced-CF": {"recommender": enhanced_cf, "type": "enhanced_cf"},
    }

    evaluator = Evaluator(movies, cfg.top_k, cfg.threshold)
    return evaluator.evaluate(models, train, test)


def aggregate(results: List[dict]) -> dict:
    metrics = ["precision", "recall", "f1", "ndcg", "mrr", "coverage"]
    out = {}

    for m in metrics:
        values = [r[m]["mean"] for r in results if m in r]
        if values:
            out[m] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }
    return out


def main() -> int:
    cfg = Config()
    seeds = [42, 123, 456, 789, 1024][: cfg.n_seeds]

    download_movielens(cfg)
    movies = load_movies(cfg)
    ratings = load_ratings(cfg)
    
    logger.info(f"数据集统计: {len(movies)}部电影, {len(ratings)}条评分")

    all_results = []
    for s in seeds:
        logger.info(f"=== Seed {s} ===")
        random.seed(s)
        np.random.seed(s)
        random.shuffle(ratings)
        all_results.append(run_once(cfg, movies, ratings))

    agg = aggregate(all_results)
    output = {
        "config": cfg.__dict__,
        "seeds": seeds,
        "aggregate": agg,
        "all_results": all_results,
        "notes": {
            "purpose": "MovieLens-100k数据集验证BERTopic-Enhanced-CF算法",
            "dataset": "MovieLens-100k",
            "n_items": len(movies),
            "n_ratings": len(ratings),
        },
    }

    out_path = "backend/experiments/movielens_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"结果已保存: {out_path}")

    print("\n=== MovieLens Results ===")
    for name, m in agg.items():
        print(f"{name}: {m.get('mean', 0):.4f} (±{m.get('std', 0):.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
