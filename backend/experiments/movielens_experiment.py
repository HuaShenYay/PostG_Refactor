#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MovieLens-100K Top-N 对比实验（本科毕业设计友好版）。

对比方法：
- CB（基于标题+类型文本的 TF-IDF）
- Item-CF
- User-CF
- BERT-Enhanced（BERTopicEnhancedCF 三路融合）

输出：
- backend/experiments/movielens_results.json
- backend/experiments/movielens_results.csv
- backend/experiments/movielens_precision_plot.png
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    dataset_path: str = "backend/experiments/data/ml-100k"
    test_ratio: float = 0.2
    random_seed: int = 42
    top_ks: Tuple[int, int] = (5, 10)
    positive_threshold: float = 4.0
    sample_users_for_case: int = 2


class ContentBasedRecommender:
    """基于文本（标题+类型）的简易 CB 推荐器。"""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8000)
        self.item_vectors = None
        self.items = []
        self.id_to_idx = {}

    def fit(self, items: List[dict]) -> None:
        self.items = items
        self.id_to_idx = {it["id"]: idx for idx, it in enumerate(items)}
        corpus = [it.get("content", "") for it in items]
        self.item_vectors = self.vectorizer.fit_transform(corpus)

    def recommend(self, user_interactions: List[dict], exclude_ids: Set[int], top_k: int) -> List[dict]:
        from sklearn.metrics.pairwise import cosine_similarity

        if self.item_vectors is None:
            return []

        rated = [x for x in user_interactions if x["poem_id"] in self.id_to_idx]
        if not rated:
            return []

        docs = [self.items[self.id_to_idx[x["poem_id"]]].get("content", "") for x in rated]
        vecs = self.vectorizer.transform(docs).toarray()
        ratings = np.array([x.get("rating", 3.0) for x in rated], dtype=float)
        weights = np.clip(ratings - 2.5, 0.0, None)

        if weights.sum() > 0:
            profile = np.average(vecs, axis=0, weights=weights)
        else:
            profile = np.mean(vecs, axis=0)

        sims = cosine_similarity([profile], self.item_vectors)[0]
        recs = []
        for idx, score in enumerate(sims):
            item_id = self.items[idx]["id"]
            if item_id in exclude_ids:
                continue
            recs.append({"poem_id": item_id, "score": float(score)})

        recs.sort(key=lambda x: x["score"], reverse=True)
        return recs[:top_k]


class UserBasedCFRecommender:
    """简化版 User-CF（Pearson + Top-K 邻居）。"""

    def __init__(self, k_neighbors: int = 40):
        self.k_neighbors = k_neighbors
        self.rating_matrix = None
        self.user_id_to_idx = {}
        self.item_id_to_idx = {}
        self.idx_to_item_id = {}
        self.user_similarity = None

    def fit(self, interactions: List[dict], item_ids: Sequence[int]) -> None:
        user_ids = sorted({x["user_id"] for x in interactions})
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        self.item_id_to_idx = {pid: idx for idx, pid in enumerate(item_ids)}
        self.idx_to_item_id = {idx: pid for pid, idx in self.item_id_to_idx.items()}

        self.rating_matrix = np.zeros((len(user_ids), len(item_ids)), dtype=float)
        for inter in interactions:
            u_idx = self.user_id_to_idx[inter["user_id"]]
            p_idx = self.item_id_to_idx.get(inter["poem_id"])
            if p_idx is not None:
                self.rating_matrix[u_idx, p_idx] = inter.get("rating", 3.0)

        self.user_similarity = self._compute_user_similarity()

    def _compute_user_similarity(self) -> np.ndarray:
        n_users = self.rating_matrix.shape[0]
        sim = np.zeros((n_users, n_users), dtype=float)
        for i in range(n_users):
            sim[i, i] = 1.0
            for j in range(i + 1, n_users):
                mask = (self.rating_matrix[i] > 0) & (self.rating_matrix[j] > 0)
                if mask.sum() == 0:
                    score = 0.0
                else:
                    vi = self.rating_matrix[i, mask]
                    vj = self.rating_matrix[j, mask]
                    vi = vi - vi.mean()
                    vj = vj - vj.mean()
                    denom = (np.sqrt((vi**2).sum()) * np.sqrt((vj**2).sum())) + 1e-8
                    score = float((vi * vj).sum() / denom)
                sim[i, j] = score
                sim[j, i] = score
        return sim

    def recommend(self, user_interactions: List[dict], exclude_ids: Set[int], top_k: int) -> List[dict]:
        if not user_interactions or self.rating_matrix is None:
            return []

        uid = user_interactions[0]["user_id"]
        target_idx = self.user_id_to_idx.get(uid)
        if target_idx is None:
            return []

        sims = self.user_similarity[target_idx].copy()
        sims[target_idx] = -np.inf
        neighbor_indices = np.argsort(sims)[-self.k_neighbors :]
        neighbors = [(idx, sims[idx]) for idx in neighbor_indices if sims[idx] > 0]
        if not neighbors:
            return []

        recs = []
        for item_idx in range(self.rating_matrix.shape[1]):
            item_id = self.idx_to_item_id[item_idx]
            if item_id in exclude_ids:
                continue

            weighted_sum = 0.0
            sim_sum = 0.0
            for n_idx, n_sim in neighbors:
                r = self.rating_matrix[n_idx, item_idx]
                if r > 0:
                    weighted_sum += n_sim * r
                    sim_sum += abs(n_sim)

            if sim_sum > 0:
                recs.append({"poem_id": item_id, "score": float(weighted_sum / (sim_sum + 1e-8))})

        recs.sort(key=lambda x: x["score"], reverse=True)
        return recs[:top_k]


def download_movielens_if_needed(cfg: Config) -> None:
    if os.path.exists(cfg.dataset_path):
        LOGGER.info("MovieLens 数据集已存在：%s", cfg.dataset_path)
        return

    os.makedirs(os.path.dirname(cfg.dataset_path), exist_ok=True)
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = cfg.dataset_path + ".zip"

    LOGGER.info("下载 MovieLens-100K ...")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as exc:
        raise RuntimeError(
            "无法自动下载 MovieLens-100K，请手动下载 ml-100k.zip 并解压到 "
            f"{cfg.dataset_path}。原始错误: {exc}"
        ) from exc

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(os.path.dirname(cfg.dataset_path))
    os.remove(zip_path)
    LOGGER.info("下载并解压完成")


def load_movies(cfg: Config) -> List[dict]:
    genre_names = [
        "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
        "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
        "Romance", "Sci-Fi", "Thriller", "War", "Western",
    ]

    path = os.path.join(cfg.dataset_path, "u.item")
    movies = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            movie_id = int(parts[0])
            title = parts[1]
            flags = parts[5:24]
            genres = [g for g, v in zip(genre_names, flags) if v == "1"]
            content = f"{title} {' '.join(genres)} movie film"
            movies.append({"id": movie_id, "title": title, "genres": genres, "content": content})

    LOGGER.info("加载电影数量: %d", len(movies))
    return movies


def load_ratings(cfg: Config) -> List[dict]:
    path = os.path.join(cfg.dataset_path, "u.data")
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            user_id, movie_id, rating, ts = line.strip().split("\t")
            out.append(
                {
                    "user_id": int(user_id),
                    "poem_id": int(movie_id),
                    "rating": float(rating),
                    "created_at": datetime.fromtimestamp(int(ts)),
                }
            )

    LOGGER.info("加载评分数量: %d", len(out))
    return out


def train_test_split_by_user(ratings: List[dict], test_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    rng = random.Random(seed)
    by_user: Dict[int, List[dict]] = defaultdict(list)
    for x in ratings:
        by_user[x["user_id"]].append(x)

    train, test = [], []
    for uid, xs in by_user.items():
        xs = xs.copy()
        rng.shuffle(xs)
        n_test = max(1, int(len(xs) * test_ratio))
        test.extend(xs[:n_test])
        train.extend(xs[n_test:])

    LOGGER.info("按用户随机划分: train=%d, test=%d", len(train), len(test))
    return train, test


def build_user_index(interactions: Iterable[dict]) -> Dict[int, List[dict]]:
    by_user: Dict[int, List[dict]] = defaultdict(list)
    for x in interactions:
        by_user[x["user_id"]].append(x)
    return by_user


def evaluate_topn(method_name: str, recommender, user_train: Dict[int, List[dict]], user_test: Dict[int, List[dict]], ks: Sequence[int], positive_threshold: float, all_train_interactions: List[dict]) -> dict:
    metrics = {k: {"precision": [], "recall": [], "hit": []} for k in ks}

    users = sorted(set(user_train.keys()) & set(user_test.keys()))
    for uid in users:
        train_inter = user_train.get(uid, [])
        test_inter = user_test.get(uid, [])

        relevant = {x["poem_id"] for x in test_inter if x.get("rating", 0.0) >= positive_threshold}
        if not train_inter or not relevant:
            continue

        exclude_ids = {x["poem_id"] for x in train_inter}
        max_k = max(ks)

        if method_name == "BERT-Enhanced":
            recs = recommender.recommend(train_inter, all_train_interactions, top_k=max_k)
        else:
            recs = recommender.recommend(train_inter, exclude_ids=exclude_ids, top_k=max_k)

        ranked = [x["poem_id"] for x in recs]

        for k in ks:
            topk = ranked[:k]
            hits = len(set(topk) & relevant)
            metrics[k]["precision"].append(hits / k)
            metrics[k]["recall"].append(hits / len(relevant))
            metrics[k]["hit"].append(1.0 if hits > 0 else 0.0)

    summary = {}
    for k in ks:
        summary[f"Precision@{k}"] = float(np.mean(metrics[k]["precision"])) if metrics[k]["precision"] else 0.0
        summary[f"Recall@{k}"] = float(np.mean(metrics[k]["recall"])) if metrics[k]["recall"] else 0.0
        summary[f"Hit@{k}"] = float(np.mean(metrics[k]["hit"])) if metrics[k]["hit"] else 0.0
    return summary


def get_case_studies(methods: Dict[str, object], movies: List[dict], user_train: Dict[int, List[dict]], user_ids: List[int], all_train_interactions: List[dict], top_k: int = 5) -> List[dict]:
    id_to_title = {m["id"]: m["title"] for m in movies}
    out = []
    for uid in user_ids:
        train_inter = user_train.get(uid, [])
        if not train_inter:
            continue

        exclude_ids = {x["poem_id"] for x in train_inter}
        row = {"user_id": uid, "recommendations": {}}
        for name, model in methods.items():
            if name == "BERT-Enhanced":
                recs = model.recommend(train_inter, all_train_interactions, top_k=top_k)
            else:
                recs = model.recommend(train_inter, exclude_ids=exclude_ids, top_k=top_k)
            row["recommendations"][name] = [id_to_title.get(x["poem_id"], str(x["poem_id"])) for x in recs[:top_k]]
        out.append(row)
    return out


def write_csv(rows: List[dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_precision(rows: List[dict], path: str) -> None:
    import matplotlib.pyplot as plt

    methods = [r["method"] for r in rows]
    p10 = [r["Precision@10"] for r in rows]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(methods, p10)
    plt.title("MovieLens-100K Precision@10 Comparison")
    plt.ylabel("Precision@10")
    plt.xticks(rotation=15)
    for b, v in zip(bars, p10):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def run_experiment(cfg: Config, skip_bertopic: bool = False) -> dict:
    from backend.core.bertopic_enhanced_cf import BERTopicEnhancedCF
    from backend.core.collaborative_filter import ItemBasedCFRecommender

    download_movielens_if_needed(cfg)
    movies = load_movies(cfg)
    ratings = load_ratings(cfg)

    train, test = train_test_split_by_user(ratings, cfg.test_ratio, cfg.random_seed)
    user_train = build_user_index(train)
    user_test = build_user_index(test)

    item_ids = [m["id"] for m in movies]

    cb = ContentBasedRecommender()
    cb.fit(movies)

    item_cf = ItemBasedCFRecommender()
    item_cf.fit(train, item_ids)

    user_cf = UserBasedCFRecommender()
    user_cf.fit(train, item_ids)

    methods = {
        "CB": cb,
        "Item-CF": item_cf,
        "User-CF": user_cf,
    }

    if not skip_bertopic:
        bert_enhanced = BERTopicEnhancedCF(item_cf_weight=0.5, user_cf_weight=0.3, topic_weight=0.2)
        bert_enhanced.fit(movies, train)
        methods["BERT-Enhanced"] = bert_enhanced
    else:
        LOGGER.warning("已跳过 BERT-Enhanced（--skip-bertopic）")

    result_rows = []
    for method_name, rec in methods.items():
        m = evaluate_topn(method_name, rec, user_train, user_test, cfg.top_ks, cfg.positive_threshold, train)
        result_rows.append({"method": method_name, **m})

    result_rows.sort(key=lambda x: x["Precision@10"], reverse=True)

    sample_users = sorted(user_test.keys())[: cfg.sample_users_for_case]
    case_studies = get_case_studies(methods, movies, user_train, sample_users, train, top_k=5)

    csv_path = "backend/experiments/movielens_results.csv"
    json_path = "backend/experiments/movielens_results.json"
    png_path = "backend/experiments/movielens_precision_plot.png"

    write_csv(result_rows, csv_path)
    plot_precision(result_rows, png_path)

    payload = {
        "config": {
            "dataset_path": cfg.dataset_path,
            "test_ratio": cfg.test_ratio,
            "random_seed": cfg.random_seed,
            "top_ks": list(cfg.top_ks),
            "positive_threshold": cfg.positive_threshold,
        },
        "dataset_stats": {
            "n_users": len({x["user_id"] for x in ratings}),
            "n_items": len(movies),
            "n_ratings": len(ratings),
            "n_train": len(train),
            "n_test": len(test),
        },
        "results": result_rows,
        "case_studies": case_studies,
        "artifacts": {
            "csv": csv_path,
            "plot": png_path,
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    LOGGER.info("实验完成，结果文件：%s", json_path)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MovieLens-100K Top-N 对比实验")
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--positive-threshold", type=float, default=4.0)
    p.add_argument("--skip-bertopic", action="store_true", help="跳过 BERT-Enhanced（便于快速验证）")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    cfg = Config(
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        positive_threshold=args.positive_threshold,
    )

    payload = run_experiment(cfg, skip_bertopic=args.skip_bertopic)

    print("\n=== MovieLens-100K Top-N Results ===")
    for row in payload["results"]:
        print(
            f"{row['method']:<14} | "
            f"P@5={row['Precision@5']:.4f} R@5={row['Recall@5']:.4f} H@5={row['Hit@5']:.4f} | "
            f"P@10={row['Precision@10']:.4f} R@10={row['Recall@10']:.4f} H@10={row['Hit@10']:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
