#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Platinum(platium)实验：对齐 EXPERIMENT_DESIGN 的对比评估脚本。

相较 gold_experiment 的改进点：
1. 修复 Hybrid 评分接口错误（不再把 interactions 当 user_id 调用）。
2. 统一推荐接口，避免 CB 路径重复调用和 profile 判定歧义。
3. 增加 MRR@K、用户冷启动分桶评估、目录覆盖率（Catalog Coverage）。
4. 训练/评估分离，保证测试集信息不泄露到模型训练。
5. 支持 BERTopic 不可用时降级，并清晰记录 warning。
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore")

notebook_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(notebook_dir, "..", ".."))
sys.path.insert(0, project_root)
os.chdir(project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    max_poems: int = 600
    n_users: int = 220
    min_ratings: int = 5
    max_ratings: int = 30
    test_ratio: float = 0.2
    n_seeds: int = 5
    top_k: int = 10
    threshold: float = 3.5


def load_poems(max_poems: int) -> List[dict]:
    data_path = os.path.join(project_root, "data", "chinese-poetry")
    poems = []

    def _load(path: str, limit: int, offset: int = 0):
        loaded = []
        if not os.path.exists(path):
            return loaded
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, p in enumerate(data[:limit]):
            content = "".join(p.get("paragraphs", [])) or p.get("title", "")
            if len(content) < 10:
                continue
            loaded.append(
                {
                    "id": offset + i,
                    "title": p.get("title", f"诗词_{offset+i}"),
                    "content": content,
                    "author": p.get("author", "未知"),
                }
            )
        return loaded

    try:
        poems.extend(_load(os.path.join(data_path, "全唐诗", "唐诗三百首.json"), max_poems // 2, 0))
        poems.extend(_load(os.path.join(data_path, "宋词", "宋词三百首.json"), max_poems // 2, len(poems)))
    except Exception as e:
        logger.warning("加载真实诗词异常: %s", e)

    if len(poems) < 100:
        logger.warning("真实数据不足，使用回退诗词样本扩充")
        base = [
            "明月几时有把酒问青天不知天上宫阙今夕是何年",
            "床前明月光疑是地上霜举头望明月低头思故乡",
            "春风又绿江南岸明月何时照我还",
            "大漠孤烟直长河落日圆",
            "会当凌绝顶一览众山小",
        ]
        for i in range(max_poems):
            poems.append({"id": len(poems) + i, "title": f"回退诗词_{i}", "content": base[i % len(base)], "author": "未知"})

    logger.info("加载诗词数量: %d", len(poems))
    return poems


def generate_interactions(poems: List[dict], cfg: Config, seed: int) -> List[dict]:
    np.random.seed(seed)
    random.seed(seed)

    themes = {
        "思乡": ["月", "乡", "归", "故", "家"],
        "送别": ["别", "离", "酒", "千里", "友"],
        "山水": ["山", "水", "云", "林", "江", "河"],
        "边塞": ["塞", "戈", "马", "沙", "战"],
        "四季": ["春", "夏", "秋", "冬", "花", "雪", "风", "雨"],
    }

    poem_ids = [p["id"] for p in poems]
    poem_content = {p["id"]: p.get("content", "") for p in poems}
    interactions = []

    for uid in range(cfg.n_users):
        user_themes = random.sample(list(themes.keys()), k=random.randint(2, 4))
        n_ratings = random.randint(cfg.min_ratings, cfg.max_ratings)
        selected = random.sample(poem_ids, k=min(n_ratings, len(poem_ids)))

        for pid in selected:
            rating = 3.0
            content = poem_content.get(pid, "")

            for t in user_themes:
                if any(kw in content for kw in themes[t]):
                    rating += 0.45

            rating += np.random.normal(0, 0.4)
            rating = float(np.clip(rating, 1.0, 5.0))

            interactions.append(
                {
                    "user_id": uid,
                    "poem_id": pid,
                    "rating": round(rating, 2),
                    "liked": rating >= cfg.threshold,
                    "created_at": datetime.now() - timedelta(days=random.randint(0, 60)),
                }
            )

    return interactions


def temporal_split(interactions: List[dict], test_ratio: float) -> Tuple[List[dict], List[dict]]:
    by_user = defaultdict(list)
    for x in interactions:
        by_user[x["user_id"]].append(x)

    train, test = [], []
    for _, xs in by_user.items():
        xs.sort(key=lambda v: v["created_at"])
        n = len(xs)
        n_test = max(1, int(n * test_ratio))
        train.extend(xs[: n - n_test])
        test.extend(xs[n - n_test :])
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


class PlatinumHybrid:
    """实验侧 Hybrid 聚合器（规避 core.HybridRecommender 的接口耦合问题）。"""

    def __init__(self, poems: List[dict], train_data: List[dict]):
        from backend.core.content_recommender import ContentBasedRecommender
        from backend.core.collaborative_filter import ItemBasedCFRecommender

        self.poems = poems
        self.pid_to_poem = {p["id"]: p for p in poems}
        self.cb = ContentBasedRecommender()
        self.cb.fit(poems)

        self.cf = ItemBasedCFRecommender()
        self.cf.fit(train_data, [p["id"] for p in poems])

        self.bt = None
        try:
            from backend.core.bertopic_recommender import BertopicRecommender

            self.bt = BertopicRecommender()
            self.bt.fit(poems, train_data)
        except Exception as e:
            logger.warning("BERTopic 不可用，降级为 CB+ItemCF: %s", e)

    @staticmethod
    def _weights(n_interactions: int) -> Dict[str, float]:
        if n_interactions == 0:
            return {"cb": 0.3, "item_cf": 0.2, "bertopic": 0.5}
        if n_interactions < 10:
            return {"cb": 0.3, "item_cf": 0.3, "bertopic": 0.4}
        return {"cb": 0.2, "item_cf": 0.3, "bertopic": 0.5}

    @staticmethod
    def _norm(recs: List[dict]) -> Dict[int, float]:
        if not recs:
            return {}
        max_score = max(r.get("score", 0.0) for r in recs)
        if max_score <= 0:
            return {}
        return {r["poem_id"]: float(r["score"]) / max_score for r in recs}

    def recommend(self, user_interactions: List[dict], exclude_ids: set, top_k: int) -> List[dict]:
        weights = self._weights(len(user_interactions))

        rated_poems = [self.pid_to_poem[i["poem_id"]] for i in user_interactions if i["poem_id"] in self.pid_to_poem]
        ratings = [i["rating"] for i in user_interactions if i["poem_id"] in self.pid_to_poem]

        profile = self.cb.get_user_profile(rated_poems, ratings) if rated_poems else None
        cb_recs = self.cb.recommend(profile, exclude_ids, top_k * 3) if profile is not None else []
        cf_recs = self.cf.recommend(user_interactions, exclude_ids, top_k * 3)
        bt_recs = self.bt.recommend(user_interactions, [], top_k * 3) if self.bt is not None else []

        cb_n, cf_n, bt_n = self._norm(cb_recs), self._norm(cf_recs), self._norm(bt_recs)
        all_ids = set(cb_n) | set(cf_n) | set(bt_n)

        combined = []
        for pid in all_ids:
            if pid in exclude_ids:
                continue
            s = cb_n.get(pid, 0.0) * weights["cb"] + cf_n.get(pid, 0.0) * weights["item_cf"] + bt_n.get(pid, 0.0) * weights["bertopic"]
            combined.append((pid, s))

        combined.sort(key=lambda x: x[1], reverse=True)
        return [{"poem_id": pid, "score": score} for pid, score in combined[:top_k]]

    def predict_rating(self, user_interactions: List[dict], poem_id: int) -> float:
        rated_poems = [self.pid_to_poem[i["poem_id"]] for i in user_interactions if i["poem_id"] in self.pid_to_poem]
        ratings = [i["rating"] for i in user_interactions if i["poem_id"] in self.pid_to_poem]
        weights = self._weights(len(user_interactions))

        cb_pred = 3.0
        if rated_poems and poem_id in self.pid_to_poem:
            profile = self.cb.get_user_profile(rated_poems, ratings)
            if profile is not None:
                poem_idx = next((idx for idx, p in enumerate(self.poems) if p["id"] == poem_id), None)
                if poem_idx is not None:
                    cb_pred = float(self.cb.predict_rating(profile, poem_idx))

        cf_pred = float(self.cf.predict_rating(user_interactions, poem_id))
        bt_pred = float(self.bt.predict_rating(user_interactions, poem_id)) if self.bt is not None else 3.0

        return float(np.clip(cb_pred * weights["cb"] + cf_pred * weights["item_cf"] + bt_pred * weights["bertopic"], 1.0, 5.0))


class Evaluator:
    def __init__(self, poems: List[dict], top_k: int, threshold: float):
        self.poems = poems
        self.top_k = top_k
        self.threshold = threshold
        self.pid_to_idx = {p["id"]: idx for idx, p in enumerate(poems)}

    def evaluate(self, models: Dict[str, dict], train_data: List[dict], test_data: List[dict]) -> dict:
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
            preds, acts = [], []
            p_list, r_list, f1_list, ndcg_list, mrr_list = [], [], [], [], []
            all_rec_items = set()

            bucket_scores = defaultdict(lambda: defaultdict(list))

            for inter in test_data:
                uid, pid, actual = inter["user_id"], inter["poem_id"], inter["rating"]
                train_items = user_train.get(uid, [])
                if not train_items:
                    continue

                try:
                    if kind == "cb":
                        rated_poems = [self.poems[self.pid_to_idx[t["poem_id"]]] for t in train_items if t["poem_id"] in self.pid_to_idx]
                        ratings = [t["rating"] for t in train_items if t["poem_id"] in self.pid_to_idx]
                        if rated_poems and pid in self.pid_to_idx:
                            prof = rec.get_user_profile(rated_poems, ratings)
                            if prof is not None:
                                preds.append(float(rec.predict_rating(prof, self.pid_to_idx[pid])))
                                acts.append(actual)
                    else:
                        preds.append(float(rec.predict_rating(train_items, pid)))
                        acts.append(actual)
                except Exception:
                    continue

            mae = float(np.mean(np.abs(np.array(preds) - np.array(acts)))) if preds else float("nan")

            for uid, test_items in user_test.items():
                train_items = user_train.get(uid, [])
                if not train_items:
                    continue

                relevant = {x["poem_id"] for x in test_items if x["rating"] >= self.threshold}
                if not relevant:
                    continue

                exclude = {x["poem_id"] for x in train_items}
                n_hist = len(train_items)
                bucket = "0" if n_hist == 0 else ("1-2" if n_hist <= 2 else ("3-5" if n_hist <= 5 else ("6-10" if n_hist <= 10 else "10+")))

                try:
                    if kind == "cb":
                        rated_poems = [self.poems[self.pid_to_idx[t["poem_id"]]] for t in train_items if t["poem_id"] in self.pid_to_idx]
                        ratings = [t["rating"] for t in train_items if t["poem_id"] in self.pid_to_idx]
                        prof = rec.get_user_profile(rated_poems, ratings) if rated_poems else None
                        rec_list = rec.recommend(prof, exclude, self.top_k) if prof is not None else []
                    else:
                        rec_list = rec.recommend(train_items, exclude, self.top_k)
                except Exception:
                    rec_list = []

                rec_ids = [x["poem_id"] for x in rec_list]
                all_rec_items.update(rec_ids)

                hit = len(set(rec_ids) & relevant)
                precision = hit / len(rec_ids) if rec_ids else 0.0
                recall = hit / len(relevant) if relevant else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                ndcg = ndcg_at_k(rec_ids, relevant, self.top_k)
                mrr = mrr_at_k(rec_ids, relevant, self.top_k)

                p_list.append(precision)
                r_list.append(recall)
                f1_list.append(f1)
                ndcg_list.append(ndcg)
                mrr_list.append(mrr)

                bucket_scores[bucket]["precision"].append(precision)
                bucket_scores[bucket]["recall"].append(recall)
                bucket_scores[bucket]["ndcg"].append(ndcg)

            results[name] = {
                "mae": mae,
                "precision": float(np.mean(p_list)) if p_list else float("nan"),
                "recall": float(np.mean(r_list)) if r_list else float("nan"),
                "f1": float(np.mean(f1_list)) if f1_list else float("nan"),
                "ndcg": float(np.mean(ndcg_list)) if ndcg_list else float("nan"),
                "mrr": float(np.mean(mrr_list)) if mrr_list else float("nan"),
                "coverage": len(all_rec_items) / len(self.poems) if self.poems else 0.0,
                "cold_start": {b: {k: float(np.mean(v)) for k, v in m.items()} for b, m in bucket_scores.items()},
            }

        return results


def aggregate(seed_results: List[dict]) -> dict:
    if not seed_results:
        return {}

    names = seed_results[0].keys()
    metrics = ["mae", "precision", "recall", "f1", "ndcg", "mrr", "coverage"]
    out = {}

    for name in names:
        out[name] = {}
        for metric in metrics:
            vals = [r[name][metric] for r in seed_results if name in r and not np.isnan(r[name][metric])]
            if vals:
                out[name][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    return out


def run_once(cfg: Config, poems: List[dict], interactions: List[dict]) -> dict:
    from backend.core.content_recommender import ContentBasedRecommender
    from backend.core.collaborative_filter import ItemBasedCFRecommender

    train, test = temporal_split(interactions, cfg.test_ratio)
    logger.info("Train=%d, Test=%d", len(train), len(test))

    cb = ContentBasedRecommender()
    cb.fit(poems)

    cf = ItemBasedCFRecommender()
    cf.fit(train, [p["id"] for p in poems])

    hybrid = PlatinumHybrid(poems, train)

    models = {
        "Content-Based": {"recommender": cb, "type": "cb"},
        "Item-CF": {"recommender": cf, "type": "cf"},
        "Hybrid": {"recommender": hybrid, "type": "hybrid"},
    }

    evaluator = Evaluator(poems, cfg.top_k, cfg.threshold)
    return evaluator.evaluate(models, train, test)


def main() -> int:
    cfg = Config()
    seeds = [42, 123, 456, 789, 1024][: cfg.n_seeds]

    poems = load_poems(cfg.max_poems)
    all_results = []

    for s in seeds:
        logger.info("=== Seed %s ===", s)
        interactions = generate_interactions(poems, cfg, s)
        all_results.append(run_once(cfg, poems, interactions))

    agg = aggregate(all_results)
    output = {
        "config": cfg.__dict__,
        "seeds": seeds,
        "aggregate": agg,
        "all_results": all_results,
        "notes": {
            "purpose": "Platinum实验：修复Gold评估漏洞并补齐设计指标",
            "added_metrics": ["MRR@K", "Catalog Coverage", "Cold-start buckets"],
        },
    }

    out_path = "backend/experiments/platium_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("结果已保存: %s", out_path)

    print("\n=== Platinum Summary ===")
    for name in ["Content-Based", "Item-CF", "Hybrid"]:
        if name not in agg:
            continue
        m = agg[name]
        print(
            f"{name:<14} "
            f"P@{cfg.top_k}={m.get('precision', {}).get('mean', float('nan')):.4f}, "
            f"R@{cfg.top_k}={m.get('recall', {}).get('mean', float('nan')):.4f}, "
            f"NDCG={m.get('ndcg', {}).get('mean', float('nan')):.4f}, "
            f"MRR={m.get('mrr', {}).get('mean', float('nan')):.4f}, "
            f"Coverage={m.get('coverage', {}).get('mean', float('nan')):.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
