#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诗词推荐系统实验 - 公平评估版本
关键改进：
1. 统一MAE计算接口 - 所有模型使用相同的预测方式
2. 公平评估 - Hybrid不使用内部存储的额外信息
3. 多种评估指标
"""

import sys
import os
import json
import numpy as np
import random
import logging
import argparse
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

notebook_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(notebook_dir, "..", ".."))
sys.path.insert(0, project_root)
os.chdir(project_root)


class ExperimentConfig:
    def __init__(self):
        self.max_poems = 300
        self.n_users = 100
        self.min_ratings = 3
        self.max_ratings = 25
        self.test_ratio = 0.2
        self.n_seeds = 5
        self.top_k = 10
        self.threshold = 3.5
        self.output_dir = "backend/experiments"


def load_poems(max_poems=300):
    data_path = os.path.join(project_root, "data", "chinese-poetry")
    poems = []

    try:
        tang_path = os.path.join(data_path, "全唐诗", "唐诗三百首.json")
        if os.path.exists(tang_path):
            with open(tang_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for i, p in enumerate(data[: max_poems // 2]):
                    paragraphs = p.get("paragraphs", [])
                    content = "".join(paragraphs) if paragraphs else p.get("title", "")
                    poems.append(
                        {
                            "id": i,
                            "title": p.get("title", f"诗{i}"),
                            "content": content,
                            "author": p.get("author", "未知"),
                            "dynasty": "唐",
                        }
                    )
            logger.info(
                f"加载唐诗 {len([p for p in poems if p.get('dynasty') == '唐'])} 首"
            )
    except Exception as e:
        logger.warning(f"加载唐诗失败: {e}")

    try:
        ci_path = os.path.join(data_path, "宋词", "宋词三百首.json")
        if os.path.exists(ci_path):
            with open(ci_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                offset = len(poems)
                for i, p in enumerate(data[: max_poems // 2]):
                    paragraphs = p.get("paragraphs", [])
                    content = "".join(paragraphs) if paragraphs else p.get("title", "")
                    poems.append(
                        {
                            "id": offset + i,
                            "title": p.get("title", f"词{i}"),
                            "content": content,
                            "author": p.get("author", "未知"),
                            "dynasty": "宋",
                        }
                    )
            logger.info(
                f"加载宋词 {len([p for p in poems if p.get('dynasty') == '宋'])} 首"
            )
    except Exception as e:
        logger.warning(f"加载宋词失败: {e}")

    if len(poems) < 50:
        logger.warning("使用生成数据")
        fallback_poems = [
            {
                "content": "明月几时有把酒问青天不知天上宫阙今夕是何年",
                "title": "水调歌头",
                "author": "苏轼",
            },
            {
                "content": "床前明月光疑是地上霜举头望明月低头思故乡",
                "title": "静夜思",
                "author": "李白",
            },
            {
                "content": "春风又绿江南岸明月何时照我还",
                "title": "泊船瓜洲",
                "author": "王安石",
            },
            {"content": "大漠孤烟直长河落日圆", "title": "使至塞上", "author": "王维"},
            {"content": "会当凌绝顶一览众山小", "title": "望岳", "author": "杜甫"},
            {"content": "海内存知己天涯若比邻", "title": "送杜少府", "author": "王勃"},
            {
                "content": "落红不是无情物化作春泥更护花",
                "title": "己亥杂诗",
                "author": "龚自珍",
            },
            {
                "content": "春蚕到死丝方尽蜡炬成灰泪始干",
                "title": "无题",
                "author": "李商隐",
            },
            {
                "content": "山重水复疑无路柳暗花明又一村",
                "title": "游山西村",
                "author": "陆游",
            },
            {
                "content": "欲穷千里目更上一层楼",
                "title": "登鹳雀楼",
                "author": "王之涣",
            },
        ]

        for i in range(max_poems):
            base = fallback_poems[i % len(fallback_poems)]
            poems.append(
                {
                    "id": i,
                    "title": f"{base['title']}_{i // len(fallback_poems) + 1}",
                    "content": base["content"],
                    "author": base["author"],
                    "dynasty": "未知",
                }
            )

    logger.info(f"总计 {len(poems)} 首诗词")
    return poems


def generate_interactions(poems, n_users, min_ratings, max_ratings, seed):
    np.random.seed(seed)
    random.seed(seed)

    interactions = []
    poem_ids = [p["id"] for p in poems]

    themes = {
        "思乡": ["月", "乡", "归", "故", "家"],
        "送别": ["别", "离", "酒", "千里"],
        "山水": ["山", "水", "云", "林"],
        "边塞": ["塞", "戈", "马", "沙"],
        "爱情": ["情", "爱", "相思"],
    }

    user_bias = {i: np.random.normal(0, 0.5) for i in range(n_users)}

    for user_id in range(n_users):
        user_themes = random.sample(list(themes.keys()), k=random.randint(2, 4))
        n_ratings = random.randint(min_ratings, max_ratings)

        selected_poems = random.sample(poem_ids, k=min(n_ratings, len(poem_ids)))

        for poem_id in selected_poems:
            poem_content = ""
            for p in poems:
                if p["id"] == poem_id:
                    poem_content = p.get("content", "")
                    break

            rating = 3.0
            for theme, keywords in themes.items():
                if theme in user_themes:
                    if any(k in poem_content for k in keywords):
                        rating += 0.5

            rating += user_bias[user_id]
            rating += np.random.normal(0, 0.5)
            rating = np.clip(rating, 1.0, 5.0)

            interactions.append(
                {
                    "user_id": user_id,
                    "poem_id": poem_id,
                    "rating": round(rating, 1),
                    "liked": rating >= 3.5,
                    "created_at": datetime.now()
                    - timedelta(days=random.randint(0, 30)),
                }
            )

    return interactions


def time_based_split(interactions, test_ratio=0.2):
    user_data = defaultdict(list)
    for inter in interactions:
        user_data[inter["user_id"]].append(inter)

    train, test = [], []

    for user_id, user_inters in user_data.items():
        user_inters = sorted(user_inters, key=lambda x: x["created_at"])
        n = len(user_inters)
        test_size = max(1, int(n * test_ratio))

        for i, inter in enumerate(user_inters):
            if i >= n - test_size:
                test.append(inter)
            else:
                train.append(inter)

    return train, test


class FairEvaluator:
    """
    公平评估器 - 关键改进：
    1. 所有模型使用相同的预测接口
    2. 不允许Hybrid使用内部存储的额外信息
    """

    def __init__(self, poems):
        self.poems = poems
        self.poem_id_to_idx = {p["id"]: i for i, p in enumerate(poems)}
        self.poem_ids = [p["id"] for p in poems]

    def build_user_data(self, train_data):
        user_train = defaultdict(list)
        for inter in train_data:
            user_train[inter["user_id"]].append(inter)
        return user_train

    def evaluate(self, models_dict, train_data, test_data, top_k=10, threshold=3.5):
        """
        统一评估所有模型
        models_dict: {
            'ModelName': {
                'recommender': recommender_obj,
                'type': 'cb' | 'cf' | 'hybrid'
            }
        }
        """
        user_train = self.build_user_data(train_data)
        user_test = defaultdict(list)
        for inter in test_data:
            user_test[inter["user_id"]].append(inter)

        results = {}

        for model_name, model_info in models_dict.items():
            recommender = model_info["recommender"]
            rec_type = model_info["type"]

            logger.info(f"  评估 {model_name}...")

            # 统一使用 predict_rating 接口 (不传入额外信息)
            predictions, actuals = [], []

            for inter in test_data:
                user_id = inter["user_id"]
                poem_id = inter["poem_id"]
                actual = inter["rating"]

                user_inters = user_train.get(user_id, [])

                try:
                    if rec_type == "cb":
                        # CB: 需要构建用户画像
                        rated = [
                            self.poems[self.poem_id_to_idx[p["poem_id"]]]
                            for p in user_inters
                            if p["poem_id"] in self.poem_id_to_idx
                        ]
                        ratings = [
                            p["rating"]
                            for p in user_inters
                            if p["poem_id"] in self.poem_id_to_idx
                        ]

                        if rated and poem_id in self.poem_id_to_idx:
                            profile = recommender.get_user_profile(rated, ratings)
                            if profile is not None:
                                pred = recommender.predict_rating(
                                    profile, self.poem_id_to_idx[poem_id]
                                )
                                predictions.append(pred)
                                actuals.append(actual)

                    elif rec_type == "cf":
                        # CF: 直接传入 user_inters
                        pred = recommender.predict_rating(user_inters, poem_id)
                        predictions.append(pred)
                        actuals.append(actual)

                    elif rec_type == "hybrid":
                        # Hybrid: 也传入 user_inters (不使用内部存储!)
                        pred = recommender.predict_rating(user_inters, poem_id)
                        predictions.append(pred)
                        actuals.append(actual)

                except Exception as e:
                    continue

            mae = (
                np.mean(np.abs(np.array(predictions) - np.array(actuals)))
                if predictions
                else float("nan")
            )
            n_pred = len(predictions)

            # 推荐列表评估
            precisions, recalls, f1s, ndcgs = [], [], [], []
            recommended_items = set()
            total_recommended = 0

            for user_id, test_items in user_test.items():
                train_items = user_train.get(user_id, [])
                if not train_items:
                    continue

                relevant = {
                    i["poem_id"] for i in test_items if i["rating"] >= threshold
                }
                if not relevant:
                    continue

                exclude = {i["poem_id"] for i in train_items}

                try:
                    if rec_type == "cb":
                        rated = [
                            self.poems[self.poem_id_to_idx[p["poem_id"]]]
                            for p in train_items
                            if p["poem_id"] in self.poem_id_to_idx
                        ]
                        ratings = [
                            p["rating"]
                            for p in train_items
                            if p["poem_id"] in self.poem_id_to_idx
                        ]
                        profile = (
                            recommender.get_user_profile(rated, ratings)
                            if rated
                            else None
                        )
                        recs = (
                            recommender.recommend(profile, exclude, top_k)
                            if profile
                            else []
                        )

                    elif rec_type == "cf":
                        recs = recommender.recommend(train_items, exclude, top_k)

                    elif rec_type == "hybrid":
                        # 关键：传入 train_items 而不是 user_id!
                        recs = recommender.recommend(train_items, exclude, top_k)

                    recommended = {r["poem_id"] for r in recs}

                    # 统计覆盖率
                    for r in recs:
                        recommended_items.add(r["poem_id"])
                        total_recommended += 1

                    # PRF
                    tp = len(recommended & relevant)
                    fp = len(recommended - relevant)
                    fn = len(relevant - recommended)

                    p = tp / (tp + fp) if (tp + fp) > 0 else 0
                    r = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f = 2 * p * r / (p + r) if (p + r) > 0 else 0

                    precisions.append(p)
                    recalls.append(r)
                    f1s.append(f)

                    # NDCG
                    dcg = sum(
                        1.0 / np.log2(i + 2)
                        for i, r in enumerate(recs)
                        if r["poem_id"] in relevant
                    )
                    idcg = sum(
                        1.0 / np.log2(i + 2) for i in range(min(len(relevant), top_k))
                    )
                    ndcg = dcg / idcg if idcg > 0 else 0
                    ndcgs.append(ndcg)

                except Exception as e:
                    continue

            # 覆盖率
            coverage = len(recommended_items) / len(self.poems) if self.poems else 0

            results[model_name] = {
                "mae": mae,
                "n_predictions": n_pred,
                "precision": np.mean(precisions) if precisions else float("nan"),
                "recall": np.mean(recalls) if recalls else float("nan"),
                "f1": np.mean(f1s) if f1s else float("nan"),
                "ndcg": np.mean(ndcgs) if ndcgs else float("nan"),
                "coverage": coverage,
                "evaluated_users": len(precisions),
            }

        return results


class HybridRecommenderFair(HybridRecommender):
    """
    公平版Hybrid推荐器 - 修复接口问题
    关键：recommend() 接受 train_items 而不是 user_id
    """

    def __init__(self):
        super().__init__()

    def recommend(self, user_interactions, exclude_ids=None, top_k=10):
        """
        公平接口 - 接受 user_interactions 而非 user_id
        不使用内部存储的 self.interactions
        """
        if not user_interactions:
            return self._get_default_recs(exclude_ids, top_k)

        exclude_ids = exclude_ids or set()

        # 计算权重
        interaction_count = len(user_interactions)

        if interaction_count == 0:
            weights = {"cb": 0.3, "item_cf": 0.2, "bertopic": 0.5}
        elif interaction_count < 10:
            weights = {"cb": 0.3, "item_cf": 0.3, "bertopic": 0.4}
        else:
            weights = {"cb": 0.2, "item_cf": 0.3, "bertopic": 0.5}

        # CB推荐
        cb_recs = []
        if weights["cb"] > 0:
            try:
                rated_poems = [
                    p
                    for p in self.poems
                    if p["id"] in set(i["poem_id"] for i in user_interactions)
                ]
                ratings = [i["rating"] for i in user_interactions]
                user_profile = self.cb_recommender.get_user_profile(
                    rated_poems, ratings
                )
                if user_profile is not None:
                    cb_recs = self.cb_recommender.recommend(
                        user_profile, exclude_ids, top_k * 2
                    )
            except:
                pass

        # Item-CF推荐
        item_cf_recs = []
        if weights["item_cf"] > 0:
            try:
                item_cf_recs = self.item_cf_recommender.recommend(
                    user_interactions, exclude_ids, top_k * 2
                )
            except:
                pass

        # BERTopic推荐 (可能失败)
        bertopic_recs = []
        if weights["bertopic"] > 0 and self.bertopic_recommender is not None:
            try:
                bertopic_recs = self.bertopic_recommender.recommend(
                    user_interactions, [], top_k * 2
                )
            except:
                pass

        # 分数归一化与合并
        def normalize(recs):
            if not recs:
                return {}
            max_s = max(r["score"] for r in recs) if recs else 1
            return {r["poem_id"]: r["score"] / max_s for r in recs if r["score"] > 0}

        cb_norm = normalize(cb_recs)
        item_cf_norm = normalize(item_cf_recs)
        bertopic_norm = normalize(bertopic_recs)

        all_pids = (
            set(cb_norm.keys()) | set(item_cf_norm.keys()) | set(bertopic_norm.keys())
        )

        combined = Counter()
        for pid in all_pids:
            combined[pid] = (
                cb_norm.get(pid, 0) * weights["cb"]
                + item_cf_norm.get(pid, 0) * weights["item_cf"]
                + bertopic_norm.get(pid, 0) * weights["bertopic"]
            )

        sorted_recs = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        return [{"poem_id": pid, "score": score} for pid, score in sorted_recs[:top_k]]

    def _get_default_recs(self, exclude_ids, top_k):
        """冷启动默认推荐"""
        if not self.item_cf_recommender:
            return []
        return self.item_cf_recommender.recommend([], exclude_ids, top_k)


class BaselineRecommender:
    def __init__(self, name):
        self.name = name
        self.poem_ids = []
        self.popular_poems = []

    def fit(self, train_data):
        poem_scores = Counter()
        for inter in train_data:
            poem_scores[inter["poem_id"]] += inter.get("rating", 3.0)
        self.popular_poems = [pid for pid, _ in poem_scores.most_common()]

    def recommend(self, train_items, exclude_ids, top_k):
        exclude_ids = set(exclude_ids) if exclude_ids else set()

        if self.name == "random":
            all_ids = [pid for pid in self.poem_ids if pid not in exclude_ids]
            random.shuffle(all_ids)
            return [{"poem_id": pid, "score": 1.0} for pid in all_ids[:top_k]]

        elif self.name == "popular":
            recs = []
            for pid in self.popular_poems:
                if pid not in exclude_ids:
                    recs.append({"poem_id": pid, "score": 1.0})
                    if len(recs) >= top_k:
                        break
            return recs

        return []

    def predict_rating(self, train_items, poem_id):
        return 3.0


def run_experiment(config, poems, interactions, seed):
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Experiment seed={seed}")
    logger.info(f"{'=' * 50}")

    train_data, test_data = time_based_split(interactions, test_ratio=config.test_ratio)

    if not train_data or not test_data:
        return None

    logger.info(f"Train: {len(train_data)}, Test: {len(test_data)}")

    evaluator = FairEvaluator(poems)
    models = {}

    # 基线模型
    for name in ["random", "popular"]:
        try:
            baseline = BaselineRecommender(name)
            baseline.fit(train_data)
            baseline.poem_ids = [p["id"] for p in poems]
            models[f"Baseline-{name}"] = {"recommender": baseline, "type": "cf"}
        except Exception as e:
            logger.error(f"  {name} failed: {e}")

    # Content-Based
    try:
        from backend.core.content_recommender import ContentBasedRecommender

        cb = ContentBasedRecommender()
        cb.fit(poems)
        models["Content-Based"] = {"recommender": cb, "type": "cb"}
    except Exception as e:
        logger.error(f"  CB failed: {e}")

    # Item-CF
    try:
        from backend.core.collaborative_filter import ItemBasedCFRecommender

        item_cf = ItemBasedCFRecommender()
        item_cf.fit(train_data, [p["id"] for p in poems])
        models["Item-CF"] = {"recommender": item_cf, "type": "cf"}
    except Exception as e:
        logger.error(f"  Item-CF failed: {e}")

    # Hybrid - 使用公平接口
    try:
        from backend.core.hybrid_strategy import HybridRecommender

        hybrid = HybridRecommender()
        hybrid.fit(poems, train_data)
        # 包装为公平接口
        hybrid_fair = HybridRecommenderFair()
        hybrid_fair.poems = hybrid.poems
        hybrid_fair.interactions = hybrid.interactions
        hybrid_fair.cb_recommender = hybrid.cb_recommender
        hybrid_fair.item_cf_recommender = hybrid.item_cf_recommender
        hybrid_fair.bertopic_recommender = hybrid.bertopic_recommender

        models["Hybrid"] = {"recommender": hybrid_fair, "type": "hybrid"}
    except Exception as e:
        logger.error(f"  Hybrid failed: {e}")
        import traceback

        traceback.print_exc()

    # 评估
    results = evaluator.evaluate(
        models, train_data, test_data, top_k=config.top_k, threshold=config.threshold
    )

    for name, metrics in results.items():
        logger.info(f"  {name}: MAE={metrics['mae']:.4f}, NDCG={metrics['ndcg']:.4f}")

    return results


def aggregate_results(all_results):
    if not all_results:
        return {}

    model_names = all_results[0].keys()
    aggregated = {}

    for name in model_names:
        metrics = {}
        for metric in ["mae", "precision", "recall", "f1", "ndcg", "coverage"]:
            values = [
                r[name][metric]
                for r in all_results
                if name in r and not np.isnan(r[name].get(metric, np.nan))
            ]
            if values:
                metrics[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "values": values,
                }
        aggregated[name] = metrics

    return aggregated


def print_results(results, all_results=None):
    print("\n" + "=" * 80)
    print("实验结果")
    print("=" * 80)

    if all_results and len(all_results) > 1:
        aggregated = aggregate_results(all_results)

        print(f"\n{'算法':<20} {'MAE':<18} {'NDCG':<18} {'F1':<18}")
        print("-" * 80)

        for name, metrics in aggregated.items():
            mae_str = (
                f"{metrics['mae']['mean']:.4f}±{metrics['mae']['std']:.4f}"
                if "mae" in metrics
                else "N/A"
            )
            ndcg_str = (
                f"{metrics['ndcg']['mean']:.4f}±{metrics['ndcg']['std']:.4f}"
                if "ndcg" in metrics
                else "N/A"
            )
            f1_str = (
                f"{metrics['f1']['mean']:.4f}±{metrics['f1']['std']:.4f}"
                if "f1" in metrics
                else "N/A"
            )

            print(f"{name:<20} {mae_str:<18} {ndcg_str:<18} {f1_str:<18}")

    elif results:
        print(
            f"\n{'算法':<20} {'MAE':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'NDCG':<12}"
        )
        print("-" * 80)

        for name, metrics in results.items():
            mae_str = (
                f"{metrics['mae']:.4f}"
                if not np.isnan(metrics.get("mae", np.nan))
                else "N/A"
            )
            p_str = (
                f"{metrics['precision']:.4f}"
                if not np.isnan(metrics.get("precision", np.nan))
                else "N/A"
            )
            r_str = (
                f"{metrics['recall']:.4f}"
                if not np.isnan(metrics.get("recall", np.nan))
                else "N/A"
            )
            f1_str = (
                f"{metrics['f1']:.4f}"
                if not np.isnan(metrics.get("f1", np.nan))
                else "N/A"
            )
            ndcg_str = (
                f"{metrics['ndcg']:.4f}"
                if not np.isnan(metrics.get("ndcg", np.nan))
                else "N/A"
            )

            print(
                f"{name:<20} {mae_str:<12} {p_str:<12} {r_str:<12} {f1_str:<12} {ndcg_str:<12}"
            )

    # 改进分析
    print("\n" + "=" * 80)
    print("改进分析")
    print("=" * 80)

    if results and "Hybrid" in results:
        for baseline in [
            "Baseline-random",
            "Baseline-popular",
            "Content-Based",
            "Item-CF",
        ]:
            if baseline in results:
                hybrid = results["Hybrid"]
                other = results[baseline]

                for metric in ["mae", "ndcg"]:
                    o_val = other.get(metric, np.nan)
                    h_val = hybrid.get(metric, np.nan)

                    if np.isnan(o_val) or np.isnan(h_val):
                        continue

                    if metric == "mae":
                        improve = (o_val - h_val) / o_val * 100 if o_val > 0 else 0
                        direction = "降低" if improve > 0 else "增加"
                    else:
                        improve = (h_val - o_val) / o_val * 100 if o_val > 0 else 0
                        direction = "提升" if improve > 0 else "下降"

                    print(
                        f"  {metric.upper()} vs {baseline}: {improve:+.1f}% ({direction})"
                    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_poems", type=int, default=300)
    parser.add_argument("--n_users", type=int, default=100)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    print("=" * 80)
    print("诗词推荐系统实验 - 公平评估版本")
    print("=" * 80)

    config = ExperimentConfig()
    config.max_poems = args.max_poems
    config.n_users = args.n_users
    config.n_seeds = args.n_seeds
    config.top_k = args.top_k

    poems = load_poems(config.max_poems)

    all_results = []
    seeds = [42, 123, 456, 789, 1024]

    for i, seed in enumerate(seeds[: config.n_seeds]):
        logger.info(f"\n{'#' * 50}")
        logger.info(f"Experiment {i + 1}/{config.n_seeds}, seed={seed}")

        interactions = generate_interactions(
            poems, config.n_users, config.min_ratings, config.max_ratings, seed
        )

        ratings = [r["rating"] for r in interactions]
        logger.info(f"生成 {len(interactions)} 条交互, 均值={np.mean(ratings):.2f}")

        result = run_experiment(config, poems, interactions, seed)

        if result:
            all_results.append(result)

    if all_results:
        print_results(all_results[-1], all_results)

        output_file = os.path.join(config.output_dir, "fair_experiment_results.json")
        os.makedirs(config.output_dir, exist_ok=True)

        output_data = {
            "config": vars(config),
            "seeds": seeds[: config.n_seeds],
            "aggregated": aggregate_results(all_results),
            "all_results": all_results,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"结果已保存至: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
