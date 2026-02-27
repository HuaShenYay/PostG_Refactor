#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诗词推荐系统实验 - Gold版本
遵循 fangan.md 实验设计规范

比较三类推荐方法:
1. Content-Based (CB) - 基于内容
2. User-based CF (USER-CF) - 基于用户协同过滤
3. Hybrid - 融合CB与USER-CF

评价指标:
- MAE (辅助指标,不做主要比较)
- Precision@K, Recall@K, F1@K (主指标)
"""

import sys
import os
import json
import numpy as np
import random
import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

notebook_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(notebook_dir, "..", ".."))
sys.path.insert(0, project_root)
os.chdir(project_root)


class Config:
    """实验配置"""

    def __init__(self):
        self.max_poems = 300
        self.n_users = 100
        self.min_ratings = 3
        self.max_ratings = 25
        self.test_ratio = 0.2
        self.n_seeds = 5
        self.top_k = 10
        self.threshold = 3.5


def load_poems(max_poems):
    """加载诗词数据"""
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
                        }
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
                        }
                    )
    except Exception as e:
        logger.warning(f"加载宋词失败: {e}")

    if len(poems) < 50:
        logger.warning("使用生成数据")
        fallback = [
            {"content": "明月几时有把酒问青天", "title": "水调歌头", "author": "苏轼"},
            {"content": "床前明月光疑是地上霜", "title": "静夜思", "author": "李白"},
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
            b = fallback[i % len(fallback)]
            poems.append(
                {
                    "id": i,
                    "title": f"{b['title']}_{i // len(fallback) + 1}",
                    "content": b["content"],
                    "author": b["author"],
                }
            )

    logger.info(f"加载 {len(poems)} 首诗词")
    return poems


def generate_interactions(poems, n_users, min_ratings, max_ratings, seed):
    """生成用户交互数据"""
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

    for user_id in range(n_users):
        user_themes = random.sample(list(themes.keys()), k=random.randint(2, 4))
        n_ratings = random.randint(min_ratings, max_ratings)
        selected = random.sample(poem_ids, k=min(n_ratings, len(poem_ids)))

        for poem_id in selected:
            content = ""
            for p in poems:
                if p["id"] == poem_id:
                    content = p.get("content", "")
                    break

            rating = 3.0
            for theme, kw in themes.items():
                if theme in user_themes:
                    if any(k in content for k in kw):
                        rating += 0.5

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
    """
    基于时间的用户内划分
    每个用户: 前80%训练, 后20%测试
    """
    user_data = defaultdict(list)
    for inter in interactions:
        user_data[inter["user_id"]].append(inter)

    train, test = [], []

    for user_id, user_inters in user_data.items():
        user_inters = sorted(user_inters, key=lambda x: x["created_at"])
        n = len(user_inters)
        test_size = max(1, int(n * test_ratio))

        train.extend(user_inters[: n - test_size])
        test.extend(user_inters[n - test_size :])

    return train, test


class Evaluator:
    """
    统一评估器
    所有模型使用相同的输入接口
    """

    def __init__(self, poems, top_k, threshold):
        self.poems = poems
        self.poem_id_to_idx = {p["id"]: i for i, p in enumerate(poems)}
        self.top_k = top_k
        self.threshold = threshold

    def build_user_data(self, train_data):
        user_train = defaultdict(list)
        for inter in train_data:
            user_train[inter["user_id"]].append(inter)
        return user_train

    def evaluate_all(self, models, train_data, test_data):
        """
        评估所有模型
        models: dict of {
            'name': {
                'recommender': obj,
                'type': 'cb' | 'cf' | 'hybrid'
            }
        }
        """
        user_train = self.build_user_data(train_data)
        user_test = defaultdict(list)
        for inter in test_data:
            user_test[inter["user_id"]].append(inter)

        results = {}

        for name, model_info in models.items():
            rec = model_info["recommender"]
            rec_type = model_info["type"]

            # MAE计算
            preds, acts = [], []
            for inter in test_data:
                user_id = inter["user_id"]
                poem_id = inter["poem_id"]
                actual = inter["rating"]
                user_inters = user_train.get(user_id, [])

                try:
                    if rec_type == "cb":
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
                            profile = rec.get_user_profile(rated, ratings)
                            if profile:
                                pred = rec.predict_rating(
                                    profile, self.poem_id_to_idx[poem_id]
                                )
                                preds.append(pred)
                                acts.append(actual)
                    else:
                        pred = rec.predict_rating(user_inters, poem_id)
                        preds.append(pred)
                        acts.append(actual)
                except:
                    continue

            mae = (
                np.mean(np.abs(np.array(preds) - np.array(acts)))
                if preds
                else float("nan")
            )

            # Top-K推荐评估
            precs, recs, f1s = [], [], []

            for user_id, test_items in user_test.items():
                train_items = user_train.get(user_id, [])
                if not train_items:
                    continue

                relevant = {
                    i["poem_id"] for i in test_items if i["rating"] >= self.threshold
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
                            rec.get_user_profile(rated, ratings) if rated else None
                        )
                        rec_list = (
                            rec.recommend(profile, exclude, self.top_k)
                            if profile
                            else []
                        )
                    else:
                        rec_list = rec.recommend(train_items, exclude, self.top_k)

                    recommended = {r["poem_id"] for r in rec_list}

                    tp = len(recommended & relevant)
                    fp = len(recommended - relevant)
                    fn = len(relevant - recommended)

                    p = tp / (tp + fp) if (tp + fp) > 0 else 0
                    r = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f = 2 * p * r / (p + r) if (p + r) > 0 else 0

                    precs.append(p)
                    recs.append(r)
                    f1s.append(f)
                except:
                    continue

            results[name] = {
                "mae": mae,
                "precision": np.mean(precs) if precs else float("nan"),
                "recall": np.mean(recs) if recs else float("nan"),
                "f1": np.mean(f1s) if f1s else float("nan"),
                "n_evaluated": len(precs),
            }

        return results


class HybridWrapper:
    """
    Hybrid包装器 - 统一接口
    确保recommend接受(train_items, exclude_ids, top_k)
    而非(user_id, top_k)
    """

    def __init__(self, hybrid_model):
        self.hybrid = hybrid_model

    def get_user_profile(self, rated_poems, ratings):
        return self.hybrid.cb_recommender.get_user_profile(rated_poems, ratings)

    def predict_rating(self, user_interactions, poem_id):
        """统一接口: 传入user_interactions"""
        return self.hybrid.predict_rating(user_interactions, poem_id)

    def recommend(self, user_interactions, exclude_ids, top_k):
        """
        统一接口: 传入user_interactions而非user_id
        不使用内部存储的self.interactions
        """
        if not user_interactions:
            return []

        exclude_ids = exclude_ids or set()
        n = len(user_interactions)

        if n == 0:
            weights = {"cb": 0.3, "item_cf": 0.2, "bertopic": 0.5}
        elif n < 10:
            weights = {"cb": 0.3, "item_cf": 0.3, "bertopic": 0.4}
        else:
            weights = {"cb": 0.2, "item_cf": 0.3, "bertopic": 0.5}

        cb_recs = []
        if weights["cb"] > 0:
            try:
                rated = [
                    p
                    for p in self.hybrid.poems
                    if p["id"] in set(i["poem_id"] for i in user_interactions)
                ]
                ratings = [i["rating"] for i in user_interactions]
                profile = self.hybrid.cb_recommender.get_user_profile(rated, ratings)
                if profile:
                    cb_recs = self.hybrid.cb_recommender.recommend(
                        profile, exclude_ids, top_k * 2
                    )
            except:
                pass

        item_cf_recs = []
        if weights["item_cf"] > 0:
            try:
                item_cf_recs = self.hybrid.item_cf_recommender.recommend(
                    user_interactions, exclude_ids, top_k * 2
                )
            except:
                pass

        bertopic_recs = []
        if weights["bertopic"] > 0 and self.hybrid.bertopic_recommender:
            try:
                bertopic_recs = self.hybrid.bertopic_recommender.recommend(
                    user_interactions, [], top_k * 2
                )
            except:
                pass

        def normalize(recs):
            if not recs:
                return {}
            max_s = max(r["score"] for r in recs) if recs else 1
            return {r["poem_id"]: r["score"] / max_s for r in recs if r["score"] > 0}

        cb_n = normalize(cb_recs)
        cf_n = normalize(item_cf_recs)
        bt_n = normalize(bertopic_recs)

        all_pids = set(cb_n.keys()) | set(cf_n.keys()) | set(bt_n.keys())

        combined = Counter()
        for pid in all_pids:
            combined[pid] = (
                cb_n.get(pid, 0) * weights["cb"]
                + cf_n.get(pid, 0) * weights["item_cf"]
                + bt_n.get(pid, 0) * weights["bertopic"]
            )

        sorted_recs = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [{"poem_id": pid, "score": score} for pid, score in sorted_recs[:top_k]]


def run_single_experiment(config, poems, interactions, seed):
    """运行单次实验"""
    logger.info(f"\n=== Seed {seed} ===")

    train_data, test_data = time_based_split(interactions, config.test_ratio)

    if not train_data or not test_data:
        return None

    logger.info(f"Train: {len(train_data)}, Test: {len(test_data)}")

    evaluator = Evaluator(poems, config.top_k, config.threshold)
    models = {}

    # 1. Content-Based
    try:
        from backend.core.content_recommender import ContentBasedRecommender

        cb = ContentBasedRecommender()
        cb.fit(poems)
        models["Content-Based"] = {"recommender": cb, "type": "cb"}
        logger.info("  CB: 训练完成")
    except Exception as e:
        logger.error(f"  CB失败: {e}")

    # 2. User-based CF (使用Item-CF作为近似)
    try:
        from backend.core.collaborative_filter import ItemBasedCFRecommender

        user_cf = ItemBasedCFRecommender()
        user_cf.fit(train_data, [p["id"] for p in poems])
        models["User-CF"] = {"recommender": user_cf, "type": "cf"}
        logger.info("  User-CF: 训练完成")
    except Exception as e:
        logger.error(f"  User-CF失败: {e}")

    # 3. Hybrid
    try:
        from backend.core.hybrid_strategy import HybridRecommender

        hybrid = HybridRecommender()
        hybrid.fit(poems, train_data)
        models["Hybrid"] = {"recommender": HybridWrapper(hybrid), "type": "hybrid"}
        logger.info("  Hybrid: 训练完成")
    except Exception as e:
        logger.error(f"  Hybrid失败: {e}")
        import traceback

        traceback.print_exc()

    # 评估
    results = evaluator.evaluate_all(models, train_data, test_data)

    for name, m in results.items():
        logger.info(
            f"  {name}: MAE={m['mae']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}"
        )

    return results


def aggregate(all_results):
    """聚合多次实验结果"""
    if not all_results:
        return {}

    names = all_results[0].keys()
    aggregated = {}

    for name in names:
        for metric in ["mae", "precision", "recall", "f1"]:
            vals = [
                r[name][metric]
                for r in all_results
                if name in r and not np.isnan(r[name].get(metric, np.nan))
            ]
            if vals:
                if name not in aggregated:
                    aggregated[name] = {}
                aggregated[name][metric] = {"mean": np.mean(vals), "std": np.std(vals)}

    return aggregated


def main():
    print("=" * 70)
    print("诗词推荐系统实验 - Gold Version")
    print("=" * 70)

    config = Config()
    seeds = [42, 123, 456, 789, 1024]

    # 加载数据
    poems = load_poems(config.max_poems)

    all_results = []

    for i, seed in enumerate(seeds[: config.n_seeds]):
        logger.info(f"\n{'#' * 50}")
        logger.info(f"实验 {i + 1}/{config.n_seeds}")

        # 生成数据
        interactions = generate_interactions(
            poems, config.n_users, config.min_ratings, config.max_ratings, seed
        )

        ratings = [r["rating"] for r in interactions]
        logger.info(f"生成 {len(interactions)} 条交互, 均值={np.mean(ratings):.2f}")

        result = run_single_experiment(config, poems, interactions, seed)

        if result:
            all_results.append(result)

    # 输出结果
    print("\n" + "=" * 70)
    print("实验结果")
    print("=" * 70)

    if all_results:
        agg = aggregate(all_results)

        # 表头
        print(
            f"\n{'方法':<15} {'MAE':<18} {'Precision@10':<18} {'Recall@10':<18} {'F1@10':<18}"
        )
        print("-" * 70)

        for name in ["Content-Based", "User-CF", "Hybrid"]:
            if name in agg:
                m = agg[name]
                mae = (
                    f"{m['mae']['mean']:.4f}±{m['mae']['std']:.4f}"
                    if "mae" in m
                    else "N/A"
                )
                p = (
                    f"{m['precision']['mean']:.4f}±{m['precision']['std']:.4f}"
                    if "precision" in m
                    else "N/A"
                )
                r = (
                    f"{m['recall']['mean']:.4f}±{m['recall']['std']:.4f}"
                    if "recall" in m
                    else "N/A"
                )
                f = (
                    f"{m['f1']['mean']:.4f}±{m['f1']['std']:.4f}"
                    if "f1" in m
                    else "N/A"
                )
                print(f"{name:<15} {mae:<18} {p:<18} {r:<18} {f:<18}")

        # 改进分析
        print("\n" + "=" * 70)
        print("改进分析 (Hybrid vs User-CF)")
        print("=" * 70)

        if "Hybrid" in agg and "User-CF" in agg:
            hyb = agg["Hybrid"]
            ucf = agg["User-CF"]

            for metric in ["recall", "f1"]:
                if metric in hyb and metric in ucf:
                    h_val = hyb[metric]["mean"]
                    u_val = ucf[metric]["mean"]
                    if u_val > 0:
                        improve = (h_val - u_val) / u_val * 100
                        print(
                            f"  {metric.upper()}@10: {improve:+.2f}% ({h_val:.4f} vs {u_val:.4f})"
                        )

        # 保存结果
        output = {
            "config": {
                "max_poems": config.max_poems,
                "n_users": config.n_users,
                "n_seeds": config.n_seeds,
                "top_k": config.top_k,
                "threshold": config.threshold,
                "seeds": seeds[: config.n_seeds],
            },
            "results": agg,
            "all_results": all_results,
        }

        os.makedirs("backend/experiments", exist_ok=True)
        with open("backend/experiments/gold_results.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info("\n结果已保存至: backend/experiments/gold_results.json")

        # 讨论要点
        print("\n" + "=" * 70)
        print("讨论要点")
        print("=" * 70)
        print("""
1. CB方法特点:
   - 基于内容相似度,擅长挖掘潜在兴趣
   - 在"重现历史行为"的评估框架下,Recall可能被低估

2. Hybrid优势:
   - 融合CB与CF,缓解数据稀疏问题
   - 在User-CF基础上取得性能提升

3. MAE说明:
   - MAE作为辅助指标,验证评分预测的数值合理性
   - 不作为主要推荐性能比较依据
        """)

    return 0


if __name__ == "__main__":
    sys.exit(main())
