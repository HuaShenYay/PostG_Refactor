#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诗词推荐系统实验 - Gold版本 (完善版)
遵循 fangan.md 实验设计规范

比较三类推荐方法:
1. Content-Based (CB) - 基于内容
2. Item-CF - 基于物品协同过滤
3. Hybrid - 融合CB与Item-CF + BERTopic

评价指标:
- MAE (辅助指标)
- Precision@K, Recall@K, F1@K (主指标)
- NDCG@K (补充指标)
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
    """实验配置 - 扩大规模"""

    def __init__(self):
        self.max_poems = 600  # 扩大诗歌数量
        self.n_users = 200  # 增加用户数
        self.min_ratings = 5  # 最少评分数
        self.max_ratings = 30  # 最多评分数
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
                    if len(content) >= 10:  # 过滤太短的诗歌
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
                    if len(content) >= 10:
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

    # 如果数据不够，生成更多
    if len(poems) < 100:
        logger.warning("数据不足，补充生成诗歌")
        base_poems = [
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
            {
                "content": "两个黄鹂鸣翠柳一行白鹭上青天",
                "title": "绝句",
                "author": "杜甫",
            },
            {
                "content": "日照香炉生紫烟遥看瀑布挂前川",
                "title": "望庐山瀑布",
                "author": "李白",
            },
            {
                "content": "朝辞白帝彩云间千里江陵一日还",
                "title": "早发白帝城",
                "author": "李白",
            },
            {
                "content": "独在异乡为异客每逢佳节倍思亲",
                "title": "九月九日忆山东兄弟",
                "author": "王维",
            },
            {"content": "空山新雨后天气晚来秋", "title": "山居秋暝", "author": "王维"},
        ]
        for i in range(max_poems):
            b = base_poems[i % len(base_poems)]
            poems.append(
                {
                    "id": len(poems) + i,
                    "title": f"{b['title']}_{i // len(base_poems) + 1}",
                    "content": b["content"],
                    "author": b["author"],
                }
            )

    logger.info(f"加载 {len(poems)} 首诗词")
    return poems


def generate_interactions(poems, n_users, min_ratings, max_ratings, seed):
    """生成用户交互数据 - 增强信号"""
    np.random.seed(seed)
    random.seed(seed)

    interactions = []
    poem_ids = [p["id"] for p in poems]

    # 扩展主题
    themes = {
        "思乡": ["月", "乡", "归", "故", "家", "故里", "亲人"],
        "送别": ["别", "离", "酒", "千里", "朋友", "知己"],
        "山水": ["山", "水", "云", "林", "江", "河", "湖", "海"],
        "边塞": ["塞", "戈", "马", "沙", "战", "将军", "胡"],
        "爱情": ["情", "爱", "相思", "红颜", "知己", "心"],
        "怀古": ["古", "迹", "怀", "往", "英雄", "历史"],
        "田园": ["田", "园", "农", "归", "耕", "隐"],
        "四季": ["春", "夏", "秋", "冬", "花", "雪", "雨", "风"],
    }

    for user_id in range(n_users):
        user_themes = random.sample(list(themes.keys()), k=random.randint(3, 5))
        n_ratings = random.randint(min_ratings, max_ratings)
        selected = random.sample(poem_ids, k=min(n_ratings, len(poem_ids)))

        for poem_id in selected:
            content = ""
            for p in poems:
                if p["id"] == poem_id:
                    content = p.get("content", "")
                    break

            rating = 3.0
            matched_themes = []
            for theme, kw in themes.items():
                if theme in user_themes:
                    for k in kw:
                        if k in content:
                            rating += 0.4
                            matched_themes.append(theme)
                            break

            # 增加噪声
            rating += np.random.normal(0, 0.4)
            rating = np.clip(rating, 1.0, 5.0)

            interactions.append(
                {
                    "user_id": user_id,
                    "poem_id": poem_id,
                    "rating": round(rating, 1),
                    "liked": rating >= 3.5,
                    "created_at": datetime.now()
                    - timedelta(days=random.randint(0, 60)),
                }
            )

    return interactions


def time_based_split(interactions, test_ratio=0.2):
    """基于时间的用户内划分"""
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


def dcg_at_k(relevances, k):
    """计算DCG"""
    relevances = np.array(relevances)[:k]
    if len(relevances) == 0:
        return 0.0
    gains = 2**relevances - 1
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return np.sum(gains / discounts)


def ndcg_at_k(recommended, relevant, k):
    """计算NDCG"""
    if not relevant:
        return 0.0

    relevances = [1 if r in relevant else 0 for r in recommended[:k]]
    ideal_relevances = sorted(relevances, reverse=True)

    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


class Evaluator:
    """统一评估器"""

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
        """评估所有模型 - 带详细日志"""
        user_train = self.build_user_data(train_data)
        user_test = defaultdict(list)
        for inter in test_data:
            user_test[inter["user_id"]].append(inter)

        results = {}

        for name, model_info in models.items():
            rec = model_info["recommender"]
            rec_type = model_info["type"]

            logger.info(f"  评估 {name}...")

            # MAE计算
            preds, acts = [], []
            mae_errors = []

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
                        ratings_list = [
                            p["rating"]
                            for p in user_inters
                            if p["poem_id"] in self.poem_id_to_idx
                        ]
                        if rated and ratings_list and poem_id in self.poem_id_to_idx:
                            profile = rec.get_user_profile(rated, ratings_list)
                            if profile is not None and np.sum(np.abs(profile)) > 1e-10:
                                pred = rec.predict_rating(
                                    profile, self.poem_id_to_idx[poem_id]
                                )
                                if pred is not None and 1.0 <= pred <= 5.0:
                                    preds.append(pred)
                                    acts.append(actual)
                    else:
                        pred = rec.predict_rating(user_inters, poem_id)
                        if pred is not None and 1.0 <= pred <= 5.0:
                            preds.append(pred)
                            acts.append(actual)
                except Exception as e:
                    mae_errors.append(str(e)[:50])
                    continue

            mae = (
                np.mean(np.abs(np.array(preds) - np.array(acts)))
                if preds
                else float("nan")
            )

            if mae_errors:
                logger.warning(
                    f"    MAE计算跳过 {len(mae_errors)} 条: {mae_errors[:3]}"
                )

            # Top-K推荐评估
            precs, recs, f1s, ndcgs = [], [], [], []
            eval_users = 0

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
                        ratings_list = [
                            p["rating"]
                            for p in train_items
                            if p["poem_id"] in self.poem_id_to_idx
                        ]
                        profile = (
                            rec.get_user_profile(rated, ratings_list)
                            if rated and ratings_list
                            else None
                        )
                        profile_valid = (
                            profile is not None and np.sum(np.abs(profile)) > 1e-10
                        )
                        rec_list = (
                            rec.recommend(profile, exclude, self.top_k)
                            if profile_valid
                            else []
                        )
                        rec_list = (
                            rec.recommend(profile, exclude, self.top_k)
                            if profile
                            else []
                        )
                    else:
                        rec_list = rec.recommend(train_items, exclude, self.top_k)

                    recommended = [r["poem_id"] for r in rec_list]

                    tp = len(set(recommended) & relevant)
                    fp = len(set(recommended) - relevant)
                    fn = len(relevant - set(recommended))

                    p = tp / (tp + fp) if (tp + fp) > 0 else 0
                    r = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f = 2 * p * r / (p + r) if (p + r) > 0 else 0

                    ndcg = ndcg_at_k(recommended, relevant, self.top_k)

                    precs.append(p)
                    recs.append(r)
                    f1s.append(f)
                    ndcgs.append(ndcg)
                    eval_users += 1

                except Exception as e:
                    logger.warning(f"    用户{user_id}评估跳过: {str(e)[:50]}")
                    continue

            results[name] = {
                "mae": mae,
                "precision": np.mean(precs) if precs else float("nan"),
                "recall": np.mean(recs) if recs else float("nan"),
                "f1": np.mean(f1s) if f1s else float("nan"),
                "ndcg": np.mean(ndcgs) if ndcgs else float("nan"),
                "n_evaluated": eval_users,
            }

            logger.info(
                f"    评估用户数: {eval_users}, Precision: {results[name]['precision']:.4f}, Recall: {results[name]['recall']:.4f}"
            )

        return results


class HybridWrapper:
    """Hybrid包装器 - 统一接口"""

    def __init__(self, hybrid_model):
        self.hybrid = hybrid_model

    def get_user_profile(self, rated_poems, ratings):
        try:
            return self.hybrid.cb_recommender.get_user_profile(rated_poems, ratings)
        except Exception as e:
            logger.warning(f"CB profile错误: {e}")
            return None

    def predict_rating(self, user_interactions, poem_id):
        return self.hybrid.predict_rating(user_interactions, poem_id)

    def recommend(self, user_interactions, exclude_ids, top_k):
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
            except Exception as e:
                logger.warning(f"CB推荐错误: {e}")

        item_cf_recs = []
        if weights["item_cf"] > 0:
            try:
                item_cf_recs = self.hybrid.item_cf_recommender.recommend(
                    user_interactions, exclude_ids, top_k * 2
                )
            except Exception as e:
                logger.warning(f"Item-CF推荐错误: {e}")

        bertopic_recs = []
        if weights["bertopic"] > 0 and self.hybrid.bertopic_recommender:
            try:
                bertopic_recs = self.hybrid.bertopic_recommender.recommend(
                    user_interactions, [], top_k * 2
                )
            except Exception as e:
                logger.warning(f"BERTopic推荐错误: {e}")

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

        # 测试CB是否能正常工作
        test_profile = cb.get_user_profile([poems[0]], [4.0])
        if test_profile is None:
            logger.warning("  CB: profile返回None，可能有问题")
        elif np.sum(np.abs(test_profile)) < 1e-10:
            logger.warning("  CB: profile全零向量，可能有问题")
        else:
            test_recs = cb.recommend(test_profile, set(), 5)
            logger.info(f"  CB: 测试推荐成功，返回{len(test_recs)}条")

        models["Content-Based"] = {"recommender": cb, "type": "cb"}
        logger.info("  CB: 训练完成")
    except Exception as e:
        logger.error(f"  CB失败: {e}")
        import traceback

        traceback.print_exc()

    # 2. Item-CF
    try:
        from backend.core.collaborative_filter import ItemBasedCFRecommender

        item_cf = ItemBasedCFRecommender()
        item_cf.fit(train_data, [p["id"] for p in poems])
        models["Item-CF"] = {"recommender": item_cf, "type": "cf"}
        logger.info("  Item-CF: 训练完成")
    except Exception as e:
        logger.error(f"  Item-CF失败: {e}")

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
            f"  {name}: MAE={m['mae']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}, NDCG={m['ndcg']:.4f}, n={m['n_evaluated']}"
        )

    return results


def aggregate(all_results):
    """聚合多次实验结果"""
    if not all_results:
        return {}

    names = all_results[0].keys()
    aggregated = {}

    for name in names:
        for metric in ["mae", "precision", "recall", "f1", "ndcg"]:
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
    print("诗词推荐系统实验 - Gold Version (完善版)")
    print("=" * 70)

    config = Config()
    seeds = [42, 123, 456, 789, 1024]

    # 加载数据
    poems = load_poems(config.max_poems)

    if len(poems) < 50:
        logger.error("诗歌数据不足，无法进行实验")
        return 1

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
            f"\n{'方法':<15} {'MAE':<15} {'P@10':<15} {'R@10':<15} {'F1@10':<15} {'NDCG@10':<15}"
        )
        print("-" * 90)

        for name in ["Content-Based", "Item-CF", "Hybrid"]:
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
                n = (
                    f"{m['ndcg']['mean']:.4f}±{m['ndcg']['std']:.4f}"
                    if "ndcg" in m
                    else "N/A"
                )
                print(f"{name:<15} {mae:<15} {p:<15} {r:<15} {f:<15} {n:<15}")

        # 改进分析
        print("\n" + "=" * 70)
        print("改进分析")
        print("=" * 70)

        if "Hybrid" in agg and "Item-CF" in agg:
            hyb = agg["Hybrid"]
            icf = agg["Item-CF"]

            print("\nHybrid vs Item-CF:")
            for metric in ["recall", "f1", "ndcg"]:
                if metric in hyb and metric in icf:
                    h_val = hyb[metric]["mean"]
                    i_val = icf[metric]["mean"]
                    if i_val > 0:
                        improve = (h_val - i_val) / i_val * 100
                        print(
                            f"  {metric.upper()}@10: {improve:+.2f}% ({h_val:.4f} vs {i_val:.4f})"
                        )

        if "Hybrid" in agg and "Content-Based" in agg:
            hyb = agg["Hybrid"]
            cb = agg["Content-Based"]

            print("\nHybrid vs Content-Based:")
            for metric in ["recall", "f1", "ndcg"]:
                if metric in hyb and metric in cb:
                    h_val = hyb[metric]["mean"]
                    c_val = cb[metric]["mean"]
                    if c_val > 0:
                        improve = (h_val - c_val) / c_val * 100
                        print(
                            f"  {metric.upper()}@10: {improve:+.2f}% ({h_val:.4f} vs {c_val:.4f})"
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
        print(f"""
数据规模:
- 诗歌数量: {config.max_poems}
- 用户数量: {config.n_users}
- 每用户评分: {config.min_ratings}-{config.max_ratings}

指标说明:
- MAE: 评分预测误差 (辅助指标)
- Precision@K: 推荐精确率
- Recall@K: 推荐召回率
- F1@K: 精确率和召回率的调和平均
- NDCG@K: 归一化折损累计增益 (更适合排序任务)

结论:
1. CB方法基于内容相似度，在重现历史行为评估中可能表现偏低
2. Item-CF基于协同过滤，依赖用户交互数据
3. Hybrid融合多种方法，通常能取得更好的综合性能
        """)

    return 0


if __name__ == "__main__":
    sys.exit(main())
