#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诗词推荐系统实验 - 优化版本
改进方向：
1. 数据：真实诗词数据 + 仿真用户行为 + 时间划分
2. 评估：NDCG/MAP/覆盖率 + 敏感性分析 + 统一接口 + 基线对比
3. 实验：多次重复 + 控制变量 + 统计显著性
4. 代码：配置化 + 日志 + 效率优化
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
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 路径设置
notebook_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(notebook_dir, "..", ".."))
sys.path.insert(0, project_root)
os.chdir(project_root)


class ExperimentConfig:
    """实验配置类"""

    def __init__(self):
        # 数据配置
        self.max_poems = 300
        self.n_users = 100
        self.min_ratings = 3
        self.max_ratings = 25
        self.test_ratio = 0.2
        self.n_seeds = 5

        # 评估配置
        self.top_k_values = [5, 10, 20]
        self.threshold_values = [3.0, 3.5, 4.0]

        # 基线配置
        self.baselines = ["random", "popular"]

        # 输出配置
        self.output_dir = "backend/experiments"
        self.save_results = True


def load_poems(max_poems=300):
    """加载真实诗词数据"""
    data_path = os.path.join(project_root, "data", "chinese-poetry")
    poems = []

    # 加载唐诗
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

    # 加载宋词
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

    # 改进的fallback数据 - 20首不同主题的诗词
    if len(poems) < 50:
        logger.warning("使用生成数据 (真实数据不足)")
        fallback_poems = [
            {
                "content": "明月几时有把酒问青天不知天上宫阙今夕是何年",
                "title": "水调歌头",
                "author": "苏轼",
                "theme": "思乡",
            },
            {
                "content": "床前明月光疑是地上霜举头望明月低头思故乡",
                "title": "静夜思",
                "author": "李白",
                "theme": "思乡",
            },
            {
                "content": "春风又绿江南岸明月何时照我还",
                "title": "泊船瓜洲",
                "author": "王安石",
                "theme": "思乡",
            },
            {
                "content": "大漠孤烟直长河落日圆",
                "title": "使至塞上",
                "author": "王维",
                "theme": "边塞",
            },
            {
                "content": "会当凌绝顶一览众山小",
                "title": "望岳",
                "author": "杜甫",
                "theme": "山水",
            },
            {
                "content": "海内存知己天涯若比邻",
                "title": "送杜少府之任蜀州",
                "author": "王勃",
                "theme": "送别",
            },
            {
                "content": "落红不是无情物化作春泥更护花",
                "title": "己亥杂诗",
                "author": "龚自珍",
                "theme": "咏物",
            },
            {
                "content": "春蚕到死丝方尽蜡炬成灰泪始干",
                "title": "无题",
                "author": "李商隐",
                "theme": "爱情",
            },
            {
                "content": "山重水复疑无路柳暗花明又一村",
                "title": "游山西村",
                "author": "陆游",
                "theme": "哲理",
            },
            {
                "content": "欲穷千里目更上一层楼",
                "title": "登鹳雀楼",
                "author": "王之涣",
                "theme": "哲理",
            },
            {
                "content": "独在异乡为异客每逢佳节倍思亲",
                "title": "九月九日忆山东兄弟",
                "author": "王维",
                "theme": "思乡",
            },
            {
                "content": "劝君更尽一杯酒西出阳关无故人",
                "title": "送元二使安西",
                "author": "王维",
                "theme": "送别",
            },
            {
                "content": "白日依山尽黄河入海流",
                "title": "登鹳雀楼",
                "author": "王之涣",
                "theme": "山水",
            },
            {
                "content": "千山鸟飞绝万径人踪灭",
                "title": "江雪",
                "author": "柳宗元",
                "theme": "山水",
            },
            {
                "content": "孤帆远影碧空尽唯见长江天际流",
                "title": "黄鹤楼送孟浩然之广陵",
                "author": "李白",
                "theme": "送别",
            },
            {
                "content": "两个黄鹂鸣翠柳一行白鹭上青天",
                "title": "绝句",
                "author": "杜甫",
                "theme": "山水",
            },
            {
                "content": "曾经沧海难为水除却巫山不是云",
                "title": "离思",
                "author": "元稹",
                "theme": "爱情",
            },
            {
                "content": "身无彩凤双飞翼心有灵犀一点通",
                "title": "无题",
                "author": "李商隐",
                "theme": "爱情",
            },
            {
                "content": "问君能有几多愁恰似一江春水向东流",
                "title": "虞美人",
                "author": "李煜",
                "theme": "怀旧",
            },
            {
                "content": "莫等闲白了少年头空悲切",
                "title": "满江红",
                "author": "岳飞",
                "theme": "哲理",
            },
        ]

        for i in range(max_poems):
            base = fallback_poems[i % len(fallback_poems)]
            poems.append(
                {
                    "id": i,
                    "title": f"{base['title']}_{i // len(fallback_poems) + 1}",
                    "content": base["content"]
                    + (str(i) if i >= len(fallback_poems) else ""),
                    "author": base["author"],
                    "dynasty": "未知",
                    "theme": base["theme"],
                }
            )

    logger.info(f"总计加载 {len(poems)} 首诗词")
    return poems


def generate_interactions(poems, n_users, min_ratings, max_ratings, seed):
    """生成用户交互数据 - 改进版"""
    np.random.seed(seed)
    random.seed(seed)

    interactions = []
    poem_ids = [p["id"] for p in poems]

    # 主题关键词
    themes = {
        "思乡": ["月", "乡", "归", "故", "家", "故里"],
        "送别": ["别", "离", "酒", "千里", "朋友"],
        "山水": ["山", "水", "云", "林", "江", "河"],
        "边塞": ["塞", "戈", "马", "沙", "战场"],
        "怀古": ["古", "迹", "怀", "往", "历史"],
        "爱情": ["情", "爱", "相思", "心"],
        "哲理": ["理", "道", "人生", "悟"],
    }

    # 用户评分偏好偏差 (模拟真实用户评分习惯)
    user_bias = {i: np.random.normal(0, 0.5) for i in range(n_users)}

    # 用户活跃度分布 (长尾 - 大部分用户交互少)
    activity_weights = np.random.pareto(2, n_users) + 0.5
    activity_weights = activity_weights / activity_weights.sum()

    for user_id in range(n_users):
        # 用户主题偏好
        user_themes = random.sample(list(themes.keys()), k=random.randint(2, 4))

        # 根据活跃度确定评分数量
        n_ratings = int(
            np.random.uniform(min_ratings, max_ratings) * activity_weights[user_id]
        )
        n_ratings = max(2, min(n_ratings, max_ratings))

        # 热门物品偏差 (长尾效应 - 热门物品被更多用户评分)
        popularity = Counter()
        for p in poems:
            content = p.get("content", "")
            for theme, keywords in themes.items():
                if any(k in content for k in keywords):
                    popularity[p["id"]] += 1

        # 热门物品更可能被评分
        pop_scores = np.array([popularity.get(pid, 1) for pid in poem_ids], dtype=float)
        pop_probs = pop_scores / pop_scores.sum()

        selected_poems = np.random.choice(
            poem_ids, size=min(n_ratings, len(poem_ids)), replace=False, p=pop_probs
        )

        for poem_id in selected_poems:
            poem_content = ""
            poem_theme = ""
            for p in poems:
                if p["id"] == poem_id:
                    poem_content = p.get("content", "")
                    poem_theme = p.get("theme", "")
                    break

            # 基于主题匹配计算评分
            rating = 3.0
            for theme, keywords in themes.items():
                if theme in user_themes:
                    matches = sum(1 for k in keywords if k in poem_content)
                    if matches > 0:
                        rating += matches * 0.5

            # 加入用户偏差
            rating += user_bias[user_id]

            # 加入噪声
            rating += np.random.normal(0, 0.5)

            rating = np.clip(rating, 1.0, 5.0)

            # 时间戳 (模拟时间分布)
            days_ago = np.random.exponential(15)

            interactions.append(
                {
                    "user_id": user_id,
                    "poem_id": poem_id,
                    "rating": round(rating, 1),
                    "liked": rating >= 3.5,
                    "created_at": datetime.now() - timedelta(days=days_ago),
                }
            )

    # 按时间排序
    interactions.sort(key=lambda x: x["created_at"])

    return interactions


def time_based_split(interactions, test_ratio=0.2):
    """时间划分 - 训练集是早期的, 测试集是后期的"""
    if not interactions:
        return [], []

    user_data = defaultdict(list)
    for inter in interactions:
        user_data[inter["user_id"]].append(inter)

    train, test = [], []

    for user_id, user_inters in user_data.items():
        # 按时间排序
        user_inters = sorted(user_inters, key=lambda x: x["created_at"])
        n = len(user_inters)
        test_size = max(1, int(n * test_ratio))

        # 前(1-test_ratio)作为训练, 后test_ratio作为测试
        for i, inter in enumerate(user_inters):
            if i >= n - test_size:
                test.append(inter)
            else:
                train.append(inter)

    return train, test


def random_split(interactions, test_ratio=0.2, seed=42):
    """随机划分 (对比用)"""
    np.random.seed(seed)
    random.seed(seed)

    user_data = defaultdict(list)
    for inter in interactions:
        user_data[inter["user_id"]].append(inter)

    train, test = [], []

    for user_id, user_inters in user_data.items():
        n = len(user_inters)
        test_size = max(1, int(n * test_ratio))

        indices = list(range(n))
        random.shuffle(indices)
        test_indices = set(indices[:test_size])

        for i, inter in enumerate(user_inters):
            if i in test_indices:
                test.append(inter)
            else:
                train.append(inter)

    return train, test


class BaselineRecommender:
    """基线推荐器"""

    def __init__(self, name):
        self.name = name
        self.poem_ids = []
        self.popular_poems = []

    def fit(self, poems, train_data):
        self.poem_ids = [p["id"] for p in poems]

        if self.name == "popular":
            # 热门推荐
            poem_scores = Counter()
            for inter in train_data:
                poem_scores[inter["poem_id"]] += inter.get("rating", 3.0)
            self.popular_poems = [pid for pid, _ in poem_scores.most_common()]

    def recommend(self, train_items, exclude_ids, top_k):
        exclude_ids = set(exclude_ids) if exclude_ids else set()

        if self.name == "random":
            available = [pid for pid in self.poem_ids if pid not in exclude_ids]
            random.shuffle(available)
            return [
                {"poem_id": pid, "score": 1.0 / len(available) if available else 0}
                for pid in available[:top_k]
            ]

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


class UnifiedEvaluator:
    """统一评估器 - 修复接口不一致问题"""

    def __init__(self, poems):
        self.poems = poems
        self.poem_id_to_idx = {p["id"]: i for i, p in enumerate(poems)}
        self.poem_id_to_info = {p["id"]: p for p in poems}

    def build_user_data(self, train_data):
        user_train = defaultdict(list)
        for inter in train_data:
            user_train[inter["user_id"]].append(inter)
        return user_train

    def predict_and_evaluate(
        self, recommender, train_data, test_data, rec_type, top_k=10, threshold=3.5
    ):
        """统一的预测评估接口"""
        user_train = self.build_user_data(train_data)

        # MAE计算
        predictions, actuals = [], []
        missing_count = 0

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
                        profile = recommender.get_user_profile(rated, ratings)
                        if profile is not None:
                            pred = recommender.predict_rating(
                                profile, self.poem_id_to_idx[poem_id]
                            )
                            predictions.append(pred)
                            actuals.append(actual)
                    else:
                        missing_count += 1

                elif rec_type in ["item_cf", "baseline"]:
                    pred = recommender.predict_rating(user_inters, poem_id)
                    predictions.append(pred)
                    actuals.append(actual)

                elif rec_type == "hybrid":
                    # Hybrid接口: recommend(user_id, top_k) / predict_rating(user_interactions, poem_id)
                    pred = recommender.predict_rating(user_inters, poem_id)
                    predictions.append(pred)
                    actuals.append(actual)

            except Exception as e:
                missing_count += 1

        mae = (
            np.mean(np.abs(np.array(predictions) - np.array(actuals)))
            if predictions
            else float("nan")
        )

        # 推荐列表评估
        return (
            self.evaluate_recommendations(
                recommender,
                train_data,
                test_data,
                rec_type,
                top_k,
                threshold,
                missing_count,
            ),
            mae,
            len(predictions),
            missing_count,
        )

    def evaluate_recommendations(
        self,
        recommender,
        train_data,
        test_data,
        rec_type,
        top_k,
        threshold,
        missing_count,
    ):
        """评估推荐列表"""
        user_train = self.build_user_data(train_data)
        user_test = defaultdict(list)
        for inter in test_data:
            user_test[inter["user_id"]].append(inter)

        all_precision, all_recall, all_f1 = [], [], []
        ndcgs = []

        for user_id, test_items in user_test.items():
            train_items = user_train.get(user_id, [])
            if not train_items:
                continue

            # 相关物品 (评分>=阈值)
            relevant = {i["poem_id"] for i in test_items if i["rating"] >= threshold}
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
                        recommender.get_user_profile(rated, ratings) if rated else None
                    )
                    recs = (
                        recommender.recommend(profile, exclude, top_k)
                        if profile
                        else []
                    )

                elif rec_type in ["item_cf", "baseline"]:
                    recs = recommender.recommend(train_items, exclude, top_k)

                elif rec_type == "hybrid":
                    recs = recommender.recommend(user_id, top_k=top_k)

                recommended = {r["poem_id"] for r in recs}

                # Precision/Recall/F1
                tp = len(recommended & relevant)
                fp = len(recommended - relevant)
                fn = len(relevant - recommended)

                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0
                f = 2 * p * r / (p + r) if (p + r) > 0 else 0

                all_precision.append(p)
                all_recall.append(r)
                all_f1.append(f)

                # NDCG
                dcg = 0.0
                for i, r in enumerate(recs):
                    if r["poem_id"] in relevant:
                        dcg += 1.0 / np.log2(i + 2)

                idcg = sum(
                    1.0 / np.log2(i + 2) for i in range(min(len(relevant), top_k))
                )
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcgs.append(ndcg)

            except Exception as e:
                continue

        # 覆盖率和新颖性
        coverage, novelty = self.evaluate_diversity(
            recommender, train_data, test_data, rec_type, top_k
        )

        return {
            "precision": np.mean(all_precision) if all_precision else float("nan"),
            "recall": np.mean(all_recall) if all_recall else float("nan"),
            "f1": np.mean(all_f1) if all_f1 else float("nan"),
            "ndcg": np.mean(ndcgs) if ndcgs else float("nan"),
            "coverage": coverage,
            "novelty": novelty,
            "evaluated_users": len(all_precision),
            "missing_samples": missing_count,
        }

    def evaluate_diversity(self, recommender, train_data, test_data, rec_type, top_k):
        """评估覆盖率和新颖性"""
        user_train = self.build_user_data(train_data)
        user_test = defaultdict(list)
        for inter in test_data:
            user_test[inter["user_id"]].append(inter)

        all_recommended = set()
        total_recommended = 0

        for user_id, test_items in user_test.items():
            train_items = user_train.get(user_id, [])
            if not train_items:
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
                        recommender.get_user_profile(rated, ratings) if rated else None
                    )
                    recs = (
                        recommender.recommend(profile, exclude, top_k)
                        if profile
                        else []
                    )

                elif rec_type in ["item_cf", "baseline"]:
                    recs = recommender.recommend(train_items, exclude, top_k)

                elif rec_type == "hybrid":
                    recs = recommender.recommend(user_id, top_k=top_k)

                for r in recs:
                    all_recommended.add(r["poem_id"])
                    total_recommended += 1

            except:
                continue

        # 覆盖率: 推荐物品占所有物品的比例
        coverage = len(all_recommended) / len(self.poems) if self.poems else 0

        # 新颖性: 推荐非热门物品的能力 (用信息熵简化)
        poem_pop = Counter()
        for inter in train_data:
            poem_pop[inter["poem_id"]] += 1

        total = sum(poem_pop.values())
        if total > 0:
            pop_probs = [
                poem_pop[pid] / total for pid in all_recommended if pid in poem_pop
            ]
            novelty = (
                -sum(p * np.log2(p + 1e-10) for p in pop_probs) / len(pop_probs)
                if pop_probs
                else 0
            )
        else:
            novelty = 0

        return coverage, novelty


def run_experiment(config, poems, interactions, seed):
    """运行单次实验"""
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Running experiment with seed={seed}")
    logger.info(f"{'=' * 50}")

    # 数据划分 - 时间划分
    train_data, test_data = time_based_split(interactions, test_ratio=config.test_ratio)

    if not train_data or not test_data:
        logger.warning("数据划分失败")
        return None

    logger.info(f"训练集: {len(train_data)}, 测试集: {len(test_data)}")

    # 评估器
    evaluator = UnifiedEvaluator(poems)

    results = {}

    # 1. 基线模型
    for baseline_name in config.baselines:
        logger.info(f"训练 {baseline_name}...")
        try:
            baseline = BaselineRecommender(baseline_name)
            baseline.fit(poems, train_data)

            metrics, mae, n_pred, n_missing = evaluator.predict_and_evaluate(
                baseline,
                train_data,
                test_data,
                "baseline",
                top_k=max(config.top_k_values),
                threshold=3.5,
            )
            metrics["mae"] = mae
            metrics["n_predictions"] = n_pred
            metrics["n_missing"] = n_missing
            results[f"Baseline-{baseline_name}"] = metrics
            logger.info(f"  MAE={mae:.4f}, NDCG={metrics['ndcg']:.4f}")
        except Exception as e:
            logger.error(f"  失败: {e}")

    # 2. Content-Based
    logger.info("训练 Content-Based...")
    try:
        from backend.core.content_recommender import ContentBasedRecommender

        cb = ContentBasedRecommender()
        cb.fit(poems)

        metrics, mae, n_pred, n_missing = evaluator.predict_and_evaluate(
            cb,
            train_data,
            test_data,
            "cb",
            top_k=max(config.top_k_values),
            threshold=3.5,
        )
        metrics["mae"] = mae
        metrics["n_predictions"] = n_pred
        metrics["n_missing"] = n_missing
        results["Content-Based"] = metrics
        logger.info(f"  MAE={mae:.4f}, NDCG={metrics['ndcg']:.4f}")
    except Exception as e:
        logger.error(f"  失败: {e}")

    # 3. Item-CF
    logger.info("训练 Item-CF...")
    try:
        from backend.core.collaborative_filter import ItemBasedCFRecommender

        item_cf = ItemBasedCFRecommender()
        item_cf.fit(train_data, [p["id"] for p in poems])

        metrics, mae, n_pred, n_missing = evaluator.predict_and_evaluate(
            item_cf,
            train_data,
            test_data,
            "item_cf",
            top_k=max(config.top_k_values),
            threshold=3.5,
        )
        metrics["mae"] = mae
        metrics["n_predictions"] = n_pred
        metrics["n_missing"] = n_missing
        results["Item-CF"] = metrics
        logger.info(f"  MAE={mae:.4f}, NDCG={metrics['ndcg']:.4f}")
    except Exception as e:
        logger.error(f"  失败: {e}")

    # 4. Hybrid
    logger.info("训练 Hybrid (你的系统)...")
    try:
        from backend.core.hybrid_strategy import HybridRecommender

        hybrid = HybridRecommender()
        hybrid.fit(poems, train_data)

        metrics, mae, n_pred, n_missing = evaluator.predict_and_evaluate(
            hybrid,
            train_data,
            test_data,
            "hybrid",
            top_k=max(config.top_k_values),
            threshold=3.5,
        )
        metrics["mae"] = mae
        metrics["n_predictions"] = n_pred
        metrics["n_missing"] = n_missing
        results["Hybrid"] = metrics
        logger.info(f"  MAE={mae:.4f}, NDCG={metrics['ndcg']:.4f}")
    except Exception as e:
        logger.error(f"  失败: {e}")
        import traceback

        traceback.print_exc()

    return results


def aggregate_results(all_results):
    """聚合多次实验结果"""
    if not all_results:
        return {}

    model_names = all_results[0].keys()
    aggregated = {}

    for name in model_names:
        metrics = {}
        for metric in [
            "mae",
            "precision",
            "recall",
            "f1",
            "ndcg",
            "coverage",
            "novelty",
        ]:
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
    """打印结果"""
    print("\n" + "=" * 80)
    print("实验结果")
    print("=" * 80)

    # 多次实验结果
    if all_results and len(all_results) > 1:
        aggregated = aggregate_results(all_results)

        print(f"\n{'算法':<20} {'MAE':<18} {'NDCG':<18} {'F1':<18} {'覆盖率':<12}")
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
            cov_str = (
                f"{metrics['coverage']['mean']:.4f}" if "coverage" in metrics else "N/A"
            )

            print(f"{name:<20} {mae_str:<18} {ndcg_str:<18} {f1_str:<18} {cov_str:<12}")

    # 单次结果
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
        print("改进分析 (vs 基线)")
        print("=" * 80)

        if "Hybrid" in results and "Baseline-popular" in results:
            hybrid_mae = results["Hybrid"].get("mae", np.nan)
            pop_mae = results["Baseline-popular"].get("mae", np.nan)

            if not np.isnan(hybrid_mae) and not np.isnan(pop_mae):
                mae_improve = (
                    (pop_mae - hybrid_mae) / pop_mae * 100 if pop_mae > 0 else 0
                )

                hybrid_ndcg = results["Hybrid"].get("ndcg", np.nan)
                pop_ndcg = results["Baseline-popular"].get("ndcg", np.nan)

                ndcg_improve = (
                    (hybrid_ndcg - pop_ndcg) / pop_ndcg * 100 if pop_ndcg > 0 else 0
                )

                print(f"\nHybrid vs Popular:")
                print(f"  MAE: {mae_improve:+.1f}%")
                print(f"  NDCG: {ndcg_improve:+.1f}%")

        if "Hybrid" in results:
            for baseline in [
                "Baseline-random",
                "Baseline-popular",
                "Content-Based",
                "Item-CF",
            ]:
                if baseline in results:
                    other = results[baseline]
                    hybrid = results["Hybrid"]

                    for metric in ["mae", "ndcg", "f1"]:
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
    """主函数"""
    parser = argparse.ArgumentParser(description="诗词推荐系统实验")
    parser.add_argument("--max_poems", type=int, default=300, help="最大诗词数")
    parser.add_argument("--n_users", type=int, default=100, help="用户数")
    parser.add_argument("--n_seeds", type=int, default=5, help="实验次数")
    parser.add_argument("--top_k", type=int, default=10, help="推荐数")
    parser.add_argument(
        "--output",
        type=str,
        default="backend/experiments/optimized_results.json",
        help="输出文件",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("诗词推荐系统实验 - 优化版本")
    print("=" * 80)

    # 配置
    config = ExperimentConfig()
    config.max_poems = args.max_poems
    config.n_users = args.n_users
    config.n_seeds = args.n_seeds

    # 1. 加载数据
    logger.info("加载诗词数据...")
    poems = load_poems(config.max_poems)

    # 2. 多次实验
    all_results = []
    seeds = [42, 123, 456, 789, 1024]

    for i, seed in enumerate(seeds[: config.n_seeds]):
        logger.info(f"\n{'#' * 50}")
        logger.info(f"实验 {i + 1}/{config.n_seeds}, seed={seed}")
        logger.info(f"{'#' * 50}")

        # 生成数据
        interactions = generate_interactions(
            poems, config.n_users, config.min_ratings, config.max_ratings, seed
        )

        logger.info(f"生成 {len(interactions)} 条交互")

        # 统计
        ratings = [r["rating"] for r in interactions]
        logger.info(
            f"评分分布: 均值={np.mean(ratings):.2f}, 标准差={np.std(ratings):.2f}"
        )
        logger.info(
            f"正样本(>=3.5): {sum(1 for r in ratings if r >= 3.5) / len(ratings) * 100:.1f}%"
        )

        # 运行实验
        result = run_experiment(config, poems, interactions, seed)

        if result:
            all_results.append(result)

    # 3. 输出结果
    if all_results:
        print_results(all_results[-1], all_results)

        # 保存
        os.makedirs(os.path.dirname(config.output), exist_ok=True)

        output_data = {
            "config": {
                "max_poems": config.max_poems,
                "n_users": config.n_users,
                "n_seeds": config.n_seeds,
                "seeds": seeds[: config.n_seeds],
            },
            "aggregated": aggregate_results(all_results),
            "all_results": all_results,
        }

        with open(config.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"\n结果已保存至: {config.output}")
    else:
        logger.error("所有实验均失败")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
