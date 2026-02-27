import numpy as np
from .content_recommender import ContentBasedRecommender
from .collaborative_filter import ItemBasedCFRecommender
from .bertopic_recommender import BertopicRecommender


class HybridRecommender:
    """
    混合推荐策略

    整合三种推荐算法：
    1. Content-Based (TF-IDF)
    2. Item-Based CF (协同过滤)
    3. BERTopic Hybrid (语义向量 + User-CF + Item-CF)

    根据用户互动数量动态调整权重
    """

    def __init__(self):
        self.cb_recommender = None
        self.item_cf_recommender = None
        self.bertopic_recommender = None

        self.poems = None
        self.interactions = None

    def fit(self, poems, interactions):
        """
        训练所有推荐模型

        Args:
            poems: list of dict with 'id', 'content', 'title'
            interactions: list of dict with 'user_id', 'poem_id', 'rating', 'created_at'
        """
        self.poems = poems
        self.interactions = interactions

        poem_ids = [p["id"] for p in poems]

        print("[Hybrid] Training Content-Based model...")
        self.cb_recommender = ContentBasedRecommender()
        self.cb_recommender.fit(poems)

        print("[Hybrid] Training Item-CF model...")
        self.item_cf_recommender = ItemBasedCFRecommender()
        self.item_cf_recommender.fit(interactions, poem_ids)

        print("[Hybrid] Training BERTopic model...")
        self.bertopic_recommender = BertopicRecommender()
        self.bertopic_recommender.fit(poems, interactions)

        print("[Hybrid] All models trained successfully")

    def _get_user_interactions(self, user_id):
        """获取指定用户的交互历史"""
        return [i for i in self.interactions if i["user_id"] == user_id]

    def recommend(self, user_id, top_k=10, method="hybrid"):
        """
        为用户推荐诗歌

        Args:
            user_id: 用户ID
            top_k: 推荐数量
            method: 'cb', 'item_cf', 'bertopic', 'hybrid'

        Returns:
            list of dict: recommended poems with scores
        """
        user_interactions = self._get_user_interactions(user_id)
        exclude_ids = set(i["poem_id"] for i in user_interactions)

        if method == "cb":
            rated_poems = [p for p in self.poems if p["id"] in exclude_ids]
            ratings = [i["rating"] for i in user_interactions]
            user_profile = self.cb_recommender.get_user_profile(rated_poems, ratings)
            return self.cb_recommender.recommend(user_profile, exclude_ids, top_k)

        elif method == "item_cf":
            return self.item_cf_recommender.recommend(
                user_interactions, exclude_ids, top_k
            )

        elif method == "bertopic":
            return self.bertopic_recommender.recommend(
                user_interactions, self.interactions, top_k
            )

        elif method == "hybrid":
            return self._hybrid_recommend(user_interactions, exclude_ids, top_k)

        return []

    def _hybrid_recommend(self, user_interactions, exclude_ids, top_k):
        """混合推荐核心逻辑"""
        interaction_count = len(user_interactions)

        if interaction_count == 0:
            weights = {"cb": 0.3, "item_cf": 0.2, "bertopic": 0.5}
        elif interaction_count < 10:
            weights = {"cb": 0.3, "item_cf": 0.3, "bertopic": 0.4}
        else:
            weights = {"cb": 0.2, "item_cf": 0.3, "bertopic": 0.5}

        cb_recs = (
            self.cb_recommender.recommend(
                self.cb_recommender.get_user_profile(
                    [
                        p
                        for p in self.poems
                        if p["id"] in set(i["poem_id"] for i in user_interactions)
                    ],
                    [i["rating"] for i in user_interactions],
                )
                if user_interactions
                else None,
                exclude_ids,
                top_k * 2,
            )
            if weights["cb"] > 0
            else []
        )

        item_cf_recs = (
            self.item_cf_recommender.recommend(
                user_interactions, exclude_ids, top_k * 2
            )
            if weights["item_cf"] > 0
            else []
        )

        bertopic_recs = (
            self.bertopic_recommender.recommend(
                user_interactions, self.interactions, top_k * 2
            )
            if weights["bertopic"] > 0
            else []
        )

        cb_scores = {r["poem_id"]: r["score"] * weights["cb"] for r in cb_recs}
        item_cf_scores = {
            r["poem_id"]: r["score"] * weights["item_cf"] for r in item_cf_recs
        }
        bertopic_scores = {
            r["poem_id"]: r["score"] * weights["bertopic"] for r in bertopic_recs
        }

        all_poem_ids = (
            set(cb_scores.keys())
            | set(item_cf_scores.keys())
            | set(bertopic_scores.keys())
        )

        combined_scores = {}
        for pid in all_poem_ids:
            combined_scores[pid] = (
                cb_scores.get(pid, 0)
                + item_cf_scores.get(pid, 0)
                + bertopic_scores.get(pid, 0)
            )

        sorted_poems = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        return [{"poem_id": pid, "score": score} for pid, score in sorted_poems[:top_k]]

    def predict_rating(self, user_id, poem_id, method="hybrid"):
        """预测用户对诗歌的评分"""
        user_interactions = self._get_user_interactions(user_id)

        if method == "cb":
            rated_poems = [
                p
                for p in self.poems
                if p["id"] in set(i["poem_id"] for i in user_interactions)
            ]
            ratings = [i["rating"] for i in user_interactions]
            user_profile = self.cb_recommender.get_user_profile(rated_poems, ratings)
            if user_profile is None:
                return 3.0
            poem_idx = next(
                (i for i, p in enumerate(self.poems) if p["id"] == poem_id), None
            )
            if poem_idx is None:
                return 3.0
            return self.cb_recommender.predict_rating(user_profile, poem_idx)

        elif method == "item_cf":
            return self.item_cf_recommender.predict_rating(user_interactions, poem_id)

        elif method == "bertopic":
            return self.bertopic_recommender.predict_rating(user_interactions, poem_id)

        elif method == "hybrid":
            cb_pred = self.predict_rating(user_id, poem_id, "cb")
            item_cf_pred = self.predict_rating(user_id, poem_id, "item_cf")
            bertopic_pred = self.predict_rating(user_id, poem_id, "bertopic")

            interaction_count = len(user_interactions)
            if interaction_count == 0:
                weights = {"cb": 0.3, "item_cf": 0.2, "bertopic": 0.5}
            elif interaction_count < 10:
                weights = {"cb": 0.3, "item_cf": 0.3, "bertopic": 0.4}
            else:
                weights = {"cb": 0.2, "item_cf": 0.3, "bertopic": 0.5}

            return (
                cb_pred * weights["cb"]
                + item_cf_pred * weights["item_cf"]
                + bertopic_pred * weights["bertopic"]
            )

        return 3.0

    def predict_all_ratings(self, user_id, method="hybrid"):
        """预测用户对所有诗歌的评分"""
        user_interactions = self._get_user_interactions(user_id)

        if method == "cb":
            rated_poems = [
                p
                for p in self.poems
                if p["id"] in set(i["poem_id"] for i in user_interactions)
            ]
            ratings = [i["rating"] for i in user_interactions]
            user_profile = self.cb_recommender.get_user_profile(rated_poems, ratings)
            if user_profile is None:
                return np.full(len(self.poems), 3.0)
            return self.cb_recommender.predict_all_ratings(user_profile)

        elif method == "item_cf":
            return self.item_cf_recommender.predict_all_ratings(user_interactions)

        elif method == "bertopic":
            return self.bertopic_recommender.predict_all_ratings(user_interactions)

        elif method == "hybrid":
            cb_preds = self.predict_all_ratings(user_id, "cb")
            item_cf_preds = self.predict_all_ratings(user_id, "item_cf")
            bertopic_preds = self.predict_all_ratings(user_id, "bertopic")

            interaction_count = len(user_interactions)
            if interaction_count == 0:
                weights = {"cb": 0.3, "item_cf": 0.2, "bertopic": 0.5}
            elif interaction_count < 10:
                weights = {"cb": 0.3, "item_cf": 0.3, "bertopic": 0.4}
            else:
                weights = {"cb": 0.2, "item_cf": 0.3, "bertopic": 0.5}

            return (
                cb_preds * weights["cb"]
                + item_cf_preds * weights["item_cf"]
                + bertopic_preds * weights["bertopic"]
            )

        return np.full(len(self.poems), 3.0)
