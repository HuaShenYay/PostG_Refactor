import numpy as np
from collections import Counter
from datetime import datetime, timedelta
import math


class ItemBasedCFRecommender:
    """
    基于物品的协同过滤算法 (Item-Based Collaborative Filtering)
    使用评分矩阵计算物品相似度进行推荐
    """

    def __init__(self, k_neighbors=30):
        self.k_neighbors = k_neighbors
        self.item_similarity = None
        self.rating_matrix = None
        self.poem_id_to_idx = None
        self.idx_to_poem_id = None

    def fit(self, interactions, poem_ids):
        """
        训练模型：构建评分矩阵和物品相似度矩阵

        Args:
            interactions: list of dict, each contains 'user_id', 'poem_id', 'rating', 'created_at'
            poem_ids: list of all poem ids
        """
        self.poem_id_to_idx = {pid: idx for idx, pid in enumerate(poem_ids)}
        self.idx_to_poem_id = {idx: pid for pid, idx in self.poem_id_to_idx.items()}

        users = set(i["user_id"] for i in interactions)
        user_id_to_idx = {uid: idx for idx, uid in enumerate(users)}

        n_users = len(users)
        n_items = len(poem_ids)

        self.rating_matrix = np.zeros((n_users, n_items))

        for inter in interactions:
            u_idx = user_id_to_idx[inter["user_id"]]
            p_idx = self.poem_id_to_idx[inter["poem_id"]]
            self.rating_matrix[u_idx, p_idx] = inter.get("rating", 3.0)

        self._compute_similarity()

        print(f"[Item-CF] 评分矩阵构建完成: {self.rating_matrix.shape}")
        print(
            f"[Item-CF] 矩阵密度: {(self.rating_matrix > 0).sum() / (n_users * n_items):.2%}"
        )

    def _compute_similarity(self):
        """计算物品相似度矩阵"""
        n_items = self.rating_matrix.shape[1]
        self.item_similarity = np.zeros((n_items, n_items))

        for i in range(n_items):
            for j in range(i, n_items):
                if i == j:
                    self.item_similarity[i, j] = 1.0
                    continue

                mask = (self.rating_matrix[:, i] > 0) & (self.rating_matrix[:, j] > 0)
                if mask.sum() == 0:
                    similarity = 0.0
                else:
                    vec_i = self.rating_matrix[mask, i]
                    vec_j = self.rating_matrix[mask, j]
                    mean_i = vec_i.mean()
                    mean_j = vec_j.mean()

                    if mean_i > 0 and mean_j > 0:
                        sim = np.sum((vec_i - mean_i) * (vec_j - mean_j)) / (
                            np.sqrt(np.sum((vec_i - mean_i) ** 2))
                            * np.sqrt(np.sum((vec_j - mean_j) ** 2))
                            + 1e-8
                        )
                        similarity = sim
                    else:
                        similarity = 0.0

                self.item_similarity[i, j] = similarity
                self.item_similarity[j, i] = similarity

        print(f"[Item-CF] 相似度矩阵构建完成")

    def get_user_ratings(self, user_id, user_interactions):
        """获取用户评分向量"""
        n_items = len(self.poem_id_to_idx)
        ratings = np.zeros(n_items)

        for inter in user_interactions:
            if inter["poem_id"] in self.poem_id_to_idx:
                p_idx = self.poem_id_to_idx[inter["poem_id"]]
                ratings[p_idx] = inter.get("rating", 3.0)

        return ratings

    def recommend(self, user_interactions, exclude_ids=None, top_k=10):
        """
        为用户推荐诗歌

        Args:
            user_interactions: list of dicts with 'poem_id', 'rating'
            exclude_ids: set of poem ids to exclude
            top_k: number of recommendations

        Returns:
            list of dict: recommended poems with scores
        """
        exclude_ids = exclude_ids or set()

        user_ratings = self.get_user_ratings(None, user_interactions)
        rated_items = np.where(user_ratings > 0)[0]

        if len(rated_items) == 0:
            return []

        scores = np.zeros(len(self.poem_id_to_idx))

        for item_idx in range(len(self.poem_id_to_idx)):
            if user_ratings[item_idx] > 0:
                continue

            neighbors = self.item_similarity[item_idx, rated_items]
            neighbor_ratings = user_ratings[rated_items]

            pos_mask = neighbors > 0
            if pos_mask.sum() > 0:
                scores[item_idx] = np.dot(
                    neighbors[pos_mask], neighbor_ratings[pos_mask]
                ) / (np.abs(neighbors[pos_mask]).sum() + 1e-8)

        results = []
        for idx, score in enumerate(scores):
            poem_id = self.idx_to_poem_id[idx]
            if poem_id not in exclude_ids and user_ratings[idx] == 0:
                results.append({"poem_id": poem_id, "score": float(score), "title": ""})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def predict_rating(self, user_interactions, poem_id):
        """预测用户对物品的评分"""
        if poem_id not in self.poem_id_to_idx:
            return 3.0

        poem_idx = self.poem_id_to_idx[poem_id]
        user_ratings = self.get_user_ratings(None, user_interactions)
        rated_items = np.where(user_ratings > 0)[0]

        if len(rated_items) == 0:
            return 3.0

        neighbors = self.item_similarity[poem_idx, rated_items]
        neighbor_ratings = user_ratings[rated_items]

        pos_mask = neighbors > 0
        if pos_mask.sum() > 0:
            return np.clip(
                np.dot(neighbors[pos_mask], neighbor_ratings[pos_mask])
                / (np.abs(neighbors[pos_mask]).sum() + 1e-8),
                1.0,
                5.0,
            )
        return 3.0

    def predict_all_ratings(self, user_interactions):
        """预测用户对所有物品的评分"""
        n_items = len(self.poem_id_to_idx)
        user_ratings = self.get_user_ratings(None, user_interactions)
        rated_items = np.where(user_ratings > 0)[0]

        if len(rated_items) == 0:
            return np.full(n_items, 3.0)

        predictions = np.zeros(n_items)

        for item_idx in range(n_items):
            if user_ratings[item_idx] > 0:
                predictions[item_idx] = user_ratings[item_idx]
                continue

            neighbors = self.item_similarity[item_idx, rated_items]
            neighbor_ratings = user_ratings[rated_items]

            pos_mask = neighbors > 0
            if pos_mask.sum() > 0:
                predictions[item_idx] = np.clip(
                    np.dot(neighbors[pos_mask], neighbor_ratings[pos_mask])
                    / (np.abs(neighbors[pos_mask]).sum() + 1e-8),
                    1.0,
                    5.0,
                )
            else:
                predictions[item_idx] = 3.0

        return predictions
