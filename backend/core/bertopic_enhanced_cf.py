import os
import numpy as np
from collections import Counter
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity


class BERTopicEnhancedCF:
    """
    BERTopic增强的协同过滤算法
    
    核心创新：融合评分矩阵相似度 + BERTopic主题向量相似度
    融合比例：0.6 × 评分相似度 + 0.4 × 主题相似度
    
    这样既保留了传统CF基于用户行为的能力，又利用BERTopic
    缓解数据稀疏问题，提升推荐效果。
    """

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
        """
        训练模型
        
        Args:
            poems: list of dict with 'id', 'content'
            interactions: list of dict with 'user_id', 'poem_id', 'rating', 'created_at'
        """
        self.poems = poems
        self.interactions = interactions
        self.poem_ids = [p["id"] for p in poems]
        self.poem_id_map = {pid: idx for idx, pid in enumerate(self.poem_ids)}
        
        print("[BERTopicEnhancedCF] 初始化组件...")
        
        print("[BERTopicEnhancedCF] 训练Item-CF模型...")
        from .collaborative_filter import ItemBasedCFRecommender
        self.item_cf = ItemBasedCFRecommender()
        self.item_cf.fit(interactions, self.poem_ids)
        
        print("[BERTopicEnhancedCF] 训练BERTopic模型...")
        from .bertopic_recommender import BertopicRecommender
        self.bertopic = BertopicRecommender()
        self.bertopic.fit(poems, interactions)
        
        print("[BERTopicEnhancedCF] 计算增强相似度矩阵...")
        self._compute_enhanced_similarity()
        
        print(f"[BERTopicEnhancedCF] 训练完成，相似度矩阵: {self.enhanced_similarity.shape}")

    def _min_max_normalize(self, matrix):
        """Min-Max归一化到[0, 1]"""
        min_val = matrix.min()
        max_val = matrix.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(matrix)
        return (matrix - min_val) / (max_val - min_val)

    def _compute_enhanced_similarity(self):
        """计算增强的物品相似度矩阵（归一化后融合）"""
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
        
        print(f"[BERTopicEnhancedCF] 融合完成: {self.rating_weight}×评分(归一化) + {self.topic_weight}×主题(归一化)")

    def recommend(self, user_interactions, all_interactions, top_k=10):
        """
        为用户推荐诗歌
        
        Args:
            user_interactions: 当前用户的交互历史
            all_interactions: 所有用户的交互历史（未使用，保留接口）
            top_k: 推荐数量
            
        Returns:
            list of dict: recommended poems with scores
        """
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
        """热门推荐备选方案"""
        exclude_ids = exclude_ids or set()
        poem_scores = Counter()
        for inter in self.interactions:
            if inter["poem_id"] not in exclude_ids:
                poem_scores[inter["poem_id"]] += inter.get("rating", 3.0)
        
        return [{"poem_id": pid, "score": float(score)} for pid, score in poem_scores.most_common(top_k)]

    def predict_rating(self, user_interactions, poem_id):
        """
        预测用户对诗歌的评分
        
        Args:
            user_interactions: 用户交互历史
            poem_id: 诗歌ID
            
        Returns:
            float: 预测评分 (1-5)
        """
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

    def predict_all_ratings(self, user_interactions):
        """预测用户对所有诗歌的评分"""
        n_items = len(self.poem_ids)
        
        if self.enhanced_similarity is None:
            return np.full(n_items, 3.0)
        
        user_ratings = np.zeros(n_items)
        for inter in user_interactions:
            if inter["poem_id"] in self.poem_id_map:
                idx = self.poem_id_map[inter["poem_id"]]
                user_ratings[idx] = inter.get("rating", 3.0)
        
        rated_indices = np.where(user_ratings > 0)[0]
        
        predictions = np.zeros(n_items)
        for i in range(n_items):
            if user_ratings[i] > 0:
                predictions[i] = user_ratings[i]
                continue
            
            neighbors = self.enhanced_similarity[i, rated_indices]
            neighbor_ratings = user_ratings[rated_indices]
            
            pos_mask = neighbors > 0
            if pos_mask.sum() > 0:
                predictions[i] = np.clip(
                    np.dot(neighbors[pos_mask], neighbor_ratings[pos_mask]) / (np.abs(neighbors[pos_mask]).sum() + 1e-8),
                    1.0, 5.0
                )
            else:
                predictions[i] = 3.0
        
        return predictions
