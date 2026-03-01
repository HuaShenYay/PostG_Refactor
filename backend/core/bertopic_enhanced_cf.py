import os
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity


class BERTopicEnhancedCF:
    """
    BERTopic增强的协同过滤算法 (三路融合版)
    
    核心创新：融合三种相似度
    - Item-CF: 基于评分矩阵的物品相似度
    - User-CF: 基于评分矩阵的用户相似度  
    - BERTopic: 主题向量相似度
    
    融合比例可配置，默认：
    - 0.5 × Item-CF + 0.3 × User-CF + 0.2 × BERTopic
    
    这样既保留了传统CF的能力，又利用BERTopic缓解数据稀疏问题。
    """

    def __init__(self, item_cf_weight=0.5, user_cf_weight=0.3, topic_weight=0.2):
        self.item_cf_weight = item_cf_weight
        self.user_cf_weight = user_cf_weight
        self.topic_weight = topic_weight
        
        self.item_cf = None
        self.bertopic = None
        
        self.poems = None
        self.interactions = None
        self.poem_ids = []
        self.poem_id_map = {}
        
        self.user_similarity = None
        self.user_id_map = {}
        self.item_similarity = None
        
        # 优化参数 - User-CF
        self.k_neighbors = 30  # Top-K neighbors
        self.min_similarity = 0.1  # 最小相似度阈值
        self.min_common_ratings = 3  # 最少共同评分数
        self.hybrid_alpha = 0.6  # 混合相似度: rating权重
        
        # NMF/SVD
        self.nmf_model = None
        self.user_factors = None
        self.item_factors = None
        
        # 时间衰减参数
        self.time_decay_half_life = 60  # 半衰期(天)
        self.recent_window = 5  # 最近交互boost窗口
        
        # 自适应融合
        self.enable_adaptive = False
        self.user_activity_bins = [10, 50, 100]
        self.user_id_map = {}
        self.item_similarity = None

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
        
        print("[BERTopicEnhancedCF] 构建评分矩阵...")
        self._build_rating_matrix(interactions)
        
        print("[BERTopicEnhancedCF] 计算Item-CF相似度...")
        self._compute_item_similarity()
        
        print("[BERTopicEnhancedCF] 计算User-CF相似度...")
        self._compute_user_similarity()
        
        print("[BERTopicEnhancedCF] 计算混合用户相似度 (Top-K + 置信度过滤)...")
        self._compute_hybrid_user_similarity()
        
        print("[BERTopicEnhancedCF] 训练BERTopic模型...")
        self._compute_user_similarity()
        
        print("[BERTopicEnhancedCF] 训练BERTopic模型...")
        from .bertopic_recommender import BertopicRecommender
        self.bertopic = BertopicRecommender()
        self.bertopic.fit(poems, interactions)
        
        print("[BERTopicEnhancedCF] 计算增强相似度矩阵...")
        self._compute_enhanced_similarity()
        
        print(f"[BERTopicEnhancedCF] 训练完成")

    def _build_rating_matrix(self, interactions):
        """构建用户-物品评分矩阵"""
        users = set(i["user_id"] for i in interactions)
        self.user_id_map = {uid: idx for idx, uid in enumerate(users)}
        self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_map.items()}
        
        n_users = len(users)
        n_items = len(self.poem_ids)
        
        self.rating_matrix = np.zeros((n_users, n_items))
        
        for inter in interactions:
            u_idx = self.user_id_map[inter["user_id"]]
            p_idx = self.poem_id_map.get(inter["poem_id"])
            if p_idx is not None:
                self.rating_matrix[u_idx, p_idx] = inter.get("rating", 3.0)
        
        print(f"[BERTopicEnhancedCF] 评分矩阵: {self.rating_matrix.shape}")

    def _compute_item_similarity(self):
        """计算物品相似度矩阵 (Item-CF)"""
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

        print(f"[BERTopicEnhancedCF] Item相似度矩阵完成")

    def _compute_user_similarity(self):
        """计算用户相似度矩阵 (User-CF)"""
        n_users = self.rating_matrix.shape[0]
        self.user_similarity = np.zeros((n_users, n_users))

        for i in range(n_users):
            for j in range(i, n_users):
                if i == j:
                    self.user_similarity[i, j] = 1.0
                    continue
                
                mask = (self.rating_matrix[i, :] > 0) & (self.rating_matrix[j, :] > 0)
                if mask.sum() == 0:
                    similarity = 0.0
                else:
                    vec_i = self.rating_matrix[i, mask]
                    vec_j = self.rating_matrix[j, mask]
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

                self.user_similarity[i, j] = similarity
                self.user_similarity[j, i] = similarity

        print(f"[BERTopicEnhancedCF] User相似度矩阵完成")

    def _min_max_normalize(self, matrix):
        """Min-Max归一化到[0, 1]"""
        min_val = matrix.min()
        max_val = matrix.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(matrix)
        return (matrix - min_val) / (max_val - min_val)

    def _compute_enhanced_similarity(self):
        """计算增强的物品相似度矩阵（三路融合归一化后）"""
        n_items = len(self.poem_ids)
        
        item_sim = self.item_similarity
        
        if self.bertopic.topic_matrix is not None:
            topic_sim = cosine_similarity(self.bertopic.topic_matrix)
        else:
            topic_sim = np.zeros((n_items, n_items))
            np.fill_diagonal(topic_sim, 1.0)
        
        if item_sim.shape != topic_sim.shape:
            min_dim = min(item_sim.shape[0], topic_sim.shape[0])
            item_sim = item_sim[:min_dim, :min_dim]
            topic_sim = topic_sim[:min_dim, :min_dim]
            n_items = min_dim
        
        item_norm = self._min_max_normalize(item_sim[:n_items, :n_items])
        topic_norm = self._min_max_normalize(topic_sim[:n_items, :n_items])
        
        self.enhanced_similarity = (
            self.item_cf_weight * item_norm + 
            self.topic_weight * topic_norm
        )
        
        print(f"[BERTopicEnhancedCF] 融合完成: {self.item_cf_weight}×Item-CF + {self.topic_weight}×主题")

    def _compute_hybrid_user_similarity(self):
        """计算混合用户相似度 (Rating + BERTopic主题向量)"""
        n_users = self.rating_matrix.shape[0]
        
        # 1. Rating-based similarity (Pearson)
        rating_sim = self.user_similarity.copy()
        
        # 2. Topic-based similarity
        if self.bertopic.topic_matrix is not None:
            # 构建用户主题向量: 用户评分过的物品的主题向量加权平均
            user_topic_vectors = np.zeros((n_users, self.bertopic.topic_matrix.shape[1]))
            
            for u_idx in range(n_users):
                rated_items = self.rating_matrix[u_idx] > 0
                if rated_items.sum() > 0:
                    ratings = self.rating_matrix[u_idx, rated_items]
                    topic_vecs = self.bertopic.topic_matrix[rated_items]
                    # 评分加权
                    weights = ratings / (ratings.sum() + 1e-8)
                    user_topic_vectors[u_idx] = np.sum(topic_vecs * weights[:, np.newaxis], axis=0)
            
            # 计算主题相似度
            topic_sim = cosine_similarity(user_topic_vectors)
            
            # 3. 混合: alpha * rating + (1-alpha) * topic
            self.hybrid_similarity = (
                self.hybrid_alpha * rating_sim + 
                (1 - self.hybrid_alpha) * topic_sim
            )
        else:
            self.hybrid_similarity = rating_sim
        
        print(f"[BERTopicEnhancedCF] 混合用户相似度完成 (α={self.hybrid_alpha})")
    
    def _apply_confidence_filter(self, similarity_matrix):
        """应用置信度过滤: 相似度阈值 + 最少共同评分数"""
        n_users = similarity_matrix.shape[0]
        filtered_sim = similarity_matrix.copy()
        
        for i in range(n_users):
            for j in range(i + 1, n_users):
                # 计算共同评分数
                common_mask = (self.rating_matrix[i] > 0) & (self.rating_matrix[j] > 0)
                common_count = common_mask.sum()
                
                # 共同评分过少, 降低置信度
                if common_count < self.min_common_ratings:
                    shrinkage = common_count / (common_count + 10)  # Jaccard shrinkage
                    filtered_sim[i, j] *= shrinkage
                    filtered_sim[j, i] *= shrinkage
                
                # 相似度过低, 直接过滤
                if abs(filtered_sim[i, j]) < self.min_similarity:
                    filtered_sim[i, j] = 0
                    filtered_sim[j, i] = 0
        
        return filtered_sim
    
    def _get_top_k_neighbors(self, target_idx, similarity_matrix, k=None):
        """获取Top-K最相似用户 (带置信度过滤)"""
        if k is None:
            k = self.k_neighbors
        
        sims = similarity_matrix[target_idx].copy()
        # 排除自己
        sims[target_idx] = -np.inf
        
        # 获取top-k
        top_k_indices = np.argsort(sims)[-k:]
        top_k_sims = sims[top_k_indices]
        
        # 过滤负相似度
        valid_mask = top_k_sims > 0
        return list(zip(top_k_indices[valid_mask], top_k_sims[valid_mask]))
    
    def _get_user_cf_scores_upgraded(self, user_interactions, exclude_ids):
        """升级版User-CF: Top-K + 混合相似度 + 置信度过滤"""
        if not user_interactions or not hasattr(self, 'hybrid_similarity'):
            return self._get_user_cf_scores_fallback(user_interactions, exclude_ids)
        
        target_idx = self.user_id_map.get(user_interactions[0]["user_id"])
        if target_idx is None:
            return self._get_user_cf_scores_fallback(user_interactions, exclude_ids)
        
        # 构建用户已评分字典
        target_ratings = {}
        for inter in user_interactions:
            if inter["poem_id"] in self.poem_id_map:
                target_ratings[inter["poem_id"]] = inter.get("rating", 3.0)
        
        if not target_ratings:
            return {}
        
        # 获取Top-K邻居 (带置信度过滤)
        filtered_sim = self._apply_confidence_filter(self.hybrid_similarity)
        neighbors = self._get_top_k_neighbors(target_idx, filtered_sim)
        
        if not neighbors:
            return {}
        
        # 预测评分: 加权平均
        scores = {}
        for item_idx in range(len(self.poem_ids)):
            item_id = self.poem_ids[item_idx]
            if item_id in exclude_ids or item_id in target_ratings:
                continue
            
            weighted_sum = 0.0
            sim_sum = 0.0
            
            for neighbor_idx, sim in neighbors:
                neighbor_rating = self.rating_matrix[neighbor_idx, item_idx]
                if neighbor_rating > 0:
                    # 用户均值中心化
                    neighbor_rated = self.rating_matrix[neighbor_idx] > 0
                    if neighbor_rated.sum() > 0:
                        neighbor_mean = self.rating_matrix[neighbor_idx, neighbor_rated].mean()
                        weighted_sum += sim * (neighbor_rating - neighbor_mean)
                        sim_sum += abs(sim)
            
            if sim_sum > 0:
                target_rated = self.rating_matrix[target_idx] > 0
                if target_rated.sum() > 0:
                    target_mean = self.rating_matrix[target_idx, target_rated].mean()
                    pred = target_mean + weighted_sum / sim_sum
                    scores[item_id] = pred
        
        return scores
    
    def _get_user_cf_scores_fallback(self, user_interactions, exclude_ids):
        """降级版User-CF (原版逻辑)"""
        if not user_interactions or self.user_similarity is None:
            return {}
        
        target_idx = self.user_id_map.get(user_interactions[0]["user_id"])
        if target_idx is None:
            return {}
        
        target_ratings = {}
        for inter in user_interactions:
            if inter["poem_id"] in self.poem_id_map:
                target_ratings[inter["poem_id"]] = inter.get("rating", 3.0)
        
        if not target_ratings:
            return {}
        
        # 找最相似用户
        best_sim_user = None
        best_sim = -1
        for u_idx in range(self.user_similarity.shape[0]):
            if u_idx != target_idx and self.user_similarity[target_idx, u_idx] > best_sim:
                best_sim = self.user_similarity[target_idx, u_idx]
                best_sim_user = u_idx
        
        if best_sim_user is None:
            return {}
        
        scores = {}
        for item_idx in range(len(self.poem_ids)):
            item_id = self.poem_ids[item_idx]
            if item_id in exclude_ids or item_id in target_ratings:
                continue
            
            rating = self.rating_matrix[best_sim_user, item_idx]
            if rating > 0:
                scores[item_id] = rating * best_sim
        
        return scores


    def _get_user_cf_scores(self, user_interactions, all_items, exclude_ids):
        """计算User-CF的推荐分数"""
        if not user_interactions or self.user_similarity is None:
            return {}
        
        target_user_idx = None
        target_ratings = {}
        for inter in user_interactions:
            if inter["poem_id"] in self.poem_id_map:
                target_ratings[inter["poem_id"]] = inter.get("rating", 3.0)
        
        if not target_ratings:
            return {}
        
        best_sim_user = None
        best_sim = -1
        target_idx = self.user_id_map.get(user_interactions[0]["user_id"])
        
        if target_idx is not None:
            for u_idx in range(self.user_similarity.shape[0]):
                if u_idx != target_idx and self.user_similarity[target_idx, u_idx] > best_sim:
                    best_sim = self.user_similarity[target_idx, u_idx]
                    best_sim_user = u_idx
        
        if best_sim_user is None:
            return {}
        
        scores = {}
        for item_idx in range(len(self.poem_ids)):
            item_id = self.poem_ids[item_idx]
            if item_id in exclude_ids or item_id in target_ratings:
                continue
            
            rating = self.rating_matrix[best_sim_user, item_idx]
            if rating > 0:
                scores[item_id] = rating * best_sim
        
        return scores

    def recommend(self, user_interactions, all_interactions, top_k=10):
        """
        为用户推荐诗歌（三路融合）
        
        Args:
            user_interactions: 当前用户的交互历史
            all_interactions: 所有用户的交互历史
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
        
        item_scores = np.zeros(len(self.poem_ids))
        for i in range(len(self.poem_ids)):
            if user_ratings[i] > 0:
                continue
            
            neighbors = self.enhanced_similarity[i, rated_indices]
            neighbor_ratings = user_ratings[rated_indices]
            
            pos_mask = neighbors > 0
            if pos_mask.sum() > 0:
                item_scores[i] = np.dot(neighbors[pos_mask], neighbor_ratings[pos_mask]) / (np.abs(neighbors[pos_mask]).sum() + 1e-8)
        
        
        # 升级版User-CF: Top-K + 混合相似度 + 置信度过滤
        user_cf_scores = self._get_user_cf_scores_upgraded(user_interactions, exclude_ids)
        
        # 如果升级版失败, 降级到原版
        if not user_cf_scores:
            user_cf_scores = self._get_user_cf_scores_fallback(user_interactions, exclude_ids)

        
        results = []
        for idx, item_score in enumerate(item_scores):
            poem_id = self.poem_ids[idx]
            if poem_id in exclude_ids:
                continue
            
            score = item_score
            
            if poem_id in user_cf_scores and self.user_cf_weight > 0:
                user_cf_score = user_cf_scores[poem_id]
                max_user_cf = max(user_cf_scores.values()) if user_cf_scores else 1
                user_cf_normalized = user_cf_score / max_user_cf if max_user_cf > 0 else 0
                score = (1 - self.user_cf_weight) * score + self.user_cf_weight * user_cf_normalized
            
            if score > 0:
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
        """预测用户对诗歌的评分"""
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

    # ========== NMF/SVD 矩阵分解 ==========
    def fit_nmf(self, n_components=20):
        """使用NMF进行矩阵分解, 缓解稀疏性问题"""
        try:
            from sklearn.decomposition import NMF
            
            print(f"[BERTopicEnhancedCF] 训练NMF模型 (components={n_components})...")
            
            # 初始化缺失值为用户/物品平均 (K1=25, K2=10 平滑)
            R = self.rating_matrix.copy()
            R_init = self._initialize_missing_values(R)
            
            # 训练NMF
            self.nmf_model = NMF(n_components=n_components, init='random', random_state=42, max_iter=200)
            self.user_factors = self.nmf_model.fit_transform(R_init)
            self.item_factors = self.nmf_model.components_
            
            print(f"[BERTopicEnhancedCF] NMF完成: user_factors={self.user_factors.shape}, item_factors={self.item_factors.shape}")
            
        except ImportError:
            print("[BERTopicEnhancedCF] NMF不可用, 跳过矩阵分解")
    
    def _initialize_missing_values(self, R, K1=25, K2=10):
        """初始化缺失值为修正均值, 缓解冷启动"""
        R_init = R.copy()
        
        # 计算全局均值
        mask = R > 0
        if mask.sum() == 0:
            return R_init
        
        global_mean = R[mask].mean()
        
        # 计算用户和物品均值
        user_means = np.zeros(R.shape[0])
        item_means = np.zeros(R.shape[1])
        
        for u in range(R.shape[0]):
            user_ratings = R[u, R[u] > 0]
            user_means[u] = user_ratings.mean() if len(user_ratings) > 0 else global_mean
        
        for i in range(R.shape[1]):
            item_ratings = R[R[:, i] > 0, i]
            item_means[i] = item_ratings.mean() if len(item_ratings) > 0 else global_mean
        
        # 填充缺失值
        for u in range(R.shape[0]):
            for i in range(R.shape[1]):
                if R[u, i] == 0:
                    # 加权组合
                    R_init[u, i] = (K1 * user_means[u] + K2 * item_means[i]) / (K1 + K2)
        
        return R_init
    
    def predict_nmf(self, user_idx, item_idx):
        """使用NMF预测评分"""
        if self.user_factors is None or self.item_factors is None:
            return None
        
        pred = np.dot(self.user_factors[user_idx], self.item_factors[:, item_idx])
        return float(np.clip(pred, 1.0, 5.0))
    
    def get_nmf_scores(self, user_interactions, exclude_ids):
        """获取NMF推荐的分数"""
        if self.user_factors is None or self.item_factors is None:
            return {}
        
        target_idx = self.user_id_map.get(user_interactions[0]["user_id"])
        if target_idx is None:
            return {}
        
        scores = {}
        user_vec = self.user_factors[target_idx]
        
        for item_idx in range(len(self.poem_ids)):
            item_id = self.poem_ids[item_idx]
            if item_id in exclude_ids:
                continue
            
            pred = np.dot(user_vec, self.item_factors[:, item_idx])
            scores[item_id] = float(np.clip(pred, 1.0, 5.0))
        
        return scores
    
    # ========== 时间衰减 ==========
    def _compute_time_decay_weights(self, interactions):
        """计算时间衰减权重"""
        if not interactions:
            return {}
        
        # 找到最新时间
        timestamps = []
        for inter in interactions:
            if 'created_at' in inter:
                ts = inter['created_at']
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                timestamps.append(ts)
        
        if not timestamps:
            return {}
        
        latest = max(timestamps)
        
        weights = {}
        half_life = self.time_decay_half_life
        
        for inter in interactions:
            uid = inter['user_id']
            if uid not in weights:
                weights[uid] = {}
            
            if 'created_at' in inter:
                ts = inter['created_at']
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                
                # 指数衰减
                days_diff = (latest - ts).days
                weight = np.power(0.5, days_diff / half_life)
                
                # 最近窗口boost
                if days_diff <= self.recent_window:
                    boost = 1.0 + (1.0 - days_diff / self.recent_window)
                    weight *= boost
                
                weights[uid][inter['poem_id']] = weight
            else:
                weights[uid][inter['poem_id']] = 1.0
        
        return weights
    
    def _apply_time_decay_to_ratings(self, user_interactions):
        """对用户评分应用时间衰减"""
        if not user_interactions:
            return user_interactions
        
        # 尝试获取时间戳
        has_time = any('created_at' in inter for inter in user_interactions)
        if not has_time:
            return user_interactions
        
        # 找到最新时间
        timestamps = []
        for inter in user_interactions:
            if 'created_at' in inter:
                ts = inter['created_at']
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                timestamps.append(ts)
        
        if not timestamps:
            return user_interactions
        
        latest = max(timestamps)
        half_life = self.time_decay_half_life
        
        weighted_interactions = []
        for inter in user_interactions:
            new_inter = inter.copy()
            
            if 'created_at' in inter:
                ts = inter['created_at']
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                
                days_diff = (latest - ts).days
                weight = np.power(0.5, days_diff / half_life)
                
                # 最近窗口boost
                if days_diff <= self.recent_window:
                    boost = 1.0 + (1.0 - days_diff / self.recent_window)
                    weight *= boost
                
                new_inter['weighted_rating'] = inter.get('rating', 3.0) * weight
            else:
                new_inter['weighted_rating'] = inter.get('rating', 3.0)
            
            weighted_interactions.append(new_inter)
        
        return weighted_interactions
    
    # ========== 自适应融合 ==========
    def _compute_adaptive_weights(self, user_interactions):
        """
        根据用户活跃度自适应调整融合权重
        - 活跃用户: 更多依赖CF
        - 不活跃用户: 更多依赖BERTopic主题
        """
        n_ratings = len(user_interactions)
        
        if n_ratings >= self.user_activity_bins[2]:
            # 高度活跃: 0.6 Item-CF + 0.3 User-CF + 0.1 BERTopic
            return 0.6, 0.3, 0.1
        elif n_ratings >= self.user_activity_bins[1]:
            # 中度活跃: 0.5 Item-CF + 0.3 User-CF + 0.2 BERTopic
            return 0.5, 0.3, 0.2
        elif n_ratings >= self.user_activity_bins[0]:
            # 低度活跃: 0.4 Item-CF + 0.2 User-CF + 0.4 BERTopic
            return 0.4, 0.2, 0.4
        else:
            # 冷启动: 0.3 Item-CF + 0.1 User-CF + 0.6 BERTopic
            return 0.3, 0.1, 0.6
    
    def recommend_with_all_features(self, user_interactions, all_interactions, top_k=10,
                                      use_nmf=False, use_time_decay=False, use_adaptive=False):
        """
        完整版推荐: 包含所有优化特性
        
        Args:
            use_nmf: 是否使用NMF
            use_time_decay: 是否使用时间衰减
            use_adaptive: 是否使用自适应权重
        """
        # 时间衰减
        if use_time_decay:
            user_interactions = self._apply_time_decay_to_ratings(user_interactions)
        
        # 自适应权重
        if use_adaptive:
            item_cf_w, user_cf_w, topic_w = self._compute_adaptive_weights(user_interactions)
        else:
            item_cf_w, user_cf_w, topic_w = self.item_cf_weight, self.user_cf_weight, self.topic_weight
        
        # 基础评分
        base_scores = self._get_base_item_scores(user_interactions)
        
        # NMF评分
        nmf_scores = {}
        if use_nmf and self.user_factors is not None:
            exclude_ids = set(i["poem_id"] for i in user_interactions)
            nmf_scores = self.get_nmf_scores(user_interactions, exclude_ids)
        
        # User-CF评分
        exclude_ids = set(i["poem_id"] for i in user_interactions)
        user_cf_scores = self._get_user_cf_scores_upgraded(user_interactions, exclude_ids)
        if not user_cf_scores:
            user_cf_scores = self._get_user_cf_scores_fallback(user_interactions, exclude_ids)
        
        # 融合所有分数
        results = []
        for idx in range(len(self.poem_ids)):
            poem_id = self.poem_ids[idx]
            if poem_id in exclude_ids:
                continue
            
            score = 0.0
            
            # Item-CF
            if poem_id in base_scores:
                score += item_cf_w * base_scores[poem_id]
            
            # User-CF
            if poem_id in user_cf_scores:
                # 归一化
                max_ucf = max(user_cf_scores.values()) if user_cf_scores else 1
                user_cf_norm = user_cf_scores[poem_id] / max_ucf if max_ucf > 0 else 0
                score += user_cf_w * user_cf_norm
            
            # BERTopic
            score += topic_w * base_scores.get(poem_id, 0)
            
            # NMF
            if use_nmf and poem_id in nmf_scores:
                # NMF作为额外信号
                score += 0.1 * (nmf_scores[poem_id] / 5.0)
            
            if score > 0:
                results.append({"poem_id": poem_id, "score": float(score)})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def _get_base_item_scores(self, user_interactions):
        """获取Item-CF基础分数"""
        exclude_ids = set(i["poem_id"] for i in user_interactions)
        
        user_ratings = np.zeros(len(self.poem_ids))
        for inter in user_interactions:
            if inter["poem_id"] in self.poem_id_map:
                idx = self.poem_id_map[inter["poem_id"]]
                rating = inter.get("weighted_rating", inter.get("rating", 3.0))
                user_ratings[idx] = rating
        
        rated_indices = np.where(user_ratings > 0)[0]
        if len(rated_indices) == 0:
            return {}
        
        item_scores = {}
        for i in range(len(self.poem_ids)):
            if user_ratings[i] > 0:
                continue
            
            neighbors = self.enhanced_similarity[i, rated_indices]
            neighbor_ratings = user_ratings[rated_indices]
            
            pos_mask = neighbors > 0
            if pos_mask.sum() > 0:
                score = np.dot(neighbors[pos_mask], neighbor_ratings[pos_mask]) / (np.abs(neighbors[pos_mask]).sum() + 1e-8)
                item_scores[self.poem_ids[i]] = score
        
        return item_scores

