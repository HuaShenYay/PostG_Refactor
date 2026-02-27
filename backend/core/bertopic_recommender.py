import os
import sys
import json
import math
import numpy as np
from collections import Counter
from datetime import datetime

np = None
cosine_similarity = None
SentenceTransformer = None
BERTopic = None
CountVectorizer = None
util = None


def _lazy_load_deps():
    """延迟加载依赖"""
    global np, cosine_similarity, SentenceTransformer, BERTopic, CountVectorizer, util

    if np is None:
        import numpy as _np

        np = _np

    if cosine_similarity is None:
        from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity

        cosine_similarity = _cosine_similarity

    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as _ST, util as _util

        SentenceTransformer = _ST
        util = _util

    if BERTopic is None:
        from bertopic import BERTopic as _BERTopic
        from sklearn.feature_extraction.text import CountVectorizer as _CV

        BERTopic = _BERTopic
        CountVectorizer = _CV


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models", "bertopic_model")


class BertopicRecommender:
    """
    BERTopic 混合推荐系统

    使用 BERTopic 进行语义向量化，结合多种推荐策略：
    - 用户协同过滤 (User-CF)
    - 物品协同过滤 (Item-CF)
    - 内容推荐 (Content-Based)
    - 热门推荐 (Popularity)
    """

    def __init__(self, cache_dir=None):
        _lazy_load_deps()

        self.bertopic_model = None
        self.embedding_model = None
        self.topic_matrix = None
        self.poem_id_map = {}
        self.poem_ids = []

        self.cache_dir = cache_dir or os.path.join(
            BASE_DIR, "saved_models", "vector_cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        self.user_vector_cache = {}
        self.cache_ttl = 300

    def _ensure_model_loaded(self):
        """确保模型已加载"""
        if self.bertopic_model is not None:
            return True

        if os.path.exists(MODEL_DIR):
            try:
                device = "cpu"
                try:
                    import torch_directml

                    device = torch_directml.device()
                except:
                    pass

                print("[BERTopic] Loading embedding model...")
                self.embedding_model = SentenceTransformer(
                    "paraphrase-multilingual-MiniLM-L12-v2", device=device
                )

                print("[BERTopic] Loading BERTopic model...")
                vectorizer = CountVectorizer(tokenizer=self._tokenize_zh)
                self.bertopic_model = BERTopic.load(
                    MODEL_DIR,
                    embedding_model=self.embedding_model,
                    vectorizer_model=vectorizer,
                )
                print("[BERTopic] Model loaded successfully")
                return True
            except Exception as e:
                print(f"[BERTopic] Failed to load model: {e}")
                return False
        return False

    def _tokenize_zh(self, text):
        """中文分词"""
        import jieba

        chinese_only = "".join(c for c in text if "\u4e00" <= c <= "\u9fff")
        words = jieba.lcut(chinese_only)
        return [w for w in words if len(w) > 1]

    def _build_poem_vector_matrix(self, poems):
        """构建诗歌向量矩阵"""
        if not self._ensure_model_loaded():
            return False

        self.poem_ids = [p["id"] for p in poems]
        self.poem_id_map = {pid: idx for idx, pid in enumerate(self.poem_ids)}

        matrix_path = os.path.join(self.cache_dir, "topic_matrix.npy")
        ids_path = os.path.join(self.cache_dir, "poem_ids.json")

        cache_valid = False
        if os.path.exists(matrix_path) and os.path.exists(ids_path):
            try:
                with open(ids_path, "r") as f:
                    cached_ids = json.load(f)
                if cached_ids == self.poem_ids:
                    self.topic_matrix = np.load(matrix_path)
                    print(
                        f"[BERTopic] Loaded {len(self.poem_ids)} poem vectors from cache"
                    )
                    cache_valid = True
            except:
                pass

        if not cache_valid:
            contents = [p.get("content", "") for p in poems]
            print(f"[BERTopic] Building vector matrix for {len(poems)} poems...")

            embeddings = self.embedding_model.encode(contents, show_progress_bar=True)
            self.topic_matrix = embeddings

            try:
                np.save(matrix_path, self.topic_matrix)
                with open(ids_path, "w") as f:
                    json.dump(self.poem_ids, f)
                print("[BERTopic] Vector matrix cached")
            except Exception as e:
                print(f"[BERTopic] Cache save failed: {e}")

        return True

    def fit(self, poems, interactions):
        """
        训练模型

        Args:
            poems: list of dict with 'id', 'content'
            interactions: list of dict with 'user_id', 'poem_id', 'rating', 'created_at'
        """
        self._build_poem_vector_matrix(poems)
        self.interactions = interactions
        print(f"[BERTopic] Recommender initialized")

    def _get_user_profile_vector(self, user_interactions):
        """构建用户偏好向量"""
        if self.topic_matrix is None or not user_interactions:
            return None

        user_vector = np.zeros(self.topic_matrix.shape[1])
        weight_sum = 0.0
        now = datetime.utcnow()

        for inter in user_interactions:
            poem_idx = self.poem_id_map.get(inter["poem_id"])
            if poem_idx is None:
                continue

            age_days = (now - inter.get("created_at", now)).total_seconds() / 86400
            decay = math.exp(-age_days / 30.0)
            rating = inter.get("rating", 3.0)
            rating_weight = max(0.2, min(1.0, rating / 5.0))
            like_boost = 1.2 if inter.get("liked", False) else 1.0

            w = decay * rating_weight * like_boost
            user_vector += self.topic_matrix[poem_idx] * w
            weight_sum += w

        if weight_sum > 0:
            user_vector /= weight_sum

        return user_vector

    def _get_similar_users(self, target_interactions, all_interactions, top_k=10):
        """寻找相似用户"""
        target_vec = self._get_user_profile_vector(target_interactions)
        if target_vec is None:
            return []

        user_ids = set(i["user_id"] for i in all_interactions)
        target_user = target_interactions[0]["user_id"] if target_interactions else None

        similarities = []
        for uid in user_ids:
            if uid == target_user:
                continue

            user_interactions = [i for i in all_interactions if i["user_id"] == uid]
            user_vec = self._get_user_profile_vector(user_interactions)

            if user_vec is not None:
                sim = cosine_similarity([target_vec], [user_vec])[0][0]
                similarities.append((uid, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _content_based_rec(self, user_vec, exclude_ids, top_k=20):
        """内容推荐"""
        if user_vec is None or self.topic_matrix is None:
            return []

        scores = cosine_similarity([user_vec], self.topic_matrix)[0]

        results = []
        for idx, score in enumerate(scores):
            poem_id = self.poem_ids[idx]
            if poem_id not in exclude_ids and score > 0:
                results.append((poem_id, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _item_cf_rec(self, user_interactions, exclude_ids, top_k=20):
        """物品协同过滤"""
        if self.topic_matrix is None or not user_interactions:
            return []

        user_indices = []
        weights = []
        for inter in user_interactions:
            poem_idx = self.poem_id_map.get(inter["poem_id"])
            if poem_idx is not None:
                user_indices.append(poem_idx)
                rating = inter.get("rating", 3.0)
                weights.append(max(0.2, min(1.0, rating / 5.0)))

        if not user_indices:
            return []

        reviewed_vectors = self.topic_matrix[user_indices]
        weights = np.array(weights)

        sim_matrix = cosine_similarity(self.topic_matrix, reviewed_vectors)

        weight_sum = weights.sum()
        if weight_sum > 0:
            scores = np.average(sim_matrix, axis=1, weights=weights)
        else:
            scores = np.mean(sim_matrix, axis=1)

        for idx in user_indices:
            scores[idx] = -1.0

        results = []
        for idx, score in enumerate(scores):
            poem_id = self.poem_ids[idx]
            if poem_id not in exclude_ids and score > 0:
                results.append((poem_id, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _user_cf_rec(self, user_interactions, all_interactions, exclude_ids, top_k=20):
        """用户协同过滤"""
        similar_users = self._get_similar_users(user_interactions, all_interactions)
        if not similar_users:
            return []

        sim_map = {uid: sim for uid, sim in similar_users}
        user_ids = list(sim_map.keys())

        user_interactions_filtered = [
            i for i in all_interactions if i["user_id"] in user_ids
        ]

        candidates = Counter()
        for inter in user_interactions_filtered:
            if inter["poem_id"] in exclude_ids:
                continue
            uid = inter["user_id"]
            sim_score = sim_map.get(uid, 0)
            rating = inter.get("rating", 3.0)
            rating_weight = max(0.2, min(1.0, rating / 5.0))

            candidates[inter["poem_id"]] += sim_score * rating_weight

        return [(pid, score) for pid, score in candidates.most_common(top_k)]

    def _popular_rec(self, exclude_ids, top_k=20):
        """热门推荐"""
        poem_scores = Counter()
        for inter in self.interactions:
            if inter["poem_id"] not in exclude_ids:
                poem_scores[inter["poem_id"]] += inter.get("rating", 3.0)

        return [
            {"poem_id": pid, "score": score}
            for pid, score in poem_scores.most_common(top_k)
        ]

    def _diversify(self, candidates, limit=10):
        """多样化排序"""
        if not candidates:
            return []

        selected = []
        remaining = list(candidates)

        while remaining and len(selected) < limit:
            best_idx = 0
            best_score = remaining[0][1]

            for idx, (pid, score) in enumerate(remaining):
                if score > best_score:
                    best_score = score
                    best_idx = idx

            selected.append(remaining[best_idx])
            remaining.pop(best_idx)

        return selected

    def recommend(self, user_interactions, all_interactions, top_k=10):
        """
        混合推荐主函数

        Args:
            user_interactions: 当前用户的交互历史
            all_interactions: 所有用户的交互历史（用于User-CF）
            top_k: 推荐数量

        Returns:
            list of dict: recommended poems with scores
        """
        if not self._ensure_model_loaded() or self.topic_matrix is None:
            return self._popular_rec(set(), top_k)

        exclude_ids = set(i["poem_id"] for i in user_interactions)

        interaction_count = len(user_interactions)

        if interaction_count == 0:
            w_content, w_item_cf, w_user_cf, w_popular = 0.4, 0.0, 0.0, 0.6
        elif interaction_count < 10:
            w_content, w_item_cf, w_user_cf, w_popular = 0.3, 0.4, 0.2, 0.1
        else:
            w_content, w_item_cf, w_user_cf, w_popular = 0.2, 0.4, 0.4, 0.0

        user_vec = self._get_user_profile_vector(user_interactions)

        content_rec = (
            self._content_based_rec(user_vec, exclude_ids) if w_content > 0 else []
        )
        item_cf_rec = (
            self._item_cf_rec(user_interactions, exclude_ids) if w_item_cf > 0 else []
        )
        user_cf_rec = (
            self._user_cf_rec(user_interactions, all_interactions, exclude_ids)
            if w_user_cf > 0
            else []
        )
        popular_rec = (
            self._popular_rec(exclude_ids)
            if w_popular > 0 or not any([content_rec, item_cf_rec, user_cf_rec])
            else []
        )

        def normalize(scores):
            if not scores:
                return {}
            max_score = max(s for _, s in scores) if scores else 1
            return {pid: s / max_score for pid, s in scores if s > 0}

        content_norm = normalize(content_rec)
        item_cf_norm = normalize(item_cf_rec)
        user_cf_norm = normalize(user_cf_rec)
        popular_norm = normalize(popular_rec)

        combined = Counter()
        for pid, score in content_norm.items():
            combined[pid] += score * w_content
        for pid, score in item_cf_norm.items():
            combined[pid] += score * w_item_cf
        for pid, score in user_cf_norm.items():
            combined[pid] += score * w_user_cf
        for pid, score in popular_norm.items():
            combined[pid] += score * (w_popular if w_popular > 0 else 0.3)

        diversified = self._diversify(list(combined.items()), top_k)

        return [{"poem_id": pid, "score": score} for pid, score in diversified]

    def predict_rating(self, user_interactions, poem_id):
        """预测评分"""
        if self.topic_matrix is None or poem_id not in self.poem_id_map:
            return 3.0

        poem_idx = self.poem_id_map[poem_id]
        user_vec = self._get_user_profile_vector(user_interactions)

        if user_vec is None:
            return 3.0

        score = cosine_similarity([user_vec], [self.topic_matrix[poem_idx]])[0][0]
        return np.clip(3.0 + score * 2.0, 1.0, 5.0)

    def predict_all_ratings(self, user_interactions):
        """预测所有评分"""
        if self.topic_matrix is None:
            return np.full(len(self.poem_ids), 3.0)

        user_vec = self._get_user_profile_vector(user_interactions)

        if user_vec is None:
            return np.full(len(self.poem_ids), 3.0)

        scores = cosine_similarity([user_vec], self.topic_matrix)[0]
        return np.clip(3.0 + scores * 2.0, 1.0, 5.0)
