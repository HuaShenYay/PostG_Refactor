from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


class SentenceTransformerEnhancedCF:
    """
    SentenceTransformer 增强协同过滤推荐系统。
    
    核心算法：三分量混合推荐
    - Semantic: 基于 SentenceTransformer 语义向量的内容匹配
    - User-CF: 基于评分矩阵的显式用户协同过滤  
    - Item-CF: 基于评分矩阵的物品协同过滤
    
    动态权重策略：根据用户评分数量自动调整各分量权重
    """

    def __init__(
        self,
        # 融合权重（离线计算用）
        cf_weight: float = 0.5,
        semantic_weight: float = 0.5,
        # 动态权重策略（在线推荐用）- 实验最优配置
        # 格式: (semantic_weight, user_cf_weight, item_cf_weight)
        cold_start_weights: Tuple[float, float, float] = (0.80, 0.15, 0.05),    # n_rated < 5
        low_activity_weights: Tuple[float, float, float] = (0.55, 0.30, 0.15),   # 5 <= n_rated < 15
        medium_activity_weights: Tuple[float, float, float] = (0.40, 0.35, 0.25), # 15 <= n_rated < 30
        high_activity_weights: Tuple[float, float, float] = (0.30, 0.40, 0.30),    # n_rated >= 30
        # 邻居数
        n_neighbors: int = 30,
        # 加速过滤策略
        fast_min_ratings: int = 10,
        fast_min_interactions: int = 5000,
    ) -> None:
        self.cf_weight = cf_weight
        self.semantic_weight = semantic_weight
        
        # 动态权重策略
        self.cold_start_weights = cold_start_weights
        self.low_activity_weights = low_activity_weights
        self.medium_activity_weights = medium_activity_weights
        self.high_activity_weights = high_activity_weights
        
        self.n_neighbors = n_neighbors

        # 来自实验的加速过滤策略
        self.fast_min_ratings = fast_min_ratings
        self.fast_min_interactions = fast_min_interactions

        self.poems: List[dict] = []
        self.interactions: List[dict] = []
        self.poem_ids: List[int] = []
        self.poem_id_map: Dict[int, int] = {}

        self.rating_matrix: Optional[np.ndarray] = None
        self.user_id_map: Dict[int, int] = {}
        self.global_mean = 3.0

        self.item_embeddings: Optional[np.ndarray] = None
        self.user_topic_profiles: Optional[np.ndarray] = None
        self.cf_item_sim: Optional[np.ndarray] = None
        self.item_semantic_sim: Optional[np.ndarray] = None
        self.user_similarity: Optional[np.ndarray] = None  # 显式User-CF
        self.enhanced_sim: Optional[np.ndarray] = None

        self.cache_dir = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def fit(self, poems: Sequence[dict], interactions: Sequence[dict]) -> None:
        poems, interactions = self._apply_fast_filter(poems, interactions)
        self.poems = list(poems)
        self.interactions = list(interactions)
        self.poem_ids = [p["id"] for p in self.poems]
        self.poem_id_map = {pid: i for i, pid in enumerate(self.poem_ids)}
        
        # 检查是否可以从缓存加载
        if self._load_all_caches():
            return

        self._build_rating_matrix(self.interactions)
        self._build_embeddings(self.poems)
        self._build_cf_similarity()
        self._build_explicit_user_similarity()

        if self.item_embeddings is None:
            self.enhanced_sim = self.cf_item_sim
            self._save_all_caches()
            return

        self._build_user_topic_profiles()
        self.item_semantic_sim = cosine_similarity(self.item_embeddings)
        self._fuse_similarities()
        
        # 保存所有缓存
        self._save_all_caches()

    def _get_cache_meta(self) -> dict:
        """生成缓存元数据，用于验证缓存有效性"""
        import hashlib
        
        # 交互数据的hash
        interaction_str = "".join(
            f"{i['user_id']}-{i['poem_id']}-{i['rating']}" 
            for i in sorted(self.interactions, key=lambda x: (x.get('user_id', 0), x.get('poem_id', 0)))
        )
        interaction_hash = hashlib.md5(interaction_str.encode("utf-8")).hexdigest()[:16]
        
        # 诗词ID列表
        poem_ids_hash = hashlib.md5(
            "|".join(map(str, self.poem_ids)).encode("utf-8")
        ).hexdigest()[:16]
        
        return {
            "interaction_hash": interaction_hash,
            "poem_ids_hash": poem_ids_hash,
            "n_poems": len(self.poem_ids),
            "n_users": len(self.user_id_map) if self.user_id_map else 0,
        }

    def _save_all_caches(self) -> None:
        """保存所有计算结果到缓存文件"""
        meta = self._get_cache_meta()
        meta_path = os.path.join(self.cache_dir, "cf_cache_meta.json")
        
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        
        if self.cf_item_sim is not None:
            np.save(os.path.join(self.cache_dir, "cf_item_sim.npy"), self.cf_item_sim)
        
        if self.user_similarity is not None:
            np.save(os.path.join(self.cache_dir, "user_similarity.npy"), self.user_similarity)
        
        if self.enhanced_sim is not None:
            np.save(os.path.join(self.cache_dir, "enhanced_sim.npy"), self.enhanced_sim)
        
        if self.user_topic_profiles is not None:
            np.save(os.path.join(self.cache_dir, "user_topic_profiles.npy"), self.user_topic_profiles)
        
        # 保存映射表
        if self.user_id_map:
            with open(os.path.join(self.cache_dir, "user_id_map.json"), "w") as f:
                json.dump(self.user_id_map, f, ensure_ascii=False)
        
        print(f"[CF-Cache] 已保存缓存: poems={len(self.poem_ids)}, users={len(self.user_id_map)}")

    def _load_all_caches(self) -> bool:
        """尝试加载所有缓存，返回是否成功"""
        import hashlib
        
        # 先检查语义向量缓存
        emb_path = os.path.join(self.cache_dir, "sem_embeddings.npy")
        ids_path = os.path.join(self.cache_dir, "sem_ids.json")
        hash_path = os.path.join(self.cache_dir, "sem_content_hash.json")
        
        if os.path.exists(emb_path) and os.path.exists(ids_path):
            try:
                with open(ids_path, "r", encoding="utf-8") as f:
                    cached_ids = json.load(f)
                cached_hash = None
                if os.path.exists(hash_path):
                    with open(hash_path, "r", encoding="utf-8") as f:
                        cached_hash = json.load(f)
                content_str = "|".join(p.get("content", "") for p in self.poems)
                content_hash = hashlib.md5(content_str.encode("utf-8")).hexdigest()
                if cached_ids == self.poem_ids and cached_hash == content_hash:
                    self.item_embeddings = np.load(emb_path)
            except Exception:
                pass

        # 检查CF缓存元数据
        meta_path = os.path.join(self.cache_dir, "cf_cache_meta.json")
        if not os.path.exists(meta_path):
            return False
            
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                cached_meta = json.load(f)
        except Exception:
            return False
        
        # 验证当前数据是否与缓存匹配
        current_meta = {}
        try:
            interaction_str = "".join(
                f"{i['user_id']}-{i['poem_id']}-{i['rating']}" 
                for i in sorted(self.interactions, key=lambda x: (x.get('user_id', 0), x.get('poem_id', 0)))
            )
            interaction_hash = hashlib.md5(interaction_str.encode("utf-8")).hexdigest()[:16]
            poem_ids_hash = hashlib.md5(
                "|".join(map(str, self.poem_ids)).encode("utf-8")
            ).hexdigest()[:16]
            current_meta = {
                "interaction_hash": interaction_hash,
                "poem_ids_hash": poem_ids_hash,
                "n_poems": len(self.poem_ids),
            }
            
            if (cached_meta.get("poem_ids_hash") != poem_ids_hash or
                cached_meta.get("interaction_hash") != interaction_hash):
                return False
        except Exception:
            return False
        
        # 加载各个缓存
        try:
            self.rating_matrix = None  # 不加载，需要时重新构建
            
            cf_sim_path = os.path.join(self.cache_dir, "cf_item_sim.npy")
            if os.path.exists(cf_sim_path):
                self.cf_item_sim = np.load(cf_sim_path)
            
            user_sim_path = os.path.join(self.cache_dir, "user_similarity.npy")
            if os.path.exists(user_sim_path):
                self.user_similarity = np.load(user_sim_path)
            
            enhanced_sim_path = os.path.join(self.cache_dir, "enhanced_sim.npy")
            if os.path.exists(enhanced_sim_path):
                self.enhanced_sim = np.load(enhanced_sim_path)
            
            user_profiles_path = os.path.join(self.cache_dir, "user_topic_profiles.npy")
            if os.path.exists(user_profiles_path) and self.item_embeddings is not None:
                self.user_topic_profiles = np.load(user_profiles_path)
            
            # 加载用户ID映射
            user_map_path = os.path.join(self.cache_dir, "user_id_map.json")
            if os.path.exists(user_map_path):
                with open(user_map_path, "r", encoding="utf-8") as f:
                    self.user_id_map = json.load(f)
            
            # 语义相似度需要重新计算（因为item_embeddings可能重新加载了）
            if self.item_embeddings is not None:
                self.item_semantic_sim = cosine_similarity(self.item_embeddings)
            
            print(f"[CF-Cache] 已加载缓存: poems={len(self.poem_ids)}, users={len(self.user_id_map)}")
            return True
            
        except Exception as e:
            print(f"[CF-Cache] 加载失败: {e}")
            return False
        poems, interactions = self._apply_fast_filter(poems, interactions)
        self.poems = list(poems)
        self.interactions = list(interactions)
        self.poem_ids = [p["id"] for p in self.poems]
        self.poem_id_map = {pid: i for i, pid in enumerate(self.poem_ids)}

        self._build_rating_matrix(self.interactions)
        self._build_embeddings(self.poems)
        self._build_cf_similarity()
        self._build_explicit_user_similarity()  # 新增：显式User-CF

        if self.item_embeddings is None:
            self.enhanced_sim = self.cf_item_sim
            return

        self._build_user_topic_profiles()
        self.item_semantic_sim = cosine_similarity(self.item_embeddings)
        self._fuse_similarities()

    def _apply_fast_filter(
        self, poems: Sequence[dict], interactions: Sequence[dict]
    ) -> Tuple[List[dict], List[dict]]:
        """实验中的“数据过滤(加速)”：大数据时过滤低频用户/物品。"""
        poems = list(poems)
        interactions = list(interactions)
        if len(interactions) < self.fast_min_interactions:
            return poems, interactions

        user_counts = Counter(i["user_id"] for i in interactions)
        item_counts = Counter(i["poem_id"] for i in interactions)
        active_users = {u for u, c in user_counts.items() if c >= self.fast_min_ratings}
        popular_items = {
            p for p, c in item_counts.items() if c >= self.fast_min_ratings
        }

        filtered_interactions = [
            i
            for i in interactions
            if i["user_id"] in active_users and i["poem_id"] in popular_items
        ]
        filtered_poems = [p for p in poems if p["id"] in popular_items]

        if not filtered_interactions or not filtered_poems:
            return poems, interactions
        return filtered_poems, filtered_interactions

    def _build_rating_matrix(self, interactions: Sequence[dict]) -> None:
        users = sorted({i["user_id"] for i in interactions})
        self.user_id_map = {uid: idx for idx, uid in enumerate(users)}
        self.rating_matrix = np.zeros(
            (len(users), len(self.poem_ids)), dtype=np.float32
        )

        ratings = []
        for inter in interactions:
            u = self.user_id_map.get(inter["user_id"])
            p = self.poem_id_map.get(inter["poem_id"])
            if u is None or p is None:
                continue
            r = float(inter.get("rating", 3.0))
            self.rating_matrix[u, p] = r
            ratings.append(r)
        self.global_mean = float(np.mean(ratings)) if ratings else 3.0

    def _build_embeddings(self, poems: Sequence[dict]) -> None:
        import hashlib
        
        emb_path = os.path.join(self.cache_dir, "sem_embeddings.npy")
        ids_path = os.path.join(self.cache_dir, "sem_ids.json")
        hash_path = os.path.join(self.cache_dir, "sem_content_hash.json")
        
        # 计算当前内容hash
        content_str = "|".join(p.get("content", "") for p in poems)
        content_hash = hashlib.md5(content_str.encode("utf-8")).hexdigest()
        
        if os.path.exists(emb_path) and os.path.exists(ids_path):
            try:
                with open(ids_path, "r", encoding="utf-8") as f:
                    cached_ids = json.load(f)
                
                # 检查hash
                cached_hash = None
                if os.path.exists(hash_path):
                    with open(hash_path, "r", encoding="utf-8") as f:
                        cached_hash = json.load(f)
                
                # 只有ID和hash都匹配才使用缓存
                if cached_ids == self.poem_ids and cached_hash == content_hash:
                    self.item_embeddings = np.load(emb_path)
                    return
            except Exception:
                pass
        emb_path = os.path.join(self.cache_dir, "sem_embeddings.npy")
        ids_path = os.path.join(self.cache_dir, "sem_ids.json")

        if os.path.exists(emb_path) and os.path.exists(ids_path):
            try:
                with open(ids_path, "r", encoding="utf-8") as f:
                    if json.load(f) == self.poem_ids:
                        self.item_embeddings = np.load(emb_path)
                        return
            except Exception:
                pass

        if SentenceTransformer is None:
            self.item_embeddings = None
            return

        try:
            device = "cpu"
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
            except Exception:
                pass

            model = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2", device=device
            )
            contents = [p.get("content", "") for p in poems]
            self.item_embeddings = model.encode(
                contents,
                batch_size=128,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            np.save(emb_path, self.item_embeddings)
            with open(ids_path, "w", encoding="utf-8") as f:
                json.dump(self.poem_ids, f, ensure_ascii=False)
            # 保存内容hash用于缓存失效检测
            with open(hash_path, "w", encoding="utf-8") as f:
                json.dump(content_hash, f)
            with open(ids_path, "w", encoding="utf-8") as f:
                json.dump(self.poem_ids, f, ensure_ascii=False)
        except Exception:
            self.item_embeddings = None

    def _build_cf_similarity(self) -> None:
        assert self.rating_matrix is not None
        R = self.rating_matrix
        mask = R > 0
        col_counts = mask.sum(axis=0).clip(min=1)
        col_means = R.sum(axis=0) / col_counts
        centered = np.where(mask, R - col_means[np.newaxis, :], 0.0)
        self.cf_item_sim = cosine_similarity(centered.T)
        np.fill_diagonal(self.cf_item_sim, 1.0)

    def _build_explicit_user_similarity(self) -> None:
        """显式User-CF：基于评分矩阵计算用户-用户相似度"""
        assert self.rating_matrix is not None
        R = self.rating_matrix
        mask = R > 0
        row_counts = mask.sum(axis=1).clip(min=1)
        row_means = R.sum(axis=1) / row_counts
        centered = np.where(mask, R - row_means[:, np.newaxis], 0.0)
        self.user_similarity = cosine_similarity(centered)
        np.fill_diagonal(self.user_similarity, 0.0)

    def _build_user_topic_profiles(self) -> None:
        assert self.rating_matrix is not None and self.item_embeddings is not None
        n_users, dim = self.rating_matrix.shape[0], self.item_embeddings.shape[1]
        profiles = np.zeros((n_users, dim), dtype=np.float32)

        for u in range(n_users):
            rated_idx = np.where(self.rating_matrix[u] > 0)[0]
            if rated_idx.size == 0:
                continue
            ratings = self.rating_matrix[u, rated_idx]
            weights = ratings - self.global_mean
            weights = np.where(np.abs(weights) < 0.3, 0.15, weights)
            profile = weights @ self.item_embeddings[rated_idx]
            norm = np.linalg.norm(profile)
            profiles[u] = profile / norm if norm > 1e-8 else profile
        self.user_topic_profiles = profiles

    def _fuse_similarities(self) -> None:
        assert self.cf_item_sim is not None and self.item_semantic_sim is not None

        def minmax(m: np.ndarray) -> np.ndarray:
            return (m - m.min()) / (m.max() - m.min() + 1e-8)

        self.enhanced_sim = self.cf_weight * minmax(
            self.cf_item_sim
        ) + self.semantic_weight * minmax(self.item_semantic_sim)
        np.fill_diagonal(self.enhanced_sim, 1.0)

    def _get_user_state(
        self, user_interactions: Sequence[dict]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[int]]:
        user_ratings = np.zeros(len(self.poem_ids), dtype=np.float32)
        user_idx = None

        for inter in user_interactions:
            p = self.poem_id_map.get(inter["poem_id"])
            if p is not None:
                user_ratings[p] = float(inter.get("rating", 3.0))
            if user_idx is None and inter.get("user_id") in self.user_id_map:
                user_idx = self.user_id_map[inter["user_id"]]

        if self.item_embeddings is None:
            return user_ratings, None, user_idx

        profile = np.zeros(self.item_embeddings.shape[1], dtype=np.float32)
        total_w = 0.0
        for inter in user_interactions:
            p = self.poem_id_map.get(inter["poem_id"])
            if p is None:
                continue
            w = float(inter.get("rating", 3.0)) - self.global_mean
            w = 0.15 if abs(w) < 0.3 else w
            profile += w * self.item_embeddings[p]
        return user_ratings, profile, user_idx

    def _explicit_user_cf_scores(
        self,
        user_idx: Optional[int],
        user_ratings: np.ndarray,
        exclude_set: Set[int],
    ) -> Optional[np.ndarray]:
        """显式User-CF：基于真实用户相似度计算评分"""
        if self.user_similarity is None or user_idx is None:
            return None

        sims = self.user_similarity[user_idx].copy()
        sims = np.clip(sims, 0, None)

        top_idx = np.argsort(sims)[::-1][: self.n_neighbors]
        top_sims = sims[top_idx]
        valid = top_sims > 0.1
        top_idx, top_sims = top_idx[valid], top_sims[valid]
        if top_idx.size == 0:
            return None

        neigh_R = self.rating_matrix[top_idx]
        w = top_sims[:, np.newaxis]
        has_rated = (neigh_R > 0).astype(np.float32)
        weighted_sum = (w * neigh_R).sum(axis=0)
        sim_total = (w * has_rated).sum(axis=0)
        scores = np.where(sim_total > 0, weighted_sum / sim_total, -np.inf)

        for pid in exclude_set:
            p = self.poem_id_map.get(pid)
            if p is not None:
                scores[p] = -np.inf
        scores[user_ratings > 0] = -np.inf
        return scores

    @staticmethod
    def _minmax_valid(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        valid = arr[arr > -np.inf]
        if valid.size == 0:
            return arr
        lo, hi = valid.min(), valid.max()
        return np.where(arr > -np.inf, (arr - lo) / (hi - lo + 1e-8), -np.inf)

    def recommend(
        self,
        user_interactions: Sequence[dict],
        exclude_ids: Optional[Set[int]] = None,
        top_k: int = 10,
    ):
        if self.enhanced_sim is None:
            return []

        exclude_set = set(exclude_ids or []) | {i["poem_id"] for i in user_interactions}
        user_ratings, user_profile, user_idx = self._get_user_state(user_interactions)
        n_rated = int((user_ratings > 0).sum())

        # 使用动态权重策略（根据构造函数配置）
        if n_rated < 5:
            sem_w, ucf_w, icf_w = self.cold_start_weights
        elif n_rated < 15:
            sem_w, ucf_w, icf_w = self.low_activity_weights
        elif n_rated < 30:
            sem_w, ucf_w, icf_w = self.medium_activity_weights
        else:
            sem_w, ucf_w, icf_w = self.high_activity_weights

        sem_scores = None
        if (
            user_profile is not None
            and self.item_embeddings is not None
            and np.linalg.norm(user_profile) > 1e-8
        ):
            sem_scores = self.item_embeddings @ user_profile
            for pid in exclude_set:
                p = self.poem_id_map.get(pid)
                if p is not None:
                    sem_scores[p] = -np.inf

        rated = np.where(user_ratings > 0)[0]
        icf_scores = None
        if rated.size > 0:
            sims = self.enhanced_sim[:, rated]
            rv = user_ratings[rated]
            pos = sims > 0
            w_sum = np.where(pos, sims * rv, 0).sum(axis=1)
            sim_sum = np.where(pos, np.abs(sims), 0).sum(axis=1) + 1e-8
            icf_scores = w_sum / sim_sum
            icf_scores[rated] = -np.inf
            for pid in exclude_set:
                p = self.poem_id_map.get(pid)
                if p is not None:
                    icf_scores[p] = -np.inf

        # 显式User-CF
        ucf_scores = self._explicit_user_cf_scores(user_idx, user_ratings, exclude_set)

        sem_scores = self._minmax_valid(sem_scores)
        icf_scores = self._minmax_valid(icf_scores)
        ucf_scores = self._minmax_valid(ucf_scores)

        combined = np.full(len(self.poem_ids), -np.inf, dtype=np.float32)
        for arr, w in ((sem_scores, sem_w), (ucf_scores, ucf_w), (icf_scores, icf_w)):
            if arr is None:
                continue
            valid = arr > -np.inf
            combined = np.where(
                valid & (combined > -np.inf), combined + w * arr, combined
            )
            combined = np.where(valid & (combined == -np.inf), w * arr, combined)

        valid_idx = np.where(combined > -np.inf)[0]
        if valid_idx.size == 0:
            return self._popular_fallback(top_k, exclude_set)
        top_idx = valid_idx[np.argsort(combined[valid_idx])[::-1][:top_k]]
        return [
            {"poem_id": self.poem_ids[i], "score": float(combined[i])} for i in top_idx
        ]

    def _popular_fallback(self, top_k: int, exclude_ids: Optional[Set[int]] = None):
        exclude_ids = exclude_ids or set()
        cnt = Counter()
        for inter in self.interactions:
            pid = inter["poem_id"]
            if pid not in exclude_ids:
                cnt[pid] += float(inter.get("rating", 3.0))
        return [
            {"poem_id": pid, "score": float(s)} for pid, s in cnt.most_common(top_k)
        ]

    def clear_cache(self, clear_type="all"):
        """
        清除缓存，强制重新计算
        
        Args:
            clear_type: "all" - 清除所有缓存
                      "sem" - 只清除语义向量缓存
                      "cf" - 只清除CF矩阵缓存
        """
        if clear_type in ("all", "sem"):
            sem_patterns = [
                os.path.join(self.cache_dir, "sem_embeddings.npy"),
                os.path.join(self.cache_dir, "sem_ids.json"),
                os.path.join(self.cache_dir, "sem_content_hash.json"),
            ]
            for pattern in sem_patterns:
                if os.path.exists(pattern):
                    try:
                        os.remove(pattern)
                    except Exception:
                        pass
        
        if clear_type in ("all", "cf"):
            cf_patterns = [
                os.path.join(self.cache_dir, "cf_item_sim.npy"),
                os.path.join(self.cache_dir, "user_similarity.npy"),
                os.path.join(self.cache_dir, "enhanced_sim.npy"),
                os.path.join(self.cache_dir, "user_topic_profiles.npy"),
                os.path.join(self.cache_dir, "user_id_map.json"),
                os.path.join(self.cache_dir, "cf_cache_meta.json"),
            ]
            for pattern in cf_patterns:
                if os.path.exists(pattern):
                    try:
                        os.remove(pattern)
                    except Exception:
                        pass
        
        print(f"[Cache] 已清除缓存: {clear_type}")
        """清除语义向量缓存，强制重新计算"""
        cache_patterns = [
            os.path.join(self.cache_dir, "sem_embeddings.npy"),
            os.path.join(self.cache_dir, "sem_ids.json"),
            os.path.join(self.cache_dir, "sem_content_hash.json"),
        ]
        for pattern in cache_patterns:
            if os.path.exists(pattern):
                try:
                    os.remove(pattern)
                except Exception:
                    pass

