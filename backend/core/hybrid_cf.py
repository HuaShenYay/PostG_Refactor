from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class HybridCFRecommender:
    """A standard hybrid recommender that blends User-CF and Item-CF."""

    def __init__(
        self,
        user_cf_weight: float = 0.5,
        item_cf_weight: float = 0.5,
        n_neighbors: int = 30,
    ) -> None:
        total_weight = user_cf_weight + item_cf_weight
        if total_weight <= 0:
            raise ValueError("HybridCFRecommender weights must be positive.")

        self.user_cf_weight = user_cf_weight / total_weight
        self.item_cf_weight = item_cf_weight / total_weight
        self.n_neighbors = n_neighbors

        self.poems: List[dict] = []
        self.interactions: List[dict] = []
        self.poem_ids: List[int] = []
        self.poem_id_map: Dict[int, int] = {}
        self.user_id_map: Dict[int, int] = {}

        self.rating_matrix: Optional[np.ndarray] = None
        self.user_means: Optional[np.ndarray] = None
        self.user_similarity: Optional[np.ndarray] = None
        self.item_similarity: Optional[np.ndarray] = None
        self.global_mean: float = 3.0
        self.item_popularity: Dict[int, float] = {}

    def fit(self, poems: Sequence[dict], interactions: Sequence[dict]) -> None:
        self.poems = list(poems)
        self.interactions = list(interactions)
        self.poem_ids = [poem["id"] for poem in self.poems]
        self.poem_id_map = {poem_id: idx for idx, poem_id in enumerate(self.poem_ids)}

        user_ids = sorted({int(item["user_id"]) for item in self.interactions})
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}

        n_users = len(self.user_id_map)
        n_items = len(self.poem_ids)
        self.rating_matrix = np.zeros((n_users, n_items), dtype=float)

        ratings = []
        item_rating_buckets: Dict[int, List[float]] = {poem_id: [] for poem_id in self.poem_ids}
        for item in self.interactions:
            user_id = int(item["user_id"])
            poem_id = int(item["poem_id"])
            if user_id not in self.user_id_map or poem_id not in self.poem_id_map:
                continue
            rating = float(item.get("rating", self.global_mean))
            self.rating_matrix[self.user_id_map[user_id], self.poem_id_map[poem_id]] = rating
            ratings.append(rating)
            item_rating_buckets[poem_id].append(rating)

        self.global_mean = float(np.mean(ratings)) if ratings else 3.0
        self.user_means = np.full(n_users, self.global_mean, dtype=float)

        if self.rating_matrix.size:
            for user_idx in range(n_users):
                row = self.rating_matrix[user_idx]
                rated = row > 0
                if np.any(rated):
                    self.user_means[user_idx] = float(np.mean(row[rated]))

            centered = self.rating_matrix.copy()
            for user_idx in range(n_users):
                rated = centered[user_idx] > 0
                centered[user_idx, rated] -= self.user_means[user_idx]

            self.user_similarity = cosine_similarity(centered) if n_users else np.zeros((0, 0))
            self.item_similarity = cosine_similarity(centered.T) if n_items else np.zeros((0, 0))

            if self.user_similarity.size:
                np.fill_diagonal(self.user_similarity, 0.0)
            if self.item_similarity.size:
                np.fill_diagonal(self.item_similarity, 0.0)
        else:
            self.user_similarity = np.zeros((n_users, n_users))
            self.item_similarity = np.zeros((n_items, n_items))

        self.item_popularity = {}
        for poem_id, poem_ratings in item_rating_buckets.items():
            if poem_ratings:
                avg_rating = float(np.mean(poem_ratings))
                review_bonus = np.log1p(len(poem_ratings))
                self.item_popularity[poem_id] = avg_rating * 0.7 + review_bonus * 0.3
            else:
                self.item_popularity[poem_id] = self.global_mean * 0.5

    def recommend(
        self,
        user_interactions: Sequence[dict],
        exclude_ids: Optional[Set[int]] = None,
        top_k: int = 20,
    ) -> List[dict]:
        exclude_ids = set(exclude_ids or set())

        if not self.poem_ids:
            return []

        active_ratings = {
            int(item["poem_id"]): float(item.get("rating", self.global_mean))
            for item in user_interactions
            if int(item["poem_id"]) in self.poem_id_map
        }

        if not active_ratings:
            return self._popular_recommendations(exclude_ids, top_k)

        candidate_ids = [
            poem_id for poem_id in self.poem_ids
            if poem_id not in exclude_ids and poem_id not in active_ratings
        ]
        if not candidate_ids:
            return []

        rated_indices = [self.poem_id_map[poem_id] for poem_id in active_ratings]
        active_mean = float(np.mean(list(active_ratings.values()))) if active_ratings else self.global_mean

        user_scores = self._score_user_cf(active_ratings, active_mean)
        item_scores = self._score_item_cf(active_ratings, rated_indices)

        rated_count = len(active_ratings)
        user_weight, item_weight = self._resolve_weights(rated_count)

        results = []
        for poem_id in candidate_ids:
            item_idx = self.poem_id_map[poem_id]
            user_score = user_scores.get(poem_id, active_mean)
            item_score = item_scores.get(poem_id, active_mean)
            popularity = self.item_popularity.get(poem_id, self.global_mean)
            hybrid_score = (
                user_weight * user_score +
                item_weight * item_score +
                0.1 * popularity
            )
            results.append(
                {
                    "poem_id": poem_id,
                    "score": float(hybrid_score),
                    "user_cf_score": float(user_score),
                    "item_cf_score": float(item_score),
                    "popularity_score": float(popularity),
                }
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]

    def _score_user_cf(self, active_ratings: Dict[int, float], active_mean: float) -> Dict[int, float]:
        if self.rating_matrix is None or self.user_means is None or self.user_similarity is None:
            return {}

        active_vector = np.zeros(len(self.poem_ids), dtype=float)
        for poem_id, rating in active_ratings.items():
            active_vector[self.poem_id_map[poem_id]] = rating

        centered_active = active_vector.copy()
        rated_mask = centered_active > 0
        centered_active[rated_mask] -= active_mean

        if self.rating_matrix.shape[0] == 0:
            return {}

        centered_train = self.rating_matrix.copy()
        for user_idx in range(centered_train.shape[0]):
            row_mask = centered_train[user_idx] > 0
            centered_train[user_idx, row_mask] -= self.user_means[user_idx]

        similarities = cosine_similarity(centered_active.reshape(1, -1), centered_train)[0]
        neighbor_indices = np.argsort(similarities)[::-1]

        scored_neighbors = []
        for user_idx in neighbor_indices:
            sim = float(similarities[user_idx])
            if sim <= 0:
                continue
            overlap = np.count_nonzero((self.rating_matrix[user_idx] > 0) & rated_mask)
            if overlap == 0:
                continue
            scored_neighbors.append((user_idx, sim))
            if len(scored_neighbors) >= self.n_neighbors:
                break

        if not scored_neighbors:
            return {}

        scores = {}
        for poem_id in self.poem_ids:
            if poem_id in active_ratings:
                continue
            item_idx = self.poem_id_map[poem_id]
            numerator = 0.0
            denominator = 0.0
            for user_idx, sim in scored_neighbors:
                neighbor_rating = self.rating_matrix[user_idx, item_idx]
                if neighbor_rating <= 0:
                    continue
                numerator += sim * (neighbor_rating - self.user_means[user_idx])
                denominator += abs(sim)
            if denominator > 0:
                scores[poem_id] = active_mean + numerator / denominator
        return scores

    def _score_item_cf(self, active_ratings: Dict[int, float], rated_indices: List[int]) -> Dict[int, float]:
        if self.item_similarity is None or not rated_indices:
            return {}

        scores = {}
        for poem_id in self.poem_ids:
            if poem_id in active_ratings:
                continue
            item_idx = self.poem_id_map[poem_id]
            similarities = self.item_similarity[item_idx, rated_indices]

            pairs = []
            for rated_idx, sim in zip(rated_indices, similarities):
                if sim <= 0:
                    continue
                rated_poem_id = self.poem_ids[rated_idx]
                pairs.append((float(sim), active_ratings[rated_poem_id]))

            if not pairs:
                continue

            pairs.sort(key=lambda pair: pair[0], reverse=True)
            pairs = pairs[: self.n_neighbors]
            numerator = sum(sim * rating for sim, rating in pairs)
            denominator = sum(abs(sim) for sim, _ in pairs)
            if denominator > 0:
                scores[poem_id] = numerator / denominator
        return scores

    def _popular_recommendations(self, exclude_ids: Set[int], top_k: int) -> List[dict]:
        candidates = [
            {"poem_id": poem_id, "score": float(score)}
            for poem_id, score in self.item_popularity.items()
            if poem_id not in exclude_ids
        ]
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:top_k]

    def _resolve_weights(self, rated_count: int) -> tuple[float, float]:
        if rated_count < 3:
            return 0.35, 0.65
        if rated_count < 10:
            return 0.45, 0.55
        return self.user_cf_weight, self.item_cf_weight
