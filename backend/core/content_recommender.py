import numpy as np
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """
    基于内容的推荐算法 (Content-Based)
    使用 TF-IDF 提取诗歌内容特征，计算余弦相似度进行推荐
    """

    def __init__(self, stopwords=None):
        self.stopwords = stopwords or set()
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.poems = None

    def _tokenize(self, text):
        """中文分词"""
        if not text:
            return ""
        chinese_only = re.sub(r"[^\u4e00-\u9fa5]", "", text)
        words = jieba.lcut(chinese_only)
        words = [w for w in words if w not in self.stopwords and len(w) > 1]
        return " ".join(words)

    def fit(self, poems):
        """
        训练模型：构建 TF-IDF 特征矩阵

        Args:
            poems: list of dict, each dict contains 'id', 'content' keys
        """
        self.poems = poems

        contents = [self._tokenize(p.get("content", "")) for p in poems]

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000, ngram_range=(1, 2), min_df=1
        )

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(contents)

        # Handle empty matrix case
        if self.tfidf_matrix.shape[1] == 0:
            print("[CB] Warning: TF-IDF matrix is empty, using fallback")
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                [" ".join(contents)]
            )

        print(f"[CB] TF-IDF 矩阵构建完成: {self.tfidf_matrix.shape}")

    def get_user_profile(self, rated_poems, ratings):
        """
        构建用户画像向量

        Args:
            rated_poems: list of poem dicts that user has rated
            ratings: list of ratings corresponding to rated_poems

        Returns:
            numpy array: user profile vector
        """
        if not rated_poems:
            return None

        rated_contents = [self._tokenize(p.get("content", "")) for p in rated_poems]
        rated_vectors = self.tfidf_vectorizer.transform(rated_contents)

        ratings = np.array(ratings)
        ratings_normalized = (ratings - 3.0) / 2.0

        weights = np.abs(ratings_normalized)
        if weights.sum() > 0:
            user_profile = np.average(rated_vectors.toarray(), axis=0, weights=weights)
        else:
            user_profile = np.mean(rated_vectors.toarray(), axis=0)

        return user_profile

    def recommend(self, user_profile, exclude_ids=None, top_k=10):
        """
        根据用户画像推荐诗歌

        Args:
            user_profile: numpy array from get_user_profile()
            exclude_ids: set of poem ids to exclude
            top_k: number of recommendations

        Returns:
            list of dict: recommended poems with scores
        """
        if user_profile is None or self.tfidf_matrix is None:
            return []

        exclude_ids = exclude_ids or set()

        similarities = cosine_similarity([user_profile], self.tfidf_matrix.toarray())[0]

        results = []
        for i, poem in enumerate(self.poems):
            if poem["id"] not in exclude_ids:
                results.append(
                    {
                        "poem_id": poem["id"],
                        "score": float(similarities[i]),
                        "title": poem.get("title", ""),
                    }
                )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def predict_rating(self, user_profile, poem_idx):
        """预测用户对物品的评分"""
        if user_profile is None:
            return 3.0

        similarity = cosine_similarity(
            [user_profile], self.tfidf_matrix[poem_idx].toarray()
        )[0][0]
        predicted_rating = 3.0 + similarity * 2.0
        return np.clip(predicted_rating, 1.0, 5.0)

    def predict_all_ratings(self, user_profile):
        """预测用户对所有物品的评分"""
        if user_profile is None:
            return np.full(len(self.poems), 3.0)

        similarities = cosine_similarity(
            user_profile.reshape(1, -1), self.tfidf_matrix.toarray()
        )[0]
        predicted_ratings = 3.0 + similarities * 2.0
        return np.clip(predicted_ratings, 1.0, 5.0)


def calculate_precision_recall(recommended, relevant):
    """计算精确率和召回率"""
    recommended_set = set(recommended)
    relevant_set = set(relevant)

    tp = len(recommended_set & relevant_set)
    fp = len(recommended_set - relevant_set)
    fn = len(relevant_set - recommended_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return precision, recall, f1
