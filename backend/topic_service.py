from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import threading
from datetime import timedelta
import os
import time

class TopicService:
    """BERTopic service for topic identification from user comments"""

    def __init__(self):
        self.model = None
        self.last_comment_count = -1
        self.last_trained_at = None
        self.refresh_lock = threading.Lock()
        self.min_refresh_interval = timedelta(minutes=5)
        self.embedding_model = self._load_embedding_model()

    def _load_embedding_model(self):
        """Load SentenceTransformer model with retry mechanism and HuggingFace mirror"""
        # 设置HuggingFace镜像源，解决中国网络问题
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "0"
        
        # 模型名称
        model_name = "BAAI/bge-small-zh-v1.5"
        
        # 模型加载超时和重试机制
        max_retries = 3
        retry_delay = 5  # 秒
        model = None
        
        for attempt in range(max_retries):
            try:
                print(f"[SentenceTransformer] 尝试加载模型 (尝试 {attempt + 1}/{max_retries})...")
                print(f"[SentenceTransformer] 使用镜像源: {os.environ.get('HF_ENDPOINT')}")
                # 使用本地缓存的模型
                local_model_path = os.path.join(os.path.dirname(__file__), "data", "cache", "bge-small-zh-v1.5")
                print(f"[SentenceTransformer] 使用本地模型: {local_model_path}")
                
                # 检查本地模型是否存在
                if os.path.exists(local_model_path):
                    model = SentenceTransformer(
                        local_model_path,
                        use_auth_token=False
                    )
                    print("[SentenceTransformer] 从本地加载模型成功!")
                else:
                    # 本地模型不存在，从镜像源下载
                    print(f"[SentenceTransformer] 本地模型不存在，从镜像源下载: {model_name}")
                    model = SentenceTransformer(
                        model_name,
                        use_auth_token=False
                    )
                    print("[SentenceTransformer] 模型下载成功!")
                    # 保存模型到本地
                    model.save(local_model_path)
                    print(f"[SentenceTransformer] 模型保存到本地: {local_model_path}")
                break
            except Exception as e:
                print(f"[SentenceTransformer] 模型加载失败: {e}")
                if attempt < max_retries - 1:
                    print(f"[SentenceTransformer] {retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                else:
                    print("[SentenceTransformer] 所有尝试均失败，使用备用方案...")
                    # 备用方案：返回None
                    return None
        
        return model

    def _initialize_model(self):
        """Initialize BERTopic model"""
        if self.model is None:
            # 检查embedding_model是否成功加载
            if self.embedding_model is None:
                print("[BERTopic] 嵌入模型加载失败，无法初始化BERTopic")
                return
                
            self.model = BERTopic(
                embedding_model=self.embedding_model,
                language='chinese', 
                min_topic_size=2,              # 关键修复
                top_n_words=1,  # 每个主题提取5个关键词
                verbose=True
            )

    def get_topics_from_comments(self, comments):
        """Extract topics from comments using BERTopic"""
        if not comments:
            print("[BERTopic] 没有评论数据，返回空列表")
            return []

        try:
            print(f"[BERTopic] 开始处理 {len(comments)} 条评论")
            self._initialize_model()

            # 检查模型是否成功初始化
            if self.model is None:
                print("[BERTopic] 模型初始化失败，返回空列表")
                return []

            # 确保评论数量足够
            if len(comments) < 5:
                print(f"[BERTopic] 评论数量不足 (仅 {len(comments)} 条)，返回空列表")
                return []

            # Train the model with comments
            print("[BERTopic] 开始训练模型...")
            try:
                topics, probabilities = self.model.fit_transform(comments)
                print(f"[BERTopic] 模型训练完成，提取到 {len(set(topics))} 个主题")
            except Exception as e:
                print(f"[BERTopic] 模型训练失败: {e}")
                # 训练失败时返回空列表
                return []

            # Get topic information
            topic_info = self.model.get_topic_info()
            print(f"[BERTopic] 主题信息：{len(topic_info)} 个主题")

            # Process topics
            result = []
            for _, row in topic_info.iterrows():
                topic_id = row['Topic']
                if topic_id == -1:  # Outlier topic
                    continue
                
                topic_words = row['Name'].split('_')[1:]
                # 提取简洁的主题名称
                if topic_words:
                    # 尝试找到最能代表主题的单个词语
                    # 过滤掉虚词和短词，选择更简洁的词语
                    valid_words = []
                    for word in topic_words:
                        # 过滤掉太长的短语，只保留2-4个字符的词语
                        if 2 <= len(word) <= 4:
                            valid_words.append(word)
                    
                    if valid_words:
                        # 选择第一个有效词语作为主题名称
                        topic_name = valid_words[0]
                    else:
                        # 如果没有有效词语，使用第一个关键词的前两个字符
                        topic_name = topic_words[0][:4] if len(topic_words[0]) >= 2 else topic_words[0]
                else:
                    topic_name = '未分类'
                topic_count = row['Count']
                
                result.append({
                    'topic_id': topic_id,
                    'topic_name': topic_name,
                    'count': topic_count,
                    'words': topic_words
                })

            # Sort by count
            result.sort(key=lambda x: x['count'], reverse=True)
            print(f"[BERTopic] 处理完成，返回 {len(result)} 个主题")

            return result
        except Exception as e:
            print(f"[BERTopic] 主题提取失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回空列表作为备用方案
            return []

    def get_user_topics(self, user_id, comments):
        """Get topics for a specific user"""
        return self.get_topics_from_comments(comments)

    def get_global_topics(self, comments):
        """Get global topics from all comments"""
        return self.get_topics_from_comments(comments)


# Create singleton instance
topic_service = TopicService()
