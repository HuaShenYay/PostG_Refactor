from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), default="123456")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    preference_topics = db.Column(db.Text)  # 冷启动用

    @staticmethod
    def _looks_like_password_hash(value):
        if not value:
            return False
        return value.startswith(("pbkdf2:", "scrypt:"))

    def set_password(self, raw_password):
        self.password_hash = generate_password_hash(
            raw_password, method="pbkdf2:sha256"
        )

    def needs_password_rehash(self):
        return not self._looks_like_password_hash(self.password_hash)

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "preference_topics": self.preference_topics,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def check_password(self, password):
        if self._looks_like_password_hash(self.password_hash):
            return check_password_hash(self.password_hash, password)
        return self.password_hash == password


class Poem(db.Model):
    __tablename__ = "poems"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text)  # 算法核心：生成语义向量
    author = db.Column(db.String(50))
    dynasty = db.Column(db.String(20))

    # 额外字段（不同诗歌类型）
    chapter = db.Column(db.String(50))  # 诗经：国风/雅/颂
    section = db.Column(db.String(50))  # 诗经：周南/召南等
    rhythmic = db.Column(db.String(50))  # 宋词：词牌名
    category = db.Column(db.String(50))  # 唐诗/宋词：诗的类型

    # 统计字段（可选，用于热门推荐）
    views = db.Column(db.Integer, default=0)
    likes = db.Column(db.Integer, default=0)

    # 主题标签（可选，用于冷启动）
    topic_tags = db.Column(db.Text)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def average_rating(self):
        reviews = Review.query.filter_by(poem_id=self.id).all()
        if reviews:
            return sum(r.rating for r in reviews) / len(reviews)
        return 3.0

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "author": self.author,
            "dynasty": self.dynasty,
            "chapter": self.chapter,
            "section": self.section,
            "rhythmic": self.rhythmic,
            "category": self.category,
            "views": self.views,
            "likes": self.likes,
            "topic_tags": self.topic_tags,
            "average_rating": self.average_rating,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Review(db.Model):
    __tablename__ = "reviews"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    poem_id = db.Column(db.Integer, db.ForeignKey("poems.id"), nullable=False)
    rating = db.Column(db.Float, default=3.0)  # 算法核心：协同过滤
    comment = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship(
        "User", backref=db.backref("reviews", cascade="all, delete-orphan")
    )
    poem = db.relationship(
        "Poem", backref=db.backref("reviews", cascade="all, delete-orphan")
    )

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "poem_id": self.poem_id,
            "rating": self.rating,
            "comment": self.comment,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
