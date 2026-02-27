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

    preference_topics = db.Column(db.Text)

    @staticmethod
    def _looks_like_password_hash(value):
        if not value:
            return False
        return value.startswith(("pbkdf2:", "scrypt:"))

    def set_password(self, raw_password):
        self.password_hash = generate_password_hash(raw_password, method='pbkdf2:sha256')

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
    author = db.Column(db.String(50))
    content = db.Column(db.Text)
    dynasty = db.Column(db.String(20))

    genre_type = db.Column(db.String(50))
    rhythm_name = db.Column(db.String(50))
    rhythm_type = db.Column(db.String(20))

    views = db.Column(db.Integer, default=0)
    likes = db.Column(db.Integer, default=0)
    shares = db.Column(db.Integer, default=0)
    review_count = db.Column(db.Integer, default=0)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    tonal_summary = db.Column(db.Text)

    Bertopic = db.Column(db.Text)
    Real_topic = db.Column(db.Text)

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
            "author": self.author,
            "content": self.content,
            "dynasty": self.dynasty,
            "genre_type": self.genre_type,
            "rhythm_name": self.rhythm_name,
            "rhythm_type": self.rhythm_type,
            "views": self.views,
            "likes": self.likes,
            "shares": self.shares,
            "review_count": self.review_count,
            "tonal_summary": self.tonal_summary,
            "Bertopic": self.Bertopic,
            "Real_topic": self.Real_topic,
            "average_rating": self.average_rating,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Review(db.Model):
    __tablename__ = "reviews"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    poem_id = db.Column(db.Integer, db.ForeignKey("poems.id"), nullable=False)
    comment = db.Column(db.Text)
    topic_names = db.Column(db.Text)

    rating = db.Column(db.Float, default=3.0)
    liked = db.Column(db.Boolean, default=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

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
            "comment": self.comment,
            "topic_names": self.topic_names,
            "rating": self.rating,
            "liked": self.liked,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
