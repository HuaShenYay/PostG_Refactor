from flask import Flask, jsonify, request
from flask_cors import CORS
from config import Config
from models import db, User, Poem, Review
from datetime import datetime, timedelta
from sqlalchemy import text, func, or_
from functools import wraps
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
import json
import os
import threading

# 延迟导入topic_service，避免启动时加载模型
topic_service = None
def get_topic_service():
    global topic_service
    if topic_service is None:
        from topic_service import topic_service as ts
        topic_service = ts
    return topic_service


app = Flask(__name__)
app.config.from_object(Config)

CORS(app)
db.init_app(app)

admin_token_serializer = URLSafeTimedSerializer(app.config["SECRET_KEY"])


def _generate_admin_token():
    return admin_token_serializer.dumps({"role": "admin"})


def _verify_admin_token(token):
    try:
        payload = admin_token_serializer.loads(
            token,
            max_age=app.config["ADMIN_TOKEN_MAX_AGE"],
        )
    except SignatureExpired:
        return None, "登录已过期，请重新登录"
    except BadSignature:
        return None, "无效的管理员凭证"

    if payload.get("role") != "admin":
        return None, "无效的管理员凭证"
    return payload, None


def admin_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        token = ""
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        if not token:
            token = request.headers.get("X-Admin-Token", "").strip()

        if not token:
            return jsonify({"status": "error", "message": "缺少管理员凭证"}), 401

        payload, error = _verify_admin_token(token)
        if error:
            return jsonify({"status": "error", "message": error}), 401

        request.admin_payload = payload
        return view_func(*args, **kwargs)

    return wrapped


def _normalize_optional_text(value):
    if value is None:
        return None
    text_value = str(value).strip()
    return text_value or None


def _poem_summary(poem):
    review_count = Review.query.filter_by(poem_id=poem.id).count()
    return {
        **poem.to_dict(),
        "review_count": review_count,
    }


def _user_summary(user):
    review_count = Review.query.filter_by(user_id=user.id).count()
    return {
        **user.to_dict(),
        "review_count": review_count,
    }


class RecommendationService:
    """Standard hybrid collaborative filtering recommender service."""

    def __init__(self):
        self.recommender = None
        self.last_review_count = -1
        self.last_trained_at = None
        self.refresh_lock = threading.Lock()
        self.min_refresh_interval = timedelta(minutes=5)

    def _ensure_recommender(self):
        if self.recommender is None:
            from core.hybrid_cf import HybridCFRecommender

            self.recommender = HybridCFRecommender(
                user_cf_weight=0.5,
                item_cf_weight=0.5,
                n_neighbors=30,
            )

    @staticmethod
    def _build_interactions():
        return [
            {
                "user_id": r.user_id,
                "poem_id": r.poem_id,
                "rating": r.rating,
                "created_at": r.created_at or datetime.utcnow(),
            }
            for r in Review.query.all()
        ]

    @staticmethod
    def _build_poems():
        return [
            {"id": p.id, "content": p.content or "", "title": p.title or ""}
            for p in Poem.query.all()
        ]

    def refresh_if_needed(self, force=False):
        now = datetime.utcnow()
        current_review_count = Review.query.count()

        is_recently_trained = (
            self.last_trained_at is not None
            and (now - self.last_trained_at) < self.min_refresh_interval
        )
        if not force and self.last_review_count == current_review_count:
            return

        if not force and is_recently_trained:
            return

        if not self.refresh_lock.acquire(blocking=False):
            return

        try:
            self._ensure_recommender()
            poems_data = self._build_poems()
            interactions = self._build_interactions()

            self.recommender.fit(poems_data, interactions)
            self.last_review_count = current_review_count
            self.last_trained_at = datetime.utcnow()
        finally:
            self.refresh_lock.release()

    def get_recommendation(self, user, skip_count=0, seen_ids=None):
        """统一的推荐方法
        
        Args:
            user: User对象
            skip_count: 跳过次数
            seen_ids: 已看过的诗歌ID列表
            
        Returns:
            推荐的诗歌对象和推荐理由
        """
        import random
        
        seen_ids = seen_ids or []
        
        # 获取用户的所有评论历史
        user_reviews = Review.query.filter_by(user_id=user.id).all()
        review_count = len(user_reviews)
        user_interactions = [
            {
                "user_id": r.user_id,
                "poem_id": r.poem_id,
                "rating": r.rating,
                "created_at": r.created_at or datetime.utcnow(),
            }
            for r in user_reviews
        ]
        
        # 构建排除列表（已评论 + 本次会话已看）
        exclude_ids = {r.poem_id for r in user_reviews}
        exclude_ids.update(seen_ids)
        
        # ========== 策略1: 探索模式（每3次强制探索）==========
        explore_frequency = 3
        should_explore = skip_count > 0 and skip_count % explore_frequency == 0
        if should_explore:
            # 策略1a: 获取所有符合条件的诗歌，然后随机选择
            subquery = (
                db.session.query(
                    Review.poem_id,
                    func.count(Review.id).label("review_count"),
                    func.avg(Review.rating).label("avg_rating"),
                )
                .group_by(Review.poem_id)
                .subquery()
            )
            
            explore_candidates = (
                Poem.query.outerjoin(subquery, Poem.id == subquery.c.poem_id)
                .filter(~Poem.id.in_(exclude_ids))
                .filter((subquery.c.review_count < 3) | (subquery.c.review_count.is_(None)))
                .order_by(func.coalesce(subquery.c.avg_rating, 4.0).desc())
                .limit(20)
                .all()
            )
            
            if explore_candidates:
                explore_poem = random.choice(explore_candidates)
                return explore_poem, "探索推荐：小众佳作"
            
            # 策略1b: 推荐从未被评论的诗歌
            reviewed_poem_ids = {
                r.poem_id for r in Review.query.with_entities(Review.poem_id).all()
            }
            unseen_candidates = (
                Poem.query.filter(
                    ~Poem.id.in_(reviewed_poem_ids), ~Poem.id.in_(exclude_ids)
                )
                .limit(20)
                .all()
            )
            
            if unseen_candidates:
                unseen_poem = random.choice(unseen_candidates)
                return unseen_poem, "探索推荐：尚未被发现的诗"
        
        # ========== 策略3: 基于 User-CF + Item-CF 的混合协同过滤 ==========
        try:
            self.refresh_if_needed()
            
            # 获取候选推荐
            recs = self.recommender.recommend(
                user_interactions, exclude_ids, top_k=100
            )
            
            # 过滤已看过的
            candidates = [rec for rec in recs if rec["poem_id"] not in exclude_ids]
            
            # 如果候选少于20首，补充随机推荐
            if len(candidates) < 20:
                random_poems = (
                    Poem.query.filter(
                        ~Poem.id.in_(exclude_ids),
                        ~Poem.id.in_([r["poem_id"] for r in candidates]),
                    )
                    .order_by(func.rand())
                    .limit(30)
                    .all()
                )
                for p in random_poems:
                    candidates.append({"poem_id": p.id, "score": 0.5, "strategy": "random_fill"})
            
            # 随机选择（增加时间和用户ID的随机性）
            if candidates:
                import time
                
                # 使用时间戳+用户ID+skip_count作为种子
                random.seed(int(time.time() * 1000) % 10000 + user.id + skip_count)
                
                # 从前20个候选中随机选择（如果候选少于20个则全部）
                pool_size = min(20, len(candidates))
                selected_idx = random.randint(0, pool_size - 1)
                selected = candidates[selected_idx]
                
                poem = Poem.query.get(selected["poem_id"])
                if poem:
                    return poem, self._build_recommend_reason(
                        review_count=review_count,
                        strategy=selected.get("strategy", "cf"),
                    )
        except Exception as e:
            print(f"Recommend error: {e}")
        
        # ========== 最终回退：完全随机 ==========
        fallback_query = Poem.query
        if exclude_ids:
            fallback_query = fallback_query.filter(~Poem.id.in_(exclude_ids))
        fallback = fallback_query.order_by(func.rand()).first()
        if fallback:
            return fallback, "随机推荐"
        
        return None, ""

    @staticmethod
    def _build_recommend_reason(review_count, strategy="cf"):
        if strategy == "random_fill":
            if review_count == 0:
                return "新用户入门推荐"
            return "为你补充发现"

        if review_count == 0:
            return "新用户入门推荐"
        if review_count < 3:
            return "根据你刚开始的评分记录推荐"
        if review_count < 10:
            return "根据你的近期评分偏好推荐"
        return "基于相似用户与相似诗作推荐"

    @staticmethod
    def _extract_preference_topic_tokens(user):
        """提取用户偏好主题标记"""
        try:
            if user and user.preference_topics:
                import json
                preferences = json.loads(user.preference_topics)
                tokens = []
                for pref in preferences:
                    if isinstance(pref, dict) and "topic_id" in pref:
                        # 这里可以根据topic_id映射到具体的主题词
                        # 暂时返回topic_id作为标记
                        tokens.append(str(pref["topic_id"]))
                return tokens[:5]  # 最多返回5个标记
        except:
            pass
        return []


rec_service = RecommendationService()


@app.route("/")
def hello_world():
    return "Poetry Recommendation Engine (Hybrid CF) is Running!"


@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"message": "请输入账号和密码", "status": "error"}), 400

    user = User.query.filter_by(username=username).first()

    if user and user.check_password(password):
        if user.needs_password_rehash():
            user.set_password(password)
            db.session.commit()
        return jsonify(
            {"message": "登录成功", "status": "success", "user": user.to_dict()}
        )
    else:
        return jsonify({"message": "账号或口令有误", "status": "error"}), 401


@app.route("/api/register", methods=["POST"])
def register():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"message": "请输入账号和密码", "status": "error"}), 400

    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return jsonify({"message": "此称谓已被占用", "status": "error"}), 400

    try:
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"message": "注册成功，即将开启诗意之旅", "status": "success"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"message": f"注册失败: {str(e)}", "status": "error"}), 500


@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    data = request.json or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"status": "error", "message": "请输入管理员账号和密码"}), 400

    if (
        username != app.config["ADMIN_USERNAME"]
        or password != app.config["ADMIN_PASSWORD"]
    ):
        return jsonify({"status": "error", "message": "管理员账号或密码错误"}), 401

    return jsonify(
        {
            "status": "success",
            "message": "管理员登录成功",
            "token": _generate_admin_token(),
            "admin": {"username": app.config["ADMIN_USERNAME"]},
        }
    )


@app.route("/api/admin/overview")
@admin_required
def get_admin_overview():
    return jsonify(
        {
            "status": "success",
            "overview": {
                "users": User.query.count(),
                "poems": Poem.query.count(),
                "reviews": Review.query.count(),
                "today_reviews": Review.query.filter(
                    Review.created_at >= datetime.utcnow().replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                ).count(),
            },
        }
    )


@app.route("/api/admin/poems")
@admin_required
def admin_get_poems():
    query = (request.args.get("q") or "").strip()
    page = max(int(request.args.get("page", 1)), 1)
    page_size = min(max(int(request.args.get("page_size", 10)), 1), 50)

    poems_query = Poem.query
    if query:
        like_query = f"%{query}%"
        poems_query = poems_query.filter(
            Poem.title.ilike(like_query)
            | Poem.author.ilike(like_query)
            | Poem.content.ilike(like_query)
            | Poem.dynasty.ilike(like_query)
            | Poem.category.ilike(like_query)
            | Poem.topic_tags.ilike(like_query)
        )

    total = poems_query.count()
    poems = (
        poems_query.order_by(Poem.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return jsonify(
        {
            "status": "success",
            "items": [_poem_summary(poem) for poem in poems],
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
            },
        }
    )


@app.route("/api/admin/poems", methods=["POST"])
@admin_required
def admin_create_poem():
    data = request.json or {}
    title = (data.get("title") or "").strip()
    content = (data.get("content") or "").strip()

    if not title or not content:
        return jsonify({"status": "error", "message": "标题和内容不能为空"}), 400

    poem = Poem(
        title=title,
        content=content,
        author=_normalize_optional_text(data.get("author")),
        dynasty=_normalize_optional_text(data.get("dynasty")),
        chapter=_normalize_optional_text(data.get("chapter")),
        section=_normalize_optional_text(data.get("section")),
        rhythmic=_normalize_optional_text(data.get("rhythmic")),
        category=_normalize_optional_text(data.get("category")),
        topic_tags=_normalize_optional_text(data.get("topic_tags")),
        views=int(data.get("views", 0) or 0),
    )
    db.session.add(poem)
    db.session.commit()
    rec_service.refresh_if_needed(force=True)

    return jsonify(
        {
            "status": "success",
            "message": "诗歌已创建",
            "item": _poem_summary(poem),
        }
    )


@app.route("/api/admin/poems/<int:poem_id>", methods=["PUT"])
@admin_required
def admin_update_poem(poem_id):
    poem = Poem.query.get(poem_id)
    if not poem:
        return jsonify({"status": "error", "message": "诗歌不存在"}), 404

    data = request.json or {}
    title = (data.get("title") or "").strip()
    content = (data.get("content") or "").strip()

    if not title or not content:
        return jsonify({"status": "error", "message": "标题和内容不能为空"}), 400

    poem.title = title
    poem.content = content
    poem.author = _normalize_optional_text(data.get("author"))
    poem.dynasty = _normalize_optional_text(data.get("dynasty"))
    poem.chapter = _normalize_optional_text(data.get("chapter"))
    poem.section = _normalize_optional_text(data.get("section"))
    poem.rhythmic = _normalize_optional_text(data.get("rhythmic"))
    poem.category = _normalize_optional_text(data.get("category"))
    poem.topic_tags = _normalize_optional_text(data.get("topic_tags"))
    poem.views = int(data.get("views", 0) or 0)

    db.session.commit()
    rec_service.refresh_if_needed(force=True)

    return jsonify(
        {
            "status": "success",
            "message": "诗歌已更新",
            "item": _poem_summary(poem),
        }
    )


@app.route("/api/admin/poems/<int:poem_id>", methods=["DELETE"])
@admin_required
def admin_delete_poem(poem_id):
    poem = Poem.query.get(poem_id)
    if not poem:
        return jsonify({"status": "error", "message": "诗歌不存在"}), 404

    db.session.delete(poem)
    db.session.commit()
    rec_service.refresh_if_needed(force=True)
    return jsonify({"status": "success", "message": "诗歌已删除"})


@app.route("/api/admin/reviews")
@admin_required
def admin_get_reviews():
    page = max(int(request.args.get("page", 1)), 1)
    page_size = min(max(int(request.args.get("page_size", 10)), 1), 50)

    query = (
        Review.query.order_by(Review.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    total = Review.query.count()

    items = []
    for review in query:
        items.append(
            {
                "id": review.id,
                "rating": review.rating,
                "comment": review.comment,
                "created_at": review.created_at.isoformat() if review.created_at else None,
                "user": review.user.username if review.user else "未知用户",
                "poem_title": review.poem.title if review.poem else "未知诗歌",
                "poem_id": review.poem_id,
            }
        )

    return jsonify(
        {
            "status": "success",
            "items": items,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
            },
        }
    )


@app.route("/api/admin/reviews/<int:review_id>", methods=["DELETE"])
@admin_required
def admin_delete_review(review_id):
    review = Review.query.get(review_id)
    if not review:
        return jsonify({"status": "error", "message": "评论不存在"}), 404

    db.session.delete(review)
    db.session.commit()
    rec_service.refresh_if_needed(force=True)
    return jsonify({"status": "success", "message": "评论已删除"})


@app.route("/api/admin/users")
@admin_required
def admin_get_users():
    query = (request.args.get("q") or "").strip()
    page = max(int(request.args.get("page", 1)), 1)
    page_size = min(max(int(request.args.get("page_size", 10)), 1), 50)

    users_query = User.query
    if query:
        like_query = f"%{query}%"
        users_query = users_query.filter(User.username.ilike(like_query))

    total = users_query.count()
    users = (
        users_query.order_by(User.created_at.desc(), User.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return jsonify(
        {
            "status": "success",
            "items": [_user_summary(user) for user in users],
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
            },
        }
    )


@app.route("/api/admin/users/<int:user_id>", methods=["PUT"])
@admin_required
def admin_update_user(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({"status": "error", "message": "用户不存在"}), 404

    data = request.json or {}
    new_username = (data.get("username") or "").strip()
    reset_password = data.get("reset_password") or ""

    if not new_username:
        return jsonify({"status": "error", "message": "用户名不能为空"}), 400

    existing_user = User.query.filter(User.username == new_username, User.id != user_id).first()
    if existing_user:
        return jsonify({"status": "error", "message": "用户名已存在"}), 400

    user.username = new_username
    if reset_password:
        user.set_password(reset_password)

    db.session.commit()

    return jsonify(
        {
            "status": "success",
            "message": "用户信息已更新",
            "item": _user_summary(user),
        }
    )


@app.route("/api/admin/users/<int:user_id>", methods=["DELETE"])
@admin_required
def admin_delete_user(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({"status": "error", "message": "用户不存在"}), 404

    db.session.delete(user)
    db.session.commit()
    rec_service.refresh_if_needed(force=True)
    return jsonify({"status": "success", "message": "用户已删除"})


@app.route("/api/user/update", methods=["POST"])
def update_user():
    data = request.json
    old_username = data.get("old_username")
    current_password = data.get("current_password")
    new_username = data.get("new_username")
    new_password = data.get("new_password")

    if not old_username or not current_password:
        return jsonify({"message": "缺少身份校验信息", "status": "error"}), 400

    user = User.query.filter_by(username=old_username).first()
    if not user:
        return jsonify({"message": "用户不存在", "status": "error"}), 404

    if not user.check_password(current_password):
        return jsonify({"message": "当前口令错误", "status": "error"}), 401

    if new_username and new_username != old_username:
        existing = User.query.filter_by(username=new_username).first()
        if existing:
            return jsonify({"message": "新称谓已被占用", "status": "error"}), 400
        user.username = new_username

    if new_password:
        user.set_password(new_password)

    try:
        db.session.commit()
        return jsonify(
            {"message": "修缮成功", "status": "success", "user": user.to_dict()}
        )
    except Exception as e:
        db.session.rollback()
        return jsonify({"message": f"修缮失败: {str(e)}", "status": "error"}), 500


@app.route("/api/poems")
def get_poems():
    poems = Poem.query.limit(20).all()
    return jsonify([p.to_dict() for p in poems])


@app.route("/api/topics")
def get_topics():
    """返回主题列表，供偏好引导页使用"""
    poems = Poem.query.filter(Poem.topic_tags.isnot(None)).all()
    counter = {}
    for poem in poems:
        for topic in (poem.topic_tags or "").split("-"):
            topic = topic.strip()
            if topic:
                counter[topic] = counter.get(topic, 0) + 1

    sorted_topics = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    if not sorted_topics:
        fallback = [
            "山水",
            "思乡",
            "边塞",
            "离别",
            "咏史",
            "田园",
            "闺怨",
            "怀古",
            "节序",
            "哲理",
            "人生",
            "家国",
            "送别",
            "咏物",
            "写景",
        ]
        sorted_topics = [(name, 1) for name in fallback]

    topics = {}
    for idx, (name, _) in enumerate(sorted_topics[:15]):
        topics[idx] = [name]

    return jsonify(topics)


@app.route("/api/poem/<int:poem_id>")
def get_poem(poem_id):
    poem = Poem.query.get(poem_id)
    if not poem:
        return jsonify({"error": "Poem not found"}), 404
    return jsonify(poem.to_dict())


@app.route("/api/search_poems")
def search_poems():
    query = request.args.get("q", "")
    if not query:
        return jsonify([])

    # 确保查询参数正确处理中文字符
    query = query.strip()
    
    # 处理简繁中文转换
    # 简单的简繁中文映射
    simplified_to_traditional = {
        '长': '長', '簟': '簟', '迎': '迎', '风': '風', '早': '早',
        '韩': '韓', '翃': '翃', '酬': '酬', '程': '程', '延': '延',
        '秋': '秋', '夜': '夜', '即': '即', '事': '事', '见': '見',
        '赠': '贈', '空': '空', '城': '城', '澹': '澹', '月': '月',
        '华': '華', '星': '星', '河': '河', '一': '一', '雁': '雁',
        '砧': '砧', '杵': '杵', '千': '千', '家': '家', '节': '節',
        '候': '候', '看': '看', '应': '應', '晚': '晚', '心': '心',
        '期': '期', '卧': '臥', '亦': '亦', '赊': '賒', '向': '向',
        '来': '來', '吟': '吟', '秀': '秀', '句': '句', '不': '不',
        '觉': '覺', '已': '已', '鸣': '鳴', '鸦': '鴉'
    }
    
    # 生成繁体查询
    traditional_query = ''.join([simplified_to_traditional.get(c, c) for c in query])
    
    # 尝试多种搜索方式，同时考虑简体和繁体
    results = Poem.query.filter(
        (Poem.title.ilike(f"%{query}%")
        | Poem.title.ilike(f"%{traditional_query}%")
        | Poem.author.ilike(f"%{query}%")
        | Poem.author.ilike(f"%{traditional_query}%")
        | Poem.content.ilike(f"%{query}%")
        | Poem.content.ilike(f"%{traditional_query}%")
        | Poem.topic_tags.ilike(f"%{query}%")
        | Poem.topic_tags.ilike(f"%{traditional_query}%")
        | Poem.category.ilike(f"%{query}%")
        | Poem.category.ilike(f"%{traditional_query}%")
        | Poem.dynasty.ilike(f"%{query}%")
        | Poem.dynasty.ilike(f"%{traditional_query}%")
        | Poem.rhythmic.ilike(f"%{query}%")
        | Poem.rhythmic.ilike(f"%{traditional_query}%")
        | Poem.chapter.ilike(f"%{query}%")
        | Poem.chapter.ilike(f"%{traditional_query}%")
        | Poem.section.ilike(f"%{query}%")
        | Poem.section.ilike(f"%{traditional_query}%")
    )).limit(20).all()

    poems_with_reason = []
    for p in results:
        poem_dict = p.to_dict()
        poem_dict["recommend_reason"] = f'匹配搜索"{query}"'
        poems_with_reason.append(poem_dict)

    return jsonify(poems_with_reason)


@app.route("/api/poem/<int:poem_id>/reviews")
def get_poem_reviews(poem_id):
    reviews = Review.query.filter_by(poem_id=poem_id).all()
    result = []
    for r in reviews:
        user = User.query.get(r.user_id)
        result.append(
            {
                "id": r.id,
                "user_id": user.username if user else "匿名",
                "rating": r.rating,
                "comment": r.comment,
            }
        )
    return jsonify(result)


@app.route("/api/poem/<int:poem_id>/allusions")
def get_poem_allusions(poem_id):
    poem = Poem.query.get(poem_id)
    if poem and poem.notes:
        try:
            return jsonify(json.loads(poem.notes))
        except:
            return jsonify([])
    return jsonify([])


@app.route("/api/poem/<int:poem_id>/helper")
def get_poem_helper(poem_id):
    poem = Poem.query.get(poem_id)
    if not poem:
        return jsonify({"author_bio": "", "background": "", "appreciation": ""})

    return jsonify(
        {
            "author_bio": poem.author_bio or "暂无作者生平信息",
            "background": f"[{poem.dynasty}]" if poem.dynasty else "",
            "appreciation": poem.appreciation or "暂无赏析",
        }
    )


@app.route("/api/poem/<int:poem_id>/analysis")
def get_single_poem_analysis(poem_id):
    poem = Poem.query.get(poem_id)
    if not poem:
        return jsonify({"matrix": [], "rhymes": []})

    import re
    from pypinyin import pinyin, Style

    lines = [l.strip() for l in re.split(r"[，。！？；\n]", poem.content) if l.strip()]
    matrix = []
    for line in lines:
        line_pinyin = pinyin(line, style=Style.TONE3, neutral_tone_with_five=True)
        line_matrix = []
        for char, py in zip(line, line_pinyin):
            s = py[0]
            tone = "?"
            if re.match(r"[\u4e00-\u9fa5]", char):
                if s and s[-1].isdigit():
                    t_num = int(s[-1])
                    if t_num in [1, 2]:
                        tone = "平"
                    elif t_num in [3, 4, 5]:
                        tone = "仄"
                else:
                    s2 = pinyin(char, style=Style.TONE2)[0][0]
                    if s2 and s2[-1].isdigit():
                        t_num = int(s2[-1])
                        tone = "平" if t_num in [1, 2] else "仄"
            line_matrix.append({"char": char, "tone": tone})
        matrix.append(line_matrix)

    rhymes = []
    for idx, line in enumerate(lines):
        if not line:
            continue
        last_char = line[-1]
        py_full = pinyin(last_char, style=Style.NORMAL)[0][0]
        vowels = "aeiouü"
        rhyme_part = py_full
        for i in range(len(py_full)):
            if py_full[i] in vowels:
                rhyme_part = py_full[i:]
                break

        rhymes.append({"line": idx + 1, "char": last_char, "rhyme": rhyme_part})

    sentiment_dict = {
        "雄浑": ["大", "长", "云", "山", "河", "壮", "万", "天", "高"],
        "忧思": ["愁", "悲", "泪", "苦", "孤", "恨", "断", "老", "梦"],
        "闲适": ["悠", "闲", "醉", "卧", "月", "酒", "归", "眠", "静"],
        "清丽": ["花", "香", "翠", "色", "红", "绿", "秀", "春", "嫩"],
        "羁旅": ["客", "路", "远", "家", "乡", "雁", "征", "帆", "渡"],
    }
    sentiment_scores = {k: 10 for k in sentiment_dict}
    for char in poem.content:
        for k, words in sentiment_dict.items():
            if char in words:
                sentiment_scores[k] += 15

    tonal_chart_data = []
    char_labels = []

    if matrix:
        for row in matrix:
            for cell in row:
                char_labels.append(cell["char"])
                tonal_chart_data.append(
                    1 if cell["tone"] == "平" else -1 if cell["tone"] == "仄" else 0
                )

    if not tonal_chart_data:
        tonal_chart_data = [0] * len(poem.content.replace("\n", ""))
        char_labels = list(poem.content.replace("\n", ""))

    return jsonify(
        {
            "matrix": matrix,
            "rhymes": rhymes,
            "chart_data": {
                "tonal_sequence": tonal_chart_data,
                "char_labels": char_labels,
                "sentiment": [
                    {"name": k, "value": v} for k, v in sentiment_scores.items()
                ],
                # 情感雷达需要的数据格式 (joy, anger, sorrow, fear, love, zen)
                "emotions": {
                    "joy": min(sentiment_scores.get("雄浑", 10) / 5, 10),
                    "anger": min(sentiment_scores.get("忧思", 10) / 5, 10),
                    "sorrow": min(sentiment_scores.get("羁旅", 10) / 5, 10),
                    "fear": min(sentiment_scores.get("忧思", 10) / 5, 10),
                    "love": min(sentiment_scores.get("闲适", 10) / 5, 10),
                    "zen": min(sentiment_scores.get("清丽", 10) / 5, 10),
                },
            },
        }
    )


def extract_topics_from_comment(comment):
    """从评论中提取主题关键词"""
    import jieba
    import jieba.analyse

    # 使用TF-IDF提取关键词
    keywords = jieba.analyse.extract_tags(comment, topK=5, withWeight=True)

    # 过滤并格式化
    topic_names = []
    for word, weight in keywords:
        if len(word) >= 2 and weight > 0.1:  # 至少2个字符，权重>0.1
            topic_names.append(word)

    return "-".join(topic_names) if topic_names else "未分类"


@app.route("/api/poem/review", methods=["POST"])
def add_review():
    data = request.json
    username = data.get("username")
    poem_id = data.get("poem_id")
    rating = data.get("rating", 5)
    comment = data.get("comment")

    if not all([username, poem_id, comment]):
        return jsonify({"message": "缺失必要信息", "status": "error"}), 400

    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"message": "用户不存在", "status": "error"}), 404

    # 提取评论主题
    topic_names = extract_topics_from_comment(comment)

    new_review = Review(
        user_id=user.id,
        poem_id=poem_id,
        rating=rating,
        comment=comment,
    )
    db.session.add(new_review)
    db.session.commit()
    rec_service.refresh_if_needed(force=True)

    # 评论成功后，返回基于 User-CF + Item-CF 的混合协同过滤推荐
    try:
        # 定义seen_ids变量
        seen_ids = [poem_id]
        # 获取推荐
        poem, reason = rec_service.get_recommendation(user, skip_count=0, seen_ids=seen_ids)
        if poem:
            recommended_poem = poem.to_dict()
            recommended_poem["recommend_reason"] = reason
            return jsonify({
                "status": "success",
                "recommended": recommended_poem
            })
        else:
            return jsonify({"status": "success"})
    except Exception as e:
        print(f"Recommend error: {e}")
        import traceback
        traceback.print_exc()

    # 最终回退：返回成功状态
    return jsonify({"status": "success"})


@app.route("/api/user_preference/<username>")
def get_user_preference(username):
    user = User.query.filter_by(username=username).first()
    if not user or not user.preference_topics:
        return jsonify(
            {"user_id": username, "preference": [], "top_interest": ["通用"]}
        )

    try:
        preference = json.loads(user.preference_topics)
    except:
        preference = []

    return jsonify(
        {
            "user_id": username,
            "preference": preference,
            "top_interest": preference[0]["keywords"] if preference else ["通用"],
        }
    )


@app.route("/api/save_initial_preferences", methods=["POST"])
def save_initial_preferences():
    data = request.json
    username = data.get("username")
    selected_topics = data.get("selected_topics", [])

    if not username:
        return jsonify({"message": "用户名不能为空", "status": "error"}), 400

    if not selected_topics:
        return jsonify({"message": "请至少选择一个主题", "status": "error"}), 400

    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"message": "用户不存在", "status": "error"}), 404

    preference = []
    for i, topic_id in enumerate(selected_topics):
        weight = 1.0 - (i * 0.15)
        preference.append({"topic_id": int(topic_id), "score": max(weight, 0.1)})

    preference.sort(key=lambda x: x["score"], reverse=True)
    user.preference_topics = json.dumps(preference)
    db.session.commit()

    return jsonify(
        {"message": "偏好设置成功", "status": "success", "preference": preference}
    )


@app.route("/api/global/stats")
def get_global_stats():
    try:
        total_users = User.query.count()
        total_poems = Poem.query.count()
        total_reviews = Review.query.count()

        total_views = db.session.query(func.sum(Poem.views)).scalar() or 0
        total_shares = 0

        avg_engagement = (
            round((total_views + total_shares) / (total_poems * 2), 2)
            if total_poems > 0
            else 0
        )

        today = datetime.utcnow().date()
        today_users = User.query.filter(func.date(User.created_at) == today).count()
        today_reviews = Review.query.filter(
            func.date(Review.created_at) == today
        ).count()

        return jsonify(
            {
                "totalUsers": total_users,
                "totalPoems": total_poems,
                "totalReviews": total_reviews,
                "totalViews": total_views,
                "totalShares": total_shares,
                "avgEngagement": f"{avg_engagement * 100}%",
                "todayNewUsers": today_users,
                "todayReviews": today_reviews,
            }
        )
    except Exception as e:
        return jsonify({"error": f"获取统计数据失败: {str(e)}"}), 500


@app.route("/api/global/popular-poems")
def get_popular_poems():
    try:
        time_range = request.args.get("time_range", "all")

        from sqlalchemy import func as sql_func
        from datetime import datetime, timedelta

        base_query = Review.query
        if time_range == "today":
            today = datetime.utcnow().date()
            base_query = base_query.filter(func.date(Review.created_at) == today)
        elif time_range == "week":
            week_ago = datetime.utcnow() - timedelta(days=7)
            base_query = base_query.filter(Review.created_at >= week_ago)
        elif time_range == "month":
            month_ago = datetime.utcnow() - timedelta(days=30)
            base_query = base_query.filter(Review.created_at >= month_ago)

        # 统计每首诗的评论数量
        review_counts = (
            base_query.with_entities(
                Review.poem_id, sql_func.count(Review.id).label("count")
            )
            .group_by(Review.poem_id)
            .all()
        )

        # 构建 poem_id -> count 的映射
        count_map = {poem_id: count for poem_id, count in review_counts}

        # 获取所有诗歌，按评论数排序
        poems = Poem.query.all()

        # 按评论数排序
        sorted_poems = sorted(
            poems, key=lambda p: count_map.get(p.id, 0), reverse=True
        )[:10]

        result = []
        for poem in sorted_poems:
            poem_dict = poem.to_dict()
            poem_dict["review_count"] = count_map.get(poem.id, 0)
            result.append(poem_dict)

        return jsonify(result)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"获取热门诗歌失败: {str(e)}"}), 500


@app.route("/api/global/theme-distribution")
def get_theme_distribution():
    """从poems表的dynasty和author字段统计主题分布"""
    try:
        theme_counts = {}

        # 从poems表统计朝代和作者分布
        poems = Poem.query.all()

        for poem in poems:
            if poem.dynasty:
                dynasty = poem.dynasty.strip()
                if dynasty and dynasty != "未知":
                    theme_counts[dynasty] = theme_counts.get(dynasty, 0) + 1
            if poem.author:
                author = poem.author.strip()
                if author and author != "未知":
                    theme_counts[author] = theme_counts.get(author, 0) + 1

        result = []
        for theme, count in sorted(
            theme_counts.items(), key=lambda x: x[1], reverse=True
        ):
            result.append({"name": theme, "value": count})

        # 如果没有数据，返回默认数据
        if not result:
            result = [
                {"name": "唐", "value": 30},
                {"name": "宋", "value": 25},
                {"name": "李白", "value": 20},
                {"name": "杜甫", "value": 15},
                {"name": "苏轼", "value": 10},
            ]

        return jsonify(result)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"获取主题分布失败: {str(e)}"}), 500


@app.route("/api/global/dynasty-distribution")
def get_dynasty_distribution():
    try:
        import os
        import json
        from datetime import date
        from sqlalchemy import func
        
        # 缓存文件路径
        cache_dir = os.path.join(os.path.dirname(__file__), "data", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "dynasty_distribution_cache.json")
        
        # 检查缓存是否存在且是今天的
        current_date = date.today().isoformat()
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                if cache_data.get("date") == current_date:
                    print("[Cache] 使用缓存的朝代分布数据")
                    return jsonify(cache_data.get("data", []))
            except Exception as e:
                print(f"[Cache] 读取缓存失败: {e}")
        
        # 缓存不存在或过期，重新计算
        print("[Cache] 缓存过期，重新计算朝代分布")
        
        # 按照评论数量统计朝代分布
        dynasty_stats = db.session.query(
            Poem.dynasty,
            func.count(Review.id).label('review_count')
        ).join(
            Review, Review.poem_id == Poem.id
        ).group_by(
            Poem.dynasty
        ).all()
        
        # 转换为字典
        dynasty_dict = {}
        for dynasty, count in dynasty_stats:
            dynasty_name = dynasty or "其他"
            dynasty_dict[dynasty_name] = count
        
        # 按评论数量排序
        sorted_dynasties = sorted(dynasty_dict.items(), key=lambda x: x[1], reverse=True)
        
        # 转换为结果格式
        result = []
        for dynasty, count in sorted_dynasties:
            result.append({"name": dynasty, "value": count})
        
        # 如果没有评论，返回默认数据
        if not result:
            result = [
                {"name": "唐", "value": 50},
                {"name": "宋", "value": 30},
                {"name": "元", "value": 10},
                {"name": "明", "value": 5},
                {"name": "清", "value": 5}
            ]
        
        # 保存到缓存
        cache_data = {
            "date": current_date,
            "data": result
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print("[Cache] 朝代分布数据已缓存")

        return jsonify(result)
    except Exception as e:
        print(f"Error in get_dynasty_distribution: {e}")
        # 返回默认数据，而不是500错误
        return jsonify([
            {"name": "唐", "value": 50},
            {"name": "宋", "value": 30},
            {"name": "元", "value": 10},
            {"name": "明", "value": 5},
            {"name": "清", "value": 5}
        ])


@app.route("/api/global/trends")
def get_global_trends():
    try:
        period = request.args.get("period", "week")

        dates = []
        user_counts = []
        review_counts = []

        if period == "week":
            days = 7
        elif period == "month":
            days = 30
        else:
            days = 90

        for i in range(days):
            date = datetime.utcnow() - timedelta(days=i)
            dates.append(date.strftime("%m-%d"))

            day_users = User.query.filter(
                func.date(User.created_at) == date.date()
            ).count()
            day_reviews = Review.query.filter(
                func.date(Review.created_at) == date.date()
            ).count()

            user_counts.append(day_users)
            review_counts.append(day_reviews)

        return jsonify(
            {
                "dates": dates[::-1],
                "users": user_counts[::-1],
                "reviews": review_counts[::-1],
            }
        )
    except Exception as e:
        return jsonify({"error": f"获取趋势数据失败: {str(e)}"}), 500


@app.route("/api/user/<username>/stats")
def get_user_profile_stats(username):
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify(
            {"totalReads": 0, "avgRating": 0, "reviewCount": 0, "activeDays": 0}
        )

    reviews = Review.query.filter_by(user_id=user.id).all()
    review_count = len(reviews)
    avg_rating = (
        sum([r.rating for r in reviews]) / review_count if review_count > 0 else 0
    )

    active_days = (datetime.utcnow() - user.created_at).days + 1
    total_reads = review_count * 3 + 5

    return jsonify(
        {
            "totalReads": total_reads,
            "avgRating": round(avg_rating, 1),
            "reviewCount": review_count,
            "activeDays": active_days,
        }
    )


@app.route("/api/user/<username>/reviews")
def get_user_reviews(username):
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify([])

    reviews = Review.query.filter_by(user_id=user.id).all()
    result = []
    for r in reviews:
        poem = Poem.query.get(r.poem_id)
        if poem:
            result.append(
                {
                    "id": r.id,
                    "poem_id": r.poem_id,
                    "poem": {
                        "id": poem.id,
                        "title": poem.title,
                        "author": poem.author,
                        "content": poem.content
                    },
                    "rating": r.rating,
                    "comment": r.comment,
                    "created_at": r.created_at.isoformat() if r.created_at else None
                }
            )
    return jsonify(result)


@app.route("/api/user/<username>/preferences")
def get_user_prefs_api(username):
    """基于用户实际评分行为动态计算偏好"""
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify(
            {
                "preferences": [
                    {"topic_name": "山水田园", "percentage": 40, "color": "#cf3f35"},
                    {"topic_name": "思乡情怀", "percentage": 35, "color": "#bfa46f"},
                    {"topic_name": "豪迈边塞", "percentage": 25, "color": "#1a1a1a"},
                ]
            }
        )

    # 获取用户评论过的诗歌
    reviewed_poems = (
        db.session.query(Poem).join(Review).filter(Review.user_id == user.id).all()
    )

    if not reviewed_poems:
        return jsonify(
            {
                "preferences": [
                    {"topic_name": "山水田园", "percentage": 40, "color": "#cf3f35"},
                    {"topic_name": "思乡情怀", "percentage": 35, "color": "#bfa46f"},
                    {"topic_name": "豪迈边塞", "percentage": 25, "color": "#1a1a1a"},
                ]
            }
        )

    # 统计作者和朝代分布
    author_counts = {}
    dynasty_counts = {}
    for poem in reviewed_poems:
        if poem.author:
            author_counts[poem.author] = author_counts.get(poem.author, 0) + 1
        if poem.dynasty:
            dynasty_counts[poem.dynasty] = dynasty_counts.get(poem.dynasty, 0) + 1

    total = len(reviewed_poems)
    preferences = []

    # 合并作者和朝代作为"主题"
    all_topics = {}
    for author, count in author_counts.items():
        all_topics[f"作者:{author}"] = count
    for dynasty, count in dynasty_counts.items():
        all_topics[f"朝代:{dynasty}"] = count

    # 排序并转换为百分比
    sorted_topics = sorted(all_topics.items(), key=lambda x: x[1], reverse=True)[:5]

    colors = ["#cf3f35", "#bfa46f", "#1a1a1a", "#1b1a8a", "#1b8a1a"]
    for i, (topic_name, count) in enumerate(sorted_topics):
        percentage = int((count / total) * 100)
        preferences.append(
            {
                "topic_name": topic_name,
                "percentage": percentage,
                "color": colors[i % len(colors)],
            }
        )

    # 如果没有统计数据，返回默认值
    if not preferences:
        preferences = [
            {"topic_name": "山水田园", "percentage": 40, "color": "#cf3f35"},
            {"topic_name": "思乡情怀", "percentage": 35, "color": "#bfa46f"},
            {"topic_name": "豪迈边塞", "percentage": 25, "color": "#1a1a1a"},
        ]

    return jsonify({"preferences": preferences})


@app.route("/api/global/wordcloud")
def get_global_wordcloud():
    """获取全局词云数据"""
    try:
        # 获取所有诗歌的 topic_tags 主题
        poems = Poem.query.all()
        word_counts = {}

        for poem in poems:
            if poem.topic_tags:
                # 主题格式: "主题1-主题2-主题3"
                topics = poem.topic_tags.split("-")
                for topic in topics:
                    topic = topic.strip()
                    if topic:
                        word_counts[topic] = word_counts.get(topic, 0) + 1

        # 转换为词云格式
        result = [{"name": k, "value": v} for k, v in word_counts.items()]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"获取词云数据失败: {str(e)}"}), 500


@app.route("/api/user/<username>/wordcloud")
def get_user_wordcloud(username):
    """获取用户词云数据"""
    try:
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify([])

        # 获取用户评论过的诗歌主题
        reviewed_poems = (
            db.session.query(Poem).join(Review).filter(Review.user_id == user.id).all()
        )

        word_counts = {}
        for poem in reviewed_poems:
            if poem.topic_tags:
                topics = poem.topic_tags.split("-")
                for topic in topics:
                    topic = topic.strip()
                    if topic:
                        word_counts[topic] = word_counts.get(topic, 0) + 1

        result = [{"name": k, "value": v} for k, v in word_counts.items()]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"获取用户词云数据失败: {str(e)}"}), 500


@app.route("/api/visual/wordcloud")
def get_visual_wordcloud():
    """获取可视化词云数据"""
    try:
        poems = Poem.query.all()
        word_counts = {}

        for poem in poems:
            if poem.topic_tags:
                topics = poem.topic_tags.split("-")
                for topic in topics:
                    topic = topic.strip()
                    if topic:
                        word_counts[topic] = word_counts.get(topic, 0) + 1

        result = [{"name": k, "value": v} for k, v in word_counts.items()]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/visual/stats")
def get_visual_stats():
    """获取可视化统计数据（雷达图和桑基图）"""
    try:
        user_id = request.args.get("user_id")

        result = {
            "total_poems": Poem.query.count(),
            "total_reviews": Review.query.count(),
            "total_users": User.query.count(),
        }

        # 雷达图数据 - 用户偏好的诗歌体裁分布
        radar_data = {"indicator": [], "value": []}

        if user_id:
            user = User.query.filter_by(username=user_id).first()
            if user:
                # 获取用户评论过的诗歌体裁分布
                reviewed_poems = (
                    db.session.query(Poem)
                    .join(Review)
                    .filter(Review.user_id == user.id)
                    .all()
                )
                genre_counts = {}
                for p in reviewed_poems:
                    genre = p.genre_type or "其他"
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1

                # 转换为雷达图数据
                for genre, count in sorted(genre_counts.items(), key=lambda x: -x[1])[
                    :6
                ]:
                    radar_data["indicator"].append({"name": genre, "max": 100})
                    radar_data["value"].append(count)

        # 如果没有用户数据，返回默认雷达图
        if not radar_data["indicator"]:
            radar_data = {
                "indicator": [
                    {"name": "诗", "max": 100},
                    {"name": "词", "max": 100},
                    {"name": "曲", "max": 100},
                    {"name": "赋", "max": 100},
                    {"name": "古体", "max": 100},
                    {"name": "近体", "max": 100},
                ],
                "value": [30, 25, 15, 10, 10, 10],
            }

        result["radar_data"] = radar_data

        # 桑基图数据 - 朝代-诗人体裁关系
        sankey_data = {"nodes": [], "links": []}

        # 获取所有诗歌的朝代和体裁信息
        dynasty_genre_map = {}
        poems = Poem.query.all()
        for p in poems:
            dynasty = p.dynasty or "未知"
            genre = p.genre_type or "其他"
            key = (dynasty, genre)
            dynasty_genre_map[key] = dynasty_genre_map.get(key, 0) + 1

        # 构建桑基图节点和链接
        dynasties = set()
        genres = set()
        links = []

        for (dynasty, genre), count in dynasty_genre_map.items():
            dynasties.add(dynasty)
            genres.add(genre)
            links.append({"source": dynasty, "target": genre, "value": count})

        sankey_data["nodes"] = [{"name": d} for d in dynasties] + [
            {"name": g} for g in genres
        ]
        sankey_data["links"] = links

        result["sankey_data"] = sankey_data

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/visual/semantic-similarity")
def get_semantic_similarity():
    """基于Item-CF的诗歌相似度可视化数据"""
    try:
        from core.hybrid_cf import HybridCFRecommender

        recommender = HybridCFRecommender()

        poems = [
            {
                "id": p.id,
                "content": p.content or "",
                "title": p.title or "",
                "author": p.author or "未知",
                "dynasty": p.dynasty or "未知",
            }
            for p in Poem.query.limit(50).all()  # 限制50首诗以保证性能
        ]
        interactions = [
            {
                "user_id": r.user_id,
                "poem_id": r.poem_id,
                "rating": r.rating,
            }
            for r in Review.query.limit(1000).all()  # 限制1000条交互
        ]
        
        recommender.fit(poems, interactions)

        if recommender.item_similarity is not None:
            nodes = []
            links = []

            for i, poem in enumerate(poems):
                nodes.append({
                    "id": poem["id"],
                    "name": poem["title"],
                    "author": poem["author"],
                    "dynasty": poem["dynasty"],
                })

            for i in range(len(poems)):
                for j in range(i + 1, len(poems)):
                    similarity = float(recommender.item_similarity[i][j])
                    if similarity > 0.35:
                        links.append({
                            "source": poems[i]["id"],
                            "target": poems[j]["id"],
                            "value": similarity
                        })
            
            return jsonify({"nodes": nodes, "links": links})
        else:
            return jsonify({"error": "协同过滤相似度未生成"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    with app.app_context():
        try:
            db.create_all()
            print("Database initialized.")
        except Exception as e:
            print(f"Database init failed: {e}")

    app.run(debug=True, port=5000)


# === Personal Analysis APIs ===


@app.route("/api/user/<username>/time-analysis")
def get_user_time_analysis(username):
    """获取用户时间分析"""
    try:
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify(
                {
                    "insights": [
                        {"time": "子时", "value": 5},
                        {"time": "丑时", "value": 3},
                        {"time": "寅时", "value": 2},
                        {"time": "卯时", "value": 8},
                        {"time": "辰时", "value": 10},
                        {"time": "巳时", "value": 12},
                        {"time": "午时", "value": 15},
                        {"time": "未时", "value": 10},
                        {"time": "申时", "value": 8},
                        {"time": "酉时", "value": 12},
                        {"time": "戌时", "value": 10},
                        {"time": "亥时", "value": 5},
                    ]
                }
            )

        # 获取用户的评论记录
        reviews = Review.query.filter_by(user_id=user.id).all()

        if not reviews:
            return jsonify(
                {
                    "insights": [
                        {"time": "子时", "value": 5},
                        {"time": "丑时", "value": 3},
                        {"time": "寅时", "value": 2},
                        {"time": "卯时", "value": 8},
                        {"time": "辰时", "value": 10},
                        {"time": "巳时", "value": 12},
                        {"time": "午时", "value": 15},
                        {"time": "未时", "value": 10},
                        {"time": "申时", "value": 8},
                        {"time": "酉时", "value": 12},
                        {"time": "戌时", "value": 10},
                        {"time": "亥时", "value": 5},
                    ]
                }
            )

        # 按小时统计
        hour_counts = {}
        for r in reviews:
            if r.created_at:
                hour = r.created_at.hour
                hour_counts[hour] = hour_counts.get(hour, 0) + 1

        # 中国传统十二时辰
        time_periods = [
            ("子时", 23, 1),
            ("丑时", 1, 3),
            ("寅时", 3, 5),
            ("卯时", 5, 7),
            ("辰时", 7, 9),
            ("巳时", 9, 11),
            ("午时", 11, 13),
            ("未时", 13, 15),
            ("申时", 15, 17),
            ("酉时", 17, 19),
            ("戌时", 19, 21),
            ("亥时", 21, 23),
        ]

        # 合并到时辰
        period_counts = {name: 0 for name, _, _ in time_periods}
        for hour, count in hour_counts.items():
            for name, start, end in time_periods:
                if start <= hour < end or (
                    start > end and (hour >= start or hour < end)
                ):
                    period_counts[name] += count
                    break

        # 转换为前端需要的格式（按传统顺序）
        ordered_periods = [
            "子时",
            "丑时",
            "寅时",
            "卯时",
            "辰时",
            "巳时",
            "午时",
            "未时",
            "申时",
            "酉时",
            "戌时",
            "亥时",
        ]
        insights = []
        for period in ordered_periods:
            count = period_counts.get(period, 0)
            if count > 0:
                insights.append({"time": period, "value": count})

        if not insights:
            return jsonify(
                {
                    "insights": [
                        {"time": "子时", "value": 5},
                        {"time": "丑时", "value": 3},
                        {"time": "寅时", "value": 2},
                        {"time": "卯时", "value": 8},
                        {"time": "辰时", "value": 10},
                        {"time": "巳时", "value": 12},
                        {"time": "午时", "value": 15},
                        {"time": "未时", "value": 10},
                        {"time": "申时", "value": 8},
                        {"time": "酉时", "value": 12},
                        {"time": "戌时", "value": 10},
                        {"time": "亥时", "value": 5},
                    ]
                }
            )

        return jsonify({"insights": insights})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/user/<username>/form-stats")
def get_user_form_stats(username):
    """获取用户偏好的诗歌体裁统计"""
    try:
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify(
                [
                    {"name": "诗", "value": 30},
                    {"name": "词", "value": 25},
                    {"name": "曲", "value": 15},
                    {"name": "赋", "value": 10},
                    {"name": "其他", "value": 20},
                ]
            )

        # 获取用户评论过的诗歌
        reviewed_poems = (
            db.session.query(Poem).join(Review).filter(Review.user_id == user.id).all()
        )

        if not reviewed_poems:
            return jsonify(
                [
                    {"name": "诗", "value": 30},
                    {"name": "词", "value": 25},
                    {"name": "曲", "value": 15},
                    {"name": "赋", "value": 10},
                    {"name": "其他", "value": 20},
                ]
            )

        # 统计体裁
        form_counts = {}
        for poem in reviewed_poems:
            genre = poem.genre_type or "其他"
            form_counts[genre] = form_counts.get(genre, 0) + 1

        result = [{"name": k, "value": v} for k, v in form_counts.items()]
        return jsonify(result)
    except Exception as e:
        print(f"Error in get_user_form_stats: {e}")
        # 返回默认数据，而不是500错误
        return jsonify(
            [
                {"name": "诗", "value": 30},
                {"name": "词", "value": 25},
                {"name": "曲", "value": 15},
                {"name": "赋", "value": 10},
                {"name": "其他", "value": 20},
            ]
        )


@app.route("/api/recommend_one/<username>", methods=["GET"])
def get_recommend_one(username):
    """推荐单首诗歌（首页使用）"""
    current_id = request.args.get("current_id", "")
    skip_count = int(request.args.get("skip_count", 0))
    seen_ids_str = request.args.get("seen_ids", "")
    seen_ids = [int(x) for x in seen_ids_str.split(",") if x.strip().isdigit()]
    
    try:
        user = User.query.filter_by(username=username).first()
        if not user:
            # 游客返回入门推荐
            poem = Poem.query.order_by(func.rand()).first()
            if poem:
                res = poem.to_dict()
                res["recommend_reason"] = "新用户入门推荐"
                return jsonify(res)
            return jsonify({"error": "No poems available"}), 404
        
        # 使用统一的推荐方法
        poem, reason = rec_service.get_recommendation(user, skip_count=skip_count, seen_ids=seen_ids)
        if poem:
            res = poem.to_dict()
            res["recommend_reason"] = reason
            return jsonify(res)
    except Exception as e:
        print(f"Recommendation error: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== 最终回退：完全随机 ==========
    fallback_query = Poem.query
    exclude_ids = set(seen_ids)
    if exclude_ids:
        fallback_query = fallback_query.filter(~Poem.id.in_(exclude_ids))
    fallback = fallback_query.order_by(func.rand()).first()
    
    if fallback:
        res = fallback.to_dict()
        res["recommend_reason"] = "随机发现"
        return jsonify(res)
    
    # 如果所有诗歌都看过，重置并随机推荐
    fallback = Poem.query.order_by(func.rand()).first()
    if fallback:
        res = fallback.to_dict()
        res["recommend_reason"] = "重新开始随机推荐"
        return jsonify(res)
    
    return jsonify({"error": "Poem list empty"}), 404


@app.route("/api/user/<username>/recommendations")
def get_user_recommendations(username):
    """获取用户推荐诗歌"""
    try:
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify({"poems": []})

        user_reviews = Review.query.filter_by(user_id=user.id).all()
        review_count = len(user_reviews)
        exclude_ids = {review.poem_id for review in user_reviews}
        user_interactions = [
            {
                "user_id": review.user_id,
                "poem_id": review.poem_id,
                "rating": review.rating,
                "created_at": review.created_at or datetime.utcnow(),
            }
            for review in user_reviews
        ]

        rec_service.refresh_if_needed()
        recommendations = rec_service.recommender.recommend(
            user_interactions,
            exclude_ids=exclude_ids,
            top_k=10,
        )

        result = []
        for rec in recommendations:
            poem = Poem.query.get(rec["poem_id"])
            if not poem:
                continue
            if review_count == 0:
                reason = "新用户入门推荐"
            elif review_count < 3:
                reason = "根据你刚开始的评分记录推荐"
            elif review_count < 10:
                reason = "根据你的近期评分偏好推荐"
            else:
                reason = "基于相似用户与相似诗作推荐"
            result.append(
                {
                    "id": poem.id,
                    "title": poem.title,
                    "author": poem.author,
                    "content": poem.content[:100] + "..."
                    if poem.content and len(poem.content) > 100
                    else poem.content,
                    "reason": reason,
                }
            )

        if not result:
            fallback_poems = Poem.query.order_by(func.rand()).limit(10).all()
            result = [
                {
                    "id": poem.id,
                    "title": poem.title,
                    "author": poem.author,
                    "content": poem.content[:100] + "..."
                    if poem.content and len(poem.content) > 100
                    else poem.content,
                    "reason": "入门推荐",
                }
                for poem in fallback_poems
            ]

        return jsonify({"poems": result})
    except Exception as e:
        return jsonify({"poems": [], "error": str(e)}), 500


@app.route("/api/user/<username>/poet-topic-sankey")
def get_user_poet_topic_sankey(username):
    """获取诗人-朝代桑基图数据"""
    try:
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify(
                {
                    "nodes": [
                        {"name": "李白"},
                        {"name": "杜甫"},
                        {"name": "苏轼"},
                        {"name": "唐"},
                        {"name": "宋"},
                        {"name": "宋"},
                    ],
                    "links": [
                        {"source": "李白", "target": "唐", "value": 15},
                        {"source": "杜甫", "target": "唐", "value": 12},
                        {"source": "苏轼", "target": "宋", "value": 20},
                    ],
                }
            )

        # 获取用户评论过的诗歌
        reviewed_poems = (
            db.session.query(Poem).join(Review).filter(Review.user_id == user.id).all()
        )

        if not reviewed_poems:
            return jsonify(
                {
                    "nodes": [
                        {"name": "李白"},
                        {"name": "杜甫"},
                        {"name": "苏轼"},
                        {"name": "唐"},
                        {"name": "宋"},
                        {"name": "宋"},
                    ],
                    "links": [
                        {"source": "李白", "target": "唐", "value": 15},
                        {"source": "杜甫", "target": "唐", "value": 12},
                        {"source": "苏轼", "target": "宋", "value": 20},
                    ],
                }
            )

        # 统计诗人-朝代关系
        poet_dynasties = {}
        for poem in reviewed_poems:
            author = poem.author or "未知作者"
            dynasty = poem.dynasty or "未知朝代"
            key = (author, dynasty)
            poet_dynasties[key] = poet_dynasties.get(key, 0) + 1

        # 构建节点和链接
        authors = set()
        dynasties = set()
        links = []

        for (author, dynasty), value in poet_dynasties.items():
            authors.add(author)
            dynasties.add(dynasty)
            links.append({"source": author, "target": dynasty, "value": value})

        nodes = [{"name": a} for a in authors] + [{"name": d} for d in dynasties]

        if not nodes:
            return jsonify(
                {
                    "nodes": [
                        {"name": "李白"},
                        {"name": "杜甫"},
                        {"name": "苏轼"},
                        {"name": "唐"},
                        {"name": "宋"},
                        {"name": "宋"},
                    ],
                    "links": [
                        {"source": "李白", "target": "唐", "value": 15},
                        {"source": "杜甫", "target": "唐", "value": 12},
                        {"source": "苏轼", "target": "宋", "value": 20},
                    ],
                }
            )

        return jsonify({"nodes": nodes, "links": links})
    except Exception as e:
        return jsonify({"nodes": [], "links": [], "error": str(e)}), 500


@app.route("/api/user/<username>/comment-topics")
def get_user_comment_topics(username):
    """获取用户评论的主题分布"""
    try:
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify([])

        # 缓存文件路径
        import os
        import json
        from datetime import date
        cache_dir = os.path.join(os.path.dirname(__file__), "data", "cache", "user_topics")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{username}_topics_cache.json")
        
        # 获取用户评论数量
        review_count = Review.query.filter_by(user_id=user.id).count()
        
        # 检查缓存是否存在且有效
        current_date = date.today().isoformat()
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                if cache_data.get("date") == current_date and cache_data.get("review_count") == review_count:
                    print(f"[Cache] 使用缓存的用户主题数据: {username}")
                    return jsonify(cache_data.get("topics", []))
            except Exception as e:
                print(f"[Cache] 读取用户主题缓存失败: {e}")
        
        # 缓存不存在或过期，重新计算
        print(f"[Cache] 缓存过期或评论数变化，重新计算用户主题: {username}")
        # 获取用户的所有评论
        reviews = Review.query.filter_by(user_id=user.id).all()
        comments = [review.comment for review in reviews if review.comment]

        if not comments:
            # 即使没有评论，也保存空缓存
            cache_data = {
                "date": current_date,
                "review_count": review_count,
                "topics": []
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            return jsonify([])

        # 使用BERTopic提取主题
        topics = get_topic_service().get_user_topics(user.id, comments)
        
        # 保存到缓存
        cache_data = {
            "date": current_date,
            "review_count": review_count,
            "topics": topics
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"[Cache] 用户主题数据已缓存: {username}")

        return jsonify(topics)
    except Exception as e:
        print(f"Error getting user comment topics: {e}")
        return jsonify([])


@app.route("/api/user/<username>/sentiment-analysis")
def get_user_sentiment_analysis(username):
    """获取用户评论的情感倾向分析"""
    try:
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify([])

        # 缓存文件路径
        import os
        import json
        from datetime import date
        cache_dir = os.path.join(os.path.dirname(__file__), "data", "cache", "user_sentiment")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{username}_sentiment_cache.json")
        
        # 获取用户评论数量
        review_count = Review.query.filter_by(user_id=user.id).count()
        
        # 检查缓存是否存在且有效
        current_date = date.today().isoformat()
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                if cache_data.get("date") == current_date and cache_data.get("review_count") == review_count:
                    print(f"[Cache] 使用缓存的用户情感数据: {username}")
                    return jsonify(cache_data.get("sentiment", []))
            except Exception as e:
                print(f"[Cache] 读取用户情感缓存失败: {e}")
        
        # 缓存不存在或过期，重新计算
        print(f"[Cache] 缓存过期或评论数变化，重新计算用户情感: {username}")
        # 获取用户的所有评论
        reviews = Review.query.filter_by(user_id=user.id).all()
        
        if not reviews:
            # 即使没有评论，也保存空缓存
            cache_data = {
                "date": current_date,
                "review_count": review_count,
                "sentiment": []
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            return jsonify([])
        
        # 情感词汇映射（喜怒哀惧爱禅）
        emotion_words = {
            "joy": ["喜", "开心", "快乐", "高兴", "愉快", "欢乐", "喜悦", "欣喜", "兴奋", "欢快"],
            "anger": ["怒", "愤怒", "生气", "恼火", "气愤", "恼怒", "愤慨", "暴怒", "盛怒"],
            "sorrow": ["哀", "悲伤", "难过", "伤心", "悲痛", "哀伤", "忧郁", "沮丧", "失落"],
            "fear": ["惧", "害怕", "恐惧", "担忧", "忧虑", "焦虑", "惶恐", "惊恐", "畏惧"],
            "love": ["爱", "喜欢", "热爱", "喜爱", "欣赏", "赞美", "敬佩", "感动", "珍惜"],
            "zen": ["禅", "平静", "宁静", "平和", "安详", "淡定", "从容", "超脱", "冥想"]
        }
        
        # 初始化情感得分
        emotion_scores = {
            "joy": 0,
            "anger": 0,
            "sorrow": 0,
            "fear": 0,
            "love": 0,
            "zen": 0
        }
        
        # 分析每条评论的情感
        sentiment_data = []
        for review in reviews:
            if review.comment:
                comment = review.comment
                review_emotions = {
                    "joy": 0,
                    "anger": 0,
                    "sorrow": 0,
                    "fear": 0,
                    "love": 0,
                    "zen": 0
                }
                
                # 计算每条评论的情感得分
                for emotion, words in emotion_words.items():
                    score = sum(1 for word in words if word in comment)
                    review_emotions[emotion] = score
                    emotion_scores[emotion] += score
                
                # 获取诗歌信息
                poem = Poem.query.get(review.poem_id)
                if poem:
                    sentiment_data.append({
                        "poem_title": poem.title,
                        "poem_author": poem.author,
                        "rating": review.rating,
                        "emotions": review_emotions,
                        "comment": comment[:50] + "..." if len(comment) > 50 else comment
                    })
        
        # 计算总得分
        total_score = sum(emotion_scores.values())
        if total_score == 0:
            total_score = 1  # 避免除以零
        
        # 转换为雷达图数据格式
        radar_data = {
            "indicator": [
                {"name": "喜", "max": 100},
                {"name": "怒", "max": 100},
                {"name": "哀", "max": 100},
                {"name": "惧", "max": 100},
                {"name": "爱", "max": 100},
                {"name": "禅", "max": 100}
            ],
            "value": [
                min((emotion_scores["joy"] / total_score) * 100, 100),
                min((emotion_scores["anger"] / total_score) * 100, 100),
                min((emotion_scores["sorrow"] / total_score) * 100, 100),
                min((emotion_scores["fear"] / total_score) * 100, 100),
                min((emotion_scores["love"] / total_score) * 100, 100),
                min((emotion_scores["zen"] / total_score) * 100, 100)
            ]
        }
        
        result = {
            "radar_data": radar_data,
            "detailed_data": sentiment_data
        }
        
        # 保存到缓存
        cache_data = {
            "date": current_date,
            "review_count": review_count,
            "sentiment": result
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"[Cache] 用户情感数据已缓存: {username}")

        return jsonify(result)
    except Exception as e:
        print(f"Error getting user sentiment analysis: {e}")
        return jsonify({"radar_data": {"indicator": [], "value": []}, "detailed_data": []})


@app.route("/api/user/<username>/reading-pattern")
def get_user_reading_pattern(username):
    """获取用户的阅读时间模式"""
    try:
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify([])

        # 缓存文件路径
        import os
        import json
        from datetime import date
        cache_dir = os.path.join(os.path.dirname(__file__), "data", "cache", "user_pattern")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{username}_pattern_cache.json")
        
        # 获取用户评论数量
        review_count = Review.query.filter_by(user_id=user.id).count()
        
        # 检查缓存是否存在且有效
        current_date = date.today().isoformat()
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                if cache_data.get("date") == current_date and cache_data.get("review_count") == review_count:
                    print(f"[Cache] 使用缓存的用户阅读模式数据: {username}")
                    return jsonify(cache_data.get("pattern", []))
            except Exception as e:
                print(f"[Cache] 读取用户阅读模式缓存失败: {e}")
        
        # 缓存不存在或过期，重新计算
        print(f"[Cache] 缓存过期或评论数变化，重新计算用户阅读模式: {username}")
        # 获取用户的所有评论
        reviews = Review.query.filter_by(user_id=user.id).all()
        
        if not reviews:
            # 即使没有评论，也保存空缓存
            cache_data = {
                "date": current_date,
                "review_count": review_count,
                "pattern": []
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            return jsonify([])
        
        # 按小时统计阅读模式
        hourly_pattern = {hour: 0 for hour in range(24)}
        for review in reviews:
            if review.created_at:
                hour = review.created_at.hour
                hourly_pattern[hour] += 1
        
        # 转换为前端需要的格式
        pattern_data = []
        for hour, count in hourly_pattern.items():
            pattern_data.append({
                "hour": hour,
                "count": count,
                "time_label": f"{hour:02d}:00"
            })
        
        # 保存到缓存
        cache_data = {
            "date": current_date,
            "review_count": review_count,
            "pattern": pattern_data
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"[Cache] 用户阅读模式数据已缓存: {username}")

        return jsonify(pattern_data)
    except Exception as e:
        print(f"Error getting user reading pattern: {e}")
        return jsonify([])


@app.route("/api/global/comment-topics")
def get_global_comment_topics():
    """获取全站评论的主题分布"""
    try:
        import os
        import json
        from datetime import datetime, date
        
        # 缓存文件路径
        cache_dir = os.path.join(os.path.dirname(__file__), "data", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "global_comment_topics_cache.json")
        
        # 检查缓存是否存在且是今天的
        current_date = date.today().isoformat()
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                if cache_data.get("date") == current_date:
                    print("[Cache] 使用缓存的主题分布数据")
                    return jsonify(cache_data.get("topics", []))
            except Exception as e:
                print(f"[Cache] 读取缓存失败: {e}")
        
        # 缓存不存在或过期，重新计算
        print("[Cache] 缓存过期，重新计算主题分布")
        # 获取所有用户的评论
        reviews = Review.query.all()
        comments = [review.comment for review in reviews if review.comment]

        if not comments:
            # 即使没有评论，也保存空缓存
            cache_data = {
                "date": current_date,
                "topics": []
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            return jsonify([])

        # 使用BERTopic提取主题
        topics = get_topic_service().get_global_topics(comments)
        
        # 保存到缓存
        cache_data = {
            "date": current_date,
            "topics": topics
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print("[Cache] 主题分布数据已缓存")

        return jsonify(topics)
    except Exception as e:
        print(f"Error getting global comment topics: {e}")
        return jsonify([])
