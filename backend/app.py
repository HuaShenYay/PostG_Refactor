from flask import Flask, jsonify, request
from flask_cors import CORS
from config import Config
from models import db, User, Poem, Review
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import text, func
import json
import os

app = Flask(__name__)
app.config.from_object(Config)

CORS(app)
db.init_app(app)

recommender = None


def init_recommender():
    """初始化推荐系统"""
    global recommender
    try:
        from core.hybrid_strategy import HybridRecommender

        with app.app_context():
            poems = Poem.query.all()
            interactions = []

            for r in Review.query.all():
                interactions.append(
                    {
                        "user_id": r.user_id,
                        "poem_id": r.poem_id,
                        "rating": r.rating,
                        "created_at": r.created_at or datetime.utcnow(),
                        "liked": r.liked,
                    }
                )

            poems_data = [
                {"id": p.id, "content": p.content, "title": p.title} for p in poems
            ]

            if poems_data and interactions:
                recommender = HybridRecommender.fit(poems_data, interactions)
                print("[System] Recommender initialized successfully")
                return recommender
    except Exception as e:
        print(f"[System] Recommender init failed: {e}")
    return None


@app.route("/")
def hello_world():
    return "Poetry Recommendation Engine (BERTopic Hybrid) is Running!"


@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"message": "请输入账号和密码", "status": "error"}), 400

    user = User.query.filter_by(username=username).first()

    if user and user.check_password(password):
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
        new_user = User(username=username, password_hash=password)
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"message": "注册成功，即将开启诗意之旅", "status": "success"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"message": f"注册失败: {str(e)}", "status": "error"}), 500


@app.route("/api/user/update", methods=["POST"])
def update_user():
    data = request.json
    old_username = data.get("old_username")
    new_username = data.get("new_username")
    new_password = data.get("new_password")

    if not old_username:
        return jsonify({"message": "无效的操作", "status": "error"}), 400

    user = User.query.filter_by(username=old_username).first()
    if not user:
        return jsonify({"message": "用户不存在", "status": "error"}), 404

    if new_username and new_username != old_username:
        existing = User.query.filter_by(username=new_username).first()
        if existing:
            return jsonify({"message": "新称谓已被占用", "status": "error"}), 400
        user.username = new_username

    if new_password:
        user.password_hash = new_password

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

    results = (
        Poem.query.filter(
            (Poem.title.ilike(f"%{query}%"))
            | (Poem.author.ilike(f"%{query}%"))
            | (Poem.content.ilike(f"%{query}%"))
        )
        .limit(20)
        .all()
    )

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
            },
        }
    )


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

    new_review = Review(
        user_id=user.id, poem_id=poem_id, rating=rating, comment=comment
    )
    db.session.add(new_review)
    db.session.commit()

    return jsonify({"message": "雅评已收录", "status": "success"})


@app.route("/api/recommend_one/<username>")
def recommend_one(username):
    """智能推荐诗歌"""
    import random

    user = User.query.filter_by(username=username).first()
    if not user:
        poems = Poem.query.limit(6).all()
        return jsonify([p.to_dict() for p in poems])

    current_id = request.args.get("current_id", type=int)
    skip_count = request.args.get("skip_count", 0, type=int)

    with app.app_context():
        user_interactions = []
        for r in Review.query.filter_by(user_id=user.id).all():
            user_interactions.append(
                {
                    "user_id": r.user_id,
                    "poem_id": r.poem_id,
                    "rating": r.rating,
                    "created_at": r.created_at,
                    "liked": r.liked,
                }
            )

        all_interactions = []
        for r in Review.query.all():
            all_interactions.append(
                {
                    "user_id": r.user_id,
                    "poem_id": r.poem_id,
                    "rating": r.rating,
                    "created_at": r.created_at,
                    "liked": r.liked,
                }
            )

        if skip_count > 0 and skip_count % 5 == 0:
            all_poem_ids = [p.id for p in Poem.query.all()]
            reviewed_poem_ids = [r.poem_id for r in Review.query.all()]
            unseen_ids = list(set(all_poem_ids) - set(reviewed_poem_ids))

            if unseen_ids:
                poem_obj = Poem.query.filter(Poem.id.in_(unseen_ids)).first()
                if poem_obj:
                    res = poem_obj.to_dict()
                    res["recommend_reason"] = "为您推荐一首尚未被发现的佳作"
                    return jsonify(res)

        try:
            from core.hybrid_strategy import HybridRecommender

            poems = Poem.query.all()
            poems_data = [
                {"id": p.id, "content": p.content, "title": p.title} for p in poems
            ]

            recommender = HybridRecommender()
            recommender.fit(poems_data, all_interactions)

            recs = recommender.recommend(user.id, top_k=10)

            exclude_ids = {
                r.poem_id for r in Review.query.filter_by(user_id=user.id).all()
            }
            if current_id:
                exclude_ids.add(current_id)

            for rec in recs:
                if rec["poem_id"] not in exclude_ids:
                    poem = Poem.query.get(rec["poem_id"])
                    if poem:
                        res = poem.to_dict()
                        res["recommend_reason"] = "基于您的偏好推荐"
                        return jsonify(res)
        except Exception as e:
            print(f"Recommend error: {e}")

    query = Poem.query
    if current_id:
        query = query.filter(Poem.id != current_id)

    all_count = query.count()
    if all_count > 0:
        poem_obj = query.offset(random.randrange(all_count)).first()
        if poem_obj:
            res = poem_obj.to_dict()
            res["recommend_reason"] = "随机选取的千古佳作"
            return jsonify(res)

    return jsonify({"error": "Poem list empty"}), 404


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

        total_likes = db.session.query(func.sum(Poem.likes)).scalar() or 0
        total_views = db.session.query(func.sum(Poem.views)).scalar() or 0
        total_shares = db.session.query(func.sum(Poem.shares)).scalar() or 0

        avg_engagement = (
            round((total_likes + total_views + total_shares) / (total_poems * 3), 2)
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
                "totalLikes": total_likes,
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
        # 基于用户评论来确定热门诗歌
        from sqlalchemy import func as sql_func

        # 统计每首诗的评论数量
        review_counts = (
            db.session.query(Review.poem_id, sql_func.count(Review.id).label("count"))
            .group_by(Review.poem_id)
            .subquery()
        )

        # 关联 Poem 获取诗歌信息
        query = (
            db.session.query(Poem, review_counts.c.count)
            .outerjoin(review_counts, Poem.id == review_counts.c.poem_id)
            .order_by(review_counts.c.count.desc())
            .limit(10)
        )

        result = []
        for poem, count in query.all():
            result.append(
                {
                    "id": poem.id,
                    "title": poem.title,
                    "dynasty": poem.dynasty,
                    "author": poem.author,
                    "likes": poem.likes or 0,
                    "reviews": count or 0,
                    "views": poem.views or 0,
                    "shares": poem.shares or 0,
                }
            )

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"获取热门诗歌失败: {str(e)}"}), 500


@app.route("/api/global/theme-distribution")
def get_theme_distribution():
    try:
        theme_counts = {}

        # 只从 reviews 表获取用户评论的主题分布
        # 关联 Review 和 Poem，获取用户评论过的诗歌的主题
        reviews = Review.query.all()

        for review in reviews:
            poem = Poem.query.get(review.poem_id)
            if poem and poem.Bertopic:
                themes = poem.Bertopic.split("-")
                for t in themes:
                    t = t.strip()
                    if t:
                        theme_counts[t] = theme_counts.get(t, 0) + 1

        result = []
        for theme, count in theme_counts.items():
            result.append({"name": theme, "value": count})

        # 如果没有评论数据，返回默认数据
        if not result:
            result = [
                {"name": "山水田园", "value": 30},
                {"name": "思乡情怀", "value": 25},
                {"name": "豪迈边塞", "value": 20},
                {"name": "离别赠答", "value": 15},
                {"name": "咏史怀古", "value": 10},
            ]

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"获取主题分布失败: {str(e)}"}), 500


@app.route("/api/global/dynasty-distribution")
def get_dynasty_distribution():
    try:
        dynasty_stats = {}

        for poem in Poem.query.all():
            dynasty = poem.dynasty or "其他"
            dynasty_stats[dynasty] = dynasty_stats.get(dynasty, 0) + 1

        dynasty_order = [
            "唐",
            "宋",
            "元",
            "明",
            "清",
            "先秦",
            "汉",
            "魏晋",
            "南北朝",
            "其他",
        ]
        result = []
        for dynasty in dynasty_order:
            if dynasty in dynasty_stats:
                result.append({"name": dynasty, "value": dynasty_stats[dynasty]})

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"获取朝代分布失败: {str(e)}"}), 500


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


@app.route("/api/user/<username>/preferences")
def get_user_prefs_api(username):
    user = User.query.filter_by(username=username).first()
    if not user or not user.preference_topics or not user.preference_topics.strip():
        return jsonify(
            {
                "preferences": [
                    {"topic_name": "山水田园", "percentage": 40, "color": "#cf3f35"},
                    {"topic_name": "思乡情怀", "percentage": 35, "color": "#bfa46f"},
                    {"topic_name": "豪迈边塞", "percentage": 25, "color": "#1a1a1a"},
                ]
            }
        )

    try:
        prefs = json.loads(user.preference_topics)
    except (json.JSONDecodeError, TypeError):
        return jsonify(
            {
                "preferences": [
                    {"topic_name": "山水田园", "percentage": 40, "color": "#cf3f35"},
                    {"topic_name": "思乡情怀", "percentage": 35, "color": "#bfa46f"},
                    {"topic_name": "豪迈边塞", "percentage": 25, "color": "#1a1a1a"},
                ]
            }
        )

    formatted = []
    colors = ["#cf3f35", "#bfa46f", "#1a1a1a", "#1b1a8a", "#1b8a1a"]

    for i, p in enumerate(prefs[:5]):
        formatted.append(
            {
                "topic_id": p["topic_id"],
                "topic_name": p.get("keywords", ["通用"])[0]
                if isinstance(p.get("keywords"), list)
                else "主题",
                "percentage": int(p["score"] * 100),
                "color": colors[i % len(colors)],
            }
        )
    return jsonify({"preferences": formatted})


@app.route("/api/global/wordcloud")
def get_global_wordcloud():
    """获取全局词云数据"""
    try:
        # 获取所有诗歌的 Bertopic 主题
        poems = Poem.query.all()
        word_counts = {}

        for poem in poems:
            if poem.Bertopic:
                # 主题格式: "主题1-主题2-主题3"
                topics = poem.Bertopic.split("-")
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
            if poem.Bertopic:
                topics = poem.Bertopic.split("-")
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
            if poem.Bertopic:
                topics = poem.Bertopic.split("-")
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
            return jsonify({"insights": []})

        # 获取用户的评论记录
        reviews = Review.query.filter_by(user_id=user.id).all()

        # 按小时统计
        hour_counts = {}
        for r in reviews:
            if r.created_at:
                hour = r.created_at.hour
                hour_counts[hour] = hour_counts.get(hour, 0) + 1

        # 转换为前端需要的格式
        insights = []
        for hour, count in sorted(hour_counts.items()):
            period = (
                "凌晨"
                if 0 <= hour < 6
                else "上午"
                if 6 <= hour < 12
                else "下午"
                if 12 <= hour < 18
                else "晚上"
            )
            insights.append({"hour": hour, "period": period, "count": count})

        return jsonify({"insights": insights})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/user/<username>/form-stats")
def get_user_form_stats(username):
    """获取用户偏好的诗歌体裁统计"""
    try:
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify([])

        # 获取用户评论过的诗歌
        reviewed_poems = (
            db.session.query(Poem).join(Review).filter(Review.user_id == user.id).all()
        )

        # 统计体裁
        form_counts = {}
        for poem in reviewed_poems:
            genre = poem.genre_type or "其他"
            form_counts[genre] = form_counts.get(genre, 0) + 1

        result = [{"name": k, "value": v} for k, v in form_counts.items()]
        return jsonify(result)
    except Exception as e:
        return jsonify([]), 500


@app.route("/api/user/<username>/recommendations")
def get_user_recommendations(username):
    """获取用户推荐诗歌"""
    try:
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify({"poems": []})

        # 获取推荐诗歌
        poems = Poem.query.limit(10).all()
        result = []
        for p in poems:
            result.append(
                {
                    "id": p.id,
                    "title": p.title,
                    "author": p.author,
                    "content": p.content[:100] + "..."
                    if p.content and len(p.content) > 100
                    else p.content,
                    "reason": "根据您的偏好推荐",
                }
            )

        return jsonify({"poems": result})
    except Exception as e:
        return jsonify({"poems": [], "error": str(e)}), 500


@app.route("/api/user/<username>/poet-topic-sankey")
def get_user_poet_topic_sankey(username):
    """获取诗人-主题桑基图数据"""
    try:
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify({"nodes": [], "links": []})

        # 获取用户评论过的诗歌
        reviewed_poems = (
            db.session.query(Poem).join(Review).filter(Review.user_id == user.id).all()
        )

        # 统计诗人-主题关系
        poet_topics = {}
        for poem in reviewed_poems:
            author = poem.author or "未知作者"
            if poem.Bertopic:
                topics = poem.Bertopic.split("-")
                for topic in topics:
                    topic = topic.strip()
                    if topic:
                        key = (author, topic)
                        poet_topics[key] = poet_topics.get(key, 0) + 1

        # 构建节点和链接
        authors = set()
        topics = set()
        links = []

        for (author, topic), value in poet_topics.items():
            authors.add(author)
            topics.add(topic)
            links.append({"source": author, "target": topic, "value": value})

        nodes = [{"name": a} for a in authors] + [{"name": t} for t in topics]

        return jsonify({"nodes": nodes, "links": links})
    except Exception as e:
        return jsonify({"nodes": [], "links": [], "error": str(e)}), 500
