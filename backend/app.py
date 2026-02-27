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

class RecommendationService:
    """BERTopic-only recommender service with lightweight refresh control."""

    def __init__(self):
        self.recommender = None
        self.last_review_count = -1
        self.last_trained_at = None

    def _ensure_recommender(self):
        if self.recommender is None:
            from core.bertopic_recommender import BertopicRecommender

            self.recommender = BertopicRecommender()

    @staticmethod
    def _build_interactions():
        return [
            {
                "user_id": r.user_id,
                "poem_id": r.poem_id,
                "rating": r.rating,
                "created_at": r.created_at or datetime.utcnow(),
                "liked": bool(r.liked),
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
        current_review_count = Review.query.count()
        if not force and self.last_review_count == current_review_count:
            return

        self._ensure_recommender()
        poems_data = self._build_poems()
        interactions = self._build_interactions()

        self.recommender.fit(poems_data, interactions)
        self.last_review_count = current_review_count
        self.last_trained_at = datetime.utcnow()


rec_service = RecommendationService()


@app.route("/")
def hello_world():
    return "Poetry Recommendation Engine (BERTopic Only) is Running!"


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
    poems = Poem.query.filter(Poem.Bertopic.isnot(None)).all()
    counter = {}
    for poem in poems:
        for topic in (poem.Bertopic or "").split("-"):
            topic = topic.strip()
            if topic:
                counter[topic] = counter.get(topic, 0) + 1

    sorted_topics = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    if not sorted_topics:
        fallback = [
            "山水", "思乡", "边塞", "离别", "咏史", "田园", "闺怨", "怀古", "节序", "哲理",
            "人生", "家国", "送别", "咏物", "写景"
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
    liked = bool(data.get("liked", False))

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
        liked=liked, 
        comment=comment,
        topic_names=topic_names
    )
    db.session.add(new_review)
    db.session.commit()
    rec_service.refresh_if_needed(force=True)

    return jsonify({"message": "雅评已收录", "status": "success", "topics": topic_names})


@app.route("/api/recommend_one/<username>")
def recommend_one(username):
    """智能推荐入口 - 解决冷启动和循环问题"""
    user = User.query.filter_by(username=username).first()
    current_id = request.args.get("current_id", type=int)
    skip_count = request.args.get("skip_count", 0, type=int)
    
    # 获取用户会话历史（用于避免循环）
    session_key = f"seen_poems_{username}"
    seen_poems = request.args.getlist("seen_ids") or []
    seen_ids = set(int(x) for x in seen_poems if str(x).isdigit())
    if current_id:
        seen_ids.add(current_id)

    if not user:
        # 访客模式：完全随机
        query = Poem.query
        if seen_ids:
            query = query.filter(~Poem.id.in_(seen_ids))
        poem_obj = query.order_by(func.rand()).first()
        if not poem_obj:
            # 如果都看过，重置
            poem_obj = Poem.query.order_by(func.rand()).first()
        if not poem_obj:
            return jsonify({"error": "Poem list empty"}), 404
        res = poem_obj.to_dict()
        res["recommend_reason"] = "访客模式随机推荐"
        return jsonify(res)

    # 获取用户评论历史
    user_reviews = Review.query.filter_by(user_id=user.id).all()
    user_interactions = [
        {
            "user_id": r.user_id,
            "poem_id": r.poem_id,
            "rating": r.rating,
            "created_at": r.created_at or datetime.utcnow(),
            "liked": bool(r.liked),
        }
        for r in user_reviews
    ]
    
    # 构建排除列表（已评论 + 本次会话已看）
    exclude_ids = {r.poem_id for r in user_reviews}
    exclude_ids.update(seen_ids)

    # ========== 策略1: 探索模式（每3次强制探索）==========
    if skip_count > 0 and skip_count % 3 == 0:
        import random
        
        # 策略1a: 获取所有符合条件的诗歌，然后随机选择
        subquery = db.session.query(
            Review.poem_id,
            func.count(Review.id).label('review_count'),
            func.avg(Review.rating).label('avg_rating')
        ).group_by(Review.poem_id).subquery()
        
        explore_candidates = Poem.query.outerjoin(
            subquery, Poem.id == subquery.c.poem_id
        ).filter(
            ~Poem.id.in_(exclude_ids)
        ).filter(
            (subquery.c.review_count < 3) | (subquery.c.review_count.is_(None))
        ).order_by(
            func.coalesce(subquery.c.avg_rating, 4.0).desc()
        ).limit(20).all()
        
        if explore_candidates:
            # 随机选择一首
            explore_poem = random.choice(explore_candidates)
            res = explore_poem.to_dict()
            res["recommend_reason"] = "探索推荐：小众佳作"
            return jsonify(res)
        
        # 策略1b: 推荐从未被评论的诗歌
        reviewed_poem_ids = {r.poem_id for r in Review.query.with_entities(Review.poem_id).all()}
        unseen_candidates = Poem.query.filter(
            ~Poem.id.in_(reviewed_poem_ids),
            ~Poem.id.in_(exclude_ids)
        ).limit(20).all()
        
        if unseen_candidates:
            unseen_poem = random.choice(unseen_candidates)
            res = unseen_poem.to_dict()
            res["recommend_reason"] = "探索推荐：尚未被发现的诗"
            return jsonify(res)

    # ========== 策略2: 冷启动优化（新用户/无评论用户）==========
    if len(user_reviews) == 0:
        import random
        import time
        
        # 新用户：从多样化的热门诗歌中选择
        popular_poems = db.session.query(
            Poem,
            func.count(Review.id).label('review_count')
        ).outerjoin(Review).group_by(Poem.id).order_by(
            func.count(Review.id).desc()
        ).limit(50).all()  # 获取更多候选
        
        if popular_poems:
            # 使用时间戳+skip_count确保每次不同
            random.seed(int(time.time() * 1000) % 10000 + skip_count)
            # 随机选择
            selected = random.choice(popular_poems)
            if selected[0].id not in exclude_ids:
                res = selected[0].to_dict()
                res["recommend_reason"] = "热门推荐"
                return jsonify(res)
        
        # 如果都不行，随机推荐
        fallback = Poem.query.filter(~Poem.id.in_(exclude_ids)).order_by(func.rand()).first()
        if fallback:
            res = fallback.to_dict()
            res["recommend_reason"] = "随机推荐"
            return jsonify(res)

    # ========== 策略3: 基于BERTopic的智能推荐（有评论用户）==========
    try:
        rec_service.refresh_if_needed()
        all_interactions = rec_service._build_interactions()
        
        # 获取候选推荐（扩大候选池到100首）
        recs = rec_service.recommender.recommend(user_interactions, all_interactions, top_k=100)
        
        # 过滤已看过的
        candidates = [rec for rec in recs if rec["poem_id"] not in exclude_ids]
        
        # 如果候选少于20首，补充随机推荐
        if len(candidates) < 20:
            random_poems = Poem.query.filter(
                ~Poem.id.in_(exclude_ids),
                ~Poem.id.in_([r["poem_id"] for r in candidates])
            ).order_by(func.rand()).limit(30).all()
            for p in random_poems:
                candidates.append({"poem_id": p.id, "score": 0.5})
        
        # 随机选择（增加时间和用户ID的随机性）
        if candidates:
            import random
            import time
            # 使用时间戳+用户ID+skip_count作为种子
            random.seed(int(time.time() * 1000) % 10000 + user.id + skip_count)
            
            # 从前20个候选中随机选择（如果候选少于20个则全部）
            pool_size = min(20, len(candidates))
            selected_idx = random.randint(0, pool_size - 1)
            selected = candidates[selected_idx]
            
            poem = Poem.query.get(selected["poem_id"])
            if poem:
                res = poem.to_dict()
                res["recommend_reason"] = "为你推荐"
                return jsonify(res)
                
    except Exception as e:
        print(f"Recommend error: {e}")
        import traceback
        traceback.print_exc()

    # ========== 最终回退：完全随机 ==========
    fallback_query = Poem.query
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
            base_query.with_entities(Review.poem_id, sql_func.count(Review.id).label("count"))
            .group_by(Review.poem_id)
            .all()
        )

        # 构建 poem_id -> count 的映射
        count_map = {poem_id: count for poem_id, count in review_counts}

        # 获取所有诗歌，按评论数排序
        poems = Poem.query.all()
        
        # 按评论数排序
        sorted_poems = sorted(poems, key=lambda p: count_map.get(p.id, 0), reverse=True)[:10]

        result = []
        for poem in sorted_poems:
            result.append(
                {
                    "id": poem.id,
                    "title": poem.title,
                    "dynasty": poem.dynasty,
                    "author": poem.author,
                    "review_count": count_map.get(poem.id, 0),
                    "likes": poem.likes or 0,
                    "views": poem.views or 0,
                    "shares": poem.shares or 0,
                }
            )

        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"获取热门诗歌失败: {str(e)}"}), 500


@app.route("/api/global/theme-distribution")
def get_theme_distribution():
    try:
        theme_counts = {}

        # 从评论表的 topic_names 字段统计主题分布
        reviews = Review.query.filter(Review.topic_names.isnot(None)).all()

        for review in reviews:
            if review.topic_names:
                # 支持多种分隔符
                topic_data = review.topic_names
                if "-" in topic_data:
                    themes = topic_data.split("-")
                elif "," in topic_data:
                    themes = topic_data.split(",")
                elif "|" in topic_data:
                    themes = topic_data.split("|")
                elif " " in topic_data:
                    themes = topic_data.split()
                else:
                    themes = [topic_data]
                
                for t in themes:
                    t = t.strip()
                    if t and t != "未知":
                        theme_counts[t] = theme_counts.get(t, 0) + 1

        result = []
        for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True):
            result.append({"name": theme, "value": count})

        # 如果没有评论数据，返回默认数据
        if not result:
            result = [
                {"name": "意境", "value": 30},
                {"name": "霸气", "value": 25},
                {"name": "恢诡", "value": 20},
                {"name": "氛围", "value": 15},
                {"name": "感受", "value": 10},
            ]

        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
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

        # 中国传统十二时辰
        time_periods = [
            ("子时", 23, 1), ("丑时", 1, 3), ("寅时", 3, 5), ("卯时", 5, 7),
            ("辰时", 7, 9), ("巳时", 9, 11), ("午时", 11, 13), ("未时", 13, 15),
            ("申时", 15, 17), ("酉时", 17, 19), ("戌时", 19, 21), ("亥时", 21, 23)
        ]

        # 合并到时辰
        period_counts = {name: 0 for name, _, _ in time_periods}
        for hour, count in hour_counts.items():
            for name, start, end in time_periods:
                if start <= hour < end or (start > end and (hour >= start or hour < end)):
                    period_counts[name] += count
                    break

        # 转换为前端需要的格式（按传统顺序）
        ordered_periods = ["子时", "丑时", "寅时", "卯时", "辰时", "巳时", "午时", "未时", "申时", "酉时", "戌时", "亥时"]
        insights = []
        for period in ordered_periods:
            count = period_counts.get(period, 0)
            if count > 0:
                insights.append({"time": period, "value": count})

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
