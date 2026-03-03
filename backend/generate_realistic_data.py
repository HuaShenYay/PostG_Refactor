#!/usr/bin/env python3
import random
import json
from datetime import datetime, timedelta

random.seed(42)

# ========== 真实用户名库 ==========
LITERARY_USERS = [
    "清风明月",
    "诗意人生",
    "书香门第",
    "静水流深",
    "云淡风轻",
    "春花秋月",
    "落霞孤鹜",
    "秋水长天",
    "孤帆远影",
    "寒江雪",
    "东篱采菊",
    "南山种豆",
    "西窗剪烛",
    "北窗高卧",
    "桃李春风",
    "江湖夜雨",
    "十年灯火",
    "一蓑烟雨",
    "青衫湿遍",
    "独立斜阳",
    "月落乌啼",
    "江枫渔火",
    "夜半钟声",
    "客船吹笛",
    "单车欲问",
    "大漠孤烟",
    "长河落日",
    "羌笛杨柳",
    "春风难度",
    "玉门关外",
    "采薇采薇",
    "在水一方",
    "蒹葭苍苍",
    "白露为霜",
    "所谓伊人",
    "青青子衿",
    "悠悠我心",
    "纵我不往",
    "子宁不嗣音",
    "挑兮达兮",
    "高山流水",
    "知音难觅",
    "伯牙子期",
    "千古知音",
    "弦断谁听",
]

ORDINARY_USERS = [
    "小张",
    "明明",
    "大伟",
    "阿强",
    "小李",
    "老王",
    "阿明",
    "小赵",
    "阿杰",
    "小陈",
    "老孙",
    "阿波",
    "小周",
    "阿华",
    "阿民",
    "阿东",
    "小刘",
    "阿超",
    "老陈",
    "阿伟",
    "小杨",
    "阿军",
    "大鹏",
    "阿磊",
    "阿亮",
    "小马",
    "阿飞",
    "阿凯",
    "阿龙",
    "阿晨",
    "阿晖",
    "阿涛",
]

LITERARY_YOUTH = [
    "北岛",
    "顾城",
    "海子",
    "舒婷",
    "食指",
    "芒克",
    "多多",
    "韩东",
    "于坚",
    "翟永明",
    "西川",
    "欧阳江河",
    "臧棣",
    "王家新",
    "孙文波",
    "张曙光",
    "萧开愚",
    "黄灿然",
    "陈东东",
    "李亚伟",
    "万夏",
    "石光华",
    "宋炜",
    "吉木狼格",
    "杨黎",
    "何小竹",
    "韩少功",
    "李钢",
    "柏桦",
    "张枣",
    "钟鸣",
    "庞培",
    "陈云虎",
    "马松",
    "京不特",
    "祝凤鸣",
]

POET_NAMES = [
    "李白",
    "杜甫",
    "白居易",
    "王维",
    "孟浩然",
    "王昌龄",
    "岑参",
    "高适",
    "李商隐",
    "杜牧",
    "刘禹锡",
    "韩愈",
    "柳宗元",
    "孟郊",
    "贾岛",
    "姚合",
    "苏轼",
    "辛弃疾",
    "李清照",
    "柳永",
    "周邦彦",
    "秦观",
    "贺铸",
    "晏几道",
    "欧阳修",
    "范仲淹",
    "王安石",
    "晏殊",
    "张先",
    "晁补之",
    "晁冲之",
    "毛滂",
    "陶渊明",
    "谢灵运",
    "鲍照",
    "庾信",
    "阴铿",
    "何逊",
    "王绩",
    "卢照邻",
    "骆宾王",
    "杨炯",
    "陈子昂",
    "张九龄",
    "王湾",
    "常建",
    "刘长卿",
    "韦应物",
    "元稹",
    "张籍",
    "王建",
    "温庭筠",
    "韦庄",
    "冯延巳",
    "李璟",
    "李煜",
]

# ========== 真实评论库 ==========
PRAISE_COMMENTS = [
    "这首诗写得真是太棒了，特别是最后两句，意境深远，读来令人回味无穷",
    "太喜欢这首诗了，每次读都有新的感受，诗人果然功底深厚",
    "千古名句不是盖的，这种胸襟气魄让人敬佩",
    "意境绝美，文字凝练，读罢久久不能忘怀",
    "这首诗把感情表达得恰到好处，不矫情不做作，真好",
    "拜服!能写出这样的作品，诗人一定是经历了太多",
    "每年秋天我都会想起这首诗真的是经典中的经典",
    "第一次读到就被深深吸引，后来又反复读了无数遍",
    "最后那句简直神来之笔，点亮了整首诗",
    "文字优美，情感真挚，这才是真正的诗",
    "读出了人生百态，世事沧桑，感人至深",
    "诗人这功底没话说，每一个字都恰到好处",
    "这首诗陪伴我度过了很多艰难的时刻，感谢诗人",
    "格律严谨，用词考究，典型的大家风范",
    "意象鲜明，画面感很强，读诗如同看画",
    "越品越有味道，这可能就是经典的魅力吧",
    "写得太好了，让我这个不懂诗的人都能感受到美",
    "堪称教科书级别的作品，后人很难超越了",
    "诗人的才情从字里行间溢出来，挡都挡不住",
    "每次读到都热血沸腾，这才是诗词的魅力",
    "语言简练但意蕴深厚，佩服佩服",
    "这首诗真的写到心里去了，说出了我想说却说不出的",
    "读完久久不能平静，情绪完全被调动起来了",
    "意境开阔，胸怀宽广，让人读了心旷神怡",
    "名不虚传!终于理解为什么这首诗流传了这么多年",
]

MEDIUM_COMMENTS = [
    "写得还行，但感觉没有传说中那么神",
    "中规中矩吧，可能是我期望太高了",
    "前面几句挺有意思，后面就一般了",
    "总体还行，但总觉得差了点什么",
    "有点名过其实了，可能是我欣赏不来",
    "还不错，适合静下心来慢慢品",
    "挺有画面感的，但情感表达不够深入",
    "作为入门级的诗词可以，深入研究就一般了",
    "这首诗有几句特别出彩，整体也不错",
    "无功无过吧，可能是我欣赏水平有限",
    "还行，只是没有特别惊艳的感觉",
    "感觉诗人想表达很多，但有些地方没表达清楚",
    "部分段落很精彩，整体略逊一筹",
    "可以看得出作者有一定功底，但还有提升空间",
    "读到中间部分有些走神，整体能打个及格分",
    "不算差但也谈不上多好吧，普通水平",
    "有点虎头蛇尾的感觉，前面很好后面一般",
    "可能需要多读几遍才能体会其中的妙处",
    "有几句我特别喜欢，但整首诗感觉一般",
    "打发时间看看还行，细究起来经不起推敲",
]

CRITIQUE_COMMENTS = [
    "真的欣赏不来，感觉都是在堆砌辞藻",
    "太长了，读了一半就读不下去",
    "这个题材被写烂了，没什么新意",
    "语法都不通，是不是写错了?",
    "空洞无物，不知道想表达什么",
    "名不副实吧，还没有现在一些网络诗写得好",
    "读起来很拗口，韵律有问题",
    "完全get不到点在哪里，可能是我水平不够",
    "感觉诗人自己在故作高深",
    "水分太多，如果精简一半可能还好点",
    "这首诗真的有那么好吗?我表示怀疑",
    "堆砌意象，无病呻吟，现在看来很一般",
    "写的什么玩意儿?完全看不懂",
    "可能是我欣赏水平达不到这个层次",
    "感觉有点强行煽情，反而让人不舒服",
    "全是大话空话，没有一点真情实感",
    "读起来味同嚼蜡，不知道好在哪里",
    "恕我直言，这首诗被高估了",
    "文字功底是有的，但内容太虚",
    "完全没有共鸣，可能是时代背景差异太大",
]

CONFUSED_COMMENTS = [
    "看不太懂想表达什么",
    "读完一脸问号",
    "作者到底想说什么?",
    "感觉作者心情很复杂的样子",
    "可能需要多读几遍或者了解一下创作背景",
    "有点抽象，不是很明白",
    "这首诗的意象我不太理解",
    "读完有些迷茫，不知道好在哪里",
    "需要查一下资料才能懂",
    "是不是有什么隐喻?反正我是没看出来",
    "似懂非懂，感觉意境到了但内容模糊",
    "说不清道不明的一种感觉",
    "可能是我理解能力有问题吧",
    "诗人想表达的和我理解的可能不一样",
    "这首诗有点超纲了",
]

SHORT_COMMENTS = [
    "不错!",
    "喜欢!",
    "经典!",
    "写得真好",
    "太美了",
    "感动",
    "很好",
    "喜欢这句",
    "绝绝子",
    "太棒了",
    "经典之作",
    "佩服",
    "真好",
    "不错不错",
    "赞",
    "美",
    "喜欢这首",
    "超级棒",
    "给力",
    "泪目",
    "震撼",
    "意境好",
    "写得好",
    "收藏了",
    "中规中矩",
    "一般般",
    "还行",
    "可以",
    "普通",
    "不太行",
    "欣赏不来",
    "无感",
    "没感觉",
    "看不懂",
]

POEM_TITLES = [
    "静夜思",
    "春晓",
    "登鹳雀楼",
    "相思",
    "登幽州台歌",
    "将进酒",
    "出塞",
    "鹿柴",
    "竹里馆",
    "送元二使安西",
    "九月九日忆山东兄弟",
    "黄鹤楼送孟浩然之广陵",
    "早发白帝城",
    "望庐山瀑布",
    "赠汪伦",
    "咏鹅",
    "悯农",
    "江雪",
    "游子吟",
    "回乡偶书",
    "绝句",
    "望岳",
    "春望",
    "石壕吏",
    "茅屋为秋风所破歌",
    "闻官军收河南河北",
    "枫桥夜泊",
    "滁州西涧",
    "暮江吟",
    "赋得古原草送别",
    "琵琶行",
    "长恨歌",
    "无题",
    "锦瑟",
    "夜雨寄北",
    "嫦娥",
    "商山早行",
    "题破山寺后禅院",
    "终南别业",
    "山居秋暝",
    "鸟鸣涧",
    "送别",
]

POEM_LINES = [
    "床前明月光",
    "疑是地上霜",
    "举头望明月",
    "低头思故乡",
    "春眠不觉晓",
    "处处闻啼鸟",
    "夜来风雨声",
    "花落知多少",
    "白日依山尽",
    "黄河入海流",
    "欲穷千里目",
    "更上一层楼",
    "红豆生南国",
    "春来发几枝",
    "愿君多采撷",
    "此物最相思",
    "前不见古人",
    "后不见来者",
    "念天地之悠悠",
    "独怆然而涕下",
    "君不见黄河之水天上来",
    "奔流到海不复回",
    "君不见高堂明镜悲白发",
    "朝如青丝暮成雪",
    "秦时明月汉时关",
    "万里长征人未还",
    "但使龙城飞将在",
    "不教胡马度阴山",
    "空山不见人",
    "但闻人语响",
    "返景入深林",
    "复照青苔上",
    "独坐幽篁里",
    "弹琴复长啸",
    "深林人不知",
    "明月来相照",
    "渭城朝雨浥轻尘",
    "客舍青青柳色新",
    "劝君更尽一杯酒",
    "西出阳关无故人",
    "独在异乡为异客",
    "每逢佳节倍思亲",
    "遥知兄弟登高处",
    "遍插茱萸少一人",
    "故人西辞黄鹤楼",
    "烟花三月下扬州",
    "孤帆远影碧空尽",
    "唯见长江天际流",
    "朝辞白帝彩云间",
    "千里江陵一日还",
    "两岸猿声啼不住",
    "轻舟已过万重山",
    "日照香炉生紫烟",
    "遥看瀑布挂前川",
    "飞流直下三千尺",
    "疑是银河落九天",
    "李白乘舟将欲行",
    "忽闻涛声踏歌声",
    "桃花潭水深千尺",
    "不及汪伦送我情",
    "千山鸟飞绝",
    "万径人踪灭",
    "孤舟蓑笠翁",
    "独钓寒江雪",
    "慈母手中线",
    "游子身上衣",
    "临行密密缝",
    "意恐迟迟归",
    "谁言寸草心",
    "报得三春晖",
    "少小离家老大回",
    "乡音无改鬓毛衰",
    "儿童相见不相识",
    "笑问客从何处来",
    "国破山河在",
    "城春草木深",
    "感时花溅泪",
    "恨别鸟惊心",
    "烽火连三月",
    "家书抵万金",
    "白头搔更短",
    "浑欲不胜簪",
]

# ========== 用户类型定义 ==========
USER_TYPES = {
    "cold_start": {
        "count": 30,
        "rating_range": (1, 4),
        "rating_distribution": [15, 25, 35, 25, 0],  # 1-5分比例(冷启动用户不给5分)
    },
    "low_active": {
        "count": 25,
        "rating_range": (5, 14),
        "rating_distribution": [10, 20, 40, 25, 5],  # 1-5分比例
    },
    "medium_active": {
        "count": 20,
        "rating_range": (15, 29),
        "rating_distribution": [8, 15, 35, 30, 12],
    },
    "high_active": {
        "count": 15,
        "rating_range": (30, 50),
        "rating_distribution": [10, 15, 25, 30, 20],
    },
}

# 评论类型分布
COMMENT_TYPE_WEIGHTS = {
    "cold_start": [20, 30, 10, 40],  # 赞美/中评/批评/困惑
    "low_active": [30, 35, 15, 20],
    "medium_active": [35, 30, 20, 15],
    "high_active": [40, 25, 20, 15],
}


def get_realistic_rating(user_type):
    dist = USER_TYPES[user_type]["rating_distribution"]
    weights = [w / sum(dist) for w in dist]
    return random.choices([1, 2, 3, 4, 5], weights=weights)[0]


def generate_comment(rating, user_type):
    weights = COMMENT_TYPE_WEIGHTS[user_type]
    comment_type = random.choices(
        ["praise", "medium", "critique", "confused"], weights=weights
    )[0]

    if rating >= 4:
        comment_type = "praise"
    elif rating <= 2:
        comment_type = random.choice(["critique", "confused"])
    else:
        comment_type = random.choice(["medium", "confused"])

    length_type = random.choices(["short", "medium", "long"], weights=[30, 50, 20])[0]

    if length_type == "short":
        base = random.choice(SHORT_COMMENTS)
        if random.random() > 0.5:
            line = random.choice(POEM_LINES)
            return f"{base}，特别是'{line}'这句"
        return base

    if comment_type == "praise":
        pool = PRAISE_COMMENTS
    elif comment_type == "medium":
        pool = MEDIUM_COMMENTS
    elif comment_type == "critique":
        pool = CRITIQUE_COMMENTS
    else:
        pool = CONFUSED_COMMENTS

    comment = random.choice(pool)

    if random.random() > 0.7:
        line = random.choice(POEM_LINES)
        prefixes = [
            f"读到'{line}'这句",
            f"关于'{line}'",
            f"' {line}'这个意象",
            f"尤其是'{line}'",
        ]
        comment = f"{random.choice(prefixes)}，{comment[2:] if comment.startswith('这首') else comment}"

    if random.random() > 0.6:
        suffixes = [
            "，个人很喜欢",
            "，推荐阅读",
            "，值得细细品味",
            "，每次读都有新感受",
            "，确实名不虚传",
        ]
        if not any(s in comment for s in suffixes):
            comment += random.choice(suffixes)

    if random.random() > 0.7:
        title = random.choice(POEM_TITLES)
        comment = f"《{title}》{comment}"

    return comment


def generate_username(used_names):
    source = random.choice([LITERARY_USERS, ORDINARY_USERS, LITERARY_YOUTH, POET_NAMES])

    if source == ORDINARY_USERS:
        name = random.choice(source)
        if name in used_names:
            suffix = random.randint(1, 99)
            name = f"{name}{suffix}"
    elif source == POET_NAMES:
        name = random.choice(source)
        if name in used_names:
            suffixes = ["爱好者", "粉丝", "研究", "的铁粉", "的诗迷", "粉丝团"]
            name = f"{name}{random.choice(suffixes)}"
    else:
        name = random.choice(source)
        if name in used_names:
            suffixes = ["斋", "居", "室", "园", "阁", "轩", "庐", "堂"]
            name = f"{name}{random.choice(suffixes)}"

    return name


def generate_users_and_reviews(poem_count, output_format="json"):
    users = []
    reviews = []
    user_id = 1
    review_id = 1
    used_names = set()

    for user_type, config in USER_TYPES.items():
        for i in range(config["count"]):
            username = generate_username(used_names)
            used_names.add(username)

            created_at = datetime.now() - timedelta(days=random.randint(30, 500))

            user = {
                "id": user_id,
                "username": username,
                "password_hash": "pbkdf2:sha256:260000$random$salt",
                "preference_topics": ",".join(
                    random.sample(
                        [
                            "思乡",
                            "送别",
                            "山水",
                            "田园",
                            "边塞",
                            "咏史",
                            "咏物",
                            "闺怨",
                            "怀古",
                            "战争",
                        ],
                        k=random.randint(2, 5),
                    )
                ),
                "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
            users.append(user)

            n_ratings = random.randint(
                config["rating_range"][0], config["rating_range"][1]
            )
            poem_ids = random.sample(
                range(1, poem_count + 1), min(n_ratings, poem_count)
            )

            for poem_id in poem_ids:
                rating = get_realistic_rating(user_type)

                days_after_user = random.randint(
                    0, min(180, (datetime.now() - created_at).days)
                )
                review_date = created_at + timedelta(days=days_after_user)

                comment = generate_comment(rating, user_type)

                review = {
                    "id": review_id,
                    "user_id": user_id,
                    "poem_id": poem_id,
                    "rating": float(rating),
                    "comment": comment,
                    "created_at": review_date.strftime("%Y-%m-%d %H:%M:%S"),
                }
                reviews.append(review)
                review_id += 1

            user_id += 1

    return users, reviews


def generate_sql(users, reviews):
    sql_statements = []

    sql_statements.append("-- 生成用户数据")
    for user in users:
        sql_statements.append(
            f"INSERT INTO users (id, username, password_hash, preference_topics, created_at) "
            f"VALUES ({user['id']}, '{user['username']}', '{user['password_hash']}', "
            f"'{user['preference_topics']}', '{user['created_at']}');"
        )

    sql_statements.append("\n-- 生成评论数据")
    for review in reviews:
        escaped_comment = review["comment"].replace("'", "''")
        sql_statements.append(
            f"INSERT INTO reviews (id, user_id, poem_id, rating, comment, created_at) "
            f"VALUES ({review['id']}, {review['user_id']}, {review['poem_id']}, "
            f"{review['rating']}, '{escaped_comment}', '{review['created_at']}');"
        )

    return "\n".join(sql_statements)


def main():
    import sys
    import os

    poem_count = 977

    print(f"开始生成诗歌评论数据...")
    print(f"诗词总数: {poem_count}")
    print(f"用户类型分布:")
    for ut, cfg in USER_TYPES.items():
        print(
            f"  - {ut}: {cfg['count']} 用户, 评分 {cfg['rating_range'][0]}-{cfg['rating_range'][1]} 条"
        )

    users, reviews = generate_users_and_reviews(poem_count)

    total_ratings = sum(
        random.randint(cfg["rating_range"][0], cfg["rating_range"][1])
        for cfg in USER_TYPES.values()
    )

    print(f"\n生成完成!")
    print(f"用户总数: {len(users)}")
    print(f"评论总数: {len(reviews)}")

    output_dir = "/home/hsy/PostG_Refactor/backend"
    os.makedirs(output_dir, exist_ok=True)

    json_output = {
        "users": users,
        "reviews": reviews,
        "summary": {
            "total_users": len(users),
            "total_reviews": len(reviews),
            "user_types": {ut: cfg["count"] for ut, cfg in USER_TYPES.items()},
        },
    }

    json_path = os.path.join(output_dir, "realistic_reviews.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    print(f"\nJSON数据已保存到: {json_path}")

    sql_output = generate_sql(users, reviews)
    sql_path = os.path.join(output_dir, "realistic_reviews.sql")
    with open(sql_path, "w", encoding="utf-8") as f:
        f.write(sql_output)
    print(f"SQL语句已保存到: {sql_path}")

    rating_dist = {}
    for r in reviews:
        rt = int(r["rating"])
        rating_dist[rt] = rating_dist.get(rt, 0) + 1

    print(f"\n评分分布:")
    for rt in sorted(rating_dist.keys()):
        pct = rating_dist[rt] / len(reviews) * 100
        print(f"  {rt}分: {rating_dist[rt]} 条 ({pct:.1f}%)")

    comment_lengths = [len(r["comment"]) for r in reviews]
    print(f"\n评论长度统计:")
    print(f"  最短: {min(comment_lengths)} 字")
    print(f"  最长: {max(comment_lengths)} 字")
    print(f"  平均: {sum(comment_lengths) / len(comment_lengths):.1f} 字")


if __name__ == "__main__":
    main()
