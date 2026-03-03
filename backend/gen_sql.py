import random

random.seed(42)

usernames = [
    "诗仙李白粉丝",
    "杜甫草堂",
    "白居易易",
    "苏轼铁粉",
    "李清照迷",
    "王维粉丝团",
    "陶渊明后人",
    "辛弃疾拥护",
    "柳永词痴",
    "李商隐研究",
    "高考语文140",
    "古文背诵机器",
    "诗词大会选手",
    "中文系小王",
    "文学社社长",
    "清风明月",
    "山水田园",
    "秋水伊人",
    "云中子",
    "空山新雨",
    "江南烟雨",
    "塞外风沙",
    "大漠孤烟",
    "长河落日",
    "东篱采菊",
    "韵律研究者",
    "平仄控",
    "意象分析君",
    "典故达人",
    "诗评家",
    "词牌爱好者",
    "唐诗宋词元曲",
    "诗经楚辞",
    "汉乐府",
    "千古诗心",
]

poem_ids = list(range(1, 978))

user_types = {
    "very_cold_1": 25,
    "very_cold_2": 15,
    "cold": 20,
    "low": 30,
    "active": 15,
    "core": 10,
}

n_ratings_map = {
    "very_cold_1": 1,
    "very_cold_2": 2,
    "cold": 3,
    "low": (4, 10),
    "active": (11, 20),
    "core": (20, 35),
}

used = set()
user_id = 1

with open("generate_reviews.sql", "a", encoding="utf-8") as f:
    for type_name, count in user_types.items():
        for i in range(count):
            while True:
                username = random.choice(usernames)
                if username not in used:
                    used.add(username)
                    break
                else:
                    username = f"{username}_{random.randint(1, 99)}"

            days = random.randint(1, 365)
            f.write(
                f"INSERT INTO users (id, username, password_hash, created_at) VALUES ({user_id}, '{username}', 'pbkdf2:sha256:260000$salt$hash', NOW() - INTERVAL {days} DAY);\n"
            )

            n = n_ratings_map[type_name]
            n = random.randint(n[0], n[1]) if isinstance(n, tuple) else n

            selected = random.sample(poem_ids, min(n, len(poem_ids)))
            for pid in selected:
                rating = random.choices(
                    [5.0, 4.5, 4.0, 3.5, 3.0], weights=[20, 30, 25, 15, 10]
                )[0]
                rdays = random.randint(0, 365)
                f.write(
                    f"INSERT INTO reviews (user_id, poem_id, rating, created_at) VALUES ({user_id}, {pid}, {rating}, NOW() - INTERVAL {rdays} DAY);\n"
                )

            user_id += 1

print(f"生成完成: {user_id - 1} 个用户")
