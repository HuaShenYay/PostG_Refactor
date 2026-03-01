#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诗词数据导入脚本
将 chinese-poetry 目录下的 JSON 文件导入到 poems 表
"""

import json
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db, Poem


def load_tang_poems():
    """加载唐诗"""
    file_path = "/home/hsy/PostG_Refactor/data/chinese-poetry/唐诗三百首.json"
    with open(file_path, "r", encoding="utf-8") as f:
        poems_data = json.load(f)

    poems = []
    for item in poems_data:
        # 将段落合并为完整诗句
        content = "".join(item.get("paragraphs", []))

        # 从 tags 提取体裁信息
        tags = item.get("tags", [])
        genre_type = None
        for tag in tags:
            if "诗" in tag:
                genre_type = tag
                break

        poems.append(
            Poem(
                title=item.get("title", ""),
                author=item.get("author", ""),
                content=content,
                dynasty="唐",
                genre_type=genre_type,
            )
        )

    return poems


def load_song_ci():
    """加载宋词"""
    file_path = "/home/hsy/PostG_Refactor/data/chinese-poetry/宋词三百首.json"
    with open(file_path, "r", encoding="utf-8") as f:
        poems_data = json.load(f)

    poems = []
    for item in poems_data:
        content = "".join(item.get("paragraphs", []))

        poems.append(
            Poem(
                title=item.get("title", ""),
                author=item.get("author", ""),
                content=content,
                dynasty="宋",
                rhythm_name=item.get("rhythmic", ""),  # 词牌名
                genre_type="宋词",
            )
        )

    return poems


def load_yuanqu():
    """加载元曲"""
    file_path = "/home/hsy/PostG_Refactor/data/chinese-poetry/yuanqu.json"
    with open(file_path, "r", encoding="utf-8") as f:
        poems_data = json.load(f)

    poems = []
    for item in poems_data:
        content = "".join(item.get("paragraphs", []))

        # 从标题中提取曲牌名
        title = item.get("title", "")
        rhythm_name = ""
        if "・" in title:
            parts = title.split("・")
            rhythm_name = parts[-1] if len(parts) > 1 else ""
            title = parts[0]

        poems.append(
            Poem(
                title=title,
                author=item.get("author", ""),
                content=content,
                dynasty=item.get("dynasty", "元").replace("yuan", "元"),
                rhythm_name=rhythm_name,
                genre_type="元曲",
            )
        )

    return poems


def import_poems():
    """执行导入"""
    with app.app_context():
        # 检查数据库连接
        try:
            db.session.execute(db.text("SELECT 1"))
            print("✓ 数据库连接成功")
        except Exception as e:
            print(f"✗ 数据库连接失败: {e}")
            print("请先确保 MySQL 已启动并初始化数据库")
            return False

        # 检查是否已有数据
        existing_count = Poem.query.count()
        if existing_count > 0:
            print(f"数据库中已有 {existing_count} 首诗词")
            response = input("是否清空现有数据并重新导入? (y/n): ")
            if response.lower() != "y":
                print("导入已取消")
                return False

            # 清空现有数据
            db.session.execute(db.text("DELETE FROM reviews"))
            db.session.execute(db.text("DELETE FROM poems"))
            db.session.commit()
            print("✓ 已清空现有数据")

        # 导入各类诗词
        all_poems = []

        print("\n正在加载唐诗...")
        tang_poems = load_tang_poems()
        all_poems.extend(tang_poems)
        print(f"  加载完成: {len(tang_poems)} 首")

        print("正在加载宋词...")
        song_ci = load_song_ci()
        all_poems.extend(song_ci)
        print(f"  加载完成: {len(song_ci)} 首")

        print("正在加载元曲...")
        yuan_qu = load_yuanqu()
        all_poems.extend(yuan_qu)
        print(f"  加载完成: {len(yuan_qu)} 首")

        # 批量插入
        print(f"\n正在导入 {len(all_poems)} 首诗词到数据库...")

        # 分批插入，每批 500 条
        batch_size = 500
        for i in range(0, len(all_poems), batch_size):
            batch = all_poems[i : i + batch_size]
            db.session.bulk_save_objects(batch)
            db.session.commit()
            print(f"  已导入 {min(i + batch_size, len(all_poems))}/{len(all_poems)}")

        # 验证
        final_count = Poem.query.count()
        print(f"\n✓ 导入完成! 数据库中共 {final_count} 首诗词")

        # 显示统计
        print("\n诗词统计:")
        stats = db.session.execute(
            db.text(
                "SELECT dynasty, genre_type, COUNT(*) as count FROM poems GROUP BY dynasty, genre_type"
            )
        ).fetchall()
        for row in stats:
            print(f"  {row[0]} {row[1] or '未知'}: {row[2]} 首")

        return True


if __name__ == "__main__":
    import_poems()
