from app import app, db, Poem

with app.app_context():
    # 测试搜索功能
    test_queries = [
        "酬程延秋夜即事見贈",  # 繁体标题
        "酬程延秋夜即事见赠",  # 简体标题
        "长簟迎风早",  # 简体内容
        "長簟迎風早",  # 繁体内容
        "韩翃",  # 简体作者
        "韓翃"   # 繁体作者
    ]
    
    # 处理简繁中文转换
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
    
    for query in test_queries:
        print(f"\nTesting search for: {query}")
        
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
        )).limit(20).all()
        
        print(f"Found {len(results)} results")
        for p in results:
            print(f'  - {p.title} by {p.author}')
