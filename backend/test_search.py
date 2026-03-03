from app import app, db, Poem

with app.app_context():
    # 获取ID为281的诗歌
    poem = Poem.query.get(281)
    print('Poem ID 281:')
    print('Title:', poem.title)
    print('Author:', poem.author)
    print('Content:', poem.content)
    
    # 尝试通过标题搜索
    print('\nSearch by title ("酬程延"):')
    title_results = Poem.query.filter(Poem.title.ilike('%酬程延%')).all()
    print(f'Found {len(title_results)} results')
    for p in title_results:
        print(f'  - {p.title} by {p.author}')
    
    # 尝试通过内容搜索
    print('\nSearch by content ("长簟迎风早"):')
    content_results = Poem.query.filter(Poem.content.ilike('%长簟迎风早%')).all()
    print(f'Found {len(content_results)} results')
    for p in content_results:
        print(f'  - {p.title} by {p.author}')
    
    # 尝试通过作者搜索
    print('\nSearch by author ("韩翃"):')
    author_results = Poem.query.filter(Poem.author.ilike('%韩翃%')).all()
    print(f'Found {len(author_results)} results')
    for p in author_results:
        print(f'  - {p.title} by {p.author}')
