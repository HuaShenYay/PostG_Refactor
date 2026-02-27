#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版实验 - 只需证明你的系统比传统算法好
优化版本：修复权重和评估问题
"""

import sys
import os
import json
import numpy as np
import random
from collections import defaultdict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置路径
notebook_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(notebook_dir, '..', '..'))
sys.path.insert(0, project_root)
os.chdir(project_root)

print("="*70)
print("诗词推荐系统实验 - 验证系统效果")
print("="*70)

# 加载模块
print("\n[1] 加载模块...")
try:
    from backend.core.content_recommender import ContentBasedRecommender
    from backend.core.collaborative_filter import ItemBasedCFRecommender
    from backend.core.hybrid_strategy import HybridRecommender
    print("  ✓ 所有推荐器加载成功")
except Exception as e:
    print(f"  ✗ 加载失败: {e}")
    sys.exit(1)

# =============================================================================
# 1. 数据加载
# =============================================================================
print("\n[2] 加载真实诗词数据...")

def load_poems(max_poems=300):
    data_path = os.path.join(project_root, 'data', 'chinese-poetry')
    poems = []
    
    try:
        tang_path = os.path.join(data_path, '全唐诗', '唐诗三百首.json')
        if os.path.exists(tang_path):
            with open(tang_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for i, p in enumerate(data[:max_poems//2]):
                    paragraphs = p.get('paragraphs', [])
                    content = ''.join(paragraphs) if paragraphs else p.get('title', '')
                    poems.append({
                        'id': i,
                        'title': p.get('title', f'诗{i}'),
                        'content': content,
                        'author': p.get('author', '未知')
                    })
            print(f"  ✓ 加载唐诗 {len([p for p in poems])} 首")
    except Exception as e:
        print(f"  ! 加载唐诗失败: {e}")
    
    try:
        ci_path = os.path.join(data_path, '宋词', '宋词三百首.json')
        if os.path.exists(ci_path):
            with open(ci_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                offset = len(poems)
                for i, p in enumerate(data[:max_poems//2]):
                    paragraphs = p.get('paragraphs', [])
                    content = ''.join(paragraphs) if paragraphs else p.get('title', '')
                    poems.append({
                        'id': offset + i,
                        'title': p.get('title', f'词{i}'),
                        'content': content,
                        'author': p.get('author', '未知')
                    })
            print(f"  ✓ 加载宋词 {len(poems) - max_poems//2} 首")
    except Exception as e:
        print(f"  ! 加载宋词失败: {e}")
    
    if len(poems) < 50:
        contents = [
            "明月几时有把酒问青天不知天上宫阙今夕是何年",
            "床前明月光疑是地上霜举头望明月低头思故乡",
            "春风又绿江南岸明月何时照我还",
            "大漠孤烟直长河落日圆",
            "会当凌绝顶一览众山小",
            "海内存知己天涯若比邻",
            "落红不是无情物化作春泥更护花",
            "春蚕到死丝方尽蜡炬成灰泪始干",
            "山重水复疑无路柳暗花明又一村",
            "欲穷千里目更上一层楼"
        ]
        for i in range(max_poems):
            poems.append({
                'id': i,
                'title': f'诗词{i}',
                'content': contents[i % len(contents)],
                'author': f'诗人{i%10}'
            })
    
    print(f"  总计: {len(poems)} 首诗词")
    return poems

poems = load_poems(300)

# =============================================================================
# 2. 生成用户交互数据 - 调整评分分布
# =============================================================================
print("\n[3] 生成用户行为数据...")

def generate_interactions(poems, n_users=100, min_ratings=8, max_ratings=20, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    
    interactions = []
    poem_ids = [p['id'] for p in poems]
    
    themes = {
        '思乡': ['月', '乡', '归', '故', '家'],
        '送别': ['别', '离', '酒', '千里'],
        '山水': ['山', '水', '云', '林', '江'],
        '边塞': ['塞', '戈', '马', '沙'],
        '怀古': ['古', '迹', '怀', '往'],
        '爱情': ['情', '爱', '相思'],
    }
    
    for user_id in range(n_users):
        user_themes = random.sample(list(themes.keys()), k=random.randint(2, 4))
        n_ratings = random.randint(min_ratings, max_ratings)
        selected_poems = random.sample(poem_ids, k=min(n_ratings, len(poem_ids)))
        
        for poem_id in selected_poems:
            poem_content = ""
            for p in poems:
                if p['id'] == poem_id:
                    poem_content = p.get('content', '')
                    break
            
            rating = 3.0
            for theme, keywords in themes.items():
                if theme in user_themes:
                    if any(k in poem_content for k in keywords):
                        rating += 0.8
            
            rating += np.random.normal(0, 0.6)
            rating = np.clip(rating, 1.0, 5.0)
            
            interactions.append({
                'user_id': user_id,
                'poem_id': poem_id,
                'rating': round(rating, 1),
                'liked': rating >= 3.5,
                'created_at': datetime.now() - timedelta(days=random.randint(0, 30))
            })
    
    return interactions

interactions = generate_interactions(poems, n_users=100, seed=42)
print(f"  生成 {len(interactions)} 条评分记录")

# 统计评分分布
ratings = [i['rating'] for i in interactions]
print(f"  评分分布: 均值={np.mean(ratings):.2f}, 标准差={np.std(ratings):.2f}")
print(f"  正样本(>=3.5): {sum(1 for r in ratings if r >= 3.5)/len(ratings)*100:.1f}%")

# =============================================================================
# 3. 数据划分
# =============================================================================
print("\n[4] 划分训练集和测试集...")

def split_data(interactions, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    
    user_data = defaultdict(list)
    for inter in interactions:
        user_data[inter['user_id']].append(inter)
    
    train, test = [], []
    
    for user_id, user_inters in user_data.items():
        n = len(user_inters)
        test_size = max(1, int(n * test_ratio))
        
        indices = list(range(n))
        random.shuffle(indices)
        test_indices = set(indices[:test_size])
        
        for i, inter in enumerate(user_inters):
            if i in test_indices:
                test.append(inter)
            else:
                train.append(inter)
    
    return train, test

train_data, test_data = split_data(interactions, test_ratio=0.2, seed=42)
print(f"  训练集: {len(train_data)} 条")
print(f"  测试集: {len(test_data)} 条")

# =============================================================================
# 4. 训练和评估
# =============================================================================
print("\n[5] 训练推荐模型...")

def evaluate_recommender(recommender, train_data, test_data, poems, rec_type, top_k=10, threshold=3.5):
    user_train = defaultdict(list)
    for inter in train_data:
        user_train[inter['user_id']].append(inter)
    
    poem_id_to_idx = {p['id']: i for i, p in enumerate(poems)}
    
    # MAE
    predictions, actuals = [], []
    for inter in test_data:
        user_id = inter['user_id']
        poem_id = inter['poem_id']
        actual = inter['rating']
        
        user_inters = user_train.get(user_id, [])
        if not user_inters:
            continue
        
        try:
            if rec_type == 'cb':
                rated = [poems[poem_id_to_idx[p['poem_id']]] 
                        for p in user_inters if p['poem_id'] in poem_id_to_idx]
                ratings = [p['rating'] for p in user_inters if p['poem_id'] in poem_id_to_idx]
                if rated:
                    profile = recommender.get_user_profile(rated, ratings)
                    if profile is not None and poem_id in poem_id_to_idx:
                        pred = recommender.predict_rating(profile, poem_id_to_idx[poem_id])
                        predictions.append(pred)
                        actuals.append(actual)
            
            elif rec_type in ['item_cf', 'hybrid']:
                pred = recommender.predict_rating(user_inters, poem_id)
                predictions.append(pred)
                actuals.append(actual)
        
        except:
            continue
    
    mae = np.mean(np.abs(np.array(predictions) - np.array(actuals))) if predictions else float('nan')
    
    # 推荐评估
    user_test = defaultdict(list)
    for inter in test_data:
        user_test[inter['user_id']].append(inter)
    
    precisions, recalls, f1s = [], [], []
    
    for user_id, test_items in user_test.items():
        train_items = user_train.get(user_id, [])
        if not train_items:
            continue
        
        relevant = {i['poem_id'] for i in test_items if i['rating'] >= threshold}
        if not relevant:
            continue
        
        exclude = {i['poem_id'] for i in train_items}
        
        try:
            if rec_type == 'cb':
                rated = [poems[poem_id_to_idx[p['poem_id']]] 
                        for p in train_items if p['poem_id'] in poem_id_to_idx]
                ratings = [p['rating'] for p in train_items if p['poem_id'] in poem_id_to_idx]
                profile = recommender.get_user_profile(rated, ratings) if rated else None
                recs = recommender.recommend(profile, exclude, top_k) if profile else []
            
            elif rec_type == 'item_cf':
                recs = recommender.recommend(train_items, exclude, top_k)
            
            elif rec_type == 'hybrid':
                recs = recommender.recommend(user_id, top_k=top_k)
            
            recommended = {r['poem_id'] for r in recs}
            
            tp = len(recommended & relevant)
            fp = len(recommended - relevant)
            fn = len(relevant - recommended)
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
        
        except:
            continue
    
    precision = np.mean(precisions) if precisions else float('nan')
    recall = np.mean(recalls) if recalls else float('nan')
    f1 = np.mean(f1s) if f1s else float('nan')
    
    return {'mae': mae, 'precision': precision, 'recall': recall, 'f1': f1}

results = {}

# Content-Based
print("\n  训练 Content-Based...")
try:
    cb = ContentBasedRecommender()
    cb.fit(poems)
    cb_metrics = evaluate_recommender(cb, train_data, test_data, poems, 'cb', threshold=3.5)
    results['Content-Based'] = cb_metrics
    print(f"    MAE: {cb_metrics['mae']:.4f}, F1: {cb_metrics['f1']:.4f}")
except Exception as e:
    print(f"    失败: {e}")

# Item-CF
print("\n  训练 Item-CF...")
try:
    item_cf = ItemBasedCFRecommender()
    item_cf.fit(train_data, [p['id'] for p in poems])
    item_cf_metrics = evaluate_recommender(item_cf, train_data, test_data, poems, 'item_cf', threshold=3.5)
    results['Item-CF'] = item_cf_metrics
    print(f"    MAE: {item_cf_metrics['mae']:.4f}, F1: {item_cf_metrics['f1']:.4f}")
except Exception as e:
    print(f"    失败: {e}")

# Hybrid
print("\n  训练 Hybrid (你的系统)...")
try:
    hybrid = HybridRecommender()
    hybrid.fit(poems, train_data)
    hybrid_metrics = evaluate_recommender(hybrid, train_data, test_data, poems, 'hybrid', threshold=3.5)
    results['Hybrid'] = hybrid_metrics
    print(f"    MAE: {hybrid_metrics['mae']:.4f}, F1: {hybrid_metrics['f1']:.4f}")
except Exception as e:
    print(f"    失败: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# 5. 结果展示
# =============================================================================
print("\n" + "="*70)
print("实验结果")
print("="*70)

print(f"\n{'算法':<20} {'MAE':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print("-"*70)

for name, metrics in results.items():
    mae_str = f"{metrics['mae']:.4f}" if not np.isnan(metrics['mae']) else "N/A"
    p_str = f"{metrics['precision']:.4f}" if not np.isnan(metrics['precision']) else "N/A"
    r_str = f"{metrics['recall']:.4f}" if not np.isnan(metrics['recall']) else "N/A"
    f1_str = f"{metrics['f1']:.4f}" if not np.isnan(metrics['f1']) else "N/A"
    print(f"{name:<20} {mae_str:<12} {p_str:<12} {r_str:<12} {f1_str:<12}")

print("\n" + "="*70)
print("结论")
print("="*70)

# 找最优
valid_results = {k:v for k,v in results.items() if not np.isnan(v['mae'])}
if valid_results:
    best_mae = min(valid_results.items(), key=lambda x: x[1]['mae'])
    best_f1_results = {k:v for k,v in valid_results.items() if not np.isnan(v['f1'])}
    if best_f1_results:
        best_f1 = max(best_f1_results.items(), key=lambda x: x[1]['f1'])
        print(f"\n✓ MAE 最低: {best_mae[0]} ({best_mae[1]['mae']:.4f})")
        print(f"✓ F1 最高: {best_f1[0]} ({best_f1[1]['f1']:.4f})")

if 'Hybrid' in results:
    print(f"\n【你的系统 vs 其他算法】")
    for name in ['Content-Based', 'Item-CF']:
        if name in results and not np.isnan(results[name]['mae']):
            other_mae = results[name]['mae']
            hybrid_mae = results['Hybrid']['mae']
            
            mae_improve = (other_mae - hybrid_mae) / other_mae * 100 if other_mae > 0 else 0
            
            print(f"  vs {name}:")
            if mae_improve > 0:
                print(f"    ✓ MAE 降低: {mae_improve:.1f}% (预测更准确)")
            else:
                print(f"    ✗ MAE 增加: {-mae_improve:.1f}%")

print("\n" + "="*70)
print("实验完成！")
print("="*70)

# 保存
output_file = 'backend/experiments/final_experiment_results.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n结果已保存至: {output_file}")
