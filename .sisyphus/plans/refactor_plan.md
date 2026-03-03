# 诗词推荐系统重构计划

## 备份当前代码

```bash
# 手动备份（建议）
cp -r backend/backend/core backend/core_backup
# 或
zip -r backend_backup.zip backend/
```

---

## 重构目标

1. 精简数据库字段
2. 修复代码Bug
3. 完善推荐系统功能

---

## 待完成任务

### Phase 1: 数据库字段精简

- [ ] 1.1 删除 Review 表无用字段
  - 删除: `comment`, `topic_names`, `liked`, `created_at`, `updated_at`

- [ ] 1.2 删除 Poem 表无用字段
  - 删除: `rhythm_name`, `rhythm_type`, `review_count`, `created_at`, `updated_at`, `tonal_summary`, `Real_topic`

- [ ] 1.3 添加 Poem 表缺失字段（如果有数据源）
  - 添加: `notes`, `author_bio`, `appreciation`

- [ ] 1.4 生成数据库迁移脚本

---

### Phase 2: 算法优化

- [ ] 2.1 修复缓存验证逻辑（更精确的用户/诗歌变化检测）

- [ ] 2.2 优化增量更新机制

- [ ] 2.3 添加预计算推荐结果功能

---

### Phase 3: 扩展功能（可选）

- [ ] 3.1 评论语义分析（基于评论内容的兴趣提取）

- [ ] 3.2 可解释推荐（告诉用户为什么推荐）

- [ ] 3.3 实时推荐（WebSocket/轮询）

---

## 关键文件清单

| 文件 | 改动类型 |
|-----|---------|
| `backend/models.py` | 精简字段 |
| `backend/app.py` | 适配新字段 |
| `backend/core/bertopic_enhanced_cf.py` | 缓存优化 |
| `backend/import_poems.py` | 适配新字段 |

---

## 执行顺序

1. 先备份当前代码
2. 修改 models.py（删除字段）
3. 生成数据库迁移SQL
4. 修改 app.py（适配）
5. 测试运行
6. 如有需要，修改 import_poems.py
