# 系统推荐算法与前后端交互的批判性审查与优化修复方案

## 1. 现状架构速览

- 后端主服务为 Flask，推荐主入口是 `GET /api/recommend_one/<username>`，每次请求都会现场拉取评论数据并重新训练 `HybridRecommender`。这会导致延迟、CPU 抖动和吞吐瓶颈。  
- 推荐策略由三路融合组成：
  - TF-IDF 内容推荐（`ContentBasedRecommender`）
  - Item-CF（`ItemBasedCFRecommender`）
  - BERTopic + 向量 + User-CF/Item-CF 的混合（`BertopicRecommender`）
- 前端 `Home.vue` 的换一首动作会直接请求 `/api/recommend_one`；`PreferenceGuide.vue` 会请求 `/api/topics` 获取主题列表。

---

## 2. 核心问题（按严重程度）

### P0（需要优先修复）

1. **推荐请求路径中“每次请求重训模型”**
   - 当前 `/api/recommend_one` 每次都会 `fit` 三个子模型（包含 BERTopic 分支），请求耗时随数据量显著增长。  
   - 风险：高并发下响应超时；模型重复计算导致资源浪费。

2. **初始化函数存在明显调用错误（死代码风险）**
   - `init_recommender()` 内部使用 `HybridRecommender.fit(poems_data, interactions)`（按类调用实例方法），语义错误，若被启用会抛异常。  
   - 当前该函数并未在主流程调用，属于“潜在炸点 + 误导代码”。

3. **前后端 API 契约不完整：`/api/topics` 缺失**
   - 前端偏好引导页会调用 `/api/topics`，后端无对应路由，用户首次配置偏好将直接失败。

### P1（高价值修复）

4. **用户“liked”信号在写入评论时丢失**
   - 前端评论提交包含 `liked` 字段，但后端创建 `Review` 时未写入 `liked`。  
   - 结果：BERTopic 分支中的 `like_boost` 失效，影响推荐质量。

5. **账号密码明文存储与比较**
   - `User.password_hash` 实际存放明文，`check_password` 直接字符串比较。  
   - 风险：安全合规不达标，任何库泄漏即全量明文暴露。

6. **统计类 API 存在 N+1 查询与 Python 端聚合**
   - 如主题分布、用户画像等处大量 `for review -> query poem` 模式；数据上来后会显著拖慢。

### P2（中期优化）

7. **融合分数归一化与多样性策略较粗糙**
   - `HybridRecommender` 融合时直接按原分数加权，不同子模型分值区间未统一，存在某一路“天然占优”。
   - BERTopic 的 `_diversify` 实际是“取最高分前 N”，未做真实多样性控制。

8. **冷启动策略偏随机，解释与可控性不足**
   - 新用户主要靠随机/热门，且规则散落在多个分支；可解释性和效果稳定性不足。

---

## 3. 可落地优化方案（建议分三期）

## 一期（1~2 周，稳定性优先）

1. **改造推荐服务为“离线训练 + 在线推断”**
   - 新增 `RecommendationService` 单例：
     - 维护模型对象与版本号（`model_version`、`trained_at`）
     - 提供 `refresh_if_needed()`（评论增量超过阈值或定时触发）
   - `/api/recommend_one` 只做：读缓存模型 + 在线过滤，不再重训。

2. **修复 API 契约缺口**
   - 新增 `/api/topics`，直接返回主题字典或主题数组，格式对齐 `PreferenceGuide.vue` 的访问方式。

3. **修复评论写入信号缺失**
   - `add_review` 增加 `liked = data.get("liked", False)`，写入 `Review(liked=...)`。

4. **清理/修复 `init_recommender()`**
   - 若保留：改为实例化后调用（`r = HybridRecommender(); r.fit(...)`）。
   - 若不用：删除并替换为新的服务初始化入口，避免误导。

5. **最小安全加固**
   - 使用 `werkzeug.security` 的 `generate_password_hash` / `check_password_hash`。
   - 兼容老数据：登录时若检测到旧明文格式，成功后自动升级为哈希。

## 二期（2~4 周，效果与性能）

1. **数据层批量化与索引优化**
   - 统一把“用户交互矩阵、主题统计、热门榜”改为 SQL 聚合 + join。
   - 建议索引：
     - `reviews(user_id, created_at)`
     - `reviews(poem_id, created_at)`
     - `poems(dynasty)`、必要时 `poems(author)`

2. **融合策略标准化**
   - 各子模型先归一化到同分布（如 z-score + sigmoid / min-max + clipping）。
   - 权重由“硬编码阈值”升级为可配置（`config.py`），并记录线上分布用于调参。

3. **真正的多样性重排**
   - 用 MMR（Maximal Marginal Relevance）或 xQuAD：在保持相关性的同时抑制同主题堆叠。

## 三期（4~8 周，智能化与可运维）

1. **在线反馈闭环**
   - 新增行为事件：曝光、点击、停留、收藏/liked、评论、跳过。
   - 使用轻量 bandit 或 rerank 学习在线探索。

2. **A/B 与监控看板**
   - 指标：CTR、评论率、次日留存、推荐覆盖率、新颖度、重复率。
   - 发布采用灰度：按用户分桶回滚可控。

3. **可解释推荐原因体系**
   - 从“基于您的偏好推荐”升级为结构化原因：
     - “因您近期偏好【山水田园】+ 与《xx》语义相近 + 同好用户高评分”。

---

## 4. 重点修复任务清单（可直接建 issue）

1. [P0] `/api/recommend_one` 拆分训练与推断，新增模型缓存层。  
2. [P0] 新增 `/api/topics` 并补接口测试。  
3. [P1] `add_review` 落库 `liked` 字段并回归测试。  
4. [P1] 密码哈希化迁移脚本 + 登录兼容升级。  
5. [P1] 高频统计 API 去 N+1，改 SQL 聚合。  
6. [P2] 融合分数标准化 + MMR 重排。  
7. [P2] 增加推荐解释字段（多因子）。

---

## 5. 建议验收标准

- 推荐接口 P95 从“按请求重训”状态下降到可接受阈值（如 < 300ms，仅在线推断路径）。
- 偏好页主题加载成功率 100%（`/api/topics` 可用）。
- `liked` 信号在 DB 可见且对推荐分值有贡献（灰度验证）。
- 登录流程完成哈希迁移后，不再存储新增明文密码。
- 统计页核心接口（热门榜、主题分布）在 10x 数据量下仍稳定。
