# gold_experiment 审查（对照 EXPERIMENT_DESIGN）

## 总体判断
`gold_experiment.py` **部分符合**设计文档，但尚未达到“系统算法设计”要求的完整性。主要问题集中在：

1. **接口不一致导致 Hybrid MAE 评估逻辑错误风险**。
2. **核心指标覆盖不全**（缺少 MRR、覆盖率/多样性指标）。
3. **冷启动评估未按设计分桶输出**。
4. **统计严谨性不足**（缺少显著性检验，仅做均值/std）。

## 关键问题（批判性）

### 1) Hybrid 评分接口调用不一致
- `HybridWrapper.predict_rating(self, user_interactions, poem_id)` 内部调用 `self.hybrid.predict_rating(user_interactions, poem_id)`。
- 但 `backend/core/hybrid_strategy.py` 的 `HybridRecommender.predict_rating` 语义是 `(user_id, poem_id, method='hybrid')`。
- 这意味着在 Gold 中，`user_interactions` 列表会被当作 `user_id`，导致预测语义偏离设计意图，可能让 MAE 结果失真。

### 2) CB 推荐逻辑重复且判定不一致
- Gold 中存在双重赋值：先根据 `profile_valid` 推荐，再根据 `if profile` 再次推荐。
- Numpy 向量的真假值判断易产生歧义或异常，且逻辑重复会降低可读性与可维护性。

### 3) 指标体系未覆盖设计关键项
- 设计文档要求的准确率体系除了 Precision/Recall/NDCG，还强调 MRR。
- 多样性方面要求至少 ILD/Coverage，Gold 缺失目录覆盖率统计。
- 冷启动实验要求按 0、1-2、3-5、6-10、10+ 分组对比，Gold 没有输出分桶结果。

### 4) 统计分析层次不足
- Gold 仅输出多 seed 均值/标准差。
- 未提供显著性检验（如 Wilcoxon / 配对 t 检验）与效应量，导致“显著优于”结论证据不足。

## Platinum(platium)版本修复方向
新脚本 `platium_experiment.py` 已实施以下修复：

1. 使用实验侧 `PlatinumHybrid` 统一接口，避免 core Hybrid 接口耦合错误。
2. 指标补齐：新增 `MRR@K`、`Catalog Coverage`。
3. 增加冷启动分桶统计（0、1-2、3-5、6-10、10+）。
4. 明确训练/测试分离，仅用训练集拟合模型。
5. BERTopic 不可用时自动降级并输出 warning，保证实验可运行。

> 备注：若要完全对齐 `EXPERIMENT_DESIGN`，下一步建议补上统计显著性检验与多样性 ILD。
