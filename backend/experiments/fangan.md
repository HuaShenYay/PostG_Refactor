实验设计（Minimal Acceptable Version）
1. 实验目的（Objective）

本实验旨在比较三类推荐方法在诗词推荐任务中的性能差异：

基于内容的推荐方法（Content-Based, CB）

基于用户的协同过滤方法（User-based CF, USER-CF）

融合内容与协同信息的混合推荐方法（Hybrid）

重点考察 Hybrid 方法是否在 USER-CF 基础上取得稳定性能提升，并分析其在不同评价指标下的表现。

2. 数据集与任务定义（Dataset & Task）
2.1 数据来源

诗词文本来自唐诗与宋词数据集；

用户–诗词交互数据通过仿真方式生成；

每条交互包含：

用户 ID

诗词 ID

显式评分（1–5 分）

2.2 任务定义

实验同时考虑两类推荐任务：

评分预测任务（Rating Prediction）

目标：预测用户对诗词的评分

评价指标：MAE

Top-K 推荐任务（Top-K Recommendation）

目标：为用户生成长度为 K 的推荐列表

正样本定义：测试集中评分 ≥ 3.5 的诗词

评价指标：Precision@K、Recall@K、F1@K

3. 数据划分策略（Data Split）

采用 基于时间的用户内划分（Time-based Split）：

对每个用户的交互按时间排序；

前 80% 作为训练集；

后 20% 作为测试集；

该划分方式更贴近真实推荐系统的在线预测场景，并避免未来信息泄漏。

4. 对比方法（Compared Methods）
4.1 Content-Based（CB）

使用诗词文本内容构建向量表示；

基于用户历史高评分诗词生成用户兴趣画像；

根据内容相似度进行推荐与评分预测。

4.2 User-based Collaborative Filtering（USER-CF）

基于用户–诗词评分矩阵；

通过用户相似度进行评分预测与推荐；

不使用任何文本或内容信息。

4.3 Hybrid 推荐方法

融合 CB 与 USER-CF 的预测结果；

所有模型在评估阶段使用相同的输入信息：

用户训练集交互

排除已交互诗词

不使用测试集或全局未来信息。

5. 评价指标（Evaluation Metrics）
5.1 MAE（Mean Absolute Error）

用于评估评分预测任务：

𝑀
𝐴
𝐸
=
1
𝑁
∑
𝑖
=
1
𝑁
∣
𝑟
^
𝑖
−
𝑟
𝑖
∣
MAE=
N
1
	​

i=1
∑
N
	​

∣
r
^
i
	​

−r
i
	​

∣

MAE 仅作为 辅助指标，不作为主要推荐性能比较依据。

5.2 Precision@K / Recall@K / F1@K

用于评估 Top-K 推荐性能：

Precision@K：推荐列表中相关诗词的比例

Recall@K：相关诗词被成功推荐的比例

F1@K：Precision 与 Recall 的调和平均

其中，相关诗词定义为测试集中评分 ≥ 3.5 的项目。

注：由于推荐任务中正负样本极度不平衡，Accuracy 指标不具区分性，故未采用。

6. 实验设置（Experimental Setup）

推荐列表长度：K = 10

用户数：固定为 N

实验重复次数：5 次（不同随机种子）

所有结果报告为 平均值 ± 标准差

7. 实验结果与分析（Results & Analysis）

实验结果从以下角度进行分析：

Hybrid 相对于 USER-CF 的性能提升

是否在 Recall@K 和 F1@K 上取得稳定改进；

CB 方法的特点

在基于历史行为的评估框架下，其 Recall 可能被低估；

MAE 指标的辅助作用

用于验证评分预测的数值合理性，而非排序能力。

8. 讨论（Discussion）

需要特别说明的是：

CB 方法更擅长挖掘潜在兴趣；

而基于测试集中“已发生行为”的 Recall 定义，天然更有利于协同过滤方法；

因此，本文主要关注 Hybrid 相对于 USER-CF 的提升幅度，而非 CB 的绝对 Recall 值。

9. 小结（Summary）

通过统一的数据划分、评价指标和信息接口，实验结果表明：

Hybrid 推荐方法在 USER-CF 基础上取得了稳定性能提升；

融合内容信息有助于缓解协同过滤的数据稀疏问题；

实验设计具备可复现性与公平性。