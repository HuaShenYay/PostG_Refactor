# 诗词推荐系统对比实验设计方案

## 文档信息

- 版本：v1.0
- 日期：2026年2月
- 目的：验证混合推荐系统相对于传统User-CF、Item-CF和CB方法的优势

---

## 1 实验背景与目标

### 1.1 实验背景

本系统是一个混合推荐系统（Hybrid Recommender），整合了三种核心推荐策略：

**基于内容的推荐（CB）**：使用TF-IDF提取诗歌内容特征，通过余弦相似度进行推荐。该方法通过分析诗歌的文本内容（标题、作者、题材、风格等）构建物品特征向量，然后根据用户历史偏好的物品特征构建用户画像，最后推荐与用户画像相似的物品。CB方法的主要优势在于可以解决新物品的冷启动问题，因为物品的内容特征是独立于用户交互行为的。

**基于物品的协同过滤（Item-CF）**：基于用户评分矩阵计算物品之间的相似度。Item-CF的核心思想是"用户喜欢某个物品，那么他也可能喜欢与该物品相似的其他物品"。该方法首先构建用户-物品评分矩阵，然后计算物品之间的相似度（常用余弦相似度或皮尔逊相关系数），最后根据用户历史评分物品的相似物品进行推荐。Item-CF的优势在于推荐结果具有较好的稳定性和可解释性。

**BERTopic混合推荐**：使用BERTopic进行语义向量化，结合User-CF、Item-CF、内容推荐和热门推荐。该方法使用预训练的多语言语义向量模型（paraphrase-multilingual-MiniLM-L12-v2）将诗歌内容转换为高维语义向量，然后融合多种推荐策略，并根据用户交互数量动态调整权重。具体权重策略为：冷启动用户（0次交互）权重为CB:0.3、Item-CF:0.2、BERTopic:0.5；低活跃用户（1-9次交互）权重为CB:0.3、Item-CF:0.3、BERTopic:0.4；高活跃用户（10次以上交互）权重为CB:0.2、Item-CF:0.3、BERTopic:0.5。

传统推荐系统通常采用单一的推荐策略，如User-CF、Item-CF或CB。这些方法各有优缺点：User-CF能够发现潜在兴趣和推荐新颖内容，但冷启动问题严重且受限于数据稀疏性；Item-CF具有较好的稳定性和可解释性，但难以发现用户的潜在兴趣；CB能够解决冷启动问题，但推荐多样性较差，容易陷入信息茧房。混合系统旨在综合多种策略的优势，弥补单一方法的不足。

### 1.2 实验目标

本实验旨在通过科学严谨的评估方法，证明混合推荐系统相对于传统User-CF、Item-CF和CB方法的核心优势。具体的实验目标包括以下几个方面。

第一个目标是验证推荐准确性提升。通过Precision@K、Recall@K、MRR（平均倒数排名）和NDCG（归一化折损累计增益）等指标，证明混合系统在推荐准确性方面的优势。

第二个目标是验证冷启动性能改善。针对不同交互历史长度的用户群体（零交互、1-5次、6-10次、10次以上），分别评估各方法的推荐质量，验证混合系统在新用户或低活跃用户场景下的优势。

第三个目标是验证推荐多样性提升。通过Intra-list Diversity（列表内多样性）和Catalog Coverage（目录覆盖率）等指标，证明混合系统能够提供更加多样化的推荐结果。

第四个目标是验证系统鲁棒性。通过在不同数据稀疏程度下（高密度、中密度、低密度）的表现，验证混合系统的稳定性和适应性。

第五个目标是分析各组件贡献。通过消融实验，分析混合系统中各推荐组件（CB、Item-CF、BERTopic）对最终性能的贡献度。

---

## 2 实验数据集与评估指标

### 2.1 数据集选择

推荐系统实验需要使用具有代表性的数据集进行验证。本实验采用以下数据集策略。

**实际交互数据**：使用本系统积累的用户-诗歌交互数据，包括浏览、点赞、评分等行为。数据按照时间顺序划分为训练集和测试集，比例建议为8:2。这种划分方式能够避免"未来信息泄露"问题，即测试集中的交互不会发生在训练集之前。

**公开标准数据集（可选）**：如果需要与学术界其他研究进行对比，可考虑使用MovieLens数据集（MovieLens 1M或MovieLens Latest）或Book-Crossing数据集。这些数据集包含大规模的用户-物品交互评分数据，适合进行推荐系统算法的对比实验。

**冷启动测试子集**：为专门验证冷启动性能，构建特殊测试子集。具体方法是从完整数据集中随机选取10%-20%的用户，将其所有交互记录仅保留在训练集中，测试集中不包含这些用户的任何历史交互，从而模拟真实的冷启动场景。

### 2.2 数据集划分策略

为确保实验结果的可靠性和可复现性，采用以下几种划分方法。

**时序划分（Temporal Split）**：按照时间顺序将数据划分为训练集和测试集。将数据按时间排序，取前80%作为训练集，后20%作为测试集。这种方法更符合实际应用场景。

**留出法（Hold-out Split）**：随机将数据划分为训练集、验证集和测试集，比例建议为60%:20%:20%。验证集用于调参，测试集用于最终评估。由于可能导致数据分布不均，需要进行多次随机划分并取平均值。

**K折交叉验证（K-Fold Cross Validation）**：将数据划分为K个大小相等的子集，每次使用K-1个子集进行训练，剩余一个子集进行测试，重复K次并取平均值。K的取值通常为5或10。K折交叉验证能够充分利用数据，减小划分带来的误差。

### 2.3 评估指标体系

推荐系统评估需要从多个维度进行，本实验采用的评估指标分为以下几类。

**准确性指标类**包括以下指标。Precision@K（精确率@K）衡量推荐列表中用户真正感兴趣物品所占的比例，计算公式为Precision@K = |推荐列表 ∩ 测试集| / K，K通常取5、10、20、50等值。Recall@K（召回率@K）衡量推荐列表覆盖用户真正感兴趣物品的比例，计算公式为Recall@K = |推荐列表 ∩ 测试集| / |测试集|。MRR（平均倒数排名）衡量推荐列表中第一个相关物品排名的倒数，取值范围为0到1，值越大表示推荐效果越好。NDCG（归一化折损累计增益）综合考虑物品的相关性和位置信息，是推荐系统评估中最常用的指标之一。RMSE（均方根误差）和MAE（平均绝对误差）用于评估评分预测的准确性。

**多样性指标类**包括以下指标。Intra-list Diversity（列表内多样性）衡量推荐列表中物品之间的相似度，计算公式为列表中所有物品对相似度的平均值，相似度越低表示多样性越高。Catalog Coverage（目录覆盖率）衡量推荐系统能够覆盖的物品比例，计算公式为至少被推荐一次的物品数量除以总物品数量。Gini系数是衡量推荐分布均匀程度的指标，值越低表示推荐分布越均匀。

**冷启动指标类**包括以下指标。新用户推荐成功率衡量在给定新用户特征的情况下，系统能够产生有效推荐的比例。预测准确率衰减衡量从有交互历史的用户到无交互历史的用户，推荐准确率下降的幅度。

**其他指标类**包括以下指标。覆盖率（Coverage）衡量推荐系统能够为多少比例的用户提供推荐。响应时间（Response Time）衡量系统生成推荐结果所需的时间。

---

## 3 对比系统设计

### 3.1 实验组设置

为全面验证系统优势，设计以下对比实验组。

**实验组一：混合推荐系统（Hybrid）**：待评估系统，采用CB、Item-CF和BERTopic的动态权重融合策略。

**实验组二：传统User-CF**：实现标准的基于用户的协同过滤算法，使用余弦相似度计算用户之间的相似度，选择最相似的K个用户，根据他们的评分预测目标用户的偏好。User-CF的核心思想是"相似用户喜欢的东西也可能是目标用户喜欢的"。

**实验组三：传统Item-CF**：基于物品相似度的协同过滤方法，使用评分矩阵计算物品之间的相似度，根据用户历史评分物品的相似物品进行推荐。Item-CF的核心思想是"用户喜欢某个物品，那么他也可能喜欢与该物品相似的其他物品"。

**实验组四：传统CB**：基于TF-IDF的Content-Based推荐，通过分析物品的内容特征构建用户画像，推荐与用户画像相似的物品。

**实验组五：简化混合系统变体**：为深入分析各组件贡献，设计以下变体。Hybrid-CB表示仅使用CB策略。Hybrid-ItemCF表示仅使用Item-CF策略。Hybrid-Only-BERTopic表示仅使用BERTopic策略（无CB和Item-CF）。

### 3.2 参数设置

为确保对比实验的公平性，需要对各系统的关键参数进行统一设置。

**相似度计算参数**：User-CF和Item-CF中统一使用余弦相似度。邻居数量K统一测试，建议测试K=10、20、30、50等值。

**CB参数**：TF-IDF特征维度设置为1000。N-gram范围设置为(1,2)，即使用unigram和bigram。最小文档频率min_df设置为1。

**混合系统权重参数**：根据用户交互数量动态调整权重。冷启动用户（0次交互）：CB权重0.3、Item-CF权重0.2、BERTopic权重0.5。低活跃用户（1-9次交互）：CB权重0.3、Item-CF权重0.3、BERTopic权重0.4。高活跃用户（10次以上交互）：CB权重0.2、Item-CF权重0.3、BERTopic权重0.5。

**BERTopic参数**：Embedding模型使用paraphrase-multilingual-MiniLM-L12-v2。语义向量维度为384维。

---

## 4 实验流程

### 4.1 数据预处理阶段

**步骤一：数据清洗**。处理缺失值、异常值和重复记录。具体操作包括：删除用户ID或物品ID为空的记录；删除评分或交互时间缺失的记录；处理异常评分（如超出1-5范围的评分）；删除同一用户对同一物品的重复交互记录，保留最后一次交互。

**步骤二：数据统计分析**。统计总用户数、总物品数、总交互数。计算用户平均交互数量、物品平均被交互次数。分析评分的分布情况。计算数据稀疏度（Sparsity），公式为：Sparsity = 1 - 总交互数 / (用户数 × 物品数)。

**步骤三：数据划分**。按照时序划分策略，将数据划分为训练集（80%）和测试集（20%）。确保测试集中的用户在训练集中有至少一定数量的交互（非冷启动场景）。

**步骤四：生成测试queries**。生成用于测试的queries，每个query包含一个用户ID和该用户在训练集中的交互历史。测试queries需要覆盖不同的用户群体。

### 4.2 模型训练与评估阶段

**步骤一：训练各对比系统**。使用训练集数据分别训练User-CF、Item-CF、CB和Hybrid系统。

**步骤二：参数调优**。使用验证集对各系统的关键参数进行调优，使用网格搜索或随机搜索进行参数优化。

**步骤三：为测试用户生成推荐**。使用训练好的模型为测试集中的每个用户生成推荐列表，推荐列表长度K设置多个值（K=5、10、20、50）。

**步骤四：计算评估指标**。使用测试集的真实交互数据，计算各推荐系统在各项评估指标上的表现。

---

## 5 冷启动实验设计

### 5.1 冷启动场景定义

冷启动问题是推荐系统中最重要的挑战之一，混合系统的一个重要优势是能够通过CB和BERTopic语义向量来缓解冷启动问题。冷启动场景主要分为三类：新用户冷启动（新注册用户没有任何交互历史）、新物品冷启动（新加入系统的物品没有被任何用户交互过）和系统冷启动（整个系统刚刚上线）。本实验重点关注新用户冷启动。

### 5.2 冷启动实验方法

将测试用户按照其在训练集中的交互历史长度分为以下几组。零交互组（0次交互）模拟新用户。极低活跃组（1-2次交互）模拟刚刚开始使用系统的用户。低活跃组（3-5次交互）模拟偶尔使用系统的用户。中活跃组（6-10次交互）模拟一般活跃用户。高活跃组（10次以上交互）模拟活跃用户。

对每个用户组，分别计算各推荐系统在Precision@K、Recall@K、MRR和NDCG等指标上的表现。重点比较各系统在零交互组和极低活跃组上的表现。

### 5.3 冷启动性能指标

设计专门的冷启动性能指标来量化系统优势。冷启动推荐成功率定义为系统能够为冷启动用户生成有效推荐的比例。冷启动准确率定义为冷启动用户的推荐列表中至少包含一个测试集中物品的比例。

通过对比实验，量化Hybrid系统在冷启动场景下相对于各单一方法的提升幅度。计算公式为：提升幅度 = (Hybrid指标值 - 最优单一方法指标值) / 最优单一方法指标值 × 100%。

---

## 6 多样性实验设计

### 6.1 多样性评估方法

**Intra-list Diversity（列表内多样性）**：计算推荐列表中物品之间的相似度平均值。相似度可以使用内容相似度（基于TF-IDF或语义向量）或协同过滤相似度。计算公式为：ILD = (1 / (K × (K-1))) × Σ_{i≠j} similarity(item_i, item_j)，其中K为推荐列表长度。ILD值越低，表示推荐列表越多样化。

**Catalog Coverage（目录覆盖率）**：计算被推荐过的物品占所有物品的比例。分别计算不同K值下的覆盖率。覆盖率越高，表示系统能够推荐更多不同的物品。

**类别分布分析**：如果物品有类别或主题标签，可以分析推荐列表中类别分布的均匀程度。使用熵或基尼系数来量化分布均匀性。

**新颖性（Novelty）**：衡量推荐物品对于用户的新颖程度，即用户之前没有见过或不知道的物品比例。

### 6.2 多样性实验流程

首先选择合适的相似度计算方法，建议同时使用内容相似度和协同过滤相似度进行评估。然后对每个测试用户，分别计算各系统推荐列表的多样性指标。接着按照用户的活跃程度进行分组，对比分析各系统在不同用户群体上的多样性表现。最后选取典型案例，可视化展示各系统推荐列表的差异。

---

## 7 鲁棒性实验设计

### 7.1 数据稀疏性实验

通过人工控制数据的稀疏程度，构建多个不同稀疏度的数据子集。具体方法是从完整数据集中随机抽取不同比例的交互记录，构建稀疏度为10%、20%、50%的数据集。在每个稀疏度下，分别训练和评估各推荐系统。

数据稀疏度（Sparsity）定义为：Sparsity = 1 - 实际交互数 / (用户数 × 物品数)。使用Precision@K和Recall@K作为主要评估指标，观察随着数据稀疏度增加，各系统性能下降的曲线。

### 7.2 噪声数据实验

在干净的数据集中注入不同比例的噪声数据，模拟真实环境中可能存在的恶意评分或随机评分行为。噪声数据注入比例设置为5%、10%、20%、30%。噪声类型包括随机噪声（对物品随机评分）和敌对噪声（故意给出与真实偏好相反的评分）。

对比各系统在有噪声和没有噪声数据下的表现差异，计算性能下降幅度，评估各系统的抗噪能力。

---

## 8 统计分析

### 8.1 统计检验方法

**描述性统计**：对各评估指标进行描述性统计，包括均值、标准差、中位数、四分位数等。

**假设检验**：进行统计假设检验，判断各系统之间的差异是否具有统计显著性。原假设H0为两种方法的评估指标没有显著差异，备择假设H1为两种方法的评估指标存在显著差异。显著性水平α设置为0.05。

对于配对样本，使用配对t检验或Wilcoxon符号秩检验。Wilcoxon检验不要求数据服从正态分布，更适合推荐系统评估。

**效应量计算**：计算Cohen's d或Cliff's Delta来量化差异的实际大小。Cohen's d的计算公式为：(方法1均值 - 方法2均值) / pooled_std。效应量解读标准：d<0.2为微小效应，0.2-0.5为小效应，0.5-0.8为中等效应，>0.8为大效应。

### 8.2 可视化方法

**柱状图对比**：使用柱状图展示各系统在各项评估指标上的表现。

**箱线图展示**：使用箱线图展示各系统评估指标的分布情况，显示数据的中位数、四分位数、异常值等信息。

**雷达图对比**：使用雷达图展示各系统在不同维度上的综合表现。

**热力图展示**：使用热力图展示不同参数设置下各系统的表现。

**学习曲线**：使用学习曲线展示系统性能随训练数据量增加的变化趋势。

---

## 9 预期结果与分析

### 9.1 准确性结果预期

**整体准确性**：Hybrid系统在Precision@K、Recall@K、MRR和NDCG等准确性指标上应该全面优于单一的User-CF、Item-CF和CB方法。

**K值影响**：随着K值增大，Recall@K会逐渐提高，但Precision@K可能会下降。在较小的K值（如K=5）时，Hybrid系统优势可能更加明显。

### 9.2 冷启动结果预期

**新用户场景（0次交互）**：Hybrid系统应该显著优于User-CF，因为User-CF无法为没有任何交互历史的用户找到相似用户。Hybrid通过CB和BERTopic中的内容推荐和热门推荐，能够为新用户提供有效的推荐。

**低活跃用户（1-5次交互）**：Hybrid系统应该优于所有单一方法。CB能够利用用户仅有的几次交互构建粗略的用户画像，BERTopic的语义向量能够捕捉更深层的兴趣偏好。

### 9.3 多样性结果预期

**列表内多样性**：Hybrid系统应该提供更多样化的推荐列表。混合系统融合了不同策略的推荐结果，不同策略倾向于推荐不同类型的物品，从而自然地增加了多样性。

**目录覆盖率**：Hybrid系统应该覆盖更多的物品种类。单一方法往往倾向于推荐热门或相似的物品，而混合系统能够发现更多长尾物品。

### 9.4 鲁棒性结果预期

**数据稀疏性**：Hybrid系统在各个稀疏度级别下都应该表现稳定。在高稀疏度时，Hybrid系统的性能下降幅度应该小于单一方法。

---

## 10 实验结论框架

基于上述实验设计，预期可以得出以下核心结论。

**结论一**：混合系统在推荐准确性方面具有显著优势。通过多指标验证，Hybrid系统在整体准确性上全面优于传统User-CF、Item-CF和CB方法，且差异具有统计显著性。

**结论二**：混合系统在冷启动场景下具有明显优势。系统能够有效解决新用户冷启动问题，在零交互和低交互用户场景下表现显著优于单一协同过滤方法。

**结论三**：混合系统能够提供更多样化的推荐。通过Intra-list Diversity和Catalog Coverage等指标的验证，证明混合系统能够为用户提供更加多样化的推荐结果。

**结论四**：混合系统具有更强的鲁棒性。在数据稀疏和存在噪声的情况下，混合系统的性能下降幅度更小，表现出更强的稳定性和适应性。

**结论五**：动态权重策略有效性。通过对比分析不同权重设置下的系统表现，验证根据用户交互数量动态调整混合权重的策略是有效的。

---

## 附录：评估指标计算代码

```python
import numpy as np

class RecommendationMetrics:
    @staticmethod
    def precision_at_k(recommendations, test_items, k):
        """计算Precision@K"""
        if not test_items:
            return 0.0
        recommended = set(r['poem_id'] for r in recommendations[:k])
        relevant = set(test_items)
        return len(recommended & relevant) / k
    
    @staticmethod
    def recall_at_k(recommendations, test_items, k):
        """计算Recall@K"""
        if not test_items:
            return 0.0
        recommended = set(r['poem_id'] for r in recommendations[:k])
        relevant = set(test_items)
        return len(recommended & relevant) / len(relevant) if relevant else 0.0
    
    @staticmethod
    def mrr(recommendations, test_items):
        """计算平均倒数排名（MRR）"""
        if not test_items:
            return 0.0
        relevant = set(test_items)
        for i, rec in enumerate(recommendations, 1):
            if rec['poem_id'] in relevant:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def ndcg(recommendations, test_items, k):
        """计算NDCG@K"""
        if not test_items:
            return 0.0
        relevant = set(test_items)
        dcg = 0.0
        for i, rec in enumerate(recommendations[:k], 1):
            if rec['poem_id'] in relevant:
                dcg += 1.0 / np.log2(i + 1)
        
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def intra_list_diversity(recommendations, similarity_matrix, poem_id_map):
        """计算列表内多样性"""
        if len(recommendations) < 2:
            return 0.0
        
        total_similarity = 0.0
        count = 0
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                pid_i = recommendations[i]['poem_id']
                pid_j = recommendations[j]['poem_id']
                if pid_i in poem_id_map and pid_j in poem_id_map:
                    idx_i = poem_id_map[pid_i]
                    idx_j = poem_id_map[pid_j]
                    total_similarity += similarity_matrix[idx_i, idx_j]
                    count += 1
        
        return total_similarity / count if count > 0 else 0.0
```

---

## 附录：实验评估框架代码

```python
import numpy as np
from collections import defaultdict

class RecommendationEvaluator:
    def __init__(self, systems_dict):
        """
        systems_dict: 字典，键为系统名称，值为推荐系统对象
        """
        self.systems = systems_dict
    
    def evaluate(self, test_users, train_data, test_data, k_values=[5, 10, 20]):
        """
        对所有推荐系统进行评估
        
        参数:
            test_users: 测试用户列表
            train_data: 训练数据，字典格式 {user_id: [(poem_id, rating), ...]}
            test_data: 测试数据，字典格式 {user_id: {poem_id: rating, ...}}
            k_values: K值列表
        """
        results = defaultdict(lambda: defaultdict(list))
        
        for user_id in test_users:
            user_train_interactions = train_data.get(user_id, [])
            
            if not user_train_interactions:
                user_test_items = list(test_data.get(user_id, {}).keys())
                for method_name in self.systems.keys():
                    for k in k_values:
                        results[method_name]['precision@k'].append(0.0)
                        results[method_name]['recall@k'].append(0.0)
                        results[method_name]['mrr'].append(0.0)
                        results[method_name]['ndcg'].append(0.0)
                continue
            
            user_test_items = list(test_data.get(user_id, {}).keys())
            
            for method_name, system in self.systems.items():
                try:
                    recommendations = system.recommend(user_train_interactions, top_k=max(k_values))
                except Exception as e:
                    print(f"Error for {method_name}: {e}")
                    recommendations = []
                
                for k in k_values:
                    prec_k = RecommendationMetrics.precision_at_k(
                        recommendations, user_test_items, k
                    )
                    rec_k = RecommendationMetrics.recall_at_k(
                        recommendations, user_test_items, k
                    )
                    mrr = RecommendationMetrics.mrr(recommendations, user_test_items)
                    ndcg = RecommendationMetrics.ndcg(recommendations, user_test_items, k)
                    
                    results[method_name]['precision@k'].append(prec_k)
                    results[method_name]['recall@k'].append(rec_k)
                    results[method_name]['mrr'].append(mrr)
                    results[method_name]['ndcg'].append(ndcg)
        
        return self._aggregate_results(results)
    
    def _aggregate_results(self, results):
        """汇总结果"""
        summary = {}
        for method, metrics in results.items():
            summary[method] = {}
            for metric_name, values in metrics.items():
                if values:
                    summary[method][metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'median': np.median(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        return summary
    
    def print_results(self, results):
        """打印评估结果"""
        print("\n" + "="*80)
        print("推荐系统评估结果")
        print("="*80)
        
        for method, metrics in results.items():
            print(f"\n【{method}】")
            for metric_name, stats in metrics.items():
                print(f"  {metric_name}: {stats['mean']:.4f} (±{stats['std']:.4f})")
```

---

*文档结束*
