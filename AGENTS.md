# 项目启动指南

## 环境要求

- **后端**: Python 3.11+, Conda
- **前端**: Node.js, Bun
- **数据库**: MySQL/PostgreSQL

---

## 启动方式

### 后端启动

```bash
# 1. 激活 conda 环境
conda activate myenv

# 2. 进入后端目录
cd backend

# 3. 启动 Flask 服务
python run_server.py
```

后端服务默认运行在 `http://localhost:5000`

### 前端启动

```bash
# 1. 进入前端目录
cd frontend

# 2. 使用 bun 启动开发服务器
bun run dev
```

前端服务默认运行在 `http://localhost:5173`

---

## 核心算法模块

| 文件 | 描述 |
|------|------|
| `core/bertopic_enhanced_cf.py` | **核心算法**: BERTopic增强的协同过滤 (0.6×评分 + 0.4×主题) |
| `core/collaborative_filter.py` | 传统Item-CF（对比组） |
| `core/content_recommender.py` | Content-Based推荐（对比组） |
| `core/bertopic_recommender.py` | BERTopic原始实现（已不推荐使用） |

---

## 实验运行

### 方案一：诗词数据集实验（合成数据）

```bash
cd backend
conda activate myenv
python -m experiments.platium_experiment
```

实验结果输出: `backend/experiments/platium_results.json`

### 方案二：MovieLens真实数据集实验（推荐）

```bash
cd backend
conda activate myenv
python -m experiments.movielens_experiment
```

实验结果输出: `backend/experiments/movielens_results.json`

该实验使用MovieLens-100k真实数据集，包含943个用户、1682部电影、10万条评分，是推荐系统领域的标准数据集。

---

## 注意事项

1. **数据库配置**: 确保 MySQL/PostgreSQL 已启动，配置文件在 `backend/config.py`

2. **BERTopic模型**: 首次运行会自动加载/训练BERTopic模型，可能需要较长时间

3. **依赖安装**: 如果依赖缺失，确保在 conda 环境中执行:
   ```bash
   pip install -r backend/requirements.txt
   ```

4. **conda环境名称**: 如环境名不是 `myenv`，请根据实际情况修改激活命令

5. **端口占用**: 如果 5000 或 5173 端口被占用，请自行修改配置

---

## 项目结构

```
PostG_Refactor/
├── backend/               # Flask 后端
│   ├── core/              # 推荐算法核心
│   │   ├── bertopic_enhanced_cf.py  # 核心算法（本项目创新点）
│   │   ├── collaborative_filter.py   # 传统Item-CF
│   │   └── content_recommender.py    # Content-Based
│   ├── experiments/       # 实验代码
│   ├── app.py             # 主服务入口
│   └── run_server.py      # 启动脚本
├── frontend/              # Vue.js 前端
│   ├── src/
│   │   ├── views/         # 页面组件
│   │   └── router.js      # 路由配置
│   └── package.json
└── data/                  # 数据文件
```
