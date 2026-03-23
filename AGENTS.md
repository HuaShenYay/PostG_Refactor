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
| `core/hybrid_cf.py` | **核心算法**: 混合协同过滤 |


---

## 注意事项

1. **数据库配置**: 确保 MySQL/PostgreSQL 已启动，配置文件在 `backend/config.py`

2. **依赖安装**: 如果依赖缺失，确保在 conda 环境中执行:
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
│   │   ├── hybrid_cf.py    # 核心算法
│   ├── app.py             # 主服务入口
│   └── run_server.py      # 启动脚本
├── frontend/              # Vue.js 前端
│   ├── src/
│   │   ├── views/         # 页面组件
│   │   └── router.js      # 路由配置
│   └── package.json
└── data/                  # 数据文件
```
