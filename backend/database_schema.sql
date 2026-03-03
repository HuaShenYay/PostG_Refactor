-- ============================================================
-- 诗词推荐系统数据库表结构
-- 基于算法需求精简设计
-- ============================================================

-- 用户表
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password_hash VARCHAR(256) DEFAULT '123456',
    preference_topics TEXT,  -- 冷启动用：用户偏好主题
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 诗歌表
CREATE TABLE IF NOT EXISTS poems (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(100) NOT NULL,
    content TEXT,  -- 算法核心：生成语义向量
    author VARCHAR(50),
    dynasty VARCHAR(20),
    views INT DEFAULT 0,  -- 热门推荐用
    likes INT DEFAULT 0,
    topic_tags TEXT,  -- 冷启动用：主题标签
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_dynasty (dynasty),
    INDEX idx_author (author)
KK|) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
#ZM|
#KR|-- 评分表（用户对诗歌的评分）

 CHARSET=utf-- 评分表（用户对诗歌的评分）
CREATE TABLE IF NOT EXISTS reviews (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    poem_id INT NOT NULL,
    rating FLOAT DEFAULT 3.0,  -- 算法核心：协同过滤
    comment TEXT COMMENT '用户评论内容',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (poem_id) REFERENCES poems(id) ON DELETE CASCADE,
    UNIQUE KEY uk_user_poem (user_id, poem_id),  -- 避免重复评分
    INDEX idx_user (user_id),
    INDEX idx_poem (poem_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================
-- 初始化数据
-- ============================================================

-- 插入测试用户
INSERT INTO users (username, password_hash) VALUES 
('test_user', 'pbkdf2:sha256:100000$test$hash'),
('admin', 'pbkdf2:sha256:100000$admin$hash');

-- 查看表结构
-- DESCRIBE users;
-- DESCRIBE poems;
-- DESCRIBE reviews;
