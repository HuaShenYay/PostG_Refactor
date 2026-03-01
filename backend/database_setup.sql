-- =============================================
-- Poetry Recommendation System Database Setup
-- Database: poetry_db
-- =============================================

-- 创建数据库
CREATE DATABASE IF NOT EXISTS poetry_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE poetry_db;

-- =============================================
-- 用户表
-- =============================================
DROP TABLE IF EXISTS `reviews`;
DROP TABLE IF EXISTS `poems`;
DROP TABLE IF EXISTS `users`;

CREATE TABLE `users` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `username` VARCHAR(50) NOT NULL,
  `password_hash` VARCHAR(256) DEFAULT '123456',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
  `preference_topics` TEXT,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_username` (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =============================================
-- 诗词表
-- =============================================
CREATE TABLE `poems` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `title` VARCHAR(100) NOT NULL,
  `author` VARCHAR(50),
  `content` TEXT,
  `dynasty` VARCHAR(20),
  `genre_type` VARCHAR(50),
  `rhythm_name` VARCHAR(50),
  `rhythm_type` VARCHAR(20),
  `views` INT DEFAULT 0,
  `likes` INT DEFAULT 0,
  `shares` INT DEFAULT 0,
  `review_count` INT DEFAULT 0,
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `tonal_summary` TEXT,
  `Bertopic` TEXT,
  `Real_topic` TEXT,
  PRIMARY KEY (`id`),
  KEY `idx_author` (`author`),
  KEY `idx_dynasty` (`dynasty`),
  KEY `idx_genre_type` (`genre_type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =============================================
-- 评论表
-- =============================================
CREATE TABLE `reviews` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `user_id` INT NOT NULL,
  `poem_id` INT NOT NULL,
  `comment` TEXT,
  `topic_names` TEXT,
  `rating` FLOAT DEFAULT 3.0,
  `liked` TINYINT(1) DEFAULT 0,
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_user_id` (`user_id`),
  KEY `idx_poem_id` (`poem_id`),
  FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON DELETE CASCADE,
  FOREIGN KEY (`poem_id`) REFERENCES `poems`(`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =============================================
-- 设置 root 用户密码并授权
-- =============================================
-- 如果需要远程访问，执行以下命令:
-- ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '123456';
-- FLUSH PRIVILEGES;

-- 或者创建专用用户:
-- CREATE USER IF NOT EXISTS 'poetry_user'@'localhost' IDENTIFIED BY 'poetry123';
-- GRANT ALL PRIVILEGES ON poetry_db.* TO 'poetry_user'@'localhost';
-- FLUSH PRIVILEGES;

SELECT 'Database setup completed!' AS status;
