import os
from huggingface_hub import snapshot_download

# 设置环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 模型名称
model_name = "BAAI/bge-small-zh-v1.5"

# 下载路径
download_path = "/home/hsy/PostG_Refactor/backend/data/cache/bge-small-zh-v1.5"

print(f"开始下载模型: {model_name}")
print(f"下载路径: {download_path}")

# 下载模型
try:
    snapshot_download(
        repo_id=model_name,
        local_dir=download_path,
        local_dir_use_symlinks=False
    )
    print("模型下载成功!")
except Exception as e:
    print(f"模型下载失败: {e}")
