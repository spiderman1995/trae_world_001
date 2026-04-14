"""
config.py — 从 .env 读取项目配置

优先级：命令行参数 > 环境变量 > .env 文件 > 硬编码默认值
"""
import os

def _load_dotenv():
    """读取项目根目录的 .env 文件到 os.environ（不覆盖已有变量）"""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value

_load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR", "D:/temp/0_tempdata8")
