# 本地 Agent 最小闭环配置（阶段一）
# 通过环境变量覆盖；PG DSN 存在时走 PostgreSQL + pgvector，否则自动降级 SQLite 脚手架。
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def _load_dotenv(path: Path) -> None:
    """极简 .env 加载：ONLY 填入尚未设置的环境变量（真实环境/IDE 配置优先）。
    不依赖 python-dotenv；支持 KEY=VALUE、可选引号与行内 # 注释。
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if not key or key in os.environ:
            continue
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        else:
            val = val.split(" #", 1)[0].strip()
        os.environ[key] = val


_load_dotenv(BASE_DIR / ".env")

# ---- 数据库 ----
# 阶段一正式载体为 PostgreSQL + pgvector；本机未装 PG 时留空即降级 SQLite（脚手架，不计入阶段一 DoD）。
PG_DSN = os.getenv("HARDNESS_PG_DSN", "").strip()
SQLITE_PATH = os.getenv("HARDNESS_SQLITE", str(DATA_DIR / "agent.sqlite3"))
DSN = PG_DSN or SQLITE_PATH

def use_postgres() -> bool:
    return bool(PG_DSN)

# ---- Provider ----
# ollama list 实测：deepseek-r1:7b + nomic-embed-text（768 维）
CHAT_MODEL = os.getenv("HARDNESS_CHAT_MODEL", "deepseek-r1:7b")
EMB_MODEL = os.getenv("HARDNESS_EMB_MODEL", "nomic-embed-text")
EMB_DIM = int(os.getenv("HARDNESS_EMB_DIM", "768"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
PROVIDER = os.getenv("HARDNESS_PROVIDER", "ollama")

# ---- RAG 检索 ----
RAG_TOP_K = int(os.getenv("HARDNESS_RAG_TOP_K", "3"))

# ---- HTTP API ----
API_HOST = os.getenv("HARDNESS_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("HARDNESS_API_PORT", "8000"))
API_VERSION = "1"
API_CORS_ORIGINS = [
    o.strip() for o in os.getenv("HARDNESS_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if o.strip()
]

# ---- 治理占位（阶段三/四启用）----
MAX_CONTEXT_TOKENS = int(os.getenv("HARDNESS_MAX_CONTEXT_TOKENS", "4096"))

# ---- 安全：LangGraph 反序列化白名单 ----
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
