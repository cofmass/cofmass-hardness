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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


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

# ---- 阶段二：存储 / 检索增强 ----
# 存储后端：空表示按 DSN 自动推导（有 DSN→pgstore，无→sql）；sql 为回退开关。
STORE_BACKEND = os.getenv("HARDNESS_STORE_BACKEND", "").strip().lower()
# ACL 上下文下推：关掉则保持阶段一「仅 user_id 请求体过滤」行为。
STORE_FILTER_ENABLED = _env_bool("HARDNESS_STORE_FILTER_ENABLED", True)
# 中文分词实测：Docker PG16 无 zhparser/pgroonga，仅 simple（按字符/空格切分）——
# 混合检索因此标记为「降级」：英文/数字召回有效，中文召回有限但不会比纯向量更差（RRF 融合）。
HYBRID_SEARCH_ENABLED = _env_bool("HARDNESS_HYBRID_SEARCH_ENABLED", True)
HYBRID_TOKENIZER = os.getenv("HARDNESS_HYBRID_TOKENIZER", "simple")  # 定案：simple（G2 实测降级）
HYBRID_RRF_K = int(os.getenv("HARDNESS_HYBRID_RRF_K", "60"))
HNSW_EF_SEARCH = int(os.getenv("HARDNESS_HNSW_EF_SEARCH", "100"))
HNSW_ITERATIVE_SCAN = os.getenv("HARDNESS_HNSW_ITERATIVE_SCAN", "relaxed_order")  # pgvector>=0.8 可设 none 关闭
DEFAULT_TTL_DAYS = int(os.getenv("HARDNESS_DEFAULT_TTL_DAYS", "30"))

# 兼容旧环境：若显式设置了后端开关，storage 门面在 import 时读取 os.environ；
# 这里把规范化后的值写回，保证 config 与 storage 用同一个口径。
if STORE_BACKEND:
    os.environ["HARDNESS_STORE_BACKEND"] = STORE_BACKEND

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
# 阶段四：RLS 启动校验开关（生产必须 true；本地无 PG 的 SQLite 脚手架自动跳过角色检查）。
RLS_ENFORCED = _env_bool("HARDNESS_RLS_ENFORCED", True)
# 阶段四：治理审计日志（TTL 清理 / 向量重建等运维动作追加写）。
AUDIT_LOG_PATH = os.getenv("HARDNESS_AUDIT_LOG", str(DATA_DIR / "governance_audit.log"))

# ---- 阶段三：工具 / 自动记忆 / SSE ----
TOOLS_ENABLED = _env_bool("HARDNESS_TOOLS_ENABLED", False)          # 默认关闭：保留阶段一顺序图回归路径
TOOL_MAX_ITERATIONS = int(os.getenv("HARDNESS_TOOL_MAX_ITERATIONS", "5"))
AUTO_MEMORY_ENABLED = _env_bool("HARDNESS_AUTO_MEMORY_ENABLED", False)  # 默认关闭：按用户灰度
MEMORY_CONFIDENCE_THRESHOLD = float(os.getenv("HARDNESS_MEMORY_CONFIDENCE_THRESHOLD", "0.8"))
MEMORY_OCCURRENCE_THRESHOLD = int(os.getenv("HARDNESS_MEMORY_OCCURRENCE_THRESHOLD", "2"))
# 自动写入白名单：namespace area -> 允许的 key 集合（为空表示不允许自动写）。
# 首位 user/tenant 由 ACL 强注入，模型不可指定；默认只开放 skills.preferences/enabled。
MEMORY_WHITELIST: dict[str, set[str]] = {"skills": {"preferences", "enabled"}}
ASYNC_ENABLED = _env_bool("HARDNESS_ASYNC_ENABLED", True)           # SSE 优先异步；不可用时 to_thread 回退
SSE_PING_INTERVAL = float(os.getenv("HARDNESS_SSE_PING_INTERVAL", "15"))

# ---- 安全：LangGraph 反序列化白名单 ----
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
