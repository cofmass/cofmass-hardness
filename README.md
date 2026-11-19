# Hardness Agent（阶段一：基础闭环）

PostgreSQL + pgvector + LangGraph + FastAPI 的最小对话闭环。**本机未安装 PG 时自动降级到 SQLite 脚手架**——可正常对话，配置 `HARDNESS_PG_DSN` 后自动切换为 PG 正式载体。

## 环境

- Python：conda `hardness`（3.11.18），`D:\Anaconda_App\envs\hardness\python.exe`
- Ollama：`deepseek-r1:7b`（对话）+ `nomic-embed-text`（768 维向量）
- 依赖：`pip install -r requirements.txt`

## 启动服务（正式入口）

```powershell
cd D:\Backend_Project\Python\hardness
& 'D:\Anaconda_App\envs\hardness\python.exe' -m uvicorn app:app --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000/docs` 看 Swagger；`/openapi.json` 为前端契约。

## 对话

```powershell
$body = @{user_id='u-1';thread_id='t-1';question='你好，请介绍一下你自己'} | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/v1/chat/invoke' -Method Post -Body $body -ContentType 'application/json'
```

> PowerShell 5.1 控制台可能把中文显示成乱码，这是终端编码问题；用 `python -X utf8` 客户端或 Swagger 查看可正常显示。

## API 契约

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/v1/health` | 健康检查 → `{"status":"ok","version":"1"}` |
| POST | `/api/v1/chat/invoke` | 同步对话；req `{user_id,thread_id,question}` |
| GET | `/api/v1/skills?user_id=` | 读 skill 偏好 |
| PUT | `/api/v1/skills` | 写偏好 `{user_id,preferences?,enabled?}` |
| POST | `/api/v1/docs/ingest` | 灌知识；RAG 按 user_id 过滤 |
| GET | `/api/v1/chat/history?thread_id=&user_id=` | checkpoint 历史 |
| POST | `/api/v1/chat/stream` | 阶段一返回 501，阶段三启用 SSE |

## 切到 PostgreSQL（阶段一正式载体）

1. 安装 PostgreSQL 15+，创建库；
2. 在目标库执行 `db/schema.sql`（含 `CREATE EXTENSION vector` + HNSW/GIN 索引）；
3. 设置环境变量 `HARDNESS_PG_DSN=postgresql://user:pass@host:port/db`；
4. 重启服务；日志出现 `backend=postgres+pgvector`。

验证扩展：`SELECT extversion FROM pg_extension WHERE extname='vector'`。

## CLI（运维）

```powershell
python cli.py init
python cli.py ingest "文本内容" --user u-1
python cli.py skills --user u-1 --lang zh --verbosity concise --skills code-review
python cli.py ask "问题" --user u-1 --thread t-1
python cli.py history --thread t-1
```

## 测试

```powershell
python -m pytest -q
```

无 PG 时 PG 集成测试明确 skip（不伪报通过）；其余契约测试用 fake LLM/Embedding，不依赖外部服务。

## 架构边界

- `graph/` 只依赖 `providers.*` / `storage.*` 抽象，无具体 SDK import（DoD 静态检查通过）；
- 三类记忆物理隔离：checkpoint（saver）/ mem_store（长期偏好）/ docs（RAG 向量）；
- 换 Provider 只改 `providers/factory.py`。
