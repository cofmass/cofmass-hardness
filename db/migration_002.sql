-- 阶段二迁移：tsvector 生成列 + GIN + mem_store 过期审计列。
-- 幂等；在 PG_DSN 指向的库执行。Docker 容器示例（避免 PowerShell 管道编码问题）：
--   docker cp db/migration_002.sql hardness-pg:/tmp/migration_002.sql
--   docker exec hardness-pg psql -U hardness -d hardness -v ON_ERROR_STOP=1 -f /tmp/migration_002.sql
-- G2 定案（2026-11-19 实测）：PG16.16 + pgvector 0.8.7；无 zhparser/pgroonga 扩展 ->
--   分词配置 = 'simple'（按空格/字符切分，中文召回有限，标记「混合检索降级」）；
--   英文/数字混合检索有效；排序层用 RRF 融合保证召回不劣于纯向量。
-- pgvector 0.8.7 支持迭代索引扫描，运行期由 storage/pg_store.py 在检索前 SET：
--   SET hnsw.iterative_scan = relaxed_order; SET hnsw.ef_search = 100;

-- docs.content_tsv：生成列随 content 自动维护，GIN 加速全文匹配。
ALTER TABLE docs ADD COLUMN IF NOT EXISTS content_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED;
CREATE INDEX IF NOT EXISTS docs_content_tsv_gin ON docs USING gin (content_tsv);

-- 官方 BaseStore.ttl 已足以实现过期；此列仅用于审计/兼容查询，不参与过期判定。
ALTER TABLE mem_store ADD COLUMN IF NOT EXISTS expires_at timestamptz;
CREATE INDEX IF NOT EXISTS mem_store_expires_idx ON mem_store (expires_at)
  WHERE expires_at IS NOT NULL;

-- langgraph 0.6.11 的原生 checkpoints 表不记录墙钟时间（写入 SQL 无时间列），
-- 而 checkpoint TTL 清理脚本需要「按时间」判断。这里补一个由 DB 维护的时间列
-- 与触发器：INSERT 写 now()，同一 checkpoint 被 ON CONFLICT UPDATE 时刷新。
-- 不改 langgraph 写 SQL，仅加列 + 触发器，对其读写完全透明。
ALTER TABLE checkpoints ADD COLUMN IF NOT EXISTS hardness_updated_at timestamptz DEFAULT now();

CREATE OR REPLACE FUNCTION hardness_touch_checkpoint() RETURNS trigger AS $$
BEGIN
  -- 仅在缺省时写入创建时间；checkpoint 是 append-only 语义，
  -- 后续 ON CONFLICT UPDATE 不应刷新 TTL 基准（允许审计/测试手工调整该列）。
  IF NEW.hardness_updated_at IS NULL THEN
    NEW.hardness_updated_at := now();
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS hardness_checkpoint_touch ON checkpoints;
CREATE TRIGGER hardness_checkpoint_touch
  BEFORE INSERT OR UPDATE ON checkpoints
  FOR EACH ROW EXECUTE FUNCTION hardness_touch_checkpoint();

CREATE INDEX IF NOT EXISTS checkpoints_hardness_updated_idx
  ON checkpoints (hardness_updated_at);
