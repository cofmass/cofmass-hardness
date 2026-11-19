-- 阶段一 PostgreSQL + pgvector schema（在 PG_DSN 指向的库执行）。
-- 验证关卡：
--   psql "$PG_DSN" -f db/schema.sql
--   psql "$PG_DSN" -c "SELECT extversion FROM pg_extension WHERE extname='vector'"
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS mem_store (
  namespace  text        NOT NULL,   -- JSON 数组字符串 '["u-1","skills"]'
  key        text        NOT NULL,
  value      jsonb       NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (namespace, key)
);
CREATE INDEX IF NOT EXISTS mem_store_value_gin ON mem_store USING gin (value);

CREATE TABLE IF NOT EXISTS docs (
  id              bigserial PRIMARY KEY,
  content         text        NOT NULL,
  metadata        jsonb       NOT NULL DEFAULT '{}',
  embedding       vector(768) NOT NULL,
  embedding_model text        NOT NULL DEFAULT 'nomic-embed-text',
  source          text,
  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS docs_embedding_hnsw ON docs USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS docs_metadata_gin   ON docs USING gin (metadata);
