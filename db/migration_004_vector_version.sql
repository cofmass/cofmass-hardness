"""向量模型版本管理迁移（阶段四 §3.4）。

docs.embedding_model 在阶段一 schema 已存在，检索 SQL 已按该列过滤；
本迁移补充「版本表 + 审计索引」，让灰度切换、重建、清理有可查询依据。
列维度变更（768→1536）的手工步骤见 db/DEPLOYMENT.md §向量模型切换。
"""
-- 版本登记表：记录每种 embedding_model 的维度与启用状态。
CREATE TABLE IF NOT EXISTS embedding_versions (
  model       text PRIMARY KEY,
  dim         integer NOT NULL,
  enabled     boolean NOT NULL DEFAULT true,
  created_at  timestamptz NOT NULL DEFAULT now(),
  notes       text
);

-- 幂等登记当前模型（维度由 config.EMB_DIM 在部署文档中说明；此处只登记不强制）。
INSERT INTO embedding_versions (model, dim, notes)
VALUES ('nomic-embed-text', 768, '阶段一默认模型')
ON CONFLICT (model) DO NOTHING;

-- 审计：按模型统计/清理向量时走该索引，避免全表扫描。
CREATE INDEX IF NOT EXISTS docs_embedding_model_idx ON docs (embedding_model);
