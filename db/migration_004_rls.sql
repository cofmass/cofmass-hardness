-- 阶段四：RLS 数据库层兜底（解决评审 blocking #2）
-- 目标：应用连接使用非超级用户、无 BYPASSRLS 的 app_user；RLS 按 current_setting 过滤。
-- 执行（用具有 CREATEROLE 的管理连接，不要用 app_user 自己执行）：
--   psql "$ADMIN_PG_DSN" -v ON_ERROR_STOP=1 -f db/migration_004_rls.sql
-- 然后配置 HARDNESS_PG_DSN=postgresql://app_user:<密码>@host:5432/hardness
-- 启动检测脚本 scripts/check_app_role.py 会拒绝超级用户/BYPASSRLS 连接。

-- 应用角色：按需改密码，禁止 SUPERUSER / BYPASSRLS / CREATEROLE。
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_user') THEN
    CREATE ROLE app_user LOGIN PASSWORD 'CHANGE_ME_APP_USER';
  END IF;
END
$$;
ALTER ROLE app_user NOSUPERUSER NOBYPASSRLS NOCREATEROLE NOCREATEDB;

-- 基础权限：三载体物理隔离，只授予必要对象。
GRANT USAGE ON SCHEMA public TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON mem_store, docs TO app_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;
-- checkpoints / checkpoint_writes / checkpoint_blobs 由 langgraph saver 读写按需授权。
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema='public' AND table_name='checkpoints') THEN
    GRANT SELECT, INSERT, UPDATE, DELETE ON checkpoints TO app_user;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema='public' AND table_name='checkpoint_writes') THEN
    GRANT SELECT, INSERT, UPDATE, DELETE ON checkpoint_writes TO app_user;
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema='public' AND table_name='checkpoint_blobs') THEN
    GRANT SELECT, INSERT, UPDATE, DELETE ON checkpoint_blobs TO app_user;
  END IF;
END
$$;

-- RLS 策略：与阶段二 ACL 字段对齐（docs.metadata.user_id/tenant_id、mem_store.namespace 首位）。
ALTER TABLE docs ENABLE ROW LEVEL SECURITY;
ALTER TABLE mem_store ENABLE ROW LEVEL SECURITY;
-- 应用角色不绕过 RLS（显式再确认一次）。
ALTER TABLE docs FORCE ROW LEVEL SECURITY;
ALTER TABLE mem_store FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS docs_acl ON docs;
CREATE POLICY docs_acl ON docs
  USING (
    (current_setting('app.user_id', true) IS NULL OR metadata->>'user_id' = current_setting('app.user_id', true))
    AND (current_setting('app.tenant_id', true) IS NULL OR metadata->>'tenant_id' = current_setting('app.tenant_id', true))
  )
  WITH CHECK (
    (current_setting('app.user_id', true) IS NULL OR metadata->>'user_id' = current_setting('app.user_id', true))
    AND (current_setting('app.tenant_id', true) IS NULL OR metadata->>'tenant_id' = current_setting('app.tenant_id', true))
  );

-- namespace 首位锁 user_id；有 tenant 时首位是 tenant_id，因此同时兼容两种前缀。
-- JSON 数组形如 ["u-1",...] 或 ["t-1","u-1",...]。
DROP POLICY IF EXISTS mem_acl ON mem_store;
CREATE POLICY mem_acl ON mem_store
  USING (
    current_setting('app.user_id', true) IS NULL
    OR namespace LIKE '["' || current_setting('app.user_id', true) || '",%'
    OR namespace LIKE '["' || coalesce(current_setting('app.tenant_id', true), '') || '","'
       || current_setting('app.user_id', true) || '",%'
  )
  WITH CHECK (
    current_setting('app.user_id', true) IS NULL
    OR namespace LIKE '["' || current_setting('app.user_id', true) || '",%'
    OR namespace LIKE '["' || coalesce(current_setting('app.tenant_id', true), '') || '","'
       || current_setting('app.user_id', true) || '",%'
  );
