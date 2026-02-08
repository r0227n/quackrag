use duckdb::Connection;

use crate::config::DuckDbConfig;

/// テーブルスキーマを初期化する
pub fn initialize_schema(conn: &Connection, config: &DuckDbConfig) -> Result<(), duckdb::Error> {
    let dim = config.embedding_dimension;

    conn.execute_batch(&format!(
        "CREATE TABLE IF NOT EXISTS documents (
            id VARCHAR PRIMARY KEY,
            source VARCHAR NOT NULL,
            content VARCHAR NOT NULL,
            chunk_index INTEGER NOT NULL,
            metadata VARCHAR DEFAULT '{{}}'
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            id VARCHAR PRIMARY KEY,
            document_id VARCHAR NOT NULL REFERENCES documents(id),
            embedding FLOAT[{dim}] NOT NULL
        );"
    ))?;

    Ok(())
}

/// VSS 拡張をロードし HNSW インデックスを作成する。
/// 失敗時は warn ログを出力して続行する（線形スキャンにフォールバック）。
pub fn try_initialize_vss(conn: &Connection, config: &DuckDbConfig) {
    if !config.use_vss {
        return;
    }

    if let Err(e) = conn.execute_batch("INSTALL vss; LOAD vss;") {
        tracing::warn!("VSS extension not available, falling back to linear scan: {e}");
        return;
    }

    let metric = config.distance_metric.hnsw_metric();
    let index_sql = format!(
        "CREATE INDEX IF NOT EXISTS embedding_hnsw_idx ON embeddings \
         USING HNSW (embedding) WITH (metric = '{metric}');"
    );

    if let Err(e) = conn.execute_batch(&index_sql) {
        tracing::warn!("Failed to create HNSW index, falling back to linear scan: {e}");
    }
}
