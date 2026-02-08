use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use duckdb::Connection;

use quackrag_core::error::{QuackragError, Result};
use quackrag_core::traits::store::VectorStore;
use quackrag_core::types::document::Document;
use quackrag_core::types::search::SearchResult;

use crate::config::{DuckDbConfig, StorageMode};

use super::schema;
use super::sql;

pub struct DuckDbStore {
    conn: Arc<Mutex<Connection>>,
    config: DuckDbConfig,
}

impl DuckDbStore {
    /// 設定から DuckDbStore を構築する
    pub fn new(config: DuckDbConfig) -> Result<Self> {
        let conn = match &config.storage_mode {
            StorageMode::InMemory => Connection::open_in_memory(),
            StorageMode::File { path } => Connection::open(path),
        }
        .map_err(|e| QuackragError::Store(format!("Failed to open DuckDB: {e}")))?;

        schema::initialize_schema(&conn, &config)
            .map_err(|e| QuackragError::Store(format!("Failed to initialize schema: {e}")))?;

        schema::try_initialize_vss(&conn, &config);

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            config,
        })
    }
}

#[async_trait]
impl VectorStore for DuckDbStore {
    async fn insert(&self, document: &Document, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.config.embedding_dimension {
            return Err(QuackragError::Store(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.config.embedding_dimension,
                embedding.len()
            )));
        }

        let conn = Arc::clone(&self.conn);
        let doc_id = document.id.clone();
        let source = document.source.clone();
        let content = document.content.clone();
        let chunk_index = document.chunk_index;
        let metadata = serde_json::to_string(&document.metadata)
            .map_err(|e| QuackragError::Store(format!("Failed to serialize metadata: {e}")))?;
        let embedding = embedding.to_vec();
        let dim = self.config.embedding_dimension;

        tokio::task::spawn_blocking(move || {
            let mut conn = conn
                .lock()
                .map_err(|e| QuackragError::Store(format!("Failed to lock connection: {e}")))?;

            // トランザクション開始
            let tx = conn
                .transaction()
                .map_err(|e| QuackragError::Store(format!("Failed to begin transaction: {e}")))?;

            // ドキュメント挿入
            tx.execute(
                sql::build_insert_document_sql(),
                duckdb::params![doc_id, source, content, chunk_index as i32, metadata],
            )
            .map_err(|e| QuackragError::Store(format!("Failed to insert document: {e}")))?;

            // 埋め込みベクトル挿入
            let embedding_literal = sql::format_embedding_literal(&embedding, dim);
            let embedding_id = uuid::Uuid::new_v4().to_string();
            let insert_sql = sql::build_insert_embedding_sql(&embedding_literal);
            tx.execute(&insert_sql, duckdb::params![embedding_id, doc_id])
                .map_err(|e| QuackragError::Store(format!("Failed to insert embedding: {e}")))?;

            // コミット
            tx.commit()
                .map_err(|e| QuackragError::Store(format!("Failed to commit transaction: {e}")))?;

            Ok(())
        })
        .await
        .map_err(|e| QuackragError::Store(format!("Task join error: {e}")))?
    }

    async fn search(&self, embedding: &[f32], top_k: usize) -> Result<Vec<SearchResult>> {
        if embedding.len() != self.config.embedding_dimension {
            return Err(QuackragError::Store(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.config.embedding_dimension,
                embedding.len()
            )));
        }

        let conn = Arc::clone(&self.conn);
        let embedding = embedding.to_vec();
        let dim = self.config.embedding_dimension;
        let config = self.config.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| QuackragError::Store(format!("Failed to lock connection: {e}")))?;

            let embedding_literal = sql::format_embedding_literal(&embedding, dim);
            let search_sql = sql::build_search_sql(&embedding_literal, &config);

            let mut stmt = conn
                .prepare(&search_sql)
                .map_err(|e| QuackragError::Store(format!("Failed to prepare search: {e}")))?;

            let results = stmt
                .query_map(duckdb::params![top_k as i32], |row| {
                    let id: String = row.get(0)?;
                    let source: String = row.get(1)?;
                    let content: String = row.get(2)?;
                    let chunk_index: i32 = row.get(3)?;
                    let metadata_str: String = row.get(4)?;
                    let distance: f32 = row.get(5)?;

                    let metadata: HashMap<String, String> = serde_json::from_str(&metadata_str)
                        .unwrap_or_else(|e| {
                            tracing::warn!(
                                "Failed to deserialize metadata for document {}: {}",
                                id,
                                e
                            );
                            HashMap::new()
                        });

                    // メトリックに応じてスコアを計算
                    let score = config.distance_metric.distance_to_score(distance);

                    Ok(SearchResult {
                        document: Document {
                            id,
                            source,
                            content,
                            chunk_index: chunk_index as usize,
                            metadata,
                        },
                        score,
                        distance,
                    })
                })
                .map_err(|e| QuackragError::Store(format!("Failed to execute search: {e}")))?;

            let mut search_results = Vec::new();
            for result in results {
                search_results.push(
                    result.map_err(|e| QuackragError::Store(format!("Failed to read row: {e}")))?,
                );
            }

            Ok(search_results)
        })
        .await
        .map_err(|e| QuackragError::Store(format!("Task join error: {e}")))?
    }

    async fn delete(&self, document_id: &str) -> Result<()> {
        let conn = Arc::clone(&self.conn);
        let document_id = document_id.to_string();

        tokio::task::spawn_blocking(move || {
            let mut conn = conn
                .lock()
                .map_err(|e| QuackragError::Store(format!("Failed to lock connection: {e}")))?;

            // トランザクション開始
            let tx = conn
                .transaction()
                .map_err(|e| QuackragError::Store(format!("Failed to begin transaction: {e}")))?;

            // 埋め込みベクトル削除
            tx.execute(
                sql::build_delete_embeddings_sql(),
                duckdb::params![document_id],
            )
            .map_err(|e| QuackragError::Store(format!("Failed to delete embeddings: {e}")))?;

            // ドキュメント削除
            tx.execute(
                sql::build_delete_document_sql(),
                duckdb::params![document_id],
            )
            .map_err(|e| QuackragError::Store(format!("Failed to delete document: {e}")))?;

            // コミット
            tx.commit()
                .map_err(|e| QuackragError::Store(format!("Failed to commit transaction: {e}")))?;

            Ok(())
        })
        .await
        .map_err(|e| QuackragError::Store(format!("Task join error: {e}")))?
    }

    async fn count(&self) -> Result<usize> {
        let conn = Arc::clone(&self.conn);

        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| QuackragError::Store(format!("Failed to lock connection: {e}")))?;

            let count: i64 = conn
                .query_row(sql::build_count_sql(), [], |row| row.get(0))
                .map_err(|e| QuackragError::Store(format!("Failed to count documents: {e}")))?;

            Ok(count as usize)
        })
        .await
        .map_err(|e| QuackragError::Store(format!("Task join error: {e}")))?
    }

    async fn clear(&self) -> Result<()> {
        let conn = Arc::clone(&self.conn);

        tokio::task::spawn_blocking(move || {
            let conn = conn
                .lock()
                .map_err(|e| QuackragError::Store(format!("Failed to lock connection: {e}")))?;

            conn.execute_batch(sql::build_clear_sql())
                .map_err(|e| QuackragError::Store(format!("Failed to clear store: {e}")))?;

            Ok(())
        })
        .await
        .map_err(|e| QuackragError::Store(format!("Task join error: {e}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn _assert_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<DuckDbStore>();
        assert_sync::<DuckDbStore>();
    }

    fn _assert_trait_object() {
        fn assert_vector_store<T: VectorStore>() {}
        assert_vector_store::<DuckDbStore>();
    }
}
