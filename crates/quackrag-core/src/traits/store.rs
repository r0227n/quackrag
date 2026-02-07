use async_trait::async_trait;

use crate::error::Result;
use crate::types::document::Document;
use crate::types::search::SearchResult;

#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn insert(&self, document: &Document, embedding: &[f32]) -> Result<()>;
    async fn search(&self, embedding: &[f32], top_k: usize) -> Result<Vec<SearchResult>>;
    async fn delete(&self, document_id: &str) -> Result<()>;
    async fn count(&self) -> Result<usize>;
    async fn clear(&self) -> Result<()>;
}
