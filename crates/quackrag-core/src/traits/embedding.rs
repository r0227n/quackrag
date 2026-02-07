use async_trait::async_trait;

use crate::error::Result;

#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    fn dimension(&self) -> usize;
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}
