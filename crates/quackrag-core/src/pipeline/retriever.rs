use crate::error::Result;
use crate::traits::embedding::EmbeddingModel;
use crate::traits::store::VectorStore;
use crate::types::search::SearchResult;

pub async fn retrieve(
    query: &str,
    top_k: usize,
    embedding: &dyn EmbeddingModel,
    store: &dyn VectorStore,
) -> Result<Vec<SearchResult>> {
    let query_embedding = embedding.embed(query).await?;
    store.search(&query_embedding, top_k).await
}
