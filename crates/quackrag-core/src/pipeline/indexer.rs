use crate::chunker::ChunkStrategy;
use crate::error::Result;
use crate::traits::embedding::EmbeddingModel;
use crate::traits::store::VectorStore;
use crate::types::document::Document;

pub async fn index_document(
    source: &str,
    content: &str,
    chunker: &dyn ChunkStrategy,
    embedding: &dyn EmbeddingModel,
    store: &dyn VectorStore,
) -> Result<Vec<Document>> {
    let chunks = chunker.chunk(content);
    if chunks.is_empty() {
        return Ok(vec![]);
    }

    let chunk_refs: Vec<&str> = chunks.iter().map(|s| s.as_str()).collect();
    let embeddings = embedding.embed_batch(&chunk_refs).await?;

    let mut documents = Vec::with_capacity(chunks.len());
    for (i, (chunk, emb)) in chunks.into_iter().zip(embeddings.iter()).enumerate() {
        let doc = Document::new(source, chunk, i);
        store.insert(&doc, emb).await?;
        documents.push(doc);
    }

    Ok(documents)
}
