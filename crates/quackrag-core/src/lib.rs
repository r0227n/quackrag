pub mod chunker;
pub mod error;
pub mod pipeline;
pub mod prompt;
pub mod traits;
pub mod types;

#[cfg(test)]
mod tests {
    use std::pin::Pin;
    use std::sync::Mutex;

    use async_trait::async_trait;
    use futures::stream;
    use futures::Stream;

    use crate::chunker::FixedSizeChunker;
    use crate::error::{QuackragError, Result};
    use crate::pipeline::RagPipeline;
    use crate::traits::embedding::EmbeddingModel;
    use crate::traits::llm::LlmBackend;
    use crate::traits::store::VectorStore;
    use crate::types::document::Document;
    use crate::types::search::SearchResult;

    // --- Mock implementations ---

    struct MockEmbedding {
        dim: usize,
    }

    #[async_trait]
    impl EmbeddingModel for MockEmbedding {
        fn dimension(&self) -> usize {
            self.dim
        }

        async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![0.1; self.dim])
        }

        async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.1; self.dim]).collect())
        }
    }

    struct MockLlm;

    #[async_trait]
    impl LlmBackend for MockLlm {
        async fn generate(&self, prompt: &str) -> Result<String> {
            Ok(format!("Mock response to: {}", prompt))
        }

        fn generate_stream(
            &self,
            prompt: String,
        ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + '_>> {
            let tokens = vec![
                Ok("Mock ".to_string()),
                Ok("response ".to_string()),
                Ok(format!("to: {}", prompt)),
            ];
            Box::pin(stream::iter(tokens))
        }

        fn model_name(&self) -> &str {
            "mock-model"
        }
    }

    struct MockStore {
        documents: Mutex<Vec<(Document, Vec<f32>)>>,
    }

    impl MockStore {
        fn new() -> Self {
            Self {
                documents: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl VectorStore for MockStore {
        async fn insert(&self, document: &Document, embedding: &[f32]) -> Result<()> {
            self.documents
                .lock()
                .unwrap()
                .push((document.clone(), embedding.to_vec()));
            Ok(())
        }

        async fn search(&self, _embedding: &[f32], top_k: usize) -> Result<Vec<SearchResult>> {
            let docs = self.documents.lock().unwrap();
            Ok(docs
                .iter()
                .take(top_k)
                .map(|(doc, _)| SearchResult {
                    document: doc.clone(),
                    score: 0.95,
                    distance: 0.05,
                })
                .collect())
        }

        async fn delete(&self, document_id: &str) -> Result<()> {
            self.documents
                .lock()
                .unwrap()
                .retain(|(doc, _)| doc.id != document_id);
            Ok(())
        }

        async fn count(&self) -> Result<usize> {
            Ok(self.documents.lock().unwrap().len())
        }

        async fn clear(&self) -> Result<()> {
            self.documents.lock().unwrap().clear();
            Ok(())
        }
    }

    // --- Tests ---

    #[test]
    fn error_display() {
        let err = QuackragError::Embedding("test error".to_string());
        assert_eq!(format!("{err}"), "Embedding error: test error");

        let err = QuackragError::Store("store fail".to_string());
        assert_eq!(format!("{err}"), "Store error: store fail");

        let err = QuackragError::Llm("llm fail".to_string());
        assert_eq!(format!("{err}"), "LLM error: llm fail");

        let err = QuackragError::Chunker("chunk fail".to_string());
        assert_eq!(format!("{err}"), "Chunker error: chunk fail");
    }

    #[test]
    fn document_creation() {
        let doc = Document::new("test.txt", "hello world", 0);
        assert_eq!(doc.source, "test.txt");
        assert_eq!(doc.content, "hello world");
        assert_eq!(doc.chunk_index, 0);
        assert!(!doc.id.is_empty());
        assert!(doc.metadata.is_empty());
    }

    #[test]
    fn document_serialization() {
        let doc = Document::new("test.txt", "hello", 0);
        let json = serde_json::to_string(&doc).unwrap();
        let deserialized: Document = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, doc.id);
        assert_eq!(deserialized.source, doc.source);
        assert_eq!(deserialized.content, doc.content);
    }

    #[test]
    fn document_unique_ids() {
        let doc1 = Document::new("a", "b", 0);
        let doc2 = Document::new("a", "b", 0);
        assert_ne!(doc1.id, doc2.id);
    }

    #[tokio::test]
    async fn mock_store_crud() {
        let store = MockStore::new();

        assert_eq!(store.count().await.unwrap(), 0);

        let doc = Document::new("test.txt", "content", 0);
        let doc_id = doc.id.clone();
        store.insert(&doc, &[0.1, 0.2, 0.3]).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);

        let results = store.search(&[0.1, 0.2, 0.3], 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document.id, doc_id);

        store.delete(&doc_id).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn mock_store_clear() {
        let store = MockStore::new();
        let doc1 = Document::new("a", "content1", 0);
        let doc2 = Document::new("b", "content2", 0);
        store.insert(&doc1, &[0.1]).await.unwrap();
        store.insert(&doc2, &[0.2]).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 2);

        store.clear().await.unwrap();
        assert_eq!(store.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn pipeline_index_and_ask() {
        let embedding = MockEmbedding { dim: 4 };
        let store = MockStore::new();
        let llm = MockLlm;

        let pipeline = RagPipeline::new(embedding, store, llm)
            .with_chunker(FixedSizeChunker::new(10, 0))
            .with_top_k(3);

        let docs = pipeline
            .index_document("test.txt", "Hello world, this is a test document.")
            .await
            .unwrap();
        assert!(!docs.is_empty());

        let answer = pipeline.ask("What is this?").await.unwrap();
        assert!(answer.contains("Mock response to:"));
    }

    #[tokio::test]
    async fn pipeline_ask_stream() {
        use futures::StreamExt;

        let embedding = MockEmbedding { dim: 4 };
        let store = MockStore::new();
        let llm = MockLlm;

        let pipeline = RagPipeline::new(embedding, store, llm);

        let doc = Document::new("test.txt", "content", 0);
        let results = vec![SearchResult {
            document: doc,
            score: 0.9,
            distance: 0.1,
        }];

        let stream = pipeline.ask_stream("question?", &results);
        let tokens: Vec<String> = stream.map(|r| r.unwrap()).collect().await;
        assert!(!tokens.is_empty());
        assert!(tokens.join("").contains("Mock response"));
    }

    #[tokio::test]
    async fn pipeline_empty_content() {
        let embedding = MockEmbedding { dim: 4 };
        let store = MockStore::new();
        let llm = MockLlm;

        let pipeline = RagPipeline::new(embedding, store, llm);
        let docs = pipeline.index_document("empty.txt", "").await.unwrap();
        assert!(docs.is_empty());
    }
}
