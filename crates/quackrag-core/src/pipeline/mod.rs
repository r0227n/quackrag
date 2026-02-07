pub mod generator;
pub mod indexer;
pub mod retriever;

use std::pin::Pin;

use futures::Stream;

use crate::chunker::{ChunkStrategy, FixedSizeChunker};
use crate::error::Result;
use crate::prompt::PromptBuilder;
use crate::traits::embedding::EmbeddingModel;
use crate::traits::llm::LlmBackend;
use crate::traits::store::VectorStore;
use crate::types::document::Document;
use crate::types::search::SearchResult;

pub struct RagPipeline<E, S, L> {
    embedding: E,
    store: S,
    llm: L,
    chunker: Box<dyn ChunkStrategy>,
    prompt_builder: PromptBuilder,
    top_k: usize,
}

impl<E, S, L> RagPipeline<E, S, L>
where
    E: EmbeddingModel,
    S: VectorStore,
    L: LlmBackend,
{
    pub fn new(embedding: E, store: S, llm: L) -> Self {
        Self {
            embedding,
            store,
            llm,
            chunker: Box::new(FixedSizeChunker::default()),
            prompt_builder: PromptBuilder::default(),
            top_k: 5,
        }
    }

    pub fn with_chunker(mut self, chunker: impl ChunkStrategy + 'static) -> Self {
        self.chunker = Box::new(chunker);
        self
    }

    pub fn with_prompt_builder(mut self, prompt_builder: PromptBuilder) -> Self {
        self.prompt_builder = prompt_builder;
        self
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    pub async fn index_document(&self, source: &str, content: &str) -> Result<Vec<Document>> {
        indexer::index_document(
            source,
            content,
            &*self.chunker,
            &self.embedding,
            &self.store,
        )
        .await
    }

    pub async fn search(&self, query: &str) -> Result<Vec<SearchResult>> {
        retriever::retrieve(query, self.top_k, &self.embedding, &self.store).await
    }

    pub async fn ask(&self, question: &str) -> Result<String> {
        let results = self.search(question).await?;
        generator::generate(question, &results, &self.prompt_builder, &self.llm).await
    }

    pub fn ask_stream(
        &self,
        question: &str,
        results: &[SearchResult],
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + '_>> {
        let prompt = generator::build_prompt(question, results, &self.prompt_builder);
        self.llm.generate_stream(prompt)
    }
}
