use thiserror::Error;

#[derive(Debug, Error)]
pub enum QuackragError {
    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Store error: {0}")]
    Store(String),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("Chunker error: {0}")]
    Chunker(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, QuackragError>;
