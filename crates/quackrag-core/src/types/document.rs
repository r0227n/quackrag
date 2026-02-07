use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub source: String,
    pub content: String,
    pub chunk_index: usize,
    pub metadata: HashMap<String, String>,
}

impl Document {
    pub fn new(source: impl Into<String>, content: impl Into<String>, chunk_index: usize) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            source: source.into(),
            content: content.into(),
            chunk_index,
            metadata: HashMap::new(),
        }
    }
}
