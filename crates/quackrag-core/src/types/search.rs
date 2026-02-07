use serde::{Deserialize, Serialize};

use super::document::Document;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub document: Document,
    pub score: f32,
    pub distance: f32,
}
