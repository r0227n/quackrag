use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::error::Result;

#[async_trait]
pub trait LlmBackend: Send + Sync {
    async fn generate(&self, prompt: &str) -> Result<String>;
    fn generate_stream(
        &self,
        prompt: String,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + '_>>;
    fn model_name(&self) -> &str;
}
