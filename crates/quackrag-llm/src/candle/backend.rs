use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use candle_core::quantized::gguf_file;
use candle_core::Device;
use candle_transformers::models::quantized_gemma3::ModelWeights;
use futures::Stream;
use tokenizers::Tokenizer;

use quackrag_core::error::{QuackragError, Result};
use quackrag_core::traits::llm::LlmBackend;

use super::generation;
use crate::config::{CandleConfig, ModelSource};

pub struct CandleBackend {
    model: Arc<Mutex<ModelWeights>>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    config: CandleConfig,
    model_name: String,
}

impl CandleBackend {
    /// 設定からCandleBackendを構築する
    ///
    /// モデルファイルとトークナイザーをロードして初期化する。
    /// HuggingFace Hubからのダウンロード、またはローカルファイルの読み込みに対応。
    pub fn new(config: CandleConfig) -> Result<Self> {
        let device = Device::Cpu;

        let (model_path, tokenizer_path, model_name) = match &config.model_source {
            ModelSource::HuggingFace {
                model_repo,
                model_file,
                tokenizer_repo,
            } => {
                let api = hf_hub::api::sync::Api::new()
                    .map_err(|e| QuackragError::Llm(format!("Failed to create HF API: {e}")))?;

                let model_path = api
                    .model(model_repo.clone())
                    .get(model_file)
                    .map_err(|e| QuackragError::Llm(format!("Failed to download model: {e}")))?;

                let tokenizer_path = api
                    .model(tokenizer_repo.clone())
                    .get("tokenizer.json")
                    .map_err(|e| {
                        QuackragError::Llm(format!("Failed to download tokenizer: {e}"))
                    })?;

                let name = format!("{model_repo}/{model_file}");
                (model_path, tokenizer_path, name)
            }
            ModelSource::Local {
                model_path,
                tokenizer_path,
            } => {
                let name = model_path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "local-model".to_string());
                (model_path.clone(), tokenizer_path.clone(), name)
            }
        };

        tracing::info!("Loading GGUF model from {:?}", model_path);
        let mut file = std::fs::File::open(&model_path).map_err(|e| {
            QuackragError::Llm(format!("Failed to open model file {model_path:?}: {e}"))
        })?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| QuackragError::Llm(format!("Failed to read GGUF content: {e}")))?;
        let model = ModelWeights::from_gguf(content, &mut file, &device)
            .map_err(|e| QuackragError::Llm(format!("Failed to load model weights: {e}")))?;
        tracing::info!("Model loaded successfully");

        tracing::info!("Loading tokenizer from {:?}", tokenizer_path);
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            QuackragError::Llm(format!("Failed to load tokenizer {tokenizer_path:?}: {e}"))
        })?;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            tokenizer: Arc::new(tokenizer),
            device,
            config,
            model_name,
        })
    }
}

#[async_trait]
impl LlmBackend for CandleBackend {
    async fn generate(&self, prompt: &str) -> Result<String> {
        let model = Arc::clone(&self.model);
        let tokenizer = Arc::clone(&self.tokenizer);
        let device = self.device.clone();
        let config = self.config.clone();
        let prompt = prompt.to_string();

        tokio::task::spawn_blocking(move || {
            let mut model = model
                .lock()
                .map_err(|e| QuackragError::Llm(format!("Failed to lock model: {e}")))?;
            generation::generate_full(&mut model, &tokenizer, &prompt, &device, &config)
        })
        .await
        .map_err(|e| QuackragError::Llm(format!("Task join error: {e}")))?
    }

    fn generate_stream(
        &self,
        prompt: String,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + '_>> {
        let model = Arc::clone(&self.model);
        let tokenizer = Arc::clone(&self.tokenizer);
        let device = self.device.clone();
        let config = self.config.clone();

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        std::thread::spawn(move || {
            let mut model = match model.lock() {
                Ok(m) => m,
                Err(e) => {
                    let _ = tx.send(Err(QuackragError::Llm(format!(
                        "Failed to lock model: {e}"
                    ))));
                    return;
                }
            };
            generation::generate_streaming(&mut model, &tokenizer, &prompt, &device, &config, tx);
        });

        Box::pin(futures::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        }))
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CandleBackendがSend + Syncを満たすことをコンパイル時に検証
    fn _assert_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<CandleBackend>();
        assert_sync::<CandleBackend>();
    }

    /// LlmBackendトレイトオブジェクトとして使用可能であることを検証
    fn _assert_trait_object() {
        fn assert_llm_backend<T: LlmBackend>() {}
        assert_llm_backend::<CandleBackend>();
    }
}
