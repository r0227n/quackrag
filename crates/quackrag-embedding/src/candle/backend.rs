use std::sync::Arc;

use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use tokenizers::Tokenizer;

use quackrag_core::error::{QuackragError, Result};
use quackrag_core::traits::embedding::EmbeddingModel;

use crate::config::{EmbeddingConfig, ModelSource};

pub struct CandleEmbedding {
    model: Arc<BertModel>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    config: EmbeddingConfig,
    hidden_size: usize,
}

impl CandleEmbedding {
    /// 設定からCandleEmbeddingを構築する
    ///
    /// モデルファイル・トークナイザー・設定をロードして初期化する。
    /// HuggingFace Hubからのダウンロード、またはローカルファイルの読み込みに対応。
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        let device = Device::Cpu;

        let (model_path, tokenizer_path, config_path) = match &config.model_source {
            ModelSource::HuggingFace { repo_id } => {
                let api = hf_hub::api::sync::Api::new().map_err(|e| {
                    QuackragError::Embedding(format!("Failed to create HF API: {e}"))
                })?;
                let repo = api.model(repo_id.clone());

                let model_path = repo.get("model.safetensors").map_err(|e| {
                    QuackragError::Embedding(format!("Failed to download model: {e}"))
                })?;
                let tokenizer_path = repo.get("tokenizer.json").map_err(|e| {
                    QuackragError::Embedding(format!("Failed to download tokenizer: {e}"))
                })?;
                let config_path = repo.get("config.json").map_err(|e| {
                    QuackragError::Embedding(format!("Failed to download config: {e}"))
                })?;

                (model_path, tokenizer_path, config_path)
            }
            ModelSource::Local {
                model_path,
                tokenizer_path,
                config_path,
            } => (
                model_path.clone(),
                tokenizer_path.clone(),
                config_path.clone(),
            ),
        };

        // config.json を読み込み、BertConfig にデシリアライズ
        tracing::info!("Loading config from {:?}", config_path);
        let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
            QuackragError::Embedding(format!("Failed to read config {config_path:?}: {e}"))
        })?;
        let bert_config: BertConfig = serde_json::from_str(&config_str)
            .map_err(|e| QuackragError::Embedding(format!("Failed to parse config: {e}")))?;
        let hidden_size = bert_config.hidden_size;

        // SafeTensors から重みをロード
        // SAFETY: モデルファイルはロード後に変更されない読み取り専用ファイルであるため、
        // メモリマップドアクセスは安全です。
        tracing::info!("Loading model from {:?}", model_path);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&model_path], DType::F32, &device)
                .map_err(|e| QuackragError::Embedding(format!("Failed to load safetensors: {e}")))?
        };
        let model = BertModel::load(vb, &bert_config)
            .map_err(|e| QuackragError::Embedding(format!("Failed to load BertModel: {e}")))?;
        tracing::info!("Model loaded successfully (hidden_size={})", hidden_size);

        // トークナイザーをロード
        tracing::info!("Loading tokenizer from {:?}", tokenizer_path);
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            QuackragError::Embedding(format!("Failed to load tokenizer {tokenizer_path:?}: {e}"))
        })?;

        Ok(Self {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            device,
            config,
            hidden_size,
        })
    }

    /// テキスト群をトークナイズし、パディング付きテンソルを生成する
    fn tokenize(&self, texts: &[&str]) -> Result<(Tensor, Tensor, Tensor)> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| QuackragError::Embedding(format!("Tokenization failed: {e}")))?;

        // トークナイザーからパディングトークンIDを取得（未設定の場合は0）
        let pad_token_id = self
            .tokenizer
            .get_padding()
            .map(|p| p.pad_id)
            .unwrap_or(0);

        // 最大シーケンス長を計算（config上限でクリップ、最小1を保証）
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(self.config.max_seq_length)
            .max(1);

        let batch_size = encodings.len();
        let mut input_ids_vec = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask_vec = Vec::with_capacity(batch_size * max_len);
        let mut token_type_ids_vec = Vec::with_capacity(batch_size * max_len);

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let type_ids = encoding.get_type_ids();
            let len = ids.len().min(max_len);

            // 有効トークンをコピー
            for i in 0..len {
                input_ids_vec.push(ids[i] as i64);
                token_type_ids_vec.push(type_ids[i] as i64);
                attention_mask_vec.push(1i64);
            }
            // パディング
            for _ in len..max_len {
                input_ids_vec.push(pad_token_id as i64);
                token_type_ids_vec.push(0i64);
                attention_mask_vec.push(0i64);
            }
        }

        let shape = (batch_size, max_len);
        let input_ids = Tensor::from_vec(input_ids_vec, shape, &self.device)
            .map_err(|e| QuackragError::Embedding(format!("Failed to create input_ids: {e}")))?;
        let token_type_ids =
            Tensor::from_vec(token_type_ids_vec, shape, &self.device).map_err(|e| {
                QuackragError::Embedding(format!("Failed to create token_type_ids: {e}"))
            })?;
        let attention_mask =
            Tensor::from_vec(attention_mask_vec, shape, &self.device).map_err(|e| {
                QuackragError::Embedding(format!("Failed to create attention_mask: {e}"))
            })?;

        Ok((input_ids, token_type_ids, attention_mask))
    }

    /// Mean Pooling: attention mask を考慮した平均プーリング
    fn mean_pooling(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // attention_mask: [batch, seq_len] → [batch, seq_len, 1]
        let mask = attention_mask
            .unsqueeze(2)
            .map_err(|e| QuackragError::Embedding(format!("Failed to unsqueeze mask: {e}")))?
            .to_dtype(DType::F32)
            .map_err(|e| QuackragError::Embedding(format!("Failed to cast mask: {e}")))?;

        // hidden_states * mask → マスク適用
        let masked = hidden_states
            .broadcast_mul(&mask)
            .map_err(|e| QuackragError::Embedding(format!("Failed to apply mask: {e}")))?;

        // sum(dim=1) / mask_sum
        let summed = masked
            .sum(1)
            .map_err(|e| QuackragError::Embedding(format!("Failed to sum: {e}")))?;
        let mask_sum = mask
            .sum(1)
            .map_err(|e| QuackragError::Embedding(format!("Failed to sum mask: {e}")))?;
        // clamp で 0 除算を防止
        let mask_sum = mask_sum
            .clamp(1e-9, f64::MAX)
            .map_err(|e| QuackragError::Embedding(format!("Failed to clamp mask_sum: {e}")))?;

        let pooled = summed
            .broadcast_div(&mask_sum)
            .map_err(|e| QuackragError::Embedding(format!("Failed to divide: {e}")))?;

        Ok(pooled)
    }

    /// L2正規化
    fn l2_normalize(&self, tensor: &Tensor) -> Result<Tensor> {
        let l2_norm = tensor
            .sqr()
            .and_then(|t| t.sum(1))
            .and_then(|t| t.sqrt())
            .and_then(|t| t.clamp(1e-12, f64::MAX))
            .and_then(|t| t.unsqueeze(1))
            .map_err(|e| QuackragError::Embedding(format!("Failed to compute L2 norm: {e}")))?;

        tensor
            .broadcast_div(&l2_norm)
            .map_err(|e| QuackragError::Embedding(format!("Failed to normalize: {e}")))
    }

    /// 内部推論: テキスト群をembeddingに変換
    fn embed_inner(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // 空配列の場合は早期リターン
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let (input_ids, token_type_ids, attention_mask) = self.tokenize(texts)?;

        // BertModel::forward
        let hidden_states = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| QuackragError::Embedding(format!("Forward pass failed: {e}")))?;

        // Mean Pooling
        let pooled = self.mean_pooling(&hidden_states, &attention_mask)?;

        // L2正規化（設定で有効な場合）
        let result = if self.config.normalize {
            self.l2_normalize(&pooled)?
        } else {
            pooled
        };

        // Tensor → Vec<Vec<f32>>
        let batch_size = texts.len();
        let mut embeddings = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let vec = result
                .get(i)
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| {
                    QuackragError::Embedding(format!("Failed to extract embedding: {e}"))
                })?;
            embeddings.push(vec);
        }

        Ok(embeddings)
    }
}

#[async_trait]
impl EmbeddingModel for CandleEmbedding {
    fn dimension(&self) -> usize {
        self.hidden_size
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let model = Arc::clone(&self.model);
        let tokenizer = Arc::clone(&self.tokenizer);
        let device = self.device.clone();
        let config = self.config.clone();
        let hidden_size = self.hidden_size;
        let text = text.to_string();

        tokio::task::spawn_blocking(move || {
            let embedding = CandleEmbedding {
                model,
                tokenizer,
                device,
                config,
                hidden_size,
            };
            let results = embedding.embed_inner(&[text.as_str()])?;
            results
                .into_iter()
                .next()
                .ok_or_else(|| QuackragError::Embedding("Empty result".to_string()))
        })
        .await
        .map_err(|e| QuackragError::Embedding(format!("Task join error: {e}")))?
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let model = Arc::clone(&self.model);
        let tokenizer = Arc::clone(&self.tokenizer);
        let device = self.device.clone();
        let config = self.config.clone();
        let hidden_size = self.hidden_size;
        let texts: Vec<String> = texts.iter().map(|t| t.to_string()).collect();

        tokio::task::spawn_blocking(move || {
            let embedding = CandleEmbedding {
                model,
                tokenizer,
                device,
                config,
                hidden_size,
            };
            let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

            // バッチサイズに分割して処理
            let mut all_results = Vec::with_capacity(text_refs.len());
            for chunk in text_refs.chunks(embedding.config.max_batch_size) {
                let mut batch_results = embedding.embed_inner(chunk)?;
                all_results.append(&mut batch_results);
            }
            Ok(all_results)
        })
        .await
        .map_err(|e| QuackragError::Embedding(format!("Task join error: {e}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CandleEmbeddingがSend + Syncを満たすことをコンパイル時に検証
    fn _assert_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<CandleEmbedding>();
        assert_sync::<CandleEmbedding>();
    }

    /// EmbeddingModelトレイトオブジェクトとして使用可能であることを検証
    fn _assert_trait_object() {
        fn assert_embedding_model<T: EmbeddingModel>() {}
        assert_embedding_model::<CandleEmbedding>();
    }
}
