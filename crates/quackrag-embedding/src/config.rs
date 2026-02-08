use std::path::PathBuf;

/// Embeddingモデルのソース指定
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// HuggingFace Hubからモデルをダウンロード
    HuggingFace { repo_id: String },
    /// ローカルファイルを直接使用
    Local {
        model_path: PathBuf,
        tokenizer_path: PathBuf,
        config_path: PathBuf,
    },
}

/// Embedding設定
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub model_source: ModelSource,
    pub max_batch_size: usize,
    pub max_seq_length: usize,
    pub normalize: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_source: ModelSource::HuggingFace {
                repo_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            },
            max_batch_size: 32,
            max_seq_length: 256,
            normalize: true,
        }
    }
}

impl EmbeddingConfig {
    pub fn with_model_source(mut self, source: ModelSource) -> Self {
        self.model_source = source;
        self
    }

    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    pub fn with_max_seq_length(mut self, length: usize) -> Self {
        self.max_seq_length = length;
        self
    }

    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.max_seq_length, 256);
        assert!(config.normalize);

        match &config.model_source {
            ModelSource::HuggingFace { repo_id } => {
                assert_eq!(repo_id, "sentence-transformers/all-MiniLM-L6-v2");
            }
            _ => panic!("Expected HuggingFace source"),
        }
    }

    #[test]
    fn builder_pattern() {
        let config = EmbeddingConfig::default()
            .with_max_batch_size(64)
            .with_max_seq_length(512)
            .with_normalize(false)
            .with_model_source(ModelSource::Local {
                model_path: PathBuf::from("/tmp/model.safetensors"),
                tokenizer_path: PathBuf::from("/tmp/tokenizer.json"),
                config_path: PathBuf::from("/tmp/config.json"),
            });

        assert_eq!(config.max_batch_size, 64);
        assert_eq!(config.max_seq_length, 512);
        assert!(!config.normalize);

        match &config.model_source {
            ModelSource::Local {
                model_path,
                tokenizer_path,
                config_path,
            } => {
                assert_eq!(model_path, &PathBuf::from("/tmp/model.safetensors"));
                assert_eq!(tokenizer_path, &PathBuf::from("/tmp/tokenizer.json"));
                assert_eq!(config_path, &PathBuf::from("/tmp/config.json"));
            }
            _ => panic!("Expected Local source"),
        }
    }
}
