use std::path::PathBuf;

/// LLMモデルのソース指定
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// HuggingFace Hubからモデルをダウンロード
    HuggingFace {
        model_repo: String,
        model_file: String,
        tokenizer_repo: String,
    },
    /// ローカルファイルを直接使用
    Local {
        model_path: PathBuf,
        tokenizer_path: PathBuf,
    },
}

/// Candleバックエンドの設定
#[derive(Debug, Clone)]
pub struct CandleConfig {
    pub model_source: ModelSource,
    pub max_tokens: usize,
    pub temperature: f64,
    pub seed: u64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl Default for CandleConfig {
    fn default() -> Self {
        Self {
            model_source: ModelSource::HuggingFace {
                model_repo: "ggml-org/gemma-3-1b-it-GGUF".to_string(),
                model_file: "gemma-3-1b-it-Q4_K_M.gguf".to_string(),
                tokenizer_repo: "google/gemma-3-1b-it".to_string(),
            },
            max_tokens: 256,
            temperature: 0.8,
            seed: 299792458,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        }
    }
}

impl CandleConfig {
    pub fn with_model_source(mut self, source: ModelSource) -> Self {
        self.model_source = source;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_repeat_penalty(mut self, penalty: f32) -> Self {
        self.repeat_penalty = penalty;
        self
    }

    pub fn with_repeat_last_n(mut self, n: usize) -> Self {
        self.repeat_last_n = n;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = CandleConfig::default();
        assert_eq!(config.max_tokens, 256);
        assert!((config.temperature - 0.8).abs() < f64::EPSILON);
        assert_eq!(config.seed, 299792458);
        assert!((config.repeat_penalty - 1.1).abs() < f32::EPSILON);
        assert_eq!(config.repeat_last_n, 64);

        match &config.model_source {
            ModelSource::HuggingFace {
                model_repo,
                model_file,
                tokenizer_repo,
            } => {
                assert_eq!(model_repo, "ggml-org/gemma-3-1b-it-GGUF");
                assert_eq!(model_file, "gemma-3-1b-it-Q4_K_M.gguf");
                assert_eq!(tokenizer_repo, "google/gemma-3-1b-it");
            }
            _ => panic!("Expected HuggingFace source"),
        }
    }

    #[test]
    fn builder_pattern() {
        let config = CandleConfig::default()
            .with_max_tokens(512)
            .with_temperature(0.5)
            .with_seed(42)
            .with_repeat_penalty(1.2)
            .with_repeat_last_n(128)
            .with_model_source(ModelSource::Local {
                model_path: PathBuf::from("/tmp/model.gguf"),
                tokenizer_path: PathBuf::from("/tmp/tokenizer.json"),
            });

        assert_eq!(config.max_tokens, 512);
        assert!((config.temperature - 0.5).abs() < f64::EPSILON);
        assert_eq!(config.seed, 42);
        assert!((config.repeat_penalty - 1.2).abs() < f32::EPSILON);
        assert_eq!(config.repeat_last_n, 128);

        match &config.model_source {
            ModelSource::Local {
                model_path,
                tokenizer_path,
            } => {
                assert_eq!(model_path, &PathBuf::from("/tmp/model.gguf"));
                assert_eq!(tokenizer_path, &PathBuf::from("/tmp/tokenizer.json"));
            }
            _ => panic!("Expected Local source"),
        }
    }
}
