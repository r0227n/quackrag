use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct AppConfig {
    #[serde(default)]
    pub database: DatabaseConfig,
    #[serde(default)]
    pub llm: LlmConfig,
    #[serde(default)]
    pub chunker: ChunkerConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseConfig {
    pub path: PathBuf,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        let path = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("quackrag")
            .join("quackrag.db");
        Self { path }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfig {
    pub model_repo: String,
    pub model_file: String,
    pub tokenizer_repo: String,
    pub max_tokens: usize,
    pub temperature: f64,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model_repo: "ggml-org/gemma-3-1b-it-GGUF".to_string(),
            model_file: "gemma-3-1b-it-Q4_K_M.gguf".to_string(),
            tokenizer_repo: "google/gemma-3-1b-it".to_string(),
            max_tokens: 256,
            temperature: 0.8,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChunkerConfig {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 64,
        }
    }
}

impl AppConfig {
    pub fn load(config_path: Option<&Path>, db_override: Option<&Path>) -> anyhow::Result<Self> {
        let mut config = if let Some(path) = config_path {
            let content = std::fs::read_to_string(path).map_err(|e| {
                anyhow::anyhow!("Failed to read config file {}: {e}", path.display())
            })?;
            toml::from_str(&content)?
        } else {
            // デフォルトパスから読み込み試行
            let default_path = dirs::config_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("quackrag")
                .join("config.toml");

            if default_path.exists() {
                let content = std::fs::read_to_string(&default_path).map_err(|e| {
                    anyhow::anyhow!("Failed to read config file {}: {e}", default_path.display())
                })?;
                toml::from_str(&content)?
            } else {
                Self::default()
            }
        };

        // CLI引数でDB pathを上書き
        if let Some(db_path) = db_override {
            config.database.path = db_path.to_path_buf();
        }

        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = AppConfig::default();
        assert_eq!(config.llm.model_repo, "ggml-org/gemma-3-1b-it-GGUF");
        assert_eq!(config.chunker.chunk_size, 512);
    }

    #[test]
    fn parse_toml() {
        let toml_str = r#"
[database]
path = "/tmp/test.db"

[llm]
model_repo = "custom/model"
model_file = "model.gguf"
tokenizer_repo = "custom/tokenizer"
max_tokens = 512
temperature = 0.5

[chunker]
chunk_size = 1024
chunk_overlap = 128
"#;
        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.database.path, PathBuf::from("/tmp/test.db"));
        assert_eq!(config.llm.model_repo, "custom/model");
        assert_eq!(config.chunker.chunk_size, 1024);
    }

    #[test]
    fn db_override() {
        let config = AppConfig::load(None, Some(Path::new("/tmp/override.db"))).unwrap();
        assert_eq!(config.database.path, PathBuf::from("/tmp/override.db"));
    }
}
