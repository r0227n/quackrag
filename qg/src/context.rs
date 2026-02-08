use std::path::PathBuf;

use indicatif::{ProgressBar, ProgressStyle};
use quackrag_embedding::{CandleEmbedding, EmbeddingConfig};
use quackrag_llm::{CandleBackend, CandleConfig};
use quackrag_store::config::{DuckDbConfig, StorageMode};
use quackrag_store::DuckDbStore;

use crate::config::AppConfig;

pub struct AppContext {
    pub store: DuckDbStore,
    pub embedding: Option<CandleEmbedding>,
    pub llm: Option<CandleBackend>,
}

impl AppContext {
    pub fn store_only(config: &AppConfig) -> anyhow::Result<Self> {
        let store = init_store(config)?;
        Ok(Self {
            store,
            embedding: None,
            llm: None,
        })
    }

    pub fn with_embedding(config: &AppConfig) -> anyhow::Result<Self> {
        let store = init_store(config)?;
        let embedding = init_embedding()?;
        Ok(Self {
            store,
            embedding: Some(embedding),
            llm: None,
        })
    }

    pub fn full(config: &AppConfig) -> anyhow::Result<Self> {
        let store = init_store(config)?;
        let embedding = init_embedding()?;
        let llm = init_llm(config)?;
        Ok(Self {
            store,
            embedding: Some(embedding),
            llm: Some(llm),
        })
    }

    pub fn embedding(&self) -> anyhow::Result<&CandleEmbedding> {
        self.embedding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Embedding model not initialized"))
    }

    pub fn llm(&self) -> anyhow::Result<&CandleBackend> {
        self.llm
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("LLM not initialized"))
    }
}

fn init_store(config: &AppConfig) -> anyhow::Result<DuckDbStore> {
    let spinner = make_spinner("Initializing database...");
    let db_path = &config.database.path;

    // 親ディレクトリが存在しなければ作成
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let store_config = DuckDbConfig::default().with_storage_mode(StorageMode::File {
        path: PathBuf::from(db_path),
    });
    let store = DuckDbStore::new(store_config)?;
    spinner.finish_with_message("Database ready");
    Ok(store)
}

fn init_embedding() -> anyhow::Result<CandleEmbedding> {
    let spinner = make_spinner("Loading embedding model...");
    let embedding = CandleEmbedding::new(EmbeddingConfig::default())?;
    spinner.finish_with_message("Embedding model ready");
    Ok(embedding)
}

fn init_llm(config: &AppConfig) -> anyhow::Result<CandleBackend> {
    let spinner = make_spinner("Loading LLM...");
    let llm_config = CandleConfig::default()
        .with_max_tokens(config.llm.max_tokens)
        .with_temperature(config.llm.temperature);
    let llm = CandleBackend::new(llm_config)?;
    spinner.finish_with_message("LLM ready");
    Ok(llm)
}

fn make_spinner(msg: &str) -> ProgressBar {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    spinner.set_message(msg.to_string());
    spinner.enable_steady_tick(std::time::Duration::from_millis(100));
    spinner
}
