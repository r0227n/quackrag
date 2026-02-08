use std::path::PathBuf;

use quackrag_embedding::{EmbeddingConfig, ModelSource};

/// デフォルト設定の検証
#[test]
fn test_default_config() {
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

/// ビルダーパターンの検証
#[test]
fn test_builder_pattern() {
    let config = EmbeddingConfig::default()
        .with_max_batch_size(64)
        .with_max_seq_length(512)
        .with_normalize(false);

    assert_eq!(config.max_batch_size, 64);
    assert_eq!(config.max_seq_length, 512);
    assert!(!config.normalize);
}

/// ModelSource::HuggingFace バリアントの検証
#[test]
fn test_model_source_huggingface() {
    let config = EmbeddingConfig::default().with_model_source(ModelSource::HuggingFace {
        repo_id: "custom/model".to_string(),
    });

    match &config.model_source {
        ModelSource::HuggingFace { repo_id } => {
            assert_eq!(repo_id, "custom/model");
        }
        _ => panic!("Expected HuggingFace source"),
    }
}

/// ModelSource::Local バリアントの検証
#[test]
fn test_model_source_local() {
    let config = EmbeddingConfig::default().with_model_source(ModelSource::Local {
        model_path: PathBuf::from("/tmp/model.safetensors"),
        tokenizer_path: PathBuf::from("/tmp/tokenizer.json"),
        config_path: PathBuf::from("/tmp/config.json"),
    });

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

/// ビルダーのチェーン呼び出し検証
#[test]
fn test_builder_chaining() {
    let config = EmbeddingConfig::default()
        .with_max_batch_size(16)
        .with_max_seq_length(128)
        .with_normalize(true)
        .with_model_source(ModelSource::HuggingFace {
            repo_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        });

    assert_eq!(config.max_batch_size, 16);
    assert_eq!(config.max_seq_length, 128);
    assert!(config.normalize);
}
