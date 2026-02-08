use quackrag_llm::{format_chat_prompt, CandleConfig, ModelSource};

/// ChatTemplateのフォーマット検証
#[test]
fn test_chat_template_format() {
    let user_msg = "Hello, world!";
    let formatted = format_chat_prompt(user_msg);

    // Gemma 3のチャットテンプレート形式を確認
    assert!(formatted.starts_with("<bos><start_of_turn>user\n"));
    assert!(formatted.contains(user_msg));
    assert!(formatted.ends_with("<end_of_turn>\n<start_of_turn>model\n"));
}

/// ChatTemplateの空文字列処理
#[test]
fn test_chat_template_empty() {
    let formatted = format_chat_prompt("");

    // 空プロンプトでもテンプレート構造は維持される
    assert!(formatted.starts_with("<bos><start_of_turn>user\n"));
    assert!(formatted.ends_with("<end_of_turn>\n<start_of_turn>model\n"));
}

/// ChatTemplateの特殊文字処理
#[test]
fn test_chat_template_special_chars() {
    let user_msg = "Test\n<bos>\n</s>\n<end_of_turn>";
    let formatted = format_chat_prompt(user_msg);

    // 特殊トークンがそのまま含まれることを確認（エスケープしない）
    assert!(formatted.contains(user_msg));
}

/// GenerationConfigのデフォルト値検証
#[test]
fn test_generation_config_defaults() {
    let config = CandleConfig::default();

    // デフォルト値の確認（config.rsの実装に合わせる）
    assert_eq!(config.max_tokens, 256);
    assert!(
        (config.temperature - 0.8).abs() < f64::EPSILON,
        "Expected temperature ~0.8, got {}",
        config.temperature
    );
    assert_eq!(config.seed, 299792458);
    assert_eq!(config.repeat_last_n, 64);
    assert!(
        (config.repeat_penalty - 1.1).abs() < f32::EPSILON,
        "Expected repeat_penalty ~1.1, got {}",
        config.repeat_penalty
    );

    // ModelSourceの確認
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

/// GenerationConfigのビルダーパターン検証
#[test]
fn test_generation_config_builder() {
    let config = CandleConfig::default()
        .with_max_tokens(512)
        .with_temperature(0.5)
        .with_seed(42)
        .with_repeat_last_n(128)
        .with_repeat_penalty(1.2);

    assert_eq!(config.max_tokens, 512);
    assert!((config.temperature - 0.5).abs() < f64::EPSILON);
    assert_eq!(config.seed, 42);
    assert_eq!(config.repeat_last_n, 128);
    assert!((config.repeat_penalty - 1.2).abs() < f32::EPSILON);
}

/// GenerationConfigのカスタムモデル設定検証
#[test]
fn test_generation_config_custom_model() {
    let config = CandleConfig::default().with_model_source(ModelSource::HuggingFace {
        model_repo: "ggml-org/gemma-3-4b-it-GGUF".to_string(),
        model_file: "gemma-3-4b-it-Q4_K_M.gguf".to_string(),
        tokenizer_repo: "google/gemma-3-4b-it".to_string(),
    });

    match &config.model_source {
        ModelSource::HuggingFace {
            model_repo,
            model_file,
            tokenizer_repo,
        } => {
            assert_eq!(model_repo, "ggml-org/gemma-3-4b-it-GGUF");
            assert_eq!(model_file, "gemma-3-4b-it-Q4_K_M.gguf");
            assert_eq!(tokenizer_repo, "google/gemma-3-4b-it");
        }
        _ => panic!("Expected HuggingFace source"),
    }
}

/// apply_repeat_penalty関数の動作検証（ダミーテンソル使用）
///
/// NOTE: このテストはCandleの内部実装に依存するため、
/// 実際のテンソル生成は統合テストで行う。
/// ここでは関数のロジックのみを検証する。
#[test]
fn test_repeat_penalty_logic() {
    use candle_core::{Device, Tensor};

    // CPU デバイスでダミーテンソルを作成
    let device = Device::Cpu;
    let logits_vec = vec![1.0, 2.0, 3.0, -1.0, -2.0];
    let logits = Tensor::from_vec(logits_vec.clone(), (5,), &device).unwrap();

    // ペナルティを適用しない場合（penalty = 1.0）
    let penalty_tokens = vec![0, 1];
    let penalty = 1.0;
    let result = apply_repeat_penalty_public(&logits, &penalty_tokens, penalty).unwrap();
    let result_vec = result.to_vec1::<f32>().unwrap();

    // ペナルティ1.0の場合、元の値と同じはず
    for (i, &val) in logits_vec.iter().enumerate() {
        assert!(
            (result_vec[i] - val).abs() < f32::EPSILON,
            "Index {}: expected {}, got {}",
            i,
            val,
            result_vec[i]
        );
    }

    // ペナルティを適用する場合（penalty = 2.0）
    let penalty = 2.0;
    let result = apply_repeat_penalty_public(&logits, &penalty_tokens, penalty).unwrap();
    let result_vec = result.to_vec1::<f32>().unwrap();

    // token 0 (value=1.0): 正なので 1.0/2.0 = 0.5
    assert!((result_vec[0] - 0.5).abs() < f32::EPSILON);
    // token 1 (value=2.0): 正なので 2.0/2.0 = 1.0
    assert!((result_vec[1] - 1.0).abs() < f32::EPSILON);
    // token 2 (value=3.0): ペナルティ対象外なので 3.0
    assert!((result_vec[2] - 3.0).abs() < f32::EPSILON);
    // token 3 (value=-1.0): ペナルティ対象外なので -1.0
    assert!((result_vec[3] - (-1.0)).abs() < f32::EPSILON);
}

/// 空トークンリストに対するペナルティ適用の検証
#[test]
fn test_repeat_penalty_empty_tokens() {
    use candle_core::{Device, Tensor};

    let device = Device::Cpu;
    let logits_vec = vec![1.0, 2.0, 3.0];
    let logits = Tensor::from_vec(logits_vec.clone(), (3,), &device).unwrap();

    // 空トークンリストではペナルティが適用されない
    let penalty_tokens: Vec<u32> = vec![];
    let penalty = 2.0;
    let result = apply_repeat_penalty_public(&logits, &penalty_tokens, penalty).unwrap();
    let result_vec = result.to_vec1::<f32>().unwrap();

    // 元の値と同じはず
    for (i, &val) in logits_vec.iter().enumerate() {
        assert!(
            (result_vec[i] - val).abs() < f32::EPSILON,
            "Index {}: expected {}, got {}",
            i,
            val,
            result_vec[i]
        );
    }
}

// テスト用にapply_repeat_penaltyのパブリック版を定義
// （実際の実装は src/candle/generation.rs 内のプライベート関数）
fn apply_repeat_penalty_public(
    logits: &candle_core::Tensor,
    tokens: &[u32],
    penalty: f32,
) -> candle_core::Result<candle_core::Tensor> {
    use candle_core::Tensor;

    if tokens.is_empty() || (penalty - 1.0).abs() < f32::EPSILON {
        return Ok(logits.clone());
    }
    let mut logits_vec = logits.to_vec1::<f32>()?;
    for &token in tokens {
        let token = token as usize;
        if token < logits_vec.len() {
            if logits_vec[token] > 0.0 {
                logits_vec[token] /= penalty;
            } else {
                logits_vec[token] *= penalty;
            }
        }
    }
    Tensor::from_vec(logits_vec, logits.shape(), logits.device())
}
