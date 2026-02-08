use quackrag_core::traits::llm::LlmBackend;
use quackrag_llm::{format_chat_prompt, CandleBackend, CandleConfig};

/// 統合テスト: モデルロード＆推論実行
///
/// このテストは実際にGGUFモデルをダウンロードして推論を実行するため、
/// デフォルトでは無視されます。明示的に実行する場合:
///
/// ```bash
/// cargo test -p quackrag-llm --test integration_test -- --ignored --nocapture
/// ```
///
/// または環境変数で有効化:
///
/// ```bash
/// ENABLE_INTEGRATION_TEST=1 cargo test -p quackrag-llm --nocapture
/// ```
#[tokio::test]
#[ignore = "Requires GGUF model download (~500MB), run with --ignored or ENABLE_INTEGRATION_TEST=1"]
async fn test_candle_backend_generation() {
    // デフォルト設定でバックエンド作成 (Gemma 3 1B Q4_K_M)
    // 出力トークン数を短くしてテスト高速化
    let config = CandleConfig::default().with_max_tokens(50);
    let backend = CandleBackend::new(config).expect("Failed to create CandleBackend");

    // モデル名確認
    println!("Model: {}", backend.model_name());

    // 英語プロンプト
    let user_msg = "What is Rust? Answer briefly.";
    let prompt = format_chat_prompt(user_msg);
    println!("\n--- Test 1: English prompt ---");
    println!("User: {}", user_msg);

    let response = backend.generate(&prompt).await.expect("Failed to generate");
    println!("Assistant: {}", response);
    assert!(!response.is_empty());

    // 日本語プロンプト
    let user_msg_ja = "Rustとは何ですか？簡潔に答えてください。";
    let prompt_ja = format_chat_prompt(user_msg_ja);
    println!("\n--- Test 2: Japanese prompt ---");
    println!("User: {}", user_msg_ja);

    let response_ja = backend
        .generate(&prompt_ja)
        .await
        .expect("Failed to generate");
    println!("Assistant: {}", response_ja);
    assert!(!response_ja.is_empty());
}

/// ストリーミング出力テスト
#[tokio::test]
#[ignore = "Requires GGUF model download (~500MB), run with --ignored or ENABLE_INTEGRATION_TEST=1"]
async fn test_candle_backend_streaming() {
    use futures::StreamExt;

    let config = CandleConfig::default().with_max_tokens(30);
    let backend = CandleBackend::new(config).expect("Failed to create CandleBackend");

    let user_msg = "Count from 1 to 5.";
    let prompt = format_chat_prompt(user_msg);
    println!("\n--- Test: Streaming output ---");
    println!("User: {}", user_msg);
    print!("Assistant: ");

    let mut stream = backend.generate_stream(prompt);
    let mut tokens = Vec::new();

    while let Some(result) = stream.next().await {
        match result {
            Ok(token) => {
                print!("{}", token);
                tokens.push(token);
            }
            Err(e) => {
                eprintln!("\nError during streaming: {}", e);
                panic!("Streaming failed");
            }
        }
    }
    println!();

    assert!(!tokens.is_empty(), "No tokens received");
    println!("\nReceived {} tokens", tokens.len());
}

/// カスタム設定テスト
#[tokio::test]
#[ignore = "Requires GGUF model download (~500MB), run with --ignored or ENABLE_INTEGRATION_TEST=1"]
async fn test_custom_config() {
    use quackrag_llm::CandleConfig;

    let config = CandleConfig::default()
        .with_max_tokens(128)
        .with_temperature(0.5)
        .with_seed(42);

    let backend = CandleBackend::new(config).expect("Failed to create backend");

    let prompt = "<bos><start_of_turn>user\nHello!<end_of_turn>\n<start_of_turn>model\n";
    let response = backend.generate(prompt).await.expect("Failed to generate");

    println!("Response: {}", response);
    assert!(!response.is_empty());
}
