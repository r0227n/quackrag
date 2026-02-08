// quackrag-llm の簡単な動作確認例
//
// 実行方法:
// cargo run --example llm_example --features quackrag-llm/candle
//
// カスタムプロンプトで実行:
// cargo run --example llm_example --features quackrag-llm/candle -- "Your question here"

use quackrag_core::traits::llm::LlmBackend;
use quackrag_llm::{CandleBackend, CandleConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ロギング設定
    tracing_subscriber::fmt::init();

    println!("=== QuackRAG LLM Example (Candle Backend) ===\n");

    // コマンドライン引数から質問を取得
    let question = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "What is Rust programming language?".to_string());

    println!("Initializing Gemma 3 (1B Q4_K_M)...");
    let backend = CandleBackend::new(CandleConfig::default())?;
    println!("Model loaded: {}\n", backend.model_name());

    // Gemma 3 チャットテンプレートでフォーマット
    let prompt = format!(
        "<bos><start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
        question
    );

    println!("Question: {}", question);
    println!("Generating response...\n");

    let response = backend.generate(&prompt).await?;

    println!("Answer: {}\n", response);
    println!("=== Done ===");

    Ok(())
}
