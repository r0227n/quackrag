# quackrag-llm

Candle ベースの LLM 推論エンジン (Gemma 3 対応)

## 特徴

- **Candle バックエンド**: HuggingFace の Rust ML フレームワーク
- **GGUF サポート**: quantized_gemma3 モジュールで量子化モデルをロード
- **非同期 API**: tokio ベースの async/await
- **ストリーミング**: トークン単位のリアルタイム出力
- **Send + Sync**: マルチスレッド対応 (`Arc<Mutex<ModelWeights>>`)

## 使用例

### 基本的な使用

```rust
use quackrag_core::traits::llm::LlmBackend;
use quackrag_llm::{CandleBackend, CandleConfig, format_chat_prompt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // バックエンド作成
    let backend = CandleBackend::new(CandleConfig::default())?;

    // プロンプト作成
    let prompt = format_chat_prompt("What is Rust?");

    // 推論実行
    let response = backend.generate(&prompt).await?;
    println!("Response: {}", response);

    Ok(())
}
```

### カスタム設定

```rust
use quackrag_llm::{CandleConfig, ModelSource};
use std::path::PathBuf;

// HuggingFace から異なるモデルを使用
let config = CandleConfig::default()
    .with_model_source(ModelSource::HuggingFace {
        model_repo: "ggml-org/gemma-3-4b-it-GGUF".to_string(),
        model_file: "gemma-3-4b-it-Q4_K_M.gguf".to_string(),
        tokenizer_repo: "google/gemma-3-4b-it".to_string(),
    })
    .with_max_tokens(512)
    .with_temperature(0.7);

let backend = CandleBackend::new(config)?;
```

### ストリーミング出力

```rust
use futures::StreamExt;

let prompt = format_chat_prompt("Tell me a story.");
let mut stream = backend.generate_stream(prompt);

while let Some(result) = stream.next().await {
    match result {
        Ok(token) => print!("{}", token),
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

## テスト

### ユニットテスト（高速、モデルダウンロード不要）

CI で自動実行されるテスト:

```bash
cargo test --package quackrag-llm
```

以下がテストされます:
- チャットテンプレートのフォーマット検証
- 設定のデフォルト値とビルダーパターン
- リピートペナルティロジックの動作確認

### 統合テスト（モデルダウンロード必要、約500MB）

実際の GGUF モデルを使用した推論テスト:

```bash
# 全ての統合テストを実行
cargo test --package quackrag-llm -- --ignored --nocapture

# 特定のテストのみ実行
cargo test --package quackrag-llm --test integration_test -- --ignored --nocapture test_candle_backend_generation
```

初回実行時は HuggingFace から Gemma 3 1B Q4_K_M モデル（約500MB）が自動ダウンロードされます。

## 依存関係

- `candle-core` 0.9
- `candle-transformers` 0.9 (quantized_gemma3)
- `tokenizers` 0.21
- `hf-hub` 0.4

## パフォーマンス (Apple M4 Pro)

| 項目 | 値 |
|------|-----|
| モデルロード時間 | 約15秒 (初回) |
| 推論速度 | トークン単位で処理 |
| メモリ使用量 | 約800MB (Gemma 3 1B Q4_K_M) |

## 制限事項

- 現在は CPU のみサポート
- GGUF フォーマットのみ対応
- Gemma 3 モデル専用

## TODO

- [ ] GPU サポート (CUDA, Metal)
- [ ] KV キャッシュの最適化
- [ ] バッチ処理の改善
- [ ] 他のモデルアーキテクチャ対応
