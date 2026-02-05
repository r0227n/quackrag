# Gemma 3 使い方ガイド

このドキュメントでは、RustでGemma 3モデルを使用してテキスト生成を行う方法を説明します。

## 1. 概要

### llama-cpp-2クレートについて

[llama-cpp-2](https://crates.io/crates/llama-cpp-2) は、[llama.cpp](https://github.com/ggerganov/llama.cpp) のRustバインディングです。llama.cppは高性能なC++推論エンジンで、GGUF形式のモデルを効率的に実行できます。

### Gemma 3モデル

Gemma 3はGoogleが開発した軽量かつ高性能な言語モデルです。

**対応モデルサイズ:**
- **1B** - 最軽量、組み込み/モバイル向け
- **4B** - バランス型、一般的なタスク向け

**量子化形式:**
- Q4_K_M - 4bit量子化（推奨、品質とサイズのバランスが良い）
- Q8_0 - 8bit量子化（より高品質だがサイズ大）

## 2. セットアップ

### 依存関係

`Cargo.toml` に以下を追加します：

```toml
[dependencies]
llama-cpp-2 = "0.1"
hf-hub = "0.4"
anyhow = "1.0"
```

### ビルド要件

llama-cpp-2をビルドするには **CMake** が必要です：

```bash
# macOS
brew install cmake

# Ubuntu/Debian
sudo apt install cmake

# Windows
winget install Kitware.CMake
```

## 3. モデルのダウンロードとロード

### HuggingFace Hubからの自動ダウンロード

`hf-hub` クレートを使用して、HuggingFaceからモデルを自動ダウンロードできます：

```rust
use hf_hub::api::sync::ApiBuilder;

let model_path = ApiBuilder::new()
    .with_progress(true)  // ダウンロード進捗を表示
    .build()
    .context("Unable to create HuggingFace API")?
    .model("ggml-org/gemma-3-1b-it-GGUF".to_string())
    .get("gemma-3-1b-it-Q4_K_M.gguf")
    .context("Unable to download model")?;
```

ダウンロードしたモデルは `~/.cache/huggingface/` にキャッシュされ、次回以降は再ダウンロード不要です。

### LlamaBackendの初期化

```rust
use llama_cpp_2::llama_backend::LlamaBackend;

let backend = LlamaBackend::init()?;
```

### モデルのロード

```rust
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;

let model_params = LlamaModelParams::default();
let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
    .context("Unable to load model")?;
```

### Apple Silicon Metal GPUの自動有効化

llama-cpp-2はApple Silicon (M1/M2/M3/M4) のMetal GPUを自動検出して有効化します。特別な設定は不要です。GPUを使用することで推論速度が大幅に向上します。

### コンテキストの作成

```rust
use std::num::NonZeroU32;
use llama_cpp_2::context::params::LlamaContextParams;

let ctx_params = LlamaContextParams::default()
    .with_n_ctx(Some(NonZeroU32::new(2048).unwrap()));  // コンテキスト長

let mut ctx = model
    .new_context(&backend, ctx_params)
    .context("Unable to create context")?;
```

## 4. 推論の実行

### Gemma 3チャットテンプレート

Gemma 3は特定のチャットテンプレート形式を必要とします：

```
<bos><start_of_turn>user
{プロンプト}<end_of_turn>
<start_of_turn>model
```

Rustでの実装：

```rust
let formatted_prompt = format!(
    "<bos><start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
    prompt
);
```

### トークナイズ

```rust
use llama_cpp_2::model::AddBos;

let tokens_list = model
    .str_to_token(&formatted_prompt, AddBos::Never)  // BOSはテンプレートに含まれているためNever
    .context("Failed to tokenize prompt")?;
```

**注意:** `AddBos::Never` を使用するのは、チャットテンプレートに既に `<bos>` が含まれているためです。

### バッチ処理

```rust
use llama_cpp_2::llama_batch::LlamaBatch;

let mut batch = LlamaBatch::new(512, 1);  // バッチサイズ512、シーケンス数1

let last_index = (tokens_list.len() - 1) as i32;
for (i, token) in (0_i32..).zip(tokens_list.iter()) {
    let is_last = i == last_index;
    batch.add(*token, i, &[0], is_last)?;
}

// プロンプトをデコード
ctx.decode(&mut batch).context("Failed to decode prompt")?;
```

### サンプリング設定

```rust
use llama_cpp_2::sampling::LlamaSampler;

let mut sampler = LlamaSampler::chain_simple([
    LlamaSampler::dist(1234),   // シード値（再現性のため）
    LlamaSampler::greedy(),     // 貪欲法サンプリング
]);
```

## 5. ストリーミング出力

トークン単位でリアルタイム出力を行う実装：

```rust
use std::io::Write;
use llama_cpp_2::model::Special;

while n_cur <= n_len {
    // トークンをサンプリング
    let token = sampler.sample(&ctx, batch.n_tokens() - 1);
    sampler.accept(token);

    // 生成終了チェック
    if model.is_eog_token(token) {
        break;
    }

    // トークンを文字列に変換して出力
    let output = model.token_to_str(token, Special::Tokenize)?;
    print!("{}", output);
    std::io::stdout().flush()?;  // バッファをフラッシュして即座に表示

    // 次のバッチを準備
    batch.clear();
    batch.add(token, n_cur, &[0], true)?;

    n_cur += 1;
    ctx.decode(&mut batch).context("Failed to decode")?;
}
```

**ポイント:**
- `stdout().flush()` を呼ぶことで、バッファリングを回避してリアルタイム表示
- `is_eog_token()` で生成終了（End of Generation）を検出
- 各生成後に `ctx.clear_kv_cache()` でKVキャッシュをクリア

## 6. パフォーマンス参考値

Apple M4 Pro での実測値：

| 項目 | 値 |
|------|-----|
| モデルロード時間 | 約15秒 |
| 推論速度 | 約120-130 tok/s |
| メモリ使用量 | 約800MB（モデル+KVキャッシュ） |

**注意:** これらの値は環境により異なります。初回ダウンロード時はネットワーク速度に依存します。

## 7. 完全なコード例

`src/main.rs`:

```rust
use std::io::Write;
use std::num::NonZeroU32;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::ggml_time_us;

fn main() -> Result<()> {
    println!("=== QuackRAG LLM Test (llama.cpp) ===\n");

    // Step 1: モデルダウンロード/ロード時間計測
    println!("Downloading/Loading Gemma 3 1B model (GGUF Q4_K_M)...");
    let load_start = Instant::now();

    // Initialize llama.cpp backend
    let backend = LlamaBackend::init()?;

    // Download model from HuggingFace
    let model_path = ApiBuilder::new()
        .with_progress(true)
        .build()
        .context("Unable to create HuggingFace API")?
        .model("ggml-org/gemma-3-1b-it-GGUF".to_string())
        .get("gemma-3-1b-it-Q4_K_M.gguf")
        .context("Unable to download model")?;

    println!("Model path: {:?}", model_path);

    // Load model (CPU only for now)
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .context("Unable to load model")?;

    let load_time = load_start.elapsed();
    println!("Model loaded in {:.2}s\n", load_time.as_secs_f64());

    // Create context
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(2048).unwrap()));
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .context("Unable to create context")?;

    // Step 2: 英語テスト
    println!("--- English Test ---");
    generate_response(&model, &mut ctx, "What is Rust programming language? Answer briefly.")?;
    println!();

    // Step 3: 日本語テスト
    println!("--- Japanese Test ---");
    generate_response(&model, &mut ctx, "Rustとは何ですか？簡潔に答えてください。")?;
    println!();

    // Step 4: RAG説明テスト
    println!("--- RAG Explanation Test ---");
    generate_response(&model, &mut ctx, "Explain what RAG (Retrieval-Augmented Generation) is in 3 sentences.")?;

    println!("\n=== Test Complete ===");
    Ok(())
}

fn generate_response(model: &LlamaModel, ctx: &mut llama_cpp_2::context::LlamaContext, prompt: &str) -> Result<()> {
    let max_tokens = 256;

    // Format prompt with chat template
    let formatted_prompt = format!(
        "<bos><start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
        prompt
    );

    // Tokenize
    let tokens_list = model
        .str_to_token(&formatted_prompt, AddBos::Never)
        .context("Failed to tokenize prompt")?;

    // Create batch
    let mut batch = LlamaBatch::new(512, 1);
    let last_index = (tokens_list.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens_list.iter()) {
        let is_last = i == last_index;
        batch.add(*token, i, &[0], is_last)?;
    }

    // Decode prompt
    ctx.decode(&mut batch).context("Failed to decode prompt")?;

    // Generation loop
    let mut n_cur = batch.n_tokens();
    let n_len = tokens_list.len() as i32 + max_tokens;
    let mut n_decode = 0;

    let t_start = ggml_time_us();

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::dist(1234),
        LlamaSampler::greedy(),
    ]);

    print!("Response: ");
    std::io::stdout().flush()?;

    while n_cur <= n_len {
        let token = sampler.sample(ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        // Check for end of generation
        if model.is_eog_token(token) {
            break;
        }

        // Convert token to string and print
        let output = model.token_to_str(token, Special::Tokenize)?;
        print!("{}", output);
        std::io::stdout().flush()?;

        // Prepare next batch
        batch.clear();
        batch.add(token, n_cur, &[0], true)?;

        n_cur += 1;
        ctx.decode(&mut batch).context("Failed to decode")?;
        n_decode += 1;
    }

    let t_end = ggml_time_us();
    let duration = Duration::from_micros((t_end - t_start) as u64);

    println!(
        "\n[Stats: {:.2}s, {} tokens, {:.1} tok/s]",
        duration.as_secs_f32(),
        n_decode,
        n_decode as f32 / duration.as_secs_f32()
    );

    // Clear KV cache for next generation
    ctx.clear_kv_cache();

    Ok(())
}
```

## 8. トラブルシューティング

### CMakeが見つからない

```
error: could not find cmake
```

→ CMakeをインストールしてください（セットアップセクション参照）

### モデルのダウンロードが遅い

→ HuggingFaceのミラーを使用するか、事前にダウンロードしてローカルパスを指定

### メモリ不足

→ より小さいモデル（1B）や、より高い量子化（Q4_K_M）を使用

## 9. コマンドライン引数

`quackrag-llm-test` は以下のコマンドライン引数をサポートしています：

### 使用可能な引数

```bash
# ヘルプを表示
cargo run --release -- --help

# デフォルトテスト実行（プロンプト未指定、HuggingFaceから自動ダウンロード）
cargo run --release

# カスタムプロンプトで推論実行
cargo run --release -- "Explain quantum computing in simple terms"

# カスタムプロンプト + 長い応答
cargo run --release -- "Write a short story about a robot" --max-tokens 1024

# ローカルGGUFファイル + カスタムプロンプト
cargo run --release -- --model-path ./models/gemma-3-1b-it-Q4_K_M.gguf "What is Rust?"

# HuggingFaceトークンを使用（プライベートリポジトリアクセス）
cargo run --release -- --hf-token hf_xxxxxxxxxxxxx "Summarize this text"

# 環境変数でHFトークンを設定
export HF_TOKEN=hf_xxxxxxxxxxxxx
cargo run --release -- "What is machine learning?"

# HuggingFaceから異なるモデルをダウンロード
cargo run --release -- --model-repo ggml-org/gemma-3-4b-it-GGUF --model-file gemma-3-4b-it-Q4_K_M.gguf

# カスタムキャッシュディレクトリを指定
cargo run --release -- --cache-dir /path/to/cache --max-tokens 512
```

### 引数詳細

| 引数 | デフォルト値 | 説明 |
|------|-------------|------|
| `PROMPT` | なし（位置引数、オプション） | モデルに送信するプロンプトテキスト。未指定の場合はデフォルトテストを実行。 |
| `--max-tokens` | `256` | 生成する最大トークン数。長い応答が必要な場合は増やしてください。 |
| `--hf-token` | なし（オプション） | HuggingFace APIトークン（プライベートリポジトリアクセス用）。環境変数 `HF_TOKEN` でも設定可能。 |
| `--model-path` | なし（オプション） | ローカルGGUFファイルへのパス。指定するとHuggingFaceからのダウンロードをスキップします。 |
| `--cache-dir` | `~/.cache/huggingface/` | HuggingFaceモデルのキャッシュディレクトリ（`--model-path` 未指定時のみ使用）。 |
| `--model-repo` | `ggml-org/gemma-3-1b-it-GGUF` | HuggingFaceモデルリポジトリID（`--model-path` 未指定時のみ使用）。 |
| `--model-file` | `gemma-3-1b-it-Q4_K_M.gguf` | リポジトリ内のGGUFファイル名（`--model-path` 未指定時のみ使用）。 |

### 実用例

#### カスタムプロンプトで質問

```bash
cargo run --release -- "What are the benefits of Rust programming language?"
```

#### より長い応答を生成

```bash
cargo run --release -- "Explain machine learning" --max-tokens 1024
```

#### ローカルにダウンロード済みのモデルを使用

```bash
# HuggingFaceキャッシュから直接パスを指定
cargo run --release -- --model-path ~/.cache/huggingface/hub/models--ggml-org--gemma-3-1b-it-GGUF/snapshots/f9c28bcd.../gemma-3-1b-it-Q4_K_M.gguf "Summarize quantum physics"

# または、別の場所に保存したモデルを使用
cargo run --release -- --model-path ./local-models/gemma-3-1b-it-Q4_K_M.gguf "What is AI?"
```

#### プライベートHuggingFaceリポジトリを使用

```bash
# コマンドライン引数でトークン指定
cargo run --release -- --hf-token hf_xxxxxxxxxxxxx --model-repo your-org/private-model "Hello"

# または環境変数で設定
export HF_TOKEN=hf_xxxxxxxxxxxxx
cargo run --release -- --model-repo your-org/private-model "Hello"
```

#### Gemma 3 4Bモデルを試す（HuggingFaceから自動ダウンロード）

```bash
cargo run --release -- --model-repo ggml-org/gemma-3-4b-it-GGUF --model-file gemma-3-4b-it-Q4_K_M.gguf "Explain RAG"
```

#### ローカル4Bモデルで長い応答を生成

```bash
cargo run --release -- --model-path ~/models/gemma-3-4b-it-Q4_K_M.gguf --max-tokens 1024 "Write a poem about AI"
```

#### カスタムキャッシュディレクトリを使用

```bash
cargo run --release -- --cache-dir /Volumes/ExternalSSD/ai-models "What is Rust?"
```

#### デフォルトテストを実行（プロンプト未指定）

```bash
cargo run --release
```

### 注意事項

- プロンプトが位置引数なので、他のオプション引数の前または後に指定できます
- プロンプトにスペースが含まれる場合は、ダブルクォートで囲んでください
- `--hf-token` は環境変数 `HF_TOKEN` からも読み取られます（コマンドライン引数が優先）
- `--model-path` と `--cache-dir` でチルダ (`~`) は自動的にホームディレクトリに展開されます
- `--model-path` を指定すると、`--model-repo`、`--model-file`、`--cache-dir` は無視されます
- ローカルファイル使用時は、HuggingFaceからのダウンロードが発生しないため起動が高速です
- モデルを変更した場合、初回実行時にダウンロードが発生します（`--model-path` 未使用時）
- `--max-tokens` を大きくすると生成時間が長くなりますが、より詳細な応答が得られます
- プロンプトを指定しない場合は、デフォルトの3つのテスト（英語・日本語・RAG説明）を実行します

## 参考リンク

- [llama-cpp-2 crate](https://crates.io/crates/llama-cpp-2)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Gemma 3 GGUF models](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF)
- [hf-hub crate](https://crates.io/crates/hf-hub)
