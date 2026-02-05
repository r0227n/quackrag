# quackrag-llm-test

Gemma 3 モデルを llama.cpp を使用してテストするための CLI ツールです。

## 概要

このツールは、Google の Gemma 3 モデル（GGUF 形式）を llama.cpp Rust バインディング (`llama-cpp-2`) で実行し、テキスト生成の動作を確認するために開発されました。HuggingFace からのモデル自動ダウンロード、ローカルファイルからの直接ロード、カスタムプロンプトの入力など、柔軟な CLI オプションを提供します。

## 必要要件

- **Rust**: 1.82 以上（推奨: 1.93+）
- **オペレーティングシステム**: macOS, Linux, Windows
- **ディスク空き容量**: モデルファイル用に最低 1GB（Gemma 3 1B Q4_K_M の場合約 700MB）

## セットアップ

### 1. Rust のインストール

```bash
# Rust がインストールされていない場合
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# バージョン確認
rustc --version  # 1.82 以上であることを確認
```

### 2. ビルド

```bash
cd /path/to/quackrag/examples/quackrag-llm-test
cargo build --release
```

初回ビルドには数分かかります。

## 使用方法

### 基本コマンド

```bash
# ヘルプを表示
cargo run --release -- --help

# デフォルトテスト実行（プロンプト未指定、HuggingFace から自動ダウンロード）
cargo run --release

# カスタムプロンプトで推論実行
cargo run --release -- "Explain quantum computing in simple terms"
```

### CLI 引数一覧

| 引数 | デフォルト値 | 説明 |
|------|-------------|------|
| `PROMPT` | なし（オプション） | モデルに送信するプロンプトテキスト。未指定の場合はデフォルトテスト（英語・日本語・RAG説明）を実行。 |
| `--max-tokens` | `256` | 生成する最大トークン数。長い応答が必要な場合は増やしてください。 |
| `--hf-token` | なし（オプション） | HuggingFace API トークン（プライベートリポジトリアクセス用）。環境変数 `HF_TOKEN` でも設定可能。 |
| `--model-path` | なし（オプション） | ローカル GGUF ファイルへのパス。指定すると HuggingFace からのダウンロードをスキップします。 |
| `--cache-dir` | `~/.cache/huggingface/` | HuggingFace モデルのキャッシュディレクトリ（`--model-path` 未指定時のみ使用）。 |
| `--model-repo` | `ggml-org/gemma-3-1b-it-GGUF` | HuggingFace モデルリポジトリ ID（`--model-path` 未指定時のみ使用）。 |
| `--model-file` | `gemma-3-1b-it-Q4_K_M.gguf` | リポジトリ内の GGUF ファイル名（`--model-path` 未指定時のみ使用）。 |

### 動作モード

#### モード 1: デフォルトテスト（プロンプト未指定）

```bash
cargo run --release
```

以下の3つのテストを自動実行します：
1. 英語テスト: "What is Rust programming language? Answer briefly."
2. 日本語テスト: "Rustとは何ですか？簡潔に答えてください。"
3. RAG説明テスト: "Explain what RAG (Retrieval-Augmented Generation) is in 3 sentences."

#### モード 2: カスタムプロンプト

```bash
cargo run --release -- "あなたのプロンプトをここに入力"
```

指定されたプロンプトで1回のみ推論を実行します。

## 使用例

### 基本的な使用例

#### デフォルトテストを実行

```bash
cargo run --release
```

#### カスタムプロンプトで質問

```bash
cargo run --release -- "What are the benefits of Rust programming language?"
```

#### より長い応答を生成

```bash
cargo run --release -- "Explain machine learning" --max-tokens 1024
```

### HuggingFace 関連

#### 環境変数でトークンを設定

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxx
cargo run --release -- "What is AI?"
```

#### コマンドライン引数でトークン指定

```bash
cargo run --release -- --hf-token hf_xxxxxxxxxxxxx "Summarize quantum physics"
```

#### プライベートリポジトリを使用

```bash
cargo run --release -- --hf-token hf_xxxxxxxxxxxxx --model-repo your-org/private-model "Hello"
```

#### Gemma 3 4B モデルを試す

```bash
cargo run --release -- --model-repo ggml-org/gemma-3-4b-it-GGUF --model-file gemma-3-4b-it-Q4_K_M.gguf "Explain RAG"
```

#### カスタムキャッシュディレクトリを使用

```bash
cargo run --release -- --cache-dir /Volumes/ExternalSSD/ai-models "What is Rust?"
```

### ローカルモデルの使用

#### HuggingFace キャッシュから直接ロード

```bash
# まずキャッシュ内のパスを確認
ls ~/.cache/huggingface/hub/models--ggml-org--gemma-3-1b-it-GGUF/snapshots/*/gemma-3-1b-it-Q4_K_M.gguf

# そのパスを使ってロード
cargo run --release -- --model-path ~/.cache/huggingface/hub/models--ggml-org--gemma-3-1b-it-GGUF/snapshots/<スナップショットID>/gemma-3-1b-it-Q4_K_M.gguf "Summarize quantum physics"
```

#### 別の場所に保存したモデルを使用

```bash
cargo run --release -- --model-path ./local-models/gemma-3-1b-it-Q4_K_M.gguf "What is AI?"
```

#### ローカル 4B モデルで長い応答を生成

```bash
cargo run --release -- --model-path ~/models/gemma-3-4b-it-Q4_K_M.gguf --max-tokens 1024 "Write a poem about AI"
```

### 複数引数の組み合わせ

```bash
# ローカルモデル + カスタムプロンプト + カスタムトークン数
cargo run --release -- --model-path ~/models/gemma-3-1b-it-Q4_K_M.gguf --max-tokens 512 "Explain quantum computing"

# プロンプトを最後に指定
cargo run --release -- --max-tokens 256 "What is AI?"

# プロンプトを最初に指定
cargo run --release -- "What is AI?" --max-tokens 256
```

## 出力例

```
=== QuackRAG LLM Test (llama.cpp) ===

Downloading/Loading model...
Downloading/Loading from HuggingFace: ggml-org/gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_K_M.gguf
Model path: "/Users/username/.cache/huggingface/hub/models--ggml-org--gemma-3-1b-it-GGUF/snapshots/.../gemma-3-1b-it-Q4_K_M.gguf"
Model loaded in 2.34s

--- User Prompt ---
Response: Rust is a systems programming language that emphasizes safety, concurrency, and performance...
[Stats: 3.21s, 87 tokens, 27.1 tok/s]

=== Test Complete ===
```

## 注意事項

- プロンプトにスペースが含まれる場合は、ダブルクォートで囲んでください
- `--hf-token` は環境変数 `HF_TOKEN` からも読み取られます（コマンドライン引数が優先）
- `--model-path` と `--cache-dir` でチルダ (`~`) は自動的にホームディレクトリに展開されます
- `--model-path` を指定すると、`--model-repo`、`--model-file`、`--cache-dir` は無視されます
- ローカルファイル使用時は、HuggingFace からのダウンロードが発生しないため起動が高速です
- モデルを変更した場合、初回実行時にダウンロードが発生します（`--model-path` 未使用時）
- `--max-tokens` を大きくすると生成時間が長くなりますが、より詳細な応答が得られます
- プロンプトを指定しない場合は、デフォルトの3つのテスト（英語・日本語・RAG説明）を実行します

## トラブルシューティング

### Rust バージョンエラー

```
error: package requires rustc 1.82 or newer
```

**解決方法**:
```bash
rustup update stable
rustup default stable
rustc --version  # 1.82 以上であることを確認
```

### Cargo.lock バージョンエラー

```
error: lock file version '4' was found, but this version of Cargo does not understand this lock file
```

**解決方法**:
```bash
rm Cargo.lock
cargo build --release
```

### HuggingFace ダウンロードエラー

プライベートリポジトリにアクセスする場合は、`--hf-token` または環境変数 `HF_TOKEN` でトークンを設定してください。

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxx
cargo run --release
```

## 参考リンク

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - C/C++ LLM 推論ライブラリ
- [llama-cpp-2](https://crates.io/crates/llama-cpp-2) - llama.cpp の Rust バインディング
- [Google Gemma](https://ai.google.dev/gemma) - Google の軽量オープンモデル
- [Gemma 3 on HuggingFace](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF) - GGUF 形式のモデルリポジトリ
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) - 量子化モデルフォーマット

## ライセンス

このプロジェクトは QuackRAG の一部です。
