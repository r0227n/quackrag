# quackrag - 仕様書

## 概要

**quackrag** は、Gemma 3 と DuckDB を組み合わせたローカルRAG（Retrieval-Augmented Generation）システムです。

### 設計思想

- **モジュラー設計**: コア機能を独立したcrateとして分離
- **クロスプラットフォーム**: iOS/Android/Desktop/CLI で共通のコアを使用
- **ローカルファースト**: すべての処理をデバイス上で完結
- **DuckDB統一**: 全プラットフォームでDuckDBをVectorStoreとして使用

---

## システムアーキテクチャ

### クレート構成

```
quackrag/
├── quackrag-core/       # コアロジック（プラットフォーム非依存）
├── quackrag-embedding/  # Embeddingエンジン（モバイル対応）
├── quackrag-llm/        # LLM推論エンジン（Gemma 3）
├── quackrag-store/      # ベクトルストア（DuckDB統一）
├── qg/                  # CLI（デスクトップ用）
└── quackrag-ffi/        # FFI bindings（iOS/Android用）
```

### レイヤー図

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Application Layer                                │
├─────────────┬─────────────┬─────────────┬─────────────┬────────────────┤
│   iOS App   │ Android App │  Flutter    │  Desktop    │   CLI (qg)     │
│   (Swift)   │  (Kotlin)   │   (Dart)    │   (Rust)    │    (Rust)      │
├─────────────┴─────────────┴─────────────┴─────────────┴────────────────┤
│                         FFI / UniFFI Layer                             │
│                        (quackrag-ffi)                                  │
├────────────────────────────────────────────────────────────────────────┤
│                         Core Layer (Pure Rust)                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      quackrag-core                               │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │  │
│  │  │    RAG      │  │   Chunker   │  │   Config    │              │  │
│  │  │  Pipeline   │  │             │  │             │              │  │
│  │  └──────┬──────┘  └─────────────┘  └─────────────┘              │  │
│  │         │                                                        │  │
│  │         ▼                                                        │  │
│  │  ┌─────────────────────────────────────────────────────────┐    │  │
│  │  │              Trait Abstractions                          │    │  │
│  │  │  ┌───────────────┐  ┌───────────────┐  ┌─────────────┐  │    │  │
│  │  │  │ EmbeddingModel│  │  LlmBackend   │  │ VectorStore │  │    │  │
│  │  │  │    (trait)    │  │    (trait)    │  │   (trait)   │  │    │  │
│  │  │  └───────┬───────┘  └───────┬───────┘  └──────┬──────┘  │    │  │
│  │  │          │                  │                 │          │    │  │
│  │  └──────────┼──────────────────┼─────────────────┼──────────┘    │  │
│  └─────────────┼──────────────────┼─────────────────┼────────────────┘  │
│                │                  │                 │                   │
│  ┌─────────────▼───────┐  ┌──────▼────────┐  ┌─────▼─────────────────┐ │
│  │ quackrag-embedding  │  │ quackrag-llm  │  │   quackrag-store      │ │
│  │                     │  │               │  │                       │ │
│  │ • Candle backend    │  │ • Gemma 3     │  │ • DuckDB (全Platform) │ │
│  │ • ONNX backend      │  │ • GGUF support│  │ • vss拡張 (HNSW)      │ │
│  │ • CoreML (iOS)      │  │ • CoreML(iOS) │  │ • In-memory option    │ │
│  │ • NNAPI (Android)   │  │ • NNAPI(Andr) │  │                       │ │
│  └─────────────────────┘  └───────────────┘  └───────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

### データフロー

```
┌──────────────────────────────────────────────────────────────────┐
│                      RAG Pipeline Flow                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    INDEXING PHASE                        │   │
│   │                                                          │   │
│   │   Document ──▶ Chunker ──▶ EmbeddingModel ──▶ DuckDB    │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    RETRIEVAL PHASE                       │   │
│   │                                                          │   │
│   │   Query ──▶ EmbeddingModel ──▶ DuckDB(HNSW) ──▶ Documents│   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    GENERATION PHASE                      │   │
│   │                                                          │   │
│   │   (Query + Documents) ──▶ PromptBuilder ──▶ LlmBackend   │   │
│   │                                      │                   │   │
│   │                                      ▼                   │   │
│   │                                   Answer                 │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## RAG (Retrieval-Augmented Generation) とは

### 基本概念

RAGは以下の3ステップで動作します：

1. **Indexing（インデックス作成）**
   - ドキュメントをチャンク（小さな断片）に分割
   - 各チャンクをEmbedding（ベクトル）に変換
   - ベクトルをデータベースに保存

2. **Retrieval（検索）**
   - ユーザーのクエリをベクトルに変換
   - データベースから類似ベクトルを検索
   - 関連するドキュメントチャンクを取得

3. **Generation（生成）**
   - 取得したドキュメントをコンテキストとしてLLMに渡す
   - LLMがコンテキストを参照して回答を生成

### Embedding（埋め込み）とは

テキストを固定長の数値ベクトルに変換したもの。意味的に似たテキストは、ベクトル空間上で近い位置になります。

```
"犬が走る" → [0.12, -0.34, 0.56, ..., 0.89]  (384次元)
"猫が歩く" → [0.11, -0.32, 0.58, ..., 0.87]  (類似 → 距離が近い)
"車を運転" → [0.78, 0.23, -0.45, ..., 0.12]  (非類似 → 距離が遠い)
```

---

## Traitベース設計

### なぜTraitベースか？

プラットフォームごとに最適な実装を切り替え可能にするため。

```rust
// quackrag-core で定義
#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    fn dimension(&self) -> usize;
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

#[async_trait]
pub trait LlmBackend: Send + Sync {
    async fn generate(&self, prompt: &str) -> Result<String>;
    fn generate_stream(&self, prompt: &str) 
        -> Pin<Box<dyn Stream<Item = Result<String>> + Send>>;
    fn model_name(&self) -> &str;
}

#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn insert(&self, doc: &Document, embedding: &[f32]) -> Result<()>;
    async fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchResult>>;
    async fn delete(&self, id: &str) -> Result<()>;
    async fn count(&self) -> Result<usize>;
}
```

### プラットフォーム別実装

| Platform | EmbeddingModel | LlmBackend | VectorStore |
|----------|---------------|------------|-------------|
| **Desktop** | Candle | llama-cpp-2 (Gemma 3 GGUF) | DuckDB + vss |
| **iOS** | CoreML / ONNX | llama-cpp-2 (GGUF) | DuckDB + vss |
| **Android** | NNAPI / ONNX | llama-cpp-2 (GGUF) | DuckDB + vss |
| **Flutter** | Platform Channel | Platform Channel | DuckDB + vss |

### Feature Flags

```toml
# quackrag-embedding/Cargo.toml
[features]
default = ["candle"]
candle = ["candle-core", "candle-nn", "candle-transformers"]
onnx = ["ort"]

# quackrag-llm/Cargo.toml
[features]
default = ["llamacpp"]
llamacpp = ["dep:llama-cpp-2", "dep:hf-hub"]

# quackrag-store/Cargo.toml
[features]
default = ["bundled"]
bundled = ["duckdb/bundled"]
```

---

## 技術スタック

### Core Libraries

| Crate | 用途 | Platform |
|-------|------|----------|
| `quackrag-core` | RAGパイプライン、Trait定義 | All |
| `quackrag-embedding` | Embedding生成 | All |
| `quackrag-llm` | LLM推論 | All |
| `quackrag-store` | ベクトルストレージ (DuckDB) | All |
| `quackrag-ffi` | FFIバインディング | Mobile |
| `qg` | CLI | Desktop |

### Dependencies

**共通 (workspace)**
```toml
[workspace.dependencies]
anyhow = "1.0"
thiserror = "2.0"
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tracing = "0.1"
async-trait = "0.1"
uuid = { version = "1.0", features = ["v4"] }
```

**Desktop (qg CLI)**
```toml
candle-core = "0.8"
candle-transformers = "0.8"
llama-cpp-2 = "0.1"
hf-hub = "0.4"
duckdb = { version = "1.1", features = ["bundled"] }
clap = { version = "4", features = ["derive"] }
```

**Mobile (quackrag-ffi)**
```toml
uniffi = "0.28"
llama-cpp-2 = "0.1"
hf-hub = "0.4"
duckdb = { version = "1.1", features = ["bundled"] }
```

---

## CLI (qg) コマンド設計

```bash
# ヘルプ
qg --help
qg <command> --help

# ドキュメントのインデックス作成
qg index <file_or_directory>
qg index ./documents/
qg index ./readme.md --chunk-size 512

# セマンティック検索（LLMなし、類似ドキュメントのみ返す）
qg search "検索クエリ"
qg search "Rustのエラーハンドリング" -k 5

# RAGクエリ（検索 + LLM生成）
qg ask "質問文"
qg ask "このプロジェクトの目的は？"
qg ask "要約して" --context-size 3

# インタラクティブチャット
qg chat
qg chat --model gemma-3-4b

# データベース管理
qg db info          # 統計情報表示
qg db clear         # 全データ削除
qg db export        # エクスポート
qg db list          # インデックス済みファイル一覧

# モデル管理
qg model list       # 利用可能モデル一覧
qg model download   # モデルダウンロード
qg model info       # 現在のモデル情報

# 設定
qg config show      # 設定表示
qg config set <key> <value>
```

---

## データベーススキーマ（DuckDB）

全プラットフォーム共通のスキーマ：

```sql
-- ドキュメントテーブル
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR PRIMARY KEY,
    source VARCHAR NOT NULL,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Embeddingテーブル
CREATE TABLE IF NOT EXISTS embeddings (
    id VARCHAR PRIMARY KEY,
    document_id VARCHAR NOT NULL REFERENCES documents(id),
    embedding FLOAT[384],  -- all-MiniLM-L6-v2は384次元
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- HNSWインデックス（ベクトル検索用）
INSTALL vss;
LOAD vss;

CREATE INDEX IF NOT EXISTS embedding_hnsw_idx 
ON embeddings USING HNSW (embedding) 
WITH (metric = 'cosine');
```

### DuckDBを全プラットフォームで使う理由

1. **一貫性**: スキーマ・クエリが全環境で同一
2. **vss拡張**: HNSWインデックスによる高速ベクトル検索
3. **軽量**: 組み込みデータベースとして優秀
4. **モバイル対応**: iOS/Androidでも動作可能
5. **Rust対応**: duckdb-rsが成熟

---

## ディレクトリ構成（モノレポ）

```
quackrag/
├── Cargo.toml                    # Workspace定義
├── Cargo.lock
├── README.md
├── SPECIFICATION.md              # この仕様書
├── TODO.md                       # 実装TODO
├── .gitignore
│
├── crates/
│   ├── quackrag-core/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── error.rs
│   │       ├── traits/
│   │       │   ├── mod.rs
│   │       │   ├── embedding.rs
│   │       │   ├── llm.rs
│   │       │   └── store.rs
│   │       ├── types/
│   │       │   ├── mod.rs
│   │       │   ├── document.rs
│   │       │   └── search.rs
│   │       ├── chunker/
│   │       │   ├── mod.rs
│   │       │   └── strategies.rs
│   │       ├── prompt/
│   │       │   ├── mod.rs
│   │       │   └── templates.rs
│   │       └── pipeline/
│   │           ├── mod.rs
│   │           ├── indexer.rs
│   │           ├── retriever.rs
│   │           └── generator.rs
│   │
│   ├── quackrag-embedding/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       └── candle/
│   │           ├── mod.rs
│   │           └── minilm.rs
│   │
│   ├── quackrag-llm/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       └── llamacpp/
│   │           ├── mod.rs
│   │           └── gemma.rs
│   │
│   ├── quackrag-store/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       └── duckdb/
│   │           ├── mod.rs
│   │           ├── connection.rs
│   │           ├── schema.rs
│   │           └── vss.rs
│   │
│   └── quackrag-ffi/
│       ├── Cargo.toml
│       ├── build.rs
│       ├── uniffi.toml
│       └── src/
│           ├── lib.rs
│           ├── engine.rs
│           └── quackrag.udl
│
├── qg/
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs
│       └── commands/
│           ├── mod.rs
│           ├── index.rs
│           ├── search.rs
│           ├── ask.rs
│           ├── chat.rs
│           ├── db.rs
│           ├── model.rs
│           └── config.rs
│
├── bindings/
│   ├── swift/
│   │   ├── Package.swift
│   │   └── Sources/Quackrag/
│   ├── kotlin/
│   │   └── quackrag/
│   └── flutter/
│       ├── pubspec.yaml
│       └── lib/
│
├── examples/
│   ├── basic_rag.rs
│   └── llm_only.rs
│
└── tests/
    ├── integration/
    └── fixtures/
```

---

## 参考リソース

### Rust
- [The Rust Programming Language](https://doc.rust-lang.org/book/)
- [Cargo Workspaces](https://doc.rust-lang.org/book/ch14-03-cargo-workspaces.html)

### DuckDB
- [DuckDB Rust Client](https://duckdb.org/docs/api/rust)
- [DuckDB VSS Extension](https://duckdb.org/docs/extensions/vss)

### Candle (Embedding)
- [Candle GitHub](https://github.com/huggingface/candle)

### llama-cpp-2 (LLM)
- [llama-cpp-2 crate](https://crates.io/crates/llama-cpp-2)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [Gemma 3 GGUF models](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF)

### Mobile FFI
- [UniFFI Documentation](https://mozilla.github.io/uniffi-rs/)

### RAG
- [RAG概念説明](https://www.promptingguide.ai/techniques/rag)
