# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**quackrag** is a local RAG (Retrieval-Augmented Generation) system combining Gemma 3 and DuckDB. The project is designed as a modular Rust workspace with cross-platform support (iOS/Android/Desktop/CLI).

### Current Status

- **Phase 0 (LLM verification)**: Completed - Gemma 3 works via llama-cpp-2
- **Phase 1 (Workspace foundation)**: Not started - workspace structure is planned but not yet created
- Only `examples/quackrag-llm-test/` exists as a working proof-of-concept

## Planned Architecture

```
quackrag/
├── crates/
│   ├── quackrag-core/       # Core traits (EmbeddingModel, LlmBackend, VectorStore)
│   ├── quackrag-embedding/  # Candle-based embedding (all-MiniLM-L6-v2, 384-dim)
│   ├── quackrag-llm/        # LLM inference (Gemma 3 via llama-cpp-2)
│   ├── quackrag-store/      # Vector store (DuckDB + vss extension)
│   └── quackrag-ffi/        # UniFFI bindings for iOS/Android
└── qg/                      # CLI application
```

## Build Commands

### LLM Test Example (currently the only buildable component)

```bash
cd examples/quackrag-llm-test

# Build
cargo build --release

# Run default tests (English, Japanese, RAG explanation)
cargo run --release

# Run with custom prompt
cargo run --release -- "Your prompt here"

# Run with local GGUF model
cargo run --release -- --model-path /path/to/model.gguf "Your prompt"

# Run with different HuggingFace model
cargo run --release -- --model-repo ggml-org/gemma-3-4b-it-GGUF --model-file gemma-3-4b-it-Q4_K_M.gguf
```

### Build Requirements

- Rust 1.82+
- CMake (required for llama-cpp-2)

## Key Technical Details

### Gemma 3 Chat Template

```
<bos><start_of_turn>user
{prompt}<end_of_turn>
<start_of_turn>model
```

When tokenizing, use `AddBos::Never` since the template already includes `<bos>`.

### Core Traits (Planned)

```rust
// EmbeddingModel: text → 384-dim vector
// LlmBackend: prompt → response (sync + streaming)
// VectorStore: insert/search/delete with HNSW index
```

### DuckDB Schema (Planned)

- `documents` table: id, source, content, chunk_index, metadata
- `embeddings` table: id, document_id, embedding FLOAT[384]
- HNSW index via vss extension for cosine similarity search

## Dependencies

Key crates used:
- `llama-cpp-2` - llama.cpp Rust bindings for LLM inference
- `hf-hub` - HuggingFace model downloads
- `duckdb` (planned) - Vector storage with vss extension
- `candle-*` (planned) - Embedding model inference
- `uniffi` (planned) - FFI for mobile platforms

## Performance Reference (Apple M4 Pro)

- Model load: ~15s
- Inference: ~120-130 tok/s
- Memory: ~800MB (Gemma 3 1B Q4_K_M)
