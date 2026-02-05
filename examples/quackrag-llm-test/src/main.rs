use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::ggml_time_us;

/// QuackRAG LLM Test - Test Gemma 3 model with llama.cpp
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Prompt text to send to the model (if not specified, runs default tests)
    #[arg(value_name = "PROMPT")]
    prompt: Option<String>,

    /// Maximum number of tokens to generate
    #[arg(long, default_value_t = 256)]
    max_tokens: i32,

    /// HuggingFace API token for accessing private repositories (can also use HF_TOKEN env var)
    #[arg(long, env = "HF_TOKEN")]
    hf_token: Option<String>,

    /// Path to local GGUF model file (if specified, ignores --model-repo and --model-file)
    #[arg(long)]
    model_path: Option<String>,

    /// HuggingFace cache directory for models (used only when --model-path is not specified)
    #[arg(long, default_value = "~/.cache/huggingface/")]
    cache_dir: String,

    /// HuggingFace model repository ID (used only when --model-path is not specified)
    #[arg(long, default_value = "ggml-org/gemma-3-1b-it-GGUF")]
    model_repo: String,

    /// GGUF model file name within the repository (used only when --model-path is not specified)
    #[arg(long, default_value = "gemma-3-1b-it-Q4_K_M.gguf")]
    model_file: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== QuackRAG LLM Test (llama.cpp) ===\n");

    // Set HuggingFace token if provided
    if let Some(token) = &args.hf_token {
        std::env::set_var("HF_TOKEN", token);
    }

    // Step 1: モデルダウンロード/ロード時間計測
    println!("Downloading/Loading model...");
    let load_start = Instant::now();

    // Initialize llama.cpp backend
    let backend = LlamaBackend::init()?;

    // Determine model path: either local file or download from HuggingFace
    let model_path = if let Some(local_path) = &args.model_path {
        // Use local GGUF file directly
        println!("Using local model: {}", local_path);

        // Expand ~ if present
        let expanded_path = if local_path.starts_with('~') {
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .context("Unable to determine HOME directory")?;
            local_path.replacen('~', &home, 1)
        } else {
            local_path.clone()
        };

        PathBuf::from(expanded_path)
    } else {
        // Download from HuggingFace
        println!("Downloading/Loading from HuggingFace: {}/{}", args.model_repo, args.model_file);

        // Expand ~ in cache_dir
        let cache_dir = if args.cache_dir.starts_with('~') {
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .context("Unable to determine HOME directory")?;
            args.cache_dir.replacen('~', &home, 1)
        } else {
            args.cache_dir.clone()
        };

        ApiBuilder::new()
            .with_progress(true)
            .with_cache_dir(cache_dir.into())
            .build()
            .context("Unable to create HuggingFace API")?
            .model(args.model_repo)
            .get(&args.model_file)
            .context("Unable to download model")?
    };

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

    // Run inference based on prompt argument
    if let Some(user_prompt) = &args.prompt {
        // Single inference with user-provided prompt
        println!("--- User Prompt ---");
        generate_response(&model, &mut ctx, user_prompt, args.max_tokens)?;
    } else {
        // Run default test suite
        // Step 2: 英語テスト
        println!("--- English Test ---");
        generate_response(&model, &mut ctx, "What is Rust programming language? Answer briefly.", args.max_tokens)?;
        println!();

        // Step 3: 日本語テスト
        println!("--- Japanese Test ---");
        generate_response(&model, &mut ctx, "Rustとは何ですか？簡潔に答えてください。", args.max_tokens)?;
        println!();

        // Step 4: RAG説明テスト
        println!("--- RAG Explanation Test ---");
        generate_response(&model, &mut ctx, "Explain what RAG (Retrieval-Augmented Generation) is in 3 sentences.", args.max_tokens)?;
    }

    println!("\n=== Test Complete ===");
    Ok(())
}

fn generate_response(
    model: &LlamaModel,
    ctx: &mut llama_cpp_2::context::LlamaContext,
    prompt: &str,
    max_tokens: i32
) -> Result<()> {

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
