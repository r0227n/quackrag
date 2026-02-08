use std::io::Write;

use futures::StreamExt;
use quackrag_core::pipeline::{generator, retriever};
use quackrag_core::prompt::PromptBuilder;
use quackrag_core::traits::llm::LlmBackend;

use crate::config::AppConfig;
use crate::context::AppContext;

pub async fn run(
    config: &AppConfig,
    question: &str,
    top_k: usize,
    max_tokens: usize,
    no_stream: bool,
) -> anyhow::Result<()> {
    // CLI引数でmax_tokensを上書き
    let mut config = config.clone();
    config.llm.max_tokens = max_tokens;
    let ctx = AppContext::full(&config)?;
    let embedding = ctx.embedding()?;
    let llm = ctx.llm()?;

    let results = retriever::retrieve(question, top_k, embedding, &ctx.store).await?;

    if results.is_empty() {
        println!("No relevant documents found. Please index some documents first.");
        return Ok(());
    }

    let prompt_builder = PromptBuilder::default();

    if no_stream {
        let answer = generator::generate(question, &results, &prompt_builder, llm).await?;
        println!("{answer}");
    } else {
        let prompt = generator::build_prompt(question, &results, &prompt_builder);
        let mut stream = llm.generate_stream(prompt);
        while let Some(token) = stream.next().await {
            let token = token?;
            print!("{token}");
            std::io::stdout().flush()?;
        }
        println!();
    }

    // ソース情報を表示
    println!("\n---\nSources:");
    for result in &results {
        println!(
            "  - {} (chunk {}, score: {:.4})",
            result.document.source, result.document.chunk_index, result.score
        );
    }

    Ok(())
}
