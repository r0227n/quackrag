use quackrag_core::pipeline::retriever;

use crate::config::AppConfig;
use crate::context::AppContext;

pub async fn run(config: &AppConfig, query: &str, top_k: usize) -> anyhow::Result<()> {
    let ctx = AppContext::with_embedding(config)?;
    let embedding = ctx.embedding()?;

    let results = retriever::retrieve(query, top_k, embedding, &ctx.store).await?;

    if results.is_empty() {
        println!("No results found.");
        return Ok(());
    }

    println!("Found {} result(s):\n", results.len());
    for (i, result) in results.iter().enumerate() {
        let preview = truncate(&result.document.content, 200);
        println!(
            "  [{i}] score: {:.4}  source: {}  chunk: {}",
            result.score, result.document.source, result.document.chunk_index
        );
        println!("      {preview}");
        println!();
    }

    Ok(())
}

fn truncate(s: &str, max_chars: usize) -> String {
    let mut chars = s.chars();
    let trimmed: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{trimmed}...")
    } else {
        trimmed
    }
}
