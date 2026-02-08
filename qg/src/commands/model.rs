use crate::cli::ModelAction;
use crate::config::AppConfig;

pub fn run(config: &AppConfig, action: ModelAction) -> anyhow::Result<()> {
    match action {
        ModelAction::Info => {
            println!("LLM Configuration:");
            println!("  Model repo:     {}", config.llm.model_repo);
            println!("  Model file:     {}", config.llm.model_file);
            println!("  Tokenizer repo: {}", config.llm.tokenizer_repo);
            println!("  Max tokens:     {}", config.llm.max_tokens);
            println!("  Temperature:    {}", config.llm.temperature);

            println!("\nEmbedding Model:");
            println!("  Model: sentence-transformers/all-MiniLM-L6-v2");
            println!("  Dimension: 384");

            println!("\nChunker:");
            println!("  Chunk size:    {}", config.chunker.chunk_size);
            println!("  Chunk overlap: {}", config.chunker.chunk_overlap);
        }
    }

    Ok(())
}
