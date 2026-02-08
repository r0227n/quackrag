use std::io::{self, BufRead, Write};

use futures::StreamExt;
use quackrag_core::pipeline::{generator, retriever};
use quackrag_core::prompt::PromptBuilder;
use quackrag_core::traits::llm::LlmBackend;

use crate::config::AppConfig;
use crate::context::AppContext;

pub async fn run(config: &AppConfig, top_k: usize, max_tokens: usize) -> anyhow::Result<()> {
    let _ = max_tokens;
    let ctx = AppContext::full(config)?;
    let embedding = ctx.embedding()?;
    let llm = ctx.llm()?;
    let prompt_builder = PromptBuilder::default();

    println!("QuackRAG interactive chat. Type /quit to exit, /help for commands.\n");

    let stdin = io::stdin();
    loop {
        print!("qg> ");
        io::stdout().flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }

        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        match input {
            "/quit" | "/exit" => {
                println!("Bye!");
                break;
            }
            "/help" => {
                println!("Commands:");
                println!("  /quit, /exit  - Exit chat");
                println!("  /help         - Show this help");
                println!("  <question>    - Ask a question using RAG");
                continue;
            }
            _ => {}
        }

        let results = retriever::retrieve(input, top_k, embedding, &ctx.store).await?;

        if results.is_empty() {
            println!("No relevant documents found.\n");
            continue;
        }

        let prompt = generator::build_prompt(input, &results, &prompt_builder);
        let mut stream = llm.generate_stream(prompt);
        while let Some(token) = stream.next().await {
            match token {
                Ok(t) => {
                    print!("{t}");
                    io::stdout().flush()?;
                }
                Err(e) => {
                    eprintln!("\nError: {e}");
                    break;
                }
            }
        }
        println!("\n");
    }

    Ok(())
}
