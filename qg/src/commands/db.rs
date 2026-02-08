use std::io::{stdin, stdout, Write};

use quackrag_core::traits::store::VectorStore;

use crate::cli::DbAction;
use crate::config::AppConfig;
use crate::context::AppContext;

pub async fn run(config: &AppConfig, action: DbAction) -> anyhow::Result<()> {
    let ctx = AppContext::store_only(config)?;

    match action {
        DbAction::Info => {
            let count = ctx.store.count().await?;
            let db_path = ctx
                .store
                .db_path()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "(in-memory)".to_string());

            println!("Database: {db_path}");
            println!("Documents: {count}");
        }
        DbAction::List => {
            let sources = ctx.store.list_sources().await?;
            if sources.is_empty() {
                println!("No indexed sources.");
            } else {
                println!("Indexed sources ({}):", sources.len());
                for source in &sources {
                    println!("  {source}");
                }
            }
        }
        DbAction::Clear => {
            // 確認プロンプト
            print!("Are you sure you want to clear all indexed data? [y/N] ");
            stdout().flush()?;
            let mut input = String::new();
            stdin().read_line(&mut input)?;
            if !matches!(input.trim().to_lowercase().as_str(), "y" | "yes") {
                println!("Aborted.");
                return Ok(());
            }
            ctx.store.clear().await?;
            println!("All indexed data cleared.");
        }
    }

    Ok(())
}
