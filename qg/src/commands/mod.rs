mod ask;
mod chat;
mod db;
mod index;
mod model;
mod search;

use crate::cli::Command;
use crate::config::AppConfig;

pub async fn dispatch(command: Command, config: &AppConfig) -> anyhow::Result<()> {
    match command {
        Command::Index {
            paths,
            chunk_size,
            chunk_overlap,
        } => index::run(config, paths, chunk_size, chunk_overlap).await,
        Command::Search { query, top_k } => search::run(config, &query, top_k).await,
        Command::Ask {
            question,
            top_k,
            max_tokens,
            no_stream,
        } => ask::run(config, &question, top_k, max_tokens, no_stream).await,
        Command::Chat { top_k, max_tokens } => chat::run(config, top_k, max_tokens).await,
        Command::Db { action } => db::run(config, action).await,
        Command::Model { action } => model::run(config, action),
    }
}
