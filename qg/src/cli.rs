use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "qg",
    version,
    about = "QuackRAG - Local RAG system with Gemma 3 and DuckDB"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    /// Path to config file
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,

    /// Path to database file
    #[arg(long, global = true)]
    pub db: Option<PathBuf>,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    pub verbose: bool,
}

#[derive(Subcommand)]
pub enum Command {
    /// Index files for RAG retrieval
    Index {
        /// Files or directories to index
        #[arg(required = true)]
        paths: Vec<PathBuf>,

        /// Chunk size in characters
        #[arg(long, default_value_t = 512)]
        chunk_size: usize,

        /// Chunk overlap in characters
        #[arg(long, default_value_t = 64)]
        chunk_overlap: usize,
    },

    /// Search indexed documents
    Search {
        /// Search query
        query: String,

        /// Number of results to return
        #[arg(short = 'k', long, default_value_t = 5)]
        top_k: usize,
    },

    /// Ask a question using RAG
    Ask {
        /// Question to ask
        question: String,

        /// Number of context documents
        #[arg(short = 'k', long, default_value_t = 5)]
        top_k: usize,

        /// Maximum tokens to generate
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,

        /// Disable streaming output
        #[arg(long)]
        no_stream: bool,
    },

    /// Interactive chat mode
    Chat {
        /// Number of context documents per turn
        #[arg(short = 'k', long, default_value_t = 5)]
        top_k: usize,

        /// Maximum tokens to generate per turn
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
    },

    /// Database management
    Db {
        #[command(subcommand)]
        action: DbAction,
    },

    /// Model information
    Model {
        #[command(subcommand)]
        action: ModelAction,
    },
}

#[derive(Subcommand)]
pub enum DbAction {
    /// Show database information
    Info,
    /// List indexed sources
    List,
    /// Clear all indexed data
    Clear,
}

#[derive(Subcommand)]
pub enum ModelAction {
    /// Show model configuration
    Info,
}
