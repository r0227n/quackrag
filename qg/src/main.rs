mod cli;
mod commands;
mod config;
mod context;

use clap::Parser;
use tracing_subscriber::EnvFilter;

use cli::Cli;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // tracing 初期化
    // RUST_LOG環境変数を優先し、なければ--verboseフラグを使用
    let filter = if cli.verbose {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug"))
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"))
    };
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let app_config = config::AppConfig::load(cli.config.as_deref(), cli.db.as_deref())?;
    commands::dispatch(cli.command, &app_config).await
}
