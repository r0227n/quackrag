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
    let filter = if cli.verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("warn")
    };
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let app_config = config::AppConfig::load(cli.config.as_deref(), cli.db.as_deref())?;

    if let Err(e) = commands::dispatch(cli.command, &app_config).await {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }

    Ok(())
}
