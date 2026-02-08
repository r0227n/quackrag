use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use quackrag_core::chunker::FixedSizeChunker;
use quackrag_core::pipeline::indexer;

use crate::config::AppConfig;
use crate::context::AppContext;

const TEXT_EXTENSIONS: &[&str] = &[
    "txt", "md", "rs", "py", "toml", "json", "yaml", "yml", "html", "css", "js", "ts", "tsx",
    "jsx", "sh", "bash", "zsh", "c", "cpp", "h", "hpp", "go", "java", "rb", "swift", "kt", "sql",
    "xml", "csv", "log", "cfg", "ini", "env",
];

pub async fn run(
    config: &AppConfig,
    paths: Vec<PathBuf>,
    chunk_size: usize,
    chunk_overlap: usize,
) -> anyhow::Result<()> {
    let ctx = AppContext::with_embedding(config)?;
    let chunker = FixedSizeChunker::new(chunk_size, chunk_overlap);

    // ファイル一覧を収集
    let mut files = Vec::new();
    for path in &paths {
        collect_files(path, &mut files)?;
    }

    if files.is_empty() {
        println!("No text files found in the given paths.");
        return Ok(());
    }

    println!("Found {} file(s) to index.", files.len());

    let progress = ProgressBar::new(files.len() as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    let embedding = ctx.embedding()?;
    let mut total_chunks = 0usize;
    let mut indexed_files = 0usize;

    for file in &files {
        let source = file.display().to_string();
        progress.set_message(source.clone());

        let content = match std::fs::read_to_string(file) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("Skipping {}: {e}", file.display());
                progress.inc(1);
                continue;
            }
        };

        if content.trim().is_empty() {
            progress.inc(1);
            continue;
        }

        let docs =
            indexer::index_document(&source, &content, &chunker, embedding, &ctx.store).await?;
        total_chunks += docs.len();
        indexed_files += 1;
        progress.inc(1);
    }

    progress.finish_and_clear();
    println!("Indexed {indexed_files} file(s), {total_chunks} chunk(s).");

    Ok(())
}

fn collect_files(path: &Path, files: &mut Vec<PathBuf>) -> anyhow::Result<()> {
    if path.is_file() {
        if is_text_file(path) {
            files.push(path.to_path_buf());
        }
    } else if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            collect_files(&entry.path(), files)?;
        }
    }
    Ok(())
}

fn is_text_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| TEXT_EXTENSIONS.contains(&ext))
        .unwrap_or(false)
}
