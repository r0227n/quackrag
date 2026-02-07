use crate::error::Result;
use crate::prompt::PromptBuilder;
use crate::traits::llm::LlmBackend;
use crate::types::search::SearchResult;

fn build_context(results: &[SearchResult]) -> String {
    results
        .iter()
        .enumerate()
        .map(|(i, r)| format!("[{}] {}", i + 1, r.document.content))
        .collect::<Vec<_>>()
        .join("\n\n")
}

pub async fn generate(
    question: &str,
    results: &[SearchResult],
    prompt_builder: &PromptBuilder,
    llm: &dyn LlmBackend,
) -> Result<String> {
    let context = build_context(results);
    let prompt = prompt_builder.build(&context, question);
    llm.generate(&prompt).await
}

pub fn build_prompt(
    question: &str,
    results: &[SearchResult],
    prompt_builder: &PromptBuilder,
) -> String {
    let context = build_context(results);
    prompt_builder.build(&context, question)
}
