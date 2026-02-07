pub mod templates;

use templates::DEFAULT_RAG_TEMPLATE;

pub struct PromptBuilder {
    template: String,
}

impl Default for PromptBuilder {
    fn default() -> Self {
        Self {
            template: DEFAULT_RAG_TEMPLATE.to_string(),
        }
    }
}

impl PromptBuilder {
    pub fn new(template: impl Into<String>) -> Self {
        Self {
            template: template.into(),
        }
    }

    pub fn build(&self, context: &str, question: &str) -> String {
        self.template
            .replace("{context}", context)
            .replace("{question}", question)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_template() {
        let builder = PromptBuilder::default();
        let result = builder.build("some context", "what is RAG?");
        assert!(result.contains("some context"));
        assert!(result.contains("what is RAG?"));
        assert!(result.contains("Answer:"));
    }

    #[test]
    fn custom_template() {
        let builder = PromptBuilder::new("Context: {context}\nQ: {question}");
        let result = builder.build("ctx", "q?");
        assert_eq!(result, "Context: ctx\nQ: q?");
    }

    #[test]
    fn empty_inputs() {
        let builder = PromptBuilder::default();
        let result = builder.build("", "");
        assert!(result.contains("Context:\n"));
        assert!(result.contains("Question: \n"));
    }
}
