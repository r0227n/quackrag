/// Gemma 3 チャットテンプレート
pub fn format_chat_prompt(user_message: &str) -> String {
    format!(
        "<bos><start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
        user_message
    )
}
