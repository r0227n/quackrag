use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_gemma3::ModelWeights;
use tokenizers::Tokenizer;

use crate::config::CandleConfig;

/// 繰り返しペナルティをlogitsに適用する
fn apply_repeat_penalty(
    logits: &Tensor,
    tokens: &[u32],
    penalty: f32,
) -> candle_core::Result<Tensor> {
    if tokens.is_empty() || (penalty - 1.0).abs() < f32::EPSILON {
        return Ok(logits.clone());
    }
    let mut logits_vec = logits.to_vec1::<f32>()?;
    for &token in tokens {
        let token = token as usize;
        if token < logits_vec.len() {
            if logits_vec[token] > 0.0 {
                logits_vec[token] /= penalty;
            } else {
                logits_vec[token] *= penalty;
            }
        }
    }
    Tensor::from_vec(logits_vec, logits.shape(), logits.device())
}

/// 全トークンを生成してから結果を返す
pub(crate) fn generate_full(
    model: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt: &str,
    device: &Device,
    config: &CandleConfig,
) -> quackrag_core::error::Result<String> {
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;
    let prompt_tokens = encoding.get_ids().to_vec();

    if prompt_tokens.is_empty() {
        return Err(quackrag_core::error::QuackragError::Llm(
            "Empty prompt after tokenization".to_string(),
        ));
    }

    let mut logits_processor = LogitsProcessor::new(config.seed, Some(config.temperature), None);

    // プロンプトトークンを1つずつ処理
    let mut logits = None;
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        let input = Tensor::new(&[token], device)
            .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;
        logits = Some(
            model
                .forward(&input, pos)
                .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?,
        );
    }
    let logits = logits.ok_or_else(|| {
        quackrag_core::error::QuackragError::Llm("No logits generated".to_string())
    })?;

    // 最後のトークン位置のlogitsを取得
    let logits = logits
        .squeeze(0)
        .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;
    let logits_dim = logits.dims();
    let logits = if logits_dim.len() > 1 {
        logits
            .get(logits_dim[0] - 1)
            .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?
    } else {
        logits
    };

    let eos_token = tokenizer
        .token_to_id("<end_of_turn>")
        .or_else(|| tokenizer.token_to_id("</s>"))
        .or_else(|| tokenizer.token_to_id("<eos>"));

    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut all_tokens = prompt_tokens.clone();

    // 最初のトークンをサンプリング
    let penalty_tokens = get_penalty_tokens(&all_tokens, config.repeat_last_n);
    let logits = apply_repeat_penalty(&logits, &penalty_tokens, config.repeat_penalty)
        .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;
    let next_token = logits_processor
        .sample(&logits)
        .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;

    if eos_token == Some(next_token) {
        return Ok(String::new());
    }
    generated_tokens.push(next_token);
    all_tokens.push(next_token);

    // 後続トークンの生成ループ
    let mut index_pos = prompt_tokens.len();
    for _ in 1..config.max_tokens {
        let input = Tensor::new(&[next_token], device)
            .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;
        let logits = model
            .forward(&input, index_pos)
            .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;
        let logits = logits
            .squeeze(0)
            .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;
        let logits_dim = logits.dims();
        let logits = if logits_dim.len() > 1 {
            logits
                .get(logits_dim[0] - 1)
                .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?
        } else {
            logits
        };

        let penalty_tokens = get_penalty_tokens(&all_tokens, config.repeat_last_n);
        let logits = apply_repeat_penalty(&logits, &penalty_tokens, config.repeat_penalty)
            .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;

        let next_token = logits_processor
            .sample(&logits)
            .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;

        if eos_token == Some(next_token) {
            break;
        }

        generated_tokens.push(next_token);
        all_tokens.push(next_token);
        index_pos += 1;
    }

    let text = tokenizer
        .decode(&generated_tokens, true)
        .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;

    Ok(text)
}

/// トークンをストリーミングで生成し、チャネル経由で送信する
pub(crate) fn generate_streaming(
    model: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt: &str,
    device: &Device,
    config: &CandleConfig,
    token_tx: tokio::sync::mpsc::UnboundedSender<quackrag_core::error::Result<String>>,
) {
    let result = generate_streaming_inner(model, tokenizer, prompt, device, config, &token_tx);
    if let Err(e) = result {
        let _ = token_tx.send(Err(e));
    }
}

fn generate_streaming_inner(
    model: &mut ModelWeights,
    tokenizer: &Tokenizer,
    prompt: &str,
    device: &Device,
    config: &CandleConfig,
    token_tx: &tokio::sync::mpsc::UnboundedSender<quackrag_core::error::Result<String>>,
) -> quackrag_core::error::Result<()> {
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;
    let prompt_tokens = encoding.get_ids().to_vec();

    if prompt_tokens.is_empty() {
        return Err(quackrag_core::error::QuackragError::Llm(
            "Empty prompt after tokenization".to_string(),
        ));
    }

    let mut logits_processor = LogitsProcessor::new(config.seed, Some(config.temperature), None);

    // プロンプトトークンを1つずつ処理
    let mut logits = None;
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        let input = Tensor::new(&[token], device)
            .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;
        logits = Some(
            model
                .forward(&input, pos)
                .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?,
        );
    }
    let logits = logits.ok_or_else(|| {
        quackrag_core::error::QuackragError::Llm("No logits generated".to_string())
    })?;

    let logits = logits
        .squeeze(0)
        .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;
    let logits_dim = logits.dims();
    let logits = if logits_dim.len() > 1 {
        logits
            .get(logits_dim[0] - 1)
            .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?
    } else {
        logits
    };

    let eos_token = tokenizer
        .token_to_id("<end_of_turn>")
        .or_else(|| tokenizer.token_to_id("</s>"))
        .or_else(|| tokenizer.token_to_id("<eos>"));

    let mut all_tokens = prompt_tokens.clone();

    let penalty_tokens = get_penalty_tokens(&all_tokens, config.repeat_last_n);
    let logits = apply_repeat_penalty(&logits, &penalty_tokens, config.repeat_penalty)
        .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;
    let mut next_token = logits_processor
        .sample(&logits)
        .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;

    if eos_token == Some(next_token) {
        return Ok(());
    }

    let text = tokenizer
        .decode(&[next_token], true)
        .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;
    if token_tx.send(Ok(text)).is_err() {
        return Ok(());
    }
    all_tokens.push(next_token);

    let mut index_pos = prompt_tokens.len();
    for _ in 1..config.max_tokens {
        let input = Tensor::new(&[next_token], device)
            .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;
        let logits = model
            .forward(&input, index_pos)
            .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;
        let logits = logits
            .squeeze(0)
            .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;
        let logits_dim = logits.dims();
        let logits = if logits_dim.len() > 1 {
            logits
                .get(logits_dim[0] - 1)
                .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?
        } else {
            logits
        };

        let penalty_tokens = get_penalty_tokens(&all_tokens, config.repeat_last_n);
        let logits = apply_repeat_penalty(&logits, &penalty_tokens, config.repeat_penalty)
            .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;

        next_token = logits_processor
            .sample(&logits)
            .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;

        if eos_token == Some(next_token) {
            break;
        }

        let text = tokenizer
            .decode(&[next_token], true)
            .map_err(|e| quackrag_core::error::QuackragError::Llm(e.to_string()))?;
        if token_tx.send(Ok(text)).is_err() {
            return Ok(());
        }
        all_tokens.push(next_token);
        index_pos += 1;
    }

    Ok(())
}

/// 繰り返しペナルティ用の直近トークンを取得
fn get_penalty_tokens(tokens: &[u32], last_n: usize) -> Vec<u32> {
    if tokens.len() <= last_n {
        tokens.to_vec()
    } else {
        tokens[tokens.len() - last_n..].to_vec()
    }
}
