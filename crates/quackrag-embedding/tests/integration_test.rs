use quackrag_core::traits::embedding::EmbeddingModel;
use quackrag_embedding::{CandleEmbedding, EmbeddingConfig};

/// 統合テスト: 単一テキストのembedding生成
///
/// このテストは実際にモデルをダウンロードして推論を実行するため、
/// デフォルトでは無視されます。明示的に実行する場合:
///
/// ```bash
/// cargo test -p quackrag-embedding --test integration_test -- --ignored --nocapture
/// ```
#[tokio::test]
#[ignore = "Requires model download (~90MB), run with --ignored"]
async fn test_single_embedding() {
    let config = EmbeddingConfig::default();
    let model = CandleEmbedding::new(config).expect("Failed to create CandleEmbedding");

    assert_eq!(model.dimension(), 384);

    let embedding = model.embed("Hello, world!").await.expect("Failed to embed");

    // 384次元であること
    assert_eq!(embedding.len(), 384);

    // L2正規化されていること（ノルム ≈ 1.0）
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "Expected L2 norm ~1.0, got {}",
        norm
    );

    println!("Embedding dimension: {}", embedding.len());
    println!("L2 norm: {:.6}", norm);
    println!("First 5 values: {:?}", &embedding[..5]);
}

/// バッチembeddingの検証
#[tokio::test]
#[ignore = "Requires model download (~90MB), run with --ignored"]
async fn test_batch_embedding() {
    let config = EmbeddingConfig::default();
    let model = CandleEmbedding::new(config).expect("Failed to create CandleEmbedding");

    let texts = vec![
        "The cat sat on the mat.",
        "A dog played in the park.",
        "Machine learning is fascinating.",
    ];

    let embeddings = model
        .embed_batch(&texts)
        .await
        .expect("Failed to batch embed");

    assert_eq!(embeddings.len(), 3);

    for (i, emb) in embeddings.iter().enumerate() {
        assert_eq!(emb.len(), 384, "Embedding {} has wrong dimension", i);

        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Embedding {} L2 norm: {} (expected ~1.0)",
            i,
            norm
        );
    }

    println!(
        "All {} embeddings are 384-dim and L2-normalized",
        embeddings.len()
    );
}

/// 意味的類似度テスト: 類似テキストの距離が近い
#[tokio::test]
#[ignore = "Requires model download (~90MB), run with --ignored"]
async fn test_semantic_similarity() {
    let config = EmbeddingConfig::default();
    let model = CandleEmbedding::new(config).expect("Failed to create CandleEmbedding");

    let texts = vec![
        "I love programming in Rust.",     // 0: Rustプログラミング
        "Rust is my favorite language.",   // 1: Rustプログラミング（類似）
        "The weather is beautiful today.", // 2: 天気（無関係）
    ];

    let embeddings = model
        .embed_batch(&texts)
        .await
        .expect("Failed to batch embed");

    // コサイン類似度（正規化済みなのでドット積）
    let sim_01 = cosine_similarity(&embeddings[0], &embeddings[1]);
    let sim_02 = cosine_similarity(&embeddings[0], &embeddings[2]);

    println!("Similarity (Rust/Rust): {:.4}", sim_01);
    println!("Similarity (Rust/Weather): {:.4}", sim_02);

    // 類似テキスト同士の方が類似度が高い
    assert!(
        sim_01 > sim_02,
        "Expected similar texts to have higher similarity: {} > {}",
        sim_01,
        sim_02
    );
}

/// 日本語テキスト対応確認
#[tokio::test]
#[ignore = "Requires model download (~90MB), run with --ignored"]
async fn test_japanese_text() {
    let config = EmbeddingConfig::default();
    let model = CandleEmbedding::new(config).expect("Failed to create CandleEmbedding");

    let embedding = model
        .embed("これは日本語のテストです。")
        .await
        .expect("Failed to embed Japanese text");

    assert_eq!(embedding.len(), 384);

    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "Expected L2 norm ~1.0, got {}",
        norm
    );

    println!("Japanese embedding OK, L2 norm: {:.6}", norm);
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
