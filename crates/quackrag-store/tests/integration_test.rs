#![cfg(feature = "duckdb")]

use quackrag_core::traits::store::VectorStore;
use quackrag_core::types::document::Document;
use quackrag_store::config::DuckDbConfig;
use quackrag_store::DuckDbStore;

fn create_test_store(dim: usize) -> DuckDbStore {
    let config = DuckDbConfig::default()
        .with_embedding_dimension(dim)
        .with_use_vss(false);
    DuckDbStore::new(config).expect("Failed to create test store")
}

fn make_embedding(dim: usize, value: f32) -> Vec<f32> {
    vec![value; dim]
}

#[tokio::test]
#[ignore]
async fn insert_and_count() {
    let store = create_test_store(4);
    assert_eq!(store.count().await.unwrap(), 0);

    let doc = Document::new("test.txt", "Hello world", 0);
    store.insert(&doc, &make_embedding(4, 0.5)).await.unwrap();

    assert_eq!(store.count().await.unwrap(), 1);
}

#[tokio::test]
#[ignore]
async fn insert_and_delete() {
    let store = create_test_store(4);

    let doc = Document::new("test.txt", "Hello world", 0);
    let doc_id = doc.id.clone();
    store.insert(&doc, &make_embedding(4, 0.5)).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 1);

    store.delete(&doc_id).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 0);
}

#[tokio::test]
#[ignore]
async fn search_returns_similar_documents() {
    let store = create_test_store(4);

    let doc1 = Document::new("a.txt", "Similar content", 0);
    let doc2 = Document::new("b.txt", "Different content", 0);
    let doc3 = Document::new("c.txt", "Very similar content", 0);

    store.insert(&doc1, &[0.9, 0.1, 0.0, 0.0]).await.unwrap();
    store.insert(&doc2, &[0.0, 0.0, 0.9, 0.1]).await.unwrap();
    store.insert(&doc3, &[0.85, 0.15, 0.0, 0.0]).await.unwrap();

    let results = store.search(&[0.9, 0.1, 0.0, 0.0], 2).await.unwrap();
    assert_eq!(results.len(), 2);
    // 最も類似するドキュメントが先頭に来る
    assert_eq!(results[0].document.source, "a.txt");
}

#[tokio::test]
#[ignore]
async fn clear_removes_all() {
    let store = create_test_store(4);

    let doc1 = Document::new("a.txt", "Content A", 0);
    let doc2 = Document::new("b.txt", "Content B", 0);

    store.insert(&doc1, &make_embedding(4, 0.1)).await.unwrap();
    store.insert(&doc2, &make_embedding(4, 0.2)).await.unwrap();
    assert_eq!(store.count().await.unwrap(), 2);

    store.clear().await.unwrap();
    assert_eq!(store.count().await.unwrap(), 0);
}

#[tokio::test]
#[ignore]
async fn dimension_mismatch_returns_error() {
    let store = create_test_store(4);

    let doc = Document::new("test.txt", "Hello", 0);
    let result = store.insert(&doc, &make_embedding(8, 0.5)).await;
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("dimension mismatch"));
}
