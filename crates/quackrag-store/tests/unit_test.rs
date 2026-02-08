use quackrag_store::config::{DistanceMetric, DuckDbConfig, StorageMode};
use std::path::PathBuf;

#[test]
fn config_default_values() {
    let config = DuckDbConfig::default();
    assert_eq!(config.embedding_dimension, 384);
    assert!(config.use_vss);
    assert_eq!(config.distance_metric, DistanceMetric::Cosine);
    assert!(matches!(config.storage_mode, StorageMode::InMemory));
}

#[test]
fn config_builder() {
    let config = DuckDbConfig::default()
        .with_storage_mode(StorageMode::File {
            path: PathBuf::from("/tmp/test.db"),
        })
        .with_embedding_dimension(768)
        .with_use_vss(false)
        .with_distance_metric(DistanceMetric::InnerProduct);

    assert_eq!(config.embedding_dimension, 768);
    assert!(!config.use_vss);
    assert_eq!(config.distance_metric, DistanceMetric::InnerProduct);
}

#[test]
fn distance_metric_sql_function_names() {
    assert_eq!(
        DistanceMetric::Cosine.sql_function(),
        "array_cosine_distance"
    );
    assert_eq!(DistanceMetric::L2.sql_function(), "array_distance");
    assert_eq!(
        DistanceMetric::InnerProduct.sql_function(),
        "array_inner_product"
    );
}

#[test]
fn distance_metric_hnsw_names() {
    assert_eq!(DistanceMetric::Cosine.hnsw_metric(), "cosine");
    assert_eq!(DistanceMetric::L2.hnsw_metric(), "l2sq");
    assert_eq!(DistanceMetric::InnerProduct.hnsw_metric(), "ip");
}

#[cfg(feature = "duckdb")]
#[test]
fn duckdb_store_is_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<quackrag_store::DuckDbStore>();
    assert_sync::<quackrag_store::DuckDbStore>();
}
