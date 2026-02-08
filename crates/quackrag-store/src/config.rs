use std::path::PathBuf;

/// ストレージモード
#[derive(Debug, Clone)]
pub enum StorageMode {
    /// インメモリDB（テスト・一時利用向け）
    InMemory,
    /// ファイルベースDB（永続化）
    File { path: PathBuf },
}

/// 距離メトリクス
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Cosine,
    L2,
    InnerProduct,
}

impl DistanceMetric {
    /// DuckDB SQL で使用する距離関数名
    pub fn sql_function(&self) -> &'static str {
        match self {
            DistanceMetric::Cosine => "array_cosine_distance",
            DistanceMetric::L2 => "array_distance",
            DistanceMetric::InnerProduct => "array_inner_product",
        }
    }

    /// HNSW インデックス作成時のメトリック名
    pub fn hnsw_metric(&self) -> &'static str {
        match self {
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::L2 => "l2sq",
            DistanceMetric::InnerProduct => "ip",
        }
    }
}

/// DuckDB ストア設定
#[derive(Debug, Clone)]
pub struct DuckDbConfig {
    pub storage_mode: StorageMode,
    pub embedding_dimension: usize,
    pub use_vss: bool,
    pub distance_metric: DistanceMetric,
}

impl Default for DuckDbConfig {
    fn default() -> Self {
        Self {
            storage_mode: StorageMode::InMemory,
            embedding_dimension: 384,
            use_vss: true,
            distance_metric: DistanceMetric::Cosine,
        }
    }
}

impl DuckDbConfig {
    pub fn with_storage_mode(mut self, mode: StorageMode) -> Self {
        self.storage_mode = mode;
        self
    }

    pub fn with_embedding_dimension(mut self, dim: usize) -> Self {
        self.embedding_dimension = dim;
        self
    }

    pub fn with_use_vss(mut self, use_vss: bool) -> Self {
        self.use_vss = use_vss;
        self
    }

    pub fn with_distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = DuckDbConfig::default();
        assert_eq!(config.embedding_dimension, 384);
        assert!(config.use_vss);
        assert_eq!(config.distance_metric, DistanceMetric::Cosine);
        assert!(matches!(config.storage_mode, StorageMode::InMemory));
    }

    #[test]
    fn builder_pattern() {
        let config = DuckDbConfig::default()
            .with_storage_mode(StorageMode::File {
                path: PathBuf::from("/tmp/test.db"),
            })
            .with_embedding_dimension(768)
            .with_use_vss(false)
            .with_distance_metric(DistanceMetric::L2);

        assert_eq!(config.embedding_dimension, 768);
        assert!(!config.use_vss);
        assert_eq!(config.distance_metric, DistanceMetric::L2);
        match &config.storage_mode {
            StorageMode::File { path } => assert_eq!(path, &PathBuf::from("/tmp/test.db")),
            _ => panic!("Expected File storage mode"),
        }
    }

    #[test]
    fn distance_metric_sql_functions() {
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
}
