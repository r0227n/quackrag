use crate::config::DuckDbConfig;

/// embedding 配列を DuckDB の FLOAT[N] リテラルとしてフォーマットする
///
/// # Panics
/// 埋め込みベクトルに NaN または Infinity が含まれている場合にパニックする
pub fn format_embedding_literal(embedding: &[f32], dimension: usize) -> String {
    // 次元数の検証
    assert_eq!(
        embedding.len(),
        dimension,
        "Embedding length mismatch: expected {}, got {}",
        dimension,
        embedding.len()
    );

    // NaN/Infinityのチェック
    for (i, &v) in embedding.iter().enumerate() {
        assert!(
            v.is_finite(),
            "Embedding contains invalid value at index {}: {:?}",
            i,
            v
        );
    }

    let values: Vec<String> = embedding.iter().map(|v| format!("{v}")).collect();
    format!("[{}]::FLOAT[{dimension}]", values.join(","))
}

/// documents テーブルへの INSERT 文
pub fn build_insert_document_sql() -> &'static str {
    "INSERT INTO documents (id, source, content, chunk_index, metadata) VALUES (?, ?, ?, ?, ?)"
}

/// embeddings テーブルへの INSERT 文（embedding リテラル埋め込み）
pub fn build_insert_embedding_sql(embedding_literal: &str) -> String {
    format!(
        "INSERT INTO embeddings (id, document_id, embedding) VALUES (?, ?, {embedding_literal})"
    )
}

/// ベクトル検索クエリ
pub fn build_search_sql(embedding_literal: &str, config: &DuckDbConfig) -> String {
    let distance_fn = config.distance_metric.sql_function();
    format!(
        "SELECT d.id, d.source, d.content, d.chunk_index, d.metadata, \
         {distance_fn}(e.embedding, {embedding_literal}) AS distance \
         FROM embeddings e \
         JOIN documents d ON e.document_id = d.id \
         ORDER BY distance ASC \
         LIMIT ?"
    )
}

/// 指定ドキュメントの embedding を削除
pub fn build_delete_embeddings_sql() -> &'static str {
    "DELETE FROM embeddings WHERE document_id = ?"
}

/// ドキュメント削除
pub fn build_delete_document_sql() -> &'static str {
    "DELETE FROM documents WHERE id = ?"
}

/// ドキュメント数カウント
pub fn build_count_sql() -> &'static str {
    "SELECT COUNT(*) FROM documents"
}

/// 全データ削除（embeddings → documents の順で削除）
pub fn build_clear_sql() -> &'static str {
    "DELETE FROM embeddings; DELETE FROM documents;"
}

/// ソース一覧取得
pub fn build_list_sources_sql() -> &'static str {
    "SELECT DISTINCT source FROM documents ORDER BY source"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DistanceMetric;

    #[test]
    fn format_embedding_literal_basic() {
        let embedding = vec![0.1, 0.2, 0.3];
        let result = format_embedding_literal(&embedding, 3);
        assert_eq!(result, "[0.1,0.2,0.3]::FLOAT[3]");
    }

    #[test]
    fn format_embedding_literal_single() {
        let embedding = vec![1.0];
        let result = format_embedding_literal(&embedding, 1);
        assert_eq!(result, "[1]::FLOAT[1]");
    }

    #[test]
    fn build_search_sql_cosine() {
        let config = DuckDbConfig::default();
        let literal = "[0.1,0.2]::FLOAT[2]";
        let sql = build_search_sql(literal, &config);
        assert!(sql.contains("array_cosine_distance"));
        assert!(sql.contains(literal));
        assert!(sql.contains("ORDER BY distance ASC"));
        assert!(sql.contains("LIMIT ?"));
    }

    #[test]
    fn build_search_sql_l2() {
        let config = DuckDbConfig::default().with_distance_metric(DistanceMetric::L2);
        let literal = "[0.1,0.2]::FLOAT[2]";
        let sql = build_search_sql(literal, &config);
        assert!(sql.contains("array_distance"));
    }

    #[test]
    fn build_insert_embedding_sql_contains_literal() {
        let literal = "[0.5,0.6]::FLOAT[2]";
        let sql = build_insert_embedding_sql(literal);
        assert!(sql.contains(literal));
        assert!(sql.contains("INSERT INTO embeddings"));
    }

    #[test]
    fn static_sql_not_empty() {
        assert!(!build_insert_document_sql().is_empty());
        assert!(!build_delete_embeddings_sql().is_empty());
        assert!(!build_delete_document_sql().is_empty());
        assert!(!build_count_sql().is_empty());
        assert!(!build_clear_sql().is_empty());
    }
}
