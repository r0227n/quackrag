pub mod config;

#[cfg(feature = "duckdb")]
mod duckdb;

#[cfg(feature = "duckdb")]
pub use crate::duckdb::DuckDbStore;
