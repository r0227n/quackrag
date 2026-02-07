pub trait ChunkStrategy: Send + Sync {
    fn chunk(&self, text: &str) -> Vec<String>;
}

pub struct FixedSizeChunker {
    pub size: usize,
    pub overlap: usize,
}

impl Default for FixedSizeChunker {
    fn default() -> Self {
        Self {
            size: 512,
            overlap: 64,
        }
    }
}

impl FixedSizeChunker {
    pub fn new(size: usize, overlap: usize) -> Self {
        assert!(size > 0, "chunk size must be greater than 0");
        Self { size, overlap }
    }
}

impl ChunkStrategy for FixedSizeChunker {
    fn chunk(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return vec![];
        }

        let chars: Vec<char> = text.chars().collect();
        let total = chars.len();

        if total <= self.size {
            return vec![text.to_string()];
        }

        let step = self.size.saturating_sub(self.overlap).max(1);
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < total {
            let end = (start + self.size).min(total);
            let chunk: String = chars[start..end].iter().collect();
            chunks.push(chunk);

            if end == total {
                break;
            }

            start += step;
        }

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string() {
        let chunker = FixedSizeChunker::default();
        let result = chunker.chunk("");
        assert!(result.is_empty());
    }

    #[test]
    fn short_text() {
        let chunker = FixedSizeChunker::new(100, 10);
        let result = chunker.chunk("Hello, world!");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "Hello, world!");
    }

    #[test]
    fn exact_size() {
        let chunker = FixedSizeChunker::new(5, 0);
        let result = chunker.chunk("abcde");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "abcde");
    }

    #[test]
    fn overlap_chunks() {
        let chunker = FixedSizeChunker::new(5, 2);
        let result = chunker.chunk("abcdefghij");
        // step = 5-2 = 3
        // chunk0: [0..5] = "abcde"
        // chunk1: [3..8] = "defgh"
        // chunk2: [6..10] = "ghij"
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], "abcde");
        assert_eq!(result[1], "defgh");
        assert_eq!(result[2], "ghij");
    }

    #[test]
    fn no_overlap() {
        let chunker = FixedSizeChunker::new(3, 0);
        let result = chunker.chunk("abcdefghi");
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], "abc");
        assert_eq!(result[1], "def");
        assert_eq!(result[2], "ghi");
    }

    #[test]
    fn japanese_multibyte() {
        let chunker = FixedSizeChunker::new(3, 1);
        let text = "こんにちは世界";
        let result = chunker.chunk(text);
        // step = 3-1 = 2
        // chunk0: [0..3] = "こんに"
        // chunk1: [2..5] = "にちは"
        // chunk2: [4..7] = "は世界"
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], "こんに");
        assert_eq!(result[1], "にちは");
        assert_eq!(result[2], "は世界");
    }

    #[test]
    fn emoji() {
        let chunker = FixedSizeChunker::new(2, 0);
        let text = "🦆🔍🤖💡";
        let result = chunker.chunk(text);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "🦆🔍");
        assert_eq!(result[1], "🤖💡");
    }

    #[test]
    fn default_values() {
        let chunker = FixedSizeChunker::default();
        assert_eq!(chunker.size, 512);
        assert_eq!(chunker.overlap, 64);
    }

    #[test]
    #[should_panic(expected = "chunk size must be greater than 0")]
    fn zero_size_validation() {
        // size=0 の場合はパニックする
        let _chunker = FixedSizeChunker::new(0, 0);
    }
}
