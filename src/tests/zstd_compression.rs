use ruzstd::encoding::{CompressionLevel, compress};
use std::io::Read;

fn compressed(original: &str) -> Vec<u8> {
    let mut out = Vec::new();
    compress(original.as_bytes(), &mut out, CompressionLevel::Fastest);
    out
}

fn decompressed(compressed: &[u8]) -> std::io::Result<String> {
    let mut out = String::new();
    ruzstd::decoding::StreamingDecoder::new(compressed).map_err(std::io::Error::other)?.read_to_string(&mut out)?;
    Ok(out)
}

#[test]
fn compress_and_decompress_successfully() {
    let original =
        "Hello, Zstd! This is a test string to be compressed. Repeating it makes it more compressible... Repeating it makes it more compressible... Repeating it makes it more compressible...";
    let result = compressed(original);
    assert!(!result.is_empty());
    assert_eq!(decompressed(&result).unwrap(), original);
}

#[test]
fn empty_string() {
    assert_eq!(decompressed(&compressed("")).unwrap(), "");
}

#[test]
fn decompress_invalid_input_fails() {
    assert!(decompressed(b"Not compressed data at all").is_err());
}
