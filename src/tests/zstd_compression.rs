fn compressed(original: &str) -> Vec<u8> {
    zstd::bulk::compress(original.as_bytes(), 1).unwrap()
}

fn decompressed(compressed: &[u8]) -> std::io::Result<String> {
    String::from_utf8(zstd::decode_all(compressed)?).map_err(std::io::Error::other)
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
