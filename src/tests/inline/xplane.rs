use super::*;

/// The varint of the byte loop, which reads at most 10 bytes.
fn slow_varint(buf: &[u8], pos: &mut usize) -> Option<u64> {
    let (mut value, mut shift) = (0u64, 0);
    while shift < 64 {
        let byte = *buf.get(*pos)?;
        *pos += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte < 0x80 {
            return Some(value);
        }
        shift += 7;
    }
    None
}

#[test]
fn a_varint_is_the_varint_of_the_byte_loop() {
    let mut state = 3u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };
    for _ in 0..200_000 {
        let length = (next() % 14) as usize;
        // Most bytes have the high bit, so that the varints have all lengths.
        let buf: Vec<u8> = (0..length)
            .map(|_| next() as u8 | if next() % 5 == 0 { 0 } else { 0x80 })
            .collect();
        for start in 0..=length {
            let (mut fast, mut slow) = (start, start);
            let value = varint(&buf, &mut fast);
            assert_eq!(value, slow_varint(&buf, &mut slow), "{buf:?} at {start}");
            if value.is_some() {
                assert_eq!(fast, slow, "{buf:?} at {start}");
            }
        }
    }
}
