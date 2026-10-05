use super::*;

#[test]
fn numbers_at_the_limits_read_back_the_same() {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let random = (0..200_000).map(|_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        f64::from_bits(state)
    });
    let limits = [f64::MAX, f64::MIN_POSITIVE, 5e-324, 1e-310, f64::from_bits(0x000f_ffff_ffff_ffff), 1.0, 0.1, 1e15, 1e16, 123456789012345680.0, 9007199254740992.0];
    for value in limits.into_iter().chain(random).filter(|value| value.is_finite()) {
        let mut out = String::new();
        number(&mut out, value);
        assert_eq!(out.parse::<f64>().ok(), Some(value), "{out}");
    }
}
