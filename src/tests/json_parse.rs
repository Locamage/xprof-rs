use crate::cli::json::J;

struct Random(u64);

impl Random {
    fn below(&mut self, limit: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) % limit as u64) as usize
    }
}

fn value(random: &mut Random, depth: usize, out: &mut String) {
    match random.below(if depth == 0 { 4 } else { 7 }) {
        0 => out.push_str(["null", "true", "-0", "12345678901234567890123", "1.5e300", "-7"][random.below(6)]),
        1..=3 => {
            let text: String = (0..random.below(12)).map(|_| ["\\", "\"", "\\\\\\", "\"\\\"", ",", "[", "}", ":", "é", "\n", "a"][random.below(11)]).collect();
            out.push_str(&serde_json::to_string(&text).unwrap());
        }
        4 | 5 => {
            out.push('[');
            for index in 0..random.below(8) {
                out.push_str(if index > 0 { " , " } else { "\n" });
                value(random, depth - 1, out);
            }
            out.push(']');
        }
        _ => {
            out.push_str("{ ");
            for index in 0..random.below(8) {
                out.push_str(if index > 0 { ",\t" } else { "" });
                out.push_str(&serde_json::to_string(["k", "\\\"", "k"][random.below(3)]).unwrap());
                out.push_str(" : ");
                value(random, depth - 1, out);
            }
            out.push('}');
        }
    }
}

fn document(random: &mut Random) -> String {
    let mut out = String::from(" {\"rows\": [");
    for index in 0..4000 {
        out.push_str(if index > 0 { "," } else { "" });
        value(random, 5, &mut out);
    }
    out.push_str("], \"rows\": {\"big\": [");
    for index in 0..4000 {
        out.push_str(if index > 0 { "," } else { "" });
        value(random, 4, &mut out);
    }
    out.push_str("]}}\n");
    out
}

#[test]
fn split_parse_matches_serde() {
    let mut random = Random(7);
    for _ in 0..4 {
        let text = document(&mut random);
        assert!(text.len() > 1 << 20);
        let expected: Option<J> = serde_json::from_str(&text).ok();
        assert!(expected.is_some());
        assert_eq!(J::parse(&text), expected);
        let bytes = text.as_bytes();
        for _ in 0..40 {
            let at = random.below(bytes.len());
            let mut broken = bytes.to_vec();
            match random.below(3) {
                0 => _ = broken.remove(at),
                1 => broken.insert(at, b"\"\\,]}"[random.below(5)]),
                _ => broken[at] = b"\"\\,[{"[random.below(5)],
            }
            let Ok(broken) = String::from_utf8(broken) else { continue };
            assert_eq!(J::parse(&broken), serde_json::from_str(&broken).ok());
        }
    }
}
