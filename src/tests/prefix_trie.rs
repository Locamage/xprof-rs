use super::xspace::XSpace;
use crate::trace::Trace;

fn found(names: &[&str], prefix: &str) -> Vec<(String, usize)> {
    let mut space = XSpace::default();
    for (index, name) in names.iter().enumerate() {
        space.host().event(1, name, index as i64 * 10, 5, &[]);
    }
    let (map, planes) = space.parsed();
    let trace = Trace::build(&planes, "localhost", &map);
    let mut keys: Vec<(String, usize)> = Vec::new();
    for index in trace.search(prefix, false) {
        let event = &trace.events[index as usize];
        keys.push((trace.names[event.name as usize].to_string(), event.ts as usize / 10));
    }
    keys.sort();
    keys
}

fn with_data(prefix: &str) -> Vec<(String, usize)> {
    found(&["abc", "abcd", "abce", "a", "abcd", "def"], prefix)
}

fn expected(keys: &[(&str, usize)]) -> Vec<(String, usize)> {
    let mut keys: Vec<(String, usize)> = keys.iter().map(|&(key, id)| (key.to_string(), id)).collect();
    keys.sort();
    keys
}

#[test]
fn prefix_search_on_empty_trie() {
    assert!(found(&[], "a").is_empty());
}

#[test]
fn prefix_search_on_trie_with_single_key() {
    assert_eq!(found(&["abc"], "a"), expected(&[("abc", 0)]));
}

#[test]
fn prefix_search_on_trie_with_multiple_keys_matching_prefix() {
    assert_eq!(with_data("abc"), expected(&[("abc", 0), ("abcd", 1), ("abcd", 4), ("abce", 2)]));
}

#[test]
fn prefix_search_on_trie_with_multiple_keys_prefix_not_found() {
    assert!(with_data("abd").is_empty());
}

#[test]
fn prefix_search_on_trie_with_multiple_keys_empty_prefix() {
    assert_eq!(with_data(""), expected(&[("abc", 0), ("abcd", 1), ("abcd", 4), ("abce", 2), ("a", 3), ("def", 5)]));
}

#[test]
fn prefix_search_on_trie_with_multiple_keys_prefix_is_an_exact_match() {
    assert_eq!(with_data("abce"), expected(&[("abce", 2)]));
}
