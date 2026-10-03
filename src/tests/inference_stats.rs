use super::xspace::{V, XSpace};
use crate::group::{Metadata, Relatives};
use crate::inference_profile::generate;

const GROUP_ID: i64 = 100;

type TensorEvent<'a> = (&'a str, i64, i64, &'a [(&'a str, V)]);

fn tensor_patterns(tensor_events: &[TensorEvent]) -> Vec<String> {
    let mut space = XSpace::default();
    let host = space.host();
    host.event(0, "SessionRun", 1000, 5000, &[("group_id", GROUP_ID.into())]);
    for &(name, offset, duration, stats) in tensor_events {
        let stats: Vec<(&str, V)> = [("group_id", V::from(GROUP_ID))].into_iter().chain(stats.iter().cloned()).collect();
        host.event(0, name, offset, duration, &stats);
    }
    let (map, planes) = space.parsed();
    let metadata = Metadata::from([(GROUP_ID, Relatives::default())]);
    generate(&planes, &map, &metadata, 0).tensor_patterns
}

#[test]
fn generate_tensor_pattern_with_tensor_shapes() {
    let patterns = tensor_patterns(&[("Linearize", 1200, 800, &[("shape", "[1,2,3]".into()), ("layout", "{0,1,2}".into())])]);
    assert_eq!(patterns, ["Linearize [1,2,3] {0,1,2}"]);
}

#[test]
fn generate_tensor_pattern_with_dimensions_and_type() {
    let patterns = tensor_patterns(&[("Linearize", 1200, 800, &[("dims", "[10,20]".into()), ("type", "F32".into()), ("layout", "{1,0}".into())])]);
    assert_eq!(patterns, ["Linearize f32[10,20] {1,0}"]);
}

#[test]
fn generate_tensor_pattern_with_dimensions_only() {
    let patterns = tensor_patterns(&[("Linearize", 1200, 800, &[("dims", "[4,8]".into()), ("layout", "{0,1}".into())])]);
    assert_eq!(patterns, ["Linearize [4,8] {0,1}"]);
}

#[test]
fn generate_tensor_pattern_multiple_sorted() {
    let patterns = tensor_patterns(&[
        ("Linearize", 1200, 800, &[("dims", "[1,2]".into()), ("type", "s32".into()), ("layout", "{1,0}".into())]),
        ("Delinearize", 2000, 500, &[("shape", "[3,4]".into()), ("layout", "{0,1}".into())]),
    ]);
    assert_eq!(patterns, ["Delinearize [3,4] {0,1}<br>Linearize s32[1,2] {1,0}"]);
}

#[test]
fn generate_tensor_pattern_missing_layout_returns_empty() {
    assert!(tensor_patterns(&[("Linearize", 1200, 800, &[("dims", "[1,2]".into())])]).is_empty());
}
