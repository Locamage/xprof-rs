use crate::hlo::Shape;
use crate::hlo::xla::{LayoutProto, SplitConfigProto, TileProto};

const F32: i32 = 11;
const S32: i32 = 4;
const U32: i32 = 8;
const U16: i32 = 7;
const COMBINE_DIMENSION: i64 = i64::MIN;

fn to_string(layout: LayoutProto) -> String {
    let dimensions = vec![2; layout.minor_to_major.len().max(1)];
    let text = Shape { element_type: F32, dimensions, layout: Some(Box::new(layout)), ..Default::default() }.text(true);
    text[text.find('{').unwrap()..].to_string()
}

fn minor_to_major(minor_to_major: &[i64]) -> LayoutProto {
    LayoutProto { minor_to_major: minor_to_major.to_vec(), ..Default::default() }
}

fn tiles(tiles: &[&[i64]]) -> Vec<TileProto> {
    tiles.iter().map(|dimensions| TileProto { dimensions: dimensions.to_vec() }).collect()
}

#[test]
fn to_string_for_empty() {
    assert_eq!(to_string(LayoutProto::default()), "{}");
}

#[test]
fn to_string_for_minor_to_major_only() {
    assert_eq!(to_string(minor_to_major(&[1, 2, 0])), "{1,2,0}");
}

#[test]
fn to_string_for_tiles() {
    assert_eq!(to_string(LayoutProto { tiles: tiles(&[&[42, 123], &[4, 5]]), ..minor_to_major(&[3, 2, 1, 0]) }), "{3,2,1,0:T(42,123)(4,5)}");
}

#[test]
fn to_string_for_tile_with_combined_dimensions() {
    assert_eq!(to_string(LayoutProto { tiles: tiles(&[&[COMBINE_DIMENSION, COMBINE_DIMENSION, 42, 123]]), ..minor_to_major(&[3, 2, 1, 0]) }), "{3,2,1,0:T(*,*,42,123)}");
}

#[test]
fn to_string_for_tail_padding_alignment() {
    assert_eq!(to_string(LayoutProto { tail_padding_alignment_in_elements: 100, ..minor_to_major(&[3, 2, 1, 0]) }), "{3,2,1,0:L(100)}");
}

#[test]
fn to_string_for_index_primitive_type() {
    assert_eq!(to_string(LayoutProto { index_primitive_type: U32, ..minor_to_major(&[3, 2, 1, 0]) }), "{3,2,1,0:#(u32)}");
}

#[test]
fn to_string_for_pointer_primitive_type() {
    assert_eq!(to_string(LayoutProto { pointer_primitive_type: U16, ..minor_to_major(&[3, 2, 1, 0]) }), "{3,2,1,0:*(u16)}");
}

#[test]
fn to_string_for_element_size() {
    assert_eq!(to_string(LayoutProto { element_size_in_bits: 42, ..minor_to_major(&[3, 2, 1, 0]) }), "{3,2,1,0:E(42)}");
}

#[test]
fn to_string_for_memory_space() {
    assert_eq!(to_string(LayoutProto { memory_space: 3, ..minor_to_major(&[3, 2, 1, 0]) }), "{3,2,1,0:S(3)}");
}

#[test]
fn to_string_for_split_configs() {
    let split_configs = vec![SplitConfigProto { dimension: 0, split_indices: vec![3] }, SplitConfigProto { dimension: 1, split_indices: vec![0, 4] }];
    assert_eq!(to_string(LayoutProto { split_configs, ..minor_to_major(&[0, 1]) }), "{0,1:SC(0:3)(1:0,4)}");
}

#[test]
fn to_string_for_physical_shape() {
    let physical_shape = Shape { element_type: S32, dimensions: vec![10, 20], layout: Some(Box::new(minor_to_major(&[1, 0]))), ..Default::default() };
    assert_eq!(to_string(LayoutProto { physical_shape: Some(Box::new(physical_shape)), ..minor_to_major(&[0, 1]) }), "{0,1:P(s32[10,20]{1,0})}");
}

#[test]
fn to_string_for_dynamic_shape_metadata_prefix_bytes() {
    assert_eq!(to_string(LayoutProto { dynamic_shape_metadata_prefix_bytes: 123, ..minor_to_major(&[0, 1]) }), "{0,1:M(123)}");
}

#[test]
fn to_string_for_mutiple_properties() {
    let layout = LayoutProto { tiles: tiles(&[&[42, 123], &[4, 5]]), tail_padding_alignment_in_elements: 100, element_size_in_bits: 42, ..minor_to_major(&[3, 2, 1, 0]) };
    assert_eq!(to_string(layout), "{3,2,1,0:T(42,123)(4,5)L(100)E(42)}");
}

#[test]
fn stream_out() {
    assert_eq!(to_string(LayoutProto { tiles: tiles(&[&[7, 8]]), ..minor_to_major(&[0, 1]) }), "{0,1:T(7,8)}");
    assert_eq!(to_string(minor_to_major(&[0, 1, 2])), "{0,1,2}");
}
