use crate::hlo::Shape;
use crate::hlo::xla::{LayoutProto, TileProto};

const F32: i32 = 11;
const U32: i32 = 8;
const S32: i32 = 4;
const OPAQUE: i32 = 14;
const TOKEN: i32 = 17;
const TUPLE: i32 = 13;
const BUFFER: i32 = 34;

fn layout(minor_to_major: &[i64]) -> Option<Box<LayoutProto>> {
    Some(Box::new(LayoutProto { minor_to_major: minor_to_major.to_vec(), ..Default::default() }))
}

fn array(element_type: i32, dimensions: &[i64], minor_to_major: &[i64]) -> Shape {
    Shape { element_type, dimensions: dimensions.to_vec(), is_dynamic_dimension: vec![false; dimensions.len()], layout: layout(minor_to_major), ..Default::default() }
}

fn tuple(elements: Vec<Shape>) -> Shape {
    Shape { element_type: TUPLE, tuple_shapes: elements, ..Default::default() }
}

#[test]
fn shape_to_string() {
    let opaque = Shape { element_type: OPAQUE, ..Default::default() };
    let token = Shape { element_type: TOKEN, ..Default::default() };
    let scalar = array(F32, &[], &[]);
    let mut scalar_with_tile = array(F32, &[], &[]);
    scalar_with_tile.layout.as_mut().unwrap().tiles.push(TileProto { dimensions: vec![256] });
    let matrix = array(U32, &[1, 2], &[1, 0]);
    let matrix2 = array(S32, &[3, 4], &[0, 1]);
    let matrix_buffer = Shape { element_type: BUFFER, tuple_shapes: vec![array(S32, &[3, 4], &[1, 0])], ..Default::default() };
    let tuple_ = tuple(vec![opaque.clone(), scalar.clone(), matrix.clone(), matrix2.clone()]);
    let nested_tuple = tuple(vec![tuple_.clone(), matrix.clone(), token.clone()]);
    assert_eq!(opaque.text(false), "opaque[]");
    assert_eq!(token.text(false), "token[]");
    assert_eq!(scalar.text(false), "f32[]");
    assert_eq!(matrix.text(false), "u32[1,2]");
    assert_eq!(matrix2.text(false), "s32[3,4]");
    assert_eq!(tuple_.text(false), "(opaque[], f32[], u32[1,2], s32[3,4])");
    assert_eq!(nested_tuple.text(false), "((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2], token[])");
    assert_eq!(opaque.text(true), "opaque[]");
    assert_eq!(scalar.text(true), "f32[]");
    assert_eq!(scalar_with_tile.text(true), "f32[]{:T(256)}");
    assert_eq!(matrix.text(true), "u32[1,2]{1,0}");
    assert_eq!(matrix2.text(true), "s32[3,4]{0,1}");
    assert_eq!(matrix_buffer.text(true), "b(s32[3,4]{1,0})");
    assert_eq!(tuple_.text(true), "(opaque[], f32[], u32[1,2]{1,0}, s32[3,4]{0,1})");
    assert_eq!(nested_tuple.text(true), "((opaque[], f32[], u32[1,2]{1,0}, s32[3,4]{0,1}), u32[1,2]{1,0}, token[])");
}

#[test]
fn dynamic_shape_to_string() {
    let mut array_shape = Shape { is_dynamic_dimension: vec![true, false, true], ..array(F32, &[23, 44, 55], &[2, 1, 0]) };
    assert_eq!(array_shape.text(false), "f32[<=23,44,<=55]");
    array_shape.is_dynamic_dimension[2] = false;
    assert_eq!(array_shape.text(false), "f32[<=23,44,55]");
    let unbounded = Shape { is_dynamic_dimension: vec![true, false], ..array(F32, &[i64::MIN, 784], &[1, 0]) };
    assert_eq!(unbounded.text(false), "f32[?,784]");
}
