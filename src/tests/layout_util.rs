use crate::hlo::Shape;
use crate::hlo::xla::{LayoutProto, TileProto};

const PRED: i32 = 1;
const F32: i32 = 11;
const BF16: i32 = 16;
const COMBINE_DIMENSION: i64 = i64::MIN;

fn dense(element_type: i32, dimensions: &[i64], minor_to_major: &[i64]) -> Shape {
    let layout = LayoutProto { minor_to_major: minor_to_major.to_vec(), ..Default::default() };
    Shape { element_type, dimensions: dimensions.to_vec(), layout: Some(Box::new(layout)), ..Default::default() }
}

fn set_tiles(shape: &mut Shape, tiles: &[&[i64]]) {
    shape.layout.as_mut().unwrap().tiles = tiles.iter().map(|dimensions| TileProto { dimensions: dimensions.to_vec() }).collect();
}

#[test]
fn human_string_with_tiling() {
    let mut shape = dense(F32, &[2, 3, 4], &[0, 1, 2]);
    assert_eq!(shape.text(true), "f32[2,3,4]{0,1,2}");
    set_tiles(&mut shape, &[&[512, 1024]]);
    assert_eq!(shape.text(true), "f32[2,3,4]{0,1,2:T(512,1024)}");
    set_tiles(&mut shape, &[&[512]]);
    assert_eq!(shape.text(true), "f32[2,3,4]{0,1,2:T(512)}");
    shape = dense(BF16, &[2, 3, 4], &[1, 2, 0]);
    set_tiles(&mut shape, &[&[16, 256], &[2, 1]]);
    assert_eq!(shape.text(true), "bf16[2,3,4]{1,2,0:T(16,256)(2,1)}");
    shape = dense(PRED, &[8, 8, 8], &[0, 2, 1]);
    set_tiles(&mut shape, &[&[8, 128]]);
    assert_eq!(shape.text(true), "pred[8,8,8]{0,2,1:T(8,128)}");
    set_tiles(&mut shape, &[&[8, 128]]);
    shape.layout.as_mut().unwrap().element_size_in_bits = 32;
    assert_eq!(shape.text(true), "pred[8,8,8]{0,2,1:T(8,128)E(32)}");
    set_tiles(&mut shape, &[]);
    assert_eq!(shape.text(true), "pred[8,8,8]{0,2,1:E(32)}");
    shape = dense(BF16, &[2, 3, 1004], &[2, 1, 0]);
    set_tiles(&mut shape, &[&[2, COMBINE_DIMENSION, 128]]);
    assert_eq!(shape.text(true), "bf16[2,3,1004]{2,1,0:T(2,*,128)}");
    shape = dense(BF16, &[8, 2, 3, 1004], &[3, 2, 1, 0]);
    set_tiles(&mut shape, &[&[2, COMBINE_DIMENSION, COMBINE_DIMENSION, 128]]);
    assert_eq!(shape.text(true), "bf16[8,2,3,1004]{3,2,1,0:T(2,*,*,128)}");
}
