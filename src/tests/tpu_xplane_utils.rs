use super::xspace::XSpace;
use crate::derive::is_tensor_core;
use crate::group::{is_sparse_core, ordinal};

#[test]
fn get_tensor_core_x_planes_from_x_space() {
    let names = ["/device:TPU:0", "/device:TPU:1", "/device:TPU:2Postfix"];
    assert_eq!(names.into_iter().filter(|name| is_tensor_core(name)).collect::<Vec<_>>(), ["/device:TPU:0", "/device:TPU:1"]);
}

#[test]
fn get_mutable_tensor_core_x_planes_from_x_space() {
    let mut space = XSpace::default();
    for name in ["/device:TPU:0", "/device:TPU:1", "/device:TPU:2Postfix"] {
        space.plane(name);
    }
    let (_, planes) = space.parsed();
    assert_eq!(planes.iter().filter(|plane| is_tensor_core(&plane.name)).map(|plane| &*plane.name).collect::<Vec<_>>(), ["/device:TPU:0", "/device:TPU:1"]);
}

#[test]
fn get_tensor_core_id_from_plane_name() {
    let mut space = XSpace::default();
    space.plane("/device:TPU:0").named_line(1, "XLA Modules");
    space.plane("/device:TPU:0").event(1, "module", 0, 1, &[]);
    let (_, planes) = space.parsed();
    assert_eq!(ordinal(&planes[0], &planes[0].lines[0]), Some(0));
}

#[test]
fn is_not_tensor_core_plane_name() {
    assert!(!is_tensor_core("/metadata:0"));
}

#[test]
fn is_not_tensor_core_plane_name_with_prefix() {
    assert!(!is_tensor_core("/prefix/device:TPU:0"));
}

#[test]
fn get_sparse_core_planes_from_x_space() {
    let names = ["/device:TPU:0", "/device:TPU:1", "/device:TPU:0 SparseCore 0", "/device:TPU:0 SparseCore 1", "/device:TPU:0 SparseCore Test 0"];
    assert_eq!(names.into_iter().filter(|name| is_tensor_core(name)).collect::<Vec<_>>(), ["/device:TPU:0", "/device:TPU:1"]);
    assert_eq!(names.map(is_sparse_core), [false, false, true, true, true]);
}
