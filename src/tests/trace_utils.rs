use crate::trace::{is_tpu_core_device_name, maybe_tpu_non_core_device_name};

#[test]
fn is_tpu_core_device_name_test() {
    assert!(is_tpu_core_device_name("/device:TPU:0"));
    assert!(is_tpu_core_device_name("TensorNode"));
    assert!(is_tpu_core_device_name("TPU Core"));
    assert!(!is_tpu_core_device_name("GPU"));
    assert!(!is_tpu_core_device_name("Host Interface"));
}

#[test]
fn maybe_tpu_host_interface_device_name_test() {
    assert!(maybe_tpu_non_core_device_name("#Chip TPU v2 Host Interface"));
    assert!(!maybe_tpu_non_core_device_name("#Chip TPU v2"));
}

#[test]
fn is_tpu_hbm_device_name_test() {
    assert!(maybe_tpu_non_core_device_name("#Chip TPU v2 HBM"));
    assert!(!maybe_tpu_non_core_device_name("#Chip TPU v2"));
}

#[test]
fn is_tpu_ici_router_device_name_test() {
    assert!(maybe_tpu_non_core_device_name("#Chip TPU v2 ICI Router"));
    assert!(!maybe_tpu_non_core_device_name("#Chip TPU v2"));
}

#[test]
fn maybe_tpu_non_core_device_name_test() {
    assert!(maybe_tpu_non_core_device_name("#Chip TPU Non-Core HBM"));
    assert!(maybe_tpu_non_core_device_name("#Chip TPU Non-Core Host Interface"));
    assert!(maybe_tpu_non_core_device_name("#Chip TPU Non-Core ICI Router"));
    assert!(!maybe_tpu_non_core_device_name("TPU v2"));
    assert!(!maybe_tpu_non_core_device_name("TPU Non-Core Other"));
}
