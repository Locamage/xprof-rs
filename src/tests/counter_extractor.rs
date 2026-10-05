use crate::tools::counter_ids::{names, names_json};

#[test]
fn test_get_all_counters_success() {
    let table = "TPU_COUNTER_ID_FOO 12345\nTPU_COUNTER_ID_BAR 67890\nTPU_COUNTER_ID_HEX 4660\n";
    assert_eq!(names_json(table), r#"[{"name": "tpu_counter_id_foo", "val": 12345}, {"name": "tpu_counter_id_bar", "val": 67890}, {"name": "tpu_counter_id_hex", "val": 4660}]"#);
}

#[test]
fn test_get_all_counters_empty_table() {
    assert_eq!(names_json(""), "[]");
    assert_eq!(names_json("\n"), "[]");
}

#[test]
fn test_get_all_counters_invalid_device_type() {
    assert!(names("../invalid").is_none());
}

#[test]
fn test_get_all_counters_security_path_separators() {
    assert!(names("v7x/../etc/passwd").is_none());
}
