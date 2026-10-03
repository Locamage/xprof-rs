use crate::counter_ids::{names, names_json};

#[test]
fn test_get_all_counters_success() {
    let header = "
    #ifndef THIRD_PARTY_XPROF_UTILS_TPU_COUNTER_IDS_H_
    #define THIRD_PARTY_XPROF_UTILS_TPU_COUNTER_IDS_H_
    namespace xprof {
    enum TpuCounterIdsTpu7x : uint64_t {
      // NOLINTBEGIN
      TPU_COUNTER_ID_FOO = 12345,
      // Some comment
      TPU_COUNTER_ID_BAR =
          67890,
      TPU_COUNTER_ID_HEX = 0x1234,
      TPU_COUNTER_ID_MULTI_EQ = 1 = 2,
      // NOLINTEND
    };
    }
    #endif
    ";
    assert_eq!(names_json(header), r#"[{"name": "tpu_counter_id_foo", "val": 12345}, {"name": "tpu_counter_id_bar", "val": 67890}, {"name": "tpu_counter_id_hex", "val": 4660}]"#);
}

#[test]
fn test_get_all_counters_no_match() {
    assert_eq!(names_json("some file without enum"), "[]");
}

#[test]
fn test_get_all_counters_empty_enum() {
    assert_eq!(names_json("enum TpuCounterIdsTpu7x : uint64_t {};"), "[]");
    assert_eq!(names_json("enum TpuCounterIdsTpu7x : uint64_t { \n };"), "[]");
}

#[test]
fn test_get_all_counters_invalid_device_type() {
    assert!(names("../invalid").is_none());
}

#[test]
fn test_get_all_counters_security_path_separators() {
    assert!(names("v7x/../etc/passwd").is_none());
}
