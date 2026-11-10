use super::hlo_fixture::{hlo_proto, module};
use crate::hlo::memory::render;

const SMALL_BUFFER: i64 = 16 * 1024;

#[test]
fn process_hlo_json_test() {
    let (data, content_type) = render(&module(&hlo_proto(r#"hlo_module { name: "test_module" }"#)), 0, SMALL_BUFFER, false).unwrap();
    assert_eq!(content_type, "application/json");
    assert!(!data.is_empty());
}

#[test]
fn process_hlo_html_test() {
    let (data, content_type) = render(&module(&hlo_proto(r#"hlo_module { name: "test_module" }"#)), 0, SMALL_BUFFER, true).unwrap();
    assert_eq!(content_type, "text/html");
    assert!(!data.is_empty());
}
