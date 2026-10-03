use crate::hlo::Module;
use crate::hlo_text::{Printer, Style};
use std::borrow::Cow;

fn printed(name: &str) -> String {
    let module = Module::parse(Cow::Owned(std::fs::read(format!("{}/tests/data/hlo_printing/{name}.pb", env!("CARGO_MANIFEST_DIR"))).unwrap()));
    Printer::new(&module, Style::Long, true).module_text().unwrap()
}

#[test]
#[ignore = "differs from upstream: XProf 2.23.2 prints the backend_config string of the proto verbatim (checked over HTTP), it does not sort its keys"]
fn check_to_string_sorts_backend_config() {
    assert!(printed("check_to_string_sorts_backend_config").contains(r#"backend_config={"tuning_knobs":{"2":"2","3":"0"}}"#));
}

#[test]
fn text_hlo_roundtrip_strict() {
    assert!(printed("text_hlo_roundtrip_strict").contains(r#"ROOT %inst = f32[] parameter(0), metadata={metadata_payload="abc"}"#));
}
