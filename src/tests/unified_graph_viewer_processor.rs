use super::hlo_fixture::{hlo_proto, session_dir, write_module};
use crate::hlo::graph::serve;
use std::collections::HashMap;

const DUMMY_HLO: &str = r#"hlo_module {
  name: "my_module"
  entry_computation_name: "my_module"
  entry_computation_id: 1
  host_program_shape {
    result {
      element_type: F32
      dimensions: [ 1 ]
    }
  }
  computations {
    name: "my_module"
    id: 1
    root_id: 10
    instructions {
      name: "constant"
      opcode: "constant"
      id: 10
      shape {
        element_type: F32
        dimensions: [ 1 ]
        layout { minor_to_major: 0 }
      }
      literal {
        shape {
          element_type: F32
          dimensions: [ 1 ]
          layout { minor_to_major: 0 }
        }
        f32s: 1.0
      }
    }
    program_shape {
      result {
        element_type: F32
        dimensions: [ 1 ]
      }
    }
  }
}"#;

fn session() -> std::path::PathBuf {
    let session_dir = session_dir("unified-graph-viewer-processor");
    std::fs::write(session_dir.join("test_host.xplane.pb"), b"").unwrap();
    session_dir
}

#[test]
fn handle_missing_hlo_proto() {
    assert!(serve(&session(), &HashMap::new()).is_err());
}

#[test]
fn handle_success() {
    let session_dir = session();
    write_module(&session_dir, "my_module", &hlo_proto(DUMMY_HLO));
    let options: HashMap<String, String> =
        [("module_name", "my_module"), ("type", "short_txt"), ("graph_width", "1"), ("merge_fusion", "false")].map(|(key, value)| (key.to_string(), value.to_string())).into();
    assert!(!serve(&session_dir, &options).unwrap().0.is_empty());
}
