use super::hlo_fixture::{hlo_proto, module};

const HLO_PROTO: &str = r#"hlo_module {
  name: "some_module"
  entry_computation_name: "some_module"
  computations {
    name: "some_module"
    instructions {
      name: "arg0.1"
      opcode: "parameter"
      shape {
        element_type: S32
        layout { tail_padding_alignment_in_elements: 1 }
      }
      id: 4294967297
    }
    instructions {
      name: "arg1.1"
      opcode: "parameter"
      shape {
        element_type: S32
        layout { tail_padding_alignment_in_elements: 1 }
      }
      parameter_number: 1
      id: 4294967298
    }
    instructions {
      name: "XLA_Retvals.1"
      opcode: "tuple"
      shape {
        element_type: TUPLE
        tuple_shapes {
          element_type: S32
          layout { tail_padding_alignment_in_elements: 1 }
        }
      }
      id: 4294967303
      operand_ids: 1
    }
    id: 1
    root_id: 4294967303
  }
  host_program_shape {
    parameters {
      element_type: S32
      layout { tail_padding_alignment_in_elements: 1 }
    }
    parameters {
      element_type: S32
      layout { tail_padding_alignment_in_elements: 1 }
    }
    result {
      element_type: TUPLE
      tuple_shapes {
        element_type: S32
        layout { tail_padding_alignment_in_elements: 1 }
      }
    }
    parameter_names: "arg0"
    parameter_names: "arg1"
  }
  id: 1
  entry_computation_id: 1
}"#;

fn assert_consecutive_ids_and_operands() {
    let module = module(&hlo_proto(HLO_PROTO));
    assert!(module.valid);
    let entry = &module.graphs[module.entry];
    assert_eq!(entry.nodes.len(), 3);
    assert_eq!(entry.nodes.iter().map(|&node| node - entry.nodes[0]).collect::<Vec<_>>(), [0, 1, 2]);
    let parameter = entry.parameters[0];
    assert_eq!(module.nodes[parameter].name, "arg0.1");
    assert_eq!(parameter - entry.nodes[0], 0);
    assert_eq!(module.nodes[entry.root].operands, [parameter]);
}

#[test]
fn fix_non_consecutive_instruction_ids() {
    assert_consecutive_ids_and_operands();
}

#[test]
fn fix_non_consecutive_instruction_ids_for_module() {
    assert_consecutive_ids_and_operands();
}
