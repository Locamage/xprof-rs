use super::{Dumper, HashMap, Module, Printer, Style, gpu_properties, render};

#[test]
fn cublas_lt_gemm_labels_show_the_backend_dot_dimensions() {
    let config = br#"{"gemm_backend_config":{"alpha_real":2,"beta":0.5,"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":["0"],"rhs_batch_dimensions":[]},"epilogue":"RELU","selected_algorithm":"7"}}"#;
    assert_eq!(
        gpu_properties("custom-call", "__cublas$lt$matmul", 11, config).as_deref(),
        Some("<br/>alpha=2<br/>beta=0.5<br/>lhs_batch_dims={0}<br/>lhs_contracting_dims={1}<br/>rhs_contracting_dims={0}<br/>algorithm=7<br/>epilogue=RELU")
    );
    let plain = br#"{"gemm_backend_config":{"alpha_real":1,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"]},"epilogue":"DEFAULT"}}"#;
    assert_eq!(gpu_properties("custom-call", "__cublas$lt$matmul$f8", 11, plain).as_deref(), Some("lhs_contracting_dims={1}<br/>rhs_contracting_dims={0}"));
    assert_eq!(gpu_properties("custom-call", "__cublas$gemm", 11, plain), None);
    assert_eq!(gpu_properties("custom-call", "__cublas$lt$matmul", 11, b"not json"), None);
    assert_eq!(gpu_properties("custom-call", "__cublas$lt$matmul", 11, br#"{"gemm_backend_config":{"unknown":1}}"#), None);
}

#[test]
fn cudnn_convolution_labels_show_scales_activation_and_engine() {
    let config = br#"{"cudnn_conv_backend_config":{"activation_mode":"kLeakyRelu","conv_result_scale":0.5,"side_input_scale":1,"leakyrelu_alpha":0.1,"algorithm":{"algo_id":"67","math_type":"TENSOR_OP_MATH","tuning_knobs":{"28":"1","2":"1","-3":"0"}}}}"#;
    assert_eq!(
        gpu_properties("custom-call", "__cudnn$convForward", 11, config).as_deref(),
        Some("<br/>conv_result_scale=0.5<br/>leakyrelu_alpha=0.1<br/>activation_mode=leakyrelu<br/>algo=eng67{k-3=0,k2=1,k28=1}")
    );
    assert_eq!(gpu_properties("custom-call", "__cudnn$convBackwardFilter", 11, b"").as_deref(), Some("<br/>conv_result_scale=0<br/>activation_mode=none<br/>algo=eng0{}"));
    assert_eq!(
        gpu_properties("custom-call", "__cudnn$convForward", 11, br#"{"cudnnConvBackendConfig":{"activationMode":9,"convResultScale":1}}"#).as_deref(),
        Some("<br/>activation_mode=unknown: 9<br/>algo=eng0{}")
    );
    assert_eq!(gpu_properties("custom-call", "__cudnn$convForward", 11, br#"{"cudnn_conv_backend_config":{},"bogus":true}"#), None);
}

fn graph_dumper_module(name: &str) -> Module<'static> {
    let path = format!("{}/tests/data/hlo_graph_dumper/{name}.pb", env!("CARGO_MANIFEST_DIR"));
    Module::parse(std::borrow::Cow::Owned(std::fs::read(path).unwrap()))
}

fn render_graph(module: &Module, computation: &str, label: &str) -> String {
    let mut callers: Vec<Vec<usize>> = vec![Vec::new(); module.graphs.len()];
    for (index, node) in module.nodes.iter().enumerate() {
        for &graph in &node.called {
            callers[graph].push(index);
        }
    }
    let mut dumper = Dumper {
        printer: Printer::new(module, Style::Graph, true),
        module,
        computation: module.find_graph(computation).unwrap(),
        label: label.into(),
        filter: None,
        backend_config: false,
        fusions: true,
        node_ids: HashMap::new(),
        edge_counts: HashMap::new(),
        cluster_ids: HashMap::new(),
        edges: Vec::new(),
        callers,
    };
    dumper.dump().unwrap()
}

#[test]
fn nested_fusion() {
    let module = graph_dumper_module("nested_fusion");
    let graph = render_graph(&module, "b", "");
    for computation in ["b", "fused_computation.inner", "fused_computation.outer"] {
        for &node in &module.graphs[module.find_graph(computation).unwrap()].nodes {
            assert!(graph.contains(&module.nodes[node].name), "{}", module.nodes[node].name);
        }
    }
    let neighborhood_graph = String::from_utf8(render(&module, "add.0", 1, false, true, "dot").unwrap().0).unwrap();
    assert!(neighborhood_graph.contains("add.0"));
}

#[test]
fn constant() {
    assert!(render_graph(&graph_dumper_module("constant"), "b", "an_empty_graph").contains("an_empty_graph"));
}

#[test]
fn tuple_constant() {
    let graph = render_graph(&graph_dumper_module("tuple_constant"), "b", "tuple_constant");
    assert!(graph.contains("tuple_constant"));
    assert!(graph.contains("constant (f32[3,2], s32[4,5])"));
}

#[test]
fn compare() {
    assert!(render_graph(&graph_dumper_module("compare"), "comp", "tuple_constant").contains("direction=LT"));
}

#[test]
fn has_statistics_viz() {
    render_graph(&graph_dumper_module("has_statistics_viz"), "comp", "tuple_constant");
}

#[test]
fn root_is_constant() {
    render_graph(&graph_dumper_module("root_is_constant"), "conditional_select", "tuple_constant");
}

#[test]
fn show_callers() {
    let module = graph_dumper_module("show_callers");
    assert!(render_graph(&module, "comp", "command_buffer").contains("ENTRY computation"));
    assert!(render_graph(&module, "command_buffer", "command_buffer").contains("Caller instructions: call.1"));
}

#[test]
fn annotate_called_computations_parameters() {
    let graph = render_graph(&graph_dumper_module("annotate_called_computations_parameters"), "command_buffer.1", "command buffer");
    assert!(graph.contains("<b>Parameter 0</b><br/><i>from add.123 in command_buffer.0</i>"));
    assert!(graph.contains("<b>Parameter 1</b><br/><i>from mul.456 in command_buffer.0</i>"));
}

#[test]
fn annotate_called_computations_parameters_tuple() {
    let graph = render_graph(&graph_dumper_module("annotate_called_computations_parameters_tuple"), "command_buffer", "command buffer");
    assert!(graph.contains("<b>Parameter 0</b><br/><i>from tuple.1 in the ENTRY computation</i>"));
}
