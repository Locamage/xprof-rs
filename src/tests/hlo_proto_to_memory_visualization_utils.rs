use super::hlo_fixture::{hlo_proto, module};
use crate::hlo::xla::HloProto;
use crate::hlo::xla::heap_simulator_trace::{Event, event::Kind};
use crate::memory_viewer::render;
use serde_json::Value;

const HLO_BASE: &str = r#"hlo_module { name: "test_module" entry_computation_name: "test_computation" computations { name: "test_computation"
  instructions { name: "fusion.1" id: 0 shape { tuple_shapes { element_type: U64 } } }
  instructions { name: "fusion.2" id: 1 shape { tuple_shapes { element_type: U64 } } }
  instructions { name: "constant.1" id: 2 shape { tuple_shapes { element_type: U64 } } } } }
buffer_assignment {
  buffer_allocations { index: 0 size: 1048576 color: 0 assigned { logical_buffer_id: 1 offset: 0 size: 524288 } assigned { logical_buffer_id: 2 offset: 524288 size: 524288 } }
  buffer_allocations { index: 1 size: 1048576 color: 0 is_constant: true assigned { logical_buffer_id: 3 offset: 0 size: 1048576 } }
  logical_buffers { id: 1 size: 524288 color: 0 defined_at { instruction_id: 0 shape_index: 0 } }
  logical_buffers { id: 2 size: 524288 color: 0 defined_at { instruction_id: 1 shape_index: 0 } }
  logical_buffers { id: 3 size: 1048576 color: 0 defined_at { instruction_id: 2 shape_index: 0 } }
  heap_simulator_traces { %s } }"#;
const ALLOC_ALLOC_FREE_FREE: &str = "events { kind: ALLOC buffer_id: 1 } events { kind: ALLOC buffer_id: 2 } events { kind: FREE buffer_id: 1 } events { kind: FREE buffer_id: 2 }";
const HALF: u64 = 524288;

/// Gives an HLO proto text with one allocation (index 0). Instruction `i` has id `i` and defines the logical buffer with id `i + 1`.
fn hlo_text(color: u8, allocation_size: u64, buffers: &[(&str, u64, u64)], trace: &str) -> String {
    let (mut instructions, mut assigned, mut logical) = (String::new(), String::new(), String::new());
    for (id, &(name, offset, size)) in buffers.iter().enumerate() {
        instructions += &format!(r#"instructions {{ name: "{name}" id: {id} shape {{ tuple_shapes {{ element_type: U64 }} }} }} "#);
        assigned += &format!("assigned {{ logical_buffer_id: {} offset: {offset} size: {size} }} ", id + 1);
        logical += &format!("logical_buffers {{ id: {} size: {size} color: {color} defined_at {{ instruction_id: {id} shape_index: 0 }} }} ", id + 1);
    }
    format!(
        r#"hlo_module {{ name: "test_module" entry_computation_name: "test_computation" computations {{ name: "test_computation" {instructions}}} }}
buffer_assignment {{ buffer_allocations {{ index: 0 size: {allocation_size} color: {color} {assigned}}} {logical}heap_simulator_traces {{ {trace} }} }}"#
    )
}

fn two_halves(color: u8, names: [&str; 2]) -> HloProto {
    hlo_proto(&hlo_text(color, 2 * HALF, &[(names[0], 0, HALF), (names[1], HALF, HALF)], ALLOC_ALLOC_FREE_FREE))
}

fn chain(trace: &str) -> HloProto {
    hlo_proto(&hlo_text(0, 3 * HALF, &[("fusion.1", 0, HALF), ("fusion.2", 0, HALF), ("fusion.3", 0, HALF)], trace))
}

fn chain_display_name(trace: &str) -> HloProto {
    hlo_proto(&hlo_text(1, 3 * HALF / 2, &[("fusion.1", 0, HALF / 2), ("fusion.2", 0, HALF / 2), ("fusion.3", 0, HALF / 2), ("fusion.4", HALF / 2, HALF)], trace))
}

fn scoped(mut proto: HloProto, sizes: &[u64]) -> HloProto {
    for (instruction, size) in proto.hlo_module.as_mut().unwrap().computations[0].instructions.iter_mut().zip(sizes) {
        instruction.backend_config = format!(r#"{{"used_scoped_memory_configs":[{{"memory_space":"1","size":"{size}"}}]}}"#).into_bytes();
    }
    proto
}

struct DoubleRectInfo {
    tooltip: String,
    pos_x: f64,
    pos_y: f64,
    width: f64,
    height: f64,
    offset: u64,
    size: u64,
    label: String,
    fontsize: f64,
}

fn preprocess(proto: &HloProto, color: i64) -> Value {
    let (body, kind) = render(&module(proto), color, 0, false).unwrap();
    assert_eq!(kind, "application/json");
    serde_json::from_str(&body).unwrap()
}

fn allocation_timeline(proto: &HloProto, color: i64) -> String {
    render(&module(proto), color, 0, true).unwrap().0
}

fn number(value: &Value, key: &str) -> f64 {
    value.get(key).map_or(0.0, |number| number.as_f64().or_else(|| number.as_str().and_then(|text| text.parse().ok())).unwrap())
}

fn text<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).map_or("", |text| text.as_str().unwrap())
}

fn list<'a>(value: &'a Value, key: &str) -> &'a [Value] {
    value.get(key).map_or(&[], |list| list.as_array().unwrap())
}

fn parse_logical_buffers_from_dot(dot: &str) -> Vec<DoubleRectInfo> {
    let find = |needle: &str, from: usize| dot[from..].find(needle).map(|position| position + from);
    let find_any = |needles: &[char], from: usize| dot[from..].find(needles).map(|position| position + from);
    let mut buffer_rects = Vec::new();
    let mut offset = 0;
    while let Some(node_end) = find("\"", offset).and_then(|node_start| find("\"", node_start + 1)) {
        offset = node_end + 1;
        let Some(open_bracket) = find("[", offset).filter(|open| open - offset <= 5) else { continue };
        let Some(tooltip_start) = find("tooltip=\"", open_bracket).map(|start| start + 9) else { continue };
        let Some(tooltip_end) = find("\"", tooltip_start) else { continue };
        let tooltip = dot[tooltip_start..tooltip_end].to_string();
        if !tooltip.contains("buffer_id:") {
            continue;
        }
        let Some(pos_start) = find("pos=\"", tooltip_end).map(|start| start + 5) else { continue };
        let Some(pos_end) = find("!", pos_start) else { continue };
        let (pos_x, pos_y) = dot[pos_start..pos_end].split_once(',').map_or((0.0, 0.0), |(x, y)| (x.parse().unwrap(), y.parse().unwrap()));
        let Some(width_start) = find("width=\"", pos_end).map(|start| start + 7) else { continue };
        let Some(width_end) = find_any(&['!', '"'], width_start) else { continue };
        let width = dot[width_start..width_end].parse::<f64>().unwrap() * 72.0;
        let Some(height_start) = find("height=\"", width_end).map(|start| start + 8) else { continue };
        let Some(height_end) = find_any(&['!', '"'], height_start) else { continue };
        let height = dot[height_start..height_end].parse::<f64>().unwrap() * 72.0;
        let fontsize = find("fontsize=", height_end).map(|start| start + 9).and_then(|start| Some(dot[start..find_any(&[',', ' ', ']'], start)?].parse().unwrap())).unwrap_or(0.0);
        let mut next_search_start = height_end;
        let mut label = String::new();
        if let Some(label_start) = find("label=\"", height_end).map(|start| start + 7)
            && let Some(label_end) = find("\"", label_start)
        {
            label = dot[label_start..label_end].to_string();
            next_search_start = label_end;
        }
        let field = |key: &str| tooltip.find(key).map_or(0, |position| tooltip[position + key.len()..].split(|character: char| !character.is_ascii_digit()).next().unwrap().parse().unwrap());
        buffer_rects.push(DoubleRectInfo { offset: field("\noffset:"), size: field("\nsize:"), tooltip, pos_x, pos_y, width, height, label, fontsize });
        offset = next_search_start + 1;
    }
    buffer_rects
}

fn boundary(rect: &DoubleRectInfo, sign: i32) -> f64 {
    if rect.pos_y == rect.pos_y as i32 as f64 && rect.height == rect.height as i32 as f64 {
        return (rect.pos_y as i32 + sign * (rect.height as i32 / 2)) as f64;
    }
    rect.pos_y + f64::from(sign) * rect.height / 2.0
}

fn rects_overlap(a: &DoubleRectInfo, b: &DoubleRectInfo) -> bool {
    let (a_left, a_right, a_bottom, a_top) = (a.pos_x - a.width / 2.0, a.pos_x + a.width / 2.0, a.pos_y - a.height / 2.0, a.pos_y + a.height / 2.0);
    let (b_left, b_right, b_bottom, b_top) = (b.pos_x - b.width / 2.0, b.pos_x + b.width / 2.0, b.pos_y - b.height / 2.0, b.pos_y + b.height / 2.0);
    let x_overlap = (a_right.min(b_right) - a_left.max(b_left)).max(0.0);
    let y_overlap = (a_top.min(b_top) - a_bottom.max(b_bottom)).max(0.0);
    x_overlap > 0.05 && y_overlap > 0.05
}

#[test]
fn test_heap_simulator_trace_share_with_1() {
    let trace = "events { kind: ALLOC buffer_id: 1 } events { kind: SHARE_WITH buffer_id: 2 share_with_canonical_id: 1 } events { kind: FREE buffer_id: 1 } events { kind: FREE buffer_id: 2 }";
    let preprocess_result = preprocess(&hlo_proto(&HLO_BASE.replace("%s", trace)), 0);
    assert_eq!(number(&preprocess_result, "peakHeapMib"), 1.5);
    assert_eq!(number(&preprocess_result, "peakUnpaddedHeapMib"), 8.0 / (1 << 20) as f64 + 1.0);
    assert_eq!(number(&preprocess_result, "totalBufferAllocationMib"), 2.0);
    assert_eq!(number(&preprocess_result, "indefiniteBufferAllocationMib"), 1.0);
}

#[test]
fn test_heap_simulator_trace_share_with_2() {
    let trace = "events { kind: ALLOC buffer_id: 1 } events { kind: FREE buffer_id: 1 } events { kind: SHARE_WITH buffer_id: 2 share_with_canonical_id: 1 } events { kind: FREE buffer_id: 2 }";
    let proto = hlo_proto(&HLO_BASE.replace("%s", trace));
    let preprocess_result = preprocess(&proto, 0);
    assert_eq!(number(&preprocess_result, "peakHeapMib"), 1.5);
    assert_eq!(number(&preprocess_result, "peakUnpaddedHeapMib"), 8.0 / (1 << 20) as f64 + 1.0);
    assert_eq!(number(&preprocess_result, "totalBufferAllocationMib"), 2.0);
    assert_eq!(number(&preprocess_result, "indefiniteBufferAllocationMib"), 1.0);
    assert!(!allocation_timeline(&proto, 0).is_empty());
}

#[test]
fn test_heap_simulator_trace_share_with_chain() {
    let trace = "events { kind: ALLOC buffer_id: 1 } events { kind: SHARE_WITH buffer_id: 2 share_with_canonical_id: 1 } events { kind: FREE buffer_id: 1 } events { kind: FREE buffer_id: 2 }
events { kind: SHARE_WITH buffer_id: 3 share_with_canonical_id: 2 } events { kind: FREE buffer_id: 3 }";
    let preprocess_result = preprocess(&chain(trace), 0);
    assert_eq!(number(&preprocess_result, "peakHeapMib"), 0.5);
    assert_eq!(list(&preprocess_result, "maxHeap").len(), 1);
}

#[test]
fn test_share_with_chain_display_name() {
    let trace = "events { kind: ALLOC buffer_id: 1 } events { kind: SHARE_WITH buffer_id: 2 share_with_canonical_id: 1 } events { kind: FREE buffer_id: 1 } events { kind: FREE buffer_id: 2 }
events { kind: ALLOC buffer_id: 4 } events { kind: SHARE_WITH buffer_id: 3 share_with_canonical_id: 2 } events { kind: FREE buffer_id: 3 } events { kind: FREE buffer_id: 4 }";
    let preprocess_result = preprocess(&chain_display_name(trace), 1);
    assert_eq!(number(&preprocess_result, "peakHeapMib"), 0.75);
    assert_eq!(list(&preprocess_result, "maxHeap").len(), 2);
    let (mut found_fusion_3, mut found_fusion_4) = (false, false);
    for heap_object in list(&preprocess_result, "maxHeap") {
        let instruction_name = text(heap_object, "instructionName");
        found_fusion_3 |= instruction_name.contains("fusion.3");
        found_fusion_4 |= instruction_name.contains("fusion.4");
        assert!(!instruction_name.contains("fusion.1"), "Bar chart shows root canonical name 'fusion.1' instead of sharer name 'fusion.3'. heap_object.instruction_name()={instruction_name}");
    }
    assert!(found_fusion_3, "Expected sharer C's instruction 'fusion.3' in the bar chart");
    assert!(found_fusion_4, "Expected independent buffer D's instruction 'fusion.4' in the bar chart");
}

#[test]
fn test_logical_buffers_do_not_overlap() {
    let verify_no_overlap = |buffers: &[(&str, u64, u64)], size: u64, trace: &str| {
        let timeline = allocation_timeline(&hlo_proto(&hlo_text(0, size, buffers, trace)), 0);
        assert!(!timeline.is_empty());
        let mut rects = parse_logical_buffers_from_dot(&timeline);
        assert!(!rects.is_empty());
        for i in 0..rects.len() {
            for j in i + 1..rects.len() {
                assert!(!rects_overlap(&rects[i], &rects[j]), "Overlap detected between:\nRect {i}: {}\nRect {j}: {}", rects[i].tooltip, rects[j].tooltip);
            }
        }
        rects.sort_by_key(|rect| rect.offset);
        for pair in rects.windows(2) {
            if pair[0].offset + pair[0].size == pair[1].offset {
                let (top, bottom) = (boundary(&pair[0], 1), boundary(&pair[1], -1));
                assert!((bottom - top).abs() <= 0.05, "Non-zero gap between adjacent buffers:\nPrev: {} (top: {top})\nNext: {} (bottom: {bottom})", pair[0].tooltip, pair[1].tooltip);
            }
        }
    };
    verify_no_overlap(&[("fusion.1", 0, HALF), ("fusion.2", HALF, HALF)], 2 * HALF, ALLOC_ALLOC_FREE_FREE);
    verify_no_overlap(&[("fusion.1", 54076112896, 134217728), ("fusion.2", 54210330624, 134217728)], 54344548352, ALLOC_ALLOC_FREE_FREE);
    verify_no_overlap(
        &[("fusion.1", 0, HALF), ("fusion.2", 0, HALF)],
        2 * HALF,
        "events { kind: ALLOC buffer_id: 1 } events { kind: FREE buffer_id: 1 } events { kind: ALLOC buffer_id: 2 } events { kind: FREE buffer_id: 2 }",
    );
}

#[test]
fn scoped_vmem_allocation_single_instruction() {
    let result = preprocess(&scoped(two_halves(1, ["fusion.1", "fusion.2"]), &[1048576]), 1);
    assert_eq!(number(&result, "maxScopedVmemAllocationMib"), 1.0);
    assert_eq!(text(&result, "maxScopedVmemInstructionName"), "fusion.1");
}

#[test]
fn scoped_vmem_allocation_max_across_instructions() {
    let result = preprocess(&scoped(two_halves(1, ["fusion.1", "fusion.2"]), &[1048576, 2097152]), 1);
    assert_eq!(number(&result, "maxScopedVmemAllocationMib"), 2.0);
    assert_eq!(text(&result, "maxScopedVmemInstructionName"), "fusion.2");
}

#[test]
fn scoped_vmem_allocation_no_scoped_configs() {
    let result = preprocess(&two_halves(1, ["fusion.1", "fusion.2"]), 1);
    assert_eq!(number(&result, "maxScopedVmemAllocationMib"), 0.0);
    assert!(text(&result, "maxScopedVmemInstructionName").is_empty());
}

#[test]
fn scoped_vmem_allocation_hbm_ignores_scoped() {
    let result = preprocess(&scoped(hlo_proto(&HLO_BASE.replace("%s", ALLOC_ALLOC_FREE_FREE)), &[1048576]), 0);
    assert_eq!(number(&result, "maxScopedVmemAllocationMib"), 0.0);
    assert!(text(&result, "maxScopedVmemInstructionName").is_empty());
}

#[test]
fn test_convert_allocation_timeline_buffer_blocks() {
    let trace = "events { kind: ALLOC buffer_id: 1 } events { kind: FREE buffer_id: 1 } events { kind: SHARE_WITH buffer_id: 2 share_with_canonical_id: 1 } events { kind: FREE buffer_id: 2 }";
    let preprocess_result = preprocess(&hlo_proto(&HLO_BASE.replace("%s", trace)), 0);
    let blocks = list(&preprocess_result, "bufferBlocks");
    assert_eq!(blocks.len(), 2);
    let container = &blocks[0];
    assert_eq!(number(container, "logicalBufferId"), -1.0);
    assert_eq!(number(container, "offset"), 0.0);
    assert_eq!(number(container, "size"), 1048576.0);
    assert_eq!(number(container, "startStep"), 0.0);
    assert_eq!(number(container, "endStep"), 4.0);
    assert_eq!(text(container, "color"), "#ffffff");
    let block = &blocks[1];
    assert_eq!(number(block, "logicalBufferId"), 1.0);
    assert_eq!(text(block, "name"), "fusion.1{0}");
    assert_eq!(number(block, "offset"), 0.0);
    assert_eq!(number(block, "size"), 524288.0);
    assert_eq!(number(block, "startStep"), 0.0);
    assert_eq!(number(block, "endStep"), 3.0);
    assert_eq!(text(block, "category"), "Temporary");
}

#[test]
fn test_allocation_timeline_labels() {
    let result = allocation_timeline(&two_halves(1, ["very_long_instruction_name_that_might_need_truncation_or_not", "short"]), 1);
    assert!(!result.is_empty());
    let rects = parse_logical_buffers_from_dot(&result);
    assert_eq!(rects.len(), 2);
    assert_eq!(rects[0].label, "very_long_instruction_name_that_might_need_truncation_or_not");
    assert_eq!(rects[1].label, "short");
}

#[test]
fn test_allocation_timeline_labels_truncation() {
    let mut proto =
        hlo_proto(&hlo_text(1, 2 * HALF, &[("very_long_instruction_name_that_should_be_truncated", 0, 2 * HALF)], "events { kind: ALLOC buffer_id: 1 } events { kind: FREE buffer_id: 1 }"));
    let trace = &mut proto.buffer_assignment.as_mut().unwrap().heap_simulator_traces[0];
    for _ in 0..49 {
        trace.events.push(Event { kind: Kind::Alloc as i32, buffer_id: 1, ..Default::default() });
        trace.events.push(Event { kind: Kind::Free as i32, buffer_id: 1, ..Default::default() });
    }
    let result = allocation_timeline(&proto, 1);
    assert!(!result.is_empty());
    let rects = parse_logical_buffers_from_dot(&result);
    assert_eq!(rects.len(), 1);
    assert_eq!(rects[0].label, "v...");
}

#[test]
fn test_allocation_timeline_font_size_scaling() {
    let trace = "events { kind: ALLOC buffer_id: 1 } events { kind: ALLOC buffer_id: 2 } events { kind: ALLOC buffer_id: 3 } events { kind: FREE buffer_id: 1 } events { kind: FREE buffer_id: 2 } events { kind: FREE buffer_id: 3 }";
    let result = allocation_timeline(&hlo_proto(&hlo_text(1, 1000000, &[("large_buffer", 0, 995000), ("medium_buffer", 995000, 4000), ("small_buffer", 999000, 1000)], trace)), 1);
    assert!(!result.is_empty());
    let rects = parse_logical_buffers_from_dot(&result);
    assert_eq!(rects.len(), 3);
    assert!((rects[0].fontsize - 14.0).abs() <= 0.01);
    assert!((rects[1].fontsize - 9.8).abs() <= 0.01);
    assert!((rects[2].fontsize - 8.0).abs() <= 0.01);
}

#[test]
fn memory_viewer_matches_xprof_on_synthetic_modules() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/memory_viewer");
    let mut text = String::new();
    std::io::Read::read_to_string(&mut flate2::read::GzDecoder::new(&include_bytes!("../../tests/data/memory_viewer/expected.json.gz")[..]), &mut text).unwrap();
    let cases: Vec<Value> = serde_json::from_str(&text).unwrap();
    for case in &cases {
        let params: std::collections::HashMap<String, String> = case["params"].as_object().unwrap().iter().map(|(key, value)| (key.clone(), value.as_str().unwrap().to_string())).collect();
        assert_eq!(crate::memory_viewer::serve(&dir, &params).unwrap().0, case["body"].as_str().unwrap(), "{params:?}");
    }
}

#[test]
fn test_heap_simulator_trace_share_with_cycle() {
    for trace in [
        "events { kind: ALLOC buffer_id: 1 }\nevents { kind: SHARE_WITH buffer_id: 1 share_with_canonical_id: 1 }\nevents { kind: FREE buffer_id: 1 }",
        "events { kind: ALLOC buffer_id: 1 }\nevents { kind: SHARE_WITH buffer_id: 2 share_with_canonical_id: 1 }\nevents { kind: SHARE_WITH buffer_id: 1 share_with_canonical_id: 2 }\nevents { kind: FREE buffer_id: 1 }",
    ] {
        preprocess(&hlo_proto(&HLO_BASE.replace("%s", trace)), 0);
    }
}
