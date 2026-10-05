#![allow(clippy::needless_raw_string_hashes)]

use crate::hlo::Module;
use crate::hlo::text::{Printer, Style};
use std::borrow::Cow;

fn module_string(fixture: &[u8], style: Style) -> String {
    let module = Module::parse(Cow::Borrowed(fixture));
    let mut printer = Printer::new(&module, style, style == Style::Long);
    (printer.operand_shapes, printer.large_constants) = (style == Style::Long, true);
    printer.module_text().unwrap_or_default()
}

fn canonical(text: &str) -> String {
    let mut text = text.trim().to_string();
    for key in [", replica_count=", ", num_partitions="] {
        while let Some(start) = text.find(key) {
            let digits = text[start + key.len()..].find(|character: char| !character.is_ascii_digit()).unwrap_or(text.len() - start - key.len());
            text.replace_range(start..start + key.len() + digits, "");
        }
    }
    text
}

macro_rules! corpus {
    ($suite:literal, $style:expr, $($(#[$attribute:meta])* $name:ident,)*) => {
        $(
            #[test]
            $(#[$attribute])*
            fn $name() {
                let fixture = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/hlo_parser_test/", $suite, "/", stringify!($name), ".pb"));
                let expected = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/hlo_parser_test/", $suite, "/", stringify!($name), ".hlo"));
                assert_eq!(super::canonical(expected), super::canonical(&super::module_string(fixture, $style)));
            }
        )*
    };
}

mod hlo_parser_test_long {
    use crate::hlo::text::Style;

    corpus! {
    "long", Style::Long,
    axpy_param,
    param_replication,
    constant_pred,
    constant_pred_array,
    constant_s32,
    constant_s32_with_statistics,
    constant_f32,
    constant_f32_r1_empty,
    constant_f32_r4_empty,
    constant4_d,
    constant_non_finite,
    constant_non_finite_e4_m3,
    constant_non_finite_e4_m3_b11,
    constant_f16,
    bf16,
    add_constants,
    tuple_constant,
    select_r1_f32,
    empty_tuple_create,
    tuple_create,
    large_tuple_round_trip,
    sharded_tuple_create,
    domain_parsing,
    while_with_scalar_s32_result,
    copy_start_and_copy_done,
    send_recv,
    send_recv_wo_channel_id,
    send_recv_with_host_transfer,
    get_tuple_element,
    call,
    composite_call,
    composite_call_with_extra_frontend_attributes,
    composite_call_optional_attributes_and_version,
    composite_call_optional_attributes,
    composite_call_optional_version,
    custom_call_with_opaque,
    custom_call_with_backend_config_in_curly_braces,
    custom_call_with_literal,
    custom_call_with_literal_tuple,
    custom_call_with_literal_r0,
    reduce_window,
    reduce_window_scalar,
    reduce_window_variadic,
    convolution,
    convolution_dynamic,
    convolution_r2,
    convolution_backward,
    reverse4_d,
    concat,
    select_and_scatter,
    select_and_scatter_scalar,
    slice,
    slice_no_stride,
    slice_r0,
    transpose,
    transpose_c128,
    triangular_solve,
    dynamic_slice,
    dynamic_slice_scalar_indices,
    dynamic_update_slice,
    dynamic_update_slice_scalar_index,
    batch_norm_training,
    batch_norm_inference,
    batch_norm_grad,
    fft,
    ifft2d,
    rfft2d,
    irfft3d,
    pad,
    pad_has_interior,
    round_nearest_even,
    pad_has_negative_padding,
    fusion,
    async_start_with_aliasing,
    fusion_with_aliasing,
    gather,
    sorted_gather,
    batch_gather,
    scatter,
    batch_scatter,
    tuple_scatter,
    sorted_scatter,
    unique_indices_scatter,
    constant_unsigned_no_underflow,
    constant_unsigned_no_overflow,
    custom_call_with_layout_constraints,
    custom_call_with_layout_constraints_no_operands,
    custom_call_with_layout_constraints_tuple_shapes,
    custom_call_with_has_side_effect,
    custom_call_with_aliasing,
    custom_call_with_schedule,
    custom_call_with_status_returning_version,
    parse_c64_literal,
    parse_c128_literal,
    indexed_conditional,
    rng_get_and_update_state,
    rng_bit_generator,
    async_ops_with_syntax_sugar,
    async_ops_with_syntax_sugar_and_thread_name,
    hlo_computation_with_parallel_thread_name,
    metadata_fields,
    original_value,
    original_value_synthetic,
    original_value_recovery_table,
    debug_attributes,
    debug_attributes_fusion_debugger,
    debug_attributes_log_mode_only,
    debug_attributes_partitioned_only,
    debug_attributes_hlo_id_only,
    original_value_recovery_table_with_nested_quotes,
    stack_frame_index,
    }
}

mod hlo_parser_test_short {
    use crate::hlo::text::Style;

    corpus! {
    "short", Style::Short,
    map,
    reduce,
    tuple_reduce,
    infeed_outfeed,
    rng,
    reduce_precision,
    sort_key,
    sort_key_value,
    sort_key_r2,
    sort_key_value_r2,
    sort_many_values,
    sort_key_stable,
    top_k,
    top_k_unstable,
    indexed_conditional,
    predicated_conditional,
    custom_call,
    custum_call_single_comp,
    custum_call_multiple_comps,
    non_default_names,
    dot,
    dot_with_algorithm,
    gather,
    all_reduce,
    all_reduce_with_subgroups,
    all_reduce_with_subgroups_iota_list,
    all_reduce_with_mesh_axes_replica_group_list,
    all_reduce_with_layout,
    all_reduce_all_reduce,
    all_reduce_start_and_done,
    reduce_scatter,
    all_gather,
    all_gather_with_layout,
    all_gather_with_subgroups,
    all_gather_with_subgroups_iota_list,
    all_to_all,
    all_to_all_with_subgroups,
    all_to_all_with_subgroups_iota_list,
    ragged_all_to_all_with_replica_groups,
    ragged_all_to_all_with_collective_device_list,
    ragged_all_to_all,
    collective_broadcast,
    collective_broadcast_dynamic_root,
    collective_reduce,
    collective_reduce_with_channel_id,
    collective_reduce_dynamic_root,
    collective_permute,
    combined_collective_permute,
    collective_permute_in_place_update,
    collective_permute_in_place_update3_d,
    collective_permute_in_place_update_multiple_read_write,
    collective_permute_in_place_update_tuple_multiple_read_write,
    collective_permute_tuple_in_place_update,
    collective_permute_start_and_done,
    combined_collective_permute_start_and_done,
    collective_permute_start_and_done_inplace_update,
    replica_id,
    partition_id,
    iota,
    custom_call_with_window_and_dim_labels_and_feature_group_count,
    custom_call_with_unknown_dim_labels,
    scheduled_module,
    after_all_with_multiple_operands,
    add_dependency,
    min_max_values,
    bitcast_convert,
    scan,
    }
}

mod hlo_non_roundtrip_parser_test {
    use crate::hlo::text::Style;

    corpus! {
    "nonroundtrip", Style::Short,
    simple_nesting,
    ambiguous_names,
    tuple_shape_inside_anonymous_instr,
    mix_anon_and_non_anon_operands,
    broadcast_of_scalar_doesnt_need_dimensions_attr,
    compact_gte,
    nested_compact_gte,
    compact_gte_multiple_uses,
    compact_gte_name_collision,
    compact_gte_mixed_nesting,
    compact_gte_on_tuple,
    }
}

mod hlo_parser_test_printing {
    use crate::hlo::Module;
    use crate::hlo::text::{Printer, Style};
    use std::borrow::Cow;
    use std::sync::LazyLock;

    static PARSER_PRINTING: LazyLock<Module<'static>> = LazyLock::new(|| fixture("parser_printing"));

    fn fixture(name: &str) -> Module<'static> {
        Module::parse(Cow::Owned(std::fs::read(format!("{}/tests/data/hlo_parser_test/printing/{name}.pb", env!("CARGO_MANIFEST_DIR"))).unwrap()))
    }

    fn instruction(name: &str) -> String {
        let mut text = String::new();
        Printer::new(&PARSER_PRINTING, Style::Long, true).instruction(PARSER_PRINTING.find(name).unwrap(), &mut text);
        text
    }

    fn module_text(name: &str, style: Style) -> String {
        let module = fixture(name);
        let mut printer = Printer::new(&module, style, true);
        printer.large_constants = false;
        printer.module_text().unwrap()
    }

    macro_rules! printed { ($($name:ident $(@ $fixture:literal)?: $key:literal = $original:literal,)*) => { $(#[test] fn $name() { assert!(instruction([$($fixture,)? stringify!($name)][0]).contains(&format!("{}={}", $key, $original))); })* } }

    macro_rules! golden { ($($name:ident),*) => { $(#[test] fn $name() { assert_eq!(module_text(stringify!($name), Style::Long), include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/hlo_parser_test/printing/", stringify!($name), ".hlo"))); })* } }

    printed! {
        parse_sharding: "sharding" = r#"{maximal device=42}"#,
        parse_named_sharding_unreduced_max: "sharding" = r#"{mesh['x'=2], [{}], unreduced=max{'x'}}"#,
        parse_sharding_partial_replication: "sharding" = r#"{devices=[2,2]0,1,2,3 last_tile_dim_replicate}"#,
        parse_sharding_sub_group: "sharding" = r#"{devices=[2,2,2,2]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 last_tile_dims={manual, replicated}}"#,
        parse_named_sharding1: "sharding" = r#"{mesh['a'=2,'b'=4,'c'=3,'d'=8], [{'a'}, {'b'}]}"#,
        parse_named_sharding2: "sharding" = r#"{mesh['a'=2,'b'=4,'c'=3,'d'=8], [{'a'}, {'c', 'b'}]}"#,
        parse_named_sharding_open_dims: "sharding" = r#"{mesh['a'=2,'b'=4,'c'=3,'d'=8], [{'b', 'a'}, {'c', 'd', ?}]}"#,
        parse_named_sharding_sub_axes1: "sharding" = r#"{mesh['a'=2,'b'=4,'c'=3,'d'=8], [{'a'}, {'b':(2)2}]}"#,
        parse_named_sharding_sub_axes2: "sharding" = r#"{mesh['a'=2,'b'=4,'c'=3,'d'=8], [{'b':(2)2}, {'d':(4)2, 'c'}]}"#,
        parse_named_sharding_sub_axes_open_dims: "sharding" = r#"{mesh['a'=2,'b'=4,'c'=3,'d'=8], [{'b':(2)2}, {'d':(4)2, 'c', ?}]}"#,
        parse_named_sharding_non_iota_mesh: "sharding" = r#"{mesh['a'=2,'b'=4,'c'=4,'d'=2], device_ids=([4,16]T(1,0)), [{'a'}]}"#,
        parse_named_sharding_non_iota_mesh_device_list: "sharding" = r#"{mesh['x'=2,'y'=2], device_ids=(0,2,1,3), [{'x'}]}"#,
        parse_named_sharding_empty_mesh_replicated: "sharding" = r#"{mesh[], replicated}"#,
        parse_named_sharding_fully_replicated: "sharding" = r#"{mesh['a'=2,'b'=4], replicated}"#,
        parse_named_sharding_replicated_axes: "sharding" = r#"{mesh['a'=2,'b'=4], [{'a'}], replicated={'b'}}"#,
        parse_named_sharding_maximal: "sharding" = r#"{maximal_mesh[device_id=5]}"#,
        parse_named_sharding_with_special_characters: "sharding" = r#"{mesh['a.b'=2,'<axis> def'=4,'z/w'=2], [{'a.b'}, {'<axis> def':(2)2, 'z/w'}]}"#,
        parse_named_sharding_fully_unreduced: "sharding" = r#"{mesh['a'=2,'b'=4], unreduced}"#,
        parse_named_sharding_unreduced_axes: "sharding" = r#"{mesh['a'=2,'b'=4,'c'=3,'d'=8], [{}, {'b'}], unreduced={'d':(4)2}}"#,
        parse_named_sharding_fully_manual: "sharding" = r#"{mesh['a'=2,'b'=4], manual}"#,
        parse_named_sharding_manual_axes: "sharding" = r#"{mesh['a'=2,'b'=4,'c'=3,'d'=8], [{'a'}], manual={'d':(4)2}}"#,
        parse_named_sharding_all_fields_with_metadata: "sharding" = r#"{mesh['a'=2,'b'=4,'c'=3,'d'=8], [{'a'}], replicated={'c'}, unreduced={'d':(4)2}, manual={'b':(2)2}, metadata={{op_name="foo"}, {op_name="bar"}}}"#,
        parse_named_sharding_fully_replicated_with_metadata: "sharding" = r#"{mesh['a'=2,'b'=4], replicated, metadata={{op_name="foo"}}}"#,
        parse_named_sharding_tuple: "sharding" = r#"{{mesh['a'=2,'b'=4,'c'=3,'d'=8], [{'d', 'c'}, {'a', 'b'}]}, {mesh['a'=2,'b'=4,'c'=3,'d'=8], replicated}, {mesh['a'=2,'b'=4,'c'=3,'d'=8], [{'d':(2)2, 'b'}, {'a', ?}], unreduced={'c'}}, {mesh['a'=2,'b'=4,'c'=3,'d'=8], [{'d', 'c'}, {'a', 'b'}], metadata={{op_name="foo"}, {op_name="bar"}}}}"#,
        parse_mixed_sharding_tuple1: "sharding" = r#"{{replicated}, {mesh['a'=2,'b'=4], replicated}, {maximal device=5}, {maximal_mesh[device_id=5]}}"#,
        parse_mixed_sharding_tuple2: "sharding" = r#"{{mesh['a'=2,'b'=2], [{'a'}, {}]}, {devices=[2,2]<=[4] last_tile_dim_replicate}}"#,
        parse_trivial_iota_sharding_partial_replication: "sharding" = r#"{devices=[2,2]<=[4] last_tile_dim_replicate}"#,
        parse_trivial_iota_sharding_sub_group: "sharding" = r#"{devices=[2,2,2,2]<=[16] last_tile_dims={manual, replicated}}"#,
        parse_transposed_iota_sharding_partial_replication: "sharding" = r#"{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"#,
        parse_transposed_iota_sharding_sub_group: "sharding" = r#"{devices=[2,2,2,2]<=[2,2,4]T(2,1,0) last_tile_dims={manual, replicated}}"#,
        parse_shard_as: "sharding" = r#"{manual shard_as 1}"#,
        parse_shard_like: "sharding" = r#"{devices=[2,2,2,2]<=[16] last_tile_dims={manual, replicated} shard_like 1}"#,
        parse_unknown_sharding: "sharding" = r#"{unknown}"#,
        parse_frontend_attributes: "frontend_attributes" = r#"{attr_a="test_a",attr_b="b",attr_c={type="s64"},attr_d="a=\"b/c\""}"#,
        parse_window: "window" = r#"{size=1x2x3}"#,
        parse_convolution_dimension_numbers: "dim_labels" = r#"b0f_0io->b0f"#,
        parse_replica_groups: "replica_groups" = r#"{{0,1},{2,3}}"#,
        parse_collective_device_list_v1 @ "parse_replica_groups": "replica_groups" = r#"{{0,1},{2,3}}"#,
        parse_collective_device_list_v2: "replica_groups" = r#"[2,2]<=[4]"#,
        parse_collective_device_list_v3 @ "parse_replica_groups_v3": "replica_groups" = r#"mesh['axis_0'=2,'axis_1'=2] {'axis_1'}"#,
        parse_padding_config_no_interior_padding: "padding" = r#"0_1x2_3"#,
        parse_padding_config_interior_padding: "padding" = r#"0_1_0x2_3_4"#,
    }

    golden!(short_constant, negative_nan, nan_payload);

    #[test]
    #[ignore = "differs from upstream: XProf 2.23.2 prints a fully unreduced named sharding as unreduced=max without its axis list (checked over HTTP); XLA prints the axes only for a strict subset of the mesh"]
    fn parse_named_sharding_scalar_unreduced_max() {
        assert!(instruction("parse_named_sharding_scalar_unreduced_max").contains("sharding={mesh['x'=2,'y'=2], unreduced=max{'x', 'y'}}"));
    }

    #[test]
    fn original_value_without_shape() {
        assert!(module_text("original_value_without_shape", Style::Short).contains(r#"origin={{"v"}}"#));
    }

    #[test]
    fn empty_leaf_in_original_value() {
        assert!(module_text("empty_leaf_in_original_value", Style::Short).contains(r#"origin={(({}, {"v2"}), {"v3"})}"#));
    }
}
