use crate::hlo::Module;
use crate::hlo_text::{Printer, Style};
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
    ($suite:literal, $style:expr, $($(#[$attribute:meta])* $name:ident: $expected:expr,)*) => {
        $(
            #[test]
            $(#[$attribute])*
            fn $name() {
                let fixture = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/hlo_parser_test/", $suite, "/", stringify!($name), ".pb"));
                assert_eq!(super::canonical($expected), super::canonical(&super::module_string(fixture, $style)));
            }
        )*
    };
}

mod hlo_parser_test_long {
    use crate::hlo_text::Style;

    corpus! {
    "long", Style::Long,
    axpy_param: r#"HloModule axpy_module, entry_computation_layout={(f32[], f32[2,4]{1,0}, f32[2,4]{1,0})->f32[2,4]{1,0}}

ENTRY %axpy.v5 (alpha: f32[], x: f32[2,4], y: f32[2,4]) -> f32[2,4] {
  %alpha = f32[] parameter(0)
  %broadcast = f32[2,4]{1,0} broadcast(f32[] %alpha), dimensions={}
  %x = f32[2,4]{1,0} parameter(1)
  %multiply = f32[2,4]{1,0} multiply(f32[2,4]{1,0} %broadcast, f32[2,4]{1,0} %x)
  %y = f32[2,4]{1,0} parameter(2)
  ROOT %add = f32[2,4]{1,0} add(f32[2,4]{1,0} %multiply, f32[2,4]{1,0} %y)
}

"#,
    param_replication: r#"HloModule param_replication_module, entry_computation_layout={(f32[], (f32[2,4]{1,0}, (f32[2,4]{1,0})))->(f32[], (f32[2,4]{1,0}, (f32[2,4]{1,0})))}

ENTRY %param_replication (a: f32[], b: (f32[2,4], (f32[2,4]))) -> (f32[], (f32[2,4], (f32[2,4]))) {
  %a = f32[] parameter(0), parameter_replication={true}
  %b = (f32[2,4]{1,0}, (f32[2,4]{1,0})) parameter(1), parameter_replication={false,true}
  ROOT %tuple = (f32[], (f32[2,4]{1,0}, (f32[2,4]{1,0}))) tuple(f32[] %a, (f32[2,4]{1,0}, (f32[2,4]{1,0})) %b)
}

"#,
    constant_pred: r#"HloModule constant_pred_module, entry_computation_layout={()->pred[]}

ENTRY %constant_pred () -> pred[] {
  ROOT %constant = pred[] constant(true), metadata={op_type="const" op_name="\"it\'s not a problem\n" source_file="path/to/test.cc" source_line=68}, backend_config="foo\" bar"
}

"#,
    constant_pred_array: r#"HloModule module, entry_computation_layout={()->pred[2,3]{1,0}}

ENTRY %constant_pred_array () -> pred[2,3] {
  ROOT %constant = pred[2,3]{1,0} constant({ { 0, 1, 0 }, { 1, 0, 1 } })
}

"#,
    constant_s32: r#"HloModule constant_s32_module, entry_computation_layout={()->s32[]}

ENTRY %constant_s32 () -> s32[] {
  ROOT %constant = s32[] constant(-42)
}

"#,
    constant_s32_with_statistics: r#"HloModule constant_s32_module, entry_computation_layout={()->s32[]}

ENTRY %constant_s32 () -> s32[] {
  ROOT %constant = s32[] constant(-42), statistics={visualizing_index=1,stat-1=33,stat-2=44}
}

"#,
    constant_f32: r#"HloModule ConstantF32_module, entry_computation_layout={()->f32[]}

ENTRY %ConstantF32.v4 () -> f32[] {
  ROOT %constant = f32[] constant(42), backend_config="this is a configuration"
}

"#,
    constant_f32_r1_empty: r#"HloModule ConstantF32Empty_module, entry_computation_layout={()->f32[0]{0}}

ENTRY %ConstantF32Empty.v4 () -> f32[0] {
  ROOT %constant = f32[0]{0} constant({})
}

"#,
    constant_f32_r4_empty: r#"HloModule ConstantF32R4Empty_module, entry_computation_layout={()->f32[2,0,4,3]{3,2,1,0}}

ENTRY %ConstantF32R4Empty.v4 () -> f32[2,0,4,3] {
  ROOT %constant = f32[2,0,4,3]{3,2,1,0} constant({ { /*i0=0*/ }, { /*i0=1*/ } })
}

"#,
    constant4_d: r#"HloModule Small_3x2x1x1_module, entry_computation_layout={()->f32[3,2,1,1]{3,2,1,0}}

ENTRY %Small_3x2x1x1.v1 () -> f32[3,2,1,1] {
  ROOT %constant = f32[3,2,1,1]{3,2,1,0} constant({ { /*i0=0*/ { /*i1=0*/ {-1} }, { /*i1=1*/ {4.1} } }, { /*i0=1*/ { /*i1=0*/ {2} }, { /*i1=1*/ {4.1} } }, { /*i0=2*/ { /*i1=0*/ {5} }, { /*i1=1*/ {4.4} } } })
}

"#,
    constant_non_finite: r#"HloModule IsFiniteR1F32s_module, entry_computation_layout={()->pred[6]{0}}

ENTRY %IsFiniteR1F32s.v2 () -> pred[6] {
  %constant = f32[6]{0} constant({nan, 7, nan, -1, inf, -inf})
  ROOT %is-finite = pred[6]{0} is-finite(f32[6]{0} %constant)
}

"#,
    constant_non_finite_e4_m3: r#"HloModule ConstantR1F8E4M3FNs_module, entry_computation_layout={()->f8e4m3fn[3]{0}}

ENTRY %IsFiniteR1F32s.v2 () -> f8e4m3fn[3] {
  ROOT %constant = f8e4m3fn[3]{0} constant({nan, 7, -nan})
}

"#,
    constant_non_finite_e4_m3_b11: r#"HloModule ConstantR1F8E4M3B11_module, entry_computation_layout={()->f8e4m3b11fnuz[2]{0}}

ENTRY %IsFiniteR1F32s.v2 () -> f8e4m3b11fnuz[2] {
  ROOT %constant = f8e4m3b11fnuz[2]{0} constant({-nan, 7})
}

"#,
    constant_f16: r#"HloModule ConstantF16_module, entry_computation_layout={()->f16[]}

ENTRY %ConstantF16.v4 () -> f16[] {
  ROOT %constant = f16[] constant(500)
}

"#,
    bf16: r#"HloModule BF16, entry_computation_layout={()->bf16[]}

ENTRY %BF16.v4 () -> bf16[] {
  ROOT %constant = bf16[] constant(500)
}

"#,
    add_constants: r#"HloModule add_constants_module, entry_computation_layout={()->f32[]}

ENTRY %add_constants () -> f32[] {
  %constant = f32[] constant(3.14)
  ROOT %add = f32[] add(f32[] %constant, f32[] %constant)
}

"#,
    tuple_constant: r#"HloModule TupleConstant_module, entry_computation_layout={()->(f32[2,1]{1,0}, f32[2]{0})}

ENTRY %TupleConstant.v1 () -> (f32[2,1], f32[2]) {
  ROOT %constant = (f32[2,1]{1,0}, f32[2]{0}) constant(( { {1}, {2} }, {2, 42} ))
}

"#,
    select_r1_f32: r#"HloModule SelectR1F32WithCmpR1F32sFromParamsSmall_module, entry_computation_layout={(f32[4]{0}, f32[4]{0})->f32[4]{0}}

ENTRY %SelectR1F32WithCmpR1F32sFromParamsSmall.v4 (v1: f32[4], v2: f32[4]) -> f32[4] {
  %v1 = f32[4]{0} parameter(0), sharding={maximal device=1}
  %v2 = f32[4]{0} parameter(1), sharding={maximal device=1}
  %greater-than = pred[4]{0} compare(f32[4]{0} %v1, f32[4]{0} %v2), direction=GT, type=TOTALORDER, sharding={replicated}
  ROOT %select = f32[4]{0} select(pred[4]{0} %greater-than, f32[4]{0} %v1, f32[4]{0} %v2), sharding={replicated}
}

"#,
    empty_tuple_create: r#"HloModule EmptyTupleCreate_module, entry_computation_layout={()->()}

ENTRY %EmptyTupleCreate.v1 () -> () {
  ROOT %tuple = () tuple()
}

"#,
    tuple_create: r#"HloModule TupleCreate_module, entry_computation_layout={(f32[], f32[3]{0}, f32[2,3]{1,0})->(f32[], f32[3]{0}, f32[2,3]{1,0})}

ENTRY %TupleCreate.v4 (v1: f32[], v2: f32[3], v3: f32[2,3]) -> (f32[], f32[3], f32[2,3]) {
  %v1 = f32[] parameter(0)
  %v2 = f32[3]{0} parameter(1)
  %v3 = f32[2,3]{1,0} parameter(2)
  ROOT %tuple = (f32[], f32[3]{0}, f32[2,3]{1,0}) tuple(f32[] %v1, f32[3]{0} %v2, f32[2,3]{1,0} %v3)
}

"#,
    large_tuple_round_trip: r#"HloModule LargeTupleRoundTrip_module, entry_computation_layout={(f32[])->(f32[], f32[], f32[], f32[], f32[], /*index=5*/f32[])}

ENTRY %TupleCreate.v4 (v: f32[]) -> (f32[], f32[], f32[], f32[], f32[], /*index=5*/f32[]) {
  %v = f32[] parameter(0)
  ROOT %tuple = (f32[], f32[], f32[], f32[], f32[], /*index=5*/f32[]) tuple(f32[] %v, f32[] %v, f32[] %v, f32[] %v, f32[] %v, /*index=5*/f32[] %v)
}

"#,
    sharded_tuple_create: r#"HloModule ShardedTupleCreate_module, entry_computation_layout={(f32[], f32[3]{0}, f32[2,3]{1,0})->(f32[], f32[3]{0}, f32[2,3]{1,0})}

ENTRY %ShardedTupleCreate.v4 (v1: f32[], v2: f32[3], v3: f32[2,3]) -> (f32[], f32[3], f32[2,3]) {
  %v1 = f32[] parameter(0), sharding={manual}
  %v2 = f32[3]{0} parameter(1)
  %v3 = f32[2,3]{1,0} parameter(2)
  ROOT %tuple = (f32[], f32[3]{0}, f32[2,3]{1,0}) tuple(f32[] %v1, f32[3]{0} %v2, f32[2,3]{1,0} %v3), sharding={{manual}, {maximal device=0}, {replicated}}
}

"#,
    domain_parsing: r#"HloModule DomainParsing_module, entry_computation_layout={(f32[])->f32[]}

ENTRY %DomainParsing (v1: f32[]) -> f32[] {
  %v1 = f32[] parameter(0)
  ROOT %dom = f32[] domain(f32[] %v1), domain={kind="sharding", entry={maximal device=0}, exit={maximal device=1}}
}

"#,
    while_with_scalar_s32_result: r#"HloModule WhileWithScalarS32Result_module, entry_computation_layout={()->s32[]}

%body.v3 (prev.1: s32[]) -> s32[] {
  %constant = s32[] constant(1)
  %prev.1 = s32[] parameter(0)
  ROOT %add = s32[] add(s32[] %constant, s32[] %prev.1)
}

%condition.v3 (prev.2: s32[]) -> pred[] {
  %constant.1 = s32[] constant(5)
  %prev.2 = s32[] parameter(0)
  ROOT %greater-than = pred[] compare(s32[] %constant.1, s32[] %prev.2), direction=GT
}

ENTRY %WhileWithScalarS32Result.v2 () -> s32[] {
  %constant.2 = s32[] constant(0)
  ROOT %while = s32[] while(s32[] %constant.2), condition=%condition.v3, body=%body.v3
}

"#,
    copy_start_and_copy_done: r#"HloModule CopyStartAndCopyDone_module, entry_computation_layout={(f32[], f32[2,3]{1,0:S(1)})->(f32[], f32[2,3]{1,0:S(2)})}

ENTRY %CopyStartAndCopyDone (v1: f32[], v2: f32[2,3]) -> (f32[], f32[2,3]) {
  %v1 = f32[] parameter(0)
  %copy-start.1 = (f32[], f32[], u32[]) copy-start(f32[] %v1), cross_program_prefetch_index=0
  %copy-done.1 = f32[] copy-done((f32[], f32[], u32[]) %copy-start.1)
  %v2 = f32[2,3]{1,0:S(1)} parameter(1)
  %copy-start.2 = (f32[2,3]{1,0:S(2)}, f32[2,3]{1,0:S(1)}, u32[]) copy-start(f32[2,3]{1,0:S(1)} %v2)
  %copy-done.2 = f32[2,3]{1,0:S(2)} copy-done((f32[2,3]{1,0:S(2)}, f32[2,3]{1,0:S(1)}, u32[]) %copy-start.2)
  ROOT %tuple = (f32[], f32[2,3]{1,0:S(2)}) tuple(f32[] %copy-done.1, f32[2,3]{1,0:S(2)} %copy-done.2)
}

"#,
    send_recv: r#"HloModule TwoSendRecvBothWayRecvFist_module, entry_computation_layout={()->(f32[], token[])}

ENTRY %TwoSendRecvBothWayRecvFist.v3 () -> (f32[], token[]) {
  %token0 = token[] after-all()
  %recv = (f32[], u32[], token[]) recv(token[] %token0), channel_id=15, sharding={{maximal device=1}, {replicated}, {replicated}}
  ROOT %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv), channel_id=15, sharding={{maximal device=1}, {replicated}}
  %constant = f32[] constant(2.1), sharding={maximal device=0}
  %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0), channel_id=16, sharding={{maximal device=1}, {replicated}, {replicated}}, control-predecessors={%recv}
  %send-done = token[] send-done((f32[], u32[], token[]) %send), channel_id=16, sharding={maximal device=0}
}

"#,
    send_recv_wo_channel_id: r#"HloModule SendRecvWoChannelID_module, entry_computation_layout={()->(f32[], token[])}

ENTRY %computation () -> (f32[], token[]) {
  %token0 = token[] after-all()
  %recv = (f32[], u32[], token[]) recv(token[] %token0)
  ROOT %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv)
  %constant = f32[] constant(2.1)
  %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0)
  %send-done = token[] send-done((f32[], u32[], token[]) %send)
}

"#,
    send_recv_with_host_transfer: r#"HloModule HostTransferSendRecv_module, entry_computation_layout={()->(f32[], token[])}

ENTRY %TwoSendRecvBothWayRecvFist.v3 () -> (f32[], token[]) {
  %token0 = token[] after-all()
  %recv = (f32[], u32[], token[]) recv(token[] %token0), channel_id=15, is_host_transfer=true
  ROOT %recv-done = (f32[], token[]) recv-done((f32[], u32[], token[]) %recv), channel_id=15, is_host_transfer=true
  %constant = f32[] constant(2.1), sharding={maximal device=0}
  %send = (f32[], u32[], token[]) send(f32[] %constant, token[] %token0), channel_id=16, is_host_transfer=true
  %send-done = token[] send-done((f32[], u32[], token[]) %send), channel_id=16, is_host_transfer=true
}

"#,
    get_tuple_element: r#"HloModule GetTupleElement_module, entry_computation_layout={()->s32[2,3]{1,0}}

ENTRY %GetTupleElement.v4 () -> s32[2,3] {
  %constant = f32[3]{0} constant({1, 2, 3})
  %constant.1 = s32[2,3]{1,0} constant({ { 1, 2, 3 }, { 4, 5, 6 } })
  %tuple = (f32[3]{0}, s32[2,3]{1,0}) tuple(f32[3]{0} %constant, s32[2,3]{1,0} %constant.1)
  ROOT %get-tuple-element = s32[2,3]{1,0} get-tuple-element((f32[3]{0}, s32[2,3]{1,0}) %tuple), index=1, sharding={maximal device=0}
}

"#,
    call: r#"HloModule CallR0F32IdentityScalar_module, entry_computation_layout={()->f32[]}

%Identity.v1 (x: f32[]) -> f32[] {
  ROOT %x = f32[] parameter(0)
}

ENTRY %CallR0F32IdentityScalar.v2 () -> f32[] {
  %constant = f32[] constant(42)
  ROOT %call = f32[] call(f32[] %constant), to_apply=%Identity.v1
}

"#,
    composite_call: r#"HloModule CompositeCall, entry_computation_layout={()->f32[]}

%add (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %constant = f32[] constant(2)
  ROOT %z = f32[] add(f32[] %x, f32[] %constant)
}

ENTRY %CompositeCall.v2 () -> f32[] {
  %constant.1 = f32[] constant(42)
  ROOT %call = f32[] call(f32[] %constant.1), to_apply=%add, is_composite=true, frontend_attributes={composite.attributes={n = 1 : i32, tensor = dense<1> : tensor<i32>},composite.name="foo.bar",composite.version="1"}
}

"#,
    composite_call_with_extra_frontend_attributes: r#"HloModule CompositeCall, entry_computation_layout={()->f32[]}

%add (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %constant = f32[] constant(2)
  ROOT %z = f32[] add(f32[] %x, f32[] %constant)
}

ENTRY %CompositeCall.v2 () -> f32[] {
  %constant.1 = f32[] constant(42)
  ROOT %call = f32[] call(f32[] %constant.1), to_apply=%add, is_composite=true, frontend_attributes={composite.attributes={n = 1 : i32, tensor = dense<1> : tensor<i32>},composite.name="foo.bar",composite.version="1",foo="bar"}
}

"#,
    composite_call_optional_attributes_and_version: r#"HloModule CompositeCall, entry_computation_layout={()->f32[]}

%add (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %constant = f32[] constant(2)
  ROOT %z = f32[] add(f32[] %x, f32[] %constant)
}

ENTRY %CompositeCall.v2 () -> f32[] {
  %constant.1 = f32[] constant(42)
  ROOT %call = f32[] call(f32[] %constant.1), to_apply=%add, is_composite=true, frontend_attributes={composite.name="foo.bar"}
}

"#,
    composite_call_optional_attributes: r#"HloModule CompositeCall, entry_computation_layout={()->f32[]}

%add (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %constant = f32[] constant(2)
  ROOT %z = f32[] add(f32[] %x, f32[] %constant)
}

ENTRY %CompositeCall.v2 () -> f32[] {
  %constant.1 = f32[] constant(42)
  ROOT %call = f32[] call(f32[] %constant.1), to_apply=%add, is_composite=true, frontend_attributes={composite.name="foo.bar",composite.version="1"}
}

"#,
    composite_call_optional_version: r#"HloModule CompositeCall, entry_computation_layout={()->f32[]}

%add (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %constant = f32[] constant(2)
  ROOT %z = f32[] add(f32[] %x, f32[] %constant)
}

ENTRY %CompositeCall.v2 () -> f32[] {
  %constant.1 = f32[] constant(42)
  ROOT %call = f32[] call(f32[] %constant.1), to_apply=%add, is_composite=true, frontend_attributes={composite.attributes={n = 1 : i32, tensor = dense<1> : tensor<i32>},composite.name="foo.bar"}
}

"#,
    custom_call_with_opaque: r#"HloModule custom_call, entry_computation_layout={()->f32[1,2,3]{0,2,1}}

ENTRY %CustomCall () -> f32[1,2,3] {
  %constant = f32[1]{0} constant({12345})
  ROOT %custom-call = f32[1,2,3]{0,2,1} custom-call(f32[1]{0} %constant), custom_call_target="foo\"bar", backend_config="this string is opaque"
}

"#,
    custom_call_with_backend_config_in_curly_braces: r#"HloModule custom_call, entry_computation_layout={()->f32[1,2,3]{0,2,1}}

ENTRY %CustomCall () -> f32[1,2,3] {
  %constant = f32[1]{0} constant({12345})
  ROOT %custom-call = f32[1,2,3]{0,2,1} custom-call(f32[1]{0} %constant), custom_call_target="foo\"bar", backend_config={key: "value"}
}

"#,
    custom_call_with_literal: r#"HloModule custom_call, entry_computation_layout={()->f32[1,2,3]{0,2,1}}

ENTRY %CustomCall () -> f32[1,2,3] {
  %constant = f32[1]{0} constant({12345})
  ROOT %custom-call = f32[1,2,3]{0,2,1} custom-call(f32[1]{0} %constant), custom_call_target="foo\"bar", literal=s32[2]{0} {1, 2}
}

"#,
    custom_call_with_literal_tuple: r#"HloModule custom_call, entry_computation_layout={()->f32[1,2,3]{0,2,1}}

ENTRY %CustomCall () -> f32[1,2,3] {
  %constant = f32[1]{0} constant({12345})
  ROOT %custom-call = f32[1,2,3]{0,2,1} custom-call(f32[1]{0} %constant), custom_call_target="foo\"bar", literal=( s32[4]{0} {4, 128, 128, 3}, pred[4]{0} {1, 0, 0, 0} )
}

"#,
    custom_call_with_literal_r0: r#"HloModule custom_call, entry_computation_layout={()->f32[1,2,3]{0,2,1}}

ENTRY %CustomCall () -> f32[1,2,3] {
  %constant = f32[1]{0} constant({12345})
  ROOT %custom-call = f32[1,2,3]{0,2,1} custom-call(f32[1]{0} %constant), custom_call_target="foo\"bar", literal=f32[] 0.1
}

"#,
    reduce_window: r#"HloModule R4UnitWindow_module, entry_computation_layout={(f32[13,12,8,15]{0,3,2,1})->f32[13,3,8,15]{0,3,2,1}}

%add_F32.v3 (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %lhs, f32[] %rhs)
}

ENTRY %R4UnitWindow.v3 (operand: f32[13,12,8,15]) -> f32[13,3,8,15] {
  %operand = f32[13,12,8,15]{0,3,2,1} parameter(0)
  %constant = f32[] constant(0)
  ROOT %reduce-window = f32[13,3,8,15]{0,3,2,1} reduce-window(f32[13,12,8,15]{0,3,2,1} %operand, f32[] %constant), window={size=1x1x7x1 stride=1x4x1x1 pad=0_0x0_0x3_3x0_0}, to_apply=%add_F32.v3
}

"#,
    reduce_window_scalar: r#"HloModule reduce_window_scalar, entry_computation_layout={()->f32[]}

%add_F32.v3 (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %lhs, f32[] %rhs)
}

ENTRY %R4UnitWindowScalar () -> f32[] {
  %constant = f32[] constant(42)
  %constant.1 = f32[] constant(1)
  ROOT %reduce-window = f32[] reduce-window(f32[] %constant, f32[] %constant.1), to_apply=%add_F32.v3
}

"#,
    reduce_window_variadic: r#"HloModule reduce_window_variadic, entry_computation_layout={()->(f32[], f32[])}

%add_F32.v3 (lhs1: f32[], lhs2: f32[], rhs1: f32[], rhs2: f32[]) -> (f32[], f32[]) {
  %lhs1 = f32[] parameter(0)
  %rhs1 = f32[] parameter(2)
  %add1 = f32[] add(f32[] %lhs1, f32[] %rhs1)
  %lhs2 = f32[] parameter(1)
  %rhs2 = f32[] parameter(3)
  %add2 = f32[] add(f32[] %lhs2, f32[] %rhs2)
  ROOT %tuple1 = (f32[], f32[]) tuple(f32[] %add1, f32[] %add2)
}

ENTRY %R4UnitWindowScalar () -> (f32[], f32[]) {
  %constant = f32[] constant(42)
  %constant.1 = f32[] constant(1)
  ROOT %reduce-window = (f32[], f32[]) reduce-window(f32[] %constant, f32[] %constant, f32[] %constant.1, f32[] %constant.1), to_apply=%add_F32.v3
}

"#,
    convolution: r#"HloModule Convolve1D1Window_0_module, entry_computation_layout={(f32[1,2,1]{2,1,0}, f32[1,1,1]{2,1,0})->f32[1,2,1]{2,0,1}}

ENTRY %Convolve1D1Window_0.v3 (input: f32[1,2,1], filter: f32[1,1,1]) -> f32[1,2,1] {
  %input = f32[1,2,1]{2,1,0} parameter(0)
  %copy = f32[1,2,1]{2,0,1} copy(f32[1,2,1]{2,1,0} %input)
  %filter = f32[1,1,1]{2,1,0} parameter(1)
  ROOT %convolution = f32[1,2,1]{2,0,1} convolution(f32[1,2,1]{2,0,1} %copy, f32[1,1,1]{2,1,0} %filter), window={size=1}, dim_labels=b0f_0io->b0f, operand_precision={high,default}
}

"#,
    convolution_dynamic: r#"HloModule Convolve1D1Window_0_module, entry_computation_layout={(f32[1,2,1]{2,1,0}, f32[1,1,1]{2,1,0})->f32[1,2,1]{2,0,1}}

ENTRY %Convolve1D1Window_0.v3 (input: f32[1,2,1], filter: f32[1,1,1]) -> f32[1,2,1] {
  %input = f32[1,2,1]{2,1,0} parameter(0)
  %copy = f32[1,2,1]{2,0,1} copy(f32[1,2,1]{2,1,0} %input)
  %filter = f32[1,1,1]{2,1,0} parameter(1)
  ROOT %custom-call.52 = f32[1,2,1]{2,0,1} custom-call(f32[1,2,1]{2,0,1} %copy, f32[1,1,1]{2,1,0} %filter), window={size=1}, dim_labels=b0f_0io->b0f, operand_precision={high,default}, custom_call_target="DynamicConvolutionForward", metadata={op_type="Conv2D" op_name="conv1d"}
}

"#,
    convolution_r2: r#"HloModule ConvolveR2_module, entry_computation_layout={(f32[1,2]{1,0}, f32[2,2]{1,0})->f32[1,2]{0,1}}

ENTRY %ConvolveR2.v3 (input: f32[1,2], filter: f32[2,2]) -> f32[1,2] {
  %input = f32[1,2]{1,0} parameter(0)
  %filter = f32[2,2]{1,0} parameter(1)
  ROOT %convolution = f32[1,2]{0,1} convolution(f32[1,2]{1,0} %input, f32[2,2]{1,0} %filter), dim_labels=bf_io->bf
}

"#,
    convolution_backward: r#"HloModule ConvolveBackward_module, entry_computation_layout={(f32[128,7,7,512]{0,3,2,1}, f32[3,3,512,512]{3,2,1,0})->f32[128,14,14,512]{0,3,2,1}}

ENTRY %ConvolveBackward (input: f32[128,7,7,512], filter: f32[3,3,512,512]) -> f32[128,14,14,512] {
  %input = f32[128,7,7,512]{0,3,2,1} parameter(0)
  %filter = f32[3,3,512,512]{3,2,1,0} parameter(1)
  ROOT %convolution-base-dilated = f32[128,14,14,512]{0,3,2,1} convolution(f32[128,7,7,512]{0,3,2,1} %input, f32[3,3,512,512]{3,2,1,0} %filter), window={size=3x3 pad=1_2x1_2 lhs_dilate=2x2 rhs_reversal=1x1}, dim_labels=b01f_01oi->b01f
}

"#,
    reverse4_d: r#"HloModule Reverse4DFloatArrayOnDim01_module, entry_computation_layout={()->f32[4,3,2,1]{0,1,2,3}}

ENTRY %Reverse4DFloatArrayOnDim01.v2 () -> f32[4,3,2,1] {
  %constant = f32[4,3,2,1]{0,1,2,3} constant({ { /*i0=0*/ { /*i1=0*/ {1}, {2} }, { /*i1=1*/ {3}, {4} }, { /*i1=2*/ {5}, {6} } }, { /*i0=1*/ { /*i1=0*/ {7}, {8} }, { /*i1=1*/ {9}, {10} }, { /*i1=2*/ {11}, {12} } }, { /*i0=2*/ { /*i1=0*/ {13}, {14} }, { /*i1=1*/ {15}, {16} }, { /*i1=2*/ {17}, {18} } }, { /*i0=3*/ { /*i1=0*/ {19}, {20} }, { /*i1=1*/ {21}, {22} }, { /*i1=2*/ {23}, {24} } } })
  ROOT %reverse = f32[4,3,2,1]{0,1,2,3} reverse(f32[4,3,2,1]{0,1,2,3} %constant), dimensions={0,1}
}

"#,
    concat: r#"HloModule Concat2x3With2x5_module, entry_computation_layout={()->f32[2,8]{1,0}}

ENTRY %Concat2x3With2x5.v3 () -> f32[2,8] {
  %constant = f32[2,3]{1,0} constant({ { 0, 1, 2 }, { 1000, 1001, 1002 } })
  %constant.1 = f32[2,5]{1,0} constant({ { 64, 65, 66, 67, 68 }, { 1064, 1065, 1066, 1067, 1068 } })
  ROOT %concatenate = f32[2,8]{1,0} concatenate(f32[2,3]{1,0} %constant, f32[2,5]{1,0} %constant.1), dimensions={1}
}

"#,
    select_and_scatter: r#"HloModule R4F32OverlapSmall_module, entry_computation_layout={()->f32[4,5,1,1]{3,2,1,0}}

%ge_F32.v3 (lhs: f32[], rhs: f32[]) -> pred[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %greater-than-or-equal-to = pred[] compare(f32[] %lhs, f32[] %rhs), direction=GE, type=TOTALORDER
}

%add_F32.v3 (lhs.1: f32[], rhs.1: f32[]) -> f32[] {
  %lhs.1 = f32[] parameter(0)
  %rhs.1 = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %lhs.1, f32[] %rhs.1)
}

ENTRY %R4F32OverlapSmall.v4 () -> f32[4,5,1,1] {
  %constant = f32[4,5,1,1]{3,2,1,0} constant({ { /*i0=0*/ { /*i1=0*/ {7} }, { /*i1=1*/ {2} }, { /*i1=2*/ {5} }, { /*i1=3*/ {3} }, { /*i1=4*/ {8} } }, { /*i0=1*/ { /*i1=0*/ {3} }, { /*i1=1*/ {8} }, { /*i1=2*/ {9} }, { /*i1=3*/ {3} }, { /*i1=4*/ {4} } }, { /*i0=2*/ { /*i1=0*/ {1} }, { /*i1=1*/ {5} }, { /*i1=2*/ {7} }, { /*i1=3*/ {5} }, { /*i1=4*/ {6} } }, { /*i0=3*/ { /*i1=0*/ {0} }, { /*i1=1*/ {6} }, { /*i1=2*/ {2} }, { /*i1=3*/ {10} }, { /*i1=4*/ {2} } } })
  %constant.1 = f32[2,2,1,1]{3,2,1,0} constant({ { /*i0=0*/ { /*i1=0*/ {2} }, { /*i1=1*/ {6} } }, { /*i0=1*/ { /*i1=0*/ {3} }, { /*i1=1*/ {1} } } })
  %constant.2 = f32[] constant(0)
  ROOT %select-and-scatter = f32[4,5,1,1]{3,2,1,0} select-and-scatter(f32[4,5,1,1]{3,2,1,0} %constant, f32[2,2,1,1]{3,2,1,0} %constant.1, f32[] %constant.2), window={size=2x3x1x1 stride=2x2x1x1}, select=%ge_F32.v3, scatter=%add_F32.v3
}

"#,
    select_and_scatter_scalar: r#"HloModule select_and_scatter_scalar, entry_computation_layout={()->f32[]}

%ge_F32.v3 (lhs: f32[], rhs: f32[]) -> pred[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %greater-than-or-equal-to = pred[] compare(f32[] %lhs, f32[] %rhs), direction=GE
}

%add_F32.v3 (lhs.1: f32[], rhs.1: f32[]) -> f32[] {
  %lhs.1 = f32[] parameter(0)
  %rhs.1 = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %lhs.1, f32[] %rhs.1)
}

ENTRY %SelectAndScatterScalar () -> f32[] {
  %constant = f32[] constant(42)
  %constant.1 = f32[] constant(1)
  %constant.2 = f32[] constant(2)
  ROOT %select-and-scatter = f32[] select-and-scatter(f32[] %constant, f32[] %constant.1, f32[] %constant.2), select=%ge_F32.v3, scatter=%add_F32.v3
}

"#,
    slice: r#"HloModule slice_module, entry_computation_layout={(f32[3,3,4,4]{3,2,1,0})->f32[3,3,2,4]{3,2,1,0}}

ENTRY %slice.v2 (p0: f32[3,3,4,4]) -> f32[3,3,2,4] {
  %p0 = f32[3,3,4,4]{3,2,1,0} parameter(0)
  ROOT %slice = f32[3,3,2,4]{3,2,1,0} slice(f32[3,3,4,4]{3,2,1,0} %p0), slice={[0:3:1], [0:3:1], [0:4:2], [0:4:1]}
}

"#,
    slice_no_stride: r#"HloModule Slice3x3x3_To_1x3x3_F32_module, entry_computation_layout={()->f32[1,3,3]{2,1,0}}

ENTRY %Slice3x3x3_To_1x3x3_F32.v2 () -> f32[1,3,3] {
  %constant = f32[3,3,3]{2,1,0} constant({ { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } }, { { 9, 10, 11 }, { 12, 13, 14 }, { 15, 16, 17 } }, { { 18, 19, 20 }, { 21, 22, 23 }, { 24, 25, 26 } } })
  ROOT %slice = f32[1,3,3]{2,1,0} slice(f32[3,3,3]{2,1,0} %constant), slice={[0:1], [0:3], [0:3]}
}

"#,
    slice_r0: r#"HloModule SliceR0_module, entry_computation_layout={()->s32[]}

ENTRY %SliceR0.v2 () -> s32[] {
  %constant = s32[] constant(1)
  ROOT %slice = s32[] slice(s32[] %constant), slice={}
}

"#,
    transpose: r#"HloModule Transpose_module, entry_computation_layout={()->s32[1,2,3]{2,1,0}}

ENTRY %Transpose.v2 () -> s32[1,2,3] {
  %constant = s32[1,2,3]{2,1,0} constant({ { { 1, 2, 3 }, { 4, 5, 6 } } })
  ROOT %transpose = s32[1,2,3]{2,1,0} transpose(s32[1,2,3]{2,1,0} %constant), dimensions={0,1,2}
}

"#,
    transpose_c128: r#"HloModule TransposeC128_module, entry_computation_layout={(c128[1,2,3]{2,1,0})->c128[1,2,3]{2,1,0}}

ENTRY %Transpose.v3 (input: c128[1,2,3]) -> c128[1,2,3] {
  %input = c128[1,2,3]{2,1,0} parameter(0)
  ROOT %transpose = c128[1,2,3]{2,1,0} transpose(c128[1,2,3]{2,1,0} %input), dimensions={0,1,2}
}

"#,
    triangular_solve: r#"HloModule TriangularSolve_module, entry_computation_layout={(f32[4,4]{1,0}, f32[3,4]{1,0})->f32[3,4]{1,0}}

ENTRY %SimpleRightLowerNotranspose.4 (a.1: f32[4,4], b.2: f32[3,4]) -> f32[3,4] {
  %a.1 = f32[4,4]{1,0} parameter(0)
  %b.2 = f32[3,4]{1,0} parameter(1)
  ROOT %triangular-solve.3 = f32[3,4]{1,0} triangular-solve(f32[4,4]{1,0} %a.1, f32[3,4]{1,0} %b.2), lower=true, transpose_a=NO_TRANSPOSE
}

"#,
    dynamic_slice: r#"HloModule DynamicSlice_module, entry_computation_layout={(s32[2,2,258]{2,1,0}, s32[1]{0})->s32[2,2,258]{2,1,0}}

ENTRY %DynamicSlice.v5 (original_parameter: s32[2,2,258], start_index: s32[1]) -> s32[2,2,258] {
  %original_parameter = s32[2,2,258]{2,1,0} parameter(0)
  %constant = s32[1]{0} constant({0})
  %start_index = s32[1]{0} parameter(1)
  %concatenate = s32[3]{0} concatenate(s32[1]{0} %constant, s32[1]{0} %constant, s32[1]{0} %start_index), dimensions={0}
  ROOT %dynamic-slice = s32[2,2,258]{2,1,0} dynamic-slice(s32[2,2,258]{2,1,0} %original_parameter, s32[3]{0} %concatenate), dynamic_slice_sizes={2,2,258}
}

"#,
    dynamic_slice_scalar_indices: r#"HloModule DynamicSlice_module, entry_computation_layout={(s32[2,2,258]{2,1,0}, s32[])->s32[2,2,258]{2,1,0}}

ENTRY %DynamicSlice.v5 (original_parameter: s32[2,2,258], start_index: s32[]) -> s32[2,2,258] {
  %original_parameter = s32[2,2,258]{2,1,0} parameter(0)
  %constant = s32[] constant(0)
  %start_index = s32[] parameter(1)
  ROOT %dynamic-slice = s32[2,2,258]{2,1,0} dynamic-slice(s32[2,2,258]{2,1,0} %original_parameter, s32[] %constant, s32[] %constant, s32[] %start_index), dynamic_slice_sizes={2,2,258}
}

"#,
    dynamic_update_slice: r#"HloModule DynamicSlice_module, entry_computation_layout={(s32[1,1,25,1]{3,2,1,0}, s32[1,1,2,1]{3,2,1,0}, s32[4]{0})->s32[1,1,25,1]{3,2,1,0}}

ENTRY %DynamicUpdateSlice.v4 (input: s32[1,1,25,1], update: s32[1,1,2,1], start_indices: s32[4]) -> s32[1,1,25,1] {
  %input = s32[1,1,25,1]{3,2,1,0} parameter(0)
  %update = s32[1,1,2,1]{3,2,1,0} parameter(1)
  %start_indices = s32[4]{0} parameter(2)
  ROOT %dynamic-update-slice = s32[1,1,25,1]{3,2,1,0} dynamic-update-slice(s32[1,1,25,1]{3,2,1,0} %input, s32[1,1,2,1]{3,2,1,0} %update, s32[4]{0} %start_indices)
}

"#,
    dynamic_update_slice_scalar_index: r#"HloModule DynamicUpdateSlice_module, entry_computation_layout={(s32[1,1,25,1]{3,2,1,0}, s32[1,1,2,1]{3,2,1,0}, s32[], s32[], s32[], /*index=5*/s32[])->s32[1,1,25,1]{3,2,1,0}}

ENTRY %DynamicUpdateSlice.v4 (input: s32[1,1,25,1], update: s32[1,1,2,1], start_index.0: s32[], start_index.1: s32[], start_index.2: s32[], start_index.3: s32[]) -> s32[1,1,25,1] {
  %input = s32[1,1,25,1]{3,2,1,0} parameter(0)
  %update = s32[1,1,2,1]{3,2,1,0} parameter(1)
  %start_index.0 = s32[] parameter(2)
  %start_index.1 = s32[] parameter(3)
  %start_index.2 = s32[] parameter(4)
  %start_index.3 = s32[] parameter(5)
  ROOT %dynamic-update-slice = s32[1,1,25,1]{3,2,1,0} dynamic-update-slice(s32[1,1,25,1]{3,2,1,0} %input, s32[1,1,2,1]{3,2,1,0} %update, s32[] %start_index.0, s32[] %start_index.1, s32[] %start_index.2, /*index=5*/s32[] %start_index.3)
}

"#,
    batch_norm_training: r#"HloModule BasicTraining_module, entry_computation_layout={()->(f32[2,2,1,2]{3,2,1,0}, f32[2]{0}, f32[2]{0})}

ENTRY %BasicTraining.v4 () -> (f32[2,2,1,2], f32[2], f32[2]) {
  %constant = f32[2,2,1,2]{3,2,1,0} constant({ { /*i0=0*/ { /*i1=0*/ { 1, 2 } }, { /*i1=1*/ { 3, 4 } } }, { /*i0=1*/ { /*i1=0*/ { 5, 6 } }, { /*i1=1*/ { 7, 8 } } } })
  %constant.1 = f32[2]{0} constant({2, 3})
  %constant.2 = f32[2]{0} constant({1, 2})
  ROOT %batch-norm-training = (f32[2,2,1,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-training(f32[2,2,1,2]{3,2,1,0} %constant, f32[2]{0} %constant.1, f32[2]{0} %constant.2), epsilon=0.001, feature_index=3
}

"#,
    batch_norm_inference: r#"HloModule BatchNormInference_module, entry_computation_layout={(f32[2,2,2,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}, f32[2]{0}, f32[2]{0})->f32[2,2,2,2]{3,2,1,0}}

ENTRY %BatchNormInference.v6 (input: f32[2,2,2,2], offset: f32[2], scale: f32[2], mean: f32[2], variance: f32[2]) -> f32[2,2,2,2] {
  %input = f32[2,2,2,2]{3,2,1,0} parameter(0)
  %offset = f32[2]{0} parameter(1)
  %scale = f32[2]{0} parameter(2)
  %mean = f32[2]{0} parameter(3)
  %variance = f32[2]{0} parameter(4)
  ROOT %batch-norm-inference = f32[2,2,2,2]{3,2,1,0} batch-norm-inference(f32[2,2,2,2]{3,2,1,0} %input, f32[2]{0} %offset, f32[2]{0} %scale, f32[2]{0} %mean, f32[2]{0} %variance), epsilon=0.001, feature_index=0
}

"#,
    batch_norm_grad: r#"HloModule BatchNormGrad_module, entry_computation_layout={(f32[2,2,2,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}, f32[2]{0}, f32[2,2,2,2]{3,2,1,0})->(f32[2,2,2,2]{3,2,1,0}, f32[2]{0}, f32[2]{0})}

ENTRY %BatchNormGrad.v4 (input: f32[2,2,2,2], scale: f32[2], mean: f32[2], variance: f32[2], grad_output: f32[2,2,2,2]) -> (f32[2,2,2,2], f32[2], f32[2]) {
  %input = f32[2,2,2,2]{3,2,1,0} parameter(0)
  %scale = f32[2]{0} parameter(1)
  %mean = f32[2]{0} parameter(2)
  %variance = f32[2]{0} parameter(3)
  %grad_output = f32[2,2,2,2]{3,2,1,0} parameter(4)
  ROOT %batch-norm-grad = (f32[2,2,2,2]{3,2,1,0}, f32[2]{0}, f32[2]{0}) batch-norm-grad(f32[2,2,2,2]{3,2,1,0} %input, f32[2]{0} %scale, f32[2]{0} %mean, f32[2]{0} %variance, f32[2,2,2,2]{3,2,1,0} %grad_output), epsilon=0.001, feature_index=0
}

"#,
    fft: r#"HloModule Fft_module, entry_computation_layout={(c64[8,32]{1,0})->c64[8,32]{1,0}}

ENTRY %Fft (input: c64[8,32]) -> c64[8,32] {
  %input = c64[8,32]{1,0} parameter(0)
  ROOT %fft = c64[8,32]{1,0} fft(c64[8,32]{1,0} %input), fft_type=FFT, fft_length={32}
}

"#,
    ifft2d: r#"HloModule Ifft2d_module, entry_computation_layout={(c64[5,8,32]{2,1,0})->c64[5,8,32]{2,1,0}}

ENTRY %Ifft2d (input: c64[5,8,32]) -> c64[5,8,32] {
  %input = c64[5,8,32]{2,1,0} parameter(0)
  ROOT %fft = c64[5,8,32]{2,1,0} fft(c64[5,8,32]{2,1,0} %input), fft_type=IFFT, fft_length={8,32}
}

"#,
    rfft2d: r#"HloModule Rfft2d_module, entry_computation_layout={(f32[5,64,32]{2,1,0})->c64[5,64,17]{2,1,0}}

ENTRY %Rfft2d (input: f32[5,64,32]) -> c64[5,64,17] {
  %input = f32[5,64,32]{2,1,0} parameter(0)
  ROOT %fft = c64[5,64,17]{2,1,0} fft(f32[5,64,32]{2,1,0} %input), fft_type=RFFT, fft_length={64,32}
}

"#,
    irfft3d: r#"HloModule Irfft3d_module, entry_computation_layout={(c64[5,64,128,33]{3,2,1,0})->f32[5,64,128,64]{3,2,1,0}}

ENTRY %Irfft3d (input: c64[5,64,128,33]) -> f32[5,64,128,64] {
  %input = c64[5,64,128,33]{3,2,1,0} parameter(0)
  ROOT %fft = f32[5,64,128,64]{3,2,1,0} fft(c64[5,64,128,33]{3,2,1,0} %input), fft_type=IRFFT, fft_length={64,128,64}
}

"#,
    pad: r#"HloModule Pad1DS3Array_module, entry_computation_layout={()->f32[7]{0}}

ENTRY %Pad1DS3Array.v3 () -> f32[7] {
  %constant = f32[3]{0} constant({1, 2, 3})
  %constant.1 = f32[] constant(0.1)
  ROOT %pad = f32[7]{0} pad(f32[3]{0} %constant, f32[] %constant.1), padding=3_1
}

"#,
    pad_has_interior: r#"HloModule PadHasInterior_module, entry_computation_layout={(f32[1,25,7,7]{3,2,1,0})->f32[1,25,17,11]{3,2,1,0}}

ENTRY %PadHasInterior.v3 (input: f32[1,25,7,7]) -> f32[1,25,17,11] {
  %input = f32[1,25,7,7]{3,2,1,0} parameter(0)
  %constant = f32[] constant(-5.123)
  ROOT %pad = f32[1,25,17,11]{3,2,1,0} pad(f32[1,25,7,7]{3,2,1,0} %input, f32[] %constant), padding=0_0_0x0_0_0x2_2_1x2_2_0
}

"#,
    round_nearest_even: r#"HloModule RoundNearestEven_module, entry_computation_layout={(f32[2,2]{1,0})->f32[2,2]{1,0}}

ENTRY %RoundNearestEven (input: f32[2,2]) -> f32[2,2] {
  %input = f32[2,2]{1,0} parameter(0)
  ROOT %round-nearest-even = f32[2,2]{1,0} round-nearest-even(f32[2,2]{1,0} %input)
}

"#,
    pad_has_negative_padding: r#"HloModule PadHasNegativePadding_module, entry_computation_layout={(f32[1,25,7,7,10]{4,3,2,1,0})->f32[1,15,6,3,35]{4,3,2,1,0}}

ENTRY %PadHasNegativePadding (input: f32[1,25,7,7,10]) -> f32[1,15,6,3,35] {
  %input = f32[1,25,7,7,10]{4,3,2,1,0} parameter(0)
  %constant = f32[] constant(-5.123)
  ROOT %pad = f32[1,15,6,3,35]{4,3,2,1,0} pad(f32[1,25,7,7,10]{4,3,2,1,0} %input, f32[] %constant), padding=0_0_0x0_-10_0x0_-1_0x-2_-2_0x-1_-1_3
}

"#,
    fusion: r#"HloModule fusion_module, entry_computation_layout={()->f32[3,2,1,1]{3,2,1,0}}

%fused_computation (constant.param_0: f32[3,2,1,1], constant.1.param_1: f32[2]) -> f32[3,2,1,1] {
  %constant.param_0 = f32[3,2,1,1]{3,2,1,0} parameter(0)
  %constant.1.param_1 = f32[2]{0} parameter(1)
  %broadcast = f32[3,2,1,1]{3,2,1,0} broadcast(f32[2]{0} %constant.1.param_1), dimensions={1}
  ROOT %subtract = f32[3,2,1,1]{3,2,1,0} subtract(f32[3,2,1,1]{3,2,1,0} %constant.param_0, f32[3,2,1,1]{3,2,1,0} %broadcast)
}

ENTRY %fusion.v3 () -> f32[3,2,1,1] {
  %constant = f32[3,2,1,1]{3,2,1,0} constant({ { /*i0=0*/ { /*i1=0*/ {-1} }, { /*i1=1*/ {4.1} } }, { /*i0=1*/ { /*i1=0*/ {2} }, { /*i1=1*/ {4.1} } }, { /*i0=2*/ { /*i1=0*/ {5} }, { /*i1=1*/ {4.4} } } })
  %constant.1 = f32[2]{0} constant({3.14, 4.25})
  ROOT %fusion = f32[3,2,1,1]{3,2,1,0} fusion(f32[3,2,1,1]{3,2,1,0} %constant, f32[2]{0} %constant.1), kind=kLoop, calls=%fused_computation
}

"#,
    async_start_with_aliasing: r#"HloModule module, entry_computation_layout={(f32[8,4,1]{0,1,2:T(4,128)})->f32[8,4,1]{0,1,2:T(4,128)}}

%async_computation (param_0.2: (f32[8,4,1], (f32[8,4,1], u32[], u32[]))) -> f32[8,4,1] {
  %param_0.2 = (f32[8,4,1]{1,2,0:T(1,128)}, (f32[8,4,1]{1,2,0:T(1,128)}, u32[]{:S(2)}, u32[]{:S(2)})) parameter(0)
  %get-tuple-element = f32[8,4,1]{1,2,0:T(1,128)} get-tuple-element((f32[8,4,1]{1,2,0:T(1,128)}, (f32[8,4,1]{1,2,0:T(1,128)}, u32[]{:S(2)}, u32[]{:S(2)})) %param_0.2), index=0
  ROOT %all-to-all0.0 = f32[8,4,1]{1,2,0:T(1,128)} all-to-all(f32[8,4,1]{1,2,0:T(1,128)} %get-tuple-element), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}
}

ENTRY %Comp_spmd (param: f32[8,4,1]) -> f32[8,4,1] {
  %param = f32[8,4,1]{0,1,2:T(4,128)} parameter(0)
  %copy = f32[8,4,1]{1,2,0:T(1,128)} copy(f32[8,4,1]{0,1,2:T(4,128)} %param)
  %custom-call = (f32[8,4,1]{1,2,0:T(1,128)}, u32[]{:S(2)}, u32[]{:S(2)}) custom-call(), custom_call_target="BarrierStart"
  %tuple = (f32[8,4,1]{1,2,0:T(1,128)}, (f32[8,4,1]{1,2,0:T(1,128)}, u32[]{:S(2)}, u32[]{:S(2)})) tuple(f32[8,4,1]{1,2,0:T(1,128)} %copy, (f32[8,4,1]{1,2,0:T(1,128)}, u32[]{:S(2)}, u32[]{:S(2)}) %custom-call)
  %all-to-all-start.1 = (((f32[8,4,1]{1,2,0:T(1,128)}, (f32[8,4,1]{1,2,0:T(1,128)}, u32[]{:S(2)}, u32[]{:S(2)}))), f32[8,4,1]{1,2,0:T(1,128)}, u32[]{:S(2)}, u32[]{:S(2)}) async-start((f32[8,4,1]{1,2,0:T(1,128)}, (f32[8,4,1]{1,2,0:T(1,128)}, u32[]{:S(2)}, u32[]{:S(2)})) %tuple), output_to_operand_aliasing={{0,0,1,0}: (0, {1,0}), {1}: (0, {1,0}), {0,0,1,1}: (0, {1,1}), {2}: (0, {1,1}), {0,0,1,2}: (0, {1,2}), {3}: (0, {1,2})}, calls=%async_computation
  %all-to-all-done = f32[8,4,1]{1,2,0:T(1,128)} async-done((((f32[8,4,1]{1,2,0:T(1,128)}, (f32[8,4,1]{1,2,0:T(1,128)}, u32[]{:S(2)}, u32[]{:S(2)}))), f32[8,4,1]{1,2,0:T(1,128)}, u32[]{:S(2)}, u32[]{:S(2)}) %all-to-all-start.1)
  ROOT %copy.1 = f32[8,4,1]{0,1,2:T(4,128)} copy(f32[8,4,1]{1,2,0:T(1,128)} %all-to-all-done)
}

"#,
    fusion_with_aliasing: r#"HloModule FusionWithAliasing, entry_computation_layout={((f32[2,2]{0,1}, f32[42,2,3]{0,1,2}), f32[123,4]{0,1})->(f32[123,4]{0,1}, f32[2,2]{0,1}, f32[1,2,3]{0,1,2})}

%FusedComp (p0: (f32[2,2], f32[42,2,3]), p1: f32[123,4]) -> (f32[123,4], f32[2,2], f32[1,2,3]) {
  %p1 = f32[123,4]{0,1} parameter(1)
  %p0 = (f32[2,2]{0,1}, f32[42,2,3]{0,1,2}) parameter(0)
  %elem1 = f32[2,2]{0,1} get-tuple-element((f32[2,2]{0,1}, f32[42,2,3]{0,1,2}) %p0), index=0
  %constant0 = f32[] constant(1)
  %broadcast0 = f32[1,2,3]{0,1,2} broadcast(f32[] %constant0), dimensions={}
  ROOT %tuple = (f32[123,4]{0,1}, f32[2,2]{0,1}, f32[1,2,3]{0,1,2}) tuple(f32[123,4]{0,1} %p1, f32[2,2]{0,1} %elem1, f32[1,2,3]{0,1,2} %broadcast0)
}

ENTRY %FusionWithAliasing (p0.1: (f32[2,2], f32[42,2,3]), p1.1: f32[123,4]) -> (f32[123,4], f32[2,2], f32[1,2,3]) {
  %p0.1 = (f32[2,2]{0,1}, f32[42,2,3]{0,1,2}) parameter(0)
  %p1.1 = f32[123,4]{0,1} parameter(1)
  ROOT %fusion = (f32[123,4]{0,1}, f32[2,2]{0,1}, f32[1,2,3]{0,1,2}) fusion((f32[2,2]{0,1}, f32[42,2,3]{0,1,2}) %p0.1, f32[123,4]{0,1} %p1.1), kind=kLoop, output_to_operand_aliasing={{0}: (1, {}), {1}: (0, {0})}, calls=%FusedComp
}

"#,
    gather: r#"HloModule StringifyGather, entry_computation_layout={(f32[50,49,48,47,46]{4,3,2,1,0}, s64[10,9,8,7,5]{4,3,2,1,0})->f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0}}

ENTRY %Gather (input_tensor: f32[50,49,48,47,46], start_indices: s64[10,9,8,7,5]) -> f32[10,9,8,7,30,29,28,27,26] {
  %input_tensor = f32[50,49,48,47,46]{4,3,2,1,0} parameter(0)
  %start_indices = s64[10,9,8,7,5]{4,3,2,1,0} parameter(1)
  ROOT %gather = f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} gather(f32[50,49,48,47,46]{4,3,2,1,0} %input_tensor, s64[10,9,8,7,5]{4,3,2,1,0} %start_indices), offset_dims={4,5,6,7,8}, collapsed_slice_dims={}, start_index_map={0,1,2,3,4}, index_vector_dim=4, slice_sizes={30,29,28,27,26}
}

"#,
    sorted_gather: r#"HloModule StringifyGather, entry_computation_layout={(f32[50,49,48,47,46]{4,3,2,1,0}, s64[10,9,8,7,5]{4,3,2,1,0})->f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0}}

ENTRY %Gather (input_tensor: f32[50,49,48,47,46], start_indices: s64[10,9,8,7,5]) -> f32[10,9,8,7,30,29,28,27,26] {
  %input_tensor = f32[50,49,48,47,46]{4,3,2,1,0} parameter(0)
  %start_indices = s64[10,9,8,7,5]{4,3,2,1,0} parameter(1)
  ROOT %gather = f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} gather(f32[50,49,48,47,46]{4,3,2,1,0} %input_tensor, s64[10,9,8,7,5]{4,3,2,1,0} %start_indices), offset_dims={4,5,6,7,8}, collapsed_slice_dims={}, start_index_map={0,1,2,3,4}, index_vector_dim=4, slice_sizes={30,29,28,27,26}, indices_are_sorted=true
}

"#,
    batch_gather: r#"HloModule StringifyGather, entry_computation_layout={(f32[50,49,48,47,46,512]{5,4,3,2,1,0}, s64[10,9,8,7,5,512]{5,4,3,2,1,0})->f32[10,9,8,7,30,29,28,27,26,512]{9,8,7,6,5,4,3,2,1,0}}

ENTRY %Gather (input_tensor: f32[50,49,48,47,46,512], start_indices: s64[10,9,8,7,5,512]) -> f32[10,9,8,7,30,29,28,27,26,512] {
  %input_tensor = f32[50,49,48,47,46,512]{5,4,3,2,1,0} parameter(0)
  %start_indices = s64[10,9,8,7,5,512]{5,4,3,2,1,0} parameter(1)
  ROOT %gather = f32[10,9,8,7,30,29,28,27,26,512]{9,8,7,6,5,4,3,2,1,0} gather(f32[50,49,48,47,46,512]{5,4,3,2,1,0} %input_tensor, s64[10,9,8,7,5,512]{5,4,3,2,1,0} %start_indices), offset_dims={4,5,6,7,8}, collapsed_slice_dims={}, start_index_map={0,1,2,3,4}, operand_batching_dims={5}, start_indices_batching_dims={5}, index_vector_dim=4, slice_sizes={30,29,28,27,26,1}
}

"#,
    scatter: r#"HloModule StringifyScatter, entry_computation_layout={(f32[50,49,48,47,46]{4,3,2,1,0}, s64[10,9,8,7,5]{4,3,2,1,0}, f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0})->f32[50,49,48,47,46]{4,3,2,1,0}}

%add_F32.v3 (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %lhs, f32[] %rhs)
}

ENTRY %Scatter (input_tensor: f32[50,49,48,47,46], scatter_indices: s64[10,9,8,7,5], updates: f32[10,9,8,7,30,29,28,27,26]) -> f32[50,49,48,47,46] {
  %input_tensor = f32[50,49,48,47,46]{4,3,2,1,0} parameter(0)
  %scatter_indices = s64[10,9,8,7,5]{4,3,2,1,0} parameter(1)
  %updates = f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} parameter(2)
  ROOT %scatter = f32[50,49,48,47,46]{4,3,2,1,0} scatter(f32[50,49,48,47,46]{4,3,2,1,0} %input_tensor, s64[10,9,8,7,5]{4,3,2,1,0} %scatter_indices, f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} %updates), update_window_dims={4,5,6,7,8}, inserted_window_dims={}, scatter_dims_to_operand_dims={0,1,2,3,4}, index_vector_dim=4, to_apply=%add_F32.v3
}

"#,
    batch_scatter: r#"HloModule StringifyScatter, entry_computation_layout={(f32[50,49,48,47,46,512]{5,4,3,2,1,0}, s64[10,9,8,7,5,512]{5,4,3,2,1,0}, f32[10,9,8,7,30,29,28,27,26,512]{9,8,7,6,5,4,3,2,1,0})->f32[50,49,48,47,46,512]{5,4,3,2,1,0}}

%add_F32.v3 (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %lhs, f32[] %rhs)
}

ENTRY %Scatter (input_tensor: f32[50,49,48,47,46,512], scatter_indices: s64[10,9,8,7,5,512], updates: f32[10,9,8,7,30,29,28,27,26,512]) -> f32[50,49,48,47,46,512] {
  %input_tensor = f32[50,49,48,47,46,512]{5,4,3,2,1,0} parameter(0)
  %scatter_indices = s64[10,9,8,7,5,512]{5,4,3,2,1,0} parameter(1)
  %updates = f32[10,9,8,7,30,29,28,27,26,512]{9,8,7,6,5,4,3,2,1,0} parameter(2)
  ROOT %scatter = f32[50,49,48,47,46,512]{5,4,3,2,1,0} scatter(f32[50,49,48,47,46,512]{5,4,3,2,1,0} %input_tensor, s64[10,9,8,7,5,512]{5,4,3,2,1,0} %scatter_indices, f32[10,9,8,7,30,29,28,27,26,512]{9,8,7,6,5,4,3,2,1,0} %updates), update_window_dims={4,5,6,7,8}, inserted_window_dims={}, scatter_dims_to_operand_dims={0,1,2,3,4}, input_batching_dims={5}, scatter_indices_batching_dims={5}, index_vector_dim=4, to_apply=%add_F32.v3
}

"#,
    tuple_scatter: r#"HloModule TupleScatter, entry_computation_layout={(f32[50,49,48,47,46]{4,3,2,1,0}, bf16[50,49,48,47,46]{4,3,2,1,0}, s64[10,9,8,7,5]{4,3,2,1,0}, f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0}, bf16[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0})->(f32[50,49,48,47,46]{4,3,2,1,0}, bf16[50,49,48,47,46]{4,3,2,1,0})}

%add_F32_mul_BF16 (lhs_0: f32[], lhs_1: bf16[], rhs_0: f32[], rhs_1: bf16[]) -> (f32[], bf16[]) {
  %lhs_0 = f32[] parameter(0)
  %rhs_0 = f32[] parameter(2)
  %add = f32[] add(f32[] %lhs_0, f32[] %rhs_0)
  %lhs_1 = bf16[] parameter(1)
  %rhs_1 = bf16[] parameter(3)
  %mul = bf16[] multiply(bf16[] %lhs_1, bf16[] %rhs_1)
  ROOT %tuple = (f32[], bf16[]) tuple(f32[] %add, bf16[] %mul)
}

ENTRY %Scatter (input_0: f32[50,49,48,47,46], input_1: bf16[50,49,48,47,46], scatter_indices: s64[10,9,8,7,5], updates_0: f32[10,9,8,7,30,29,28,27,26], updates_1: bf16[10,9,8,7,30,29,28,27,26]) -> (f32[50,49,48,47,46], bf16[50,49,48,47,46]) {
  %input_0 = f32[50,49,48,47,46]{4,3,2,1,0} parameter(0)
  %input_1 = bf16[50,49,48,47,46]{4,3,2,1,0} parameter(1)
  %scatter_indices = s64[10,9,8,7,5]{4,3,2,1,0} parameter(2)
  %updates_0 = f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} parameter(3)
  %updates_1 = bf16[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} parameter(4)
  ROOT %scatter = (f32[50,49,48,47,46]{4,3,2,1,0}, bf16[50,49,48,47,46]{4,3,2,1,0}) scatter(f32[50,49,48,47,46]{4,3,2,1,0} %input_0, bf16[50,49,48,47,46]{4,3,2,1,0} %input_1, s64[10,9,8,7,5]{4,3,2,1,0} %scatter_indices, f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} %updates_0, bf16[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} %updates_1), update_window_dims={4,5,6,7,8}, inserted_window_dims={}, scatter_dims_to_operand_dims={0,1,2,3,4}, index_vector_dim=4, to_apply=%add_F32_mul_BF16
}

"#,
    sorted_scatter: r#"HloModule StringifySortedScatter, entry_computation_layout={(f32[50,49,48,47,46]{4,3,2,1,0}, s64[10,9,8,7,5]{4,3,2,1,0}, f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0})->f32[50,49,48,47,46]{4,3,2,1,0}}

%add_F32.v3 (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %lhs, f32[] %rhs)
}

ENTRY %Scatter (input_tensor: f32[50,49,48,47,46], scatter_indices: s64[10,9,8,7,5], updates: f32[10,9,8,7,30,29,28,27,26]) -> f32[50,49,48,47,46] {
  %input_tensor = f32[50,49,48,47,46]{4,3,2,1,0} parameter(0)
  %scatter_indices = s64[10,9,8,7,5]{4,3,2,1,0} parameter(1)
  %updates = f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} parameter(2)
  ROOT %scatter = f32[50,49,48,47,46]{4,3,2,1,0} scatter(f32[50,49,48,47,46]{4,3,2,1,0} %input_tensor, s64[10,9,8,7,5]{4,3,2,1,0} %scatter_indices, f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} %updates), update_window_dims={4,5,6,7,8}, inserted_window_dims={}, scatter_dims_to_operand_dims={0,1,2,3,4}, index_vector_dim=4, indices_are_sorted=true, to_apply=%add_F32.v3
}

"#,
    unique_indices_scatter: r#"HloModule StringifyUniqueIndicesScatter, entry_computation_layout={(f32[50,49,48,47,46]{4,3,2,1,0}, s64[10,9,8,7,5]{4,3,2,1,0}, f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0})->f32[50,49,48,47,46]{4,3,2,1,0}}

%add_F32.v3 (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %add = f32[] add(f32[] %lhs, f32[] %rhs)
}

ENTRY %Scatter (input_tensor: f32[50,49,48,47,46], scatter_indices: s64[10,9,8,7,5], updates: f32[10,9,8,7,30,29,28,27,26]) -> f32[50,49,48,47,46] {
  %input_tensor = f32[50,49,48,47,46]{4,3,2,1,0} parameter(0)
  %scatter_indices = s64[10,9,8,7,5]{4,3,2,1,0} parameter(1)
  %updates = f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} parameter(2)
  ROOT %scatter = f32[50,49,48,47,46]{4,3,2,1,0} scatter(f32[50,49,48,47,46]{4,3,2,1,0} %input_tensor, s64[10,9,8,7,5]{4,3,2,1,0} %scatter_indices, f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} %updates), update_window_dims={4,5,6,7,8}, inserted_window_dims={}, scatter_dims_to_operand_dims={0,1,2,3,4}, index_vector_dim=4, unique_indices=true, to_apply=%add_F32.v3
}

"#,
    constant_unsigned_no_underflow: r#"HloModule ConstantUnsignedNoUnderflow_module, entry_computation_layout={()->u64[]}

ENTRY %ConstantUnsignedNoUnderflow () -> u64[] {
  ROOT %constant = u64[] constant(1)
}

"#,
    constant_unsigned_no_overflow: r#"HloModule ConstantUnsignedNoOverflow_module, entry_computation_layout={()->u64[]}

ENTRY %ConstantUnsignedNoOverflow () -> u64[] {
  ROOT %constant = u64[] constant(9223372036854775807)
}

"#,
    custom_call_with_layout_constraints: r#"HloModule CustomCallWithLayoutConstraints, entry_computation_layout={(f32[42,2,3]{0,1,2}, f32[123,4]{0,1})->f32[1,2,3]{0,2,1}}

ENTRY %CustomCallWithLayoutConstraints (p0: f32[42,2,3], p1: f32[123,4]) -> f32[1,2,3] {
  %p0 = f32[42,2,3]{0,1,2} parameter(0)
  %p1 = f32[123,4]{0,1} parameter(1)
  ROOT %custom-call = f32[1,2,3]{0,2,1} custom-call(f32[42,2,3]{0,1,2} %p0, f32[123,4]{0,1} %p1), custom_call_target="baz", operand_layout_constraints={f32[42,2,3]{0,1,2}, f32[123,4]{1,0}}
}

"#,
    custom_call_with_layout_constraints_no_operands: r#"HloModule CustomCallWithLayoutConstraintsNoOperands, entry_computation_layout={()->f32[1,2,3]{0,2,1}}

ENTRY %CustomCallWithLayoutConstraints () -> f32[1,2,3] {
  ROOT %custom-call = f32[1,2,3]{0,2,1} custom-call(), custom_call_target="baz", operand_layout_constraints={}
}

"#,
    custom_call_with_layout_constraints_tuple_shapes: r#"HloModule CustomCallWithLayoutConstraintsTupleShapes, entry_computation_layout={((f32[2,2]{0,1}, f32[42,2,3]{0,1,2}), f32[123,4]{0,1})->(f32[1,2,3]{0,2,1}, f32[1,2,3]{1,2,0})}

ENTRY %CustomCallWithLayoutConstraints (p0: (f32[2,2], f32[42,2,3]), p1: f32[123,4]) -> (f32[1,2,3], f32[1,2,3]) {
  %p0 = (f32[2,2]{0,1}, f32[42,2,3]{0,1,2}) parameter(0)
  %p1 = f32[123,4]{0,1} parameter(1)
  ROOT %custom-call = (f32[1,2,3]{0,2,1}, f32[1,2,3]{1,2,0}) custom-call((f32[2,2]{0,1}, f32[42,2,3]{0,1,2}) %p0, f32[123,4]{0,1} %p1), custom_call_target="baz", operand_layout_constraints={(f32[2,2]{1,0}, f32[42,2,3]{2,0,1}), f32[123,4]{1,0}}
}

"#,
    custom_call_with_has_side_effect: r#"HloModule CustomCallWithHasSideEffect, entry_computation_layout={((f32[2,2]{0,1}, f32[42,2,3]{0,1,2}), f32[123,4]{0,1})->(f32[1,2,3]{0,2,1}, f32[1,2,3]{1,2,0})}

ENTRY %CustomCallWithHasSideEffect (p0: (f32[2,2], f32[42,2,3]), p1: f32[123,4]) -> (f32[1,2,3], f32[1,2,3]) {
  %p0 = (f32[2,2]{0,1}, f32[42,2,3]{0,1,2}) parameter(0)
  %p1 = f32[123,4]{0,1} parameter(1)
  ROOT %custom-call = (f32[1,2,3]{0,2,1}, f32[1,2,3]{1,2,0}) custom-call((f32[2,2]{0,1}, f32[42,2,3]{0,1,2}) %p0, f32[123,4]{0,1} %p1), custom_call_target="baz", custom_call_has_side_effect=true
}

"#,
    custom_call_with_aliasing: r#"HloModule CustomCallWithAliasing, entry_computation_layout={((f32[2,2]{0,1}, f32[42,2,3]{0,1,2}), f32[123,4]{0,1})->(f32[123,4]{0,1}, f32[2,2]{0,1}, f32[1,2,3]{0,1,2})}

ENTRY %CustomCallWithAliasing (p0: (f32[2,2], f32[42,2,3]), p1: f32[123,4]) -> (f32[123,4], f32[2,2], f32[1,2,3]) {
  %p0 = (f32[2,2]{0,1}, f32[42,2,3]{0,1,2}) parameter(0)
  %p1 = f32[123,4]{0,1} parameter(1)
  ROOT %custom-call = (f32[123,4]{0,1}, f32[2,2]{0,1}, f32[1,2,3]{0,1,2}) custom-call((f32[2,2]{0,1}, f32[42,2,3]{0,1,2}) %p0, f32[123,4]{0,1} %p1), custom_call_target="baz", output_to_operand_aliasing={{0}: (1, {}), {1}: (0, {0})}
}

"#,
    custom_call_with_schedule: r#"HloModule custom_call, entry_computation_layout={()->f32[1,2,3]{0,2,1}}

ENTRY %CustomCall () -> f32[1,2,3] {
  %constant = f32[1]{0} constant({12345})
  %custom-call.0 = f32[1,2,3]{0,2,1} custom-call(f32[1]{0} %constant), custom_call_target="foo", schedule=SCHEDULE_EARLIEST
  ROOT %custom-call.1 = f32[1,2,3]{0,2,1} custom-call(f32[1,2,3]{0,2,1} %custom-call.0), custom_call_target="bar", schedule=SCHEDULE_LATEST
}

"#,
    custom_call_with_status_returning_version: r#"HloModule custom_call, entry_computation_layout={()->f32[1,2,3]{0,2,1}}

ENTRY %CustomCall () -> f32[1,2,3] {
  %constant = f32[1]{0} constant({12345})
  ROOT %custom-call.1 = f32[1,2,3]{0,2,1} custom-call(f32[1]{0} %constant), custom_call_target="foo", api_version=API_VERSION_STATUS_RETURNING
}

"#,
    parse_c64_literal: r#"HloModule ParseC64Literal, entry_computation_layout={()->c64[2]{0}}

ENTRY %ParseC64Literal () -> c64[2] {
  ROOT %c = c64[2]{0} constant({(1, 2), (-inf, nan)})
}

"#,
    parse_c128_literal: r#"HloModule ParseC128Literal, entry_computation_layout={()->c128[2]{0}}

ENTRY %ParseC128Literal () -> c128[2] {
  ROOT %c = c128[2]{0} constant({(1, 2), (-inf, nan)})
}

"#,
    indexed_conditional: r#"HloModule indexed_conditional, entry_computation_layout={()->f32[]}

%Negate (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  ROOT %negate = f32[] negate(f32[] %x)
}

%Identity (y: f32[]) -> f32[] {
  %y = f32[] parameter(0)
  ROOT %copy = f32[] copy(f32[] %y)
}

%Floor (z: f32[]) -> f32[] {
  %z = f32[] parameter(0)
  ROOT %floor = f32[] floor(f32[] %z)
}

ENTRY %Parameters1.v4 () -> f32[] {
  %constant = s32[] constant(1)
  %constant.1 = f32[] constant(56)
  %constant.2 = f32[] constant(12)
  %constant.3 = f32[] constant(13)
  ROOT %conditional = f32[] conditional(s32[] %constant, f32[] %constant.1, f32[] %constant.2, f32[] %constant.3), branch_computations={%Negate, %Identity, %Floor}
}

"#,
    rng_get_and_update_state: r#"HloModule rng_get_and_update_state, entry_computation_layout={()->u64[2]{0}}

ENTRY %RngGetAndUpdateState () -> u64[2] {
  ROOT %rng-get-and-update-state = u64[2]{0} rng-get-and-update-state(), delta=4096
}

"#,
    rng_bit_generator: r#"HloModule gng_bit_generator, entry_computation_layout={(u64[2]{0})->(u64[2]{0}, u32[11,17]{1,0})}

ENTRY %RngBitGenerator (p0: u64[2]) -> (u64[2], u32[11,17]) {
  %p0 = u64[2]{0} parameter(0)
  ROOT %rand = (u64[2]{0}, u32[11,17]{1,0}) rng-bit-generator(u64[2]{0} %p0), algorithm=rng_three_fry
}

"#,
    async_ops_with_syntax_sugar: r#"HloModule AsyncOpsWithSyntaxSugar, entry_computation_layout={(f32[10]{0})->f32[20]{0}}

ENTRY %Entry (p0: f32[10]) -> f32[20] {
  %p0 = f32[10]{0} parameter(0)
  %async-start = ((f32[10]{0}), f32[20]{0}, s32[]) custom-call-start(f32[10]{0} %p0), custom_call_target="foo"
  %async-update = ((f32[10]{0}), f32[20]{0}, s32[]) custom-call-update(((f32[10]{0}), f32[20]{0}, s32[]) %async-start)
  ROOT %async-done = f32[20]{0} custom-call-done(((f32[10]{0}), f32[20]{0}, s32[]) %async-update)
}

"#,
    async_ops_with_syntax_sugar_and_thread_name: r#"HloModule AsyncOpsWithSyntaxSugarAndThreadName, entry_computation_layout={(f32[10]{0})->f32[20]{0}}

ENTRY %Entry (p0: f32[10]) -> f32[20] {
  %p0 = f32[10]{0} parameter(0)
  %async-start = ((f32[10]{0}), f32[20]{0}, s32[]) custom-call-start(f32[10]{0} %p0), async_execution_thread="parallel_thread", custom_call_target="foo"
  %async-update = ((f32[10]{0}), f32[20]{0}, s32[]) custom-call-update(((f32[10]{0}), f32[20]{0}, s32[]) %async-start)
  ROOT %async-done = f32[20]{0} custom-call-done(((f32[10]{0}), f32[20]{0}, s32[]) %async-update)
}

"#,
    hlo_computation_with_parallel_thread_name: r#"HloModule HloComputationWithParallelThreadName, entry_computation_layout={(f32[10]{0})->f32[20]{0}}

ENTRY %Entry (p0: f32[10]) -> f32[20] {
  %p0 = f32[10]{0} parameter(0)
  %async-start = ((f32[10]{0}), f32[20]{0}, s32[]) custom-call-start(f32[10]{0} %p0), async_execution_thread="parallel_thread", custom_call_target="foo"
  %async-update = ((f32[10]{0}), f32[20]{0}, s32[]) custom-call-update(((f32[10]{0}), f32[20]{0}, s32[]) %async-start)
  ROOT %async-done = f32[20]{0} custom-call-done(((f32[10]{0}), f32[20]{0}, s32[]) %async-update)
}, execution_thread="main_thread"

"#,
    metadata_fields: r#"HloModule test, entry_computation_layout={(f32[100]{0})->u32[100]{0}}

ENTRY %test (p: f32[100]) -> u32[100] {
  %p = f32[100]{0} parameter(0)
  ROOT %root = u32[100]{0} bitcast-convert(f32[100]{0} %p), metadata={op_type="a" op_name="b" source_file="c" source_line=1 profile_type={1} deduplicated_name="d" scheduling_name="foo"}
}

"#,
    original_value: r#"HloModule test, entry_computation_layout={(f32[], f32[3]{0}, f32[2,3]{1,0})->((f32[], f32[3]{0}), f32[2,3]{1,0})}

ENTRY %test (v1: f32[], v2: f32[3], v3: f32[2,3]) -> ((f32[], f32[3]), f32[2,3]) {
  %v1 = f32[] parameter(0), origin={{"v1"}}
  %v2 = f32[3]{0} parameter(1), origin={{"v2"}}
  %tuple = (f32[], f32[3]{0}) tuple(f32[] %v1, f32[3]{0} %v2), origin={({"v1"}, {"v2"})}
  %v3 = f32[2,3]{1,0} parameter(2), origin={{"v3"}}
  ROOT %nested_tuple = ((f32[], f32[3]{0}), f32[2,3]{1,0}) tuple((f32[], f32[3]{0}) %tuple, f32[2,3]{1,0} %v3), origin={(({"v1"}, {"v2"}), {"v3"})}
}

"#,
    original_value_synthetic: r#"HloModule test, entry_computation_layout={(f32[])->f32[]}

ENTRY %test (v1: f32[]) -> f32[] {
  %v1 = f32[] parameter(0), origin={[synthetic_call]}
  ROOT %add = f32[] add(f32[] %v1, f32[] %v1), origin={[synthetic_call]}
}

"#,
    original_value_recovery_table: r#"HloModule module, entry_computation_layout={()->s32[1,3]{1,0}}, num_partitions=2, origin_recovery_table={
  {"constant"} : {"constant__ovp0"},
  "
    HloModule recovery_module, entry_computation_layout={(s32[2,3]{1,0})->s32[2,3]{1,0}}

    %add (x: s32[], y: s32[]) -> s32[] {
      %x = s32[] parameter(0)
      %y = s32[] parameter(1)
      ROOT %add = s32[] add(%x, %y)
    }

    %add.clone (x.1: s32[], y.1: s32[]) -> s32[] {
      %x.1 = s32[] parameter(0)
      %y.1 = s32[] parameter(1)
      ROOT %add.1 = s32[] add(%x.1, %y.1)
    }

    ENTRY %recovery_computation (param: s32[2,3]) -> s32[2,3] {
      %partition-id = u32[] partition-id()
      %constant = u32[] constant(0)
      %compare = pred[] compare(%partition-id, %constant), direction=EQ
      %broadcast = pred[2,3]{1,0} broadcast(%compare), dimensions={}
      %param = s32[2,3]{1,0} parameter(0), sharding={maximal device=0}
      %constant.1 = s32[] constant(0)
      %broadcast.1 = s32[2,3]{1,0} broadcast(%constant.1), dimensions={}
      %select = s32[2,3]{1,0} select(%broadcast, %param, %broadcast.1)
      ROOT %all-reduce = s32[2,3]{1,0} all-reduce(%select), channel_id=1, replica_groups={{0,1}}, use_global_device_ids=true, to_apply=%add.clone, sharding={replicated}
    }


  "
}


%add.clone (x.1: s32[], y.1: s32[]) -> s32[] {
  %x.1 = s32[] parameter(0)
  %y.1 = s32[] parameter(1)
  ROOT %add.1 = s32[] add(s32[] %x.1, s32[] %y.1)
}

ENTRY %entry_spmd () -> s32[1,3] {
  %partition-id = u32[] partition-id()
  %constant.2 = u32[] constant(0)
  %compare = pred[] compare(u32[] %partition-id, u32[] %constant.2), direction=EQ
  %broadcast = pred[2,3]{1,0} broadcast(pred[] %compare), dimensions={}
  %constant.1 = s32[2,3]{1,0} constant({ { 1, 1, 1 }, { 1, 1, 1 } }), origin={{"constant__ovp0"}}
  %constant.3 = s32[] constant(0)
  %broadcast.1 = s32[2,3]{1,0} broadcast(s32[] %constant.3), dimensions={}
  %select = s32[2,3]{1,0} select(pred[2,3]{1,0} %broadcast, s32[2,3]{1,0} %constant.1, s32[2,3]{1,0} %broadcast.1)
  %all-reduce = s32[2,3]{1,0} all-reduce(s32[2,3]{1,0} %select), channel_id=1, replica_groups={{0,1}}, use_global_device_ids=true, to_apply=%add.clone
  %constant.4 = s32[2]{0} constant({1, 0})
  %dynamic-slice = s32[1]{0} dynamic-slice(s32[2]{0} %constant.4, u32[] %partition-id), dynamic_slice_sizes={1}
  %reshape = s32[] reshape(s32[1]{0} %dynamic-slice)
  %dynamic-slice.1 = s32[1,3]{1,0} dynamic-slice(s32[2,3]{1,0} %all-reduce, s32[] %reshape, s32[] %constant.3), dynamic_slice_sizes={1,3}
  ROOT %copy.1 = s32[1,3]{1,0} copy(s32[1,3]{1,0} %dynamic-slice.1)
}

"#,
    debug_attributes: r#"HloModule module, entry_computation_layout={()->s32[1,3]{1,0}},
debug_attributes={
  {"constant"}:({log_mode=default,callback_id=123,partitioned=true})
}

ENTRY %e () -> s32[1,3] {
  ROOT %c = s32[1,3]{1,0} constant({ { 0, 1, 2 } }), origin={{"constant"}}
}

"#,
    debug_attributes_fusion_debugger: r#"HloModule module, entry_computation_layout={()->s32[1,3]{1,0}},
debug_attributes={
  {"constant"}:({log_mode=fusion_debugger,callback_id=123})
}

ENTRY %e () -> s32[1,3] {
  ROOT %c = s32[1,3]{1,0} constant({ { 0, 1, 2 } }), origin={{"constant"}}
}

"#,
    debug_attributes_log_mode_only: r#"HloModule module, entry_computation_layout={()->s32[1,3]{1,0}},
debug_attributes={
  {"constant"}:({log_mode=default})
}

ENTRY %e () -> s32[1,3] {
  ROOT %c = s32[1,3]{1,0} constant({ { 0, 1, 2 } }), origin={{"constant"}}
}

"#,
    debug_attributes_partitioned_only: r#"HloModule module, entry_computation_layout={()->s32[1,3]{1,0}},
debug_attributes={
  {"constant"}:({partitioned=true})
}

ENTRY %e () -> s32[1,3] {
  ROOT %c = s32[1,3]{1,0} constant({ { 0, 1, 2 } }), origin={{"constant"}}
}

"#,
    debug_attributes_hlo_id_only: r#"HloModule module, entry_computation_layout={()->s32[1,3]{1,0}},
debug_attributes={
  {"constant"}:({callback_id=123})
}

ENTRY %e () -> s32[1,3] {
  ROOT %c = s32[1,3]{1,0} constant({ { 0, 1, 2 } }), origin={{"constant"}}
}

"#,
    original_value_recovery_table_with_nested_quotes: r#"HloModule module, entry_computation_layout={()->s32[1,3]{1,0}}, num_partitions=2, origin_recovery_table={
  {"constant"} : {"constant__ovp0"},
  "
    HloModule recovery_module, entry_computation_layout={(s32[2,3]{1,0})->s32[2,3]{1,0}}, frontend_attributes={foo=\"bar\"}

    %add (x: s32[], y: s32[]) -> s32[] {
      %x = s32[] parameter(0)
      %y = s32[] parameter(1)
      ROOT %add = s32[] add(%x, %y)
    }

    ENTRY %recovery_computation (param: s32[2,3]) -> s32[2,3] {
      ROOT %param = s32[2,3]{1,0} parameter(0)
    }


  "
}


ENTRY %entry_spmd () -> s32[1,3] {
  ROOT %constant.1 = s32[1,3]{1,0} constant({ { 1, 1, 1 } }), origin={{"constant__ovp0"}}
}

"#,
    stack_frame_index: r#"HloModule m, entry_computation_layout={()->pred[]}

FileNames
1 "<embedded module>"
2 "yet/another/test.py"

FunctionNames
1 "main"
2 "method"

FileLocations
1 {file_name_id=1 function_name_id=1 line=153 end_line=153 column=2 end_column=31}
2 {file_name_id=2 function_name_id=2 line=35 end_line=35 column=2 end_column=24}

StackFrames
1 {file_location_id=1 parent_frame_id=1}
2 {file_location_id=2 parent_frame_id=2}


ENTRY %constant_pred () -> pred[] {
  ROOT %constant = pred[] constant(true), metadata={op_type="const" op_name="opname" stack_frame_id=2}
}

"#,
        }
}

mod hlo_parser_test_short {
    use crate::hlo_text::Style;

    corpus! {
    "short", Style::Short,
    map: r#"HloModule MapBinaryAdder_module, entry_computation_layout={(f32[4]{0}, f32[4]{0})->f32[4]{0}}

add_F32.v3 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY MapBinaryAdder.v3 {
  param0 = f32[4]{0} parameter(0)
  param1 = f32[4]{0} parameter(1)
  ROOT map = f32[4]{0} map(param0, param1), dimensions={0}, to_apply=add_F32.v3
}

"#,
    reduce: r#"HloModule ReduceR3ToR2_module, entry_computation_layout={(f32[8,16,256]{2,1,0})->f32[8,16]{1,0}}

add_F32.v3 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY ReduceR3ToR2.v3 {
  input = f32[8,16,256]{2,1,0} parameter(0)
  constant = f32[] constant(0)
  ROOT reduce = f32[8,16]{1,0} reduce(input, constant), dimensions={2}, to_apply=add_F32.v3
}

"#,
    tuple_reduce: r#"HloModule TupleReduce, entry_computation_layout={(f32[1024]{0}, s32[1024]{0})->(f32[], s32[])}

max_argmax {
  value = f32[] parameter(2)
  prev_max = f32[] parameter(0)
  is_next_larger = pred[] compare(value, prev_max), direction=GE
  max = f32[] select(is_next_larger, value, prev_max)
  index = s32[] parameter(3)
  prev_argmax = s32[] parameter(1)
  argmax = s32[] select(is_next_larger, index, prev_argmax)
  ROOT pair = (f32[], s32[]) tuple(max, argmax)
}

ENTRY reduce_entry {
  values = f32[1024]{0} parameter(0)
  indices = s32[1024]{0} parameter(1)
  init_value = f32[] constant(-inf)
  init_index = s32[] constant(-1)
  ROOT result = (f32[], s32[]) reduce(values, indices, init_value, init_index), dimensions={0}, to_apply=max_argmax
}

"#,
    infeed_outfeed: r#"HloModule outfeed_module, entry_computation_layout={()->((u32[3]{0}, pred[]), token[])}

ENTRY InfeedToOutfeed {
  token0 = token[] after-all()
  infeed = ((u32[3]{0}, pred[]), token[]) infeed(token0)
  infeed.data = (u32[3]{0}, pred[]) get-tuple-element(infeed), index=0
  outfeed = token[] outfeed(infeed.data, token0), outfeed_shape=(u32[3]{0}, pred[])
  ROOT infeed.1 = ((u32[3]{0}, pred[]), token[]) infeed(token0)
  infeed.1.data = (u32[3]{0}, pred[]) get-tuple-element(infeed.1), index=0
  infeed.1.token = token[] get-tuple-element(infeed.1), index=1
  outfeed.1 = token[] outfeed(infeed.1.data, infeed.1.token), outfeed_shape=(u32[3]{0}, pred[])
}

"#,
    rng: r#"HloModule rng_module, entry_computation_layout={()->f32[8]{0}}

ENTRY Rng {
  constant = f32[] constant(0)
  constant.1 = f32[] constant(1)
  ROOT rng = f32[8]{0} rng(constant, constant.1), distribution=rng_uniform
}

"#,
    reduce_precision: r#"HloModule reduce_precision, entry_computation_layout={()->f32[1]{0}}

ENTRY ReducePrecision {
  constant = f32[1]{0} constant({3.14159})
  ROOT reduce-precision = f32[1]{0} reduce-precision(constant), exponent_bits=8, mantissa_bits=10
}

"#,
    sort_key: r#"HloModule sort, entry_computation_layout={(f32[1024]{0})->f32[1024]{0}}

compare {
  p.0.lhs = f32[] parameter(0)
  p.0.rhs = f32[] parameter(1)
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
}

ENTRY Sort {
  x = f32[1024]{0} parameter(0)
  ROOT sorted = f32[1024]{0} sort(x), dimensions={0}, to_apply=compare
}

"#,
    sort_key_value: r#"HloModule sort, entry_computation_layout={(f32[1024]{0}, s32[1024]{0})->(f32[1024]{0}, s32[1024]{0})}

compare {
  p.1.lhs = s32[] parameter(2)
  p.1.rhs = s32[] parameter(3)
  p.0.lhs = f32[] parameter(0)
  p.0.rhs = f32[] parameter(1)
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
}

ENTRY Sort {
  keys = f32[1024]{0} parameter(0)
  values = s32[1024]{0} parameter(1)
  ROOT sorted = (f32[1024]{0}, s32[1024]{0}) sort(keys, values), dimensions={0}, to_apply=compare
}

"#,
    sort_key_r2: r#"HloModule sort, entry_computation_layout={(f32[1024,16]{0,1})->f32[1024,16]{0,1}}

compare {
  p.0.lhs = f32[] parameter(0)
  p.0.rhs = f32[] parameter(1)
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
}

ENTRY Sort {
  x = f32[1024,16]{0,1} parameter(0)
  ROOT sorted = f32[1024,16]{0,1} sort(x), dimensions={0}, to_apply=compare
}

"#,
    sort_key_value_r2: r#"HloModule sort, entry_computation_layout={(f32[1024,16]{0,1}, s32[1024,16]{0,1})->(f32[1024,16]{0,1}, s32[1024,16]{0,1})}

compare {
  p.1.lhs = s32[] parameter(2)
  p.1.rhs = s32[] parameter(3)
  p.0.lhs = f32[] parameter(0)
  p.0.rhs = f32[] parameter(1)
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
}

ENTRY Sort {
  keys = f32[1024,16]{0,1} parameter(0)
  values = s32[1024,16]{0,1} parameter(1)
  ROOT sorted = (f32[1024,16]{0,1}, s32[1024,16]{0,1}) sort(keys, values), dimensions={0}, to_apply=compare
}

"#,
    sort_many_values: r#"HloModule sort, entry_computation_layout={(f32[1024,16]{0,1}, s32[1024,16]{0,1}, u32[1024,16]{0,1}, f32[1024,16]{0,1})->(f32[1024,16]{0,1}, s32[1024,16]{0,1}, u32[1024,16]{0,1}, f32[1024,16]{0,1})}

compare {
  p.1.lhs = s32[] parameter(2)
  p.1.rhs = s32[] parameter(3)
  p.2.lhs = u32[] parameter(4)
  p.2.rhs = u32[] parameter(5)
  p.3.lhs = f32[] parameter(6)
  p.3.rhs = f32[] parameter(7)
  p.0.lhs = f32[] parameter(0)
  p.0.rhs = f32[] parameter(1)
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
}

ENTRY Sort {
  keys = f32[1024,16]{0,1} parameter(0)
  values.0 = s32[1024,16]{0,1} parameter(1)
  values.1 = u32[1024,16]{0,1} parameter(2)
  values.2 = f32[1024,16]{0,1} parameter(3)
  ROOT sorted = (f32[1024,16]{0,1}, s32[1024,16]{0,1}, u32[1024,16]{0,1}, f32[1024,16]{0,1}) sort(keys, values.0, values.1, values.2), dimensions={0}, to_apply=compare
}

"#,
    sort_key_stable: r#"HloModule sort, entry_computation_layout={(f32[1024]{0})->f32[1024]{0}}

compare {
  p.0.lhs = f32[] parameter(0)
  p.0.rhs = f32[] parameter(1)
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
}

ENTRY Sort {
  x = f32[1024]{0} parameter(0)
  ROOT sorted = f32[1024]{0} sort(x), dimensions={0}, is_stable=true, to_apply=compare
}

"#,
    top_k: r#"HloModule topk, entry_computation_layout={(f32[10,10]{0,1})->(f32[10,2]{0,1}, s32[10,2]{0,1})}

ENTRY TopK {
  x = f32[10,10]{0,1} parameter(0)
  ROOT topk = (f32[10,2]{0,1}, s32[10,2]{0,1}) topk(x), k=2, largest=true, is_stable=true
}

"#,
    top_k_unstable: r#"HloModule topk, entry_computation_layout={(f32[8,1024]{0,1})->(f32[8,24]{0,1}, s32[8,24]{0,1})}

ENTRY TopK {
  x = f32[8,1024]{0,1} parameter(0)
  ROOT topk = (f32[8,24]{0,1}, s32[8,24]{0,1}) topk(x), k=24, largest=true, is_stable=false
}

"#,
    indexed_conditional: r#"HloModule indexed_conditional, entry_computation_layout={()->f32[]}

Negate {
  x = f32[] parameter(0)
  ROOT negate = f32[] negate(x)
}

Identity {
  y = f32[] parameter(0)
  ROOT copy = f32[] copy(y)
}

Floor {
  z = f32[] parameter(0)
  ROOT floor = f32[] floor(z)
}

ENTRY Parameters1.v4 {
  constant = s32[] constant(1)
  constant.1 = f32[] constant(56)
  constant.2 = f32[] constant(12)
  constant.3 = f32[] constant(13)
  ROOT conditional = f32[] conditional(constant, constant.1, constant.2, constant.3), branch_computations={Negate, Identity, Floor}
}

"#,
    predicated_conditional: r#"HloModule pred_conditional, entry_computation_layout={()->f32[]}

Negate {
  x = f32[] parameter(0)
  ROOT negate = f32[] negate(x)
}

Identity {
  y = f32[] parameter(0)
  ROOT copy = f32[] copy(y)
}

ENTRY Parameters1.v4 {
  constant = pred[] constant(true)
  constant.1 = f32[] constant(56)
  constant.2 = f32[] constant(12)
  ROOT conditional = f32[] conditional(constant, constant.1, constant.2), true_computation=Negate, false_computation=Identity
}

"#,
    custom_call: r#"HloModule custom_call, entry_computation_layout={()->f32[1,2,3]{0,2,1}}

ENTRY CustomCall {
  constant = f32[1]{0} constant({12345})
  ROOT custom-call = f32[1,2,3]{0,2,1} custom-call(constant), custom_call_target="foo\"bar"
}

"#,
    custum_call_single_comp: r#"HloModule custom_call_with_comp, entry_computation_layout={()->f32[1,2,3]{0,2,1}}

max_F32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT maximum = f32[] maximum(lhs, rhs)
}

ENTRY CustomCall {
  constant = f32[1]{0} constant({12345})
  ROOT custom-call = f32[1,2,3]{0,2,1} custom-call(constant), custom_call_target="foo\"bar", called_computations={max_F32}
}

"#,
    custum_call_multiple_comps: r#"HloModule custom_call_with_comps, entry_computation_layout={()->f32[1,2,3]{0,2,1}}

max_F32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT maximum = f32[] maximum(lhs, rhs)
}

ENTRY CustomCall {
  constant = f32[1]{0} constant({12345})
  ROOT custom-call = f32[1,2,3]{0,2,1} custom-call(constant), custom_call_target="foo\"bar", called_computations={max_F32, max_F32}
}

"#,
    non_default_names: r#"HloModule add_constants_module, entry_computation_layout={()->f32[]}

ENTRY add_constants {
  foo = f32[] constant(3.14)
  ROOT bar = f32[] add(foo, foo)
}

"#,
    dot: r#"HloModule dot, entry_computation_layout={(f32[2,10]{1,0}, f32[10,2]{1,0})->f32[2]{0}}

ENTRY dot {
  a = f32[2,10]{1,0} parameter(0)
  b = f32[10,2]{1,0} parameter(1)
  ROOT dot = f32[2]{0} dot(a, b), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={1}, rhs_contracting_dims={0}
}

"#,
    dot_with_algorithm: r#"HloModule dot, entry_computation_layout={(f32[2,10]{1,0}, f32[10,2]{1,0})->f32[2]{0}}

ENTRY dot {
  a = f32[2,10]{1,0} parameter(0)
  b = f32[10,2]{1,0} parameter(1)
  ROOT dot = f32[2]{0} dot(a, b), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={1}, rhs_contracting_dims={0}, algorithm=dot_tf32_tf32_f32
}

"#,
    gather: r#"HloModule gather, entry_computation_layout={(f32[50,49,48,47,46]{4,3,2,1,0}, s64[10,9,8,7,5]{4,3,2,1,0})->f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0}}

ENTRY Gather {
  input_tensor = f32[50,49,48,47,46]{4,3,2,1,0} parameter(0)
  start_indices = s64[10,9,8,7,5]{4,3,2,1,0} parameter(1)
  ROOT gather = f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} gather(input_tensor, start_indices), offset_dims={4,5,6,7,8}, collapsed_slice_dims={}, start_index_map={0,1,2,3,4}, index_vector_dim=4, slice_sizes={30,29,28,27,26}
}

"#,
    all_reduce: r#"HloModule CRS, entry_computation_layout={(f32[8]{0})->f32[8]{0}}

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY CRS {
  input = f32[8]{0} parameter(0)
  ROOT crs = f32[8]{0} all-reduce(input), replica_groups={}, to_apply=add
}

"#,
    all_reduce_with_subgroups: r#"HloModule CRS_Subgroups, entry_computation_layout={(f32[128,32]{0,1})->f32[128,32]{0,1}}, replica_count=4

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY AllReduceWithSubgroups {
  input = f32[128,32]{0,1} parameter(0)
  ROOT all-reduce = f32[128,32]{0,1} all-reduce(input), replica_groups={{0,1},{2,3}}, to_apply=add
}

"#,
    all_reduce_with_subgroups_iota_list: r#"HloModule CRS_Subgroups, entry_computation_layout={(f32[128,32]{0,1})->f32[128,32]{0,1}}, replica_count=20

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY AllReduceWithSubgroupsIotaList {
  input = f32[128,32]{0,1} parameter(0)
  ROOT all-reduce = f32[128,32]{0,1} all-reduce(input), replica_groups=[2,10]<=[20], to_apply=add
}

"#,
    all_reduce_with_mesh_axes_replica_group_list: r#"HloModule CRS_Subgroups, entry_computation_layout={(f32[128,32]{0,1})->f32[128,32]{0,1}}, replica_count=4

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY AllReduceWithMeshAxesReplicaGroupList {
  input = f32[128,32]{0,1} parameter(0)
  ROOT all-reduce = f32[128,32]{0,1} all-reduce(input), replica_groups=mesh['axis_0'=2,'axis_1'=2] {'axis_1'}, to_apply=add
}

"#,
    all_reduce_with_layout: r#"HloModule CRS, entry_computation_layout={(f32[8]{0})->f32[8]{0}}

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY CRS {
  input = f32[8]{0} parameter(0)
  ROOT crs = f32[8]{0} all-reduce(input), replica_groups={}, constrain_layout=true, to_apply=add
}

"#,
    all_reduce_all_reduce: r#"HloModule CRS, entry_computation_layout={(f32[8]{0})->f32[8]{0}}

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY CRS {
  input = f32[8]{0} parameter(0)
  crs.1 = f32[8]{0} all-reduce(input), channel_id=1, replica_groups={{0}}, to_apply=add
  ROOT crs.0 = f32[8]{0} all-reduce(input), channel_id=1, replica_groups={{0}}, to_apply=add
}

"#,
    all_reduce_start_and_done: r#"HloModule CRS, entry_computation_layout={(f32[8]{0})->f32[8]{0}}

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY CRS {
  input = f32[8]{0} parameter(0)
  crs = f32[8]{0} all-reduce-start(input), replica_groups={}, to_apply=add
  ROOT done = f32[8]{0} all-reduce-done(crs)
}

"#,
    reduce_scatter: r#"HloModule RS, entry_computation_layout={(f32[8]{0})->f32[4]{0}}

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY CRS {
  input = f32[8]{0} parameter(0)
  ROOT ars = f32[4]{0} reduce-scatter(input), replica_groups={{0,1}}, dimensions={0}, to_apply=add
}

"#,
    all_gather: r#"HloModule AllGather, entry_computation_layout={(f32[128,32]{0,1})->f32[128,128]{0,1}}

ENTRY AllGather {
  input = f32[128,32]{0,1} parameter(0)
  ROOT ag = f32[128,128]{0,1} all-gather(input), replica_groups={}, dimensions={1}
}

"#,
    all_gather_with_layout: r#"HloModule AllGather, entry_computation_layout={(f32[128,32]{0,1})->f32[128,128]{0,1}}

ENTRY AllGather {
  input = f32[128,32]{0,1} parameter(0)
  ROOT ag = f32[128,128]{0,1} all-gather(input), replica_groups={}, constrain_layout=true, dimensions={1}
}

"#,
    all_gather_with_subgroups: r#"HloModule AllGatherWithSubgroups, entry_computation_layout={(f32[128,32]{0,1})->f32[128,64]{0,1}}, replica_count=4

ENTRY AllGatherWithSubgroups {
  input = f32[128,32]{0,1} parameter(0)
  ROOT ag = f32[128,64]{0,1} all-gather(input), replica_groups={{0,1},{2,3}}, dimensions={1}
}

"#,
    all_gather_with_subgroups_iota_list: r#"HloModule AllGatherWithSubgroupsIotaList, entry_computation_layout={(f32[128,32]{0,1})->f32[128,320]{0,1}}, replica_count=30

ENTRY AllGatherWithSubgroupsIotaList {
  input = f32[128,32]{0,1} parameter(0)
  ROOT ag = f32[128,320]{0,1} all-gather(input), replica_groups=[3,10]<=[6,5]T(1,0), dimensions={1}
}

"#,
    all_to_all: r#"HloModule AllToAll, entry_computation_layout={(f32[128,32]{0,1})->(f32[128,32]{0,1})}

ENTRY AllToAll {
  input = f32[128,32]{0,1} parameter(0)
  ROOT a2a = (f32[128,32]{0,1}) all-to-all(input), replica_groups={}
}

"#,
    all_to_all_with_subgroups: r#"HloModule AllToAllWithSubgroups, entry_computation_layout={(f32[128,32]{0,1}, f32[128,32]{0,1})->(f32[128,32]{0,1}, f32[128,32]{0,1})}, replica_count=4

ENTRY AllToAllWithSubgroups {
  p0 = f32[128,32]{0,1} parameter(0)
  p1 = f32[128,32]{0,1} parameter(1)
  ROOT a2a = (f32[128,32]{0,1}, f32[128,32]{0,1}) all-to-all(p0, p1), replica_groups={{1,2},{3,0}}
}

"#,
    all_to_all_with_subgroups_iota_list: r#"HloModule AllToAllWithSubgroupsIotaList, entry_computation_layout={(f32[128,32]{0,1})->f32[128,32]{0,1}}, replica_count=32

ENTRY AllToAllWithSubgroupsIotaList {
  p0 = f32[128,32]{0,1} parameter(0)
  ROOT a2a = f32[128,32]{0,1} all-to-all(p0), replica_groups=[4,8]<=[4,8]T(1,0), dimensions={0}
}

"#,
    ragged_all_to_all_with_replica_groups: r#"HloModule RaggedAllToAll, entry_computation_layout={(bf16[1024,256]{1,0}, bf16[1024,256]{1,0}, s32[8]{0}, s32[8]{0}, s32[8]{0}, /*index=5*/s32[8]{0})->bf16[1024,256]{1,0}}, replica_count=8

ENTRY AllToAll {
  input = bf16[1024,256]{1,0} parameter(0)
  output = bf16[1024,256]{1,0} parameter(1)
  input_offsets = s32[8]{0} parameter(2)
  send_sizes = s32[8]{0} parameter(3)
  output_offsets = s32[8]{0} parameter(4)
  recv_sizes = s32[8]{0} parameter(5)
  ROOT ra2a = bf16[1024,256]{1,0} ragged-all-to-all(input, output, input_offsets, send_sizes, output_offsets, recv_sizes), replica_groups={{0,1,2,3,4,5,6,7}}
}

"#,
    ragged_all_to_all_with_collective_device_list: r#"HloModule RaggedAllToAll, entry_computation_layout={(bf16[1024,256]{1,0}, bf16[1024,256]{1,0}, s32[8]{0}, s32[8]{0}, s32[8]{0}, /*index=5*/s32[8]{0})->bf16[1024,256]{1,0}}, replica_count=8

ENTRY AllToAll {
  input = bf16[1024,256]{1,0} parameter(0)
  output = bf16[1024,256]{1,0} parameter(1)
  input_offsets = s32[8]{0} parameter(2)
  send_sizes = s32[8]{0} parameter(3)
  output_offsets = s32[8]{0} parameter(4)
  recv_sizes = s32[8]{0} parameter(5)
  ROOT ra2a = bf16[1024,256]{1,0} ragged-all-to-all(input, output, input_offsets, send_sizes, output_offsets, recv_sizes), replica_groups=[2,4]<=[4,2]T(1,0)
}

"#,
    ragged_all_to_all: r#"HloModule RaggedAllToAll, entry_computation_layout={(bf16[1024,256]{1,0}, bf16[1024,256]{1,0}, s32[8]{0}, s32[8]{0}, s32[8]{0}, /*index=5*/s32[8]{0})->bf16[1024,256]{1,0}}, replica_count=8

ENTRY AllToAll {
  input = bf16[1024,256]{1,0} parameter(0)
  output = bf16[1024,256]{1,0} parameter(1)
  input_offsets = s32[8]{0} parameter(2)
  send_sizes = s32[8]{0} parameter(3)
  output_offsets = s32[8]{0} parameter(4)
  recv_sizes = s32[8]{0} parameter(5)
  ROOT ra2a = bf16[1024,256]{1,0} ragged-all-to-all(input, output, input_offsets, send_sizes, output_offsets, recv_sizes), replica_groups={}
}

"#,
    collective_broadcast: r#"HloModule CollectiveBroadcast, entry_computation_layout={(f32[128,32]{0,1})->f32[128,32]{0,1}}, replica_count=4

ENTRY CollectiveBroadcast {
  input = f32[128,32]{0,1} parameter(0)
  ROOT cb = f32[128,32]{0,1} collective-broadcast(input), replica_groups={{1,0},{2,3}}, has_dynamic_root=false
}

"#,
    collective_broadcast_dynamic_root: r#"HloModule CollectiveBroadcast, entry_computation_layout={(f32[128,32]{0,1}, f32[1]{0})->f32[128,32]{0,1}}, replica_count=4

ENTRY CollectiveBroadcast {
  input = f32[128,32]{0,1} parameter(0)
  root = f32[1]{0} parameter(1)
  ROOT cb = f32[128,32]{0,1} collective-broadcast(input, root), replica_groups={{1,0},{2,3}}, has_dynamic_root=true
}

"#,
    collective_reduce: r#"HloModule CR, entry_computation_layout={(f32[8]{0})->f32[8]{0}}, replica_count=4

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY CR {
  input = f32[8]{0} parameter(0)
  ROOT cr = f32[8]{0} collective-reduce(input), replica_groups={{0,1},{2,3}}, has_dynamic_root=false, to_apply=add
}

"#,
    collective_reduce_with_channel_id: r#"HloModule CR, entry_computation_layout={(f32[8]{0})->f32[8]{0}}, replica_count=2

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY CR {
  input = f32[8]{0} parameter(0)
  ROOT cr = f32[8]{0} collective-reduce(input), channel_id=1, replica_groups={{0,1}}, use_global_device_ids=true, has_dynamic_root=false, to_apply=add
}

"#,
    collective_reduce_dynamic_root: r#"HloModule CR, entry_computation_layout={(f32[8]{0}, s32[1]{0})->f32[8]{0}}, replica_count=4

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY CR {
  input = f32[8]{0} parameter(0)
  root = s32[1]{0} parameter(1)
  ROOT cr = f32[8]{0} collective-reduce(input, root), replica_groups={{0,1},{2,3}}, has_dynamic_root=true, to_apply=add
}

"#,
    collective_permute: r#"HloModule CollectivePermute, entry_computation_layout={(f32[128,32]{0,1})->f32[128,32]{0,1}}, replica_count=4

ENTRY CollectivePermute {
  input = f32[128,32]{0,1} parameter(0)
  ROOT root = f32[128,32]{0,1} collective-permute(input), source_target_pairs={{0,1},{1,2},{2,3}}
}

"#,
    combined_collective_permute: r#"HloModule CombinedCollectivePermute, entry_computation_layout={(f32[128,32]{0,1}, f32[128,32]{0,1})->(f32[128,32]{0,1}, f32[128,32]{0,1})}, replica_count=4

ENTRY CollectivePermute {
  input.0 = f32[128,32]{0,1} parameter(0)
  input.1 = f32[128,32]{0,1} parameter(1)
  ROOT root = (f32[128,32]{0,1}, f32[128,32]{0,1}) collective-permute(input.0, input.1), source_target_pairs={{0,1},{1,2},{2,3}}
}

"#,
    collective_permute_in_place_update: r#"HloModule CollectivePermuteInPlaceUpdate, entry_computation_layout={(f32[128,32]{0,1})->f32[128,128]{0,1}}, replica_count=4

ENTRY CollectivePermuteInPlaceUpdate {
  input = f32[128,32]{0,1} parameter(0)
  constant = f32[] constant(1)
  output = f32[128,128]{0,1} broadcast(constant), dimensions={}
  constant.1 = s32[] constant(0)
  tuple.1 = (s32[], s32[]) tuple(constant.1, constant.1)
  constant.2 = s32[] constant(64)
  tuple.2 = (s32[], s32[]) tuple(constant.1, constant.2)
  ROOT root = f32[128,128]{0,1} collective-permute(input, output, tuple.1, tuple.2), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{128,32}}
}

"#,
    collective_permute_in_place_update3_d: r#"HloModule CollectivePermuteInPlaceUpdate3D, entry_computation_layout={(f32[128,128,32]{0,1,2})->f32[128,128,128]{0,1,2}}, replica_count=4

ENTRY CollectivePermuteInPlaceUpdate {
  input = f32[128,128,32]{0,1,2} parameter(0)
  constant = f32[] constant(1)
  output = f32[128,128,128]{0,1,2} broadcast(constant), dimensions={}
  constant.1 = s32[] constant(0)
  tuple.1 = (s32[], s32[], s32[]) tuple(constant.1, constant.1, constant.1)
  constant.2 = s32[] constant(64)
  tuple.2 = (s32[], s32[], s32[]) tuple(constant.1, constant.1, constant.2)
  ROOT root = f32[128,128,128]{0,1,2} collective-permute(input, output, tuple.1, tuple.2), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{128,128,32}}
}

"#,
    collective_permute_in_place_update_multiple_read_write: r#"HloModule CollectivePermuteInPlaceUpdateMultipleReadWrite, entry_computation_layout={(f32[8,8,128]{2,1,0})->f32[8,8,128]{2,1,0}}, replica_count=4

ENTRY CollectivePermuteInPlaceUpdate {
  constant.3 = s32[] constant(2)
  constant.1 = s32[] constant(0)
  output_offset.3 = (s32[], s32[], s32[]) tuple(constant.3, constant.1, constant.1)
  constant.4 = s32[] constant(3)
  output_offset.4 = (s32[], s32[], s32[]) tuple(constant.4, constant.1, constant.1)
  input = f32[8,8,128]{2,1,0} parameter(0)
  constant = f32[] constant(1)
  output = f32[8,8,128]{2,1,0} broadcast(constant), dimensions={}
  input_offset.1 = (s32[], s32[], s32[]) tuple(constant.1, constant.1, constant.1)
  constant.2 = s32[] constant(1)
  input_offset.2 = (s32[], s32[], s32[]) tuple(constant.2, constant.1, constant.1)
  input_offset = ((s32[], s32[], s32[]), (s32[], s32[], s32[])) tuple(input_offset.1, input_offset.2)
  output_offset = ((s32[], s32[], s32[]), (s32[], s32[], s32[])) tuple(input_offset.1, input_offset.2)
  ROOT root = f32[8,8,128]{2,1,0} collective-permute(input, output, input_offset, output_offset), source_target_pairs={{0,1},{1,2},{2,3},{0,3},{2,1},{3,2}}, slice_sizes={{1,8,128},{1,8,128}}
}

"#,
    collective_permute_in_place_update_tuple_multiple_read_write: r#"HloModule hlo_runner_test_0.1, entry_computation_layout={()->(u32[2,8,128]{2,1,0:T(2,128)}, u32[4,8,128]{2,1,0:T(2,128)})}, replica_count=4

ENTRY hlo_runner_test_0.1 {
  replica_id = u32[] replica-id()
  broadcast.0 = u32[2,8,128]{2,1,0:T(2,128)} broadcast(replica_id), dimensions={}
  tuple.input = (u32[2,8,128]{2,1,0:T(2,128)}, u32[2,8,128]{2,1,0:T(2,128)}) tuple(broadcast.0, broadcast.0)
  constant.1 = u32[] constant(1000)
  broadcast.1 = u32[2,8,128]{2,1,0:T(2,128)} broadcast(constant.1), dimensions={}
  broadcast.2 = u32[4,8,128]{2,1,0:T(2,128)} broadcast(constant.1), dimensions={}
  tuple.output = (u32[2,8,128]{2,1,0:T(2,128)}, u32[4,8,128]{2,1,0:T(2,128)}) tuple(broadcast.1, broadcast.2)
  constant.2 = s32[] constant(0)
  tuple.2 = (s32[], s32[], s32[]) tuple(constant.2, constant.2, constant.2)
  constant.3 = s32[] constant(1)
  tuple.3 = (s32[], s32[], s32[]) tuple(constant.3, constant.2, constant.2)
  tuple.4 = ((s32[], s32[], s32[]), (s32[], s32[], s32[])) tuple(tuple.2, tuple.3)
  tuple.7 = ((s32[], s32[], s32[]), (s32[], s32[], s32[])) tuple(tuple.2, tuple.2)
  tuple.8 = (((s32[], s32[], s32[]), (s32[], s32[], s32[])), ((s32[], s32[], s32[]), (s32[], s32[], s32[]))) tuple(tuple.4, tuple.7)
  constant.4 = s32[] constant(2)
  tuple.5 = (s32[], s32[], s32[]) tuple(constant.4, constant.2, constant.2)
  tuple.6 = ((s32[], s32[], s32[]), (s32[], s32[], s32[])) tuple(tuple.2, tuple.5)
  tuple.9 = (((s32[], s32[], s32[]), (s32[], s32[], s32[])), ((s32[], s32[], s32[]), (s32[], s32[], s32[]))) tuple(tuple.4, tuple.6)
  ROOT collective-permute.53 = (u32[2,8,128]{2,1,0:T(2,128)}, u32[4,8,128]{2,1,0:T(2,128)}) collective-permute(tuple.input, tuple.output, tuple.8, tuple.9), source_target_pairs={{0,1},{1,2},{2,3},{3,0},{0,3},{3,2},{2,1},{1,0}}, slice_sizes={{1,8,128},{1,8,128},{2,8,128},{2,8,128}}
}

"#,
    collective_permute_tuple_in_place_update: r#"HloModule CollectivePermuteTupleInPlaceUpdate, entry_computation_layout={(f32[128,32]{0,1})->(f32[128,128]{0,1}, f32[128,128]{0,1})}, replica_count=4

ENTRY CollectivePermuteInPlaceUpdate {
  input = f32[128,32]{0,1} parameter(0)
  tuple.input = (f32[128,32]{0,1}, f32[128,32]{0,1}) tuple(input, input)
  constant = f32[] constant(1)
  output = f32[128,128]{0,1} broadcast(constant), dimensions={}
  tuple.output = (f32[128,128]{0,1}, f32[128,128]{0,1}) tuple(output, output)
  constant.1 = s32[] constant(0)
  tuple.1 = (s32[], s32[]) tuple(constant.1, constant.1)
  constant.2 = s32[] constant(64)
  tuple.2 = (s32[], s32[]) tuple(constant.2, constant.1)
  tuple.3 = ((s32[], s32[]), (s32[], s32[])) tuple(tuple.1, tuple.2)
  tuple.4 = (s32[], s32[]) tuple(constant.1, constant.1)
  tuple.5 = (s32[], s32[]) tuple(constant.2, constant.2)
  tuple.6 = ((s32[], s32[]), (s32[], s32[])) tuple(tuple.4, tuple.5)
  ROOT root = (f32[128,128]{0,1}, f32[128,128]{0,1}) collective-permute(tuple.input, tuple.output, tuple.3, tuple.6), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{64,32},{64,32}}
}

"#,
    collective_permute_start_and_done: r#"HloModule CollectivePermuteStartAndDone, entry_computation_layout={(f32[128,32]{0,1})->f32[128,32]{0,1}}, replica_count=4

ENTRY CollectivePermuteStartAndDone {
  input = f32[128,32]{0,1} parameter(0)
  collective-permute-start.1 = (f32[128,32]{0,1}, f32[128,32]{0,1}, u32[], u32[]) collective-permute-start(input), source_target_pairs={{0,1},{1,2},{2,3}}
  ROOT collective-permute-done.1 = f32[128,32]{0,1} collective-permute-done(collective-permute-start.1)
}

"#,
    combined_collective_permute_start_and_done: r#"HloModule CombinedCollectivePermuteStartAndDone, entry_computation_layout={(f32[128,32]{0,1}, f32[128,32]{0,1})->(f32[128,32]{0,1}, f32[128,32]{0,1})}, replica_count=4

ENTRY CombinedCollectivePermuteStartAndDone {
  input.0 = f32[128,32]{0,1} parameter(0)
  input.1 = f32[128,32]{0,1} parameter(1)
  collective-permute-start.1 = ((f32[128,32]{0,1}, f32[128,32]{0,1}), (f32[128,32]{0,1}, f32[128,32]{0,1})) collective-permute-start(input.0, input.1), source_target_pairs={{0,1},{1,2},{2,3}}
  ROOT collective-permute-done.1 = (f32[128,32]{0,1}, f32[128,32]{0,1}) collective-permute-done(collective-permute-start.1)
}

"#,
    collective_permute_start_and_done_inplace_update: r#"HloModule CollectivePermuteStartAndDoneInplaceUpdate, entry_computation_layout={(f32[128,32]{0,1})->f32[128,128]{0,1}}, replica_count=4

ENTRY CollectivePermuteStartAndDoneInplaceUpdate {
  input = f32[128,32]{0,1} parameter(0)
  constant = f32[] constant(1)
  output = f32[128,128]{0,1} broadcast(constant), dimensions={}
  constant.1 = s32[] constant(0)
  tuple.1 = (s32[], s32[]) tuple(constant.1, constant.1)
  constant.2 = s32[] constant(64)
  tuple.2 = (s32[], s32[]) tuple(constant.1, constant.2)
  collective-permute-start.1 = (f32[128,32]{0,1}, f32[128,128]{0,1}, u32[], u32[]) collective-permute-start(input, output, tuple.1, tuple.2), source_target_pairs={{0,1},{1,2},{2,3}}, slice_sizes={{64,32}}
  ROOT collective-permute-done.1 = f32[128,128]{0,1} collective-permute-done(collective-permute-start.1)
}

"#,
    replica_id: r#"HloModule replica-id, entry_computation_layout={()->u32[]}

ENTRY Replica-id {
  ROOT replica-id = u32[] replica-id()
}

"#,
    partition_id: r#"HloModule partition-id, entry_computation_layout={()->u32[]}

ENTRY PartitionId {
  ROOT id = u32[] partition-id()
}

"#,
    iota: r#"HloModule iota, entry_computation_layout={()->f32[100]{0}}

ENTRY Iota {
  ROOT iota = f32[100]{0} iota(), iota_dimension=0
}

"#,
    custom_call_with_window_and_dim_labels_and_feature_group_count: r#"HloModule CustomCallWithWindowAndDimLabelsAndFeatureGroupCount, entry_computation_layout={()->f32[100]{0}}

ENTRY Computation {
  ROOT r = f32[100]{0} custom-call(), window={size=2x2}, dim_labels=b01f_01io->b01f, feature_group_count=2, custom_call_target="target"
}

"#,
    custom_call_with_unknown_dim_labels: r#"HloModule CustomCallWithUnknownDimLabels, entry_computation_layout={()->f32[100]{0}}

ENTRY Computation {
  ROOT r = f32[100]{0} custom-call(), window={size=2x2}, dim_labels=?b01f_0?1io->b01?f, custom_call_target="target"
}

"#,
    scheduled_module: r#"HloModule scheduled_module, is_scheduled=true, entry_computation_layout={(f32[1024]{0}, s32[1024]{0})->(f32[1024]{0}, s32[1024]{0})}

compare {
  p.1.lhs = s32[] parameter(2)
  p.1.rhs = s32[] parameter(3)
  p.0.lhs = f32[] parameter(0)
  p.0.rhs = f32[] parameter(1)
  ROOT lhs = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
}

ENTRY Sort {
  keys = f32[1024]{0} parameter(0)
  values = s32[1024]{0} parameter(1)
  ROOT sorted = (f32[1024]{0}, s32[1024]{0}) sort(keys, values), dimensions={0}, to_apply=compare
}

"#,
    after_all_with_multiple_operands: r#"HloModule AfterAllWithMultipleOperands, entry_computation_layout={(f32[])->token[]}

ENTRY AfterAllWithMultipleOperands {
  p0 = f32[] parameter(0)
  token0 = token[] after-all()
  token1 = token[] after-all()
  ROOT after-all = token[] after-all(p0, token0, token1)
}

"#,
    add_dependency: r#"HloModule AddDependency, entry_computation_layout={(f32[])->f32[]}

ENTRY AddDependency {
  p = f32[] parameter(0)
  neg = f32[] negate(p)
  token0 = token[] after-all(neg)
  p_after_token = f32[] add-dependency(p, token0)
  exp = f32[] exponential(p_after_token)
  ROOT sum = f32[] add(neg, exp)
}

"#,
    min_max_values: r#"HloModule MinMaxValues, entry_computation_layout={()->c128[2]{0}}

ENTRY MinMaxValues {
  x.s4 = s4[2]{0} constant({-8, 7})
  x.s8 = s8[2]{0} constant({-128, 127})
  x.s16 = s16[2]{0} constant({-32768, 32767})
  x.s32 = s32[2]{0} constant({-2147483648, 2147483647})
  x.u4 = u4[2]{0} constant({0, 15})
  x.u8 = u8[2]{0} constant({0, 255})
  x.u16 = u16[2]{0} constant({0, 65535})
  x.u32 = u32[2]{0} constant({0, 4294967295})
  x.f16 = f16[2]{0} constant({-65504, 65504})
  x.bf16 = bf16[2]{0} constant({-3.39e+38, 3.39e+38})
  x.f32 = f32[2]{0} constant({-3.40282e+38, 3.40282e+38})
  x.f64 = f64[2]{0} constant({-1.79769e+308, 1.79769e+308})
  x.c64 = c64[2]{0} constant({(-3.40282e+38, 3.40282e+38), (3.40282e+38, -3.40282e+38)})
  ROOT c.c128 = c128[2]{0} constant({(-1.79769e+308, 1.79769e+308), (1.79769e+308, -1.79769e+308)})
}

"#,
    bitcast_convert: r#"HloModule BitcastConvert, entry_computation_layout={(f32[100]{0})->u32[100]{0}}

ENTRY BitcastConvertUsage {
  p = f32[100]{0} parameter(0)
  ROOT out = u32[100]{0} bitcast-convert(p)
}

"#,
    scan: r#"HloModule scan_module, entry_computation_layout={(f32[4]{0})->(f32[4]{0}, f32[])}

add_F32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  add = f32[] add(lhs, rhs)
  ROOT t = (f32[], f32[]) tuple(add, add)
}

ENTRY Scan {
  input = f32[4]{0} parameter(0)
  init = f32[] constant(0)
  ROOT scan = (f32[4]{0}, f32[]) scan(input, init), dimensions={0}, num_carries=1, is_reverse=true, is_associative=true, to_apply=add_F32
}

"#,
        }
}

mod hlo_non_roundtrip_parser_test {
    use crate::hlo_text::Style;

    corpus! {
    "nonroundtrip", Style::Short,
    simple_nesting: r#"HloModule test, entry_computation_layout={(f32[10]{0}, f32[10]{0}, f32[10]{0})->f32[10]{0}}

ENTRY test {
  parameter.anon = f32[10]{0} parameter(0)
  parameter.anon.1 = f32[10]{0} parameter(1)
  parameter.anon.2 = f32[10]{0} parameter(2)
  multiply.anon = f32[10]{0} multiply(parameter.anon.1, parameter.anon.2)
  ROOT root = f32[10]{0} add(parameter.anon, multiply.anon)
}"#,
    ambiguous_names: r#"HloModule test, entry_computation_layout={(f32[10]{0}, f32[10]{0})->f32[10]{0}}

ENTRY test {
  parameter.anon = f32[10]{0} parameter(0)
  parameter.anon.1 = f32[10]{0} parameter(1)
  add = f32[10]{0} add(parameter.anon, parameter.anon.1)
  add.anon = f32[10]{0} add(add, add)
  ROOT add2 = f32[10]{0} add(add, add.anon)
}"#,
    tuple_shape_inside_anonymous_instr: r#"HloModule test, entry_computation_layout={(f32[10]{0}, f16[10]{0})->f32[10]{0}}

ENTRY test {
  parameter.anon = f32[10]{0} parameter(0)
  parameter.anon.1 = f16[10]{0} parameter(1)
  tuple.anon = (f32[10]{0}, f16[10]{0}) tuple(parameter.anon, parameter.anon.1)
  ROOT root = f32[10]{0} get-tuple-element(tuple.anon), index=0
}"#,
    mix_anon_and_non_anon_operands: r#"HloModule test, entry_computation_layout={(f32[10]{0}, f32[10]{0})->(f32[10]{0}, f32[10]{0}, f32[10]{0})}

ENTRY test {
  parameter.anon = f32[10]{0} parameter(0)
  parameter.anon.1 = f32[10]{0} parameter(1)
  add = f32[10]{0} add(parameter.anon, parameter.anon.1)
  add.anon = f32[10]{0} add(add, add)
  ROOT root = (f32[10]{0}, f32[10]{0}, f32[10]{0}) tuple(add, add.anon, add)
}"#,
    broadcast_of_scalar_doesnt_need_dimensions_attr: r#"HloModule test, entry_computation_layout={(f32[])->f32[10,10]{1,0}}

ENTRY test {
  parameter.anon = f32[] parameter(0)
  broadcast.anon = f32[10,10]{1,0} broadcast(parameter.anon), dimensions={}
  ROOT root = f32[10,10]{1,0} sqrt(broadcast.anon)
}"#,
    compact_gte: r#"HloModule test, entry_computation_layout={((f32[10]{0}, f16[10]{0}))->f32[10]{0}}

ENTRY test {
  p0 = (f32[10]{0}, f16[10]{0}) parameter(0)
  p0_0 = f32[10]{0} get-tuple-element(p0), index=0
  ROOT root = f32[10]{0} add(p0_0, p0_0)
}"#,
    nested_compact_gte: r#"HloModule test, entry_computation_layout={((f32[10]{0}, (f32[20]{0}, f32[30]{0})))->f32[20]{0}}

ENTRY test {
  p0 = (f32[10]{0}, (f32[20]{0}, f32[30]{0})) parameter(0)
  p0_1 = (f32[20]{0}, f32[30]{0}) get-tuple-element(p0), index=1
  p0_1_0 = f32[20]{0} get-tuple-element(p0_1), index=0
  ROOT root = f32[20]{0} add(p0_1_0, p0_1_0)
}"#,
    compact_gte_multiple_uses: r#"HloModule test, entry_computation_layout={((f32[10]{0}, f16[10]{0}))->f32[10]{0}}

ENTRY test {
  p0 = (f32[10]{0}, f16[10]{0}) parameter(0)
  p0_0 = f32[10]{0} get-tuple-element(p0), index=0
  gte_use1 = f32[10]{0} negate(p0_0)
  gte_use2 = f32[10]{0} log(p0_0)
  ROOT root = f32[10]{0} add(gte_use1, gte_use2)
}"#,
    compact_gte_name_collision: r#"HloModule test, entry_computation_layout={((f32[10]{0}, f16[10]{0}), f32[10]{0})->f32[10]{0}}

ENTRY test {
  p0 = (f32[10]{0}, f16[10]{0}) parameter(0)
  p0_0.1 = f32[10]{0} get-tuple-element(p0), index=0
  p0_0 = f32[10]{0} parameter(1)
  ROOT root = f32[10]{0} add(p0_0.1, p0_0)
}"#,
    compact_gte_mixed_nesting: r#"HloModule test, entry_computation_layout={((f32[10]{0}, (f32[20]{0}, f32[30]{0})))->((f32[20]{0}, f32[30]{0}), f32[20]{0})}

ENTRY test {
  p0 = (f32[10]{0}, (f32[20]{0}, f32[30]{0})) parameter(0)
  p0_1 = (f32[20]{0}, f32[30]{0}) get-tuple-element(p0), index=1
  p0_1_0 = f32[20]{0} get-tuple-element(p0_1), index=0
  ROOT root = ((f32[20]{0}, f32[30]{0}), f32[20]{0}) tuple(p0_1, p0_1_0)
}"#,
    compact_gte_on_tuple: r#"HloModule test, entry_computation_layout={(f32[10]{0}, f16[10]{0})->f32[10]{0}}

ENTRY test {
  p0 = f32[10]{0} parameter(0)
  p1 = f16[10]{0} parameter(1)
  t = (f32[10]{0}, f16[10]{0}) tuple(p0, p1)
  t_0 = f32[10]{0} get-tuple-element(t), index=0
  ROOT root = f32[10]{0} negate(t_0)
}"#,
        }
}

mod hlo_parser_test_printing {
    use crate::hlo::Module;
    use crate::hlo_text::{Printer, Style};
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

    #[test]
    fn parse_sharding() {
        let original = r#"{maximal device=42}"#;
        assert!(instruction("parse_sharding").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_named_sharding_unreduced_max() {
        let original = r#"{mesh['x'=2], [{}], unreduced=max{'x'}}"#;
        assert!(instruction("parse_named_sharding_unreduced_max").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_sharding_partial_replication() {
        let original = r#"{devices=[2,2]0,1,2,3 last_tile_dim_replicate}"#;
        assert!(instruction("parse_sharding_partial_replication").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_sharding_sub_group() {
        let original = r#"{devices=[2,2,2,2]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 last_tile_dims={manual, replicated}}"#;
        assert!(instruction("parse_sharding_sub_group").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_named_sharding1() {
        let original = r#"{mesh['a'=2,'b'=4,'c'=3,'d'=8], [{'a'}, {'b'}]}"#;
        assert!(instruction("parse_named_sharding1").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_named_sharding2() {
        let original = r#"{mesh['a'=2,'b'=4,'c'=3,'d'=8], [{'a'}, {'c', 'b'}]}"#;
        assert!(instruction("parse_named_sharding2").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_named_sharding_open_dims() {
        let original = r#"{mesh['a'=2,'b'=4,'c'=3,'d'=8], [{'b', 'a'}, {'c', 'd', ?}]}"#;
        assert!(instruction("parse_named_sharding_open_dims").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_named_sharding_sub_axes1() {
        let original = r#"{mesh['a'=2,'b'=4,'c'=3,'d'=8], [{'a'}, {'b':(2)2}]}"#;
        assert!(instruction("parse_named_sharding_sub_axes1").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_named_sharding_sub_axes2() {
        let original = r#"{mesh['a'=2,'b'=4,'c'=3,'d'=8], [{'b':(2)2}, {'d':(4)2, 'c'}]}"#;
        assert!(instruction("parse_named_sharding_sub_axes2").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_named_sharding_sub_axes_open_dims() {
        let original = r#"{mesh['a'=2,'b'=4,'c'=3,'d'=8], [{'b':(2)2}, {'d':(4)2, 'c', ?}]}"#;
        assert!(instruction("parse_named_sharding_sub_axes_open_dims").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_named_sharding_non_iota_mesh() {
        let original = r#"{mesh['a'=2,'b'=4,'c'=4,'d'=2], device_ids=([4,16]T(1,0)), [{'a'}]}"#;
        assert!(instruction("parse_named_sharding_non_iota_mesh").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_named_sharding_non_iota_mesh_device_list() {
        let original = r#"{mesh['x'=2,'y'=2], device_ids=(0,2,1,3), [{'x'}]}"#;
        assert!(instruction("parse_named_sharding_non_iota_mesh_device_list").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_named_sharding_empty_mesh_replicated() {
        let original = r#"{mesh[], replicated}"#;
        assert!(instruction("parse_named_sharding_empty_mesh_replicated").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_named_sharding_fully_replicated() {
        let original = r#"{mesh['a'=2,'b'=4], replicated}"#;
        assert!(instruction("parse_named_sharding_fully_replicated").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_named_sharding_replicated_axes() {
        let original = r#"{mesh['a'=2,'b'=4], [{'a'}], replicated={'b'}}"#;
        assert!(instruction("parse_named_sharding_replicated_axes").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_named_sharding_maximal() {
        let original = r#"{maximal_mesh[device_id=5]}"#;
        assert!(instruction("parse_named_sharding_maximal").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_named_sharding_with_special_characters() {
        let original = r#"{mesh['a.b'=2,'<axis> def'=4,'z/w'=2], [{'a.b'}, {'<axis> def':(2)2, 'z/w'}]}"#;
        assert!(instruction("parse_named_sharding_with_special_characters").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_named_sharding_fully_unreduced() {
        let original = r#"{mesh['a'=2,'b'=4], unreduced}"#;
        assert!(instruction("parse_named_sharding_fully_unreduced").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_named_sharding_unreduced_axes() {
        let original = r#"{mesh['a'=2,'b'=4,'c'=3,'d'=8], [{}, {'b'}], unreduced={'d':(4)2}}"#;
        assert!(instruction("parse_named_sharding_unreduced_axes").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_named_sharding_fully_manual() {
        let original = r#"{mesh['a'=2,'b'=4], manual}"#;
        assert!(instruction("parse_named_sharding_fully_manual").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_named_sharding_manual_axes() {
        let original = r#"{mesh['a'=2,'b'=4,'c'=3,'d'=8], [{'a'}], manual={'d':(4)2}}"#;
        assert!(instruction("parse_named_sharding_manual_axes").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_named_sharding_all_fields_with_metadata() {
        let original = r#"{mesh['a'=2,'b'=4,'c'=3,'d'=8], [{'a'}], replicated={'c'}, unreduced={'d':(4)2}, manual={'b':(2)2}, metadata={{op_name="foo"}, {op_name="bar"}}}"#;
        assert!(instruction("parse_named_sharding_all_fields_with_metadata").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_named_sharding_fully_replicated_with_metadata() {
        let original = r#"{mesh['a'=2,'b'=4], replicated, metadata={{op_name="foo"}}}"#;
        assert!(instruction("parse_named_sharding_fully_replicated_with_metadata").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_named_sharding_tuple() {
        let original = r#"{{mesh['a'=2,'b'=4,'c'=3,'d'=8], [{'d', 'c'}, {'a', 'b'}]}, {mesh['a'=2,'b'=4,'c'=3,'d'=8], replicated}, {mesh['a'=2,'b'=4,'c'=3,'d'=8], [{'d':(2)2, 'b'}, {'a', ?}], unreduced={'c'}}, {mesh['a'=2,'b'=4,'c'=3,'d'=8], [{'d', 'c'}, {'a', 'b'}], metadata={{op_name="foo"}, {op_name="bar"}}}}"#;
        assert!(instruction("parse_named_sharding_tuple").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_mixed_sharding_tuple1() {
        let original = r#"{{replicated}, {mesh['a'=2,'b'=4], replicated}, {maximal device=5}, {maximal_mesh[device_id=5]}}"#;
        assert!(instruction("parse_mixed_sharding_tuple1").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_mixed_sharding_tuple2() {
        let original = r#"{{mesh['a'=2,'b'=2], [{'a'}, {}]}, {devices=[2,2]<=[4] last_tile_dim_replicate}}"#;
        assert!(instruction("parse_mixed_sharding_tuple2").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_trivial_iota_sharding_partial_replication() {
        let original = r#"{devices=[2,2]<=[4] last_tile_dim_replicate}"#;
        assert!(instruction("parse_trivial_iota_sharding_partial_replication").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_trivial_iota_sharding_sub_group() {
        let original = r#"{devices=[2,2,2,2]<=[16] last_tile_dims={manual, replicated}}"#;
        assert!(instruction("parse_trivial_iota_sharding_sub_group").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_transposed_iota_sharding_partial_replication() {
        let original = r#"{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"#;
        assert!(instruction("parse_transposed_iota_sharding_partial_replication").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_transposed_iota_sharding_sub_group() {
        let original = r#"{devices=[2,2,2,2]<=[2,2,4]T(2,1,0) last_tile_dims={manual, replicated}}"#;
        assert!(instruction("parse_transposed_iota_sharding_sub_group").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_shard_as() {
        let original = r#"{manual shard_as 1}"#;
        assert!(instruction("parse_shard_as").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_shard_like() {
        let original = r#"{devices=[2,2,2,2]<=[16] last_tile_dims={manual, replicated} shard_like 1}"#;
        assert!(instruction("parse_shard_like").contains(&format!("sharding={original}")));
    }

    #[test]
    fn parse_unknown_sharding() {
        let original = r#"{unknown}"#;
        assert!(instruction("parse_unknown_sharding").contains(&format!("sharding={original}")));
    }

    #[test]
    #[ignore = "differs from upstream: XProf 2.23.2 prints a fully unreduced named sharding as unreduced=max without its axis list (checked over HTTP); XLA prints the axes only for a strict subset of the mesh"]
    fn parse_named_sharding_scalar_unreduced_max() {
        assert!(instruction("parse_named_sharding_scalar_unreduced_max").contains("sharding={mesh['x'=2,'y'=2], unreduced=max{'x', 'y'}}"));
    }

    #[test]
    fn parse_frontend_attributes() {
        let original = r#"{attr_a="test_a",attr_b="b",attr_c={type="s64"},attr_d="a=\"b/c\""}"#;
        assert!(instruction("parse_frontend_attributes").contains(&format!("frontend_attributes={original}")));
    }

    #[test]
    fn parse_window() {
        let original = r#"{size=1x2x3}"#;
        assert!(instruction("parse_window").contains(&format!("window={original}")));
    }

    #[test]
    fn parse_convolution_dimension_numbers() {
        let original = r#"b0f_0io->b0f"#;
        assert!(instruction("parse_convolution_dimension_numbers").contains(&format!("dim_labels={original}")));
    }

    #[test]
    fn parse_replica_groups() {
        let original = r#"{{0,1},{2,3}}"#;
        assert!(instruction("parse_replica_groups").contains(&format!("replica_groups={original}")));
    }

    #[test]
    fn parse_collective_device_list_v1() {
        let original = r#"{{0,1},{2,3}}"#;
        assert!(instruction("parse_replica_groups").contains(&format!("replica_groups={original}")));
    }

    #[test]
    fn parse_collective_device_list_v2() {
        let original = r#"[2,2]<=[4]"#;
        assert!(instruction("parse_collective_device_list_v2").contains(&format!("replica_groups={original}")));
    }

    #[test]
    fn parse_collective_device_list_v3() {
        let original = r#"mesh['axis_0'=2,'axis_1'=2] {'axis_1'}"#;
        assert!(instruction("parse_replica_groups_v3").contains(&format!("replica_groups={original}")));
    }

    #[test]
    fn parse_padding_config_no_interior_padding() {
        let original = r#"0_1x2_3"#;
        assert!(instruction("parse_padding_config_no_interior_padding").contains(&format!("padding={original}")));
    }

    #[test]
    fn parse_padding_config_interior_padding() {
        let original = r#"0_1_0x2_3_4"#;
        assert!(instruction("parse_padding_config_interior_padding").contains(&format!("padding={original}")));
    }

    #[test]
    fn short_constant() {
        let original = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/hlo_parser_test/printing/short_constant.hlo"));
        assert_eq!(module_text("short_constant", Style::Long), original);
    }

    #[test]
    fn negative_nan() {
        let original = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/hlo_parser_test/printing/negative_nan.hlo"));
        assert_eq!(module_text("negative_nan", Style::Long), original);
    }

    #[test]
    fn nan_payload() {
        let original = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/hlo_parser_test/printing/nan_payload.hlo"));
        assert_eq!(module_text("nan_payload", Style::Long), original);
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
