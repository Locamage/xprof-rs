use super::xspace::{XSpace, assert_db};
use crate::tools::opstats::{Builder, Db, EventReader, templates};

fn build(space: &XSpace, tensor_core: bool) -> Db {
    let (map, planes) = space.parsed();
    let (templates, reader) = (templates(&planes[0], &map), EventReader::new(&planes[0]));
    let mut builder = Builder::new(&templates);
    for event in planes[0].lines.iter().flat_map(|line| &line.events) {
        builder.add(event, &reader.read(&map, event), (event.dur, event.dur), tensor_core);
    }
    builder.finish()
}

#[test]
fn from_x_event_handles_missing_occurrences() {
    let mut space = XSpace::default();
    let plane = space.add_plane();
    plane.event_metadata("metadata").display_name = "display_name".into();
    plane.metadata_stats(
        "metadata",
        &[
            ("program_id", 1.into()),
            ("symbol_id", 2.into()),
            ("deduplicated_name", "deduplicated_name".into()),
            ("tf_op", "tf_op".into()),
            ("hlo_category", "tf_op_category".into()),
            ("flops", 3.into()),
            ("model_flops", 4.into()),
            ("bytes_accessed", 5.into()),
            ("source", "my_file.py:123".into()),
            ("source_stack", "my_file.py:123\nmy_other_file.py:456".into()),
        ],
    );
    plane.event(0, "metadata", 0, 100, &[]);
    let db = build(&space, false);
    assert_db(
        &Db { metrics: db.metrics, ..Default::default() },
        r#"metrics_db {
             occurrences: 1
             time_ps: 100
             self_time_ps: 100
             dma_stall_ps: 0
             hlo_module_id: 1
             flops: 3
             flops_v2: 3
             model_flops: 4
             model_flops_v2: 4
             bytes_accessed: 5
             name: "display_name"
             long_name: "metadata"
             deduplicated_name: "deduplicated_name"
             category: "tf_op_category"
             provenance: "tf_op"
             min_time_ps: 100
             num_cores: 1
             source_info {
               file_name: "my_file.py"
               line_number: 123
               stack_frame: "my_file.py:123\nmy_other_file.py:456"
             }
           }"#,
    );
}

#[test]
fn add_op_metric() {
    let mut space = XSpace::default();
    let plane = space.add_plane();
    for (name, display, ids) in
        [("m1", "display_name1", &[("program_id", 1), ("symbol_id", 1)][..]), ("m2", "display_name2", &[("program_id", 1), ("symbol_id", 2)]), ("m3", "display_name3", &[("symbol_id", 1)])]
    {
        plane.event_metadata(name).display_name = display.into();
        plane.metadata_stats(name, &ids.iter().map(|&(key, value)| (key, value.into())).collect::<Vec<_>>());
    }
    plane.event(0, "m1", 0, 100, &[("Time Scale Multiplier", 0.5.into()), ("vdd_energy_j", 10.0.into())]);
    plane.event(0, "m1", 100, 100, &[("Time Scale Multiplier", 1.0.into()), ("vdd_energy_j", 20.0.into())]);
    plane.event(0, "m2", 0, 100, &[("Time Scale Multiplier", 2.0.into())]);
    plane.event(0, "m3", 0, 100, &[]);
    let mut db = build(&space, true);
    db.metrics.iter_mut().for_each(|metrics| metrics.core_type = 0);
    assert_db(
        &db,
        r#"metrics_db {
             hlo_module_id: 1
             self_time_ps: 200
             occurrences: 2
             name: "display_name1"
             long_name: "m1"
             time_ps: 200
             normalized_time_ps: 150
             min_time_ps: 100
             num_cores: 1
             vdd_energy_j: 30
           }
           metrics_db {
             hlo_module_id: 1
             self_time_ps: 100
             occurrences: 1
             name: "display_name2"
             long_name: "m2"
             time_ps: 100
             min_time_ps: 100
             normalized_time_ps: 200
             num_cores: 1
           }
           total_op_time_ps: 300
           normalized_total_op_time_ps: 350"#,
    );
}
