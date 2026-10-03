use super::cli_support::{Fake, args, json, parse, same};
use crate::cli::json::J;
use crate::cli::overview::get_utilization_viewer;

type Case<'a> = (&'a str, Option<String>, Vec<(&'a str, J)>, &'a str);

const HEADER: &str = "Host,Device,Sample,Node,Name,Achieved,Peak,Unit\n";
const DATATABLE: &str = r#"{"cols": [{"id": "host", "label": "Host", "type": "number"}, {"id": "device", "label": "Device", "type": "number"}, {"id": "sample", "label": "Sample", "type": "number"}, {"id": "node", "label": "Node", "type": "number"}, {"id": "name", "label": "Name", "type": "string"}, {"id": "achieved", "label": "Achieved", "type": "number"}, {"id": "peak", "label": "Peak", "type": "number"}, {"id": "unit", "label": "Unit", "type": "string"}], "rows": [{"c": [{"v": 0}, {"v": 0}, {"v": 0}, {"v": 0}, {"v": "Vector ALUs"}, {"v": 1.0}, {"v": 2.0}, {"v": "instructions"}]}, {"c": [{"v": 0}, {"v": 0}, {"v": 0}, {"v": 0}, {"v": "Scalar Unit"}, {"v": 2.0}, {"v": 4.0}, {"v": "instructions"}]}, {"c": [{"v": 0}, {"v": 0}, {"v": 0}, {"v": 0}, {"v": "MXU0"}, {"v": 6.0}, {"v": 12.0}, {"v": "instructions"}]}, {"c": [{"v": 0}, {"v": 0}, {"v": 0}, {"v": 0}, {"v": "HBM Rd+Wr (per chip)"}, {"v": 8.0}, {"v": 16.0}, {"v": "bytes"}]}]}"#;
const SUCCESS_CSV: &str = "0,0,0,0,Vector ALUs,1.0,2.0,instructions
0,0,0,0,Scalar Unit,2.0,4.0,instructions
0,0,0,0,Vmem/Cmem Stores,3.0,6.0,instructions
0,0,0,0,Vmem Loads,4.0,8.0,instructions
0,0,0,0,Cmem Loads,5.0,10.0,instructions
0,0,0,0,MXU0,6.0,12.0,instructions
0,0,0,0,XLU0,7.0,14.0,instructions
0,0,0,0,HBM Rd+Wr,8.0,16.0,bytes
0,0,0,0,ICI (Read),9.0,18.0,bytes
0,0,0,0,ICI (Write),10.0,20.0,bytes
0,0,0,0,MXU_BF16,2.0,5.0,instructions
0,0,0,0,MXU_I8,3.0,10.0,instructions
0,0,0,0,Avg MXU Busy,4.0,10.0,instructions
";
const NONZERO_CSV: &str = "1,2,0,3,Vector ALUs,1.0,2.0,instructions
1,2,0,3,Scalar Unit,2.0,4.0,instructions
1,2,0,3,MXU0,6.0,12.0,instructions
";
const NONZERO_EXPECTED: &str = r#"{"vector_alu_utilization_percent": 50.0, "scalar_unit_utilization_percent": 50.0, "mxu_utilization_percent": 50.0, "idleness_percent": 50.0, "metrics": {"Vector ALUs": 50.0, "Scalar Unit": 50.0, "MXU0": 50.0}}"#;

#[test]
fn test_get_utilization_viewer() {
    let csv = |rows: &str| Some(format!("{HEADER}{rows}"));
    let cases: [Case; 10] = [
        (
            "success_csv",
            csv(SUCCESS_CSV),
            vec![],
            r#"{"vector_alu_utilization_percent": 50.0, "scalar_unit_utilization_percent": 50.0, "vmem_cmem_stores_utilization_percent": 50.0, "vmem_loads_utilization_percent": 50.0, "cmem_loads_utilization_percent": 50.0, "hbm_bandwidth_utilization_percent": 50.0, "ici_read_utilization_percent": 50.0, "ici_write_utilization_percent": 50.0, "xlu_utilization_percent": 50.0, "mxu_utilization_percent": 40.0, "idleness_percent": 50.0, "metrics": {"Vector ALUs": 50.0, "Scalar Unit": 50.0, "Vmem/Cmem Stores": 50.0, "Vmem Loads": 50.0, "Cmem Loads": 50.0, "MXU0": 50.0, "XLU0": 50.0, "HBM Rd+Wr": 50.0, "ICI (Read)": 50.0, "ICI (Write)": 50.0, "MXU_BF16": 40.0, "MXU_I8": 30.0, "Avg MXU Busy": 40.0}}"#,
        ),
        (
            "success_datatable_json",
            Some(DATATABLE.into()),
            vec![],
            r#"{"vector_alu_utilization_percent": 50.0, "scalar_unit_utilization_percent": 50.0, "mxu_utilization_percent": 50.0, "hbm_bandwidth_utilization_percent": 50.0, "idleness_percent": 50.0, "metrics": {"Vector ALUs": 50.0, "Scalar Unit": 50.0, "MXU0": 50.0, "HBM Rd+Wr (per chip)": 50.0}}"#,
        ),
        ("no_data_none", None, vec![], r#"{"status": "NO_DATA", "message": "No data returned for session test-session"}"#),
        ("no_data_empty_datatable", Some(r#"{"cols": [], "rows": []}"#.into()), vec![], r#"{"status": "NO_DATA", "message": "No hardware performance counter events found in trace"}"#),
        ("no_node0_data", csv("1,0,0,0,Vector ALUs,1.0,2.0,instructions\n"), vec![], r#"{"status": "NO_DATA", "message": "No data found for Host 0 Device 0 Node 0"}"#),
        ("success_non_zero_parameters", csv(NONZERO_CSV), vec![("host", J::Int(1)), ("device", J::Int(2)), ("node", J::Int(3))], NONZERO_EXPECTED),
        (
            "missing_parameters",
            csv("1,2,0,3,Vector ALUs,1.0,2.0,instructions\n"),
            vec![("host", J::Int(1)), ("device", J::Int(2)), ("node", J::Int(4))],
            r#"{"status": "NO_DATA", "message": "No data found for Host 1 Device 2 Node 4"}"#,
        ),
        ("type_coercion", csv(NONZERO_CSV), vec![("host", J::from("1")), ("device", J::from("2")), ("node", J::from("3"))], NONZERO_EXPECTED),
        ("with_no_mxu_busy", csv("0,0,0,0,No MXU Busy,30.0,100.0,instructions\n"), vec![], r#"{"idleness_percent": 30.0, "metrics": {"No MXU Busy": 30.0}}"#),
        (
            "missing_node_column_warning",
            Some("Host,Device,Sample,Name,Achieved,Peak,Unit\n0,0,0,Vector ALUs,1.0,2.0,instructions\n".into()),
            vec![("node", J::Int(1))],
            r#"{"vector_alu_utilization_percent": 50.0, "idleness_percent": 100.0, "metrics": {"Vector ALUs": 50.0}, "warnings": ["Node column missing; ignoring node=1 filter"]}"#,
        ),
    ];
    for (name, payload, flags, expected) in cases {
        let fake = Fake::new(move |_, _| Ok(Some(payload.clone().unwrap_or_default().into_bytes())));
        let result = json(get_utilization_viewer(&fake, &args("test-session", &flags)));
        assert!(same(&result, &parse(expected)), "{name}: {}", result.dumps());
    }
}
