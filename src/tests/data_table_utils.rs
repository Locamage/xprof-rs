use crate::table::{Cell, Table};

const COLUMNS: [(&str, &str, &str); 7] = [
    ("rank", "number", "Rank"),
    ("program_id", "string", "Program Id"),
    ("op_category", "string", "Op Category"),
    ("op_name", "string", "Op Name"),
    ("bytes_accessed", "number", "Bytes Accessed"),
    ("model_flops", "number", "Model Flops"),
    ("occurrences", "number", "#Occurrences"),
];

fn test_rows() -> serde_json::Value {
    serde_json::json!([[1, "11111", "category1", "op1", 200000000, 123123123, 10], [2, "22222", "category2", "op2", 1000000, 0, 20], [3, "33333", "category3", "op3", 3000000, 565656, 30]])
}

fn row_json(cell: Cell) -> serde_json::Value {
    let mut table = Table::default();
    table.row().push(cell);
    assert_eq!(table.rows[0].len(), 1);
    serde_json::from_str::<serde_json::Value>(&table.json()).unwrap()["rows"][0]["c"][0].clone()
}

#[test]
fn to_json() {
    let mut table = Table::new(&COLUMNS);
    for row in test_rows().as_array().unwrap() {
        let cells = table.row();
        for (value, column) in row.as_array().unwrap().iter().zip(COLUMNS) {
            cells.push(if column.1 == "number" { Cell::Number(value.as_f64().unwrap()) } else { Cell::Text(value.as_str().unwrap().into()) });
        }
    }
    let parsed: serde_json::Value = serde_json::from_str(&table.json()).unwrap();
    assert_eq!(parsed["cols"].as_array().unwrap().len(), COLUMNS.len());
    assert_eq!(parsed["rows"].as_array().unwrap().len(), 3);
    for (index, (id, kind, label)) in COLUMNS.iter().enumerate() {
        assert_eq!((&parsed["cols"][index]["id"], &parsed["cols"][index]["label"], &parsed["cols"][index]["type"]), (&serde_json::json!(id), &serde_json::json!(label), &serde_json::json!(kind)));
    }
    for (row, expected) in test_rows().as_array().unwrap().iter().enumerate() {
        for (column, value) in expected.as_array().unwrap().iter().enumerate() {
            let cell = &parsed["rows"][row]["c"][column]["v"];
            assert!(cell == value || cell.as_f64() == value.as_f64() && value.is_number(), "{cell} {value}");
        }
    }
}

#[test]
fn to_json_with_custom_properties() {
    let mut table = Table::default();
    table.prop("key1", "value1");
    table.prop("key2", "value2");
    let parsed: serde_json::Value = serde_json::from_str(&table.json()).unwrap();
    assert_eq!(parsed["p"], serde_json::json!({"key1": "value1", "key2": "value2"}));
}

#[test]
fn add_number_cell() {
    assert_eq!(row_json(Cell::Number(123.45)), serde_json::json!({"v": 123.45}));
}

#[test]
fn add_text_cell() {
    assert_eq!(row_json(Cell::Text("test_string".into())), serde_json::json!({"v": "test_string"}));
}

#[test]
fn add_boolean_cell() {
    assert_eq!(row_json(Cell::Boolean(true)), serde_json::json!({"v": true}));
}
