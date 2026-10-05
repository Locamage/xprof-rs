use crate::tools::hlo_stats::source_text;
use crate::tools::opstats::{Metrics, Source};

fn source_info_formatted_text(file_name: &str, line_number: i32, stack_frame: &str) -> String {
    source_text(&Metrics { source: Some(Source { file: file_name.into(), line: line_number, stack: stack_frame.into() }), ..Default::default() })
}

#[test]
fn valid_source_info() {
    assert_eq!(source_info_formatted_text("foo.cc", 42, "frame1\nframe2"), "<div class='source-info-cell' title='frame1\nframe2'>foo.cc:42</div>");
}

#[test]
fn empty_source_info() {
    assert_eq!(source_info_formatted_text("", 0, ""), "");
}

#[test]
fn source_info_missing_line() {
    assert_eq!(source_info_formatted_text("foo.cc", -1, ""), "");
}

#[test]
fn stack_frame_with_html() {
    assert_eq!(source_info_formatted_text("foo.cc", 42, "<embedded module '_launcher'>"), "<div class='source-info-cell' title='&lt;embedded module &#39;_launcher&#39;&gt;'>foo.cc:42</div>");
}

#[test]
fn file_name_with_html() {
    assert_eq!(source_info_formatted_text("a<b>c.cc", 42, ""), "<div class='source-info-cell' title=''>a&lt;b&gt;c.cc:42</div>");
}
