use crate::xplane::steps::Span;

fn timespan(begin: u64, duration: u64) -> Span {
    Span { begin, duration }
}

fn includes_time(span: Span, time: u64) -> bool {
    span.includes(timespan(time, 0))
}

#[test]
fn non_instant_span_includes_single_time_tests() {
    assert!(includes_time(timespan(10, 2), 12));
    assert!(includes_time(timespan(12, 1), 12));
}

#[test]
fn non_instant_span_includes_instant_span_tests() {
    assert!(timespan(10, 2).includes(timespan(10, 0)));
    assert!(timespan(10, 2).includes(timespan(12, 0)));
}

#[test]
fn non_instant_span_includes_non_instant_span_tests() {
    assert!(timespan(10, 5).includes(timespan(10, 4)));
    assert!(timespan(10, 5).includes(timespan(10, 5)));
    assert!(!timespan(10, 5).includes(timespan(10, 6)));
}

#[test]
fn instant_span_includes_single_time_tests() {
    assert!(includes_time(timespan(10, 0), 10));
    assert!(!includes_time(timespan(10, 0), 9));
}

#[test]
fn instant_span_includes_instant_span_tests() {
    assert!(timespan(10, 0).includes(timespan(10, 0)));
    assert!(!timespan(10, 0).includes(timespan(8, 0)));
}

#[test]
fn instant_span_includes_non_instant_span_tests() {
    assert!(!timespan(10, 0).includes(timespan(10, 1)));
    assert!(!timespan(12, 0).includes(timespan(9, 100)));
}
