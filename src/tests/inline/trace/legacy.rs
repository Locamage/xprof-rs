use super::*;
use crate::xplane::Meta;

fn plane(lines: &[Vec<u64>]) -> Plane {
    let event = |(index, &ts): (usize, &u64)| Ev {
        ts,
        dur: 0,
        group: 0,
        raw: (0, 0),
        meta: (index % 7 == 0) as u32,
        eager: None,
        has_stats: false,
        linked: false,
    };
    let lines = lines
        .iter()
        .map(|times| Line {
            events: times.iter().enumerate().map(event).collect(),
            ..Default::default()
        })
        .collect();
    let (mut plane, mut internal) = (Plane::default(), Meta::default());
    internal.internal = true;
    (plane.lines, plane.meta) = (lines, vec![Meta::default(), internal]);
    plane
}

fn check(plane: &Plane) {
    let lines: Vec<_> = plane.lines.iter().map(|line| (plane, line)).collect();
    let values = || {
        lines.par_iter().flat_map_iter(|&(plane, line)| {
            line.events
                .iter()
                .filter(move |event| kept(plane, line, event))
                .map(move |event| time(plane, event))
        })
    };
    for limit in [
        1,
        2,
        1000,
        SAMPLES / 2,
        SAMPLES,
        2 * SAMPLES,
        3 * SAMPLES,
        4 * SAMPLES,
    ] {
        assert_eq!(cut(&lines, limit), select(values, limit), "limit {limit}");
    }
}

#[test]
fn the_sampled_cut_is_the_selected_cut() {
    let mut state = 1u64;
    let mut next = |range: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % range
    };
    let lines: Vec<Vec<u64>> = (0..5)
        .map(|line| {
            (0..(line + 1) * SAMPLES / 4)
                .map(|_| next(1 << (8 * line + 8)))
                .collect()
        })
        .collect();
    check(&plane(&lines));
}

#[test]
fn the_cut_is_exact_when_the_sample_misses_the_place_of_the_limit() {
    // Each sampled event is late, so the sample gives a range after the place of the limit.
    let late = |count: usize| {
        (0..count as u64)
            .map(|index| {
                if index % 3 == 0 {
                    1 << 40 | index
                } else {
                    index
                }
            })
            .collect()
    };
    check(&plane(&[late(3 * SAMPLES), late(SAMPLES / 2)]));
}
