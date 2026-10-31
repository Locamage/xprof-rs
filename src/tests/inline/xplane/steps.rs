use super::*;
use crate::xplane::gpu::tests::fixture::{Stat, event, line, plane};

#[test]
fn gpu_step_events_classify_correlated_kernels_inside_derived_step_markers() {
    let stats = ["correlation_id", "group_id", "tensor_shapes"];
    let kernel = |meta: u64, offset: u64, group: u64| {
        event(
            meta,
            offset,
            10,
            &[(1, Stat::Int(1)), (2, Stat::Int(group))],
        )
    };
    let lines = [
        line(
            1,
            "Stream #1(Compute)",
            &[
                kernel(1, 0, 0),
                kernel(2, 20, 0),
                kernel(3, 40, 0),
                event(1, 60, 10, &[(2, Stat::Int(1))]),
                kernel(4, 100, 2),
            ],
        ),
        line(
            2,
            "Stream #2(MemcpyH2D)",
            &[
                kernel(5, 5, 0),
                event(
                    1,
                    80,
                    10,
                    &[
                        (1, Stat::Int(1)),
                        (2, Stat::Int(1)),
                        (3, Stat::Text("(half[2])")),
                    ],
                ),
            ],
        ),
    ];
    let map = plane(
        "/device:GPU:3",
        &lines,
        &[
            "fusion",
            "ncclKernel",
            "MemcpyDToD",
            "hgemm_fp16",
            "MemcpyHToD",
        ],
        &stats,
    );
    let planes = crate::xplane::parse(&map).unwrap();
    let events = gpu_device(&planes[0], &map, 0);
    let mut steps: Vec<i64> = events.keys().copied().collect();
    steps.sort_unstable();
    assert_eq!(steps, vec![0, 1, 2]);
    let kinds = |step: i64| {
        events[&step]
            .events
            .iter()
            .map(|(kind, span)| (*kind, span.begin))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        kinds(0),
        vec![
            (DEVICE_COMPUTE_32, 1_000_000),
            (DEVICE_COLLECTIVES, 1_000_020),
            (DEVICE_TO_DEVICE, 1_000_040),
            (HOST_TO_DEVICE, 1_000_005)
        ]
    );
    assert_eq!(kinds(1), vec![(DEVICE_COMPUTE_16, 1_000_080)]);
    assert_eq!(kinds(2), vec![(DEVICE_COMPUTE_16, 1_000_100)]);
    assert_eq!(
        events[&0]
            .markers
            .iter()
            .map(|marker| (marker.core, marker.span))
            .collect::<Vec<_>>(),
        vec![(
            Some(0),
            Span {
                begin: 1_000_000,
                duration: 50
            }
        )]
    );
    assert_eq!(
        events[&1]
            .markers
            .iter()
            .map(|marker| marker.span)
            .collect::<Vec<_>>(),
        vec![Span {
            begin: 1_000_060,
            duration: 30
        }]
    );
    assert_eq!(events[&0].collectives[&0], vec![(0, 1_000_020, 30)]);
}

#[test]
fn joined_intervals_keep_the_merged_active_time() {
    let mut state = 1u64;
    let mut next = |limit: u64| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % limit
    };
    for _ in 0..200 {
        let streams: Vec<Vec<(u64, u64)>> = (0..3)
            .map(|_| {
                let mut begin = next(20);
                (0..next(40))
                    .map(|_| {
                        begin = if next(8) == 0 {
                            next(200)
                        } else {
                            begin + next(6)
                        };
                        (
                            begin,
                            if next(10) == 0 {
                                begin.wrapping_sub(next(5) + 1)
                            } else {
                                begin + next(12)
                            },
                        )
                    })
                    .collect()
            })
            .collect();
        let joined = streams.iter().flat_map(|stream| {
            let mut intervals = Vec::new();
            for &(begin, end) in stream {
                add_interval(&mut intervals, begin, end);
            }
            intervals
        });
        assert_eq!(
            merged_active(joined.collect()),
            merged_active(streams.concat())
        );
    }
}
