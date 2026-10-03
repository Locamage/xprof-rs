use super::xspace::{V, XSpace, name};
use crate::xplane::{Links, Plane, slice};

const THREADPOOL_EVENT: i64 = 15;

type Link = Option<(u64, u64)>;

fn links_by_name(space: &XSpace) -> Vec<(String, Links)> {
    let (map, planes) = space.parsed();
    let plane = &planes[0];
    plane.lines.iter().flat_map(|line| &line.events).map(|event| (name(plane, event).to_string(), plane.links(event.meta, slice(&map, event.raw), None))).collect()
}

fn legacy_pair(producer: &[(&str, V)], consumer: &[(&str, V)]) -> (Link, Link) {
    let mut space = XSpace::default();
    let plane = space.add_plane();
    plane.event(0, "ExecutorState::Process", 100, 100, producer);
    plane.event(1, "TpuExecuteOp", 200, 100, consumer);
    let links = links_by_name(&space);
    (links[0].1.producer, links[1].1.consumer)
}

#[test]
fn is_root_stats_test() {
    let mut space = XSpace::default();
    let plane = space.add_plane();
    plane.event(0, "ProcessBatch", 100, 100, &[]);
    plane.event(0, "BatchingSessionRun", 200, 100, &[]);
    let roots: Vec<(String, Option<i64>)> = links_by_name(&space).into_iter().map(|(name, links)| (name, links.root)).collect();
    assert_eq!(roots, [("ProcessBatch".to_string(), Some(2)), ("BatchingSessionRun".to_string(), Some(1))]);
}

#[test]
fn producer_consumer_test() {
    let stats = [("id", V::from(123)), ("iter_num", 456.into())];
    let (producer, consumer) = legacy_pair(&stats, &stats);
    assert!(producer.is_some() && consumer.is_some());
    assert_eq!(producer, consumer);
}

#[test]
fn producer_consumer_not_matched_test() {
    let (producer, consumer) = legacy_pair(&[("id", 123.into()), ("iter_num", 456.into()), ("device_ordinal", 789.into())], &[("id", 123.into()), ("iter_num", 789.into())]);
    assert!(producer.is_some() && consumer.is_some());
    assert_ne!(producer, consumer);
}

#[test]
fn missing_legacy_stat_test() {
    let (producer, consumer) = legacy_pair(&[("id", 123.into())], &[("id", 123.into())]);
    assert_eq!((producer, consumer), (None, None));
}

fn regions(space: &XSpace) -> Vec<(u64, u64)> {
    let (map, mut planes) = space.parsed();
    planes.iter_mut().for_each(|plane| plane.add_threadpool_regions(&map));
    let plane: &Plane = &planes[0];
    plane.lines.iter().flat_map(|line| &line.events).filter(|event| name(plane, event) == "ThreadpoolListener::Region").map(|event| (event.ts, event.dur)).collect()
}

fn threadpool_space(regions: &[(i64, i64)]) -> XSpace {
    let mut space = XSpace::default();
    let plane = space.add_plane();
    plane.event(0, "ThreadpoolListener::Record", 100, 100, &[("_pt", THREADPOOL_EVENT.into()), ("_p", 123.into())]);
    let consumer = [("_ct", V::from(THREADPOOL_EVENT)), ("_c", 123.into())];
    for &(start, stop) in regions {
        plane.event(1, "ThreadpoolListener::StartRegion", start, 0, &consumer);
        plane.event(1, "ThreadpoolListener::StopRegion", stop, 0, &consumer);
    }
    space
}

#[test]
fn thread_pool_preprocessor_test() {
    assert_eq!(regions(&threadpool_space(&[(200, 300)])), [(200, 100)]);
}

#[test]
fn thread_pool_preprocessor_sorted_test() {
    let timestamps: Vec<u64> = regions(&threadpool_space(&[(400, 500), (200, 300)])).into_iter().map(|region| region.0).collect();
    assert_eq!(timestamps, [200, 400]);
}
