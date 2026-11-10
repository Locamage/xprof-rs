use crate::cli::client::{Client, Params};
use crate::cli::json::J;
use crate::cli::{Args, Error, Out};
use std::cell::RefCell;
use std::path::{Path, PathBuf};

type Fetch = Box<dyn Fn(&str, &Params) -> Result<Option<Vec<u8>>, Error>>;
type Calls = RefCell<Vec<(String, Vec<(String, String)>)>>;

pub struct Fake {
    pub fetch: Fetch,
    pub hosts: Option<Result<Vec<String>, Error>>,
    pub dir: PathBuf,
    pub logdir: Option<PathBuf>,
    pub calls: Calls,
}

impl Fake {
    pub fn new(fetch: impl Fn(&str, &Params) -> Result<Option<Vec<u8>>, Error> + 'static) -> Self {
        Self { fetch: Box::new(fetch), hosts: None, dir: PathBuf::new(), logdir: None, calls: RefCell::default() }
    }

    pub fn tools(answer: impl Fn(&str) -> Option<String> + 'static) -> Self {
        Self::new(move |tool, _| Ok(answer(tool).map(String::into_bytes)))
    }

    pub fn fixed(data: &str) -> Self {
        let data = data.to_string();
        Self::tools(move |_| Some(data.clone()))
    }

    pub fn with_hosts(mut self, hosts: &[&str]) -> Self {
        self.hosts = Some(Ok(hosts.iter().map(std::string::ToString::to_string).collect()));
        self
    }

    pub fn failing_hosts(mut self, error: Error) -> Self {
        self.hosts = Some(Err(error));
        self
    }

    pub fn in_dir(mut self, dir: &Path) -> Self {
        self.dir = dir.to_path_buf();
        self
    }

    pub fn fetched(&self) -> Vec<String> {
        self.calls.borrow().iter().map(|(tool, _)| tool.clone()).collect()
    }
}

impl Client for Fake {
    fn fetch(&self, tool: &str, _: &str, params: &Params) -> Result<Option<Vec<u8>>, Error> {
        self.calls.borrow_mut().push((tool.to_string(), params.iter().map(|(key, value)| (key.to_string(), value.clone())).collect()));
        (self.fetch)(tool, params)
    }

    fn run_dir(&self, _: &str) -> Result<PathBuf, Error> {
        Ok(self.dir.clone())
    }

    fn logdir(&self) -> Option<&Path> {
        self.logdir.as_deref()
    }

    fn barrier_durations(&self, session: &str) -> Option<Vec<f64>> {
        let mut params = vec![("trace_filter_config", r#"{"device_regexes": ["TPU:0", "GPU:0"]}"#.to_string())];
        params.extend(self.hosts(session).ok()?.first().map(|host| ("hosts", host.clone())));
        let trace = serde_json::from_slice::<serde_json::Value>(&self.fetch("trace_viewer.json", session, &params).ok().flatten()?).ok()?;
        let events = trace.get("traceEvents")?.as_array()?;
        Some(
            events
                .iter()
                .filter(|event| event.get("name").and_then(|name| name.as_str()) == Some("barrier-cores"))
                .map(|event| event.get("dur").and_then(serde_json::Value::as_f64).unwrap_or(0.0))
                .collect(),
        )
    }

    fn category_times(&self, session: &str, params: &Params) -> Option<[f64; 3]> {
        let profile = self.fetch_text("hlo_op_profile.json", session, params).ok().flatten().and_then(|raw| J::parse(&raw));
        let times = profile
            .as_ref()
            .map(|profile| crate::cli::ops::field(profile, "by_category"))
            .filter(|node| matches!(node, J::Map(_)) && node.truthy())
            .map(hlo_times);
        crate::release(profile);
        times
    }

    fn roofline_properties(&self, session: &str, params: &Params) -> Result<Option<String>, Error> {
        self.fetch_text("roofline_model.json", session, params)
    }

    fn roofline_total(&self, session: &str) -> Option<String> {
        self.fetch_text("roofline_model.json", session, &[]).ok()?.or_else(|| self.fetch_text("roofline_model", session, &[]).ok()?)
    }

    fn op_profile(&self, session: &str, params: &Params) -> Result<Option<J>, Error> {
        let data = self.fetch_either("op_profile", "hlo_op_profile.json", session, params)?;
        data.map(|data| {
            J::parse(&data)
                .ok_or_else(|| Error::new(crate::cli::Kind::Value, "Failed to parse op_profile proto: ParseError('Failed to load JSON')"))
        })
        .transpose()
    }

    fn hosts(&self, session: &str) -> Result<Vec<String>, Error> {
        if let Some(hosts) = &self.hosts {
            hosts.clone()
        } else {
            let paths = self.xspace_paths(&self.run_dir(session)?)?;
            Ok(paths.iter().map(|path| crate::cli::client::host(path)).collect())
        }
    }
}

pub fn args(session: &str, pairs: &[(&str, J)]) -> Args {
    let mut values = vec![("session_id".to_string(), J::from(session))];
    values.extend(pairs.iter().map(|(key, value)| (key.to_string(), value.clone())));
    Args { values }
}

pub fn text(out: Result<Out, Error>) -> String {
    match out.unwrap() {
        Out::Text(text) => text,
        other => panic!("expected text, got {other:?}"),
    }
}

pub fn json(out: Result<Out, Error>) -> J {
    match out.unwrap() {
        Out::Text(text) => J::parse(&text).unwrap_or_else(|| panic!("not JSON: {text}")),
        Out::Value(value) => value,
        other @ Out::Bytes(_) => panic!("expected JSON, got {other:?}"),
    }
}

pub fn parse(text: &str) -> J {
    J::parse(text).unwrap()
}

pub fn run(argv: &[&str]) -> (i32, String, String) {
    let argv: Vec<String> = argv.iter().map(std::string::ToString::to_string).collect();
    let (code, out, err) = crate::cli::execute(&argv).expect("a CLI command");
    (code, String::from_utf8_lossy(&out).into_owned(), err)
}

pub fn ok(argv: &[&str]) -> J {
    let (code, out, _) = run(argv);
    assert_eq!(code, 0, "{out}");
    parse(&out)
}

pub fn scratch(name: &str) -> PathBuf {
    crate::tests::scratch(&format!("cli-{name}"))
}

pub fn same(left: &J, right: &J) -> bool {
    match (left, right) {
        (J::Map(left), J::Map(right)) => {
            left.len() == right.len() && left.iter().all(|(key, value)| right.iter().any(|(other, item)| other == key && same(value, item)))
        }
        (J::List(left), J::List(right)) => left.len() == right.len() && left.iter().zip(right).all(|(left, right)| same(left, right)),
        (J::Int(_) | J::Float(_), J::Int(_) | J::Float(_)) => left.float() == right.float(),
        _ => left == right,
    }
}

/// The compute, HBM and ICI sums of the times of the leaves under `node`.
pub fn hlo_times(node: &J) -> [f64; 3] {
    let metrics = node.at("metrics");
    let children = node.at("children").items();
    let mut times = [0.0; 3];
    if children.is_empty() && metrics.at("occurrences").float().unwrap_or(0.0) > 0.0 {
        let category = node.at("xla").get("category").map_or(String::new(), J::text);
        times[crate::cli::steps::time_class(&node.at("name").text(), &category)] +=
            Some(metrics.at("rawTime")).filter(|value| value.truthy()).and_then(J::float).unwrap_or(0.0);
    }
    for child in children.iter().filter(|child| matches!(child, J::Map(_))) {
        let child = hlo_times(child);
        times = std::array::from_fn(|index| times[index] + child[index]);
    }
    times
}
