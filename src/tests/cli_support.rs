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
    pub fn new(fetch: impl Fn(&str, &Params) -> Result<Option<Vec<u8>>, Error> + 'static) -> Fake {
        Fake { fetch: Box::new(fetch), hosts: None, dir: PathBuf::new(), logdir: None, calls: RefCell::default() }
    }

    pub fn tools(answer: impl Fn(&str) -> Option<String> + 'static) -> Fake {
        Fake::new(move |tool, _| Ok(answer(tool).map(String::into_bytes)))
    }

    pub fn fixed(data: &str) -> Fake {
        let data = data.to_string();
        Fake::tools(move |_| Some(data.clone()))
    }

    pub fn with_hosts(mut self, hosts: &[&str]) -> Fake {
        self.hosts = Some(Ok(hosts.iter().map(|host| host.to_string()).collect()));
        self
    }

    pub fn failing_hosts(mut self, error: Error) -> Fake {
        self.hosts = Some(Err(error));
        self
    }

    pub fn in_dir(mut self, dir: &Path) -> Fake {
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

    fn hosts(&self, session: &str) -> Result<Vec<String>, Error> {
        match &self.hosts {
            Some(hosts) => hosts.clone(),
            None => {
                let paths = self.xspace_paths(&self.run_dir(session)?)?;
                Ok(paths.iter().map(|path| crate::cli::client::host(path)).collect())
            }
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
        other => panic!("expected JSON, got {other:?}"),
    }
}

pub fn parse(text: &str) -> J {
    J::parse(text).unwrap()
}

pub fn run(argv: &[&str]) -> (i32, String, String) {
    let argv: Vec<String> = argv.iter().map(|argument| argument.to_string()).collect();
    let (code, out, err) = crate::cli::execute(&argv).expect("a CLI command");
    (code, String::from_utf8_lossy(&out).into_owned(), err)
}

pub fn scratch(name: &str) -> PathBuf {
    let dir = crate::tests::temp_dir().join(format!("xprof-rs-cli-{}-{name}", std::process::id()));
    _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

pub fn same(left: &J, right: &J) -> bool {
    match (left, right) {
        (J::Map(left), J::Map(right)) => left.len() == right.len() && left.iter().all(|(key, value)| right.iter().any(|(other, item)| other == key && same(value, item))),
        (J::List(left), J::List(right)) => left.len() == right.len() && left.iter().zip(right).all(|(left, right)| same(left, right)),
        (J::Int(_) | J::Float(_), J::Int(_) | J::Float(_)) => left.float() == right.float(),
        _ => left == right,
    }
}
