use crate::hlo::Module;
use crate::hlo::xla::HloProto;
use prost::Message;
use prost_reflect::{DescriptorPool, DynamicMessage};
use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicUsize, Ordering};

static POOL: LazyLock<DescriptorPool> = LazyLock::new(|| {
    let mut pool = DescriptorPool::global();
    pool.decode_file_descriptor_set(&include_bytes!("../hlo_descriptors.pb")[..]).unwrap();
    pool
});
static DIRS: AtomicUsize = AtomicUsize::new(0);

pub fn hlo_proto(text: &str) -> HloProto {
    let message = DynamicMessage::parse_text_format(POOL.get_message_by_name("xla.HloProto").unwrap(), text).unwrap();
    HloProto::decode(&message.encode_to_vec()[..]).unwrap()
}

pub fn module(proto: &HloProto) -> Module<'static> {
    Module::parse(Cow::Owned(proto.encode_to_vec()))
}

pub fn session_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xprof-rs-{name}-{}-{}", std::process::id(), DIRS.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

pub fn write_module(dir: &std::path::Path, name: &str, proto: &HloProto) {
    std::fs::write(dir.join(format!("{name}.hlo_proto.pb")), proto.encode_to_vec()).unwrap();
}
