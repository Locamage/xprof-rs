fn main() {
    println!("cargo:rerun-if-changed=src/data/hlo_descriptors.pb");
    let mut pool = prost_reflect::DescriptorPool::global();
    pool.decode_file_descriptor_set(&include_bytes!("src/data/hlo_descriptors.pb")[..]).unwrap();
    let mut config = prost_build::Config::new();
    for field in pool.get_message_by_name("xla.HloInstructionProto").unwrap().fields().filter(|field| field.kind().as_message().is_some() && !field.is_list()) {
        config.boxed(field.full_name());
    }
    let mut files: Vec<_> = pool.file_descriptor_protos().cloned().collect();
    let hlo = files.iter_mut().find(|file| file.name() == "xla/service/hlo.proto").unwrap();
    let instruction = hlo.message_type.iter().find(|message| message.name() == "HloInstructionProto").unwrap().clone();
    for (name, numbers) in [("HloInstructionLite", &[1, 2, 3, 7, 9, 35, 36, 37, 38][..]), ("HloInstructionMeta", &[1, 7][..])] {
        let mut lite = instruction.clone();
        lite.name = Some(name.into());
        lite.field.retain(|field| numbers.contains(&field.number()));
        lite.oneof_decl.clear();
        hlo.message_type.push(lite);
    }
    config.compile_fds(prost_types::FileDescriptorSet { file: files }).unwrap();
}
