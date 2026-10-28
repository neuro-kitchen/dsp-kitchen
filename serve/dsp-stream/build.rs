//! Generates the protocol types from `proto/dsp_stream.proto`.

const PROTO_DIR: &str = "proto";
const PROTO_FILE: &str = "proto/dsp_stream.proto";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed={PROTO_FILE}");
    let descriptors = protox::compile([PROTO_FILE], [PROTO_DIR])?;
    prost_build::Config::new().compile_fds(descriptors)?;
    Ok(())
}
