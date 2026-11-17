//! Generates the protocol types from `proto/dspk_stream.proto` (sessions with a server) and
//! `proto/dspk_live.proto` (devices; it uses the first one's types, generated once).

const PROTO_DIR: &str = "proto";
const STREAM_PROTO: &str = "proto/dspk_stream.proto";
const LIVE_PROTO: &str = "proto/dspk_live.proto";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed={STREAM_PROTO}");
    println!("cargo:rerun-if-changed={LIVE_PROTO}");
    let stream = protox::compile([STREAM_PROTO], [PROTO_DIR])?;
    prost_build::Config::new().compile_fds(stream)?;
    // The live package refers to dspk_stream.v1 types: point them at the module already generated
    // (`protocol::wire`) instead of generating them a second time.
    let live = protox::compile([LIVE_PROTO], [PROTO_DIR])?;
    prost_build::Config::new()
        .extern_path(".dspk_stream.v1", "crate::protocol::wire")
        .compile_fds(live)?;
    Ok(())
}
