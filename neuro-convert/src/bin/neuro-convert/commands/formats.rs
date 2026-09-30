pub fn run() -> anyhow::Result<()> {
    println!("Input formats:");
    for f in neuro_convert::input_formats() {
        println!("  {:10} {}", f.name(), f.description());
        for v in f.versions() {
            println!("  {:10}   - {v}", "");
        }
    }
    println!("\nOutput formats:");
    println!("  nwb        Neurodata Without Borders {} (Zarr v3 store, hdmf-zarr layout)", neuro_convert::outputs::nwb::schema::NWB_VERSION);
    Ok(())
}
