use dsp_synapse::probe::{neuropixels_1_0, neuropixels_2_0, tetrode, utah_array};

pub fn run_probe(model: &str) {
    println!("=== Electrode Probe Inspector ===");
    let probe = match model {
        "neuropixels-1" | "np1" => neuropixels_1_0(),
        "neuropixels-2" | "np2" => neuropixels_2_0(),
        "tetrode" => tetrode(),
        "utah" | "utah-array" => utah_array(),
        other => {
            println!("Unknown probe model: '{}'. Supported models: neuropixels-1, neuropixels-2, tetrode, utah", other);
            return;
        }
    };

    println!("Probe Name:        {}", probe.name);
    println!("Total Channels:    {}", probe.total_channels());
    println!("Active Channels:   {}", probe.active_channels());

    if let Ok(c0) = probe.get_contact(0) {
        println!(
            "Contact 0 Position:   ({:.1} um, {:.1} um, {:.1} um) [shank {}]",
            c0.position.x_um, c0.position.y_um, c0.position.z_um, c0.shank_id
        );
    }
    let last_idx = probe.total_channels().saturating_sub(1);
    if let Ok(c_last) = probe.get_contact(last_idx) {
        println!(
            "Contact {} Position: ({:.1} um, {:.1} um, {:.1} um) [shank {}]",
            last_idx, c_last.position.x_um, c_last.position.y_um, c_last.position.z_um, c_last.shank_id
        );
    }
}
