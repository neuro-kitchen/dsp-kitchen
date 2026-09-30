use cubecl::prelude::*;
use dsp_base::{ComputeTarget, ComputeTask, LaunchGeometry};

struct Inspect {
    channels: usize,
    samples: usize,
}

impl ComputeTask for Inspect {
    type Output = ();
    fn run<R: Runtime>(self, client: ComputeClient<R>) {
        let (channels, samples) = (self.channels, self.samples);
        let hw = &client.properties().hardware;
        println!("Runtime:                 {}", R::name(&client));
        println!("Max cube count:          {:?}", hw.max_cube_count);
        println!("Simulated matrix:        {channels} channels × {samples} samples");

        let show = |label: &str, g: LaunchGeometry| {
            println!(
                "  {label:<34} CubeDim ({}, {}, {})  CubeCount {:?}",
                g.cube_dim.x, g.cube_dim.y, g.cube_dim.z, g.cube_count
            );
        };
        println!("\n[Launch geometries]");
        show("elementwise (scale / clamp)", LaunchGeometry::elementwise(&client, channels * samples));
        show("channels × samples (median / TKEO)", LaunchGeometry::channels_samples(&client, channels, samples));
        show("per sample (CAR)", LaunchGeometry::per_sample(&client, samples));
        show("per channel (IIR / threshold)", LaunchGeometry::per_channel(channels));
    }
}

pub fn run_inspect(target: ComputeTarget, channels: usize, samples: usize) -> anyhow::Result<()> {
    println!("=== Launch Geometry Inspector ===");
    let available: Vec<_> = ComputeTarget::available().iter().map(|t| t.name()).collect();
    println!("Compiled-in runtimes:    {}", available.join(", "));
    target.run(Inspect { channels, samples })?;
    Ok(())
}
