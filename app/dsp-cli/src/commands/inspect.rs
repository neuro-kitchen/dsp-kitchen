//! `inspect`: the launch geometries the selected runtime gives a `[channels, samples]` buffer.

use clap::Args;
use cubecl::prelude::*;
use dsp_core::compute::{ComputeTarget, ComputeTask, LaunchGeometry};

/// Default buffer: a Neuropixels probe for one second at 30 kHz.
const DEFAULT_CHANNELS: usize = 384;
const DEFAULT_SAMPLES: usize = 30_000;

#[derive(Args, Debug)]
pub struct InspectArgs {
    /// Channels of the buffer
    #[arg(short, long, default_value_t = DEFAULT_CHANNELS)]
    channels: usize,
    /// Samples per channel
    #[arg(short, long, default_value_t = DEFAULT_SAMPLES)]
    samples: usize,
}

struct Inspect {
    channels: usize,
    samples: usize,
}

impl ComputeTask for Inspect {
    type Output = ();
    fn run(self, client: Client) {
        let (channels, samples) = (self.channels, self.samples);
        let hw = &client.properties().hardware;
        println!("Runtime:          {}", client.name());
        println!("Max cube count:   {:?}", hw.max_cube_count);
        println!("Plane lanes:      {}", LaunchGeometry::plane_lanes(&client));
        println!("Buffer:           {channels} channels × {samples} samples\n");
        let show = |label: &str, g: LaunchGeometry| {
            println!("  {label:<36} CubeDim ({}, {}, {})  CubeCount {:?}", g.cube_dim.x, g.cube_dim.y, g.cube_dim.z, g.cube_count);
        };
        show("elementwise (scale, clamp)", LaunchGeometry::elementwise(&client, channels * samples));
        show("channels × samples (median, TKEO)", LaunchGeometry::channels_samples(&client, channels, samples));
        show("per sample (common reference)", LaunchGeometry::per_sample(&client, samples));
        show("per channel (IIR, detection)", LaunchGeometry::per_channel(&client, channels));
        show("per row (reductions, one per channel)", LaunchGeometry::per_row(&client, channels, samples));
    }
}

pub fn run(target: ComputeTarget, args: &InspectArgs) -> anyhow::Result<()> {
    let compiled: Vec<&str> = ComputeTarget::available().iter().map(|t| t.name()).collect();
    println!("Compiled-in runtimes: {}", compiled.join(", "));
    target.run(Inspect { channels: args.channels, samples: args.samples })?;
    Ok(())
}
