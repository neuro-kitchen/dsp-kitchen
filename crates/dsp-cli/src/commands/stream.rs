use dsp_core::time::SampleRate;
use dsp_stream::StreamPurpose;

pub fn run_stream(mode: &str, buckets: usize, channels: usize, sample_rate: f64) -> anyhow::Result<()> {
    println!("=== Stream Mode & Decimation Inspector ===");
    let sr = SampleRate::new(sample_rate)?;
    println!("Source Sampling Rate: {:.1} Hz", sr.rate_hz());
    println!("Channel Count:        {}", channels);

    let purpose = if mode == "processing" {
        StreamPurpose::Processing
    } else {
        StreamPurpose::Visualization {
            target_points_per_channel: buckets,
        }
    };

    match purpose {
        StreamPurpose::Processing => {
            let bytes_per_sec = channels as f64 * sample_rate * 4.0;
            println!("Mode: FULL-FIDELITY PROCESSING STREAM");
            println!("  Lossless:          YES");
            println!("  Rate:              {:.0} samples/sec/channel", sample_rate);
            println!("  Bandwidth ({}ch):   {:.2} MB/s (float32)", channels, bytes_per_sec / (1024.0 * 1024.0));
        }
        StreamPurpose::Visualization { target_points_per_channel } => {
            println!("Mode: DECIMATED VISUALIZATION STREAM (LOD)");
            println!("  Target Points:     {} points/channel/frame", target_points_per_channel);
            println!("  Decimation Kernel: Min-Max Envelope Preserving");
            println!("  Viewport Transfer: Minimized to display pixel density");
        }
    }

    Ok(())
}
