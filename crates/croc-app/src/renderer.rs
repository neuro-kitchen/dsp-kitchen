use slint::{Rgba8Pixel, SharedPixelBuffer};

/// Palette for multi-channel visualization (vibrant, modern dark-theme colors)
pub const CHANNEL_COLORS: [Rgba8Pixel; 8] = [
    Rgba8Pixel { r: 56,  g: 189, b: 248, a: 255 }, // Cyan #38bdf8
    Rgba8Pixel { r: 52,  g: 211, b: 153, a: 255 }, // Emerald #34d399
    Rgba8Pixel { r: 168, g: 85,  b: 247, a: 255 }, // Purple #a855f7
    Rgba8Pixel { r: 251, g: 191, b: 36,  a: 255 }, // Amber #fbbf24
    Rgba8Pixel { r: 244, g: 63,  b: 94,  a: 255 }, // Rose #f43f5e
    Rgba8Pixel { r: 96,  g: 165, b: 250, a: 255 }, // Blue #60a5fa
    Rgba8Pixel { r: 249, g: 115, b: 22,  a: 255 }, // Orange #f97316
    Rgba8Pixel { r: 45,  g: 212, b: 191, a: 255 }, // Teal #2dd4bf
];

const BG_COLOR: Rgba8Pixel = Rgba8Pixel { r: 9,  g: 13, b: 19, a: 255 }; // Deep dark #090d13
const GRID_COLOR: Rgba8Pixel = Rgba8Pixel { r: 22, g: 27, b: 34, a: 255 }; // Grid line #161b22
const BASELINE_COLOR: Rgba8Pixel = Rgba8Pixel { r: 33, g: 38, b: 45, a: 255 }; // Baseline #21262d
const TEXT_COLOR: Rgba8Pixel = Rgba8Pixel { r: 139, g: 148, b: 158, a: 255 }; // Gray text #8b949e
const SPIKE_MARKER_COLOR: Rgba8Pixel = Rgba8Pixel { r: 250, g: 204, b: 21, a: 255 }; // Gold #facc15

/// Configuration parameters for waveform rendering.
pub struct RenderConfig<'a> {
    pub raw_data: &'a [f32],
    pub total_samples: usize,
    pub total_channels: usize,
    pub channel_offset: usize,
    pub visible_channels: usize,
    pub window_start_sec: f64,
    pub visible_window_sec: f64,
    pub sample_rate: f64,
    pub amplitude_scale: f32,
    pub spike_times: &'a [f64],
    pub spike_channels: &'a [usize],
}

/// Renders multi-channel signals into a Slint SharedPixelBuffer using screen-space Min-Max LOD decimation.
pub fn render_waveforms_lod(
    width: u32,
    height: u32,
    config: &RenderConfig,
) -> SharedPixelBuffer<Rgba8Pixel> {
    let width = width.max(100);
    let height = height.max(100);
    let mut pixel_buffer = SharedPixelBuffer::new(width, height);
    let pixels = pixel_buffer.make_mut_bytes();

    // 1. Fill background (4 bytes per RGBA pixel)
    for chunk in pixels.chunks_exact_mut(4) {
        chunk[0] = BG_COLOR.r;
        chunk[1] = BG_COLOR.g;
        chunk[2] = BG_COLOR.b;
        chunk[3] = BG_COLOR.a;
    }

    let left_margin = 70usize;
    let right_margin = 20usize;
    let top_margin = 20usize;
    let bottom_margin = 30usize;

    if (width as usize) <= left_margin + right_margin || (height as usize) <= top_margin + bottom_margin {
        return pixel_buffer;
    }

    let plot_w = (width as usize) - left_margin - right_margin;
    let plot_h = (height as usize) - top_margin - bottom_margin;

    let num_channels = config.visible_channels.min(config.total_channels.saturating_sub(config.channel_offset));
    if num_channels == 0 || config.total_samples == 0 {
        return pixel_buffer;
    }

    let lane_h = plot_h as f32 / num_channels as f32;

    // Helper closure to set a pixel safely
    let w_usize = width as usize;
    let h_usize = height as usize;
    let mut set_pixel = |x: usize, y: usize, color: Rgba8Pixel| {
        if x < w_usize && y < h_usize {
            let idx = (y * w_usize + x) * 4;
            pixels[idx] = color.r;
            pixels[idx + 1] = color.g;
            pixels[idx + 2] = color.b;
            pixels[idx + 3] = color.a;
        }
    };

    // 2. Draw vertical time grid lines
    let num_grid_cols = 10;
    for g in 0..=num_grid_cols {
        let gx = left_margin + (g * plot_w) / num_grid_cols;
        for y in top_margin..(top_margin + plot_h) {
            set_pixel(gx, y, GRID_COLOR);
        }
    }

    // 3. Compute visible sample range
    let start_sample_f = (config.window_start_sec * config.sample_rate).round().max(0.0);
    let end_sample_f = ((config.window_start_sec + config.visible_window_sec) * config.sample_rate).round().min(config.total_samples as f64);

    let start_sample = (start_sample_f as usize).min(config.total_samples);
    let end_sample = (end_sample_f as usize).min(config.total_samples);
    let window_samples = end_sample.saturating_sub(start_sample);

    // 4. Render each channel
    for ch_idx in 0..num_channels {
        let actual_ch = config.channel_offset + ch_idx;
        let color = CHANNEL_COLORS[ch_idx % CHANNEL_COLORS.len()];
        let lane_center_y = top_margin as f32 + (ch_idx as f32 + 0.5) * lane_h;
        let channel_data_offset = actual_ch * config.total_samples;

        // Draw horizontal baseline
        let baseline_y = lane_center_y.round() as usize;
        for x in left_margin..(left_margin + plot_w) {
            if x % 4 != 0 {
                set_pixel(x, baseline_y, BASELINE_COLOR);
            }
        }

        // Draw simple channel label indicator (e.g. 5x7 dot pattern for "C" and channel number)
        draw_channel_marker(&mut set_pixel, 12, baseline_y, actual_ch, color);

        if window_samples == 0 {
            continue;
        }

        // Screen-space Min-Max Decimation across plot_w columns
        let mut prev_y_min = baseline_y;
        let mut prev_y_max = baseline_y;

        // Typical extracellular voltage scale (approx +/- 80 uV maps to 80% of lane half-height)
        let nominal_scale = 80.0f32;
        let half_lane = lane_h * 0.42;

        for col in 0..plot_w {
            let px = left_margin + col;

            // Map pixel column to continuous sample range in data
            let s0 = start_sample + ((col as f64 / plot_w as f64) * window_samples as f64) as usize;
            let s1 = (start_sample + (((col + 1) as f64 / plot_w as f64) * window_samples as f64) as usize)
                .min(config.total_samples)
                .max(s0 + 1);

            // Compute min and max within this column bucket
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;

            let slice = &config.raw_data[channel_data_offset + s0..channel_data_offset + s1];
            for &v in slice {
                if v < min_val { min_val = v; }
                if v > max_val { max_val = v; }
            }

            if min_val.is_infinite() {
                min_val = 0.0;
                max_val = 0.0;
            }

            // Apply amplitude scaling and map to screen pixels
            let scaled_min = (min_val * config.amplitude_scale) / nominal_scale;
            let scaled_max = (max_val * config.amplitude_scale) / nominal_scale;

            // Invert Y for screen coords: positive voltage points UP
            let y0 = (lane_center_y - scaled_max * half_lane).clamp(top_margin as f32, (top_margin + plot_h - 1) as f32).round() as usize;
            let y1 = (lane_center_y - scaled_min * half_lane).clamp(top_margin as f32, (top_margin + plot_h - 1) as f32).round() as usize;

            let cur_y_min = y0.min(y1);
            let cur_y_max = y0.max(y1);

            // Connect vertical span with previous column to eliminate aliasing gaps
            let draw_min = if col == 0 { cur_y_min } else { cur_y_min.min(prev_y_max) };
            let draw_max = if col == 0 { cur_y_max } else { cur_y_max.max(prev_y_min) };

            for y in draw_min..=draw_max {
                set_pixel(px, y, color);
            }

            prev_y_min = cur_y_min;
            prev_y_max = cur_y_max;
        }
    }

    // 5. Draw Spike Events / Detected Action Potential Markers
    let win_start = config.window_start_sec;
    let win_end = config.window_start_sec + config.visible_window_sec;

    for (&st, &sch) in config.spike_times.iter().zip(config.spike_channels.iter()) {
        if st >= win_start && st <= win_end {
            if sch >= config.channel_offset && sch < config.channel_offset + num_channels {
                let local_ch = sch - config.channel_offset;
                let lane_center_y = top_margin as f32 + (local_ch as f32 + 0.5) * lane_h;
                let col_ratio = (st - win_start) / config.visible_window_sec;
                let sx = left_margin + ((col_ratio * plot_w as f64).round() as usize).min(plot_w - 1);
                let sy = (lane_center_y - lane_h * 0.40).round() as usize;

                // Draw diamond / pip marker (3x3 pixels)
                set_pixel(sx, sy, SPIKE_MARKER_COLOR);
                set_pixel(sx - 1, sy, SPIKE_MARKER_COLOR);
                set_pixel(sx + 1, sy, SPIKE_MARKER_COLOR);
                set_pixel(sx, sy - 1, SPIKE_MARKER_COLOR);
                set_pixel(sx, sy + 1, SPIKE_MARKER_COLOR);
            }
        }
    }

    // 6. Draw Bottom Time Axis Label & Scale Bar
    let scale_y = top_margin + plot_h + 12;
    let scale_x = width as usize - right_margin - 80;
    // Draw 10ms scale bar
    for x in scale_x..(scale_x + 60) {
        set_pixel(x, scale_y, TEXT_COLOR);
    }
    set_pixel(scale_x, scale_y - 3, TEXT_COLOR);
    set_pixel(scale_x, scale_y - 2, TEXT_COLOR);
    set_pixel(scale_x, scale_y - 1, TEXT_COLOR);
    set_pixel(scale_x + 60, scale_y - 3, TEXT_COLOR);
    set_pixel(scale_x + 60, scale_y - 2, TEXT_COLOR);
    set_pixel(scale_x + 60, scale_y - 1, TEXT_COLOR);

    pixel_buffer
}

/// Draws a stylized channel indicator on the left margin.
fn draw_channel_marker<F>(set_pixel: &mut F, x: usize, y: usize, ch: usize, color: Rgba8Pixel)
where
    F: FnMut(usize, usize, Rgba8Pixel),
{
    // Draw colored channel dot
    for dy in 0..4 {
        for dx in 0..4 {
            set_pixel(x + dx, y - 2 + dy, color);
        }
    }

    // Draw simple digit / index indicator bars (binary-coded / notch tally for readability)
    let bar_x = x + 10;
    let units = ch % 10;
    let tens = ch / 10;

    for i in 0..tens.min(3) {
        for dy in 0..6 {
            set_pixel(bar_x + i * 3, y - 3 + dy, TEXT_COLOR);
        }
    }

    for i in 0..units.min(9) {
        for dy in 0..3 {
            set_pixel(bar_x + 12 + i * 2, y - 1 + dy, TEXT_COLOR);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_waveforms_lod_basic() {
        let channels = 8;
        let samples = 10000;
        let data = vec![10.0f32; channels * samples];
        let spikes = vec![0.050, 0.120];
        let spike_channels = vec![0, 1];

        let config = RenderConfig {
            raw_data: &data,
            total_samples: samples,
            total_channels: channels,
            channel_offset: 0,
            visible_channels: 4,
            window_start_sec: 0.0,
            visible_window_sec: 0.200,
            sample_rate: 30000.0,
            amplitude_scale: 1.0,
            spike_times: &spikes,
            spike_channels: &spike_channels,
        };

        let buf = render_waveforms_lod(800, 400, &config);
        assert_eq!(buf.width(), 800);
        assert_eq!(buf.height(), 400);
    }

    #[test]
    fn test_render_waveforms_lod_edge_cases() {
        // Zero samples
        let empty_data = Vec::new();
        let config = RenderConfig {
            raw_data: &empty_data,
            total_samples: 0,
            total_channels: 0,
            channel_offset: 0,
            visible_channels: 0,
            window_start_sec: 0.0,
            visible_window_sec: 0.100,
            sample_rate: 30000.0,
            amplitude_scale: 1.0,
            spike_times: &[],
            spike_channels: &[],
        };

        let buf = render_waveforms_lod(400, 300, &config);
        assert_eq!(buf.width(), 400);
        assert_eq!(buf.height(), 300);
    }
}

