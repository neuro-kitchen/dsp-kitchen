//! Main Application ViewModel (Android MVVM style).
//!
//! Orchestrates domain models (`Dataset`, `TimelineState`, `SpikeEventStore`),
//! maintains UI presentation state, processes user intents from the View,
//! and produces updated view states and rendered buffers.

use std::time::Instant;
use slint::{Rgba8Pixel, SharedPixelBuffer};
use crate::model::{Dataset, SpikeEventStore, TimelineState};
use crate::view::{RenderConfig, WaveformRenderer};

pub struct AppViewModel {
    pub dataset: Dataset,
    pub timeline: TimelineState,
    pub events: SpikeEventStore,
    pub amplitude_scale: f32,
    pub channel_offset: usize,
    pub visible_channels: usize,
    pub canvas_width: u32,
    pub canvas_height: u32,
    last_frame_instant: Instant,
}

impl AppViewModel {
    pub fn new(dataset: Dataset) -> Self {
        let total_duration = dataset.total_duration_sec();
        let timeline = TimelineState::new(total_duration);
        let events = SpikeEventStore::detect_from_raw(
            &dataset.raw_data,
            dataset.total_channels,
            dataset.total_samples,
            dataset.sample_rate,
        );

        Self {
            dataset,
            timeline,
            events,
            amplitude_scale: 1.0,
            channel_offset: 0,
            visible_channels: 8,
            canvas_width: 1200,
            canvas_height: 550,
            last_frame_instant: Instant::now(),
        }
    }

    // ========================================================================
    // UI State Formatting Helpers
    // ========================================================================

    pub fn channel_label_range(&self) -> String {
        let end_ch = (self.channel_offset + self.visible_channels).min(self.dataset.total_channels);
        format!(
            "Ch {} - Ch {} (of {})",
            self.channel_offset,
            end_ch.saturating_sub(1),
            self.dataset.total_channels
        )
    }

    pub fn time_readout(&self) -> String {
        self.timeline.format_time_readout()
    }

    /// Renders the current viewport waveform buffer using screen-space LOD decimation.
    pub fn render_waveform_buffer(&self, width: u32, height: u32) -> SharedPixelBuffer<Rgba8Pixel> {
        let config = RenderConfig {
            raw_data: &self.dataset.raw_data,
            total_samples: self.dataset.total_samples,
            total_channels: self.dataset.total_channels,
            channel_offset: self.channel_offset,
            visible_channels: self.visible_channels,
            window_start_sec: self.timeline.window_start_sec,
            visible_window_sec: self.timeline.visible_window_sec,
            sample_rate: self.dataset.sample_rate,
            amplitude_scale: self.amplitude_scale,
            spike_times: &self.events.times_sec,
            spike_channels: &self.events.channels,
        };
        WaveformRenderer::render(width, height, &config)
    }

    // ========================================================================
    // User Intent Handlers
    // ========================================================================

    pub fn on_toggle_play(&mut self) {
        self.timeline.toggle_play();
        self.last_frame_instant = Instant::now();
    }

    pub fn on_step_forward(&mut self) {
        self.timeline.step_forward(0.033);
    }

    pub fn on_step_backward(&mut self) {
        self.timeline.step_backward(0.033);
    }

    pub fn on_jump_to_start(&mut self) {
        self.timeline.scrub_to(0.0);
    }

    pub fn on_jump_to_end(&mut self) {
        let dur = self.timeline.total_duration_sec;
        self.timeline.scrub_to(dur);
    }

    pub fn on_toggle_loop(&mut self) {
        self.timeline.toggle_loop();
    }

    pub fn on_set_playback_speed(&mut self, speed: f32) {
        self.timeline.playback_speed = speed as f64;
    }

    pub fn on_set_window_duration(&mut self, duration_sec: f32) {
        self.timeline.set_window_duration(duration_sec as f64);
    }

    pub fn on_scrub_to_ratio(&mut self, ratio: f32) {
        self.timeline.scrub_ratio(ratio as f64);
    }

    pub fn on_zoom_time(&mut self, factor: f32) {
        self.timeline.zoom_time(factor as f64);
    }

    pub fn on_pan_time(&mut self, dt_sec: f32) {
        self.timeline.pan_time(dt_sec as f64);
    }

    pub fn on_zoom_amplitude(&mut self, factor: f32) {
        self.amplitude_scale = (self.amplitude_scale * factor).clamp(0.1, 20.0);
    }

    pub fn on_next_channel_page(&mut self) {
        if self.channel_offset + self.visible_channels < self.dataset.total_channels {
            self.channel_offset += self.visible_channels;
        }
    }

    pub fn on_prev_channel_page(&mut self) {
        self.channel_offset = self.channel_offset.saturating_sub(self.visible_channels);
    }

    pub fn on_canvas_resized(&mut self, width: f32, height: f32) {
        self.canvas_width = width.max(100.0) as u32;
        self.canvas_height = height.max(100.0) as u32;
    }

    /// Periodic frame tick handler (called at ~60 Hz). Returns `true` if UI state needs refreshing.
    pub fn on_tick(&mut self) -> bool {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame_instant).as_secs_f64();
        self.last_frame_instant = now;

        if self.timeline.is_playing {
            self.timeline.advance(dt);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viewmodel_intents_and_pagination() {
        let ds = Dataset::generate_synthetic(16, 10_000.0, 2.0);
        let mut vm = AppViewModel::new(ds);

        assert_eq!(vm.channel_offset, 0);
        assert_eq!(vm.channel_label_range(), "Ch 0 - Ch 7 (of 16)");

        // Page down
        vm.on_next_channel_page();
        assert_eq!(vm.channel_offset, 8);
        assert_eq!(vm.channel_label_range(), "Ch 8 - Ch 15 (of 16)");

        // Page down again (should clamp at last page)
        vm.on_next_channel_page();
        assert_eq!(vm.channel_offset, 8);

        // Page up
        vm.on_prev_channel_page();
        assert_eq!(vm.channel_offset, 0);

        // Gain scaling
        vm.on_zoom_amplitude(2.0);
        assert!((vm.amplitude_scale - 2.0).abs() < 1e-5);

        // Scrubbing
        vm.on_scrub_to_ratio(0.5);
        assert!((vm.timeline.current_time_sec - 1.0).abs() < 1e-5);
    }
}
