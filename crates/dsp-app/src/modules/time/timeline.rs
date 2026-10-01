//! Timeline and playback state model.
//!
//! Provides playhead management, transport controls, loop semantics,
//! scrubbing, and visible time window clamping.

/// Rerun-style interactive timeline state controller.
#[derive(Debug, Clone)]
pub struct TimelineState {
    pub current_time_sec: f64,
    pub total_duration_sec: f64,
    pub visible_window_sec: f64,
    pub window_start_sec: f64,
    pub is_playing: bool,
    pub loop_playback: bool,
    pub playback_speed: f64,
}

impl TimelineState {
    pub fn new(total_duration_sec: f64) -> Self {
        let total_duration_sec = total_duration_sec.max(0.001);
        let visible_window_sec = 0.100f64.min(total_duration_sec);
        Self {
            current_time_sec: 0.0,
            total_duration_sec,
            visible_window_sec,
            window_start_sec: 0.0,
            is_playing: false,
            loop_playback: true,
            playback_speed: 1.0,
        }
    }

    /// Advances playback by a physical delta time in seconds.
    pub fn advance(&mut self, dt_sec: f64) {
        if !self.is_playing {
            return;
        }

        self.current_time_sec += dt_sec * self.playback_speed;

        if self.current_time_sec >= self.total_duration_sec {
            if self.loop_playback {
                self.current_time_sec = 0.0;
            } else {
                self.current_time_sec = self.total_duration_sec;
                self.is_playing = false;
            }
        }

        self.follow_playhead();
    }

    /// Sets playhead time directly (scrubbing), clamping to bounds.
    pub fn scrub_to(&mut self, time_sec: f64) {
        self.current_time_sec = time_sec.clamp(0.0, self.total_duration_sec);
        self.follow_playhead();
    }

    /// Scrubs to a normalized ratio [0.0, 1.0] along the timeline ruler.
    pub fn scrub_ratio(&mut self, ratio: f64) {
        let t = ratio.clamp(0.0, 1.0) * self.total_duration_sec;
        self.scrub_to(t);
    }

    /// Steps forward by a given step time (e.g. 1/30s).
    pub fn step_forward(&mut self, step_sec: f64) {
        self.current_time_sec = (self.current_time_sec + step_sec).min(self.total_duration_sec);
        self.follow_playhead();
    }

    /// Steps backward by a given step time.
    pub fn step_backward(&mut self, step_sec: f64) {
        self.current_time_sec = (self.current_time_sec - step_sec).max(0.0);
        self.follow_playhead();
    }

    /// Toggles play/pause state.
    pub fn toggle_play(&mut self) {
        self.is_playing = !self.is_playing;
    }

    /// Toggles loop mode.
    pub fn toggle_loop(&mut self) {
        self.loop_playback = !self.loop_playback;
    }

    /// Adjusts visible window duration (zoom in / out in time).
    pub fn set_window_duration(&mut self, dur_sec: f64) {
        self.visible_window_sec = dur_sec.clamp(0.005, self.total_duration_sec);
        self.follow_playhead();
    }

    /// Scales the visible window around an anchor point, keeping the time under
    /// `anchor_ratio` (0.0 = window start, 1.0 = window end) fixed on screen.
    pub fn zoom_at(&mut self, factor: f64, anchor_ratio: f64) {
        let ratio = anchor_ratio.clamp(0.0, 1.0);
        let anchor_time = self.window_start_sec + ratio * self.visible_window_sec;
        self.visible_window_sec =
            (self.visible_window_sec * factor).clamp(0.005, self.total_duration_sec);
        let max_start = (self.total_duration_sec - self.visible_window_sec).max(0.0);
        self.window_start_sec =
            (anchor_time - ratio * self.visible_window_sec).clamp(0.0, max_start);
    }

    /// Sets the visible window to `[start, end]` seconds (min 5 ms), clamped to the recording.
    pub fn set_window_range(&mut self, start_sec: f64, end_sec: f64) {
        self.visible_window_sec = (end_sec - start_sec).clamp(0.005, self.total_duration_sec);
        let max_start = (self.total_duration_sec - self.visible_window_sec).max(0.0);
        self.window_start_sec = start_sec.clamp(0.0, max_start);
    }

    /// Pans the visible time window by delta seconds without moving the playhead.
    pub fn pan_time(&mut self, dt_sec: f64) {
        let max_start = (self.total_duration_sec - self.visible_window_sec).max(0.0);
        self.window_start_sec = (self.window_start_sec + dt_sec).clamp(0.0, max_start);
    }

    /// Keeps the visible window centered on the playhead.
    fn follow_playhead(&mut self) {
        let half_win = self.visible_window_sec * 0.5;
        let max_start = (self.total_duration_sec - self.visible_window_sec).max(0.0);
        self.window_start_sec = (self.current_time_sec - half_win).clamp(0.0, max_start);
    }

    /// Formats current time readout as "MM:SS.mmm / MM:SS.mmm".
    pub fn format_time_readout(&self) -> String {
        format!(
            "{:02}:{:06.3} / {:02}:{:06.3}",
            (self.current_time_sec / 60.0).floor() as u32,
            self.current_time_sec % 60.0,
            (self.total_duration_sec / 60.0).floor() as u32,
            self.total_duration_sec % 60.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeline_playback_and_loop() {
        let mut tl = TimelineState::new(10.0);
        assert_eq!(tl.current_time_sec, 0.0);
        assert!(!tl.is_playing);

        tl.toggle_play();
        assert!(tl.is_playing);

        // Advance 2.5 seconds
        tl.advance(2.5);
        assert_eq!(tl.current_time_sec, 2.5);

        // Advance past end with loop enabled
        tl.advance(8.0);
        assert_eq!(tl.current_time_sec, 0.0);
        assert!(tl.is_playing);

        // Disable loop and advance past end
        tl.toggle_loop();
        tl.advance(11.0);
        assert_eq!(tl.current_time_sec, 10.0);
        assert!(!tl.is_playing);
    }

    #[test]
    fn test_timeline_scrubbing_and_clamping() {
        let mut tl = TimelineState::new(5.0);
        tl.scrub_to(2.5);
        assert_eq!(tl.current_time_sec, 2.5);

        // Clamp negative
        tl.scrub_to(-1.0);
        assert_eq!(tl.current_time_sec, 0.0);

        // Clamp excess
        tl.scrub_to(10.0);
        assert_eq!(tl.current_time_sec, 5.0);

        // Scrub ratio
        tl.scrub_ratio(0.5);
        assert_eq!(tl.current_time_sec, 2.5);
    }

    #[test]
    fn test_timeline_zoom_and_pan() {
        let mut tl = TimelineState::new(10.0);
        tl.scrub_to(5.0);
        tl.set_window_duration(0.200);
        assert_eq!(tl.visible_window_sec, 0.200);
        assert!((tl.window_start_sec - 4.900).abs() < 1e-6);

        // Pan forward 0.1s
        tl.pan_time(0.1);
        assert!((tl.window_start_sec - 5.000).abs() < 1e-6);
    }

    #[test]
    fn test_set_window_range_clamps() {
        let mut tl = TimelineState::new(10.0);
        tl.set_window_range(2.0, 3.5);
        assert_eq!((tl.window_start_sec, tl.visible_window_sec), (2.0, 1.5));

        // Minimum width and right-edge clamping
        tl.set_window_range(9.999, 9.9995);
        assert!((tl.visible_window_sec - 0.005).abs() < 1e-12);
        assert!((tl.window_start_sec - 9.995).abs() < 1e-9);
    }

    #[test]
    fn test_zoom_at_keeps_anchor_fixed() {
        let mut tl = TimelineState::new(10.0);
        tl.window_start_sec = 4.0;
        tl.visible_window_sec = 1.0;

        // Anchor at 25% of the window -> t = 4.25 must stay at 25%
        tl.zoom_at(0.5, 0.25);
        assert!((tl.visible_window_sec - 0.5).abs() < 1e-9);
        let anchor = tl.window_start_sec + 0.25 * tl.visible_window_sec;
        assert!((anchor - 4.25).abs() < 1e-9);

        // Zooming out past the start clamps the window to 0
        tl.window_start_sec = 0.1;
        tl.zoom_at(4.0, 0.0);
        assert!((tl.visible_window_sec - 2.0).abs() < 1e-9);
        assert_eq!(tl.window_start_sec, 0.1);
        tl.zoom_at(1.0, 1.0);
        tl.zoom_at(2.0, 1.0);
        assert_eq!(tl.window_start_sec, 0.0);
    }
}
