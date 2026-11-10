//! Analytics, units aligned to events: for each unit of the active view's sorting, its spikes
//! around every event (a raster, one row per event) over its peri-event time histogram (PETH,
//! spikes/s). The numbers come from `dspk_synapse::metrics` (`compute_aligned_raster`,
//! `compute_psth`); this module only draws them. Cards are kept until the sorting, the events or
//! the window change.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use dspk_app_core::data::store::Store;
use dspk_app_core::ui::plot::canvas::{Frame, Pixel};
use dspk_app_core::ui::theme::palette::Palette;
use dspk_synapse::metrics::{compute_aligned_raster, compute_psth, PsthOptions};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _};
use gpui_kit::{div, img, px, AnyElement, App, Entity, FontWeight, IntoElement, ParentElement as _, RenderImage, Styled as _};

use crate::sorting::SortingData;

/// Most units shown (those with the most spikes near the events first).
pub const MAX_UNITS: usize = 24;
/// PETH bin width (ms).
pub const PETH_BIN_MS: f64 = 10.0;
const CARD_W: f32 = 260.;
const CARD_H: f32 = 130.;
/// Share of a card's height the raster takes (the PETH below).
const RASTER_SHARE: f32 = 0.6;

/// One unit's card.
#[derive(Clone)]
pub struct UnitCard {
    pub unit_id: i32,
    pub index: usize,
    /// Spikes within the windows, over all events.
    pub spikes: usize,
    pub peak_hz: f32,
    pub image: Option<Arc<RenderImage>>,
}

/// What the cards were made for.
#[derive(Debug, Clone, PartialEq)]
struct Key {
    sorting: PathBuf,
    events: usize,
    first: Option<u64>,
    last: Option<u64>,
    pre_ms: f64,
    post_ms: f64,
    dark: bool,
}

/// The last cards made (the plugin keeps one set).
#[derive(Default)]
pub struct Cache(Mutex<Option<(Key, Arc<Vec<UnitCard>>)>>);

/// The units of `sorting` aligned to `events` (samples), drawn: raster over PETH.
pub fn unit_cards(sorting: &SortingData, events: &[u64], pre_ms: f64, post_ms: f64, palette: &Palette) -> Vec<UnitCard> {
    let rate = sorting.sample_rate;
    let mut per_unit: Vec<Vec<u64>> = vec![Vec::new(); sorting.units.len()];
    for (&t, u) in sorting.times.iter().zip(&sorting.unit_of) {
        if let Some(u) = u {
            per_unit[*u as usize].push(t);
        }
    }
    let options = PsthOptions { pre_ms, post_ms, bin_ms: PETH_BIN_MS };
    let mut rows: Vec<(usize, dspk_synapse::metrics::AlignedRaster)> =
        sorting.alive().map(|u| (u, compute_aligned_raster(&per_unit[u], events, rate, pre_ms, post_ms))).filter(|(_, r)| !r.trial.is_empty()).collect();
    rows.sort_by_key(|(_, r)| std::cmp::Reverse(r.trial.len()));
    rows.truncate(MAX_UNITS);
    rows.into_iter()
        .map(|(u, raster)| {
            let peth = compute_psth(&per_unit[u], events, rate, &options);
            let peak_hz = peth.mean_rate_hz.iter().copied().fold(0.0f32, f32::max);
            let frame = draw(&raster, &peth.mean_rate_hz, pre_ms, post_ms, palette.unit(u), palette, 2.0);
            let (w, h) = (frame.width, frame.height);
            let image = image::RgbaImage::from_raw(w, h, frame.into_bgra()).map(|b| Arc::new(RenderImage::new(vec![image::Frame::new(b)])));
            UnitCard { unit_id: sorting.units[u].info.unit_id as i32, index: u, spikes: raster.trial.len(), peak_hz, image }
        })
        .collect()
}

/// Raster (a dot per spike, a row per event) over the PETH bars, with a line at the event.
fn draw(raster: &dspk_synapse::metrics::AlignedRaster, peth: &[f32], pre_ms: f64, post_ms: f64, colour: Pixel, palette: &Palette, scale: f32) -> Frame {
    let (width, height) = ((CARD_W * scale) as u32, (CARD_H * scale) as u32);
    let mut frame = Frame::new(width, height);
    let mut c = frame.canvas();
    c.pixels.fill(palette.background);
    let (w, h) = (width as f32, height as f32);
    let split = h * RASTER_SHARE;
    let span = (pre_ms + post_ms) as f32;
    let x = |ms: f64| ((ms + pre_ms) as f32 / span) * (w - 1.0);
    let event_x = x(0.0);
    c.rect(event_x - 0.5 * scale, 0.0, event_x + 0.5 * scale, h, palette.grid, 1.0);
    c.rect(0.0, split - 0.5, w, split + 0.5, palette.grid, 0.8);
    let rows = raster.num_trials.max(1) as f32;
    let row_h = (split - 2.0) / rows;
    let dot = row_h.clamp(1.0, 3.0 * scale);
    for (&trial, &ms) in raster.trial.iter().zip(&raster.time_ms) {
        let (px, py) = (x(ms), 1.0 + (trial as f32 + 0.5) * row_h);
        c.rect(px - 0.75 * scale, py - dot / 2.0, px + 0.75 * scale, py + dot / 2.0, colour, 0.9);
    }
    let peak = peth.iter().copied().fold(0.0f32, f32::max).max(1e-6);
    let bw = w / peth.len().max(1) as f32;
    for (i, &r) in peth.iter().enumerate() {
        let top = h - (r / peak) * (h - split - 3.0);
        c.rect(i as f32 * bw + 0.5, top, (i + 1) as f32 * bw - 0.5, h, colour, 0.75);
    }
    frame
}

impl Cache {
    /// The cards of the active view's sorting for `events_s`, made once per sorting, events and
    /// window; `None` when the view has no loaded sorting.
    pub fn cards(&self, store: &Entity<Store>, events_s: &[f64], pre_ms: f64, post_ms: f64, cx: &App) -> Option<(Arc<SortingData>, Arc<Vec<UnitCard>>)> {
        let s = store.read(cx);
        let view = s.active_view()?;
        let path = crate::sorting_of(view)?;
        let sorting = crate::state(cx)?.read(cx).sorting(&path)?;
        let events: Vec<u64> = events_s.iter().map(|&t| (t * sorting.sample_rate).round().max(0.0) as u64).collect();
        let palette = s.palette();
        let key = Key { sorting: path, events: events.len(), first: events.first().copied(), last: events.last().copied(), pre_ms, post_ms, dark: palette.dark };
        let mut cache = self.0.lock().unwrap();
        if let Some((k, cards)) = cache.as_ref() {
            if *k == key {
                return Some((sorting, cards.clone()));
            }
        }
        let cards = Arc::new(unit_cards(&sorting, &events, pre_ms, post_ms, &palette));
        *cache = Some((key, cards.clone()));
        Some((sorting, cards))
    }
}

/// The Analytics section of the units: a card per unit.
pub fn section(cache: &Cache, store: &Entity<Store>, events_s: &[f64], pre_ms: f64, post_ms: f64, cx: &App) -> Option<AnyElement> {
    let (sorting, cards) = cache.cards(store, events_s, pre_ms, post_ms, cx)?;
    let t = cx.theme();
    let (muted, border) = (t.muted_foreground, t.border);
    let title = format!("UNITS · {} · raster over PETH ({PETH_BIN_MS:.0} ms bins)", sorting.sorter);
    let body = cards.iter().map(|c| {
        v_flex()
            .w(px(CARD_W + 16.))
            .p_2()
            .gap_1()
            .rounded_md()
            .border_1()
            .border_color(border)
            .child(
                h_flex()
                    .gap_2()
                    .child(div().text_sm().font_weight(FontWeight::SEMIBOLD).child(format!("Unit {}", c.unit_id)))
                    .child(div().flex_1())
                    .child(div().text_xs().text_color(muted).child(format!("{} spikes · peak {:.0} Hz", c.spikes, c.peak_hz))),
            )
            .children(c.image.clone().map(|i| img(i).w(px(CARD_W)).h(px(CARD_H))))
            .child(h_flex().text_xs().text_color(muted).child(format!("−{pre_ms:.0} ms")).child(div().flex_1()).child("0").child(div().flex_1()).child(format!("+{post_ms:.0} ms")))
            .into_any_element()
    });
    Some(
        v_flex()
            .gap_3()
            .child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(muted).child(title))
            .child(if cards.is_empty() {
                div().text_sm().text_color(muted).child("No spikes of the linked sorting fall near these events.").into_any_element()
            } else {
                h_flex().flex_wrap().gap_3().children(body).into_any_element()
            })
            .into_any_element(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sorting::tests::sorting_of_units;

    #[test]
    fn cards_of_units_with_spikes_near_the_events() {
        // 1 kHz; unit id 1: 50 spikes every 10 samples from 0; unit id 2: 5 spikes from 500 on
        let s = sorting_of_units(&[(1, 0, 50), (2, 0, 5)]);
        // Events at 105 and 305 (spikes mid-bin), window −20 … +50 ms: 7 spikes of unit 1 around
        // each, none of unit 2
        let cards = unit_cards(&s, &[105, 305], 20.0, 50.0, &Palette::of(true));
        assert_eq!(cards.len(), 1, "units without spikes near the events are left out");
        let c = &cards[0];
        assert_eq!((c.unit_id, c.spikes), (1, 14));
        // One spike per 10 ms bin in each trial: 100 spikes/s
        assert!((c.peak_hz - 100.0).abs() < 1e-3, "{}", c.peak_hz);
        assert!(c.image.is_some());
    }
}
