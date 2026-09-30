//! A module's docked workspace: its views, the dock tree placing them, focus, and the
//! flattened layout. Generic so every module (time, spikes, …) docks the same way.

use serde::{Deserialize, Serialize};

use super::dock::{dragged_ratio, lay_out, Dock, DockLayout, DropSide, ViewId};

/// Seat chrome around a plot, in logical px. The Slint seat uses the same values (pushed to
/// the `Metrics` global at startup) so plot canvases map 1:1 onto their images.
pub const SEAT_HEADER: f32 = 30.0;
pub const GUTTER: f32 = 64.0;
pub const AXIS: f32 = 22.0;

/// What a workspace needs from a view.
pub trait DockView {
    fn id(&self) -> ViewId;
    /// Plot canvas in physical pixels.
    fn set_canvas(&mut self, width: u32, height: u32, scale: f32);
    fn mark_dirty(&mut self);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockWorkspace<V> {
    pub views: Vec<V>,
    pub dock: Dock,
    pub focused: Option<ViewId>,
    /// Workspace size (logical px) and display scale.
    #[serde(skip)]
    pub size: (f32, f32),
    #[serde(skip, default = "one")]
    pub scale: f32,
    /// Flattened layout, logical px relative to the workspace.
    #[serde(skip)]
    pub layout: DockLayout,
    #[serde(skip)]
    divider_drag: Option<(usize, f32)>,
}

fn one() -> f32 {
    1.0
}

impl<V: DockView> DockWorkspace<V> {
    pub fn new(views: Vec<V>, dock: Dock, focused: Option<ViewId>) -> Self {
        Self { views, dock, focused, size: (1000.0, 600.0), scale: 1.0, layout: DockLayout::default(), divider_drag: None }
    }

    /// Takes over a restored workspace, keeping the current size and marking views stale.
    pub fn adopt(&mut self, mut restored: Self) {
        restored.size = self.size;
        restored.scale = self.scale;
        if restored.focused.is_none_or(|f| !restored.dock.contains(f)) {
            restored.focused = restored.dock.views().first().copied();
        }
        *self = restored;
        for v in &mut self.views {
            v.mark_dirty();
        }
        self.relayout();
    }

    /// Structurally sound: every placed view exists once, ids are unique.
    pub fn is_consistent(&self) -> bool {
        let mut ids: Vec<ViewId> = self.views.iter().map(|v| v.id()).collect();
        ids.sort_unstable();
        let unique = ids.windows(2).all(|w| w[0] != w[1]);
        let mut placed = self.dock.views();
        placed.sort_unstable();
        unique && placed.windows(2).all(|w| w[0] != w[1]) && placed.iter().all(|p| ids.binary_search(p).is_ok())
    }

    pub fn next_id(&self) -> ViewId {
        self.views.iter().map(|v| v.id()).max().unwrap_or(0) + 1
    }

    pub fn view(&self, id: ViewId) -> Option<&V> {
        self.views.iter().find(|v| v.id() == id)
    }

    pub fn view_mut(&mut self, id: ViewId) -> Option<&mut V> {
        self.views.iter_mut().find(|v| v.id() == id)
    }

    pub fn focused_view(&self) -> Option<&V> {
        self.focused.and_then(|id| self.view(id))
    }

    pub fn focused_view_mut(&mut self) -> Option<&mut V> {
        let id = self.focused?;
        self.view_mut(id)
    }

    pub fn mark_all_dirty(&mut self) {
        for v in &mut self.views {
            v.mark_dirty();
        }
    }

    /// Replaces views and arrangement (layout presets).
    pub fn reset(&mut self, views: Vec<V>, dock: Dock, focused: Option<ViewId>) {
        self.views = views;
        self.dock = dock;
        self.focused = focused;
        self.relayout();
    }

    /// Docks `view` beside the focused view (or anywhere if nothing is focused) and focuses it.
    pub fn add(&mut self, view: V, side: DropSide) -> ViewId {
        let id = view.id();
        self.views.push(view);
        let target = self.focused.unwrap_or(0);
        if !self.dock.insert(id, target, side) {
            let any = self.dock.views().first().copied().unwrap_or(0);
            self.dock.insert(id, any, side);
        }
        self.focused = Some(id);
        self.relayout();
        id
    }

    pub fn close(&mut self, id: ViewId) {
        self.dock.remove(id);
        self.views.retain(|v| v.id() != id);
        if self.focused == Some(id) {
            self.focused = self.dock.views().first().copied();
        }
        self.relayout();
    }

    /// Returns true when focus changed.
    pub fn focus(&mut self, id: ViewId) -> bool {
        if self.view(id).is_some() && self.focused != Some(id) {
            self.focused = Some(id);
            return true;
        }
        false
    }

    pub fn activate(&mut self, id: ViewId) {
        self.dock.activate(id);
        self.focused = Some(id);
        self.relayout();
    }

    pub fn drop_view(&mut self, view: ViewId, target: ViewId, side: DropSide) {
        if self.dock.move_view(view, target, side) {
            self.focused = Some(view);
            self.relayout();
        }
    }

    /// Every view as a tab of one seat.
    pub fn compact(&mut self) {
        let ids = self.dock.views();
        let active = self.focused.and_then(|f| ids.iter().position(|&v| v == f)).unwrap_or(0);
        self.dock = Dock::Tabs { views: ids, active };
        self.relayout();
    }

    pub fn resized(&mut self, width: f32, height: f32, scale: f32) {
        if (width, height, scale) != (self.size.0, self.size.1, self.scale) {
            self.size = (width, height);
            self.scale = scale.max(0.1);
            self.relayout();
        }
    }

    pub fn divider_pressed(&mut self, index: usize) {
        self.divider_drag = self.dock.ratio(index).map(|r| (index, r));
    }

    /// Divider drag by `delta` logical px since the press.
    pub fn divider_dragged(&mut self, index: usize, delta: f32) {
        let Some((i, start)) = self.divider_drag else { return };
        let Some(extent) = self.layout.dividers.iter().find(|d| d.index == index).map(|d| d.extent) else { return };
        if i == index {
            self.dock.set_ratio(index, dragged_ratio(start, delta, extent));
            self.relayout();
        }
    }

    /// Recomputes seat geometry and the canvas of every visible view.
    pub fn relayout(&mut self) {
        let (w, h) = self.size;
        self.layout = lay_out(&self.dock, (0.0, 0.0, w, h));
        let scale = self.scale;
        let sizes: Vec<(ViewId, u32, u32)> = self
            .layout
            .seats
            .iter()
            .filter_map(|s| {
                let id = s.active_view()?;
                let pw = ((s.width - GUTTER) * scale).max(1.0) as u32;
                let ph = ((s.height - SEAT_HEADER - AXIS) * scale).max(1.0) as u32;
                Some((id, pw, ph))
            })
            .collect();
        for (id, pw, ph) in sizes {
            if let Some(v) = self.view_mut(id) {
                v.set_canvas(pw, ph, scale);
            }
        }
    }

    /// Views on screen (the active tab of each seat).
    pub fn visible(&self) -> Vec<ViewId> {
        self.layout.seats.iter().filter_map(|s| s.active_view()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct V {
        id: ViewId,
        canvas: (u32, u32),
        dirty: bool,
    }

    impl DockView for V {
        fn id(&self) -> ViewId {
            self.id
        }
        fn set_canvas(&mut self, width: u32, height: u32, _scale: f32) {
            self.canvas = (width, height);
        }
        fn mark_dirty(&mut self) {
            self.dirty = true;
        }
    }

    fn v(id: ViewId) -> V {
        V { id, canvas: (0, 0), dirty: false }
    }

    #[test]
    fn test_workspace_add_close_focus_and_canvas() {
        let mut ws = DockWorkspace::new(vec![v(1), v(2)], Dock::stacked(1, 2, 0.5), Some(1));
        ws.resized(1000.0, 606.0, 1.0);
        assert_eq!(ws.view(1).unwrap().canvas, (1000 - GUTTER as u32, (300.0 - SEAT_HEADER - AXIS) as u32));

        let id = ws.next_id();
        assert_eq!(id, 3);
        ws.add(v(id), DropSide::Right);
        assert_eq!(ws.focused, Some(3));
        assert_eq!(ws.visible().len(), 3);

        ws.close(3);
        assert_eq!(ws.visible(), vec![1, 2]);
        assert!(ws.focus(2));
        assert!(!ws.focus(2));
        assert!(ws.is_consistent());

        ws.compact();
        assert_eq!(ws.visible(), vec![2]);
    }

    #[test]
    fn test_adopt_restored_workspace() {
        let mut ws = DockWorkspace::new(vec![v(1)], Dock::single(1), Some(1));
        ws.resized(800.0, 400.0, 2.0);
        let json = serde_json::to_string(&DockWorkspace::new(vec![v(5), v(6)], Dock::stacked(5, 6, 0.5), Some(99))).unwrap();
        ws.adopt(serde_json::from_str(&json).unwrap());
        assert_eq!(ws.scale, 2.0);
        assert_eq!(ws.focused, Some(5), "invalid focus falls back to the first view");
        assert!(ws.views.iter().all(|v| v.dirty));
        assert_eq!(ws.layout.seats.len(), 2);
    }
}
