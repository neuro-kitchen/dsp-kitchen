//! Workspace arrangement: a tree of splits whose leaves are tabbed seats of views.
//!
//! Slint components cannot contain themselves, so the tree lives here and is walked flat
//! into positioned seats and dividers that the UI draws with plain `for` loops.

use serde::{Deserialize, Serialize};

pub type ViewId = u32;

/// Smallest size a seat may be squeezed to by a divider drag (logical px).
pub const SEAT_MIN: f32 = 120.0;
/// Gutter between the two halves of a split (logical px).
pub const GAP: f32 = 6.0;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Dock {
    /// A seat: one or more views behind tabs, `active` is the one shown.
    Tabs { views: Vec<ViewId>, active: usize },
    /// Two nodes side by side (`columns`) or stacked; `ratio` is the first node's share
    /// of the extent left after the gutter.
    Split { columns: bool, ratio: f32, first: Box<Dock>, second: Box<Dock> },
}

/// Where a dragged view lands on a seat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropSide {
    /// Join the seat as a tab.
    Centre,
    Left,
    Right,
    Top,
    Bottom,
}

/// A seat, placed.
#[derive(Debug, Clone, PartialEq)]
pub struct SeatBox {
    pub views: Vec<ViewId>,
    pub active: usize,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl SeatBox {
    pub fn active_view(&self) -> Option<ViewId> {
        self.views.get(self.active).copied()
    }
}

/// The gutter of one split. `index` is the split's position in a parent-first walk.
#[derive(Debug, Clone, PartialEq)]
pub struct DividerBox {
    pub index: usize,
    pub columns: bool,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    /// Extent shared by the two halves (split size minus the gutter), to turn px into ratio.
    pub extent: f32,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DockLayout {
    pub seats: Vec<SeatBox>,
    pub dividers: Vec<DividerBox>,
}

impl Dock {
    pub fn single(view: ViewId) -> Self {
        Dock::Tabs { views: vec![view], active: 0 }
    }

    pub fn empty() -> Self {
        Dock::Tabs { views: Vec::new(), active: 0 }
    }

    /// Two views stacked: `top` gets `ratio` of the height.
    pub fn stacked(top: ViewId, bottom: ViewId, ratio: f32) -> Self {
        Dock::Split {
            columns: false,
            ratio,
            first: Box::new(Dock::single(top)),
            second: Box::new(Dock::single(bottom)),
        }
    }

    /// Every view in the tree, in walk order.
    pub fn views(&self) -> Vec<ViewId> {
        let mut out = Vec::new();
        self.walk_views(&mut out);
        out
    }

    fn walk_views(&self, out: &mut Vec<ViewId>) {
        match self {
            Dock::Tabs { views, .. } => out.extend(views),
            Dock::Split { first, second, .. } => {
                first.walk_views(out);
                second.walk_views(out);
            }
        }
    }

    pub fn contains(&self, view: ViewId) -> bool {
        self.seat_path(view).is_some()
    }

    fn at(&self, path: &[bool]) -> &Dock {
        match (path.split_first(), self) {
            (Some((step, rest)), Dock::Split { first, second, .. }) => {
                if *step { second.at(rest) } else { first.at(rest) }
            }
            _ => self,
        }
    }

    fn at_mut(&mut self, path: &[bool]) -> &mut Dock {
        match (path.split_first(), self) {
            (Some((step, rest)), Dock::Split { first, second, .. }) => {
                if *step { second.at_mut(rest) } else { first.at_mut(rest) }
            }
            (_, node) => node,
        }
    }

    /// Path (false = first, true = second) to the seat holding `view`.
    fn seat_path(&self, view: ViewId) -> Option<Vec<bool>> {
        fn walk(node: &Dock, view: ViewId, path: &mut Vec<bool>) -> bool {
            match node {
                Dock::Tabs { views, .. } => views.contains(&view),
                Dock::Split { first, second, .. } => {
                    for (step, child) in [(false, first), (true, second)] {
                        path.push(step);
                        if walk(child, view, path) {
                            return true;
                        }
                        path.pop();
                    }
                    false
                }
            }
        }
        let mut path = Vec::new();
        walk(self, view, &mut path).then_some(path)
    }

    /// Path to the `index`-th split in a parent-first walk (the divider numbering).
    fn split_path(&self, index: usize) -> Option<Vec<bool>> {
        fn walk(node: &Dock, want: usize, seen: &mut usize, path: &mut Vec<bool>) -> bool {
            let Dock::Split { first, second, .. } = node else { return false };
            if *seen == want {
                return true;
            }
            *seen += 1;
            for (step, child) in [(false, first), (true, second)] {
                path.push(step);
                if walk(child, want, seen, path) {
                    return true;
                }
                path.pop();
            }
            false
        }
        let (mut seen, mut path) = (0, Vec::new());
        walk(self, index, &mut seen, &mut path).then_some(path)
    }

    /// The views sharing a seat with `view` (including it).
    pub fn seat_views(&self, view: ViewId) -> Vec<ViewId> {
        match self.seat_path(view).map(|p| self.at(&p)) {
            Some(Dock::Tabs { views, .. }) => views.clone(),
            _ => Vec::new(),
        }
    }

    /// Makes `view` the visible tab of its seat.
    pub fn activate(&mut self, view: ViewId) -> bool {
        let Some(path) = self.seat_path(view) else { return false };
        if let Dock::Tabs { views, active } = self.at_mut(&path) {
            if let Some(i) = views.iter().position(|&v| v == view) {
                *active = i;
                return true;
            }
        }
        false
    }

    /// Removes `view`; an emptied seat disappears and its sibling takes the space.
    pub fn remove(&mut self, view: ViewId) -> bool {
        let Some(path) = self.seat_path(view) else { return false };
        let now_empty = match self.at_mut(&path) {
            Dock::Tabs { views, active } => {
                let i = views.iter().position(|&v| v == view).unwrap_or(0);
                views.remove(i);
                if *active >= i && *active > 0 {
                    *active -= 1;
                }
                views.is_empty()
            }
            Dock::Split { .. } => false,
        };
        if now_empty {
            if let Some((last, above)) = path.split_last() {
                let parent = self.at_mut(above);
                if let Dock::Split { first, second, .. } = parent {
                    let survivor = if *last { first } else { second };
                    let survivor = std::mem::replace(survivor.as_mut(), Dock::empty());
                    *parent = survivor;
                }
            }
            // At the root an empty seat stays: the workspace shows its "add a view" hint
        }
        true
    }

    /// Places `view` relative to the seat holding `target`: as a tab (centre) or in a new
    /// seat split off the given side. `view` must not already be in the tree.
    pub fn insert(&mut self, view: ViewId, target: ViewId, side: DropSide) -> bool {
        let path = match self.seat_path(target) {
            Some(p) => p,
            // Empty workspace: the root seat takes the view
            None if matches!(self, Dock::Tabs { views, .. } if views.is_empty()) => Vec::new(),
            None => return false,
        };
        let node = self.at_mut(&path);
        match side {
            DropSide::Centre => {
                if let Dock::Tabs { views, active } = node {
                    views.push(view);
                    *active = views.len() - 1;
                    return true;
                }
                false
            }
            _ => {
                let existing = Box::new(std::mem::replace(node, Dock::empty()));
                let new = Box::new(Dock::single(view));
                let (columns, first, second) = match side {
                    DropSide::Left => (true, new, existing),
                    DropSide::Right => (true, existing, new),
                    DropSide::Top => (false, new, existing),
                    _ => (false, existing, new),
                };
                *node = Dock::Split { columns, ratio: 0.5, first, second };
                true
            }
        }
    }

    /// Moves `view` next to (or into) the seat of `target`. Dropping a view onto its own
    /// single-view seat is a no-op; onto its own multi-tab seat it splits off the others.
    pub fn move_view(&mut self, view: ViewId, target: ViewId, side: DropSide) -> bool {
        if !self.contains(view) {
            return false;
        }
        // Resolve the target seat by a view that will still be there after removal
        let anchor = if target == view {
            match self.seat_views(view).into_iter().find(|&v| v != view) {
                Some(other) if side != DropSide::Centre => other,
                _ => return false,
            }
        } else {
            target
        };
        if side == DropSide::Centre && self.seat_views(anchor).contains(&view) {
            return self.activate(view);
        }
        self.remove(view);
        self.insert(view, anchor, side)
    }

    /// Sets the ratio of the `index`-th split (parent-first numbering).
    pub fn set_ratio(&mut self, index: usize, value: f32) {
        if let Some(path) = self.split_path(index) {
            if let Dock::Split { ratio, .. } = self.at_mut(&path) {
                *ratio = value.clamp(0.0, 1.0);
            }
        }
    }

    pub fn ratio(&self, index: usize) -> Option<f32> {
        match self.split_path(index).map(|p| self.at(&p)) {
            Some(Dock::Split { ratio, .. }) => Some(*ratio),
            _ => None,
        }
    }
}

/// New ratio for a divider dragged by `delta` px from `start_ratio`, keeping both halves at
/// least `SEAT_MIN` (or half the extent when the split is smaller than two minimum seats).
pub fn dragged_ratio(start_ratio: f32, delta: f32, extent: f32) -> f32 {
    if extent <= 0.0 {
        return start_ratio;
    }
    let min = (SEAT_MIN / extent).min(0.5);
    (start_ratio + delta / extent).clamp(min, 1.0 - min)
}

/// Walks the tree into positioned seats and dividers inside `(x, y, w, h)`.
pub fn lay_out(dock: &Dock, rect: (f32, f32, f32, f32)) -> DockLayout {
    let mut out = DockLayout::default();
    walk(dock, rect, &mut out);
    out
}

fn walk(node: &Dock, (x, y, w, h): (f32, f32, f32, f32), out: &mut DockLayout) {
    match node {
        Dock::Tabs { views, active } => out.seats.push(SeatBox {
            views: views.clone(),
            active: (*active).min(views.len().saturating_sub(1)),
            x,
            y,
            width: w.max(0.0),
            height: h.max(0.0),
        }),
        Dock::Split { columns, ratio, first, second } => {
            // Numbered before the children so the order is parent-first
            let index = out.dividers.len();
            let along = if *columns { w } else { h };
            let extent = (along - GAP).max(0.0);
            let head = (extent * ratio).clamp(0.0, extent);
            out.dividers.push(if *columns {
                DividerBox { index, columns: true, x: x + head, y, width: GAP, height: h.max(0.0), extent }
            } else {
                DividerBox { index, columns: false, x, y: y + head, width: w.max(0.0), height: GAP, extent }
            });
            if *columns {
                walk(first, (x, y, head, h), out);
                walk(second, (x + head + GAP, y, extent - head, h), out);
            } else {
                walk(first, (x, y, w, head), out);
                walk(second, (x, y + head + GAP, w, extent - head), out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_splits_space_and_numbers_dividers() {
        let dock = Dock::Split {
            columns: true,
            ratio: 0.5,
            first: Box::new(Dock::single(1)),
            second: Box::new(Dock::stacked(2, 3, 0.25)),
        };
        let l = lay_out(&dock, (0.0, 0.0, 206.0, 106.0));
        assert_eq!(l.seats.len(), 3);
        assert_eq!(l.dividers.len(), 2);
        // Outer split: 200 px shared, 100 each
        assert_eq!((l.seats[0].x, l.seats[0].width), (0.0, 100.0));
        assert_eq!((l.seats[1].x, l.seats[1].width), (106.0, 100.0));
        // Inner split: 100 px tall shared, 25 / 75
        assert_eq!(l.seats[1].height, 25.0);
        assert_eq!((l.seats[2].y, l.seats[2].height), (31.0, 75.0));
        assert_eq!((l.dividers[0].index, l.dividers[0].columns), (0, true));
        assert_eq!((l.dividers[1].index, l.dividers[1].columns), (1, false));
        assert_eq!(l.dividers[1].extent, 100.0);
    }

    #[test]
    fn test_insert_sides_and_tabs() {
        let mut dock = Dock::single(1);
        assert!(dock.insert(2, 1, DropSide::Right));
        assert!(dock.insert(3, 2, DropSide::Bottom));
        assert!(dock.insert(4, 3, DropSide::Centre));
        assert_eq!(dock.views(), vec![1, 2, 3, 4]);
        assert_eq!(dock.seat_views(4), vec![3, 4]);

        let l = lay_out(&dock, (0.0, 0.0, 400.0, 400.0));
        assert_eq!(l.seats.len(), 3);
        // New tab is active
        assert_eq!(l.seats[2].active_view(), Some(4));
        assert!(dock.activate(3));
        assert_eq!(lay_out(&dock, (0.0, 0.0, 400.0, 400.0)).seats[2].active_view(), Some(3));
    }

    #[test]
    fn test_remove_collapses_split_and_keeps_root_seat() {
        let mut dock = Dock::stacked(1, 2, 0.5);
        dock.insert(3, 2, DropSide::Right);
        assert!(dock.remove(2));
        // Seat of 2 is gone; 3 took its place under the root split
        assert_eq!(dock, Dock::stacked(1, 3, 0.5));
        dock.remove(1);
        assert_eq!(dock, Dock::single(3));
        dock.remove(3);
        assert_eq!(dock, Dock::empty());
        assert_eq!(lay_out(&dock, (0.0, 0.0, 10.0, 10.0)).seats.len(), 1);
        // An empty workspace accepts a view again
        assert!(dock.insert(7, 0, DropSide::Centre));
        assert_eq!(dock.views(), vec![7]);
    }

    #[test]
    fn test_remove_tab_keeps_seat_and_fixes_active() {
        let mut dock = Dock::Tabs { views: vec![1, 2, 3], active: 2 };
        dock.remove(3);
        assert_eq!(dock, Dock::Tabs { views: vec![1, 2], active: 1 });
        dock.remove(1);
        assert_eq!(dock, Dock::Tabs { views: vec![2], active: 0 });
    }

    #[test]
    fn test_move_view() {
        let mut dock = Dock::stacked(1, 2, 0.5);
        // Onto another seat's centre -> becomes a tab there
        assert!(dock.move_view(1, 2, DropSide::Centre));
        assert_eq!(dock, Dock::Tabs { views: vec![2, 1], active: 1 });
        // Out of its own tabbed seat to the left
        assert!(dock.move_view(1, 1, DropSide::Left));
        assert_eq!(dock.views(), vec![1, 2]);
        assert!(matches!(dock, Dock::Split { columns: true, .. }));
        // Onto its own single seat: no-op
        let before = dock.clone();
        assert!(!dock.move_view(1, 1, DropSide::Right));
        assert_eq!(dock, before);
    }

    #[test]
    fn test_divider_drag_clamps_to_min_seat() {
        assert_eq!(dragged_ratio(0.5, 100.0, 1000.0), 0.6);
        assert_eq!(dragged_ratio(0.5, -1000.0, 1000.0), SEAT_MIN / 1000.0);
        assert_eq!(dragged_ratio(0.5, 1000.0, 1000.0), 1.0 - SEAT_MIN / 1000.0);
        // Too small for two minimum seats: pinned to the middle
        assert_eq!(dragged_ratio(0.2, 50.0, 100.0), 0.5);

        let mut dock = Dock::stacked(1, 2, 0.5);
        dock.set_ratio(0, 0.7);
        assert_eq!(dock.ratio(0), Some(0.7));
        assert_eq!(dock.ratio(1), None);
    }
}
