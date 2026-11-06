//! Every unit's template in a grid (largest channels overlaid, the unit's colour, its id under
//! it), one image per row drawn in the background and shown through a virtual list (only the rows
//! on screen are laid out; no image is larger than a row, which keeps every image within the
//! GPU's texture limits). A click selects a unit, Ctrl / Shift+click adds it; the owner shows the
//! selection on the timeline; selected cells are framed.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use dspk_app_core::ui::plot::canvas::Frame;
use dspk_app_core::ui::theme::palette::Palette;
use gpui_kit::component::{h_flex, ActiveTheme as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    canvas, div, img, px, uniform_list, App, AppContext as _, Bounds, Context, InteractiveElement as _, IntoElement, MouseButton, ParentElement as _, Pixels, Render, RenderImage,
    Styled as _, Task, Window,
};

use crate::plots;
use crate::sorting::SortingData;

/// Cell size (logical px), the label line under each template, and the images' oversampling.
pub const CELL_W: f32 = 136.;
pub const CELL_H: f32 = 96.;
const LABEL_H: f32 = 16.;
const SCALE: f32 = 1.5;
/// Template rows (largest channels) overlaid per cell.
const ROWS_PER_CELL: usize = 3;

pub type OnSelect = Rc<dyn Fn(usize, bool, &mut Window, &mut App)>;

pub struct TemplateGrid {
    on_select: OnSelect,
    /// What the rows were drawn for: sorting, columns.
    drawn: Option<(*const SortingData, usize)>,
    /// One image per row of the grid.
    rows: Vec<Arc<RenderImage>>,
    /// Units in grid order (largest first), and their ids.
    order: Vec<usize>,
    ids: Vec<usize>,
    cols: usize,
    selected: Vec<usize>,
    /// What to draw again when the width changes.
    sorting: Option<Arc<SortingData>>,
    palette: Option<Palette>,
    bounds: Rc<Cell<Bounds<Pixels>>>,
    _work: Option<Task<()>>,
}

impl TemplateGrid {
    pub fn new(on_select: OnSelect) -> Self {
        Self {
            on_select,
            drawn: None,
            rows: Vec::new(),
            order: Vec::new(),
            ids: Vec::new(),
            cols: 1,
            selected: Vec::new(),
            sorting: None,
            palette: None,
            bounds: Rc::new(Cell::new(Bounds::default())),
            _work: None,
        }
    }

    /// Shows `sorting`'s templates, `selected` framed.
    pub fn set(&mut self, sorting: Option<Arc<SortingData>>, selected: Vec<usize>, palette: Palette, cx: &mut Context<Self>) {
        self.selected = selected;
        self.sorting = sorting.clone();
        self.palette = Some(palette);
        let Some(sorting) = sorting else {
            self.rows.clear();
            self.drawn = None;
            cx.notify();
            return;
        };
        let cols = cols_for(self.bounds.get().size.width.as_f32());
        let key = (Arc::as_ptr(&sorting), cols);
        if self.drawn == Some(key) {
            cx.notify();
            return;
        }
        self.drawn = Some(key);
        let mut order: Vec<usize> = sorting.alive().collect();
        order.sort_by_key(|&u| std::cmp::Reverse(sorting.units[u].info.num_spikes));
        self.ids = order.iter().map(|&u| sorting.units[u].info.unit_id).collect();
        self.order = order.clone();
        self.cols = cols;
        let work = cx.background_spawn(async move { order.chunks(cols).filter_map(|row| plots::to_image(draw_row(&sorting, row, cols, &palette))).collect::<Vec<_>>() });
        self._work = Some(cx.spawn(async move |this, cx| {
            let rows = work.await;
            let _ = this.update(cx, |g, cx| {
                g.rows = rows;
                cx.notify();
            });
        }));
        cx.notify();
    }
}

fn cols_for(width: f32) -> usize {
    ((width / CELL_W).floor() as usize).max(1)
}

/// One row of the grid: the units of `row`, `cols` cells wide.
fn draw_row(sorting: &SortingData, row: &[usize], cols: usize, palette: &Palette) -> Frame {
    let (cw, ch) = ((CELL_W * SCALE) as u32, ((CELL_H - LABEL_H) * SCALE) as u32);
    let mut frame = Frame::new(cw * cols as u32, ch);
    {
        let canvas = frame.canvas();
        canvas.pixels.fill(palette.background);
    }
    for (k, &u) in row.iter().enumerate() {
        let tpl = &sorting.units[u].template;
        plots::template_cell(&mut frame, k as u32 * cw, 0, cw, ch, &tpl[..tpl.len().min(ROWS_PER_CELL)], palette.unit(u), palette, SCALE);
    }
    frame
}

impl Render for TemplateGrid {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let (muted, frame, fg) = (t.muted_foreground, t.primary, t.foreground);
        // Drawn for another width: draw again for this one
        let width = self.bounds.get().size.width.as_f32();
        if width > 0.0 && cols_for(width) != self.cols {
            if let (Some(sorting), Some(palette)) = (self.sorting.clone(), self.palette) {
                let selected = self.selected.clone();
                let me = cx.entity().downgrade();
                cx.defer(move |cx| {
                    let _ = me.update(cx, |g, cx| g.set(Some(sorting), selected, palette, cx));
                });
            }
        }
        let cols = self.cols;
        let (rows, order, ids, selected, on_select) = (self.rows.clone(), self.order.clone(), self.ids.clone(), self.selected.clone(), self.on_select.clone());
        let n_rows = if rows.is_empty() { 0 } else { order.len().div_ceil(cols) };
        let cell = self.bounds.clone();
        let list = uniform_list("template-rows", n_rows, move |range, _, _| {
            range
                .map(|r| {
                    let units: Vec<usize> = order.iter().copied().skip(r * cols).take(cols).collect();
                    let unit_ids: Vec<usize> = ids.iter().copied().skip(r * cols).take(cols).collect();
                    let on_select = on_select.clone();
                    let image = rows.get(r).cloned();
                    let (units2, cell) = (units.clone(), cell.clone());
                    div()
                        .id(("template-row", r))
                        .relative()
                        .h(px(CELL_H))
                        .w(px(cols as f32 * CELL_W))
                        .on_mouse_down(MouseButton::Left, move |e, window, cx| {
                            // The cell under the pointer (its row's origin: the row's left edge)
                            let b = cell.get();
                            let k = ((e.position.x - b.origin.x).as_f32() / CELL_W).floor().max(0.0) as usize;
                            if let Some(&u) = units2.get(k) {
                                let add = e.modifiers.control || e.modifiers.platform || e.modifiers.shift;
                                on_select(u, add, window, cx);
                            }
                        })
                        .children(image.map(|i| img(i).absolute().top_0().left_0().w(px(cols as f32 * CELL_W)).h(px(CELL_H - LABEL_H))))
                        .child(
                            h_flex().absolute().left_0().bottom_0().h(px(LABEL_H)).children(unit_ids.iter().zip(&units).map(|(id, u)| {
                                let on = selected.contains(u);
                                div().w(px(CELL_W)).text_xs().text_center().text_color(if on { fg } else { muted }).child(format!("Unit {id}"))
                            })),
                        )
                        .children(units.iter().enumerate().filter(|(_, u)| selected.contains(u)).map(|(k, _)| {
                            div().absolute().top_0().left(px(k as f32 * CELL_W)).w(px(CELL_W)).h(px(CELL_H)).border_2().border_color(frame).rounded_sm()
                        }))
                })
                .collect()
        })
        .size_full();
        let cell_paint = self.bounds.clone();
        div()
            .id("template-grid")
            .relative()
            .size_full()
            .child(canvas(move |b, _, _| cell_paint.set(b), |_, _, _, _| {}).absolute().size_full())
            .child(list)
            .when(self.rows.is_empty(), |d| d.child(div().absolute().top_0().left_0().p_3().text_xs().text_color(muted).child("Drawing the templates…")))
    }
}
