//! Every unit's template in a grid (largest channels overlaid, the unit's colour), drawn once per
//! sorting and width into one image; a click selects a unit (the owner shows it on the
//! timeline), the selection drawn as frames over the image.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use dspk_app_core::ui::plot::canvas::Frame;
use dspk_app_core::ui::theme::palette::Palette;
use gpui_kit::component::ActiveTheme as _;
use gpui_kit::{
    canvas, div, img, px, App, AppContext as _, Bounds, Context, FontWeight, InteractiveElement as _, IntoElement, MouseButton, ParentElement as _, Pixels, Render, RenderImage,
    StatefulInteractiveElement as _, Styled as _, Task, Window,
};

use crate::plots;

use crate::sorting::SortingData;

/// Cell size (logical px) and the image's oversampling.
pub const CELL_W: f32 = 132.;
pub const CELL_H: f32 = 84.;
const SCALE: f32 = 1.5;
/// Template rows (largest channels) overlaid per cell.
const ROWS_PER_CELL: usize = 3;

pub type OnSelect = Rc<dyn Fn(usize, bool, &mut Window, &mut App)>;

pub struct TemplateGrid {
    on_select: OnSelect,
    /// What the image was drawn for: sorting, columns.
    drawn: Option<(*const SortingData, usize)>,
    image: Option<Arc<RenderImage>>,
    /// Units in grid order (largest first).
    order: Vec<usize>,
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
        Self { on_select, drawn: None, image: None, order: Vec::new(), cols: 1, selected: Vec::new(), sorting: None, palette: None, bounds: Rc::new(Cell::new(Bounds::default())), _work: None }
    }

    /// Shows `sorting`'s templates, `selected` framed.
    pub fn set(&mut self, sorting: Option<Arc<SortingData>>, selected: Vec<usize>, palette: Palette, cx: &mut Context<Self>) {
        self.selected = selected;
        self.sorting = sorting.clone();
        self.palette = Some(palette);
        let Some(sorting) = sorting else {
            self.image = None;
            self.drawn = None;
            cx.notify();
            return;
        };
        let width = self.bounds.get().size.width.as_f32();
        let cols = cols_for(width);
        let key = (Arc::as_ptr(&sorting), cols);
        if self.drawn == Some(key) {
            cx.notify();
            return;
        }
        self.drawn = Some(key);
        let mut order: Vec<usize> = sorting.alive().collect();
        order.sort_by_key(|&u| std::cmp::Reverse(sorting.units[u].info.num_spikes));
        self.order = order.clone();
        self.cols = cols;
        let work = cx.background_spawn(async move { plots::to_image(draw(&sorting, &order, cols, &palette)) });
        self._work = Some(cx.spawn(async move |this, cx| {
            let image = work.await;
            let _ = this.update(cx, |g, cx| {
                g.image = image;
                cx.notify();
            });
        }));
        cx.notify();
    }
}

fn cols_for(width: f32) -> usize {
    ((width / CELL_W).floor() as usize).max(1)
}

/// The grid: one cell per unit in `order`, `cols` per row.
fn draw(sorting: &SortingData, order: &[usize], cols: usize, palette: &Palette) -> Frame {
    let rows = order.len().div_ceil(cols).max(1);
    let (cw, ch) = ((CELL_W * SCALE) as u32, (CELL_H * SCALE) as u32);
    let mut frame = Frame::new(cw * cols as u32, ch * rows as u32);
    {
        let canvas = frame.canvas();
        canvas.pixels.fill(palette.background);
    }
    let cells: Vec<(usize, &Vec<Vec<f32>>)> = order.iter().map(|&u| (u, &sorting.units[u].template)).collect();
    for (k, (u, tpl)) in cells.into_iter().enumerate() {
        let (x0, y0) = ((k % cols) as u32 * cw, (k / cols) as u32 * ch);
        plots::template_cell(&mut frame, x0, y0, cw, ch, &tpl[..tpl.len().min(ROWS_PER_CELL)], palette.unit(u), palette, SCALE);
    }
    frame
}

impl Render for TemplateGrid {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let t = cx.theme();
        let (muted, frame) = (t.muted_foreground, t.primary);
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
        let rows = self.order.len().div_ceil(self.cols).max(1);
        let height = rows as f32 * CELL_H;
        let cols = self.cols;
        let cell = self.bounds.clone();
        let cell_paint = self.bounds.clone();
        let (order, on_select) = (self.order.clone(), self.on_select.clone());
        let frames = self.selected.iter().filter_map(|u| self.order.iter().position(|o| o == u)).map(|k| {
            div()
                .absolute()
                .left(px((k % cols) as f32 * CELL_W))
                .top(px((k / cols) as f32 * CELL_H))
                .w(px(CELL_W))
                .h(px(CELL_H))
                .border_2()
                .border_color(frame)
        });
        div()
            .id("template-grid")
            .size_full()
            .overflow_y_scroll()
            .child(
                div()
                    .relative()
                    .w_full()
                    .h(px(height))
                    .on_mouse_down(MouseButton::Left, move |e, window, cx| {
                        let b = cell.get();
                        let (x, y) = ((e.position.x - b.origin.x).as_f32(), (e.position.y - b.origin.y).as_f32());
                        let k = (y / CELL_H).floor() as usize * cols + (x / CELL_W).floor() as usize;
                        if (x / CELL_W) < cols as f32 {
                            if let Some(&u) = order.get(k) {
                                let add = e.modifiers.control || e.modifiers.platform || e.modifiers.shift;
                                on_select(u, add, window, cx);
                            }
                        }
                    })
                    .child(canvas(move |b, _, _| cell_paint.set(b), |_, _, _, _| {}).absolute().size_full())
                    .children(self.image.clone().map(|i| img(i).absolute().top_0().left_0().w(px(cols as f32 * CELL_W)).h(px(height))))
                    .children(frames)
                    .when(self.image.is_none(), |d| d.child(div().p_3().text_xs().text_color(muted).font_weight(FontWeight::NORMAL).child("Drawing the templates…"))),
            )
    }
}

use gpui_kit::prelude::FluentBuilder as _;
