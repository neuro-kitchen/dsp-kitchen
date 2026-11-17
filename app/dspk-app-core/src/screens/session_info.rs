//! A recording's session and subject, in NWB's terms ([`dspk_session::session_meta`]), in its
//! metadata card. What the data says is shown as it is; what it does not say is typed here and
//! kept in the recording's `<rec>.dspk.zarr` (never in the recording). The people of the app can
//! be added as experimenters with a click.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use dspk_session::session_meta::{Field, SessionMetadata, FIELDS};
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Input, InputState};
use gpui_kit::component::{h_flex, v_flex, ActiveTheme as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{div, px, AnyElement, AppContext as _, Context, Entity, FontWeight, InteractiveElement as _, IntoElement, ParentElement as _, Render, SharedString, StatefulInteractiveElement as _, Styled as _, Window};

use crate::ui::widgets::row;

/// The first field of the subject's group.
const SUBJECT_FROM: &str = "subject.subject_id";

pub struct SessionInfoForm {
    /// The recording shown.
    path: Option<PathBuf>,
    /// What the data says (shown, not edited).
    from_data: SessionMetadata,
    /// One input per field.
    inputs: Vec<(&'static Field, Entity<InputState>)>,
    open: bool,
    /// The last save: whether it worked, and what to say.
    message: Option<(bool, String)>,
}

impl SessionInfoForm {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let inputs = FIELDS.iter().map(|f| (f, cx.new(|cx| InputState::new(window, cx).placeholder(f.hint)))).collect();
        Self { path: None, from_data: SessionMetadata::default(), inputs, open: false, message: None }
    }

    /// Shows `path`'s session: what its data says (an NWB-Zarr file's own fields over what its
    /// reader knows, `reader`) and what was filled in before.
    pub fn show(&mut self, path: &Path, reader: &BTreeMap<String, String>, window: &mut Window, cx: &mut Context<Self>) {
        if self.path.as_deref() == Some(path) {
            return;
        }
        self.path = Some(path.to_path_buf());
        let is_nwb_zarr = path.is_dir() && path.to_string_lossy().trim_end_matches('/').ends_with(".nwb.zarr");
        let nwb = if is_nwb_zarr { dspk_session::derived::session_meta::from_nwb_zarr(path) } else { SessionMetadata::default() };
        self.from_data = nwb.over(&SessionMetadata::from_reader(reader));
        let filled = dspk_session::derived::session_meta::read(path);
        for (f, input) in &self.inputs {
            let v = filled.get(f.key);
            input.update(cx, |i, cx| i.set_value(v, window, cx));
        }
        self.message = None;
        cx.notify();
    }

    /// What is filled in now (the fields the data does not say).
    fn filled(&self, cx: &Context<Self>) -> SessionMetadata {
        let mut meta = SessionMetadata::default();
        for (f, input) in &self.inputs {
            if self.from_data.get(f.key).is_empty() {
                meta.set(f.key, &input.read(cx).value());
            }
        }
        meta
    }

    fn save(&mut self, cx: &mut Context<Self>) {
        let Some(path) = self.path.clone() else { return };
        let meta = self.filled(cx);
        self.message = Some(match dspk_session::derived::session_meta::write(&path, &meta) {
            Ok(store) => (true, format!("Kept in {}", dspk_session::paths::short_path(&store))),
            Err(e) => (false, format!("Not kept: {e}")),
        });
        cx.notify();
    }

    /// Adds `name` to the experimenters typed in.
    fn add_experimenter(&mut self, name: &str, window: &mut Window, cx: &mut Context<Self>) {
        let Some((_, input)) = self.inputs.iter().find(|(f, _)| f.key == "experimenter") else { return };
        let mut meta = SessionMetadata::default();
        meta.set("experimenter", &input.read(cx).value());
        if !meta.experimenter.iter().any(|e| e == name) {
            meta.experimenter.push(name.to_string());
        }
        let line = meta.get("experimenter");
        input.update(cx, |i, cx| i.set_value(line, window, cx));
    }

    fn field_row(&self, f: &'static Field, input: &Entity<InputState>, cx: &mut Context<Self>) -> AnyElement {
        let muted = cx.theme().muted_foreground;
        let label = if f.required { format!("{} *", f.label) } else { f.label.to_string() };
        let known = self.from_data.get(f.key);
        if !known.is_empty() {
            let value = v_flex().w_full().child(div().text_xs().child(known)).child(div().text_xs().text_color(muted).child("from the data"));
            return row(label, value).into_any_element();
        }
        let control = v_flex().w_full().gap_1().child(Input::new(input).small()).when(f.key == "experimenter", |d| {
            let people = kitchen_shell::people::people();
            d.when(!people.is_empty(), |d| {
                d.child(h_flex().flex_wrap().gap_1().children(people.into_iter().enumerate().map(|(i, p)| {
                    let name = p.name.clone();
                    Button::new(SharedString::from(format!("session-add-person-{i}"))).ghost().small().label(format!("+ {}", p.name)).on_click(cx.listener(move |this, _, window, cx| this.add_experimenter(&name, window, cx)))
                })))
            })
        });
        row(label, control).into_any_element()
    }
}

impl Render for SessionInfoForm {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let muted = cx.theme().muted_foreground;
        let shown = self.from_data.over(&self.filled(cx));
        let missing = shown.missing_required().len();
        let summary = match missing {
            0 => "Session and subject (NWB): the required fields are there".to_string(),
            n => format!("Session and subject (NWB): {n} required field{} missing", if n == 1 { "" } else { "s" }),
        };
        let header = Button::new("session-info-toggle").ghost().small().label(format!("{} {summary}", if self.open { "▾" } else { "▸" })).on_click(cx.listener(|this, _, _, cx| {
            this.open = !this.open;
            cx.notify();
        }));
        let mut body = v_flex().gap_2();
        if self.open {
            let split = self.inputs.iter().position(|(f, _)| f.key == SUBJECT_FROM).unwrap_or(self.inputs.len());
            let rows: Vec<AnyElement> = self.inputs.clone().iter().map(|(f, input)| self.field_row(f, input, cx)).collect();
            let mut rows = rows.into_iter();
            let session: Vec<AnyElement> = rows.by_ref().take(split).collect();
            let subject: Vec<AnyElement> = rows.collect();
            let heading = |t: &'static str| div().text_xs().font_weight(FontWeight::SEMIBOLD).child(t);
            body = body
                .child(div().text_xs().text_color(muted).child("What the data does not say is kept in the recording's .dspk.zarr, never in the recording. * NWB requires it."))
                .child(v_flex().gap_1().child(heading("Session")).children(session))
                .child(v_flex().gap_1().pt_2().child(heading("Subject")).children(subject))
                .children(self.message.clone().map(|(ok, m)| div().text_xs().text_color(if ok { muted } else { cx.theme().danger }).child(m)))
                .child(h_flex().justify_end().child(Button::new("session-info-save").primary().small().label("Keep").on_click(cx.listener(|this, _, _, cx| this.save(cx)))));
        }
        v_flex().id("session-info").pt_2().gap_1().child(header).when(self.open, |d| d.child(div().id("session-info-body").max_h(px(420.)).overflow_y_scroll().child(body)))
    }
}
