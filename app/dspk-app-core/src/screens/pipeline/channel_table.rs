//! The channel table of a Remap node (Pipeline inspector): one row per output channel, in output
//! order, reading one input channel. Rows move up and down, leave (their input is dropped) and come
//! back; the whole map resets, reverses, imports from and exports to a map file (TOML or JSON, the
//! schema of `dspk_io::read_channel_map`). Every edit is one `SetNode` (undo works as for any op).

use std::path::PathBuf;

use dspk_core::{ChannelMap, ChannelMapFile, ChannelPair};
use dspk_pipeline::Node;
use gpui_kit::assets::IconName;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::menu::{DropdownMenu as _, PopupMenu, PopupMenuItem};
use gpui_kit::component::{h_flex, v_flex, Disableable as _, Sizable as _};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{div, px, AnyElement, App, Entity, Hsla, InteractiveElement as _, IntoElement, ParentElement as _, PathPromptOptions, SharedString, Styled as _};

use crate::data::store::Store;
use crate::domain::command::SessionCommand;
use crate::ui::widgets::icon_button;

/// A map being edited: the input (0-based) and the name of each output, in output order.
#[derive(Debug, Clone, PartialEq)]
pub struct Rows {
    pub base: usize,
    pub inputs: Vec<usize>,
    pub names: Vec<Option<String>>,
}

impl Rows {
    pub fn of(map: &ChannelMap) -> Result<Self, String> {
        Ok(Self { base: map.base, inputs: map.order().map_err(|e| e.to_string())?, names: map.names() })
    }

    /// Every input in order (what a new Remap starts from).
    pub fn identity(channels: usize) -> Self {
        Self { base: 0, inputs: (0..channels).collect(), names: vec![None; channels] }
    }

    pub fn map(&self) -> ChannelMap {
        let pairs = self
            .inputs
            .iter()
            .zip(&self.names)
            .enumerate()
            .map(|(to, (&from, name))| ChannelPair { from: from + self.base, to: to + self.base, name: name.clone() })
            .collect();
        ChannelMap { base: self.base, pairs }
    }

    pub fn node(&self) -> Node {
        let map = self.map();
        Node::Remap { base: map.base, pairs: map.pairs, path: None }
    }

    fn swap(mut self, a: usize, b: usize) -> Self {
        self.inputs.swap(a, b);
        self.names.swap(a, b);
        self
    }

    fn remove(mut self, row: usize) -> Self {
        self.inputs.remove(row);
        self.names.remove(row);
        self
    }

    fn push(mut self, input: usize) -> Self {
        self.inputs.push(input);
        self.names.push(None);
        self
    }

    fn reversed(mut self) -> Self {
        self.inputs.reverse();
        self.names.reverse();
        self
    }
}

/// The map's text for a file: TOML, or JSON when `path` ends in `.json`.
pub fn map_text(map: &ChannelMap, path: &std::path::Path) -> String {
    if path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("json")) {
        serde_json::to_string_pretty(&ChannelMapFile { remap: map.clone() }).expect("a channel map serialises") + "\n"
    } else {
        dspk_io::channel_map_toml(map)
    }
}

/// The editor for Remap node `id` holding `rows`, over an input of `channels` channels (`None`:
/// unknown, the recording not loaded), `label(c)` naming input channel `c`.
pub fn editor(store: &Entity<Store>, id: &str, rows: Rows, channels: Option<usize>, label: impl Fn(usize) -> Option<String>, muted: Hsla) -> AnyElement {
    let set = {
        let (store, id) = (store.clone(), id.to_string());
        move |rows: Rows, cx: &mut App| {
            let _ = store.update(cx, |s, cx| s.execute(SessionCommand::SetNode { id: id.clone(), node: rows.node() }, cx));
        }
    };
    let n = rows.inputs.len();
    let dropped: Vec<usize> = channels.map(|c| (0..c).filter(|i| !rows.inputs.contains(i)).collect()).unwrap_or_default();
    let missing: Vec<usize> = channels.map(|c| rows.inputs.iter().copied().filter(|&i| i >= c).collect()).unwrap_or_default();

    let table = v_flex().gap_0p5().children(rows.inputs.iter().enumerate().map(|(row, &input)| {
        let name = rows.names[row].clone().or_else(|| label(input));
        let mv = |dir: isize, icon: IconName, tip: &'static str| {
            let to = row as isize + dir;
            let (set, r) = (set.clone(), rows.clone());
            icon_button(SharedString::from(format!("remap-{row}-{tip}")), icon, tip).xsmall().disabled(to < 0 || to >= n as isize).on_click(move |_, _, cx| set(r.clone().swap(row, to as usize), cx))
        };
        let (set_rm, r_rm) = (set.clone(), rows.clone());
        h_flex()
            .id(("remap-row", row))
            .h(px(24.))
            .gap_1()
            .child(div().w(px(36.)).text_xs().text_color(muted).child(format!("{}", row + rows.base)))
            .child(div().text_xs().text_color(muted).child("←"))
            .child(div().w(px(40.)).text_xs().when(missing.contains(&input), |d| d.text_color(gpui_kit::red())).child(format!("{}", input + rows.base)))
            .child(div().flex_1().min_w_0().truncate().text_xs().text_color(muted).child(name.unwrap_or_default()))
            .child(mv(-1, IconName::ArrowUp, "Earlier"))
            .child(mv(1, IconName::ArrowDown, "Later"))
            .child(icon_button(SharedString::from(format!("remap-{row}-drop")), IconName::Close, "Drop this channel").xsmall().disabled(n <= 1).on_click(move |_, _, cx| set_rm(r_rm.clone().remove(row), cx)))
    }));

    let n_dropped = dropped.len();
    let add_back = (!dropped.is_empty()).then(|| {
        let (set, r) = (set.clone(), rows.clone());
        Button::new("remap-add-back").outline().xsmall().icon(IconName::Plus).label(format!("Add back ({})", dropped.len())).dropdown_menu(move |m: PopupMenu, _, _| {
            let mut m = m.scrollable(true).max_h(px(360.));
            for &c in &dropped {
                let (set, r) = (set.clone(), r.clone());
                m = m.item(PopupMenuItem::new(format!("Input {}", c + r.base)).on_click(move |_, _, cx| set(r.clone().push(c), cx)));
            }
            m
        })
    });
    let reset = channels.map(|c| {
        let set = set.clone();
        Button::new("remap-reset").ghost().xsmall().label("All in order").tooltip("Every input channel, in its own order").on_click(move |_, _, cx| set(Rows::identity(c), cx))
    });
    let reverse = {
        let (set, r) = (set.clone(), rows.clone());
        Button::new("remap-reverse").ghost().xsmall().label("Reverse").on_click(move |_, _, cx| set(r.clone().reversed(), cx))
    };
    let import = {
        let set = set.clone();
        let store = store.clone();
        Button::new("remap-import").outline().xsmall().icon(IconName::FolderOpen).label("Import…").tooltip("A map file (TOML or JSON: [remap] base, pairs)").on_click(move |_, _, cx| {
            let rx = cx.prompt_for_paths(PathPromptOptions { files: true, directories: false, multiple: false, prompt: Some("Import a channel map".into()) });
            let (set, store) = (set.clone(), store.clone());
            cx.spawn(async move |cx| {
                let Ok(Ok(Some(paths))) = rx.await else { return };
                let Some(path) = paths.into_iter().next() else { return };
                let _ = cx.update(|cx| match dspk_io::read_channel_map(&path).map_err(|e| e.to_string()).and_then(|m| Rows::of(&m)) {
                    Ok(rows) => set(rows, cx),
                    Err(e) => store.update(cx, |s, cx| s.set_status(format!("{}: {e}", path.display()), cx)),
                });
            })
            .detach();
        })
    };
    let export = {
        let (map, store) = (rows.map(), store.clone());
        Button::new("remap-export").outline().xsmall().icon(IconName::ArrowDown).label("Export…").tooltip("Save the map (.toml, or .json)").on_click(move |_, _, cx| {
            let dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let rx = cx.prompt_for_new_path(&dir, Some("channel-map.toml"));
            let (map, store) = (map.clone(), store.clone());
            cx.spawn(async move |cx| {
                let Ok(Ok(Some(path))) = rx.await else { return };
                let result = std::fs::write(&path, map_text(&map, &path));
                let _ = cx.update(|cx| {
                    let text = match result {
                        Ok(()) => format!("Channel map written to {}", path.display()),
                        Err(e) => format!("{}: {e}", path.display()),
                    };
                    store.update(cx, |s, cx| s.set_status(text, cx));
                });
            })
            .detach();
        })
    };

    v_flex()
        .id("remap-editor")
        .gap_2()
        .child(div().text_xs().text_color(muted).child(match channels {
            Some(c) => format!("{n} output channel{} from {c} input{} ({} dropped). Output ← input · name.", if n == 1 { "" } else { "s" }, if c == 1 { "" } else { "s" }, n_dropped),
            None => format!("{n} output channels. Open the recording to check the inputs."),
        }))
        .when(!missing.is_empty(), |d| d.child(div().text_xs().text_color(gpui_kit::red()).child(format!("{} input(s) do not exist in this recording.", missing.len()))))
        .child(h_flex().gap_1().flex_wrap().children(add_back).children(reset).child(reverse))
        .child(div().max_h(px(420.)).overflow_hidden().child(table))
        .child(h_flex().gap_1().child(import).child(export))
        .into_any_element()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_round_trip_through_the_map_with_their_base() {
        let map = ChannelMap { base: 1, pairs: vec![ChannelPair { from: 3, to: 1, name: Some("A1".into()) }, ChannelPair { from: 1, to: 2, name: None }] };
        let rows = Rows::of(&map).unwrap();
        assert_eq!(rows.inputs, vec![2, 0]);
        assert_eq!(rows.map(), map);
        let moved = rows.clone().swap(0, 1);
        assert_eq!(moved.map().order().unwrap(), vec![0, 2]);
        assert_eq!(moved.names, vec![None, Some("A1".into())], "names move with their channel");
        assert_eq!(rows.clone().remove(0).push(5).map().order().unwrap(), vec![0, 5]);
        assert!(Rows::identity(4).node().stage().is_ok());
    }

    #[test]
    fn map_text_follows_the_extension() {
        let map = Rows::identity(2).map();
        for name in ["m.toml", "m.json"] {
            let path = std::path::Path::new(name);
            let parsed = dspk_io::parse_channel_map(&map_text(&map, path), name.ends_with(".toml")).unwrap();
            assert_eq!(parsed, map, "{name}");
        }
    }
}
