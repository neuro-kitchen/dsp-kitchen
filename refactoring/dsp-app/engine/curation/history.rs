//! The edit history of a curation: every merge, split and label is a command that knows how to
//! undo itself (phy's `G`, `K`, `Alt/Ctrl+G/M/N/U`, `Ctrl+Z` / `Ctrl+Y`), with the selection before
//! and after it.

use std::collections::BTreeSet;

use super::{ClusterGroup, ClusterId, ClusterMeta, SortingData};

/// Reversible edit command on `SortingData`.
#[derive(Debug, Clone)]
pub enum CurationCommand {
    Merge {
        sources: Vec<ClusterId>,
        target: ClusterId,
        /// `(spike_index, old_cluster_id)` for every spike moved into `target`.
        prev_spikes: Vec<(usize, ClusterId)>,
        prev_metas: Vec<ClusterMeta>,
        new_meta: ClusterMeta,
        prev_selection: Vec<ClusterId>,
        next_selection: Vec<ClusterId>,
    },
    Split {
        source: ClusterId,
        created_inside: ClusterId,
        created_outside: ClusterId,
        /// `(spike_index, new_cluster_id)` for every spike originally in `source`.
        assigned_spikes: Vec<(usize, ClusterId)>,
        prev_meta: ClusterMeta,
        inside_meta: ClusterMeta,
        outside_meta: ClusterMeta,
        prev_selection: Vec<ClusterId>,
        next_selection: Vec<ClusterId>,
    },
    Label {
        changes: Vec<(ClusterId, ClusterGroup, ClusterGroup)>,
        prev_selection: Vec<ClusterId>,
        next_selection: Vec<ClusterId>,
    },
    CustomLabel {
        key: String,
        changes: Vec<(ClusterId, Option<String>, Option<String>)>,
    },
}

impl CurationCommand {
    pub fn touched_clusters(&self) -> Vec<ClusterId> {
        match self {
            Self::Merge { sources, target, .. } => {
                let mut v = sources.clone();
                v.push(*target);
                v
            }
            Self::Split { source, created_inside, created_outside, .. } => vec![*source, *created_inside, *created_outside],
            Self::Label { changes, .. } => changes.iter().map(|(id, _, _)| *id).collect(),
            Self::CustomLabel { changes, .. } => changes.iter().map(|(id, _, _)| *id).collect(),
        }
    }
}

impl SortingData {
    /// Merges `sources` (at least 2 clusters) into a single new cluster id (`G` in Phy).
    /// Returns `(new_cluster_id, touched_clusters)`.
    pub fn merge(&mut self, sources: &[ClusterId], prev_selection: &[ClusterId]) -> Option<(ClusterId, Vec<ClusterId>)> {
        let mut uniq: Vec<ClusterId> = sources.iter().copied().filter(|c| self.clusters.contains_key(c)).collect();
        uniq.sort_unstable();
        uniq.dedup();
        if uniq.len() < 2 {
            return None;
        }
        let target = self.next_cluster_id;
        self.next_cluster_id += 1;

        let source_set: BTreeSet<ClusterId> = uniq.iter().copied().collect();
        let mut prev_spikes = Vec::new();
        for (i, c) in self.sorting.spike_clusters.iter_mut().enumerate() {
            if source_set.contains(c) {
                prev_spikes.push((i, *c));
                *c = target;
            }
        }
        let prev_metas: Vec<ClusterMeta> = uniq.iter().filter_map(|c| self.clusters.remove(c)).collect();
        let new_meta = self.compute_meta_for(target, prev_metas.first());
        self.clusters.insert(target, new_meta.clone());

        let next_selection = vec![target];
        let cmd = CurationCommand::Merge {
            sources: uniq,
            target,
            prev_spikes,
            prev_metas,
            new_meta,
            prev_selection: prev_selection.to_vec(),
            next_selection,
        };
        let touched = cmd.touched_clusters();
        self.undo_stack.push(cmd);
        self.redo_stack.clear();
        Some((target, touched))
    }

    /// Splits `source` into two new clusters (`created_inside` for spikes in `inside_spike_indices`
    /// and `created_outside` for the remaining spikes of `source`, matching Phy's `K` split).
    pub fn split(
        &mut self,
        source: ClusterId,
        inside_spike_indices: &[usize],
        prev_selection: &[ClusterId],
    ) -> Option<(ClusterId, ClusterId, Vec<ClusterId>)> {
        let prev_meta = self.clusters.get(&source)?.clone();
        let inside_set: BTreeSet<usize> = inside_spike_indices.iter().copied().collect();
        let all_spikes = self.spike_indices(source);
        let inside_count = all_spikes.iter().filter(|i| inside_set.contains(i)).count();
        if inside_count == 0 || inside_count == all_spikes.len() {
            return None;
        }
        let created_inside = self.next_cluster_id;
        let created_outside = self.next_cluster_id + 1;
        self.next_cluster_id += 2;

        let mut assigned_spikes = Vec::with_capacity(all_spikes.len());
        for i in all_spikes {
            let next_c = if inside_set.contains(&i) { created_inside } else { created_outside };
            self.sorting.spike_clusters[i] = next_c;
            assigned_spikes.push((i, next_c));
        }
        self.clusters.remove(&source);
        let inside_meta = self.compute_meta_for(created_inside, Some(&prev_meta));
        let outside_meta = self.compute_meta_for(created_outside, Some(&prev_meta));
        self.clusters.insert(created_inside, inside_meta.clone());
        self.clusters.insert(created_outside, outside_meta.clone());

        let next_selection = vec![created_inside, created_outside];
        let cmd = CurationCommand::Split {
            source,
            created_inside,
            created_outside,
            assigned_spikes,
            prev_meta,
            inside_meta,
            outside_meta,
            prev_selection: prev_selection.to_vec(),
            next_selection,
        };
        let touched = cmd.touched_clusters();
        self.undo_stack.push(cmd);
        self.redo_stack.clear();
        Some((created_inside, created_outside, touched))
    }

    /// Sets the quality group (`good` / `mua` / `noise` / `unsorted`) on `targets`.
    pub fn set_group(
        &mut self,
        targets: &[ClusterId],
        group: ClusterGroup,
        prev_selection: &[ClusterId],
        next_selection: &[ClusterId],
    ) -> bool {
        let mut changes = Vec::new();
        for &cid in targets {
            if let Some(meta) = self.clusters.get_mut(&cid)
                && meta.group != group {
                    changes.push((cid, meta.group, group));
                    meta.group = group;
                }
        }
        if changes.is_empty() {
            return false;
        }
        self.undo_stack.push(CurationCommand::Label {
            changes,
            prev_selection: prev_selection.to_vec(),
            next_selection: next_selection.to_vec(),
        });
        self.redo_stack.clear();
        true
    }

    /// Sets a custom key=value label on `targets` (and registers `key` as a table column).
    pub fn set_custom_label(&mut self, targets: &[ClusterId], key: &str, value: &str) -> bool {
        let key = key.trim().to_string();
        if key.is_empty() {
            return false;
        }
        let val_opt = (!value.trim().is_empty()).then(|| value.trim().to_string());
        let mut changes = Vec::new();
        for &cid in targets {
            if let Some(meta) = self.clusters.get_mut(&cid) {
                let prev = meta.custom.get(&key).cloned();
                if prev != val_opt {
                    match &val_opt {
                        Some(v) => {
                            meta.custom.insert(key.clone(), v.clone());
                        }
                        None => {
                            meta.custom.remove(&key);
                        }
                    }
                    changes.push((cid, prev, val_opt.clone()));
                }
            }
        }
        if changes.is_empty() {
            return false;
        }
        if !self.custom_keys.contains(&key) {
            self.custom_keys.push(key.clone());
        }
        self.undo_stack.push(CurationCommand::CustomLabel { key, changes });
        self.redo_stack.clear();
        true
    }

    /// Undoes the last command. Returns `(touched_clusters, restored_selection)`.
    pub fn undo(&mut self) -> Option<(Vec<ClusterId>, Option<Vec<ClusterId>>)> {
        let cmd = self.undo_stack.pop()?;
        let touched = cmd.touched_clusters();
        let sel = match &cmd {
            CurationCommand::Merge { target, prev_spikes, prev_metas, prev_selection, .. } => {
                for &(i, old_c) in prev_spikes {
                    self.sorting.spike_clusters[i] = old_c;
                }
                self.clusters.remove(target);
                for m in prev_metas {
                    self.clusters.insert(m.id, m.clone());
                }
                Some(prev_selection.clone())
            }
            CurationCommand::Split { source, created_inside, created_outside, assigned_spikes, prev_meta, prev_selection, .. } => {
                for &(i, _) in assigned_spikes {
                    self.sorting.spike_clusters[i] = *source;
                }
                self.clusters.remove(created_inside);
                self.clusters.remove(created_outside);
                self.clusters.insert(*source, prev_meta.clone());
                Some(prev_selection.clone())
            }
            CurationCommand::Label { changes, prev_selection, .. } => {
                for (cid, prev_g, _) in changes {
                    if let Some(m) = self.clusters.get_mut(cid) {
                        m.group = *prev_g;
                    }
                }
                Some(prev_selection.clone())
            }
            CurationCommand::CustomLabel { key, changes } => {
                for (cid, prev_v, _) in changes {
                    if let Some(m) = self.clusters.get_mut(cid) {
                        match prev_v {
                            Some(v) => {
                                m.custom.insert(key.clone(), v.clone());
                            }
                            None => {
                                m.custom.remove(key);
                            }
                        }
                    }
                }
                None
            }
        };
        self.redo_stack.push(cmd);
        Some((touched, sel))
    }

    /// Redoes the last undone command. Returns `(touched_clusters, restored_selection)`.
    pub fn redo(&mut self) -> Option<(Vec<ClusterId>, Option<Vec<ClusterId>>)> {
        let cmd = self.redo_stack.pop()?;
        let touched = cmd.touched_clusters();
        let sel = match &cmd {
            CurationCommand::Merge { sources, target, prev_spikes, new_meta, next_selection, .. } => {
                for &(i, _) in prev_spikes {
                    self.sorting.spike_clusters[i] = *target;
                }
                for s in sources {
                    self.clusters.remove(s);
                }
                self.clusters.insert(*target, new_meta.clone());
                Some(next_selection.clone())
            }
            CurationCommand::Split { source, created_inside, created_outside, assigned_spikes, inside_meta, outside_meta, next_selection, .. } => {
                for &(i, new_c) in assigned_spikes {
                    self.sorting.spike_clusters[i] = new_c;
                }
                self.clusters.remove(source);
                self.clusters.insert(*created_inside, inside_meta.clone());
                self.clusters.insert(*created_outside, outside_meta.clone());
                Some(next_selection.clone())
            }
            CurationCommand::Label { changes, next_selection, .. } => {
                for (cid, _, next_g) in changes {
                    if let Some(m) = self.clusters.get_mut(cid) {
                        m.group = *next_g;
                    }
                }
                Some(next_selection.clone())
            }
            CurationCommand::CustomLabel { key, changes } => {
                for (cid, _, next_v) in changes {
                    if let Some(m) = self.clusters.get_mut(cid) {
                        match next_v {
                            Some(v) => {
                                m.custom.insert(key.clone(), v.clone());
                            }
                            None => {
                                m.custom.remove(key);
                            }
                        }
                    }
                }
                None
            }
        };
        self.undo_stack.push(cmd);
        Some((touched, sel))
    }
}
