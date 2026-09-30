//! Snip stores → [`SnippetSeries`]: each record points at one waveform in the TEV.
//! Sort codes come from the TSQ, or from an offline sort (`sort::load`) indexed by record.

use super::codes;
use super::tsq::{session_time, StoreIndex};
use crate::common::codec::decode_into;
use crate::common::mapped::MappedFile;
use crate::error::{Error, Result};
use crate::model::SnippetSeries;

pub fn build(store: &StoreIndex, tev: &MappedFile, block_start: f64, sort: Option<(&str, &[u8])>, warnings: &mut Vec<String>) -> Result<SnippetSeries> {
    let ty = codes::sample_type(store.format)
        .ok_or_else(|| Error::Unsupported(format!("TDT snip store {}: data format code {}", store.name, store.format)))?;
    let points = (store.packet_bytes / ty.bytes() as u64) as usize;
    let mut s = SnippetSeries {
        name: store.name.clone(),
        description: "TDT snippet store".into(),
        sample_rate: store.frequency,
        samples_per_snippet: points,
        unit: if matches!(ty, crate::model::SampleType::F32 | crate::model::SampleType::F64) { "V".into() } else { "a.u.".into() },
        ..Default::default()
    };
    let bytes = tev.bytes();
    let mut skipped = 0;
    let mut unsorted = 0;
    let mut wave = vec![0.0f32; points];
    for r in &store.records {
        let (at, len) = (r.offset as usize, points * ty.bytes());
        if at + len > bytes.len() || r.data_bytes() as usize != len {
            skipped += 1;
            continue;
        }
        decode_into(ty, &bytes[at..at + len], &mut wave, 1.0, 0.0);
        s.data.extend_from_slice(&wave);
        s.timestamps.push(session_time(r.timestamp, block_start));
        s.channels.push(r.chan);
        s.sort_codes.push(match sort {
            Some((_, codes)) => codes.get(r.seq as usize).map_or_else(
                || {
                    unsorted += 1;
                    r.sortcode
                },
                |&c| c as u16,
            ),
            None => r.sortcode,
        });
    }
    if let Some((id, _)) = sort {
        s.description = format!("TDT snippet store, sort codes from offline sort {id:?}");
        if unsorted > 0 {
            warnings.push(format!("{}: sort {id:?} has no code for {unsorted} snippets (online codes kept)", store.name));
        }
    }
    if skipped > 0 {
        warnings.push(format!("{}: {skipped} snippets were outside the TEV or had a different size", store.name));
    }
    Ok(s)
}
