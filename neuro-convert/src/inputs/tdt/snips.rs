//! Snip stores → [`SnippetSeries`]: each record points at one waveform in the TEV.
//! Sort codes come from the TSQ; `sort/<name>/*.SortResult` overlays are not applied yet.

use super::codes;
use super::tsq::{session_time, StoreIndex};
use crate::common::codec::decode_into;
use crate::common::mapped::MappedFile;
use crate::error::{Error, Result};
use crate::model::SnippetSeries;

pub fn build(store: &StoreIndex, tev: &MappedFile, block_start: f64, warnings: &mut Vec<String>) -> Result<SnippetSeries> {
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
        s.sort_codes.push(r.sortcode);
    }
    if skipped > 0 {
        warnings.push(format!("{}: {skipped} snippets were outside the TEV or had a different size", store.name));
    }
    Ok(s)
}
