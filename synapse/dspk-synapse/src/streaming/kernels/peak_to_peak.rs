use cubecl::prelude::*;
use dspk_core::compute::element_position;
use dspk_core::compute::LaunchGeometry;

/// Peak-to-peak height of every row of `[rows, snippet_len]` snippets (`rows` = spikes × K
/// neighbours): one unit per row.
#[cube(launch)]
pub fn snippet_peak_to_peak_kernel(snippets: &[f32], out: &mut [f32], rows: u32, snippet_len: u32) {
    let row = element_position() as u32;
    if row < rows {
        let base = row * snippet_len;
        let mut lo = snippets[base as usize];
        let mut hi = lo;
        let mut t = 1u32;
        while t < snippet_len {
            let v = snippets[(base + t) as usize];
            lo = f32::min(lo, v);
            hi = f32::max(hi, v);
            t += 1u32;
        }
        out[row as usize] = hi - lo;
    }
}

/// Peak-to-peak height per `(spike, neighbour)` of extracted `[spikes, k_neighbors, snippet_len]`
/// snippets in VRAM, downloaded (`spikes × k_neighbors` values: what localisation needs, not the
/// waveforms).
pub fn execute_snippet_peak_to_peak(client: &Client, snippets: &cubecl::server::Handle, spikes: usize, k_neighbors: usize, snippet_len: usize) -> Vec<f32> {
    let rows = spikes * k_neighbors;
    if rows == 0 || snippet_len == 0 {
        return Vec::new();
    }
    let out = client.empty(rows * std::mem::size_of::<f32>());
    let geom = LaunchGeometry::elementwise(client, rows);
    unsafe {
        snippet_peak_to_peak_kernel::launch(
            client,
            geom.cube_count,
            geom.cube_dim,
            BufferArg::from_raw_parts(snippets.clone(), rows * snippet_len),
            BufferArg::from_raw_parts(out.clone(), rows),
            rows as u32,
            snippet_len as u32,
        );
    }
    f32::from_bytes(&client.read_one(out).expect("VRAM read peak-to-peak")).to_vec()
}
