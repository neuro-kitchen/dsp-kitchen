//! The TDT reader against TDT's own Python reader (`tdt` 0.7.6, `read_block`) on a real
//! Synapse block. Skipped when the block is not on disk (it is not committed: 19 GB).

use std::path::Path;

use neuro_convert::OpenOptions;

const BLOCK: &str = "../playground/data/15-25-33_meps";

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn matches_tdt_python_reader() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(BLOCK);
    if !path.exists() {
        eprintln!("skipped: {} not found", path.display());
        return;
    }
    let s = neuro_convert::open(&path, &OpenOptions::default()).unwrap();
    assert_eq!(s.provenance.version.as_deref(), Some("Synapse 53575"));
    assert!(s.provenance.warnings.is_empty(), "{:?}", s.provenance.warnings);

    // read_block(..., store='HDEG', t1=100, t2=100.001): first sample = ceil(100 * fs)
    let expect_stream = |name: &str, want: &[f32]| {
        let r = s.recording(name).unwrap();
        let first = (100.0 * r.info().sample_rate).ceil() as u64;
        let mut out = vec![0.0f32; want.len()];
        r.read(&[0], first..first + want.len() as u64, &mut out).unwrap();
        for (i, (&a, &b)) in out.iter().zip(want).enumerate() {
            assert!(close(a as f64, b as f64, 1e-7 + b.abs() as f64 * 1e-5), "{name}[{i}]: {a} vs {b}");
        }
    };
    expect_stream("HDEG", &[-0.000_420_93, -0.000_429_12, -0.000_431_81, -0.000_421_25, -0.000_413_57]);
    expect_stream("MonA", &[0.0; 5]);
    expect_stream("Teme", &[36.142_517, 36.130_31]);

    let met = s.event_series("MET/").unwrap();
    let offsets = met.offsets.as_ref().unwrap();
    for (i, (on, off)) in [(0.588_267_52, 0.588_308_48), (1.972_019_2, 1.972_060_16), (3.361_873_92, 3.361_914_88)].iter().enumerate() {
        assert!(close(met.onsets[i], *on, 1e-8), "onset {i}: {}", met.onsets[i]);
        assert!(close(offsets[i], *off, 1e-8), "offset {i}: {}", offsets[i]);
    }

    let stim = s.event_series("eS1p").unwrap();
    assert_eq!((stim.channels, stim.len()), (24, 800));
    for (i, want) in [0.491_519_99, 1.0, -750.0, 0.400_000_006].iter().enumerate() {
        assert!(close(stim.values[i], *want, 1e-6), "eS1p[{i}] = {}", stim.values[i]);
    }
    assert!(close(stim.onsets[0], 454.239_395_84, 1e-8));
    assert!(close(stim.onsets[1], 455.648_788_48, 1e-8));
}
