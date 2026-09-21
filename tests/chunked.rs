//! A chunked, deflated and shuffled four-dimensional dataset, as netCDF4
//! writes a `MultiSpeakerBRIR` (`Data.IR` is `[M][R][E][N]`).
//!
//! The fixture is 6 × 2 × 3 × 1000 doubles in chunks of 1 × 1 × 1 × 300:
//! 144 chunks, so the version 1 B-tree has an inner node, and the last chunk
//! of every row is an edge chunk. Every element is its own row-major index,
//! so a wrong stride, offset, unshuffle or edge handling shows up as a value.

use sofar::reader::OpenOptions;

fn fixture() -> Vec<u8> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/chunked_multispeaker_brir.sofa"
    );
    std::fs::read(path).expect("read fixture")
}

#[test]
fn chunked_four_dimensional_dataset_reads_every_element() {
    let sofa = OpenOptions::new()
        .sample_rate(48000.0)
        .normalized(false)
        .open_data(fixture())
        .expect("open");
    let h = sofa.hrtf();
    let (m, r, e, n) = (6usize, 2usize, 3usize, 1000usize);
    assert_eq!(h.data_ir.values.len(), m * r * e * n);
    for (i, &v) in h.data_ir.values.iter().enumerate() {
        assert_eq!(v, i as f32, "element {i}");
    }
    // The small chunked datasets came through the same reader.
    assert_eq!(h.emitter_position.values.len(), e * 3);
    assert_eq!(h.listener_view.values.len(), m * 3);
    assert_eq!(h.data_delay.values.len(), r * e);
    assert_eq!(h.data_sampling_rate.values, vec![48000.0]);
}
