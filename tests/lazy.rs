//! Reading a few measurements of `Data.IR` without the rest.
//!
//! Three `MultiSpeakerBRIR` fixtures written by netCDF4, each `Data.IR`
//! element its own row-major index:
//!
//! - `chunked_multispeaker_brir.sofa`: 6 × 2 × 3 × 1000 in 1 × 1 × 1 × 300
//!   chunks, deflated and shuffled (one measurement per chunk, an inner
//!   B-tree node, edge chunks);
//! - `rows_multispeaker_brir.sofa`: 7 × 2 × 3 × 50 in 3 × 1 × 2 × 20 chunks,
//!   deflated and shuffled, so a chunk holds several measurements and the
//!   last chunk of every dimension is an edge chunk; `Data.Delay` is
//!   `[M][R][E]` and `ListenerView` turns 10° per measurement;
//! - `contiguous_multispeaker_brir.sofa`: 5 × 2 × 2 × 40, contiguous and
//!   unfiltered.

use sofar::reader::{LazySofa, OpenOptions};

fn fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(path).expect("read fixture")
}

const FIXTURES: [(&str, [usize; 4]); 3] = [
    ("chunked_multispeaker_brir.sofa", [6, 2, 3, 1000]),
    ("rows_multispeaker_brir.sofa", [7, 2, 3, 50]),
    ("contiguous_multispeaker_brir.sofa", [5, 2, 2, 40]),
];

/// Every run of measurements reads as the same values the whole set holds.
#[test]
fn any_run_of_measurements_reads_as_the_whole_set_holds_it() {
    for (name, shape) in FIXTURES {
        let bytes = fixture(name);
        let lazy = LazySofa::open(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(lazy.ir_shape(), shape, "{name}");
        let row = shape[1] * shape[2] * shape[3];
        for first in 0..shape[0] {
            for count in 1..=shape[0] - first {
                let values = lazy.read_ir(first, count).expect("reads");
                assert_eq!(values.len(), count * row, "{name} {first}+{count}");
                for (i, &v) in values.iter().enumerate() {
                    assert_eq!(v, (first * row + i) as f32, "{name} {first}+{count} [{i}]");
                }
            }
        }
    }
}

/// Opened lazily, a file holds what `open_data` reads of it but the
/// responses, in the same coordinates.
#[test]
fn a_lazy_open_reads_everything_but_the_responses() {
    for (name, _) in FIXTURES {
        let bytes = fixture(name);
        let whole = OpenOptions::new()
            .sample_rate(48000.0)
            .normalized(false)
            .open_data(&bytes)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let lazy = LazySofa::open(&bytes).unwrap();
        let (w, l) = (whole.hrtf(), lazy.hrtf());
        assert!(l.data_ir.values.is_empty(), "{name}");
        for (what, a, b) in [
            (
                "ListenerPosition",
                &w.listener_position,
                &l.listener_position,
            ),
            (
                "ReceiverPosition",
                &w.receiver_position,
                &l.receiver_position,
            ),
            ("SourcePosition", &w.source_position, &l.source_position),
            ("EmitterPosition", &w.emitter_position, &l.emitter_position),
            ("ListenerUp", &w.listener_up, &l.listener_up),
            ("ListenerView", &w.listener_view, &l.listener_view),
            (
                "Data.SamplingRate",
                &w.data_sampling_rate,
                &l.data_sampling_rate,
            ),
            ("Data.Delay", &w.data_delay, &l.data_delay),
        ] {
            assert!(!a.values.is_empty(), "{name} {what}");
            assert_eq!(a.values, b.values, "{name} {what}");
            assert_eq!(a.attributes, b.attributes, "{name} {what}");
        }
        assert_eq!(w.get_attribute("SOFAConventions"), Some("MultiSpeakerBRIR"));
        assert_eq!(
            w.get_attribute("SOFAConventions"),
            l.get_attribute("SOFAConventions")
        );
    }
}

#[test]
fn measurements_beyond_the_set_are_refused() {
    for (name, shape) in FIXTURES {
        let bytes = fixture(name);
        let lazy = LazySofa::open(&bytes).unwrap();
        let m = shape[0];
        assert!(lazy.read_ir(m, 1).is_err(), "{name}");
        assert!(lazy.read_ir(m - 1, 2).is_err(), "{name}");
        assert!(lazy.read_ir(usize::MAX, 2).is_err(), "{name}");
        assert_eq!(
            lazy.read_ir(0, 0).expect("reads nothing"),
            Vec::<f32>::new()
        );
    }
}

/// Byte range of the twelve chunks of `rows_multispeaker_brir.sofa` that
/// hold measurements 3 to 5 (h5py `get_chunk_info`: offsets 0xe553 to
/// 0xea8b, the last 88 bytes long).
const ROWS_3_TO_5: std::ops::Range<usize> = 0xe553..0xeae3;

/// The chunks that hold none of the measurements read are not read: with
/// those of measurements 3 to 5 destroyed, the others still read, and any
/// run that reaches 3 to 5 is refused.
#[test]
fn only_the_chunks_holding_the_measurements_are_read() {
    let mut bytes = fixture("rows_multispeaker_brir.sofa");
    bytes[ROWS_3_TO_5].fill(0xff);
    let lazy = LazySofa::open(&bytes).expect("the header is intact");
    let row = 2 * 3 * 50;
    for (first, count) in [(0, 3), (6, 1), (2, 1)] {
        let values = lazy.read_ir(first, count).expect("reads");
        assert_eq!(values.len(), count * row);
        assert!(
            values
                .iter()
                .enumerate()
                .all(|(i, &v)| v == (first * row + i) as f32)
        );
    }
    for (first, count) in [(3, 1), (5, 1), (2, 2), (0, 7)] {
        assert!(lazy.read_ir(first, count).is_err(), "{first}+{count}");
    }
}

/// Damaged copies open lazily and read, or are refused: never a panic.
#[test]
fn damaged_copies_read_lazily_or_are_refused_without_panicking() {
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    for (name, shape) in FIXTURES {
        let bytes = fixture(name);
        for case in 0..60usize {
            let mut damaged = bytes.clone();
            if case < 30 {
                damaged.truncate(bytes.len() * case / 30);
            } else {
                for _ in 0..=case % 8 {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let at = (seed % damaged.len() as u64) as usize;
                    damaged[at] = (seed >> 32) as u8;
                }
            }
            if let Ok(lazy) = LazySofa::open(&damaged) {
                let _ = lazy.ir_shape();
                for first in 0..shape[0] {
                    let _ = lazy.read_ir(first, 1);
                }
                let _ = lazy.read_ir(0, shape[0]);
            }
        }
    }
}
