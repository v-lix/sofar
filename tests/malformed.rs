//! Corrupt files are refused with an error, in debug builds too.
//!
//! The parser used to report malformed input through `ErrMode::assert`,
//! which panics when `debug_assertions` are on: the libmysofa crash corpus
//! below, and damaged copies of the valid fixtures, crashed every debug
//! build reading them.

use sofar::reader::OpenOptions;

const CORPUS_DIR: &str = "libmysofa-sys/libmysofa/tests";

fn manifest(path: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path)
}

/// Parsing `bytes` returns, whatever they hold.
fn parse(bytes: &[u8]) {
    let _ = OpenOptions::new().sample_rate(48000.0).open_data(bytes);
    let _ = sofar::hdf::parse_with_children(bytes).map(|hdf| {
        for name in ["Data.IR", "Data.SamplingRate", "SourcePosition"] {
            let _ = hdf.get_child(name);
        }
    });
}

#[test]
fn the_libmysofa_crash_corpus_is_refused_without_panicking() {
    let mut files: Vec<_> = std::fs::read_dir(manifest(CORPUS_DIR))
        .expect("the libmysofa submodule is checked out")
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("fail-issue-"))
        })
        .collect();
    files.sort();
    assert!(!files.is_empty());
    for file in files {
        println!("{}", file.display());
        parse(&std::fs::read(&file).unwrap());
    }
}

#[test]
fn damaged_copies_of_the_fixtures_are_refused_without_panicking() {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    for fixture in [
        "tests/data/sofasonix_netcdf4.sofa",
        "tests/data/chunked_multispeaker_brir.sofa",
    ] {
        let bytes = std::fs::read(manifest(fixture)).unwrap();
        for case in 0..100usize {
            let mut damaged = bytes.clone();
            if case < 50 {
                damaged.truncate(bytes.len() * case / 50);
            } else {
                for _ in 0..=case % 8 {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let at = (seed % damaged.len() as u64) as usize;
                    damaged[at] = (seed >> 32) as u8;
                }
            }
            parse(&damaged);
        }
    }
}
