//! String attributes kept in a fractal heap whose datatype says UTF-8, as
//! the SOFA C++ API writes them (the IRCAM LISTEN sets among others), or
//! null- or space-padded: the string class bit field holds the padding type
//! and the character set, and only its upper 16 bits are reserved. Both
//! record forms read it: `Title` here is a name-value pair, and the IRCAM
//! sets' attributes are the longer form.
//!
//! The fixture is libmysofa's `tests/tester2.sofa` (CC BY 4.0, derived from
//! `Pulse.sofa` by Piotr Majdak, ARI Vienna), whose `Title` is such an
//! attribute written as null-terminated ASCII; the tests rewrite its bit
//! field in memory.

fn fixture() -> Vec<u8> {
    let path = format!("{}/tests/data/tester2.sofa", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(path).unwrap()
}

/// Where `Title`'s bit field sits: after the name, its NUL padding and the
/// 0x13 of the string class.
fn bit_field_at(data: &[u8]) -> usize {
    let name = data
        .windows(6)
        .position(|w| w == b"Title\0")
        .expect("Title attribute in the fixture");
    let class = name + data[name..].iter().position(|&b| b == 0x13).unwrap();
    assert_eq!(
        &data[class + 1..class + 4],
        &[0, 0, 0],
        "ASCII, null-terminated"
    );
    class + 1
}

fn title(data: &[u8]) -> Result<Option<String>, String> {
    let parsed = sofar::hdf::parse_with_children(data).map_err(|e| format!("{e:?}"))?;
    Ok(parsed
        .root
        .parsed_attributes
        .iter()
        .find(|a| a.name == "Title")
        .and_then(|a| a.value.clone()))
}

fn with_bit_field(bytes: [u8; 3]) -> Vec<u8> {
    let mut data = fixture();
    let at = bit_field_at(&data);
    data[at..at + 3].copy_from_slice(&bytes);
    data
}

#[test]
fn a_utf8_or_padded_string_attribute_is_read() {
    let ascii = title(&fixture()).unwrap();
    assert!(ascii.is_some(), "the fixture's Title is read");
    // UTF-8; null-padded; space-padded; UTF-8 and space-padded.
    for bits in [0x10, 0x01, 0x02, 0x12] {
        assert_eq!(
            title(&with_bit_field([bits, 0, 0])).unwrap(),
            ascii,
            "{bits:#x}"
        );
    }
}

#[test]
fn the_reserved_bits_are_still_checked() {
    for bytes in [
        [0x00, 0x01, 0x00],
        [0x00, 0x00, 0x80],
        [0x03, 0, 0],
        [0x20, 0, 0],
    ] {
        assert!(title(&with_bit_field(bytes)).is_err(), "{bytes:02x?}");
    }
}
