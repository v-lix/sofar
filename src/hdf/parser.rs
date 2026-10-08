use winnow::prelude::*;
use winnow::stream::{LocatingSlice, Location, Stateful, Stream};
use winnow::token::take;

use super::btree::tree;
use super::data_object::{DataObject, DataSpace, MAX_DATASET_BYTES, Storage, data_object};
use super::super_block::{SuperBlock, super_block};

pub(crate) type Input<'a> = Stateful<LocatingSlice<&'a [u8]>, State>;

/// Context state that is available after parsing Super Block
#[derive(Debug, Clone)]
pub(crate) struct State {
    size_of_lengths: u8,
    size_of_offsets: u8,
    end_of_file_address: u64,
    recursive_counter: u32,
    data_space: Option<DataSpace>,
    /// Filter identifiers of the last filter pipeline message, in pipeline
    /// order; empty when the object declared none.
    filters: Vec<u16>,
    /// Whether a dataset's elements are read along with its header.
    read_data: bool,
}

impl State {
    pub fn new(block: &SuperBlock) -> Self {
        Self {
            size_of_lengths: block.size_of_lengths,
            size_of_offsets: block.size_of_offsets,
            end_of_file_address: block.end_of_file_address,
            recursive_counter: 0,
            data_space: None,
            filters: Vec::new(),
            read_data: true,
        }
    }

    pub(crate) fn read_data(&self) -> bool {
        self.read_data
    }

    pub fn size_of_lengths(&self) -> u8 {
        self.size_of_lengths
    }

    pub fn size_of_offsets(&self) -> u8 {
        self.size_of_offsets
    }

    #[allow(dead_code)]
    pub fn end_of_file_address(&self) -> u64 {
        self.end_of_file_address
    }

    pub fn is_address_valid(&self, address: u64) -> bool {
        address > 0 && address < self.end_of_file_address
    }

    pub(crate) fn data_space(&self) -> Option<DataSpace> {
        self.data_space.clone()
    }

    pub(crate) fn set_data_space(&mut self, data_space: DataSpace) {
        self.data_space = Some(data_space);
    }

    pub(crate) fn filters(&self) -> Vec<u16> {
        self.filters.clone()
    }

    pub(crate) fn set_filters(&mut self, filters: Vec<u16>) {
        self.filters = filters;
    }

    pub(crate) fn recursive_counter(&self) -> u32 {
        self.recursive_counter
    }

    pub(crate) fn recursive_counter_inc(&mut self) {
        self.recursive_counter = self.recursive_counter.saturating_add(1);
    }

    pub(crate) fn recursive_counter_dec(&mut self) {
        self.recursive_counter = self.recursive_counter.saturating_sub(1);
    }
}

/// Parsed HDF5 file with ability to navigate to child objects.
pub struct ParsedHdf<'a> {
    data: &'a [u8],
    state: State,
    /// The root data object
    pub root: DataObject,
}

impl<'a> ParsedHdf<'a> {
    /// Parse a child data object by address.
    ///
    /// Use addresses from `root.child_directories` to navigate the tree.
    pub fn parse_child(&self, name: &str, address: u64) -> ModalResult<DataObject> {
        self.parse_object(name, address, true)
    }

    /// Find a child by name and parse it.
    pub fn get_child(&self, name: &str) -> Option<ModalResult<DataObject>> {
        self.root
            .child_directories
            .iter()
            .find(|d| d.name == name)
            .map(|d| self.parse_child(&d.name, d.address))
    }

    /// Find a child by name and parse its header only: shape, type and
    /// attributes as [`Self::get_child`] parses them, `data` left empty.
    /// [`Self::read_rows`] then reads the part of the elements a caller
    /// needs, so a dataset of hundreds of MB costs nothing until then.
    pub fn get_child_header(&self, name: &str) -> Option<ModalResult<DataObject>> {
        self.root
            .child_directories
            .iter()
            .find(|d| d.name == name)
            .map(|d| self.parse_object(&d.name, d.address, false))
    }

    /// Rows `first..first + count` of the first dimension of `object`, a
    /// dataset of this file parsed by [`Self::get_child_header`] or
    /// [`Self::get_child`]: the elements as stored, row-major, in the
    /// file's element size. Only the storage holding those rows is read; a
    /// chunked dataset's other chunks are neither read nor unfiltered.
    ///
    /// # Errors
    ///
    /// The rows lie beyond the dataset, the dataset has no storage to read
    /// (no data written, or a compact layout), or the storage is damaged.
    pub fn read_rows(&self, object: &DataObject, first: u64, count: u64) -> ModalResult<Vec<u8>> {
        let mut stream = Input {
            input: LocatingSlice::new(self.data),
            state: self.state.clone(),
        };
        let invalid = |stream: &Input, why| Err(crate::hdf::helpers::invalid(stream, why));
        let dims = object.ds.dimensionality as usize;
        if dims == 0 || dims > object.ds.dimension_size.len() {
            return invalid(&stream, "read_rows: not an array");
        }
        let rows = object.ds.dimension_size[0];
        if first.checked_add(count).is_none_or(|hi| hi > rows) {
            return invalid(&stream, "read_rows: rows beyond the dataset");
        }
        match &object.storage {
            None => invalid(&stream, "read_rows: no storage to read"),
            Some(Storage::Contiguous { address, size }) => {
                // Elements per row, and the element size the type declares.
                let row = object.ds.dimension_size[1..dims]
                    .iter()
                    .try_fold(u64::from(object.dt.size), |acc, &d| acc.checked_mul(d));
                let span = row.and_then(|row| {
                    let all = row.checked_mul(rows)?;
                    let at = address.checked_add(first.checked_mul(row)?)?;
                    let len = count.checked_mul(row)?;
                    if all > *size || len > MAX_DATASET_BYTES {
                        return None;
                    }
                    Some((usize::try_from(at).ok()?, usize::try_from(len).ok()?))
                });
                let Some((at, len)) = span else {
                    return invalid(&stream, "read_rows: contiguous size mismatch");
                };
                let _skip = take(at).parse_next(&mut stream)?;
                let bytes = take(len).parse_next(&mut stream)?;
                Ok(bytes.to_vec())
            }
            Some(Storage::Chunked {
                address,
                layout,
                space,
                filters,
            }) => {
                let Ok(at) = usize::try_from(*address) else {
                    return invalid(&stream, "read_rows: chunk index beyond the file");
                };
                let _skip = take(at).parse_next(&mut stream)?;
                tree(space.clone(), layout.clone(), filters.clone(), first, count)
                    .parse_next(&mut stream)
            }
        }
    }

    /// Parse the object at `address`, with or without its data.
    fn parse_object(&self, name: &str, address: u64, read_data: bool) -> ModalResult<DataObject> {
        if !self.state.is_address_valid(address) {
            return Err(crate::hdf::helpers::invalid(
                &self.data,
                "Invalid child object address",
            ));
        }

        let input = LocatingSlice::new(self.data);
        let mut state = self.state.clone();
        state.read_data = read_data;
        let mut stream = Input { input, state };

        let _skip = take(address as usize).parse_next(&mut stream)?;
        data_object(name).parse_next(&mut stream)
    }
}

/// Parse an HDF5/SOFA file and return a navigable structure.
pub fn parse_with_children(input: &[u8]) -> ModalResult<ParsedHdf<'_>> {
    let mut slice = input;
    let cp = slice.checkpoint();
    let super_block = super_block.parse_next(&mut slice)?;

    slice.reset(&cp);

    if super_block.end_of_file_address as usize != slice.eof_offset() {
        return Err(crate::hdf::helpers::invalid(&slice, "File size mismatch"));
    }

    let state = State::new(&super_block);
    let locating = LocatingSlice::new(slice);

    let mut stream = Input {
        input: locating,
        state: state.clone(),
    };

    let _skip =
        take(super_block.root_group_object_header_address as usize).parse_next(&mut stream)?;
    let root = data_object("root").parse_next(&mut stream)?;

    Ok(ParsedHdf {
        data: input,
        state,
        root,
    })
}

pub fn parse(mut input: &[u8]) -> ModalResult<DataObject> {
    let cp = input.checkpoint();
    let super_block = super_block.parse_next(&mut input)?;

    input.reset(&cp);

    if super_block.end_of_file_address as usize != input.eof_offset() {
        log::error!(
            "File size mismatch: header says {}, actual {}",
            super_block.end_of_file_address,
            input.eof_offset()
        );
        return Err(crate::hdf::helpers::invalid(&input, "File size mismatch"));
    }

    log::debug!(
        "SuperBlock parsed: offsets={}, lengths={}, root_addr={:#x}",
        super_block.size_of_offsets,
        super_block.size_of_lengths,
        super_block.root_group_object_header_address
    );

    let state = State::new(&super_block);
    let input = LocatingSlice::new(input);

    let mut stream = Input { input, state };

    // jump to the first object
    let _skip =
        take(super_block.root_group_object_header_address as usize).parse_next(&mut stream)?;

    log::debug!(
        "About to parse root data_object at position {:#x}",
        stream.input.current_token_start()
    );

    match data_object("root").parse_next(&mut stream) {
        Ok(obj) => {
            log::debug!("Root object parsed successfully: {}", obj.name);
            Ok(obj)
        }
        Err(e) => {
            log::error!("Failed to parse root object: {:?}", e);
            Err(e)
        }
    }
}
