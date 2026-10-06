pub(crate) use super::{
    data_object::{DataLayout, DataSpace, MAX_DATASET_BYTES},
    helpers::varint_size,
    parser::Input,
};

use miniz_oxide::inflate::decompress_to_vec_zlib_with_limit;
use winnow::ModalResult;
use winnow::Parser;
use winnow::binary::{le_u8, le_u16, le_u32, le_u64};
use winnow::combinator::repeat;
use winnow::error::StrContext;
use winnow::stream::Stream;
use winnow::token::{literal, take};

use log::info;

/// ASCII C format:                 [   T,    R,    E,    E]
pub const TREE_SIGNATURE: [u8; 4] = [0x54, 0x52, 0x45, 0x45];

/// Filter identifiers of the HDF5 filter pipeline (spec, "Filter Pipeline
/// Message").
const FILTER_DEFLATE: u16 = 1;
const FILTER_SHUFFLE: u16 = 2;

/// Deepest chunk B-tree accepted (a leaf-only tree is depth 0).
const MAX_TREE_DEPTH: u8 = 8;
/// Largest single chunk accepted once unfiltered, bytes.
const MAX_CHUNK_BYTES: u64 = 0x1000_0000;

/// What the chunk reader needs to know about the dataset.
struct Geometry {
    /// Rank of the dataset (1 to 4).
    dims: usize,
    /// Chunk extent per dimension, elements.
    chunk: Vec<u64>,
    /// Dataset extent per dimension, elements.
    dataset: Vec<u64>,
    /// Row-major dataset stride per dimension, elements.
    stride: Vec<u64>,
    /// Element size, bytes.
    size: u64,
    /// Elements per chunk.
    elements: u64,
    /// Whether the pipeline holds the deflate / shuffle filters.
    deflate: bool,
    shuffle: bool,
}

/// Read a chunked dataset through its version 1 B-tree (node type 1),
/// scattering every chunk into a row-major buffer of `data_len` bytes.
///
/// Chunks pass through the filters the object's pipeline message declared
/// (deflate, shuffle; nothing else is supported), the tree may have inner
/// nodes, and the dataset may have up to four dimensions — what a
/// `MultiSpeakerBRIR` written by netCDF4 needs (`Data.IR` is `[M][R][E][N]`,
/// chunked, deflated and shuffled, hundreds of MB).
pub(crate) fn tree(
    data_len: usize,
    data_space: DataSpace,
    data_layout: DataLayout,
) -> impl FnMut(&mut Input) -> ModalResult<Vec<u8>> {
    move |input| {
        if data_len as u64 > MAX_DATASET_BYTES {
            return Err(crate::hdf::helpers::invalid(
                input,
                "Tree data_len exceeds maximum allowed size",
            ));
        }
        let dims = data_space.dimensionality as usize;
        if dims == 0 || dims > data_space.dimension_size.len() || dims >= data_layout.len() {
            return Err(crate::hdf::helpers::invalid(
                input,
                "Tree dimensionality unsupported",
            ));
        }
        let chunk: Vec<u64> = data_layout[..dims].iter().map(|&c| c as u64).collect();
        let size = data_layout[dims] as u64;
        let elements = chunk
            .iter()
            .try_fold(1u64, |acc, &c| acc.checked_mul(c))
            .unwrap_or(0);
        if elements == 0
            || size == 0
            || size > 0x10
            || elements
                .checked_mul(size)
                .is_none_or(|b| b > MAX_CHUNK_BYTES)
        {
            return Err(crate::hdf::helpers::invalid(
                input,
                "Invalid tree elements or size",
            ));
        }
        let dataset: Vec<u64> = data_space.dimension_size[..dims].to_vec();
        let mut stride = vec![1u64; dims];
        for d in (0..dims.saturating_sub(1)).rev() {
            stride[d] = stride[d + 1].saturating_mul(dataset[d + 1]);
        }
        // The pipeline message precedes the layout message in files written
        // by the HDF5 library; a dataset without one holds raw chunks.
        let filters = input.state.filters();
        let geometry = Geometry {
            dims,
            chunk,
            dataset,
            stride,
            size,
            elements,
            deflate: filters.contains(&FILTER_DEFLATE),
            shuffle: filters.contains(&FILTER_SHUFFLE),
        };
        info!(
            "Tree: {} dims, chunk {:?}, dataset {:?}, element size {}, filters {:?}",
            dims, geometry.chunk, geometry.dataset, size, filters
        );

        let mut data = vec![0u8; data_len];
        read_node(input, &geometry, &mut data, 0)?;
        Ok(data)
    }
}

/// One B-tree node at the current position: its keys and children, reading
/// chunks at the leaves and recursing into inner nodes.
fn read_node(
    input: &mut Input,
    geometry: &Geometry,
    data: &mut [u8],
    depth: u8,
) -> ModalResult<()> {
    if depth > MAX_TREE_DEPTH {
        return Err(crate::hdf::helpers::invalid(
            input,
            "Tree deeper than supported",
        ));
    }
    let size_of_offsets = input.state.size_of_offsets();

    let _signature = literal(TREE_SIGNATURE).parse_next(input)?;
    let _node_type = le_u8
        .verify(|t| *t == 1)
        .context(StrContext::Label("Tree node type"))
        .context(StrContext::Expected("1 (raw data chunks)".into()))
        .parse_next(input)?;
    let node_level = le_u8.parse_next(input)?;
    let entries_used = le_u16
        .verify(|e| *e <= 0x1000)
        .context(StrContext::Label("Tree entries used"))
        .context(StrContext::Expected("<= 0x1000".into()))
        .parse_next(input)?;
    let _address_of_left_sibling = varint_size(size_of_offsets).parse_next(input)?;
    let _address_of_right_sibling = varint_size(size_of_offsets).parse_next(input)?;

    for _ in 0..entries_used {
        // Key: chunk size on disk, filter mask, one offset per dimension
        // plus the (always zero) offset in the element-size dimension.
        let chunk_bytes = le_u32.parse_next(input)?;
        let filter_mask = le_u32
            .verify(|m| *m == 0)
            .context(StrContext::Label("TREE filter mask"))
            .context(StrContext::Expected(
                "all filters must be enabled (0)".into(),
            ))
            .parse_next(input)?;
        let start: Vec<u64> = repeat(geometry.dims, le_u64).parse_next(input)?;
        let _element_offset = le_u64.parse_next(input)?;
        let child = varint_size(size_of_offsets).parse_next(input)?;
        if !input.state.is_address_valid(child) {
            return Err(crate::hdf::helpers::invalid(
                input,
                "Invalid child pointer address",
            ));
        }
        let _ = filter_mask;

        let cp = input.checkpoint();
        input.input.reset_to_start();
        let _skip = take(child as usize).parse_next(input)?;
        if node_level > 0 {
            read_node(input, geometry, data, depth + 1)?;
        } else {
            info!(" chunk at {child:#x} len {chunk_bytes} start {start:?}");
            read_chunk(input, geometry, data, chunk_bytes as usize, &start)?;
        }
        input.reset(&cp);
    }
    // The trailing key closes the last child; nothing in it is needed.
    let _trailing = take(4 + 4 + 8 * (geometry.dims + 1)).parse_next(input)?;
    Ok(())
}

/// One chunk at the current position: unfilter it and scatter its elements
/// into `data` at `start`, dropping whatever lies beyond the dataset (the
/// edge chunks of a dataset whose extent is not a multiple of the chunk).
fn read_chunk(
    input: &mut Input,
    g: &Geometry,
    data: &mut [u8],
    chunk_bytes: usize,
    start: &[u64],
) -> ModalResult<()> {
    let raw = take(chunk_bytes).parse_next(input)?;
    let olen = (g.elements * g.size) as usize;
    let inflated;
    let buf: &[u8] = if g.deflate {
        inflated = decompress_to_vec_zlib_with_limit(raw, olen)
            .map_err(|_err| crate::hdf::helpers::invalid(input, "Failed to inflate btree data"))?;
        &inflated
    } else {
        raw
    };
    if buf.len() != olen {
        return Err(crate::hdf::helpers::invalid(
            input,
            "Invalid tree chunk length",
        ));
    }

    let last = g.dims - 1;
    let run_len = g.chunk[last];
    let runs = g.elements / run_len;
    // Elements of the innermost dimension that fall inside the dataset.
    let keep = g.dataset[last].saturating_sub(start[last]).min(run_len);
    let size = g.size as usize;
    let elements = g.elements as usize;
    for run in 0..runs {
        // Multi-index of the run over the outer dimensions, row-major.
        let mut rem = run;
        let mut base = start[last];
        let mut inside = true;
        for d in (0..last).rev() {
            let local = rem % g.chunk[d];
            rem /= g.chunk[d];
            let x = local + start[d];
            if x >= g.dataset[d] {
                inside = false;
                break;
            }
            base += x * g.stride[d];
        }
        if !inside {
            continue;
        }
        let first = (run * run_len) as usize;
        for k in 0..keep as usize {
            let e = first + k;
            let dst = (base as usize + k) * size;
            if dst + size > data.len() {
                break;
            }
            if g.shuffle {
                for b in 0..size {
                    data[dst + b] = buf[b * elements + e];
                }
            } else {
                data[dst..dst + size].copy_from_slice(&buf[e * size..e * size + size]);
            }
        }
    }
    Ok(())
}
