//! SOFA file reader for HRTF data.
//!
//! This module provides the primary API for reading `HRTF` filters from
//! `SOFA` files (Spatially Oriented Format for Acoustics).
//!
//! # Examples
//!
//! Open a SOFA file with default options:
//!
//! ```no_run
//! use sofar::reader::Sofar;
//! use sofar::reader::Filter;
//!
//! let sofa = Sofar::open("path/to/file.sofa").unwrap();
//! let filt_len = sofa.filter_len();
//!
//! let mut filter = Filter::new(filt_len);
//! sofa.filter(0.0, 1.0, 0.0, &mut filter);
//! ```
//!
//! Open with custom options:
//!
//! ```no_run
//! use sofar::reader::OpenOptions;
//!
//! let sofa = OpenOptions::new()
//!     .sample_rate(44100.0)
//!     .open("path/to/file.sofa")
//!     .unwrap();
//! ```
//!
//! Open from in-memory bytes:
//!
//! ```no_run
//! use sofar::reader::Sofar;
//!
//! let data = std::fs::read("path/to/file.sofa").unwrap();
//! let sofa = Sofar::open_data(&data).unwrap();
//! ```
//!
//! Read a few measurements of a large file without the rest:
//!
//! ```no_run
//! use sofar::reader::LazySofa;
//!
//! let data = std::fs::read("path/to/room.sofa").unwrap();
//! let sofa = LazySofa::open(&data).unwrap();
//! let views = &sofa.hrtf().listener_view.values;
//! let first = sofa.read_ir(0, 1).unwrap();
//! ```

use std::path::Path;

use winnow::error::{ContextError, ErrMode};

use crate::hdf::{self, DataObject, ParsedHdf};
use crate::sofa::{
    Hrtf, InterpolatedFilter, Lookup, Neighborhood, get_filter_nointerp, interpolate, normalize,
    resample, validate,
};

const DEFAULT_NORMALIZED: bool = true;
const DEFAULT_SAMPLE_RATE: f32 = 48000.0;
const DEFAULT_NEIGHBOR_ANGLE_STEP: f32 = 0.5;
const DEFAULT_NEIGHBOR_RADIUS_STEP: f32 = 0.01;

#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(#[from] crate::sofa::Error),
    #[error("Invalid format")]
    InvalidFormat,
    #[error("Failed to build spatial lookup")]
    LookupBuildFailed,
    #[error("Resampling failed: {0}")]
    ResampleFailed(String),
}

/// Options for opening SOFA files.
#[derive(Clone, Debug)]
pub struct OpenOptions {
    sample_rate: f32,
    neighbor_angle_step: f32,
    neighbor_radius_step: f32,
    normalized: bool,
}

impl OpenOptions {
    /// Create a new set of open options with defaults.
    ///
    /// Default values:
    /// - `sample_rate`: 48000.0
    /// - `neighbor_angle_step`: 0.5°
    /// - `neighbor_radius_step`: 0.01m
    /// - `normalized`: true
    ///
    /// # Example
    ///
    /// ```no_run
    /// use sofar::reader::OpenOptions;
    ///
    /// let sofa = OpenOptions::new()
    ///     .sample_rate(44100.0)
    ///     .normalized(true)
    ///     .open("path/to/file.sofa")
    ///     .unwrap();
    /// ```
    pub fn new() -> Self {
        Default::default()
    }

    /// Set sampling rate of HRTF data. If requested sampling rate is different
    /// than what is in a SOFA file, the data will be resampled. Default value
    /// is 48_000.0.
    pub fn sample_rate(&mut self, sample_rate: f32) -> &mut Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Neighbor search angle step measured in degrees. Default value is 0.5.
    pub fn neighbor_angle_step(&mut self, neighbor_angle_step: f32) -> &mut Self {
        self.neighbor_angle_step = neighbor_angle_step;
        self
    }

    /// Neighbor search radius step measured in meters. Default value is 0.01.
    pub fn neighbor_radius_step(&mut self, neighbor_radius_step: f32) -> &mut Self {
        self.neighbor_radius_step = neighbor_radius_step;
        self
    }

    /// Apply normalization upon opening a SOFA file. Default value is `true`
    pub fn normalized(&mut self, normalized: bool) -> &mut Self {
        self.normalized = normalized;
        self
    }

    /// Open a SOFA file at `path` with open options specified in `self`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use sofar::reader::OpenOptions;
    ///
    /// let sofa = OpenOptions::new()
    ///     .sample_rate(44100.0)
    ///     .open("path/to/file.sofa")
    ///     .unwrap();
    /// ```
    pub fn open<P: AsRef<Path>>(&self, path: P) -> Result<Sofar, Error> {
        let data = std::fs::read(path)?;
        self.open_data(&data)
    }

    /// Open a SOFA file from in-memory bytes with open options specified in `self`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use sofar::reader::OpenOptions;
    ///
    /// let data = std::fs::read("path/to/file.sofa").unwrap();
    /// let sofa = OpenOptions::new()
    ///     .open_data(&data)
    ///     .unwrap();
    /// ```
    pub fn open_data<B: AsRef<[u8]>>(&self, bytes: B) -> Result<Sofar, Error> {
        let mut hrtf = Hrtf::from_bytes(bytes.as_ref())?;

        // Convert all position arrays to cartesian coordinates.
        // SOFA files may store positions in spherical coordinates.
        hrtf.convert_to_cartesian();

        // Validate (log warnings but don't fail)
        if let Err(e) = validate(&hrtf) {
            log::warn!("SOFA validation: {}", e);
        }

        // Resample if needed
        let original_rate = hrtf.sample_rate();
        if (original_rate - self.sample_rate).abs() > 0.1 {
            resample(&mut hrtf, self.sample_rate).map_err(Error::ResampleFailed)?;
        }

        // Normalize if requested
        if self.normalized {
            let _ = normalize(&mut hrtf);
        }

        // Build spatial lookup
        let lookup = Lookup::new(&hrtf).ok_or(Error::LookupBuildFailed)?;

        // Build neighborhood for interpolation
        let neighborhood = Neighborhood::new(
            &hrtf,
            &lookup,
            self.neighbor_angle_step,
            self.neighbor_radius_step,
        );

        let filter_len = hrtf.filter_len();

        Ok(Sofar {
            hrtf,
            lookup,
            neighborhood,
            filter_len,
        })
    }
}

impl Default for OpenOptions {
    fn default() -> Self {
        OpenOptions {
            sample_rate: DEFAULT_SAMPLE_RATE,
            neighbor_angle_step: DEFAULT_NEIGHBOR_ANGLE_STEP,
            neighbor_radius_step: DEFAULT_NEIGHBOR_RADIUS_STEP,
            normalized: DEFAULT_NORMALIZED,
        }
    }
}

pub use crate::filter::Filter;

/// SOFA reader providing access to HRTF filter data.
///
/// Wraps parsed HRTF data with spatial lookup and neighbor interpolation.
/// Use [`OpenOptions`] for fine-grained control, or the convenience methods
/// [`Sofar::open`] and [`Sofar::open_data`].
pub struct Sofar {
    hrtf: Hrtf,
    lookup: Lookup,
    neighborhood: Neighborhood,
    filter_len: usize,
}

impl Sofar {
    /// Open a SOFA file with the default open options.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use sofar::reader::Sofar;
    ///
    /// let sofa = Sofar::open("path/to/file.sofa").unwrap();
    /// println!("Filter length: {}", sofa.filter_len());
    /// ```
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Sofar, Error> {
        OpenOptions::new().open(path)
    }

    /// Open a SOFA file from in-memory bytes with the default open options.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use sofar::reader::Sofar;
    ///
    /// let data = std::fs::read("path/to/file.sofa").unwrap();
    /// let sofa = Sofar::open_data(&data).unwrap();
    /// ```
    pub fn open_data<B: AsRef<[u8]>>(bytes: B) -> Result<Sofar, Error> {
        OpenOptions::new().open_data(bytes)
    }

    /// Get the filter length (number of IR taps per channel).
    pub fn filter_len(&self) -> usize {
        self.filter_len
    }

    /// Get the HRTF filter for a given cartesian position using interpolation.
    ///
    /// Uses inverse distance weighting to combine the nearest measurement with
    /// up to 6 directional neighbors for smooth spatial transitions.
    ///
    /// # Arguments
    ///
    /// * `x` - Forward/backward position in meters
    /// * `y` - Left/right position in meters
    /// * `z` - Up/down position in meters
    /// * `filter` - Output filter to fill with interpolated IR data
    ///
    /// # Example
    ///
    /// ```no_run
    /// use sofar::reader::{Sofar, Filter};
    ///
    /// let sofa = Sofar::open("path/to/file.sofa").unwrap();
    /// let mut filter = Filter::new(sofa.filter_len());
    ///
    /// // Get filter for a source 1 meter in front
    /// sofa.filter(1.0, 0.0, 0.0, &mut filter);
    /// ```
    pub fn filter(&self, x: f32, y: f32, z: f32, filter: &mut Filter) {
        let position = [x, y, z];

        if let Some(nearest_idx) = self.lookup.find(&position)
            && let Some(interp) =
                interpolate(&self.hrtf, &self.neighborhood, nearest_idx, &position)
        {
            self.fill_filter(filter, &interp);
            return;
        }

        // Fallback: zero filter
        filter.left.iter_mut().for_each(|s| *s = 0.0);
        filter.right.iter_mut().for_each(|s| *s = 0.0);
        filter.ldelay = 0.0;
        filter.rdelay = 0.0;
    }

    /// Get the HRTF filter for a given position without interpolation.
    ///
    /// Returns the nearest measurement without blending with neighbors.
    /// Faster than [`filter`](Sofar::filter) but may produce audible
    /// discontinuities when the source position changes.
    pub fn filter_nointerp(&self, x: f32, y: f32, z: f32, filter: &mut Filter) {
        let position = [x, y, z];

        if let Some(nearest_idx) = self.lookup.find(&position)
            && let Some(interp) = get_filter_nointerp(&self.hrtf, nearest_idx)
        {
            self.fill_filter(filter, &interp);
            return;
        }

        // Fallback: zero filter
        filter.left.iter_mut().for_each(|s| *s = 0.0);
        filter.right.iter_mut().for_each(|s| *s = 0.0);
        filter.ldelay = 0.0;
        filter.rdelay = 0.0;
    }

    fn fill_filter(&self, filter: &mut Filter, interp: &InterpolatedFilter) {
        let copy_len = interp.left.len().min(filter.left.len());
        filter.left[..copy_len].copy_from_slice(&interp.left[..copy_len]);
        if copy_len < filter.left.len() {
            filter.left[copy_len..].iter_mut().for_each(|s| *s = 0.0);
        }

        let copy_len = interp.right.len().min(filter.right.len());
        filter.right[..copy_len].copy_from_slice(&interp.right[..copy_len]);
        if copy_len < filter.right.len() {
            filter.right[copy_len..].iter_mut().for_each(|s| *s = 0.0);
        }

        // Convert delay from samples to seconds
        let sample_rate = self.hrtf.sample_rate();
        filter.ldelay = interp.delay_left / sample_rate;
        filter.rdelay = interp.delay_right / sample_rate;
    }

    /// Get access to the underlying HRTF data.
    pub fn hrtf(&self) -> &Hrtf {
        &self.hrtf
    }

    /// Get access to the spatial lookup.
    pub fn lookup(&self) -> &Lookup {
        &self.lookup
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> f32 {
        self.hrtf.sample_rate()
    }

    /// Get the number of measurements.
    pub fn num_measurements(&self) -> u32 {
        self.hrtf.m()
    }
}

/// A SOFA file held in memory, opened without reading its impulse
/// responses.
///
/// Every other variable is read and converted to cartesian coordinates as
/// [`OpenOptions::open_data`] does; nothing is resampled or normalised, and
/// no lookup is built. `Data.IR` stays in the file until
/// [`LazySofa::read_ir`] reads the measurements a caller picks, unfiltering
/// only the stored chunks that hold them: one head orientation of a room
/// response set that runs to hundreds of MB, read in a fraction of the time
/// and memory the whole set takes.
pub struct LazySofa<'a> {
    parsed: ParsedHdf<'a>,
    ir: DataObject,
    hrtf: Hrtf,
}

impl<'a> LazySofa<'a> {
    /// Open the SOFA file in `bytes`, reading everything but `Data.IR`'s
    /// elements.
    ///
    /// # Errors
    ///
    /// The bytes are not a SOFA file, or it has no `Data.IR`.
    pub fn open(bytes: &'a [u8]) -> Result<Self, Error> {
        let parsed = hdf::parse_with_children(bytes).map_err(crate::sofa::Error::from)?;
        let mut hrtf = Hrtf::from_parsed_hdf(&parsed, false)?;
        hrtf.convert_to_cartesian();
        let ir = parsed
            .get_child_header("Data.IR")
            .ok_or(crate::sofa::Error::MissingArray("Data.IR"))?
            .map_err(hdf_error)?;
        Ok(Self { parsed, ir, hrtf })
    }

    /// The file's variables but `Data.IR`, whose values are left empty.
    pub fn hrtf(&self) -> &Hrtf {
        &self.hrtf
    }

    /// The shape of `Data.IR` as stored, outermost first: `[M, R, N]`, or
    /// `[M, R, E, N]` where a convention measures several emitters at once
    /// (`MultiSpeakerBRIR`).
    pub fn ir_shape(&self) -> Vec<usize> {
        let ds = &self.ir.ds;
        ds.dimension_size
            .iter()
            .take(ds.dimensionality as usize)
            .map(|&d| d as usize)
            .collect()
    }

    /// Measurements `first..first + count` of `Data.IR`: every value of
    /// each, row-major, as `f32`.
    ///
    /// # Errors
    ///
    /// The measurements lie beyond the file's, or their storage is damaged
    /// or of a type that is not a number.
    pub fn read_ir(&self, first: usize, count: usize) -> Result<Vec<f32>, Error> {
        let bytes = self
            .parsed
            .read_rows(&self.ir, first as u64, count as u64)
            .map_err(hdf_error)?;
        Hrtf::convert_data_to_f32(&bytes, &self.ir.dt)
            .ok_or(Error::Parse(crate::sofa::Error::UnsupportedDataType))
    }
}

/// An HDF parse error as this module reports it.
fn hdf_error(e: ErrMode<ContextError>) -> Error {
    let e = match e {
        ErrMode::Backtrack(e) | ErrMode::Cut(e) => e,
        ErrMode::Incomplete(_) => ContextError::new(),
    };
    Error::Parse(crate::sofa::Error::from(e))
}
