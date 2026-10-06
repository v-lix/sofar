use winnow::ModalResult;
use winnow::Parser;
use winnow::stream::{AsBytes, Stream, StreamIsPartial, ToUsize};
use winnow::token::take;

/// Parse a variable-sized little-endian integer (1-8 bytes).
#[inline]
pub(crate) fn varint_size<S>(size: impl ToUsize) -> impl FnMut(&mut S) -> ModalResult<u64>
where
    S: StreamIsPartial + Stream,
    S::Slice: AsBytes,
{
    use winnow::error::ErrMode;

    move |input| {
        let size = size.to_usize();
        if size > 8 {
            return Err(ErrMode::Cut(winnow::error::ContextError::new()));
        }

        let mut size_of_chunk = [0u8; 8];
        let bytes = take(size).parse_next(input)?;

        size_of_chunk[..size].copy_from_slice(bytes.as_bytes());
        Ok(u64::from_le_bytes(size_of_chunk))
    }
}

/// A malformed file: an error carrying `message`, in every build.
///
/// `ErrMode::assert` is meant for parser bugs, not bad input: it panics when
/// `debug_assertions` are on, so a corrupt file crashed debug builds of every
/// application reading it. This is the error `assert` returns in release,
/// with the message kept as context.
pub(crate) fn invalid<I: Stream>(
    input: &I,
    message: &'static str,
) -> winnow::error::ErrMode<winnow::error::ContextError> {
    use winnow::error::{AddContext, ContextError, ErrMode, StrContext};
    ErrMode::Backtrack(ContextError::new().add_context(
        input,
        &input.checkpoint(),
        StrContext::Label(message),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_2_bytes() {
        let mut input: &[u8] = &[0x34, 0x12];
        let result = varint_size(2usize).parse_next(&mut input).unwrap();
        assert_eq!(result, 0x1234);
    }

    #[test]
    fn test_varint_4_bytes() {
        let mut input: &[u8] = &[0x78, 0x56, 0x34, 0x12];
        let result = varint_size(4usize).parse_next(&mut input).unwrap();
        assert_eq!(result, 0x12345678);
    }

    #[test]
    fn test_varint_8_bytes() {
        let mut input: &[u8] = &[0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01];
        let result = varint_size(8usize).parse_next(&mut input).unwrap();
        assert_eq!(result, 0x0123456789ABCDEF);
    }

    #[test]
    fn test_varint_size_too_large_returns_error() {
        let mut input: &[u8] = &[0; 16];
        let result = varint_size(9usize).parse_next(&mut input);
        assert!(result.is_err(), "Expected error for size > 8");
    }
}
