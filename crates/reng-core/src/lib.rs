//! Core types for the Reciprocating Engine.
//!
//! These are the small, dependency-light building blocks shared by every other
//! crate in the workspace: element data types, tensor shapes, device
//! identifiers, and the common error type. Nothing here is specific to a single
//! accelerator vendor.

use thiserror::Error;

/// Numeric type of a single tensor element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    /// IEEE 754 single precision.
    F32,
    /// IEEE 754 half precision.
    F16,
    /// Brain floating point (8-bit exponent, 7-bit mantissa).
    Bf16,
    /// Signed 32-bit integer.
    I32,
    /// Signed 16-bit integer.
    I16,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 8-bit integer.
    U8,
}

impl Dtype {
    /// Size of a single element in bytes.
    #[must_use]
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F16 | Dtype::Bf16 | Dtype::I16 => 2,
            Dtype::I8 | Dtype::U8 => 1,
        }
    }

    /// Whether this is a floating-point type.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Dtype::F32 | Dtype::F16 | Dtype::Bf16)
    }
}

/// Row-major shape of a dense tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Create a shape from a list of dimensions.
    #[must_use]
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self { dims: dims.into() }
    }

    /// The shape of a scalar (rank 0), which holds a single element.
    #[must_use]
    pub fn scalar() -> Self {
        Self { dims: Vec::new() }
    }

    /// The dimensions, outermost first.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Number of dimensions.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Total number of elements. A scalar has one element.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Bytes needed to store a dense tensor of this shape and `dtype`.
    #[must_use]
    pub fn size_in_bytes(&self, dtype: Dtype) -> usize {
        self.numel() * dtype.size_in_bytes()
    }
}

/// Kind of compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    /// The host CPU.
    Cpu,
    /// An Intel Gaudi2 (HL-225) accelerator.
    Gaudi2,
}

/// Identifier of a device within the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId(pub u32);

/// Errors produced by the core and hardware layers.
#[derive(Debug, Error)]
pub enum Error {
    /// A device with the given index was not found.
    #[error("device {0} not found")]
    DeviceNotFound(u32),
    /// An operation does not support the given data type.
    #[error("unsupported dtype for this operation: {0:?}")]
    UnsupportedDtype(Dtype),
    /// An underlying I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Any other error, described by the message.
    #[error("{0}")]
    Other(String),
}

/// Result alias used across the workspace.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_sizes() {
        assert_eq!(Dtype::F32.size_in_bytes(), 4);
        assert_eq!(Dtype::I32.size_in_bytes(), 4);
        assert_eq!(Dtype::Bf16.size_in_bytes(), 2);
        assert_eq!(Dtype::F16.size_in_bytes(), 2);
        assert_eq!(Dtype::I8.size_in_bytes(), 1);
        assert_eq!(Dtype::U8.size_in_bytes(), 1);
    }

    #[test]
    fn dtype_is_float() {
        assert!(Dtype::F32.is_float());
        assert!(Dtype::Bf16.is_float());
        assert!(!Dtype::I8.is_float());
        assert!(!Dtype::I32.is_float());
    }

    #[test]
    fn shape_numel_and_bytes() {
        let s = Shape::new([2, 3, 4]);
        assert_eq!(s.rank(), 3);
        assert_eq!(s.dims(), &[2, 3, 4]);
        assert_eq!(s.numel(), 24);
        assert_eq!(s.size_in_bytes(Dtype::Bf16), 48);
        assert_eq!(s.size_in_bytes(Dtype::F32), 96);
    }

    #[test]
    fn scalar_shape() {
        let s = Shape::scalar();
        assert_eq!(s.rank(), 0);
        assert_eq!(s.numel(), 1);
        assert_eq!(s.size_in_bytes(Dtype::F32), 4);
    }
}
