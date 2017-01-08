use std::error;
use std::result;
use std::fmt;

#[derive(Debug)]
#[derive(PartialEq)]
pub enum ErrorType {
    NonFiniteEntry,
    ZeroNorm,
}

#[derive(Debug)]
pub struct Error {
    pub error_type: ErrorType,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:#?}", self.error_type)
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        match self.error_type {
            ErrorType::ZeroNorm => "Zero-length vectors present in input data.",
            ErrorType::NonFiniteEntry => "Infinite or NaN values present in input data.",
        }
    }
}

pub type Result<T> = result::Result<T, Error>;
