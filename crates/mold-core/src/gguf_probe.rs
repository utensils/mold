//! Bounded, header-only GGUF metadata used by pre-download/runtime qualification.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
const MAX_METADATA_ENTRIES: u64 = 16_384;
const MAX_TENSORS: u64 = 100_000;
const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ARRAY_ITEMS: u64 = 1_000_000;
const MAX_DIMS: u32 = 8;

#[derive(Debug, Clone, PartialEq)]
pub enum GgufMetadataValue {
    String(String),
    Bool(bool),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<GgufMetadataValue>),
}

impl GgufMetadataValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorInfo {
    /// Dimensions in PyTorch/Candle order (GGUF stores them reversed).
    pub shape: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub metadata: HashMap<String, GgufMetadataValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
}

struct Reader<R> {
    inner: R,
    consumed: u64,
}

impl<R: Read> Reader<R> {
    fn read_exact<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        self.consumed = self
            .consumed
            .checked_add(N as u64)
            .filter(|total| *total <= MAX_HEADER_BYTES)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "GGUF header is too large")
            })?;
        let mut bytes = [0; N];
        self.inner.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.read_exact::<1>()?[0])
    }

    fn u16(&mut self) -> io::Result<u16> {
        Ok(u16::from_le_bytes(self.read_exact()?))
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.read_exact()?))
    }

    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.read_exact()?))
    }

    fn string(&mut self) -> io::Result<String> {
        let len = self.u64()?;
        if len > MAX_STRING_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("GGUF string length {len} exceeds the header limit"),
            ));
        }
        self.consumed = self
            .consumed
            .checked_add(len)
            .filter(|total| *total <= MAX_HEADER_BYTES)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "GGUF header is too large")
            })?;
        let mut bytes = vec![0; len as usize];
        self.inner.read_exact(&mut bytes)?;
        String::from_utf8(bytes)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err.to_string()))
    }

    fn value(&mut self, value_type: u32, array_depth: usize) -> io::Result<GgufMetadataValue> {
        let value = match value_type {
            0 => GgufMetadataValue::U64(self.u8()? as u64),
            1 => GgufMetadataValue::I64(self.u8()? as i8 as i64),
            2 => GgufMetadataValue::U64(self.u16()? as u64),
            3 => GgufMetadataValue::I64(self.u16()? as i16 as i64),
            4 => GgufMetadataValue::U64(self.u32()? as u64),
            5 => GgufMetadataValue::I64(self.u32()? as i32 as i64),
            6 => GgufMetadataValue::F64(f32::from_le_bytes(self.read_exact()?) as f64),
            7 => match self.u8()? {
                0 => GgufMetadataValue::Bool(false),
                1 => GgufMetadataValue::Bool(true),
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid GGUF boolean value {other}"),
                    ));
                }
            },
            8 => GgufMetadataValue::String(self.string()?),
            9 if array_depth == 0 => {
                let item_type = self.u32()?;
                if item_type == 9 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "nested GGUF metadata arrays are unsupported",
                    ));
                }
                let len = self.u64()?;
                if len > MAX_ARRAY_ITEMS {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("GGUF metadata array length {len} exceeds the limit"),
                    ));
                }
                let mut values = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    values.push(self.value(item_type, array_depth + 1)?);
                }
                GgufMetadataValue::Array(values)
            }
            10 => GgufMetadataValue::U64(self.u64()?),
            11 => GgufMetadataValue::I64(self.u64()? as i64),
            12 => GgufMetadataValue::F64(f64::from_le_bytes(self.read_exact()?)),
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported GGUF metadata value type {other}"),
                ));
            }
        };
        Ok(value)
    }
}

pub fn read_gguf_header(path: &Path) -> io::Result<GgufHeader> {
    let file = File::open(path)?;
    let mut reader = Reader {
        inner: file,
        consumed: 0,
    };
    if &reader.read_exact::<4>()? != b"GGUF" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "missing GGUF magic",
        ));
    }
    let version = reader.u32()?;
    if version != 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported GGUF version {version}; expected v3"),
        ));
    }
    let tensor_count = reader.u64()?;
    let metadata_count = reader.u64()?;
    if tensor_count > MAX_TENSORS || metadata_count > MAX_METADATA_ENTRIES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "GGUF tensor or metadata count exceeds the qualification limit",
        ));
    }

    let mut metadata = HashMap::with_capacity(metadata_count as usize);
    for _ in 0..metadata_count {
        let key = reader.string()?;
        let value_type = reader.u32()?;
        let value = reader.value(value_type, 0)?;
        if metadata.insert(key.clone(), value).is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("duplicate GGUF metadata key {key}"),
            ));
        }
    }

    let mut tensors = HashMap::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = reader.string()?;
        let n_dims = reader.u32()?;
        if n_dims == 0 || n_dims > MAX_DIMS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("GGUF tensor {name} has invalid rank {n_dims}"),
            ));
        }
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let dim = reader.u64()?;
            if dim == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("GGUF tensor {name} has a zero dimension"),
                ));
            }
            shape.push(dim);
        }
        shape.reverse();
        let ggml_type = reader.u32()?;
        let offset = reader.u64()?;
        if tensors
            .insert(
                name.clone(),
                GgufTensorInfo {
                    shape,
                    ggml_type,
                    offset,
                },
            )
            .is_some()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("duplicate GGUF tensor name {name}"),
            ));
        }
    }

    Ok(GgufHeader {
        version,
        metadata,
        tensors,
    })
}
