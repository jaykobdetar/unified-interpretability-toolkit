use crate::{headroom, require, sha, Result};
use serde::{
    de::{self, MapAccess, SeqAccess, Visitor},
    Deserialize, Serialize,
};
use serde_json::{json, Value};
use std::{
    fmt,
    fs::{File, Metadata, OpenOptions},
    os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
};

// Reject duplicate keys at every JSON depth, including tensor metadata and index.
struct Unique(Value);
impl<'de> Deserialize<'de> for Unique {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Unique;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("JSON without duplicate keys")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Unique, A::Error> {
                let mut m = serde_json::Map::new();
                while let Some((k, v)) = a.next_entry::<String, Unique>()? {
                    if m.insert(k, v.0).is_some() {
                        return Err(de::Error::custom("Duplicate JSON key"));
                    }
                }
                Ok(Unique(Value::Object(m)))
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut a: A,
            ) -> std::result::Result<Unique, A::Error> {
                let mut v = Vec::new();
                while let Some(x) = a.next_element::<Unique>()? {
                    v.push(x.0)
                }
                Ok(Unique(Value::Array(v)))
            }
            fn visit_bool<E: de::Error>(self, v: bool) -> std::result::Result<Unique, E> {
                Ok(Unique(json!(v)))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Unique, E> {
                Ok(Unique(json!(v)))
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Unique, E> {
                Ok(Unique(json!(v)))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Unique, E> {
                Ok(Unique(json!(v)))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Unique, E> {
                Ok(Unique(json!(v)))
            }
            fn visit_string<E: de::Error>(self, v: String) -> std::result::Result<Unique, E> {
                Ok(Unique(json!(v)))
            }
            fn visit_unit<E: de::Error>(self) -> std::result::Result<Unique, E> {
                Ok(Unique(Value::Null))
            }
        }
        d.deserialize_any(V)
    }
}
pub fn json_unique(raw: &[u8]) -> Result<Value> {
    Ok(serde_json::from_slice::<Unique>(raw)?.0)
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Fingerprint {
    pub size: u64,
    pub dev: u64,
    pub inode: u64,
    pub mtime: i64,
    pub mtime_ns: i64,
    pub ctime: i64,
    pub ctime_ns: i64,
}
impl From<Metadata> for Fingerprint {
    fn from(m: Metadata) -> Self {
        Self {
            size: m.len(),
            dev: m.dev(),
            inode: m.ino(),
            mtime: m.mtime(),
            mtime_ns: m.mtime_nsec(),
            ctime: m.ctime(),
            ctime_ns: m.ctime_nsec(),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    Bf16,
    F16,
    F32,
}
impl Dtype {
    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "BF16" => Ok(Self::Bf16),
            "F16" => Ok(Self::F16),
            "F32" => Ok(Self::F32),
            _ => Err("Only BF16, F16 and F32 tensors are supported".into()),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F16 => "F16",
            Self::F32 => "F32",
        }
    }
    pub fn bytes(self) -> usize {
        if self == Self::F32 {
            4
        } else {
            2
        }
    }
    pub fn bits(self, raw: &[u8]) -> u32 {
        if self == Self::F32 {
            u32::from_le_bytes(raw.try_into().unwrap())
        } else {
            u16::from_le_bytes(raw.try_into().unwrap()) as u32
        }
    }
    pub fn finite(self, bits: u32) -> bool {
        let mask = match self {
            Self::Bf16 => 0x7f80,
            Self::F16 => 0x7c00,
            Self::F32 => 0x7f80_0000,
        };
        bits & mask != mask
    }
    pub fn value(self, bits: u32) -> f64 {
        match self {
            Self::Bf16 => value(bits as u16),
            Self::F32 => f32::from_bits(bits) as f64,
            Self::F16 => {
                let sign = if bits & 0x8000 == 0 { 1. } else { -1. };
                let e = (bits >> 10) & 31;
                let f = bits & 1023;
                if e == 31 {
                    if f == 0 {
                        sign * f64::INFINITY
                    } else {
                        f64::NAN
                    }
                } else if e == 0 {
                    sign * (f as f64) * 2f64.powi(-24)
                } else {
                    sign * ((1024 + f) as f64) * 2f64.powi(e as i32 - 25)
                }
            }
        }
    }
}
/// Storage widths only: this does not authorize interpreting quantized weights.
/// Unknown future encodings remain unavailable with extent-only validation.
fn storage_bits(name: &str) -> Option<u64> {
    match name {
        "BOOL" | "I8" | "U8" | "F8_E5M2" | "F8_E4M3" | "F8_E8M0" | "F8_E4M3FNUZ"
        | "F8_E5M2FNUZ" => Some(8),
        "I16" | "U16" | "F16" | "BF16" => Some(16),
        "I32" | "U32" | "F32" => Some(32),
        "I64" | "U64" | "F64" | "C64" => Some(64),
        "F4" => Some(4),
        "F6_E2M3" | "F6_E3M2" => Some(6),
        _ => None,
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tensor {
    pub id: usize,
    pub name: String,
    pub shape: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
    pub count: usize,
    pub dtype: String,
    pub element_bytes: usize,
    pub available: bool,
    pub unavailable_reason: Option<String>,
    pub shard: String,
    pub shard_id: usize,
    pub byte_offset: u64,
    pub max_level: u32,
    pub min_level: u32,
}
#[derive(Clone, Debug, Serialize)]
pub struct Shard {
    pub name: String,
    pub fingerprint: Fingerprint,
    pub header_sha256: String,
    pub data_start: u64,
}
pub struct Source {
    pub root: PathBuf,
    pub tensors: Vec<Tensor>,
    pub shards: Vec<Shard>,
    pub files: Vec<File>,
    pub identity: String,
    pub bytes: u64,
    pub header_bytes: u64,
    pub index_fingerprint: Option<Fingerprint>,
}
fn safe_name(name: &str) -> bool {
    !name.is_empty()
        && !name.contains('/')
        && !name.contains('\\')
        && name != "."
        && name != ".."
        && name.ends_with(".safetensors")
}
fn regular(path: &Path) -> Result<Metadata> {
    let m = std::fs::symlink_metadata(path)?;
    require(m.is_file(),"Only regular files inside the chosen model directory are supported; no shard/index symlinks")?;
    Ok(m)
}
pub fn read_small(path: &Path, maximum: u64) -> Result<Vec<u8>> {
    use std::io::Read;
    require(regular(path)?.len() <= maximum, "Metadata file too large")?;
    let f = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?;
    require(
        f.metadata()?.is_file() && f.metadata()?.len() <= maximum,
        "Metadata file changed",
    )?;
    let mut bytes = Vec::new();
    f.take(maximum + 1).read_to_end(&mut bytes)?;
    require(
        bytes.len() as u64 <= maximum,
        "Metadata file grew beyond limit",
    )?;
    Ok(bytes)
}

impl Source {
    pub fn open(root: &Path) -> Result<Self> {
        headroom()?;
        let root = root.canonicalize()?;
        require(root.is_dir(), "--model must be a directory")?;
        let ip = root.join("model.safetensors.index.json");
        let (index, ifp) = if ip.exists() {
            (
                Some(json_unique(&read_small(&ip, 8 * 1024 * 1024)?)?),
                Some(Fingerprint::from(regular(&ip)?)),
            )
        } else {
            (None, None)
        };
        let expected = if let Some(v) = &index {
            Some(
                v["weight_map"]
                    .as_object()
                    .ok_or("Missing index weight_map")?,
            )
        } else {
            None
        };
        let mut names = if let Some(m) = expected {
            m.values()
                .map(|v| v.as_str().ok_or("Invalid shard name").map(str::to_owned))
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            std::fs::read_dir(&root)?
                .map(|e| e.map(|e| e.file_name().to_string_lossy().into_owned()))
                .collect::<std::io::Result<Vec<_>>>()?
                .into_iter()
                .filter(|s| s.ends_with(".safetensors"))
                .collect()
        };
        names.sort();
        names.dedup();
        require(
            !names.is_empty() && names.len() <= 64,
            "Expected 1–64 safetensors shards",
        )?;
        let (mut tensors, mut shards, mut files) = (Vec::new(), Vec::new(), Vec::new());
        let mut hb = 0;
        for name in names {
            require(
                safe_name(&name),
                "Shard name must be a local safetensors filename",
            )?;
            let path = root.join(&name);
            let fingerprint = Fingerprint::from(regular(&path)?);
            let f = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW)
                .open(path)?;
            require(
                Fingerprint::from(f.metadata()?) == fingerprint,
                "Source changed while opening",
            )?;
            let mut length = [0u8; 8];
            f.read_exact_at(&mut length, 0)?;
            let n = u64::from_le_bytes(length);
            require(
                (2..=8 * 1024 * 1024).contains(&n) && n + 8 <= fingerprint.size,
                "Invalid safetensors header size",
            )?;
            hb += n + 8;
            require(
                hb <= 32 * 1024 * 1024,
                "Combined headers exceed 32 MiB limit",
            )?;
            let mut raw = vec![0; n as usize];
            f.read_exact_at(&mut raw, 8)?;
            let header = json_unique(&raw)?;
            let map = header.as_object().ok_or("Header must be an object")?;
            let mut intervals = Vec::new();
            let sid = files.len();
            for (key, v) in map {
                if key == "__metadata__" {
                    require(
                        v.as_object()
                            .is_some_and(|m| m.values().all(Value::is_string)),
                        "Safetensors metadata must contain strings",
                    )?;
                    continue;
                }
                require(
                    !key.is_empty() && key.len() < 512,
                    "Tensor names must contain 1–511 bytes",
                )?;
                let dtype_name = v["dtype"].as_str().ok_or("Missing dtype")?;
                require(
                    !dtype_name.is_empty() && dtype_name.len() <= 64,
                    "Invalid dtype name",
                )?;
                let storage_bits = storage_bits(dtype_name);
                let shape = v["shape"]
                    .as_array()
                    .ok_or("Missing shape")?
                    .iter()
                    .map(|d| {
                        d.as_u64()
                            .ok_or("Shape must be positive integers")
                            .and_then(|n| {
                                if usize::try_from(n).is_ok() {
                                    Ok(n as usize)
                                } else {
                                    Err("Dimension exceeds address space")
                                }
                            })
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                require(
                    shape.len() <= 32,
                    "Tensor rank exceeds metadata limit of 32",
                )?;
                let count = shape
                    .iter()
                    .try_fold(1usize, |a, &b| a.checked_mul(b))
                    .ok_or("Shape overflow")?;
                let offsets = v["data_offsets"].as_array().ok_or("Missing offsets")?;
                require(offsets.len() == 2, "Two offsets required")?;
                let a = offsets[0].as_u64().ok_or("Invalid offset")?;
                let b = offsets[1].as_u64().ok_or("Invalid offset")?;
                let bytes = storage_bits
                    .map(|bits| {
                        u64::try_from(count)
                            .ok()
                            .and_then(|n| n.checked_mul(bits))
                            .filter(|n| n % 8 == 0)
                            .map(|n| n / 8)
                            .ok_or("Tensor byte length overflow or sub-byte misalignment")
                    })
                    .transpose()?;
                let byte_offset = n
                    .checked_add(8)
                    .and_then(|v| v.checked_add(a))
                    .ok_or("Tensor offset overflow")?;
                require(
                    a <= b
                        && bytes.is_none_or(|bytes| b - a == bytes)
                        && b <= fingerprint.size - n - 8,
                    "Tensor offsets/shape exceed source bounds",
                )?;
                let (rows, cols) = match shape.len() {
                    0 => (0, 0),
                    1 => (1, shape[0]),
                    rank => (shape[rank - 2], shape[rank - 1]),
                };
                let unavailable_reason = if Dtype::parse(dtype_name).is_err() {
                    Some(if storage_bits.is_some() {
                        "Storage extent validated; numeric encoding/quantization scale semantics unsupported".to_string()
                    } else {
                        "Unknown storage encoding; declared extent checked only, no numeric interpretation".to_string()
                    })
                } else if shape.is_empty() || count == 0 {
                    Some("Scalar/empty tensor display unsupported".to_string())
                } else if rows > 200000 || cols > 200000 {
                    Some("Dimension exceeds the unchanged 200000 display-axis limit".to_string())
                } else {
                    None
                };
                let max = rows.max(cols);
                let max_level = if max == 0 {
                    0
                } else {
                    usize::BITS - (max - 1).leading_zeros()
                };
                tensors.push(Tensor {
                    id: 0,
                    name: key.clone(),
                    shape,
                    rows,
                    cols,
                    count,
                    dtype: dtype_name.into(),
                    element_bytes: storage_bits
                        .filter(|bits| bits % 8 == 0)
                        .map_or(0, |bits| (bits / 8) as usize),
                    available: unavailable_reason.is_none(),
                    unavailable_reason,
                    shard: name.clone(),
                    shard_id: sid,
                    byte_offset,
                    max_level,
                    min_level: 0,
                });
                intervals.push((a, b));
            }
            intervals.sort();
            let mut cursor = 0;
            for (a, b) in intervals {
                require(a == cursor, "Gaps or overlaps in safetensors data")?;
                cursor = b
            }
            require(
                n.checked_add(8).and_then(|v| v.checked_add(cursor)) == Some(fingerprint.size),
                "Trailing or missing tensor bytes",
            )?;
            shards.push(Shard {
                name,
                fingerprint,
                header_sha256: sha(&raw),
                data_start: 8 + n,
            });
            files.push(f);
            require(tensors.len() <= 10000, "Tensor catalog exceeds 10000 limit")?;
        }
        tensors.sort_by(|a, b| a.name.cmp(&b.name));
        for i in 0..tensors.len() {
            require(
                i == 0 || tensors[i - 1].name != tensors[i].name,
                "Duplicate tensor across shards",
            )?;
            tensors[i].id = i;
        }
        require(!tensors.is_empty(), "No tensors")?;
        tensors
            .iter()
            .try_fold(0usize, |n, t| n.checked_add(t.count))
            .ok_or("Catalog element count overflow")?;
        if let Some(m) = expected {
            require(
                m.len() == tensors.len()
                    && tensors
                        .iter()
                        .all(|t| m.get(&t.name).and_then(Value::as_str) == Some(&t.shard)),
                "Index/header tensor coverage mismatch",
            )?
        }
        let identity = sha(&serde_json::to_vec(
            &json!({"schema":1,"root":root,"shards":shards,"index":index,"index_stat":ifp}),
        )?);
        let bytes = shards.iter().map(|s| s.fingerprint.size).sum();
        let source = Self {
            root,
            tensors,
            shards,
            files,
            identity,
            bytes,
            header_bytes: hb,
            index_fingerprint: ifp,
        };
        source.check()?;
        Ok(source)
    }
    pub fn check(&self) -> Result<()> {
        for (s, f) in self.shards.iter().zip(&self.files) {
            require(Fingerprint::from(regular(&self.root.join(&s.name))?)==s.fingerprint&&Fingerprint::from(f.metadata()?)==s.fingerprint,"Source identity changed; stop and reopen model. Cached calibration/tiles are invalid.")?
        }
        let p = self.root.join("model.safetensors.index.json");
        let now = if p.exists() {
            Some(Fingerprint::from(regular(&p)?))
        } else {
            None
        };
        require(now == self.index_fingerprint, "Index changed; reopen model")
    }
    pub fn tensor(&self, id: usize) -> Result<&Tensor> {
        self.tensors.get(id).ok_or_else(|| "Unknown tensor".into())
    }
    /// Compatibility reader for BF16 callers; generic consumers use scalar_bits.
    pub fn scalar(&self, t: &Tensor, row: usize, col: usize) -> Result<(u16, u64)> {
        require(t.dtype == "BF16", "BF16 scalar reader requires BF16 tensor")?;
        let (bits, offset) = self.scalar_bits(t, row, col)?;
        Ok((bits as u16, offset))
    }
    pub fn scalar_bits(&self, t: &Tensor, row: usize, col: usize) -> Result<(u32, u64)> {
        self.check()?;
        require(
            t.available,
            t.unavailable_reason
                .as_deref()
                .unwrap_or("Tensor unavailable"),
        )?;
        require(
            t.shape.len() <= 2,
            "Higher-rank tensor requires an explicit 2D slice",
        )?;
        require(row < t.rows && col < t.cols, "Address outside tensor")?;
        let dtype = Dtype::parse(&t.dtype)?;
        let index = row
            .checked_mul(t.cols)
            .and_then(|v| v.checked_add(col))
            .ok_or("Scalar index overflow")?;
        let off = element_offset(t, index, dtype)?;
        let mut raw = [0; 4];
        self.files[t.shard_id].read_exact_at(&mut raw[..dtype.bytes()], off)?;
        self.check()?;
        Ok((dtype.bits(&raw[..dtype.bytes()]), off))
    }
}
pub fn element_offset(t: &Tensor, index: usize, dtype: Dtype) -> Result<u64> {
    require(index < t.count, "Element outside tensor")?;
    u64::try_from(index)?
        .checked_mul(dtype.bytes() as u64)
        .and_then(|v| t.byte_offset.checked_add(v))
        .ok_or_else(|| "Element offset overflow".into())
}
pub fn value(bits: u16) -> f64 {
    f32::from_bits((bits as u32) << 16) as f64
}

/// Mathematical decimal expansion of an original BF16 value.
pub fn exact_decimal(bits: u16) -> String {
    exact_decimal_for(Dtype::Bf16, bits as u32)
}
/// Exact binary-rational decimal expansion; no shortest-roundtrip/display rounding.
/// Finite binary32 needs at most 149 decimal places; its largest integer fits u128.
pub fn exact_decimal_for(dtype: Dtype, bits: u32) -> String {
    let (fraction_bits, exponent_bits, bias) = match dtype {
        Dtype::Bf16 => (7, 8, 127),
        Dtype::F16 => (10, 5, 15),
        Dtype::F32 => (23, 8, 127),
    };
    let negative = bits & (1 << (fraction_bits + exponent_bits)) != 0;
    let exponent_mask = (1 << exponent_bits) - 1;
    let exponent = (bits >> fraction_bits) & exponent_mask;
    let fraction = bits & ((1 << fraction_bits) - 1);
    if exponent == exponent_mask {
        return if fraction != 0 {
            "NaN".into()
        } else if negative {
            "-Infinity".into()
        } else {
            "Infinity".into()
        };
    }
    if exponent == 0 && fraction == 0 {
        return if negative {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    let (mantissa, power) = if exponent == 0 {
        (fraction as u128, 1 - bias - fraction_bits)
    } else {
        (
            ((1 << fraction_bits) + fraction) as u128,
            exponent as i32 - bias - fraction_bits,
        )
    };
    let mut result = if power >= 0 {
        (mantissa << power).to_string()
    } else {
        let places = (-power) as usize;
        let mut digits = mantissa
            .to_string()
            .bytes()
            .rev()
            .map(|b| b - b'0')
            .collect::<Vec<_>>();
        for _ in 0..places {
            let mut carry = 0;
            for d in &mut digits {
                let n = *d * 5 + carry;
                *d = n % 10;
                carry = n / 10;
            }
            if carry > 0 {
                digits.push(carry);
            }
        }
        while digits.len() <= places {
            digits.push(0);
        }
        let mut text = digits
            .iter()
            .rev()
            .map(|d| (b'0' + d) as char)
            .collect::<String>();
        text.insert(text.len() - places, '.');
        while text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
        text
    };
    if negative {
        result.insert(0, '-');
    }
    result
}
