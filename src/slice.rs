//! Explicit contiguous trailing-two-axis views; original metadata stays unchanged.
use crate::{
    require, sha,
    source::{Dtype, Source, Tensor},
    Result,
};
use serde_json::{json, Value};

#[derive(Clone, Debug)]
pub struct TensorSlice {
    pub tensor: Tensor,
    pub native_shape: Vec<usize>,
    pub leading: Vec<usize>,
    pub element_start: usize,
    pub source_identity: String,
    pub identity: String,
}

impl TensorSlice {
    pub fn new(source: &Source, id: usize, leading: &[usize]) -> Result<Self> {
        let original = source.tensor(id)?;
        require(
            original.available,
            original
                .unavailable_reason
                .as_deref()
                .unwrap_or("Tensor unavailable"),
        )?;
        let dtype = Dtype::parse(&original.dtype)?;
        let rank = original.shape.len();
        require(
            leading.len() == rank.saturating_sub(2),
            "Every leading axis requires an explicit slice index",
        )?;
        let mut plane = 0usize;
        for (&index, &dimension) in leading.iter().zip(&original.shape) {
            require(index < dimension, "Slice index outside native shape")?;
            plane = plane
                .checked_mul(dimension)
                .and_then(|v| v.checked_add(index))
                .ok_or("Slice index overflow")?;
        }
        let count = original
            .rows
            .checked_mul(original.cols)
            .ok_or("Slice size overflow")?;
        let element_start = plane.checked_mul(count).ok_or("Slice offset overflow")?;
        require(
            element_start
                .checked_add(count)
                .is_some_and(|end| end <= original.count),
            "Slice exceeds original tensor",
        )?;
        let mut tensor = original.clone();
        tensor.byte_offset = original
            .byte_offset
            .checked_add(
                u64::try_from(element_start)?
                    .checked_mul(dtype.bytes() as u64)
                    .ok_or("Slice byte offset overflow")?,
            )
            .ok_or("Slice byte offset overflow")?;
        tensor.count = count;
        tensor.shape = if rank == 1 {
            original.shape.clone()
        } else {
            vec![original.rows, original.cols]
        };
        let identity = sha(json!([
            "weight-atlas-trailing-slice-v1",
            source.identity,
            id,
            original.name,
            original.shape,
            original.dtype,
            leading
        ])
        .to_string()
        .as_bytes());
        Ok(Self {
            tensor,
            native_shape: original.shape.clone(),
            leading: leading.to_vec(),
            element_start,
            source_identity: source.identity.clone(),
            identity,
        })
    }

    pub fn check(&self, source: &Source) -> Result<()> {
        require(
            self.source_identity == source.identity,
            "Slice belongs to another source",
        )?;
        source.check()
    }

    pub fn native_indices(&self, row: usize, col: usize) -> Result<Vec<usize>> {
        require(
            row < self.tensor.rows && col < self.tensor.cols,
            "Address outside slice",
        )?;
        let mut result = self.leading.clone();
        if self.native_shape.len() > 1 {
            result.push(row);
        }
        result.push(col);
        Ok(result)
    }

    pub fn binding(&self, model_identity: &str) -> Value {
        json!({"version":2,"model_identity":model_identity,"source_identity":self.source_identity,
            "tensor":self.tensor.id,"name":self.tensor.name,"dtype":self.tensor.dtype,"shape":self.native_shape,
            "rows":self.tensor.rows,"cols":self.tensor.cols,
            "slice":{"leading_indices":self.leading,"display_axes":if self.native_shape.len()==1 {vec![0]} else {vec![self.native_shape.len()-2,self.native_shape.len()-1]}}})
    }

    pub fn descriptor(&self) -> Value {
        json!({"version":1,"identity":self.identity,"leading_indices":self.leading,
            "native_shape":self.native_shape,"display_axes":if self.native_shape.len()==1 {vec![0]} else {vec![self.native_shape.len()-2,self.native_shape.len()-1]},
            "element_start":self.element_start,"count":self.tensor.count,"byte_offset":self.tensor.byte_offset})
    }
}

pub fn parse_indices(text: &str) -> Result<Vec<usize>> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    require(text.len() <= 256, "Slice index list too long")?;
    text.split(',')
        .map(|part| {
            require(
                !part.is_empty() && part.bytes().all(|c| c.is_ascii_digit()),
                "Slice indices must be unsigned decimal integers",
            )?;
            Ok(part.parse()?)
        })
        .collect()
}
