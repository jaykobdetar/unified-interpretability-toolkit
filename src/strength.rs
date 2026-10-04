//! Bounded, explicitly advanced whole-slice mean-magnitude profiles.
//! No threads, HTTP routes, automatic continuation, disk cache or reordering.
use crate::{
    headroom, require, sha,
    slice::TensorSlice,
    source::{element_offset, Dtype, Source},
    Result,
};
use serde_json::{json, Value};
use std::{
    os::unix::fs::FileExt,
    time::{Duration, Instant},
};

pub mod snapshot;

pub const MAX_STATE_BYTES: usize = 32 * 1024 * 1024;
pub const MAX_PAGE: usize = 1024;
pub const CHUNK_VALUES: usize = 65536;
const BATCH_VALUES: usize = 4096;
const MAX_GRANT: Duration = Duration::from_secs(5);
const CHUNK_TIME: Duration = Duration::from_millis(50);

#[derive(Clone, Default)]
struct Sum {
    sum: f64,
    correction: f64,
    count: u64,
}
impl Sum {
    fn add(&mut self, value: f64) {
        let y = value - self.correction;
        let next = self.sum + y;
        self.correction = (next - self.sum) - y;
        self.sum = next;
        self.count += 1;
    }
    fn record(&self, index: usize, expected: usize) -> Value {
        json!({"index":index,"sum_abs":self.sum,"visited_count":self.count,
            "expected_count":expected,"mean_abs":if self.count==0 {None} else {Some(self.sum/self.count as f64)},
            "complete":self.count==expected as u64})
    }
}

fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Each round conditionally swaps a pair i,(pivot-i) mod n. Both members
/// make the same decision, so each round and their composition are bijections.
/// Eight rounds are an experimental structured control, not a uniform draw
/// from all permutations and not a calibrated statistical null distribution.
pub fn control_destination(mut index: usize, count: usize, seed: u64) -> usize {
    assert!(count > 0 && index < count);
    let n = count as u64;
    for round in 0..8u64 {
        let key = mix(seed ^ round.wrapping_mul(0x9e3779b97f4a7c15));
        let pivot = key % n;
        let other = if pivot >= index as u64 {
            pivot - index as u64
        } else {
            n - (index as u64 - pivot)
        } as usize;
        if mix(key ^ index.min(other) as u64) & 1 != 0 {
            index = other;
        }
    }
    index
}

pub struct StrengthProfile {
    slice: TensorSlice,
    binding: Value,
    identity: String,
    seed: u64,
    rows: Vec<Sum>,
    columns: Vec<Sum>,
    control_rows: Vec<Sum>,
    control_columns: Vec<Sum>,
    visited: usize,
    authorized_until: usize,
    remaining_time: Duration,
    active_time: Duration,
    allocated_bytes: usize,
    error: Option<String>,
}

impl StrengthProfile {
    pub fn new(
        source: &Source,
        slice: TensorSlice,
        model_identity: &str,
        seed: u64,
        values: usize,
        time: Duration,
    ) -> Result<Self> {
        slice.check(source)?;
        require(
            model_identity.len() == 64 && model_identity.bytes().all(|b| b.is_ascii_hexdigit()),
            "Model revision identity required",
        )?;
        let t = &slice.tensor;
        require(
            values > 0 && values <= t.count,
            "Scan allowance must fit remaining slice",
        )?;
        require(
            !time.is_zero() && time <= MAX_GRANT,
            "Scan time allowance must be positive and at most five seconds",
        )?;
        let axes = t
            .rows
            .checked_add(t.cols)
            .ok_or("Profile dimensions overflow")?;
        let allocated_bytes = axes
            .checked_mul(2 * std::mem::size_of::<Sum>())
            .and_then(|n| n.checked_add(BATCH_VALUES * 4))
            .ok_or("Profile memory estimate overflow")?;
        require(
            allocated_bytes <= MAX_STATE_BYTES,
            "Profile axes exceed 32 MiB state budget; no allocation performed",
        )?;
        headroom()?;
        fn zeros(n: usize) -> Result<Vec<Sum>> {
            let mut v = Vec::new();
            v.try_reserve_exact(n)?;
            v.resize(n, Sum::default());
            Ok(v)
        }
        let binding = slice.binding(model_identity);
        let identity = sha(json!([
            "weight-atlas-strength-v1",
            binding,
            seed,
            "swap-or-not-8-v1"
        ])
        .to_string()
        .as_bytes());
        let mut result = Self {
            rows: zeros(t.rows)?,
            columns: zeros(t.cols)?,
            control_rows: zeros(t.rows)?,
            control_columns: zeros(t.cols)?,
            slice,
            binding,
            identity,
            seed,
            visited: 0,
            authorized_until: 0,
            remaining_time: Duration::ZERO,
            active_time: Duration::ZERO,
            allocated_bytes,
            error: None,
        };
        result.authorize(values, time)?;
        Ok(result)
    }

    /// Caller must invoke only for a new explicit user action. Replaces remaining
    /// allowance; repeated scheduler ticks must call advance(), never authorize().
    pub fn authorize(&mut self, values: usize, time: Duration) -> Result<()> {
        require(self.error.is_none(), "Invalid profile must be discarded")?;
        require(
            values > 0 && values <= self.slice.tensor.count - self.visited,
            "Scan allowance must fit remaining slice",
        )?;
        require(
            !time.is_zero() && time <= MAX_GRANT,
            "Scan time allowance must be positive and at most five seconds",
        )?;
        self.authorized_until = self.visited + values;
        self.remaining_time = time;
        Ok(())
    }

    pub fn advance(&mut self, source: &Source) -> Result<()> {
        self.advance_until(source, CHUNK_VALUES, Instant::now() + MAX_GRANT)
    }

    /// Clamp one chunk to the caller's remaining values and absolute deadline.
    /// This never authorizes more work. An expired external deadline exhausts
    /// this grant; only a separately admitted explicit action may authorize again.
    /// The deadline must be constructed in this process's monotonic clock domain.
    /// Disposable-worker CPU and cleanup bounds remain the host's responsibility.
    pub fn advance_until(
        &mut self,
        source: &Source,
        max_values: usize,
        external_deadline: Instant,
    ) -> Result<()> {
        require(self.error.is_none(), "Invalid profile must be discarded")?;
        require(max_values > 0, "Positive external value allowance required")?;
        let start = Instant::now();
        if start >= external_deadline {
            self.remaining_time = Duration::ZERO;
            return Ok(());
        }
        let result = self.advance_inner(source, start, max_values, external_deadline);
        let elapsed = start.elapsed();
        self.active_time += elapsed;
        self.remaining_time = self.remaining_time.saturating_sub(elapsed);
        if Instant::now() >= external_deadline {
            self.remaining_time = Duration::ZERO;
        }
        if let Err(error) = &result {
            self.error = Some(error.to_string());
        }
        result
    }

    fn advance_inner(
        &mut self,
        source: &Source,
        start: Instant,
        max_values: usize,
        external_deadline: Instant,
    ) -> Result<()> {
        self.slice.check(source)?;
        if self.visited >= self.authorized_until || self.remaining_time.is_zero() {
            return Ok(());
        }
        headroom()?;
        let deadline = (start + self.remaining_time.min(CHUNK_TIME)).min(external_deadline);
        let end = self
            .authorized_until
            .min(self.visited.saturating_add(CHUNK_VALUES.min(max_values)));
        let t = &self.slice.tensor;
        let dtype = Dtype::parse(&t.dtype)?;
        let mut raw = vec![0u8; BATCH_VALUES * dtype.bytes()];
        while self.visited < end && Instant::now() < deadline {
            let n = BATCH_VALUES.min(end - self.visited);
            source.files[t.shard_id].read_exact_at(
                &mut raw[..n * dtype.bytes()],
                element_offset(t, self.visited, dtype)?,
            )?;
            for bytes in raw[..n * dtype.bytes()].chunks_exact(dtype.bytes()) {
                let word = dtype.bits(bytes);
                require(
                    dtype.finite(word),
                    "Nonfinite source value invalidates the profile; no replacement value",
                )?;
                let value = dtype.value(word).abs();
                let i = self.visited;
                self.rows[i / t.cols].add(value);
                self.columns[i % t.cols].add(value);
                let j = control_destination(i, t.count, self.seed);
                self.control_rows[j / t.cols].add(value);
                self.control_columns[j % t.cols].add(value);
                self.visited += 1;
            }
        }
        // Cooperative deadline: at most one already-started 4096-value batch
        // overshoots. Filesystem I/O latency is not a hard real-time guarantee.
        self.slice.check(source)
    }

    /// Small status only: no vectors or source scanning.
    pub fn progress(&self) -> Value {
        let complete = self.error.is_none() && self.visited == self.slice.tensor.count;
        json!({"schema":"weight-atlas.strength.v1","identity":self.identity,"binding":self.binding,
            "state":if self.error.is_some(){"invalid"}else if complete{"complete"}else if self.visited>=self.authorized_until || self.remaining_time.is_zero(){"paused"}else{"ready"},
            "complete":complete,"visited_values":self.visited,"total_values":self.slice.tensor.count,
            "authorized_remaining_values":self.authorized_until-self.visited,"remaining_active_ms":self.remaining_time.as_secs_f64()*1000.,
            "active_seconds":self.active_time.as_secs_f64(),"allocated_state_bytes":self.allocated_bytes,"error":self.error,
            "coverage":"Original: row-major prefix. Partial means describe only visited cells, not representative samples. Control uses exactly the same visited multiset at permuted destinations; axis coverage may differ until complete.",
            "control":{"seed":self.seed,"algorithm":"swap-or-not-8-v1","exact_multiset":true,"uniform_permutation":false,
                "interpretation":"Experimental coordinate permutation control. Compare coverage counts. No calibrated significance, concept or causal claim."}})
    }

    pub fn page(&self, source: &Source, axis: &str, start: usize, count: usize) -> Result<Value> {
        self.slice.check(source)?;
        require(
            self.error.is_none(),
            "Invalid profile has no publishable statistics",
        )?;
        let (original, control, expected) = match axis {
            "rows" => (&self.rows, &self.control_rows, self.slice.tensor.cols),
            "columns" => (&self.columns, &self.control_columns, self.slice.tensor.rows),
            _ => return Err("Profile axis must be rows or columns".into()),
        };
        require(
            count > 0 && count <= MAX_PAGE && start < original.len(),
            "Profile page outside bounded range",
        )?;
        let end = original.len().min(start + count);
        Ok(
            json!({"progress":self.progress(),"axis":axis,"start":start,"end":end,"axis_length":original.len(),
            "original":(start..end).map(|i|original[i].record(i,expected)).collect::<Vec<_>>(),
            "control":(start..end).map(|i|control[i].record(i,expected)).collect::<Vec<_>>() }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn controls_are_bijections_for_even_odd_and_prime_domains() {
        for count in 1..=257 {
            for seed in [0, 1, 17, u64::MAX] {
                let mut mapped = (0..count)
                    .map(|i| control_destination(i, count, seed))
                    .collect::<Vec<_>>();
                mapped.sort_unstable();
                assert_eq!(mapped, (0..count).collect::<Vec<_>>());
            }
        }
    }
}
