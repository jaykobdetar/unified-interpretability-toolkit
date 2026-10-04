//! Private, fixed-format session snapshot. Never restores a work/time grant.
use super::*;
use sha2::{Digest, Sha256};
use std::io::{BufReader, BufWriter, Read, Write};

pub const HEADER_BYTES: usize = 168;
pub const MAX_BINDING_BYTES: usize = 16384;
pub const BUFFER_BYTES: usize = 16384;
pub const ALGORITHM: &str = "weight-atlas-strength-snapshot-v1:kahan-f64-abs:swap-or-not-8-v1";
const MAGIC: &[u8; 8] = b"WAPROF01";
// Covers small metadata, IO buffers and at most one bounded page. Both memfd
// generations count, even though Linux process RSS need not account for them.
const FIXED_RESERVE: usize = 2 * 1024 * 1024 + 128 * 1024;

#[derive(Debug)]
pub struct Layout {
    pub frame_bytes: usize,
    pub live_bytes: usize,
}

pub fn layout(rows: usize, cols: usize, binding_bytes: usize) -> Result<Layout> {
    require(
        rows > 0 && cols > 0 && rows <= 200000 && cols <= 200000,
        "Snapshot display dimensions invalid",
    )?;
    require(
        binding_bytes > 0 && binding_bytes <= MAX_BINDING_BYTES,
        "Snapshot binding too large",
    )?;
    let axes = rows.checked_add(cols).ok_or("Snapshot axes overflow")?;
    let frame_bytes = axes
        .checked_mul(48)
        .and_then(|n| n.checked_add(HEADER_BYTES + binding_bytes))
        .ok_or("Snapshot size overflow")?;
    let live_bytes = axes
        .checked_mul(2 * std::mem::size_of::<Sum>() + 8)
        .and_then(|n| n.checked_add(2 * frame_bytes))
        .and_then(|n| n.checked_add(FIXED_RESERVE))
        .ok_or("Snapshot live size overflow")?;
    require(
        live_bytes <= MAX_STATE_BYTES,
        "Snapshot, overlap and restore state exceed 32 MiB; no allocation",
    )?;
    Ok(Layout {
        frame_bytes,
        live_bytes,
    })
}

fn check_time(deadline: Instant) -> Result<()> {
    require(
        Instant::now() < deadline,
        "Snapshot deadline exhausted; explicit Restart may be required",
    )
}
fn hash_bytes(s: &str) -> Result<[u8; 32]> {
    require(
        s.len() == 64
            && s.bytes()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "Invalid snapshot digest",
    )?;
    let mut out = [0; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)?;
    }
    Ok(out)
}
fn number(raw: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        raw[offset..offset + 8]
            .try_into()
            .expect("fixed header range"),
    )
}
fn put(raw: &mut [u8], offset: usize, n: u64) {
    raw[offset..offset + 8].copy_from_slice(&n.to_le_bytes());
}
fn identity(binding: &Value, seed: u64) -> String {
    sha(json!([
        "weight-atlas-strength-v1",
        binding,
        seed,
        "swap-or-not-8-v1"
    ])
    .to_string()
    .as_bytes())
}
fn read_hashed(
    reader: &mut impl Read,
    hash: &mut Sha256,
    raw: &mut [u8],
    deadline: Instant,
) -> Result<()> {
    check_time(deadline)?;
    reader.read_exact(raw)?;
    hash.update(raw);
    Ok(())
}

impl StrengthProfile {
    pub fn visited_values(&self) -> usize {
        self.visited
    }

    /// Writes one bounded frame. Caller owns a private output memfd and must seal
    /// and validate it before publishing. Partial output is never a snapshot.
    pub fn write_snapshot(
        &self,
        source: &Source,
        output: impl Write,
        deadline: Instant,
    ) -> Result<String> {
        check_time(deadline)?;
        self.slice.check(source)?;
        require(
            self.error.is_none(),
            "Invalid profile cannot be snapshotted",
        )?;
        require(
            self.seed <= u32::MAX as u64,
            "Snapshot seed must fit uint32",
        )?;
        let binding = self.binding.to_string();
        let t = &self.slice.tensor;
        let size = layout(t.rows, t.cols, binding.len())?;
        let mut header = [0u8; HEADER_BYTES];
        header[..8].copy_from_slice(MAGIC);
        header[8..12].copy_from_slice(&1u32.to_le_bytes());
        header[12..16].copy_from_slice(&(binding.len() as u32).to_le_bytes());
        for (offset, n) in [
            (16, t.rows),
            (24, t.cols),
            (32, t.count),
            (40, self.visited),
            (48, self.seed as usize),
            (56, 2 * (t.rows + t.cols)),
            (64, size.frame_bytes - HEADER_BYTES - binding.len()),
        ] {
            put(&mut header, offset, n as u64);
        }
        header[72..104].copy_from_slice(&hash_bytes(&sha(binding.as_bytes()))?);
        header[104..136].copy_from_slice(&hash_bytes(&sha(ALGORITHM.as_bytes()))?);
        header[136..168].copy_from_slice(&hash_bytes(&self.identity)?);
        let mut hash = Sha256::new();
        let mut writer = BufWriter::with_capacity(BUFFER_BYTES, output);
        for bytes in [&header[..], binding.as_bytes()] {
            writer.write_all(bytes)?;
            hash.update(bytes);
        }
        for (i, sum) in self
            .rows
            .iter()
            .chain(&self.columns)
            .chain(&self.control_rows)
            .chain(&self.control_columns)
            .enumerate()
        {
            if i % 256 == 0 {
                check_time(deadline)?;
            }
            let mut raw = [0u8; 24];
            raw[..8].copy_from_slice(&sum.sum.to_le_bytes());
            raw[8..16].copy_from_slice(&sum.correction.to_le_bytes());
            raw[16..].copy_from_slice(&sum.count.to_le_bytes());
            writer.write_all(&raw)?;
            hash.update(raw);
        }
        writer.flush()?;
        self.slice.check(source)?;
        check_time(deadline)?;
        Ok(format!("{:x}", hash.finalize()))
    }

    /// Restore is exhausted. A trusted host must admit a new explicit action
    /// before calling authorize. expected_revision must be coordinator-retained,
    /// not a digest submitted by a visitor alongside arbitrary snapshot bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn restore_snapshot(
        source: &Source,
        slice: TensorSlice,
        model_identity: &str,
        seed: u64,
        input: impl Read,
        encoded_len: usize,
        expected_revision: &str,
        deadline: Instant,
    ) -> Result<Self> {
        check_time(deadline)?;
        slice.check(source)?;
        require(seed <= u32::MAX as u64, "Snapshot seed must fit uint32")?;
        let expected_hash = hash_bytes(expected_revision)?;
        let binding = slice.binding(model_identity);
        let canonical = binding.to_string();
        let t = &slice.tensor;
        let size = layout(t.rows, t.cols, canonical.len())?;
        require(encoded_len == size.frame_bytes, "Snapshot length mismatch")?;
        let mut reader = BufReader::with_capacity(BUFFER_BYTES, input);
        let mut hash = Sha256::new();
        let mut header = [0u8; HEADER_BYTES];
        read_hashed(&mut reader, &mut hash, &mut header, deadline)?;
        require(
            &header[..8] == MAGIC
                && header[8..12] == 1u32.to_le_bytes()
                && header[12..16] == (canonical.len() as u32).to_le_bytes(),
            "Unknown snapshot format/binding length",
        )?;
        let visited = usize::try_from(number(&header, 40))?;
        require(visited <= t.count, "Snapshot cursor outside slice")?;
        for (offset, n) in [
            (16, t.rows),
            (24, t.cols),
            (32, t.count),
            (48, seed as usize),
            (56, 2 * (t.rows + t.cols)),
            (64, 48 * (t.rows + t.cols)),
        ] {
            require(
                number(&header, offset) == n as u64,
                "Snapshot shape/seed/layout mismatch",
            )?;
        }
        require(
            header[72..104] == hash_bytes(&sha(canonical.as_bytes()))?
                && header[104..136] == hash_bytes(&sha(ALGORITHM.as_bytes()))?
                && header[136..168] == hash_bytes(&identity(&binding, seed))?,
            "Snapshot identity/algorithm mismatch",
        )?;
        let mut raw_binding = vec![0u8; canonical.len()];
        read_hashed(&mut reader, &mut hash, &mut raw_binding, deadline)?;
        require(
            raw_binding == canonical.as_bytes(),
            "Snapshot canonical binding mismatch",
        )?;
        let finite_max = match Dtype::parse(&slice.tensor.dtype)? {
            Dtype::F16 => 65504.0,
            Dtype::Bf16 => f32::from_bits(0x7f7f0000) as f64,
            Dtype::F32 => f32::MAX as f64,
        };
        // The one-nanosecond constructor grant is never advanced and is cleared
        // immediately; no stored or restored allowance can survive this function.
        let mut p = Self::new(
            source,
            slice,
            model_identity,
            seed,
            1,
            Duration::from_nanos(1),
        )?;
        p.visited = visited;
        p.authorized_until = visited;
        p.remaining_time = Duration::ZERO;
        for (i, sum) in p
            .rows
            .iter_mut()
            .chain(&mut p.columns)
            .chain(&mut p.control_rows)
            .chain(&mut p.control_columns)
            .enumerate()
        {
            if i % 256 == 0 {
                check_time(deadline)?;
            }
            let mut raw = [0u8; 24];
            read_hashed(&mut reader, &mut hash, &mut raw, deadline)?;
            sum.sum = f64::from_le_bytes(raw[..8].try_into()?);
            sum.correction = f64::from_le_bytes(raw[8..16].try_into()?);
            sum.count = number(&raw, 16);
            require(
                sum.sum.is_finite() && sum.sum >= 0. && sum.correction.is_finite(),
                "Nonfinite or negative snapshot sum",
            )?;
            require(
                sum.correction.abs() <= 64. * f64::EPSILON * sum.sum,
                "Snapshot compensation outside bound",
            )?;
            require(
                sum.count != 0 || (sum.sum == 0. && sum.correction == 0.),
                "Empty snapshot record has a sum",
            )?;
            require(
                sum.sum <= sum.count as f64 * finite_max,
                "Snapshot sum exceeds source dtype bound",
            )?;
        }
        let mut trailing = [0u8; 1];
        require(reader.read(&mut trailing)? == 0, "Trailing snapshot bytes")?;
        require(
            hash.finalize().as_slice() == expected_hash,
            "Snapshot revision mismatch",
        )?;
        p.validate_snapshot_counts(deadline)?;
        source.check()?;
        check_time(deadline)?;
        Ok(p)
    }

    fn validate_snapshot_counts(&self, deadline: Instant) -> Result<()> {
        let t = &self.slice.tensor;
        let mut counts = Vec::<u64>::new();
        counts.try_reserve_exact(t.rows + t.cols)?;
        counts.resize(t.rows + t.cols, 0);
        if self.visited == t.count {
            counts[..t.rows].fill(t.cols as u64);
            counts[t.rows..].fill(t.rows as u64);
        } else {
            // No arbitrary prefix cap: O(visited), cooperatively time bounded.
            // Refusal is preferable to trusting incorrect control coverage.
            for i in 0..self.visited {
                if i % 256 == 0 {
                    check_time(deadline)?;
                }
                let j = control_destination(i, t.count, self.seed);
                counts[j / t.cols] += 1;
                counts[t.rows + j % t.cols] += 1;
            }
        }
        let mut totals = [0f64; 4];
        for (group, sums) in [
            &self.rows,
            &self.columns,
            &self.control_rows,
            &self.control_columns,
        ]
        .iter()
        .enumerate()
        {
            let mut aggregate = Sum::default();
            let mut count = 0u64;
            for (i, s) in sums.iter().enumerate() {
                if i % 256 == 0 {
                    check_time(deadline)?;
                }
                let expected = match group {
                    0 => self.visited.saturating_sub(i * t.cols).min(t.cols) as u64,
                    1 => (self.visited / t.cols + usize::from(i < self.visited % t.cols)) as u64,
                    2 => counts[i],
                    _ => counts[t.rows + i],
                };
                require(
                    s.count == expected,
                    "Snapshot axis prefix coverage mismatch",
                )?;
                count = count
                    .checked_add(s.count)
                    .ok_or("Snapshot count overflow")?;
                aggregate.add(s.sum);
                totals[group] = aggregate.sum;
            }
            require(
                count == self.visited as u64,
                "Snapshot paired total count mismatch",
            )?;
        }
        for total in totals {
            require(
                (total - totals[0]).abs() <= 128. * f64::EPSILON * total.max(totals[0]),
                "Snapshot paired sums inconsistent",
            )?;
        }
        check_time(deadline)
    }
}
