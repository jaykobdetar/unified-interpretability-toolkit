pub use crate::rules::IDS as RULES;
use crate::{
    headroom, require,
    rules::{self, Definition, Scope, Transform},
    source::{element_offset, Dtype, Source, Tensor},
    Result,
};
use flate2::{write::ZlibEncoder, Compression};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::BTreeMap, io::Write, os::unix::fs::FileExt, time::Instant};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stats {
    pub count: usize,
    pub max_abs: f64,
    pub median_nonzero_abs: f64,
    pub q99: f64,
    pub robust_clipped_count: u64,
    pub histogram_sha256: Option<String>,
    pub exact_zero_count: u64,
    pub unique_bit_patterns: Option<usize>,
    pub dtype: String,
    pub calibration_method: String,
    pub seconds: f64,
}
const RAW_BAND_BYTES: usize = 2 * 1048576;
fn scan_words(source: &Source, t: &Tensor, mut visit: impl FnMut(u32)) -> Result<()> {
    source.check()?;
    let dtype = Dtype::parse(&t.dtype)?;
    let capacity = RAW_BAND_BYTES / dtype.bytes();
    let mut buf = vec![0u8; RAW_BAND_BYTES];
    let mut done = 0;
    while done < t.count {
        headroom()?;
        let n = (t.count - done).min(capacity);
        source.files[t.shard_id].read_exact_at(
            &mut buf[..n * dtype.bytes()],
            element_offset(t, done, dtype)?,
        )?;
        for raw in buf[..n * dtype.bytes()].chunks_exact(dtype.bytes()) {
            let bits = dtype.bits(raw);
            require(
                dtype.finite(bits),
                "Nonfinite source value: calibration refused",
            )?;
            visit(bits);
        }
        done += n;
    }
    source.check()
}
pub fn calibrate(source: &Source, t: &Tensor) -> Result<(Stats, Vec<u64>)> {
    let start = Instant::now();
    require(
        t.available,
        t.unavailable_reason
            .as_deref()
            .unwrap_or("Tensor unavailable"),
    )?;
    let dtype = Dtype::parse(&t.dtype)?;
    if dtype == Dtype::F32 {
        return calibrate_f32(source, t, start);
    }
    let mut hist = vec![0u64; 65536];
    scan_words(source, t, |bits| hist[bits as usize] += 1)?;
    let abs = (0..32768)
        .map(|i| hist[i] + hist[i + 32768])
        .collect::<Vec<_>>();
    let q99 = quantile_for(&abs, 0.99, false, dtype);
    let divisor = if q99 == 0. { 1. } else { q99 };
    let stats = Stats {
        count: t.count,
        max_abs: dtype.value(abs.iter().rposition(|&n| n > 0).unwrap_or(0) as u32),
        median_nonzero_abs: quantile_for(&abs, 0.5, true, dtype),
        q99,
        robust_clipped_count: abs
            .iter()
            .enumerate()
            .filter(|(i, _)| dtype.value(*i as u32) > divisor)
            .map(|(_, &n)| n)
            .sum(),
        histogram_sha256: Some(histogram_digest(&hist)),
        exact_zero_count: abs[0],
        unique_bit_patterns: Some(hist.iter().filter(|&&n| n > 0).count()),
        dtype: dtype.name().into(),
        calibration_method: "exact-16-bit-histogram".into(),
        seconds: start.elapsed().as_secs_f64(),
    };
    Ok((stats, hist))
}
// Positive finite IEEE binary32 words have exactly the same order as their values.
// Quantiles refine at most four high-word buckets, plus the divisor-one fallback
// bucket for an exact robust clipping count. Memory is independent of tensor size.
fn rank_bucket(hist: &[u64], mut rank: u64) -> (usize, u64) {
    for (bucket, &count) in hist.iter().enumerate() {
        if rank < count {
            return (bucket, rank);
        }
        rank -= count;
    }
    unreachable!("rank is within the complete finite tensor")
}
fn calibrate_f32(source: &Source, t: &Tensor, start: Instant) -> Result<(Stats, Vec<u64>)> {
    let mut high = vec![0u64; 65536];
    let (mut maximum, mut zeros) = (0u32, 0u64);
    scan_words(source, t, |bits| {
        let magnitude = bits & 0x7fff_ffff;
        high[(magnitude >> 16) as usize] += 1;
        maximum = maximum.max(magnitude);
        zeros += u64::from(magnitude == 0);
    })?;
    let count = t.count as u64;
    let nonzero = count - zeros;
    let q = (count - 1) as f64 * 0.99;
    let m = nonzero.saturating_sub(1) as f64 * 0.5;
    let ranks = [
        q.floor() as u64,
        q.ceil() as u64,
        if nonzero > 0 {
            zeros + m.floor() as u64
        } else {
            0
        },
        if nonzero > 0 {
            zeros + m.ceil() as u64
        } else {
            0
        },
    ];
    let plans = ranks.map(|r| rank_bucket(&high, r));
    let mut low = BTreeMap::<usize, Vec<u64>>::new();
    for (bucket, _) in plans {
        low.entry(bucket).or_insert_with(|| vec![0; 65536]);
    }
    low.entry((1f32.to_bits() >> 16) as usize)
        .or_insert_with(|| vec![0; 65536]);
    scan_words(source, t, |bits| {
        let magnitude = bits & 0x7fff_ffff;
        if let Some(hist) = low.get_mut(&((magnitude >> 16) as usize)) {
            hist[(magnitude & 65535) as usize] += 1;
        }
    })?;
    let selected = plans.map(|(bucket, rank)| {
        let (suffix, _) = rank_bucket(&low[&bucket], rank);
        f32::from_bits(((bucket as u32) << 16) | suffix as u32) as f64
    });
    let q99 = selected[0] + (selected[1] - selected[0]) * (q - q.floor());
    let divisor = if q99 == 0. { 1. } else { q99 };
    // Round threshold downward to a representable nonnegative binary32 word.
    // This makes strict source-value > divisor counts exact, including ties.
    let mut threshold = (divisor as f32).to_bits();
    if (f32::from_bits(threshold) as f64) > divisor {
        threshold -= 1;
    }
    let bucket = (threshold >> 16) as usize;
    let within = if let Some(hist) = low.get(&bucket) {
        hist[..=(threshold & 65535) as usize].iter().sum::<u64>()
    } else {
        require(high[bucket] == 0, "Unrefined nonempty quantile bucket")?;
        0
    };
    let robust_clipped_count = count - high[..bucket].iter().sum::<u64>() - within;
    Ok((
        Stats {
            count: t.count,
            max_abs: f32::from_bits(maximum) as f64,
            median_nonzero_abs: if nonzero > 0 {
                selected[2] + (selected[3] - selected[2]) * (m - m.floor())
            } else {
                0.
            },
            q99,
            robust_clipped_count,
            histogram_sha256: None,
            exact_zero_count: zeros,
            unique_bit_patterns: None,
            dtype: "F32".into(),
            calibration_method: "exact-two-pass-radix-quantiles".into(),
            seconds: start.elapsed().as_secs_f64(),
        },
        Vec::new(),
    ))
}
pub fn histogram_digest(hist: &[u64]) -> String {
    crate::sha(
        &hist
            .iter()
            .flat_map(|n| n.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
pub fn quantile(abs: &[u64], q: f64, exclude_zero: bool) -> f64 {
    quantile_for(abs, q, exclude_zero, Dtype::Bf16)
}
fn quantile_for(abs: &[u64], q: f64, exclude_zero: bool, dtype: Dtype) -> f64 {
    let n: u64 = abs
        .iter()
        .enumerate()
        .filter(|(i, _)| !exclude_zero || *i != 0)
        .map(|(_, n)| n)
        .sum();
    if n == 0 {
        return 0.;
    }
    let p = (n - 1) as f64 * q;
    let lo = p.floor() as u64;
    let hi = p.ceil() as u64;
    let (mut count, mut a, mut b) = (0, 0., 0.);
    let mut got = false;
    for (i, &k) in abs.iter().enumerate() {
        if exclude_zero && i == 0 {
            continue;
        }
        count += k;
        if !got && count > lo {
            a = dtype.value(i as u32);
            got = true
        }
        if count > hi {
            b = dtype.value(i as u32);
            break;
        }
    }
    a + (b - a) * (p - lo as f64)
}
pub fn rule_info(rule: &str) -> Result<Value> {
    rules::definition(rule)?.metadata()
}
pub fn validate_rule_dtype(rule: &str, dtype: Dtype) -> Result<()> {
    rules::definition(rule)?.validate_dtype(dtype)
}
pub fn legend(rule: &str, stats: Option<&Stats>, global: Option<f64>) -> Result<Value> {
    let definition = rules::definition(rule)?;
    let mut l = definition.metadata()?;
    if let Some(stats) = stats {
        validate_rule_dtype(rule, Dtype::parse(&stats.dtype)?)?;
    }
    let g = definition.scope == Scope::Checkpoint;
    let bound = if g {
        global.ok_or("Full checkpoint calibration is not ready")?
    } else {
        stats
            .ok_or("Complete selected-tensor calibration is not ready")?
            .max_abs
    };
    let bound = if definition.quantile_bound() {
        let q = stats.unwrap().q99;
        if q == 0. {
            1.
        } else {
            q
        }
    } else if definition.requires_histogram() {
        1.
    } else {
        bound
    };
    let s = if definition.asinh() {
        Some(if g {
            if bound > 0. {
                bound / 100.
            } else {
                1.
            }
        } else {
            let m = stats.unwrap().median_nonzero_abs;
            if m > 0. {
                m
            } else {
                1.
            }
        })
    } else {
        None
    };
    let m = l.as_object_mut().unwrap();
    for(k,v)in json!({"min":-bound,"zero":0.,"max":bound,"s":s,"scope":if g{"complete checkpoint"}else{"complete original tensor"},"units":if s.is_some(){"raw weight on nonlinear scale"}else{"raw weight"},"clipped_fraction":0.,"color_warning":"Finite 8-bit colors merge nearby weights; neutral does not prove exact zero."}).as_object().unwrap(){m.insert(k.clone(),v.clone());}
    if definition.unsigned() {
        m.insert("min".into(), json!(0.));
        m.insert("units".into(), json!("native: absolute raw weight |x|; pooled: mean absolute raw weight mean(|x|); normalized field = mean(|x|)/M"));
        m.insert("palette".into(), json!("sequential-purple-v1"));
        m.insert("field_min".into(), json!(0.));
        m.insert("field_max".into(), json!(if bound == 0. { 0. } else { 1. }));
        m.insert("color_warning".into(), json!("Unsigned magnitude discards sign only for color; original values stay signed. Finite 8-bit colors merge nearby magnitudes; light color does not prove exact zero."));
    }
    if definition.quantile_bound() {
        let stats = stats.unwrap();
        m.insert("q99".into(), json!(stats.q99));
        m.insert("quantile_order_statistics".into(), json!("exact"));
        m.insert(
            "quantile_interpolation".into(),
            json!("linear in F64; final floating-point rounding possible"),
        );
        m.insert("effective_divisor".into(), json!(bound));
        m.insert("zero_quantile_fallback".into(), json!(stats.q99 == 0.));
        m.insert(
            "clipped_fraction".into(),
            json!(stats.robust_clipped_count as f64 / stats.count as f64),
        );
        m.insert("clipped_count".into(), json!(stats.robust_clipped_count));
        m.insert("units".into(), json!("native: raw weight clipped at effective divisor D; pooled: mean of clipped x/D (not a raw weight mean)"));
    }
    if definition.transform == Transform::MagnitudeAsinh {
        m.insert("units".into(), json!("native: transformed absolute weight on median-nonzero asinh scale; pooled: mean of transformed magnitudes, not mean raw magnitude"));
        m.insert("control_semantics".into(), json!("Pointwise permutation-equivariant rule; use identical original-tensor calibration for a matched shuffled field"));
    }
    if definition.requires_histogram() {
        m.insert(
            "rank_method".into(),
            json!("exact absolute-value mid-CDF including zeros in N"),
        );
        m.insert("rank_approximation".into(), json!(false));
        m.insert(
            "histogram_sha256".into(),
            json!(stats.unwrap().histogram_sha256),
        );
        m.insert("units".into(), json!("native: signed absolute-magnitude mid-percentile, dimensionless; pooled: mean signed percentile, not a raw weight or the percentile of a pooled weight"));
        m.insert("color_warning".into(), json!("Tied magnitudes share a percentile; both signed zeros map to zero. Rank colors hide absolute magnitudes and distances."));
    }
    Ok(l)
}
pub struct Mapping {
    pub dtype: Dtype,
    table: Option<Vec<f64>>,
    rule: rules::Selected,
    bound: f64,
    scale: f64,
}
fn transformed(rule: &Definition, v: f64, bound: f64, s: f64) -> f64 {
    if (rule.transform == Transform::MagnitudeLinear && bound == 0.) || !v.is_finite() {
        return 0.;
    }
    let divisor = if rule.asinh() {
        (bound / s).asinh()
    } else {
        bound
    };
    let divisor = if divisor == 0. { 1. } else { divisor };
    (if rule.asinh() {
        (if rule.transform == Transform::MagnitudeAsinh {
            v.abs()
        } else {
            v
        } / s)
            .asinh()
            / divisor
    } else if rule.unsigned() {
        v.abs() / divisor
    } else {
        v / divisor
    })
    .clamp(-1., 1.)
}
impl Mapping {
    pub fn value(&self, bits: u32) -> f64 {
        if let Some(table) = &self.table {
            table[bits as usize]
        } else {
            transformed(self.rule.0, self.dtype.value(bits), self.bound, self.scale)
        }
    }
}
pub fn mapping(rule: &str, l: &Value, dtype: Dtype) -> Result<Mapping> {
    let definition = rules::definition(rule)?;
    definition.validate_dtype(dtype)?;
    require(
        !definition.requires_histogram(),
        "Percentile mapping requires an exact histogram",
    )?;
    let bound = l["max"].as_f64().unwrap();
    let scale = l["s"].as_f64().unwrap_or(1.);
    Ok(Mapping {
        dtype,
        table: if dtype == Dtype::F32 {
            None
        } else {
            Some(
                (0..65536)
                    .map(|bits| transformed(definition, dtype.value(bits), bound, scale))
                    .collect(),
            )
        },
        rule: rules::Selected(definition),
        bound,
        scale,
    })
}
pub fn percentile_mapping(dtype: Dtype, hist: &[u64], count: usize) -> Result<Mapping> {
    rules::SIGNED_PERCENTILE.validate_dtype(dtype)?;
    require(
        hist.len() == 65536 && count > 0,
        "Invalid percentile histogram",
    )?;
    let total = hist
        .iter()
        .try_fold(0u64, |n, &v| n.checked_add(v))
        .ok_or("Histogram count overflow")?;
    require(
        total == count as u64 && total <= u64::MAX / 2,
        "Histogram count mismatch/overflow",
    )?;
    let mut table = vec![0.; 65536];
    let mut before = 0u64;
    for i in 0..32768 {
        let ties = hist[i] + hist[i + 32768];
        require(
            dtype.finite(i as u32) || ties == 0,
            "Nonfinite histogram value",
        )?;
        let mid = (2 * before + ties) as f64 / (2. * total as f64);
        if i != 0 && dtype.finite(i as u32) {
            table[i] = mid;
            table[i + 32768] = -mid;
        }
        before += ties;
    }
    Ok(Mapping {
        dtype,
        table: Some(table),
        rule: rules::Selected(rules::SIGNED_PERCENTILE),
        bound: 1.,
        scale: 1.,
    })
}
pub fn lookup(rule: &str, l: &Value) -> Vec<f64> {
    mapping(rule, l, Dtype::Bf16).unwrap().table.unwrap()
}
#[derive(Clone, Serialize)]
pub struct Metrics {
    pub source_count: usize,
    pub source_bytes_read: usize,
    pub max_raw_band_values: usize,
    pub factor: usize,
    pub width: usize,
    pub height: usize,
}
pub fn tile_fields(
    source: &Source,
    t: &Tensor,
    luts: &[&Mapping],
    level: u32,
    x: usize,
    y: usize,
) -> Result<(Vec<Vec<f64>>, Metrics)> {
    require(level <= t.max_level, "Invalid pyramid level")?;
    let f = 1usize << (t.max_level - level);
    let c0 = x.checked_mul(256 * f).ok_or("Tile coordinate overflow")?;
    let r0 = y.checked_mul(256 * f).ok_or("Tile coordinate overflow")?;
    require(r0 < t.rows && c0 < t.cols, "Tile outside tensor")?;
    let r1 = (r0 + 256 * f).min(t.rows);
    let c1 = (c0 + 256 * f).min(t.cols);
    region_fields(source, t, luts, (r0, r1, c0, c1), f)
}
pub fn region_fields(
    source: &Source,
    t: &Tensor,
    luts: &[&Mapping],
    region: (usize, usize, usize, usize),
    f: usize,
) -> Result<(Vec<Vec<f64>>, Metrics)> {
    region_fields_with_threads(
        source,
        t,
        luts,
        region,
        f,
        crate::resources::policy().cpu_count,
    )
}
fn region_fields_with_threads(
    source: &Source,
    t: &Tensor,
    luts: &[&Mapping],
    region: (usize, usize, usize, usize),
    f: usize,
    requested_threads: usize,
) -> Result<(Vec<Vec<f64>>, Metrics)> {
    require(
        (1..=8).contains(&requested_threads),
        "Rendering threads must be 1–8",
    )?;
    require(
        t.available && t.shape.len() <= 2,
        "Rendering requires an available explicit 2D slice",
    )?;
    let (r0, r1, c0, c1) = region;
    require(
        f > 0 && r0 < r1 && r1 <= t.rows && c0 < c1 && c1 <= t.cols && r0 % f == 0 && c0 % f == 0,
        "Invalid aligned source region",
    )?;
    let dtype = Dtype::parse(&t.dtype)?;
    require(
        !luts.is_empty() && luts.len() <= 4 && luts.iter().all(|l| l.dtype == dtype),
        "Invalid lookups",
    )?;
    let (w, h) = ((c1 - c0).div_ceil(f), (r1 - r0).div_ceil(f));
    require(w * h <= 1048576, "Output exceeds bounded image limit")?;
    source.check()?;
    let threads = requested_threads.min(h);
    let parallel = threads > 1 && (r1 - r0).saturating_mul(c1 - c0) >= 65536;
    let workspace = (w * h * luts.len() * if parallel { 16 } else { 8 }) as u64
        + threads as u64 * (RAW_BAND_BYTES as u64 + 2 * 1024 * 1024);
    let _permit = crate::resources::reserve(workspace.max(crate::resources::JOB_WORKSPACE_BYTES))?;
    if !parallel {
        return region_fields_serial(source, t, luts, region, f);
    }
    // Output-row partitions never split a pooling block. Every pixel retains
    // the original row/chunk arithmetic order; all workers join before return.
    let rows_per_worker = h.div_ceil(threads);
    let results = std::thread::scope(|scope| -> Result<Vec<_>> {
        let mut handles = Vec::new();
        for output_row in (0..h).step_by(rows_per_worker) {
            let begin = r0 + output_row * f;
            let end = (r0 + (output_row + rows_per_worker).min(h) * f).min(r1);
            handles.push(
                std::thread::Builder::new()
                    .name("atlas-render".into())
                    .stack_size(2 * 1024 * 1024)
                    .spawn_scoped(scope, move || {
                        region_fields_serial(source, t, luts, (begin, end, c0, c1), f)
                    })?,
            );
        }
        // Consume ALL joins, including after any earlier worker error. Scoped
        // ownership also joins previously spawned threads if a spawn fails.
        let mut results = Vec::new();
        let mut failure = None;
        for handle in handles {
            match handle.join() {
                Ok(Ok(result)) => results.push(result),
                Ok(Err(error)) => {
                    if failure.is_none() {
                        failure = Some(error);
                    }
                }
                Err(_) => {
                    if failure.is_none() {
                        failure = Some("Render worker failed".into());
                    }
                }
            }
        }
        if let Some(error) = failure {
            Err(error)
        } else {
            Ok(results)
        }
    })?;
    source.check()?;
    let mut fields = (0..luts.len())
        .map(|_| Vec::with_capacity(w * h))
        .collect::<Vec<_>>();
    let mut metrics = Metrics {
        source_count: 0,
        source_bytes_read: 0,
        max_raw_band_values: 0,
        factor: f,
        width: w,
        height: h,
    };
    for (mut parts, part) in results {
        for (field, values) in fields.iter_mut().zip(parts.iter_mut()) {
            field.append(values);
        }
        metrics.source_count += part.source_count;
        metrics.source_bytes_read += part.source_bytes_read;
        metrics.max_raw_band_values = metrics.max_raw_band_values.max(part.max_raw_band_values);
    }
    Ok((fields, metrics))
}

fn region_fields_serial(
    source: &Source,
    t: &Tensor,
    luts: &[&Mapping],
    region: (usize, usize, usize, usize),
    f: usize,
) -> Result<(Vec<Vec<f64>>, Metrics)> {
    let (r0, r1, c0, c1) = region;
    let dtype = Dtype::parse(&t.dtype)?;
    let (w, h) = ((c1 - c0).div_ceil(f), (r1 - r0).div_ceil(f));
    let mut out = vec![vec![0.; w * h]; luts.len()];
    let mut max_raw = 0;
    let mut read = 0;
    // Batch adjacent narrow rows using their real on-disk stride. Read gaps
    // only when amplification is at most 4x; otherwise keep a single narrow row.
    // The entire span, including gaps, remains <=2 MiB.
    let width = c1 - c0;
    let capacity = RAW_BAND_BYTES / dtype.bytes();
    require(width <= capacity, "Source row exceeds raw band budget")?;
    let band_rows = if t.cols <= width.saturating_mul(4) {
        (1 + (capacity - width) / t.cols).min(r1 - r0)
    } else {
        1
    };
    let mut buf = vec![0u8; ((band_rows - 1) * t.cols + width) * dtype.bytes()];
    for row in (r0..r1).step_by(band_rows) {
        headroom()?;
        let rows = band_rows.min(r1 - row);
        let n = (rows - 1) * t.cols + width;
        max_raw = max_raw.max(n);
        source.files[t.shard_id].read_exact_at(
            &mut buf[..dtype.bytes() * n],
            element_offset(
                t,
                row.checked_mul(t.cols)
                    .and_then(|v| v.checked_add(c0))
                    .ok_or("Region offset overflow")?,
                dtype,
            )?,
        )?;
        read += dtype.bytes() * n;
        for rr in 0..rows {
            let oy = (row + rr - r0) / f;
            let raw = &buf[rr * t.cols * dtype.bytes()..(rr * t.cols + width) * dtype.bytes()];
            for (ox, chunk) in raw.chunks(f * dtype.bytes()).enumerate() {
                let index = oy * w + ox;
                for (lut, field) in luts.iter().zip(out.iter_mut()) {
                    let mut sum = 0.;
                    for b in chunk.chunks_exact(dtype.bytes()) {
                        let bits = dtype.bits(b);
                        require(
                            dtype.finite(bits),
                            "Nonfinite source value: rendering refused",
                        )?;
                        sum += lut.value(bits)
                    }
                    field[index] += sum
                }
            }
        }
    }

    source.check()?;
    for field in &mut out {
        for row in 0..h {
            for col in 0..w {
                let count = (r1 - (r0 + row * f)).min(f) * (c1 - (c0 + col * f)).min(f);
                field[row * w + col] /= count as f64
            }
        }
    }
    Ok((
        out,
        Metrics {
            source_count: (r1 - r0) * (c1 - c0),
            source_bytes_read: read,
            max_raw_band_values: max_raw,
            factor: f,
            width: w,
            height: h,
        },
    ))
}
pub fn rgba(field: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(field.len() * 4);
    for &v in field {
        let v = v.clamp(-1., 1.);
        let ends = if v >= 0. {
            [242., 38., 38.]
        } else {
            [37., 99., 242.]
        };
        for e in ends {
            out.push((242. + v.abs() * (e - 242.)).round_ties_even() as u8)
        }
        out.push(255)
    }
    out
}
/// Sequential unsigned palette. Legacy signed palette remains byte-for-byte unchanged.
pub fn rgba_for_rule(rule: &str, field: &[f64]) -> Vec<u8> {
    if !rules::find(rule).is_some_and(Definition::unsigned) {
        return rgba(field);
    }
    let mut out = Vec::with_capacity(field.len() * 4);
    for &v in field {
        let v = v.clamp(0., 1.);
        for (lo, hi) in [(247., 84.), (244., 39.), (249., 143.)] {
            out.push((lo + v * (hi - lo)).round_ties_even() as u8);
        }
        out.push(255);
    }
    out
}
fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc = crc32fast::Hasher::new();
    crc.update(kind);
    crc.update(data);
    out.extend_from_slice(&crc.finalize().to_be_bytes());
}
pub fn png(field: &[f64], w: usize, h: usize) -> Result<Vec<u8>> {
    require(w * h == field.len() && w > 0 && h > 0, "Invalid PNG size")?;
    encode_png(&rgba(field), w, h)
}
pub fn png_for_rule(rule: &str, field: &[f64], w: usize, h: usize) -> Result<Vec<u8>> {
    rule_info(rule)?;
    require(w * h == field.len() && w > 0 && h > 0, "Invalid PNG size")?;
    encode_png(&rgba_for_rule(rule, field), w, h)
}
fn encode_png(pixels: &[u8], w: usize, h: usize) -> Result<Vec<u8>> {
    let mut data = Vec::with_capacity(pixels.len() + h);
    // Sub filter improves weight-image compression without a large image dependency.
    for row in pixels.chunks_exact(w * 4) {
        data.push(1);
        for i in 0..row.len() {
            data.push(row[i].wrapping_sub(if i >= 4 { row[i - 4] } else { 0 }))
        }
    }
    let mut z = ZlibEncoder::new(Vec::new(), Compression::new(3));
    z.write_all(&data)?;
    let compressed = z.finish()?;
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &compressed);
    chunk(&mut out, b"IEND", &[]);
    Ok(out)
}

#[cfg(test)]
mod resource_numeric_tests {
    use super::*;
    use std::io::Write;

    // This oracle has no Mapping/LUT dependency. It transforms each real address
    // before pooling and counts the partial edges independently of band reads.
    fn scalar_oracle(
        values: &[f64; 6],
        counts: &[u64; 6],
        cols: usize,
        region: (usize, usize, usize, usize),
        factor: usize,
        rule: &str,
    ) -> Vec<f64> {
        let (r0, r1, c0, c1) = region;
        let (width, height) = ((c1 - c0).div_ceil(factor), (r1 - r0).div_ceil(factor));
        let mut expected = vec![0.; width * height];
        for row in r0..r1 {
            for col in c0..c1 {
                let v = values[(row * cols + col) % values.len()];
                let transformed = match rule {
                    "tensor_linear" => v / 2.,
                    "tensor_asinh" => v.asinh() / 2f64.asinh(),
                    "tensor_magnitude" => v.abs() / 2.,
                    "tensor_robust99" => v.clamp(-1., 1.),
                    "tensor_signed_percentile" if v == 0. => 0.,
                    "tensor_signed_percentile" => {
                        let (mut before, mut ties) = (0u64, 0u64);
                        for (&other, &count) in values.iter().zip(counts) {
                            if other.abs() < v.abs() {
                                before += count;
                            }
                            if other.abs() == v.abs() {
                                ties += count;
                            }
                        }
                        v.signum() * (2 * before + ties) as f64
                            / (2. * counts.iter().sum::<u64>() as f64)
                    }
                    _ => panic!("Unexpected oracle rule"),
                };
                expected[(row - r0) / factor * width + (col - c0) / factor] += transformed;
            }
        }
        for y in 0..height {
            for x in 0..width {
                expected[y * width + x] /= ((r1 - r0 - y * factor).min(factor)
                    * (c1 - c0 - x * factor).min(factor))
                    as f64;
            }
        }
        expected
    }

    #[test]
    fn bounded_reads_and_scoped_rows_match_an_independent_scalar_oracle() {
        let _isolation = crate::resources::test_workspace_guard();
        let root = std::env::temp_dir().join(format!("atlas-render-bands-{}", std::process::id()));
        // Total fixture payload: 5,773,320 bytes. The supported 131072-column
        // F32 case reaches a real 2 MiB raw span; source axis limits stay intact.
        for (trial, dtype, rows, cols, factor) in [
            (0, Dtype::Bf16, 257, 513, 4),
            (1, Dtype::F16, 257, 513, 4),
            (2, Dtype::F32, 257, 513, 4),
            (3, Dtype::F32, 9, 131072, 4),
        ] {
            let model = root.join(format!("model-{trial}"));
            std::fs::create_dir_all(&model).unwrap();
            let header = serde_json::json!({"matrix":{"dtype":dtype.name(),"shape":[rows,cols],
                "data_offsets":[0,rows*cols*dtype.bytes()]}})
            .to_string();
            let mut file = std::fs::File::create(model.join("fixture.safetensors")).unwrap();
            file.write_all(&(header.len() as u64).to_le_bytes())
                .unwrap();
            file.write_all(header.as_bytes()).unwrap();
            // Exact signed values and subnormals exercise cancellation and finite
            // dtype decoding. The independent oracle uses their mathematical values.
            let (patterns, tiny): ([u32; 6], f64) = match dtype {
                Dtype::Bf16 => ([0xbf80, 0x8001, 0x3f80, 1, 0, 0x4000], 2f64.powi(-133)),
                Dtype::F16 => ([0xbc00, 0x8001, 0x3c00, 1, 0, 0x4000], 2f64.powi(-24)),
                Dtype::F32 => (
                    [0xbf800000, 0x80000001, 0x3f800000, 1, 0, 0x40000000],
                    2f64.powi(-149),
                ),
            };
            let values = [-1., -tiny, 1., tiny, 0., 2.];
            let mut counts = [0u64; 6];
            let mut bytes = Vec::with_capacity(rows * cols * dtype.bytes());
            for index in 0..rows * cols {
                bytes.extend_from_slice(
                    &patterns[index % patterns.len()].to_le_bytes()[..dtype.bytes()],
                );
                counts[index % patterns.len()] += 1;
            }
            file.write_all(&bytes).unwrap();
            drop(file);
            drop(bytes);
            let source = Source::open(&model).unwrap();
            let tensor = source.tensor(0).unwrap();
            assert!(
                tensor.available,
                "{}",
                tensor
                    .unavailable_reason
                    .as_deref()
                    .unwrap_or("unavailable")
            );
            let linear = mapping("tensor_linear", &serde_json::json!({"max":2.0}), dtype).unwrap();
            let asinh = mapping(
                "tensor_asinh",
                &serde_json::json!({"max":2.0,"s":1.0}),
                dtype,
            )
            .unwrap();
            let magnitude =
                mapping("tensor_magnitude", &serde_json::json!({"max":2.0}), dtype).unwrap();
            let robust =
                mapping("tensor_robust99", &serde_json::json!({"max":1.0}), dtype).unwrap();
            let percentile = if dtype == Dtype::F32 {
                assert!(validate_rule_dtype("tensor_signed_percentile", dtype).is_err());
                None
            } else {
                let mut hist = vec![0u64; 65536];
                for (&bits, &count) in patterns.iter().zip(&counts) {
                    hist[bits as usize] += count;
                }
                Some(percentile_mapping(dtype, &hist, tensor.count).unwrap())
            };
            let mut groups = vec![
                vec![&linear, &asinh],
                vec![&linear, &asinh, &magnitude, &robust],
            ];
            if let Some(percentile) = &percentile {
                groups.push(vec![&linear, &asinh, &magnitude, percentile]);
            }
            for region in [
                (0, rows, 0, cols),
                (0, rows, 4, 260),
                (4, rows, 4, cols - 1),
            ] {
                let (r0, r1, c0, c1) = region;
                for luts in &groups {
                    let expected = luts
                        .iter()
                        .map(|lut| scalar_oracle(&values, &counts, cols, region, factor, &lut.rule))
                        .collect::<Vec<_>>();
                    let (serial, _) =
                        region_fields_with_threads(&source, tensor, luts, region, factor, 1)
                            .unwrap();
                    for threads in [1, 2, 4, 8] {
                        let (fields, metrics) = region_fields_with_threads(
                            &source, tensor, luts, region, factor, threads,
                        )
                        .unwrap();
                        for ((field, baseline), oracle) in fields.iter().zip(&serial).zip(&expected)
                        {
                            assert_eq!(
                                field.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                baseline.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                            );
                            for (&actual, &expected) in field.iter().zip(oracle) {
                                assert!(
                                    (actual - expected).abs() <= 1e-14,
                                    "{actual} != {expected}"
                                );
                            }
                        }
                        assert!(metrics.max_raw_band_values * dtype.bytes() <= RAW_BAND_BYTES);
                        if trial == 3 && region == (0, rows, 0, cols) {
                            assert_eq!(metrics.max_raw_band_values * dtype.bytes(), RAW_BAND_BYTES);
                        }
                        assert!(
                            metrics.source_bytes_read <= (r1 - r0) * (c1 - c0) * dtype.bytes() * 4
                        );
                        assert_eq!(metrics.source_count, (r1 - r0) * (c1 - c0));
                        // Widely-strided narrow rows read exactly their selected addresses.
                        if trial == 3 && region == (0, rows, 4, 260) {
                            assert_eq!(
                                metrics.source_bytes_read,
                                metrics.source_count * dtype.bytes()
                            );
                        }
                    }
                }
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}
