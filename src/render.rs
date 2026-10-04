use crate::{
    headroom, require,
    source::{element_offset, Dtype, Source, Tensor},
    Result,
};
use flate2::{write::ZlibEncoder, Compression};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::BTreeMap, io::Write, os::unix::fs::FileExt, time::Instant};
pub const RULES: [&str; 8] = [
    "global_linear",
    "global_asinh",
    "tensor_linear",
    "tensor_asinh",
    "tensor_magnitude",
    "tensor_magnitude_asinh",
    "tensor_robust99",
    "tensor_signed_percentile",
];
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
    let (title, formula) = match rule {
        "global_linear" => ("Global linear", "clip(x/G, -1, 1)"),
        "global_asinh" => (
            "Global asinh",
            "clip(asinh(x/s)/asinh(G/s), -1, 1); s = G/100 (all-zero fallback 1)",
        ),
        "tensor_linear" => ("Tensor linear", "clip(x/M, -1, 1)"),
        "tensor_asinh" => (
            "Tensor asinh",
            "clip(asinh(x/s)/asinh(M/s), -1, 1); s = median nonzero |x| (all-zero fallback 1)",
        ),
        "tensor_magnitude" => (
            "Tensor magnitude",
            "f(x) = abs(x)/M; M = max(abs(x)) over the complete tensor; M = 0 => f(x) = 0",
        ),
        "tensor_magnitude_asinh" => ("Tensor typical magnitude", "min(1, asinh(abs(x)/s)/asinh(D/s)); s = median nonzero |x| (all-zero fallback 1); D = Q99 including zeros (zero-Q99 fallback 1)"),
        "tensor_robust99" => ("Tensor robust 99%", "clip(x/D, -1, 1); Q99 = linear-interpolated 99th percentile of all original |x|; D = Q99 if Q99 != 0, else 1"),
        "tensor_signed_percentile" => ("Tensor signed percentile", "sign(x) * (count(|X| < |x|) + count(|X| <= |x|))/(2*N); X = complete original tensor including zeros; x = 0 => 0"),
        _ => return Err("Unknown color rule".into()),
    };
    Ok(
        json!({"id":rule,"title":title,"formula":formula,"supported_dtypes":if rule=="tensor_signed_percentile"{vec!["BF16","F16"]}else{vec!["BF16","F16","F32"]},"availability_note":if rule=="tensor_signed_percentile"{"F32 exact absolute-value rank index is not implemented; no approximate ranks are substituted."}else{""}}),
    )
}
pub fn validate_rule_dtype(rule: &str, dtype: Dtype) -> Result<()> {
    rule_info(rule)?;
    require(!(rule == "tensor_signed_percentile" && dtype == Dtype::F32), "Tensor signed percentile is unsupported for F32: exact absolute-value rank index is pending; no approximation is used")
}
pub fn legend(rule: &str, stats: Option<&Stats>, global: Option<f64>) -> Result<Value> {
    let mut l = rule_info(rule)?;
    if let Some(stats) = stats {
        validate_rule_dtype(rule, Dtype::parse(&stats.dtype)?)?;
    }
    let g = rule.starts_with("global");
    let bound = if g {
        global.ok_or("Full checkpoint calibration is not ready")?
    } else {
        stats
            .ok_or("Complete selected-tensor calibration is not ready")?
            .max_abs
    };
    let bound = if matches!(rule, "tensor_robust99" | "tensor_magnitude_asinh") {
        let q = stats.unwrap().q99;
        if q == 0. {
            1.
        } else {
            q
        }
    } else if rule == "tensor_signed_percentile" {
        1.
    } else {
        bound
    };
    let s = if rule.ends_with("asinh") {
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
    if matches!(rule, "tensor_magnitude" | "tensor_magnitude_asinh") {
        m.insert("min".into(), json!(0.));
        m.insert("units".into(), json!("native: absolute raw weight |x|; pooled: mean absolute raw weight mean(|x|); normalized field = mean(|x|)/M"));
        m.insert("palette".into(), json!("sequential-purple-v1"));
        m.insert("field_min".into(), json!(0.));
        m.insert("field_max".into(), json!(if bound == 0. { 0. } else { 1. }));
        m.insert("color_warning".into(), json!("Unsigned magnitude discards sign only for color; original values stay signed. Finite 8-bit colors merge nearby magnitudes; light color does not prove exact zero."));
    }
    if matches!(rule, "tensor_robust99" | "tensor_magnitude_asinh") {
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
    if rule == "tensor_magnitude_asinh" {
        m.insert("units".into(), json!("native: transformed absolute weight on median-nonzero asinh scale; pooled: mean of transformed magnitudes, not mean raw magnitude"));
        m.insert("control_semantics".into(), json!("Pointwise permutation-equivariant rule; use identical original-tensor calibration for a matched shuffled field"));
    }
    if rule == "tensor_signed_percentile" {
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
    rule: String,
    bound: f64,
    scale: f64,
}
fn transformed(rule: &str, v: f64, bound: f64, s: f64) -> f64 {
    if (rule == "tensor_magnitude" && bound == 0.) || !v.is_finite() {
        return 0.;
    }
    let divisor = if rule.ends_with("asinh") {
        (bound / s).asinh()
    } else {
        bound
    };
    let divisor = if divisor == 0. { 1. } else { divisor };
    (if rule.ends_with("asinh") {
        (if rule == "tensor_magnitude_asinh" {
            v.abs()
        } else {
            v
        } / s)
            .asinh()
            / divisor
    } else if matches!(rule, "tensor_magnitude" | "tensor_magnitude_asinh") {
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
            transformed(&self.rule, self.dtype.value(bits), self.bound, self.scale)
        }
    }
}
pub fn mapping(rule: &str, l: &Value, dtype: Dtype) -> Result<Mapping> {
    validate_rule_dtype(rule, dtype)?;
    require(
        rule != "tensor_signed_percentile",
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
                    .map(|bits| transformed(rule, dtype.value(bits), bound, scale))
                    .collect(),
            )
        },
        rule: rule.into(),
        bound,
        scale,
    })
}
pub fn percentile_mapping(dtype: Dtype, hist: &[u64], count: usize) -> Result<Mapping> {
    validate_rule_dtype("tensor_signed_percentile", dtype)?;
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
        rule: "tensor_signed_percentile".into(),
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
    let mut out = vec![vec![0.; w * h]; luts.len()];
    let mut max_raw = 0;
    let mut read = 0;
    // Batch adjacent narrow rows using their real on-disk stride. Read gaps
    // only when amplification is at most 4x; otherwise keep a single narrow row.
    // The entire span, including gaps, remains <=2 MiB.
    let width = c1 - c0;
    let capacity = RAW_BAND_BYTES / dtype.bytes();
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
    if !matches!(rule, "tensor_magnitude" | "tensor_magnitude_asinh") {
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
