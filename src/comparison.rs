//! Read-only, explicitly paired checkpoints. Comparison coordinates are never edit targets.
use crate::{
    atomic_write, disk_guard, headroom, render, require, sha,
    source::{element_offset, exact_decimal_for, Dtype, Format, Source, Tensor},
    state::{intended_cache_path, TileCache},
    Result,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    os::{fd::AsRawFd, unix::fs::FileExt},
    path::{Path, PathBuf},
    sync::Mutex,
};

mod response;

pub const VERSION: &str = "checkpoint-comparison-v1";
const RENDERER: &str = "comparison-v2-exact-json-transform-before-mean";
const BAND: usize = 2 * 1024 * 1024;
#[derive(Clone, Serialize)]
pub struct Pair {
    pub id: usize,
    pub name: String,
    pub shape: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
    pub count: usize,
    pub max_level: u32,
    pub dtype_a: Format,
    pub dtype_b: Format,
    a: usize,
    b: usize,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Scales {
    pub count: usize,
    pub shared_raw_max: f64,
    pub difference_max: f64,
    pub nonzero_difference_count: usize,
}
#[derive(Clone, Serialize, Deserialize)]
struct Saved {
    version: u32,
    comparison_identity: String,
    tensors: BTreeMap<usize, Scales>,
}
#[derive(Default, Serialize)]
pub struct Progress {
    pub active: Option<usize>,
    pub error: Option<String>,
}
pub struct Comparison {
    pub a: Source,
    pub b: Source,
    pub pairs: Vec<Pair>,
    pub identity: String,
    root: PathBuf,
    calibration: Mutex<Saved>,
    pub calibration_note: String,
    pub progress: Mutex<Progress>,
    cache: Mutex<TileCache>,
    compute: Mutex<()>,
    _lock: File,
}
fn compatible(a: &Source, b: &Source) -> Result<Vec<Pair>> {
    let by_b = b
        .tensors
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect::<BTreeMap<_, _>>();
    let by_a = a
        .tensors
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect::<BTreeMap<_, _>>();
    let missing = by_a
        .keys()
        .filter(|n| !by_b.contains_key(**n))
        .copied()
        .collect::<Vec<_>>();
    let extra = by_b
        .keys()
        .filter(|n| !by_a.contains_key(**n))
        .copied()
        .collect::<Vec<_>>();
    let shape = a
        .tensors
        .iter()
        .filter(|t| {
            by_b.get(t.name.as_str())
                .is_some_and(|u| t.shape != u.shape)
        })
        .map(|t| t.name.as_str())
        .collect::<Vec<_>>();
    require(
        missing.is_empty() && extra.is_empty() && shape.is_empty(),
        &format!(
            "Complete named-shape compatibility failed: {}",
            json!({"missing_in_b_count":missing.len(),"extra_in_b_count":extra.len(),"shape_mismatch_count":shape.len(),"missing_in_b":missing.iter().take(20).collect::<Vec<_>>(),"extra_in_b":extra.iter().take(20).collect::<Vec<_>>(),"shape_mismatch":shape.iter().take(20).collect::<Vec<_>>() })
        ),
    )?;
    a.tensors
        .iter()
        .enumerate()
        .map(|(id, t)| {
            let u = by_b[t.name.as_str()];
            require(t.available && u.available && t.shape.len() <= 2 && u.shape.len() <= 2, "Comparison currently requires supported vectors/matrices; higher-rank slices unavailable")?;
            Dtype::parse(&t.dtype)?;
            Dtype::parse(&u.dtype)?;
            Ok(Pair {
                id,
                name: t.name.clone(),
                shape: t.shape.clone(),
                rows: t.rows,
                cols: t.cols,
                count: t.count,
                max_level: t.max_level,
                dtype_a: t.dtype.clone(),
                dtype_b: u.dtype.clone(),
                a: t.id,
                b: u.id,
            })
        })
        .collect()
}
impl Comparison {
    pub fn open(model_a: &Path, model_b: &Path, cache: &Path) -> Result<Self> {
        let a = Source::open(model_a)?;
        let b = Source::open(model_b)?;
        let pairs = compatible(&a, &b)?;
        let identity = sha(&serde_json::to_vec(
            &json!({"schema":VERSION,"a":a.identity,"b":b.identity,"pairs":pairs,"direction":"B-A"}),
        )?);
        let intended = intended_cache_path(cache)?;
        require(
            !intended.starts_with(&a.root) && !intended.starts_with(&b.root),
            "Comparison cache must be outside both source directories",
        )?;
        std::fs::create_dir_all(&intended)?;
        let root = intended.canonicalize()?;
        require(
            !root.starts_with(&a.root) && !root.starts_with(&b.root),
            "Comparison cache must be outside both source directories",
        )?;
        disk_guard(&root)?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.join("atlas.lock"))?;
        // SAFETY: live owned descriptor; lock released when Comparison is dropped.
        require(
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "Cache is in use; choose a separate --cache directory",
        )?;
        let loaded = (|| -> Result<Saved> {
            let path = root.join("comparison-calibration.json");
            let bytes = crate::source::read_small(&path, 8 * 1024 * 1024)?;
            let envelope: Value = serde_json::from_slice(&bytes)?;
            let payload = envelope["payload"]
                .as_str()
                .ok_or("Invalid comparison calibration")?;
            require(
                envelope["sha256"] == sha(payload.as_bytes()),
                "Comparison calibration checksum mismatch",
            )?;
            let saved: Saved = serde_json::from_str(payload)?;
            require(
                saved.version == 2 && saved.comparison_identity == identity,
                "Comparison calibration identity/schema changed",
            )?;
            for (&id, s) in &saved.tensors {
                let p = pairs.get(id).ok_or("Unknown calibration pair")?;
                require(
                    s.count == p.count
                        && s.shared_raw_max.is_finite()
                        && s.shared_raw_max >= 0.
                        && s.difference_max.is_finite()
                        && s.difference_max >= 0.
                        && s.difference_max <= 2. * s.shared_raw_max
                        && s.nonzero_difference_count <= s.count,
                    "Invalid comparison calibration scales",
                )?;
            }
            Ok(saved)
        })();
        let (saved, note) = match loaded {
            Ok(s) => (s, "Validated comparison calibration resumed".into()),
            Err(e) => (
                Saved {
                    version: 2,
                    comparison_identity: identity.clone(),
                    tensors: BTreeMap::new(),
                },
                format!("Comparison calibration not resumed: {e}"),
            ),
        };
        let cache = TileCache::open(&root.join("tiles"))?;
        let result = Self {
            a,
            b,
            pairs,
            identity,
            root,
            calibration: Mutex::new(saved),
            calibration_note: note,
            progress: Mutex::new(Progress::default()),
            cache: Mutex::new(cache),
            compute: Mutex::new(()),
            _lock: lock,
        };
        result.check()?;
        Ok(result)
    }
    pub fn output_prefix(&self, path: &Path) -> Result<PathBuf> {
        let intended = intended_cache_path(path)?;
        require(
            !intended.starts_with(&self.a.root) && !intended.starts_with(&self.b.root),
            "Comparison output must be outside both source directories",
        )?;
        Ok(intended)
    }
    pub fn check(&self) -> Result<()> {
        self.a.check()?;
        self.b.check()
    }
    pub fn pair(&self, id: usize) -> Result<&Pair> {
        self.pairs
            .get(id)
            .ok_or_else(|| "Unknown comparison pair".into())
    }
    pub fn scales(&self, id: usize) -> Option<Scales> {
        self.calibration.lock().unwrap().tensors.get(&id).cloned()
    }
    pub fn identity_metadata(&self) -> Value {
        Value::from(response::Identity {
            version: VERSION,
            comparison_identity: &self.identity,
            source_a_identity: &self.a.identity,
            source_a_directory: &self.a.root,
            source_b_identity: &self.b.identity,
            source_b_directory: &self.b.root,
        })
    }
    pub fn model(&self) -> Result<Value> {
        self.check()?;
        let v = self.identity_metadata();
        let saved = self.calibration.lock().unwrap();
        let catalog = self
            .pairs
            .iter()
            .map(|p| {
                let mut t = json!(p);
                t["calibration_complete"] = json!(saved.tensors.contains_key(&p.id));
                t["scales"] = json!(saved.tensors.get(&p.id));
                t
            })
            .collect::<Vec<_>>();
        let mut v = Value::from(response::Model {
            identity: v,
            catalog,
            tensor_count: self.pairs.len(),
            calibration_note: &self.calibration_note,
        });
        v["progress"] = json!(*self.progress.lock().unwrap());
        Ok(v)
    }
    /// Both input buffers together stay <= BAND, even for wide rows or mixed dtypes.
    fn scan_range(
        &self,
        p: &Pair,
        start: usize,
        count: usize,
        mut visit: impl FnMut(usize, f64, f64),
    ) -> Result<usize> {
        let a = self.a.tensor(p.a)?;
        let b = self.b.tensor(p.b)?;
        let da = Dtype::parse(&a.dtype)?;
        let db = Dtype::parse(&b.dtype)?;
        require(
            start.checked_add(count).is_some_and(|end| end <= p.count),
            "Comparison source range out of bounds",
        )?;
        let capacity = (BAND / (da.bytes() + db.bytes())).min(count.max(1));
        let mut ba = vec![0; capacity * da.bytes()];
        let mut bb = vec![0; capacity * db.bytes()];
        for done in (0..count).step_by(capacity) {
            headroom()?;
            let n = capacity.min(count - done);
            self.a.files[a.shard_id].read_exact_at(
                &mut ba[..n * da.bytes()],
                element_offset(a, start + done, da)?,
            )?;
            self.b.files[b.shard_id].read_exact_at(
                &mut bb[..n * db.bytes()],
                element_offset(b, start + done, db)?,
            )?;
            for i in 0..n {
                let va = da.bits(&ba[i * da.bytes()..(i + 1) * da.bytes()]);
                let vb = db.bits(&bb[i * db.bytes()..(i + 1) * db.bytes()]);
                require(
                    da.finite(va) && db.finite(vb),
                    "Nonfinite paired source value: comparison calibration/rendering refused",
                )?;
                visit(start + done + i, da.value(va), db.value(vb));
            }
        }
        Ok(capacity * (da.bytes() + db.bytes()))
    }
    pub fn calibrate_one(&self, id: usize) -> Result<()> {
        let _compute = self.compute.lock().unwrap();
        self.check()?;
        let p = self.pair(id)?;
        if self.scales(id).is_some() {
            return Ok(());
        }
        {
            let mut progress = self.progress.lock().unwrap();
            progress.active = Some(id);
            progress.error = None;
        }
        let result = (|| -> Result<()> {
            let mut s = Scales {
                count: p.count,
                shared_raw_max: 0.,
                difference_max: 0.,
                nonzero_difference_count: 0,
            };
            self.scan_range(p, 0, p.count, |_, a, b| {
                s.shared_raw_max = s.shared_raw_max.max(a.abs()).max(b.abs());
                let d = b - a;
                s.difference_max = s.difference_max.max(d.abs());
                s.nonzero_difference_count += usize::from(d != 0.);
            })?;
            self.check()?;
            let mut next = self.calibration.lock().unwrap().clone();
            next.tensors.insert(id, s);
            let payload = serde_json::to_string(&next)?;
            atomic_write(
                &self.root.join("comparison-calibration.json"),
                &serde_json::to_vec(&json!({"sha256":sha(payload.as_bytes()),"payload":payload}))?,
            )?;
            *self.calibration.lock().unwrap() = next;
            Ok(())
        })();
        let mut progress = self.progress.lock().unwrap();
        progress.active = None;
        progress.error = result.as_ref().err().map(|e| e.to_string());
        result
    }
    pub fn legend(&self, id: usize, quantity: &str, mapping: &str) -> Result<Value> {
        validate(quantity, mapping)?;
        self.pair(id)?;
        let s = self
            .scales(id)
            .ok_or("Complete paired-tensor calibration is not ready")?;
        let derived = quantity == "delta" || quantity == "abs_delta";
        let bound = if derived {
            s.difference_max
        } else {
            s.shared_raw_max
        };
        let unsigned = quantity == "abs_delta" || mapping == "magnitude";
        Ok(
            json!({"quantity":quantity,"mapping":mapping,"original_source_values":!derived,"derived":derived,"difference_direction":"B-A","min":if unsigned{0.}else{-bound},"max":bound,"bound":bound,"s":if mapping=="asinh"{Some(if bound==0.{1.}else{bound/100.})}else{None},"scope":if derived{"complete paired tensor; shared derived B-A scale"}else{"complete paired tensor; shared original A+B scale"},"calibration_domain":if derived{"difference"}else{"shared_raw"},"palette":if unsigned{"sequential-purple-v1"}else{"signed-blue-red"},"formula":match mapping{"linear"=>"v/bound; bound=0 => 0","asinh"=>"asinh(v/s)/asinh(bound/s); s=bound/100; bound=0 => 0",_=>"abs(v)/bound; bound=0 => 0"},"value_definition":match quantity{"a"=>"original source A weight","b"=>"original source B weight","delta"=>"derived F64 arithmetic B-A; not an original weight",_=>"derived abs(F64 B-A); not an original weight"},"units":"native: selected quantity on labeled scale; pooled: mean of pointwise transformed quantity, not transform of a pooled weight","rounding":"A/B decode exactly to F64; derived subtraction and field arithmetic may round"}),
        )
    }
    pub fn view(&self, id: usize, left: &str, right: &str, mapping: &str) -> Result<Value> {
        self.check()?;
        let p = self.pair(id)?;
        let v = self.identity_metadata();
        let pair = json!(p);
        let legends =
            json!({"left":self.legend(id,left,mapping)?,"right":self.legend(id,right,mapping)?});
        Ok(Value::from(response::View {
            identity: v,
            pair,
            legends,
        }))
    }
    pub fn inspect(&self, id: usize, row: usize, col: usize) -> Result<Value> {
        self.check()?;
        let p = self.pair(id)?;
        let (a, av) = scalar(&self.a, self.a.tensor(p.a)?, row, col)?;
        let (b, bv) = scalar(&self.b, self.b.tensor(p.b)?, row, col)?;
        self.check()?;
        let finite = av.is_finite() && bv.is_finite();
        let delta = finite.then_some(bv - av);
        let v = self.identity_metadata();
        Ok(Value::from(response::Inspection {
            identity: v,
            pair_id: id,
            name: &p.name,
            row,
            col,
            native_indices: if p.shape.len() == 1 {
                vec![col]
            } else {
                vec![row, col]
            },
            original_a: a,
            original_b: b,
            delta,
            finite,
        }))
    }
    pub fn fields(
        &self,
        id: usize,
        quantity: &str,
        mapping: &str,
        level: u32,
        x: usize,
        y: usize,
    ) -> Result<(Vec<f64>, render::Metrics)> {
        let l = self.legend(id, quantity, mapping)?;
        let p = self.pair(id)?;
        let (factor, r0, r1, c0, c1, w, h) = geometry(p, level, x, y)?;
        self.check()?;
        let mut field = vec![0.; w * h];
        let mut max_band = 0;
        let bound = l["bound"].as_f64().unwrap();
        for row in r0..r1 {
            max_band =
                max_band.max(
                    self.scan_range(p, row * p.cols + c0, c1 - c0, |index, a, b| {
                        let ox = (index % p.cols - c0) / factor;
                        let oy = (row - r0) / factor;
                        let value = match quantity {
                            "a" => a,
                            "b" => b,
                            "delta" => b - a,
                            _ => (b - a).abs(),
                        };
                        field[oy * w + ox] += transform(value, mapping, bound);
                    })?,
                );
        }
        self.check()?;
        for oy in 0..h {
            for ox in 0..w {
                field[oy * w + ox] /= ((r1 - r0 - oy * factor).min(factor)
                    * (c1 - c0 - ox * factor).min(factor))
                    as f64;
            }
        }
        let count = (r1 - r0) * (c1 - c0);
        let bytes = Dtype::parse(&p.dtype_a)?.bytes() + Dtype::parse(&p.dtype_b)?.bytes();
        Ok((
            field,
            render::Metrics {
                source_count: count * 2,
                source_bytes_read: count * bytes,
                max_raw_band_values: max_band / bytes * 2,
                factor,
                width: w,
                height: h,
            },
        ))
    }
    pub fn tile(
        &self,
        id: usize,
        quantity: &str,
        mapping: &str,
        level: u32,
        x: usize,
        y: usize,
    ) -> Result<(Vec<u8>, bool, render::Metrics)> {
        self.check()?;
        let legend = self.legend(id, quantity, mapping)?;
        let p = self.pair(id)?;
        let (factor, _, _, _, _, width, height) = geometry(p, level, x, y)?;
        let key = sha(&serde_json::to_vec(
            &json!({"renderer":RENDERER,"comparison_identity":self.identity,"pair":p,"direction":"B-A","quantity":quantity,"mapping":mapping,"legend":legend,"level":level,"x":x,"y":y}),
        )?);
        if let Some(png) = self.cache.lock().unwrap().get(&key)? {
            return Ok((
                png,
                true,
                render::Metrics {
                    source_count: 0,
                    source_bytes_read: 0,
                    max_raw_band_values: 0,
                    factor,
                    width,
                    height,
                },
            ));
        }
        let _compute = self.compute.lock().unwrap();
        let (field, metrics) = self.fields(id, quantity, mapping, level, x, y)?;
        let rule = if quantity == "abs_delta" || mapping == "magnitude" {
            "tensor_magnitude"
        } else {
            "tensor_linear"
        };
        let png = render::png_for_rule(rule, &field, metrics.width, metrics.height)?;
        self.cache.lock().unwrap().put(&key, &png)?;
        Ok((png, false, metrics))
    }
}
pub fn validate(quantity: &str, mapping: &str) -> Result<()> {
    require(
        ["a", "b", "delta", "abs_delta"].contains(&quantity),
        "Unsupported comparison quantity",
    )?;
    require(
        ["linear", "asinh", "magnitude"].contains(&mapping),
        "Unsupported comparison mapping; exact paired robust/percentile calibration is pending",
    )
}
fn transform(v: f64, mapping: &str, bound: f64) -> f64 {
    if bound == 0. {
        return 0.;
    }
    match mapping {
        "linear" => v / bound,
        "magnitude" => v.abs() / bound,
        _ => {
            let s = bound / 100.;
            (v / s).asinh() / (bound / s).asinh()
        }
    }
    .clamp(-1., 1.)
}
fn scalar(source: &Source, t: &Tensor, row: usize, col: usize) -> Result<(Value, f64)> {
    let (bits, offset) = source.scalar_bits(t, row, col)?;
    let dtype = Dtype::parse(&t.dtype)?;
    let value = dtype.value(bits);
    let raw = bits.to_le_bytes()[..dtype.bytes()]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    Ok((
        json!({"raw_exact":exact_decimal_for(dtype,bits),"raw_hex_le":raw,"dtype":dtype.name(),"element_bytes":dtype.bytes(),"shard":t.shard,"byte_offset":offset,"classification":if value.is_nan(){"nan"}else if value.is_infinite(){"infinity"}else{"finite"},"original_source_value":true}),
        value,
    ))
}
fn geometry(
    p: &Pair,
    level: u32,
    x: usize,
    y: usize,
) -> Result<(usize, usize, usize, usize, usize, usize, usize)> {
    require(level <= p.max_level, "Invalid comparison level")?;
    let f = 1usize << (p.max_level - level);
    let span = 256usize.checked_mul(f).ok_or("Tile span overflow")?;
    let c0 = x.checked_mul(span).ok_or("Tile x overflow")?;
    let r0 = y.checked_mul(span).ok_or("Tile y overflow")?;
    require(
        c0 < p.cols && r0 < p.rows,
        "Comparison tile outside native shape",
    )?;
    let c1 = c0 + span.min(p.cols - c0);
    let r1 = r0 + span.min(p.rows - r0);
    Ok((
        f,
        r0,
        r1,
        c0,
        c1,
        (c1 - c0).div_ceil(f),
        (r1 - r0).div_ceil(f),
    ))
}
