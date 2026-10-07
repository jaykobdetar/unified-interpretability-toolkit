mod response;
use crate::{
    atomic_write, disk_guard,
    render::{self, Stats, RULES},
    require, sha,
    slice::TensorSlice,
    source::{Dtype, Source, Tensor},
    Result,
};
use flate2::{read::ZlibDecoder, write::ZlibEncoder, Compression};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, VecDeque},
    fs::File,
    io::{Read, Write},
    os::fd::AsRawFd,
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Instant, SystemTime},
};
pub const CALIBRATION_VERSION: u32 = 4;
pub const RENDER_VERSION: &str = "rust-rgba-v5-exact-json-f64-row-sums-subfilter";

// Resolve existing components (including symlinks) and normalize missing ones
// without creating anything. mkdir uses this destination, not the original path.
pub(crate) fn intended_cache_path(cache: &Path) -> Result<PathBuf> {
    let absolute = if cache.is_absolute() {
        cache.to_owned()
    } else {
        std::env::current_dir()?.join(cache)
    };
    let mut resolved = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => continue,
            Component::ParentDir => {
                resolved.pop();
                continue;
            }
            _ => resolved.push(component.as_os_str()),
        }
        match resolved.canonicalize() {
            Ok(path) => resolved = path,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // A dangling symlink is not a missing directory to create.
                match std::fs::symlink_metadata(&resolved) {
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => return Err(e.into()),
                    Ok(_) => return Err(e.into()),
                }
            }
            Err(e) => return Err(e.into()),
        }
    }
    Ok(resolved)
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Calibration {
    pub version: u32,
    pub source_identity: String,
    pub tensors: BTreeMap<usize, Stats>,
}
#[derive(Default)]
pub struct Progress {
    pub active: Option<usize>,
    pub all_requested: bool,
    pub error: Option<String>,
}
pub struct State {
    pub source: Source,
    pub root: PathBuf,
    pub calibration: Mutex<Calibration>,
    pub progress: Mutex<Progress>,
    pub calibration_note: String,
    pub started: Instant,
    pub name: String,
    pub revision: String,
    pub lookups: Mutex<VecDeque<(String, Arc<render::Mapping>)>>,
    pub cache: Mutex<TileCache>,
    pub compute: Mutex<()>,
    pub fresh_hashes: Mutex<Option<Value>>,
    _lock: File,
}
// Domain-separated, opaque source/revision binding; no filesystem path is emitted.
fn revision_identity(source_identity: &str, revision: &str) -> String {
    sha(json!(["weight-atlas-model-v1", source_identity, revision])
        .to_string()
        .as_bytes())
}
fn browser_tile_binding(
    model: &str,
    source: &str,
    slice: &str,
    rule: &str,
    legend: &Value,
) -> String {
    sha(json!([
        "weight-atlas-browser-tile-v1",
        model,
        source,
        slice,
        RENDER_VERSION,
        rule,
        legend
    ])
    .to_string()
    .as_bytes())
}
impl State {
    pub fn model_identity(&self) -> String {
        revision_identity(&self.source.identity, &self.revision)
    }
    pub fn source_binding(&self, tensor: &Tensor) -> Value {
        // Compatibility helper for original vectors/matrices only.
        TensorSlice::new(&self.source, tensor.id, &[])
            .map(|s| s.binding(&self.model_identity()))
            .unwrap_or(Value::Null)
    }
    pub fn slice_binding(&self, slice: &TensorSlice) -> Value {
        slice.binding(&self.model_identity())
    }
    pub fn open(
        model: &Path,
        cache: &Path,
        name: Option<String>,
        revision: Option<String>,
    ) -> Result<Self> {
        let started = Instant::now();
        let source = Source::open(model)?;
        let intended = intended_cache_path(cache)?;
        require(
            !intended.starts_with(&source.root),
            "Cache must be outside the read-only chosen model directory",
        )?;
        std::fs::create_dir_all(&intended)?;
        let root = intended.canonicalize()?;
        require(
            !root.starts_with(&source.root),
            "Cache must be outside the read-only chosen model directory",
        )?;
        disk_guard(&root)?;
        let lock = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.join("atlas.lock"))?;
        // SAFETY: valid owned descriptor; flock is released with File lifetime.
        require(
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "Cache is in use; choose a separate --cache directory",
        )?;
        let empty = Calibration {
            version: CALIBRATION_VERSION,
            source_identity: source.identity.clone(),
            tensors: BTreeMap::new(),
        };
        let loaded = (|| -> Result<Calibration> {
            let p = root.join("calibration.json");
            require(
                std::fs::metadata(&p)?.len() < 8 * 1024 * 1024,
                "Calibration cache too large",
            )?;
            let v: Value = serde_json::from_slice(&std::fs::read(p)?)?;
            let payload = v["payload"].as_str().ok_or("Invalid calibration cache")?;
            require(
                v["sha256"] == sha(payload.as_bytes()),
                "Calibration checksum mismatch",
            )?;
            let c: Calibration = serde_json::from_str(payload)?;
            require(
                c.version == CALIBRATION_VERSION && c.source_identity == source.identity,
                "Calibration identity/schema changed",
            )?;
            for (&id, s) in &c.tensors {
                let t = source.tensor(id)?;
                require(
                    t.available
                        && t.count == s.count
                        && t.dtype == s.dtype
                        && s.q99.is_finite()
                        && s.q99 >= 0.
                        && s.q99 <= s.max_abs
                        && s.max_abs.is_finite()
                        && s.max_abs >= 0.
                        && s.median_nonzero_abs.is_finite()
                        && s.median_nonzero_abs >= 0.
                        && s.median_nonzero_abs <= s.max_abs
                        && s.exact_zero_count <= s.count as u64
                        && s.robust_clipped_count <= s.count as u64
                        && if t.dtype == "F32" {
                            s.histogram_sha256.is_none()
                        } else {
                            s.histogram_sha256.as_ref().is_some_and(|v| {
                                v.len() == 64 && v.bytes().all(|b| b.is_ascii_hexdigit())
                            })
                        },
                    "Invalid calibration stats",
                )?
            }
            Ok(c)
        })();
        let (calibration,note)=match loaded{Ok(c)=>(c,"Reused calibration after path, size, inode/device, mtime/ctime, header/index SHA identity checks; full source SHA-256 NOT recomputed on open.".into()),Err(e)=>(empty,format!("At startup no valid saved calibration: {e}. Header/file identity checks do not constitute full source SHA verification."))};
        let name = name.unwrap_or_else(|| {
            source
                .root
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned()
        });
        let cache = TileCache::open(&root.join("tiles"))?;
        Ok(Self {
            source,
            root,
            calibration: Mutex::new(calibration),
            progress: Mutex::new(Progress::default()),
            calibration_note: note,
            started,
            name,
            revision: revision.unwrap_or_else(|| "local selection; revision not asserted".into()),
            lookups: Mutex::new(VecDeque::new()),
            cache: Mutex::new(cache),
            compute: Mutex::new(()),
            fresh_hashes: Mutex::new(None),
            _lock: lock,
        })
    }
    pub fn global(&self) -> Option<f64> {
        let c = self.calibration.lock().unwrap();
        if c.tensors.len() == self.source.tensors.len() {
            Some(c.tensors.values().map(|s| s.max_abs).fold(0., f64::max))
        } else {
            None
        }
    }
    pub fn stats(&self, id: usize) -> Option<Stats> {
        self.calibration.lock().unwrap().tensors.get(&id).cloned()
    }
    pub fn persist(&self) -> Result<()> {
        let _compute = self.compute.lock().unwrap();
        let snapshot = self.calibration.lock().unwrap().clone();
        self.persist_calibration(&snapshot)
    }
    fn persist_calibration(&self, calibration: &Calibration) -> Result<()> {
        self.source.check()?;
        let payload = serde_json::to_string(calibration)?;
        atomic_write(
            &self.root.join("calibration.json"),
            &serde_json::to_vec(&json!({"sha256":sha(payload.as_bytes()),"payload":payload}))?,
        )
    }
    // The caller holds compute, serializing calibration writers. Readers retain
    // the previously committed snapshot until publication has actually succeeded.
    fn commit_calibration(
        &self,
        id: usize,
        stats: Stats,
        publish: impl FnOnce(&Calibration) -> Result<()>,
    ) -> Result<()> {
        let mut next = self.calibration.lock().unwrap().clone();
        next.tensors.insert(id, stats);
        publish(&next)?;
        *self.calibration.lock().unwrap() = next;
        Ok(())
    }
    pub fn calibrate_one(&self, id: usize) -> Result<()> {
        let _compute = self.compute.lock().unwrap();
        if self.stats(id).is_some() {
            return Ok(());
        }
        let _workspace = crate::resources::reserve(crate::resources::JOB_WORKSPACE_BYTES)?;
        let t = self.source.tensor(id)?;
        self.progress.lock().unwrap().active = Some(id);
        let result = (|| {
            let (s, hist) = render::calibrate(&self.source, t)?;
            if !hist.is_empty() {
                let hdir = self.root.join("histograms");
                std::fs::create_dir_all(&hdir)?;
                let mut z = ZlibEncoder::new(Vec::new(), Compression::new(3));
                for n in hist {
                    z.write_all(&n.to_le_bytes())?;
                }
                atomic_write(
                    &hdir.join(format!("{}-{id:04}.u64le.zlib", self.source.identity)),
                    &z.finish()?,
                )?;
            }
            let seconds = s.seconds;
            self.commit_calibration(id, s, |next| self.persist_calibration(next))?;
            eprintln!(
                "{}",
                json!({"calibrated":id,"name":t.name,"seconds":seconds,"values":t.count})
            );
            Ok(())
        })();
        let mut p = self.progress.lock().unwrap();
        p.active = None;
        p.error = result.as_ref().err().map(|e| format!("{e}"));
        result
    }
    pub fn legends(&self, t: &Tensor, left: &str, right: &str) -> Result<Value> {
        let dtype = Dtype::parse(&t.dtype)?;
        render::validate_rule_dtype(left, dtype)?;
        render::validate_rule_dtype(right, dtype)?;
        let s = self.stats(t.id);
        let g = self.global();
        Ok(
            json!({"left":render::legend(left,s.as_ref(),g)?,"right":render::legend(right,s.as_ref(),g)?}),
        )
    }
    pub fn mapping(&self, rule: &str, l: &Value, dtype: Dtype) -> Result<Arc<render::Mapping>> {
        let key = format!("{}:{rule}:{}:{}", dtype.name(), l["max"], l["s"]);
        let mut cache = self.lookups.lock().unwrap();
        if let Some(i) = cache.iter().position(|(k, _)| *k == key) {
            let entry = cache.remove(i).unwrap();
            let result = entry.1.clone();
            cache.push_back(entry);
            return Ok(result);
        }
        let lut = Arc::new(render::mapping(rule, l, dtype)?);
        if cache.len() >= 24 {
            cache.pop_front();
        }
        cache.push_back((key, lut.clone()));
        Ok(lut)
    }
    pub fn tensor_mapping(
        &self,
        t: &Tensor,
        rule: &str,
        l: &Value,
    ) -> Result<Arc<render::Mapping>> {
        let dtype = Dtype::parse(&t.dtype)?;
        render::validate_rule_dtype(rule, dtype)?;
        if !crate::rules::definition(rule)?.requires_histogram() {
            return self.mapping(rule, l, dtype);
        }
        self.source.check()?;
        let stats = self
            .stats(t.id)
            .ok_or("Complete selected-tensor calibration is not ready")?;
        let digest = stats
            .histogram_sha256
            .as_deref()
            .ok_or("Exact percentile histogram is not ready")?;
        require(
            l["histogram_sha256"].as_str() == Some(digest),
            "Percentile calibration mismatch",
        )?;
        let key = format!(
            "{}:{}:{}:{rule}:{digest}",
            self.source.identity,
            t.id,
            dtype.name()
        );
        {
            let mut cache = self.lookups.lock().unwrap();
            if let Some(i) = cache.iter().position(|(k, _)| *k == key) {
                let entry = cache.remove(i).unwrap();
                let result = entry.1.clone();
                cache.push_back(entry);
                return Ok(result);
            }
        }
        let path = self
            .root
            .join("histograms")
            .join(format!("{}-{:04}.u64le.zlib", self.source.identity, t.id));
        let mut compressed = Vec::new();
        File::open(path).map_err(|e| format!("Exact percentile histogram unavailable ({e}); fresh cache and recalibration required"))?
            .take(1048577)
            .read_to_end(&mut compressed)?;
        require(
            compressed.len() <= 1048576,
            "Percentile histogram compressed size exceeds limit",
        )?;
        let mut bytes = Vec::new();
        ZlibDecoder::new(compressed.as_slice())
            .take(65536 * 8 + 1)
            .read_to_end(&mut bytes)?;
        require(bytes.len() == 65536 * 8 && sha(&bytes) == digest, "Exact percentile histogram length/checksum mismatch; fresh cache and recalibration required")?;
        let hist = bytes
            .chunks_exact(8)
            .map(|v| u64::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>();
        let mapping = Arc::new(render::percentile_mapping(
            dtype,
            &hist,
            self.source.tensor(t.id)?.count,
        )?);
        self.source.check()?;
        let mut cache = self.lookups.lock().unwrap();
        if cache.len() >= 24 {
            cache.pop_front();
        }
        cache.push_back((key, mapping.clone()));
        Ok(mapping)
    }
    /// Opaque browser identity includes the exact committed legend and selected slice.
    pub fn tile_binding_for(&self, slice: &TensorSlice, rule: &str, legend: &Value) -> String {
        browser_tile_binding(
            &self.model_identity(),
            &self.source.identity,
            &slice.identity,
            rule,
            legend,
        )
    }
    pub fn tile_binding(&self, id: usize, leading: &[usize], rule: &str) -> Result<String> {
        self.source.check()?;
        let slice = TensorSlice::new(&self.source, id, leading)?;
        render::validate_rule_dtype(rule, Dtype::parse(&slice.tensor.dtype)?)?;
        let legend = render::legend(rule, self.stats(id).as_ref(), self.global())?;
        Ok(self.tile_binding_for(&slice, rule, &legend))
    }

    pub fn tile(
        &self,
        id: usize,
        rule: &str,
        level: u32,
        x: usize,
        y: usize,
    ) -> Result<(Vec<u8>, bool, render::Metrics)> {
        self.tile_slice(id, &[], rule, level, x, y)
    }
    pub fn tile_slice(
        &self,
        id: usize,
        leading: &[usize],
        rule: &str,
        level: u32,
        x: usize,
        y: usize,
    ) -> Result<(Vec<u8>, bool, render::Metrics)> {
        self.tile_slice_bound(id, leading, rule, level, x, y, None)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn tile_slice_bound(
        &self,
        id: usize,
        leading: &[usize],
        rule: &str,
        level: u32,
        x: usize,
        y: usize,
        binding: Option<&str>,
    ) -> Result<(Vec<u8>, bool, render::Metrics)> {
        let _compute = self.compute.lock().unwrap();
        self.source.check()?;
        let slice = TensorSlice::new(&self.source, id, leading)?;
        let t = &slice.tensor;
        render::validate_rule_dtype(rule, Dtype::parse(&t.dtype)?)?;
        let s = self.stats(id);
        let l = render::legend(rule, s.as_ref(), self.global())?;
        if let Some(expected) = binding {
            require(
                expected == self.tile_binding_for(&slice, rule, &l),
                "Tile binding changed; reload the selected view",
            )?;
        }
        require(level <= t.max_level, "Invalid tile level")?;
        let factor = 1usize << (t.max_level - level);
        require(
            x < t.cols.div_ceil(256 * factor) && y < t.rows.div_ceil(256 * factor),
            "Tile outside tensor",
        )?;
        let width = (t.cols - x * 256 * factor)
            .min(256 * factor)
            .div_ceil(factor);
        let height = (t.rows - y * 256 * factor)
            .min(256 * factor)
            .div_ceil(factor);
        let key = sha(format!(
            "{}:{RENDER_VERSION}:{id}:{rule}:{level}:{x}:{y}:{l}:{}",
            self.source.identity, slice.identity
        )
        .as_bytes());
        if let Some(bytes) = self.cache.lock().unwrap().get(&key)? {
            self.source.check()?;
            return Ok((
                bytes,
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
        let lut = self.tensor_mapping(t, rule, &l)?;
        let (fields, m) = render::tile_fields(&self.source, t, &[&lut], level, x, y)?;
        let png = render::png_for_rule(rule, &fields[0], m.width, m.height)?;
        self.cache.lock().unwrap().put(&key, &png)?;
        Ok((png, false, m))
    }
    /// Explicit CLI preparation of one <=256x256 overview, at most four rules.
    /// No calibration, model scanning beyond the selected slice or background work.
    pub fn prepare_overview(
        &self,
        id: usize,
        leading: &[usize],
        rules: &[&str],
        max_values: usize,
    ) -> Result<Value> {
        require(
            (1..=64 * 1024 * 1024).contains(&max_values),
            "Overview max-values must be 1–67108864",
        )?;
        require(
            !rules.is_empty() && rules.len() <= 4,
            "Overview requires 1–4 rules",
        )?;
        let _compute = self.compute.lock().unwrap();
        self.source.check()?;
        let slice = TensorSlice::new(&self.source, id, leading)?;
        let t = &slice.tensor;
        require(
            t.count <= max_values,
            "Selected slice exceeds explicit overview value budget",
        )?;
        let level = t.max_level.min(8);
        let mut legends = Vec::new();
        let mut mappings = Vec::new();
        for rule in rules {
            render::validate_rule_dtype(rule, Dtype::parse(&t.dtype)?)?;
            let legend = render::legend(rule, self.stats(id).as_ref(), self.global())?;
            mappings.push(self.tensor_mapping(t, rule, &legend)?);
            legends.push(legend);
        }
        let refs = mappings.iter().map(|v| v.as_ref()).collect::<Vec<_>>();
        let (fields, metrics) = render::tile_fields(&self.source, t, &refs, level, 0, 0)?;
        let mut records = Vec::new();
        for ((rule, field), legend) in rules.iter().zip(&fields).zip(&legends) {
            let key = sha(format!(
                "{}:{RENDER_VERSION}:{id}:{rule}:{level}:0:0:{legend}:{}",
                self.source.identity, slice.identity
            )
            .as_bytes());
            let png = render::png_for_rule(rule, field, metrics.width, metrics.height)?;
            self.cache.lock().unwrap().put(&key, &png)?;
            records.push(json!({"rule":rule,"level":level,"x":0,"y":0,
                "binding":self.tile_binding_for(&slice,rule,legend),"png_bytes":png.len()}));
        }
        self.source.check()?;
        Ok(
            json!({"api_version":1,"source_binding":self.slice_binding(&slice),"max_values":max_values,
            "metrics":metrics,"tiles":records,"coverage":"one selected-slice overview per rule; fine tiles remain on demand"}),
        )
    }

    /// Small readiness projection. Never construct the full tensor catalog here.
    pub fn status(&self, selected: Option<usize>) -> Result<Value> {
        self.source.check()?;
        let global = self.global();
        let c = self.calibration.lock().unwrap();
        let p = self.progress.lock().unwrap();
        let tensor = if let Some(id) = selected {
            let t = self.source.tensor(id)?;
            let s = c.tensors.get(&id);
            Some(Value::from(response::TensorStatus {
                id,
                calibration_complete: s.is_some(),
                max_abs: s.map(|s| s.max_abs),
                median_nonzero_abs: s.map(|s| s.median_nonzero_abs),
                q99: s.map(|s| s.q99),
                robust_clipped_count: s.map(|s| s.robust_clipped_count),
                exact_zero_count: s.map(|s| s.exact_zero_count),
                unique_bit_patterns: s.and_then(|s| s.unique_bit_patterns),
                calibration_method: s.map(|s| s.calibration_method.as_str()),
                signed_percentile_status: if !t.available {
                    "unavailable: numeric tensor viewing unsupported"
                } else if t.dtype == "F32" {
                    "unsupported: exact F32 absolute-value rank index pending"
                } else if s.is_none() {
                    "calibration pending"
                } else {
                    "supported dtype; requires a valid exact histogram"
                },
            }))
        } else {
            None
        };
        let supported = self.source.tensors.iter().filter(|t| t.available).count();
        let unsupported = self.source.tensors.len() - supported;
        let supported_complete = supported > 0
            && self
                .source
                .tensors
                .iter()
                .filter(|t| t.available)
                .all(|t| c.tensors.contains_key(&t.id));
        let tiles = self.cache.lock().unwrap();
        let result = Value::from(response::Status {
            source_identity: &self.source.identity,
            model_identity: self.model_identity(),
            global_max: global,
            calibration_complete: global.is_some(),
            coverage: response::Coverage {
                statistics_complete: global.is_some(),
                values_streamed: c.tensors.values().map(|s| s.count).sum::<usize>(),
                calibrated_tensors: c.tensors.len(),
                active_tensor: p.active,
                all_requested: p.all_requested,
                calibration_error: p.error.as_ref().map(|_| "calibration_failed"),
                materialized_tiles: tiles.entries.len(),
                materialized_bytes: tiles.bytes,
                supported_tensors: supported,
                unavailable_tensors: unsupported,
            },
            tensor_status: tensor,
            global_calibration_supported: unsupported == 0,
            supported_calibration_complete: supported_complete,
        });
        require(
            serde_json::to_vec(&result)?.len() <= 16384,
            "Status exceeds 16 KiB",
        )?;
        Ok(result)
    }
    pub fn model(&self) -> Result<Value> {
        self.source.check()?;
        let g = self.global();
        let c = self.calibration.lock().unwrap();
        let p = self.progress.lock().unwrap();
        let total: usize = self.source.tensors.iter().map(|t| t.count).sum();
        let supported = self.source.tensors.iter().filter(|t| t.available).count();
        let unsupported = self.source.tensors.len() - supported;
        let supported_complete = supported > 0
            && self
                .source
                .tensors
                .iter()
                .filter(|t| t.available)
                .all(|t| c.tensors.contains_key(&t.id));
        let global_unavailable_reason = if unsupported > 0 {
            Some("Catalog includes unavailable tensors; global calibration cannot complete. Supported tensors remain eligible for local calibration.")
        } else {
            None
        };
        let catalog = self
            .source
            .tensors
            .iter()
            .map(|t| {
                let v = serde_json::to_value(t).unwrap();
                let s = c.tensors.get(&t.id);
                Value::from(response::ModelTensor {
                    tensor: v,
                    calibration_complete: s.is_some(),
                    slice_required: t.available && t.shape.len() > 2,
                    display_axes: if t.shape.len() == 1 {
                        vec![0]
                    } else if t.shape.len() > 1 {
                        vec![t.shape.len() - 2, t.shape.len() - 1]
                    } else {
                        vec![]
                    },
                    max_abs: s.map(|s| s.max_abs),
                    median_nonzero_abs: s.map(|s| s.median_nonzero_abs),
                    q99: s.map(|s| s.q99),
                    quantile_order_statistics: s.map(|_| "exact"),
                    quantile_interpolation: s
                        .map(|_| "linear in F64; final floating-point rounding possible"),
                    robust_clipped_count: s.map(|s| s.robust_clipped_count),
                    signed_percentile_status: if !t.available {
                        "unavailable: numeric tensor viewing unsupported"
                    } else if t.dtype == "F32" {
                        "unsupported: exact F32 absolute-value rank index pending"
                    } else if s.is_none() {
                        "calibration pending"
                    } else {
                        "supported dtype; requires a valid exact histogram"
                    },
                    exact_zero_count: s.map(|s| s.exact_zero_count),
                    unique_bit_patterns: s.and_then(|s| s.unique_bit_patterns),
                    calibration_method: s.map(|s| s.calibration_method.as_str()),
                })
            })
            .collect::<Vec<_>>();
        let cache = self.cache.lock().unwrap();
        let fresh = self.fresh_hashes.lock().unwrap().clone();
        let hashes = fresh.as_ref().and_then(|v| v["shards"].as_array());
        let hashed = hashes.map_or(0, Vec::len);
        let matched = hashes.map_or(0, |records| {
            records
                .iter()
                .filter(|r| r["matches_saved_expected_sha"] == true)
                .count()
        });
        let missing_expectation = hashes.map_or(0, |records| {
            records
                .iter()
                .filter(|r| r["matches_saved_expected_sha"].is_null())
                .count()
        });
        let model = Value::from(response::Model {
            name: &self.name,
            revision: &self.revision,
            parameter_count: total,
            global_max: g,
            calibration_complete: g.is_some(),
            source_directory: &self.source.root,
            source_bytes: self.source.bytes,
            header_bytes_read: self.source.header_bytes,
            source_identity: &self.source.identity,
            model_identity: self.model_identity(),
            identity_validation: &self.calibration_note,
            fresh_source_hashes: &fresh,
            catalog: &catalog,
            rules: RULES
                .iter()
                .map(|r| render::rule_info(r).unwrap())
                .collect::<Vec<_>>(),
            coverage: response::ModelCoverage {
                sha_hashed_shards: hashed,
                sha_expected_matched_shards: matched,
                sha_missing_expected_shards: missing_expectation,
                sha_verified_shards: matched,
                statistics_complete: g.is_some(),
                values_streamed: c.tensors.values().map(|s| s.count).sum::<usize>(),
                calibrated_tensors: c.tensors.len(),
                active_tensor: p.active,
                calibration_error: &p.error,
                all_requested: p.all_requested,
                materialized_tiles: cache.entries.len(),
                materialized_bytes: cache.bytes,
                cache_budget_bytes: cache.budget_bytes,
                fine_tile_file_cap: cache.file_cap,
                supported_tensors: supported,
                unavailable_tensors: unsupported,
            },
            global_calibration_supported: unsupported == 0,
            global_calibration_unavailable_reason: global_unavailable_reason,
            supported_calibration_complete: supported_complete,
        });
        Ok(model)
    }
}
pub struct TileCache {
    pub root: PathBuf,
    pub entries: BTreeMap<String, (u64, SystemTime)>,
    pub bytes: u64,
    budget_bytes: u64,
    file_cap: usize,
}
impl TileCache {
    pub fn open(root: &Path) -> Result<Self> {
        Self::open_with_limits(root, crate::resources::policy())
    }
    fn open_with_limits(root: &Path, limits: &crate::resources::Resources) -> Result<Self> {
        limits.validate()?;
        std::fs::create_dir_all(root)?;
        let mut c = Self {
            root: root.into(),
            entries: BTreeMap::new(),
            bytes: 0,
            budget_bytes: limits.tile_cache_bytes,
            file_cap: limits.tile_cache_files,
        };
        for e in std::fs::read_dir(root)? {
            let e = e?;
            let name = e.file_name().to_string_lossy().to_string();
            if name.ends_with(".png") && e.file_type()?.is_file() {
                let m = e.metadata()?;
                c.bytes += m.len();
                c.entries.insert(name, (m.len(), m.modified()?));
            }
        }
        c.trim(0)?;
        Ok(c)
    }
    fn trim(&mut self, extra: u64) -> Result<()> {
        while !self.entries.is_empty()
            && (self.bytes.saturating_add(extra) > self.budget_bytes
                || self.entries.len() >= self.file_cap)
        {
            let key = self
                .entries
                .iter()
                .min_by_key(|(_, (_, when))| *when)
                .unwrap()
                .0
                .clone();
            match std::fs::remove_file(self.root.join(&key)) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
            self.forget(&key);
        }
        Ok(())
    }
    pub fn get(&mut self, key: &str) -> Result<Option<Vec<u8>>> {
        self.get_with(key, |path| {
            let mut bytes = Vec::new();
            File::open(path)?
                .take(1024 * 1024 + 1)
                .read_to_end(&mut bytes)?;
            Ok(bytes)
        })
    }
    fn forget(&mut self, name: &str) {
        if let Some((size, _)) = self.entries.remove(name) {
            self.bytes -= size;
        }
    }
    fn get_with(
        &mut self,
        key: &str,
        read: impl FnOnce(&Path) -> std::io::Result<Vec<u8>>,
    ) -> Result<Option<Vec<u8>>> {
        let name = format!("{key}.png");
        if let Some(&(size, _)) = self.entries.get(&name) {
            if size <= 1024 * 1024 {
                match read(&self.root.join(&name)) {
                    Ok(bytes) if bytes.len() as u64 == size => {
                        self.entries.get_mut(&name).unwrap().1 = SystemTime::now();
                        return Ok(Some(bytes));
                    }
                    Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                        self.forget(&name);
                        return Err(e.into());
                    }
                    _ => {}
                }
            }
            // A missing, unreadable, or changed cache file is not source data.
            // Forget its accounting; State::tile can regenerate from the source.
            self.forget(&name);
        }
        Ok(None)
    }
    pub fn put(&mut self, key: &str, data: &[u8]) -> Result<()> {
        require(data.len() <= 1024 * 1024, "Tile cache entry exceeds 1 MiB")?;
        let name = format!("{key}.png");
        if self.entries.contains_key(&name) {
            return Ok(());
        }
        self.trim(data.len() as u64)?;
        atomic_write(&self.root.join(&name), data)?;
        self.entries
            .insert(name, (data.len() as u64, SystemTime::now()));
        self.bytes += data.len() as u64;
        Ok(())
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    fn model_reports_the_captured_effective_cache_limits() {
        let root =
            std::env::temp_dir().join(format!("atlas-effective-cache-{}", std::process::id()));
        let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/tiny-bf16");
        let state = State::open(&model, &root, None, None).unwrap();
        for limits in [
            crate::resources::Resources::default(),
            crate::resources::Resources {
                tile_cache_bytes: 16 * crate::resources::MIB,
                tile_cache_files: 64,
                ..Default::default()
            },
            crate::resources::Resources {
                tile_cache_bytes: 8 * crate::resources::GIB,
                tile_cache_files: 8000,
                ..Default::default()
            },
        ] {
            *state.cache.lock().unwrap() =
                TileCache::open_with_limits(&root.join("tiles"), &limits).unwrap();
            let report = state.model().unwrap();
            assert_eq!(
                report["coverage"]["cache_budget_bytes"],
                limits.tile_cache_bytes
            );
            assert_eq!(
                report["coverage"]["fine_tile_file_cap"],
                limits.tile_cache_files
            );
        }
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn small_cache_captures_limits_and_trims_before_an_admitted_write() {
        let root = std::env::temp_dir().join(format!("atlas-small-cache-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        // Sparse regular files exercise accounting without allocating 16 MiB of test RAM.
        for index in 0..64 {
            File::create(root.join(format!("{index}.png")))
                .unwrap()
                .set_len(256 * 1024)
                .unwrap();
        }
        let limits = crate::resources::Resources {
            tile_cache_bytes: 16 * crate::resources::MIB,
            tile_cache_files: 64,
            ..Default::default()
        };
        let mut cache = TileCache::open_with_limits(&root, &limits).unwrap();
        assert!(cache.entries.len() < limits.tile_cache_files);
        let incoming = vec![0; 1024 * 1024];
        cache.put("new", &incoming).unwrap();
        assert!(cache.bytes <= limits.tile_cache_bytes);
        assert!(cache.entries.len() <= limits.tile_cache_files);
        assert_eq!(
            cache.bytes,
            cache.entries.values().map(|&(n, _)| n).sum::<u64>()
        );
        assert_eq!(cache.get("new").unwrap().unwrap(), incoming);
        let prior = (cache.bytes, cache.entries.len());
        assert!(cache.put("too-large", &vec![0; 1024 * 1024 + 1]).is_err());
        assert_eq!((cache.bytes, cache.entries.len()), prior);
        assert!(!root.join("too-large.png").exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bound_tiles_and_prepared_overviews_match_legacy_pixels() {
        let _isolation = crate::resources::test_workspace_guard();
        let root =
            std::env::temp_dir().join(format!("atlas-bound-overview-{}", std::process::id()));
        let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/tiny-bf16");
        let state = State::open(&model, &root, None, None).unwrap();
        state.calibrate_one(0).unwrap();
        let level = state.source.tensor(0).unwrap().max_level.min(8);
        let binding = state.tile_binding(0, &[], "tensor_linear").unwrap();
        assert!(state
            .tile_slice_bound(0, &[], "tensor_linear", level, 0, 0, Some("stale"))
            .is_err());
        let prepared = state
            .prepare_overview(0, &[], &["tensor_linear"], 1024)
            .unwrap();
        assert_eq!(prepared["tiles"][0]["binding"], binding);
        let (bound, hit, _) = state
            .tile_slice_bound(0, &[], "tensor_linear", level, 0, 0, Some(&binding))
            .unwrap();
        assert!(hit);
        let (legacy, hit, _) = state.tile(0, "tensor_linear", level, 0, 0).unwrap();
        assert!(hit);
        assert_eq!(bound, legacy);
        assert!(state
            .prepare_overview(0, &[], &["tensor_linear"], 1)
            .is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn browser_identity_changes_for_every_content_input() {
        let original = browser_tile_binding(
            "model-A",
            "source-A",
            "slice-A",
            "tensor_linear",
            &json!({"max":1}),
        );
        assert_eq!(original.len(), 64);
        assert_eq!(
            original,
            browser_tile_binding(
                "model-A",
                "source-A",
                "slice-A",
                "tensor_linear",
                &json!({"max":1})
            )
        );
        for changed in [
            browser_tile_binding(
                "model-B",
                "source-A",
                "slice-A",
                "tensor_linear",
                &json!({"max":1}),
            ),
            browser_tile_binding(
                "model-A",
                "source-B",
                "slice-A",
                "tensor_linear",
                &json!({"max":1}),
            ),
            browser_tile_binding(
                "model-A",
                "source-A",
                "slice-B",
                "tensor_linear",
                &json!({"max":1}),
            ),
            browser_tile_binding(
                "model-A",
                "source-A",
                "slice-A",
                "tensor_asinh",
                &json!({"max":1}),
            ),
            browser_tile_binding(
                "model-A",
                "source-A",
                "slice-A",
                "tensor_linear",
                &json!({"max":2}),
            ),
        ] {
            assert_ne!(original, changed);
        }
    }

    #[test]
    fn revision_identity_is_domain_separated_and_deterministic() {
        let source = "a".repeat(64);
        assert_eq!(
            revision_identity(&source, "revision-A"),
            "5942af533f3964e00a6804e0a918c2ce28e040ccd3756fcb142362517a3d7cd1"
        );
        assert_ne!(
            revision_identity(&source, "revision-A"),
            revision_identity(&source, "revision-B")
        );
        assert_ne!(
            revision_identity(&source, "revision-A"),
            revision_identity(&"b".repeat(64), "revision-A")
        );
    }

    #[test]
    fn model_and_scalar_share_a_revision_bound_identity_without_paths() {
        let root =
            std::env::temp_dir().join(format!("atlas-workspace-binding-{}", std::process::id()));
        let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/tiny-bf16");
        let mut bindings = Vec::new();
        for revision in ["revision-A", "revision-B"] {
            let state = State::open(&model, &root, None, Some(revision.into())).unwrap();
            let metadata = state.model().unwrap();
            let scalar = crate::server::inspect(&state, &crate::server::query(&[])).unwrap();
            let binding = scalar["source_binding"].clone();
            assert_eq!(
                binding,
                state.source_binding(state.source.tensor(0).unwrap())
            );
            assert_eq!(binding["model_identity"], metadata["model_identity"]);
            assert_eq!(binding["source_identity"], metadata["source_identity"]);
            assert!(!binding.to_string().contains(revision));
            assert!(!binding.to_string().contains(model.to_str().unwrap()));
            assert!(binding.get("source_directory").is_none());
            assert!(binding.get("shard").is_none());
            bindings.push(binding);
        }
        assert_eq!(
            bindings[0]["source_identity"],
            bindings[1]["source_identity"]
        );
        assert_eq!(bindings[0]["shape"], bindings[1]["shape"]);
        assert_ne!(bindings[0]["model_identity"], bindings[1]["model_identity"]);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn missing_unreadable_and_changed_tiles_become_accounted_misses() {
        let root =
            std::env::temp_dir().join(format!("atlas-cache-recovery-{}", std::process::id()));
        let mut cache = TileCache::open(&root).unwrap();
        let key = "fixture";
        cache.put(key, b"first").unwrap();
        std::fs::remove_file(root.join("fixture.png")).unwrap();
        assert!(cache.get(key).unwrap().is_none());
        assert!(cache.entries.is_empty());
        assert_eq!(cache.bytes, 0);
        cache.put(key, b"regenerated").unwrap();
        assert_eq!(cache.get(key).unwrap().unwrap(), b"regenerated");
        assert!(cache
            .get_with(key, |_| Err(std::io::Error::from(
                std::io::ErrorKind::PermissionDenied
            )))
            .is_err());
        assert!(cache.entries.is_empty());
        assert_eq!(cache.bytes, 0);
        cache.put(key, b"next").unwrap();
        std::fs::write(root.join("fixture.png"), b"different length").unwrap();
        assert!(cache.get(key).unwrap().is_none());
        assert_eq!(cache.bytes, 0);
        cache.put(key, b"final").unwrap();
        std::fs::remove_file(root.join("fixture.png")).unwrap();
        cache.trim(2 * 1024 * 1024 * 1024).unwrap();
        assert!(cache.entries.is_empty());
        assert_eq!(cache.bytes, 0);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn readers_observe_only_committed_calibration_across_publication_barriers() {
        let _isolation = crate::resources::test_workspace_guard();
        use std::sync::mpsc::sync_channel;
        use std::time::Duration;
        let root =
            std::env::temp_dir().join(format!("atlas-publication-barrier-{}", std::process::id()));
        let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/tiny-bf16");
        let state = Arc::new(State::open(&model, &root, None, None).unwrap());
        let last = state.source.tensors.last().unwrap().id;
        for id in 0..last {
            state.calibrate_one(id).unwrap();
        }
        let prior = std::fs::read(root.join("calibration.json")).unwrap();
        let (stats, _) =
            render::calibrate(&state.source, state.source.tensor(last).unwrap()).unwrap();
        for fail in [true, false] {
            let (entered, ready) = sync_channel(0);
            let (release, barrier) = sync_channel(0);
            let publisher_state = state.clone();
            let stats = stats.clone();
            let publisher = std::thread::Builder::new()
                .stack_size(1024 * 1024)
                .spawn(move || {
                    let _compute = publisher_state.compute.lock().unwrap();
                    publisher_state.commit_calibration(last, stats, |next| {
                        assert_eq!(next.tensors.len(), publisher_state.source.tensors.len());
                        entered.send(()).unwrap();
                        barrier.recv_timeout(Duration::from_secs(2)).unwrap();
                        if fail {
                            Err(std::io::Error::other("Injected publisher failure").into())
                        } else {
                            publisher_state.persist_calibration(next)
                        }
                    })
                })
                .unwrap();
            ready.recv_timeout(Duration::from_secs(2)).unwrap();
            // Observe through the actual readiness readers while publication is paused.
            assert!(state.stats(last).is_none());
            assert!(state.global().is_none());
            let metadata = state.model().unwrap();
            assert_eq!(metadata["calibration_complete"], false);
            assert_eq!(metadata["catalog"][last]["calibration_complete"], false);
            assert_eq!(std::fs::read(root.join("calibration.json")).unwrap(), prior);
            release.send(()).unwrap();
            let result = publisher.join().unwrap();
            assert_eq!(result.is_err(), fail);
            if fail {
                assert!(state.stats(last).is_none());
                assert_eq!(std::fs::read(root.join("calibration.json")).unwrap(), prior);
            }
        }
        assert!(state.stats(last).is_some());
        assert_eq!(state.global(), Some(34.));
        assert_eq!(state.model().unwrap()["calibration_complete"], true);
        drop(state);
        let reopened = State::open(&model, &root, None, None).unwrap();
        assert_eq!(reopened.global(), Some(34.));
        drop(reopened);
        std::fs::remove_dir_all(root).unwrap();
    }
}
