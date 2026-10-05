//! Standalone viewer resource policy. Hosted/inference ownership remains separate.
use crate::{require, Result};
use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex, OnceLock,
};

pub const MIB: u64 = 1024 * 1024;
pub const GIB: u64 = 1024 * MIB;
pub const JOB_WORKSPACE_BYTES: u64 = 64 * MIB;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StartupScope {
    Legacy,
    Standalone,
}
impl StartupScope {
    pub fn for_command(command: &str) -> Self {
        if matches!(
            command,
            "metadata"
                | "serve"
                | "calibrate"
                | "verify"
                | "tile"
                | "overview"
                | "inspect"
                | "bench"
        ) {
            Self::Standalone
        } else {
            Self::Legacy
        }
    }
    pub fn admit(
        self,
        limits: &Resources,
        allowed_cpus: usize,
        host_available: u64,
        effective_available: u64,
        hard_as: u64,
    ) -> Result<()> {
        match self {
            Self::Standalone => limits.admit(allowed_cpus, effective_available, hard_as),
            Self::Legacy => require(
                host_available >= 3 * GIB,
                "PAUSED: fewer than 3 GiB available RAM; retry when memory is available",
            ),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Resources {
    pub version: u8,
    pub cpu_count: usize,
    pub address_space_bytes: u64,
    pub available_floor_bytes: u64,
    pub disk_reserve_bytes: u64,
    pub workspace_bytes: u64,
    pub tile_cache_bytes: u64,
    pub tile_cache_files: usize,
}
impl Default for Resources {
    fn default() -> Self {
        Self {
            version: 1,
            cpu_count: 1,
            address_space_bytes: 768 * MIB,
            available_floor_bytes: 3 * GIB,
            disk_reserve_bytes: 25 * GIB,
            workspace_bytes: JOB_WORKSPACE_BYTES,
            tile_cache_bytes: 2 * GIB,
            tile_cache_files: 1000,
        }
    }
}
impl Resources {
    pub fn validate(&self) -> Result<()> {
        require(self.version == 1, "Unsupported resources version")?;
        require((1..=8).contains(&self.cpu_count), "cpu_count must be 1–8")?;
        require(
            (256 * MIB..=8 * GIB).contains(&self.address_space_bytes),
            "Address-space budget must be 256 MiB–8 GiB",
        )?;
        require(
            (512 * MIB..=128 * GIB).contains(&self.available_floor_bytes),
            "Available RAM floor must be 512 MiB–128 GiB",
        )?;
        require(
            (GIB..=1024 * GIB).contains(&self.disk_reserve_bytes),
            "Disk reserve must be 1–1024 GiB",
        )?;
        require(
            (JOB_WORKSPACE_BYTES..=GIB).contains(&self.workspace_bytes)
                && self.workspace_bytes <= self.address_space_bytes / 2,
            "Workspace budget must be 64 MiB–1 GiB and fit within half the address-space budget",
        )?;
        require(
            (16 * MIB..=64 * GIB).contains(&self.tile_cache_bytes),
            "Tile cache must be 16 MiB–64 GiB",
        )?;
        require(
            (64..=100000).contains(&self.tile_cache_files),
            "Tile file budget must be 64–100000",
        )
    }
    /// Admission reserves the entire process budget above the unchanged free-RAM floor.
    pub fn admit(&self, allowed_cpus: usize, available_bytes: u64, hard_as: u64) -> Result<()> {
        self.validate()?;
        require(
            self.cpu_count <= allowed_cpus,
            "Requested CPUs exceed allowed affinity",
        )?;
        require(
            self.address_space_bytes <= hard_as,
            "Requested address space exceeds inherited hard limit",
        )?;
        let required = self
            .available_floor_bytes
            .checked_add(self.address_space_bytes)
            .ok_or("RAM admission overflow")?;
        require(
            available_bytes >= required,
            "PAUSED: insufficient effective RAM for floor plus process budget",
        )
    }
}

static POLICY: OnceLock<Resources> = OnceLock::new();
static LEGACY_POLICY: OnceLock<Resources> = OnceLock::new();
static STANDALONE_ACTIVE: AtomicBool = AtomicBool::new(false);
pub fn standalone_active() -> bool {
    STANDALONE_ACTIVE.load(Ordering::Acquire)
}
pub(crate) fn activate_scope(scope: StartupScope) {
    STANDALONE_ACTIVE.store(scope == StartupScope::Standalone, Ordering::Release);
}
pub(crate) fn legacy_address_space(hard_as: u64) -> u64 {
    hard_as.min(768 * MIB)
}
pub fn policy() -> &'static Resources {
    if standalone_active() {
        POLICY.get_or_init(Resources::default)
    } else {
        LEGACY_POLICY.get_or_init(Resources::default)
    }
}
pub fn initialize(resources: Resources) -> Result<()> {
    resources.validate()?;
    POLICY
        .set(resources)
        .map_err(|_| "Resource policy already initialized".into())
}

pub fn snapshot() -> Result<serde_json::Value> {
    let mut affinity: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    let mut limit: libc::rlimit = unsafe { std::mem::zeroed() };
    // SAFETY: initialized writable structures, with return values checked.
    require(
        unsafe { libc::sched_getaffinity(0, std::mem::size_of_val(&affinity), &mut affinity) } == 0,
        "Cannot measure CPU affinity",
    )?;
    require(
        unsafe { libc::getrlimit(libc::RLIMIT_AS, &mut limit) } == 0,
        "Cannot measure address-space limit",
    )?;
    let cpu_count = (0..libc::CPU_SETSIZE as usize)
        .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &affinity) })
        .count();
    let ledger = LEDGER
        .lock()
        .map_err(|_| "Resource accounting unavailable")?;
    Ok(
        serde_json::json!({"configured":policy(),"effective":{"affinity_cpu_count":cpu_count,
        "cpu_quota_count":cpu_quota()?.min(cpu_count),"address_space_bytes":limit.rlim_cur,
        "available_bytes":available_bytes()?,"reserved_workspace_bytes":ledger.used,"active_numeric_operations":ledger.active},
        "scope":"standalone Rust process; hosted/inference/build policies are separate"}),
    )
}

/// One process-global ledger, shared by every rendering call, including scoped threads.
/// Successful permits release on return/error/unwind; no unbounded waiting queue.
struct Ledger {
    used: u64,
    active: usize,
}
static LEDGER: Mutex<Ledger> = Mutex::new(Ledger { used: 0, active: 0 });
pub struct Permit {
    bytes: u64,
}

/// Host MemAvailable used by the existing hosted/comparison/profile policy.
pub fn host_available_bytes() -> Result<u64> {
    let text = std::fs::read_to_string("/proc/meminfo")?;
    let kb = text
        .lines()
        .find(|l| l.starts_with("MemAvailable:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .ok_or("MemAvailable missing")?
        .parse::<u64>()?;
    kb.checked_mul(1024)
        .ok_or_else(|| "MemAvailable overflow".into())
}

/// Linux effective free RAM: host MemAvailable capped by every cgroup-v2 ancestor.
/// Unreadable or malformed discovered constraints fail closed.
pub fn available_bytes() -> Result<u64> {
    let mut available = host_available_bytes()?;
    let root = std::path::Path::new("/sys/fs/cgroup");
    if root.join("cgroup.controllers").exists() {
        let membership = std::fs::read_to_string("/proc/self/cgroup")?;
        let relative = membership
            .lines()
            .find_map(|l| l.strip_prefix("0::"))
            .ok_or("Cannot find cgroup-v2 membership")?;
        require(
            !relative.split('/').any(|c| c == ".."),
            "Invalid cgroup membership",
        )?;
        let mut group = root.join(relative.trim_start_matches('/'));
        loop {
            let limit = group.join("memory.max");
            if limit.exists() {
                let maximum = std::fs::read_to_string(limit)?;
                if maximum.trim() != "max" {
                    let maximum = maximum.trim().parse::<u64>()?;
                    let current = std::fs::read_to_string(group.join("memory.current"))?
                        .trim()
                        .parse::<u64>()?;
                    available = available.min(maximum.saturating_sub(current));
                }
            }
            if group == root {
                break;
            }
            require(
                group.pop() && group.starts_with(root),
                "Invalid cgroup hierarchy",
            )?;
        }
    }
    Ok(available)
}

pub fn cpu_quota() -> Result<usize> {
    let root = std::path::Path::new("/sys/fs/cgroup");
    let mut cpus = usize::MAX;
    if root.join("cgroup.controllers").exists() {
        let membership = std::fs::read_to_string("/proc/self/cgroup")?;
        let relative = membership
            .lines()
            .find_map(|l| l.strip_prefix("0::"))
            .ok_or("Cannot find cgroup-v2 membership")?;
        require(
            !relative.split('/').any(|c| c == ".."),
            "Invalid cgroup membership",
        )?;
        let mut group = root.join(relative.trim_start_matches('/'));
        loop {
            let path = group.join("cpu.max");
            if path.exists() {
                let value = std::fs::read_to_string(path)?;
                let mut parts = value.split_whitespace();
                let quota = parts.next().ok_or("Invalid CPU quota")?;
                let period = parts.next().ok_or("Invalid CPU period")?.parse::<u64>()?;
                require(period > 0 && parts.next().is_none(), "Invalid CPU quota")?;
                if quota != "max" {
                    let quota = quota.parse::<u64>()?;
                    require(quota > 0, "Invalid CPU quota")?;
                    cpus = cpus.min((quota / period).max(1).try_into()?);
                }
            }
            if group == root {
                break;
            }
            require(
                group.pop() && group.starts_with(root),
                "Invalid cgroup hierarchy",
            )?;
        }
    }
    Ok(cpus)
}
pub fn reserve(bytes: u64) -> Result<Permit> {
    let mut ledger = LEDGER
        .lock()
        .map_err(|_| "Resource accounting unavailable")?;
    let next = ledger
        .used
        .checked_add(bytes)
        .ok_or("Workspace accounting overflow")?;
    require(
        bytes > 0 && next <= policy().workspace_bytes && ledger.active < 1,
        "Global numeric workspace busy or exhausted",
    )?;
    ledger.used = next;
    ledger.active += 1;
    Ok(Permit { bytes })
}
impl Drop for Permit {
    fn drop(&mut self) {
        // Recover poisoned accounting only to release this already-owned reservation.
        let mut ledger = LEDGER.lock().unwrap_or_else(|e| e.into_inner());
        ledger.used -= self.bytes;
        ledger.active -= 1;
    }
}

#[cfg(test)]
pub(crate) fn test_workspace_guard() -> std::sync::MutexGuard<'static, ()> {
    // Independent unit tests share the real process-global workspace ledger.
    // Isolate their fixtures without changing admission, accounting or release.
    static ISOLATION: Mutex<()> = Mutex::new(());
    ISOLATION.lock().unwrap_or_else(|error| error.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn standalone_admission_does_not_replace_legacy_floor() {
        let defaults = Resources::default();
        let between = 13 * GIB / 4;
        assert!(StartupScope::Legacy
            .admit(&defaults, 1, between, between, u64::MAX)
            .is_ok());
        assert!(StartupScope::Standalone
            .admit(&defaults, 1, between, between, u64::MAX)
            .is_err());
        assert!(StartupScope::Legacy
            .admit(&defaults, 1, 3 * GIB - 1, 16 * GIB, u64::MAX)
            .is_err());
        let admitted = defaults.available_floor_bytes + defaults.address_space_bytes;
        assert!(StartupScope::Standalone
            .admit(
                &defaults,
                1,
                admitted,
                admitted,
                defaults.address_space_bytes
            )
            .is_ok());
        assert!(StartupScope::Standalone
            .admit(
                &defaults,
                1,
                admitted,
                admitted - 1,
                defaults.address_space_bytes
            )
            .is_err());
        assert!(StartupScope::Standalone
            .admit(
                &defaults,
                1,
                admitted,
                admitted,
                defaults.address_space_bytes - 1
            )
            .is_err());
        // Legacy startup retains the inherited hard-limit clamp, not a new refusal.
        assert_eq!(legacy_address_space(512 * MIB), 512 * MIB);
        assert_eq!(legacy_address_space(u64::MAX), 768 * MIB);
        assert!(StartupScope::Legacy
            .admit(&defaults, 1, 3 * GIB, 0, 512 * MIB)
            .is_ok());
        // Container process reservations belong to the standalone contract.
        assert!(StartupScope::Standalone
            .admit(&defaults, 1, 16 * GIB, 3 * GIB, u64::MAX)
            .is_err());
        for command in [
            "hosted-renderer",
            "compare-serve",
            "compare-metadata",
            "compare-tile",
            "compare-calibrate",
            "compare-inspect",
            "profile-worker",
        ] {
            assert_eq!(StartupScope::for_command(command), StartupScope::Legacy);
        }
        for command in [
            "serve",
            "metadata",
            "tile",
            "overview",
            "calibrate",
            "verify",
            "inspect",
            "bench",
        ] {
            assert_eq!(StartupScope::for_command(command), StartupScope::Standalone);
        }
    }
    #[test]
    fn defaults_and_admission_preserve_guards() {
        let r = Resources::default();
        r.validate().unwrap();
        assert_eq!(r.cpu_count, 1);
        assert!(r.admit(1, 3 * GIB + 768 * MIB, u64::MAX).is_ok());
        assert!(r.admit(0, 16 * GIB, u64::MAX).is_err());
        assert!(r.admit(8, 3 * GIB, u64::MAX).is_err());
        assert!(r.admit(8, 16 * GIB, 512 * MIB).is_err());
        for text in [
            r#"{"cpu_count":0}"#,
            r#"{"cpu_count":9}"#,
            r#"{"available_floor_bytes":0}"#,
            r#"{"disk_reserve_bytes":0}"#,
            r#"{"workspace_bytes":1073741824}"#,
        ] {
            assert!(serde_json::from_str::<Resources>(text)
                .unwrap()
                .validate()
                .is_err());
        }
        for text in [
            r#"{"cpu_count":true}"#,
            r#"{"cpu_count":1.0}"#,
            r#"{"unknown":1}"#,
            r#"{"cpu_count":1,"cpu_count":2}"#,
        ] {
            assert!(serde_json::from_str::<Resources>(text).is_err());
        }
    }
    #[test]
    fn global_workspace_is_exclusive_and_released() {
        let _isolation = test_workspace_guard();
        let partial = reserve(1).unwrap();
        assert_eq!(LEDGER.lock().unwrap().used, 1);
        assert_eq!(LEDGER.lock().unwrap().active, 1);
        assert!(reserve(1).is_err());
        drop(partial);
        assert_eq!(LEDGER.lock().unwrap().used, 0);
        assert_eq!(LEDGER.lock().unwrap().active, 0);
        let owned = reserve(JOB_WORKSPACE_BYTES).unwrap();
        assert!(reserve(1).is_err());
        drop(owned);
        assert!(reserve(JOB_WORKSPACE_BYTES).is_ok());
        assert!(reserve(u64::MAX).is_err());
        assert!(reserve(0).is_err());
    }
}
