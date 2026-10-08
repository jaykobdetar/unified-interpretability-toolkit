pub mod api;
pub mod comparison;
pub mod comparison_http;
pub mod hosted_renderer;
mod http_status;
mod page_assets;
pub mod profile_worker;
pub mod render;
pub mod resources;
mod rules;
pub mod server;
pub mod slice;
pub mod source;
pub mod state;
pub mod strength;
use std::{ffi::CString, path::Path};
pub type Result<T> = std::result::Result<T, Error>;
pub fn require(ok: bool, message: &str) -> Result<()> {
    if !ok {
        Err(message.into())
    } else {
        Ok(())
    }
}
pub fn sha(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}
pub fn headroom() -> Result<u64> {
    let available = if resources::standalone_active() {
        resources::available_bytes()?
    } else {
        resources::host_available_bytes()?
    };
    if resources::standalone_active() {
        require(
            available >= resources::policy().available_floor_bytes,
            "PAUSED: effective available RAM is below configured reserve",
        )?;
    } else {
        resources::StartupScope::Legacy.admit(
            resources::policy(),
            1,
            available,
            available,
            u64::MAX,
        )?;
    }
    Ok(available)
}
pub fn disk_guard(path: &Path) -> Result<()> {
    disk_guard_for(path, 0)
}
fn disk_guard_for(path: &Path, additional_bytes: u64) -> Result<()> {
    use std::os::unix::ffi::OsStrExt;
    let c = CString::new(path.as_os_str().as_bytes())?;
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: valid nul-terminated path and writable statvfs storage.
    require(
        unsafe { libc::statvfs(c.as_ptr(), stat.as_mut_ptr()) } == 0,
        "Cannot check disk reserve",
    )?;
    let stat = unsafe { stat.assume_init() };
    let additional_bytes = if resources::standalone_active() {
        additional_bytes
    } else {
        0
    };
    require(
        stat.f_bavail as u128 * stat.f_frsize as u128
            >= resources::policy().disk_reserve_bytes as u128 + additional_bytes as u128,
        "PAUSED: disk free space is below configured reserve",
    )
}
pub fn configure() -> Result<usize> {
    resources::activate_scope(resources::StartupScope::Legacy);
    headroom()?;
    // SAFETY: initialized cpu_set_t/rlimit pointers, checked syscalls. All spawned threads inherit affinity.
    unsafe {
        let mut available: libc::cpu_set_t = std::mem::zeroed();
        require(
            libc::sched_getaffinity(0, std::mem::size_of_val(&available), &mut available) == 0,
            "Cannot read affinity",
        )?;
        let default = (0..libc::CPU_SETSIZE as usize)
            .find(|&i| libc::CPU_ISSET(i, &available))
            .ok_or("No allowed CPU")?;
        let cpu = std::env::var("ATLAS_CPU")
            .ok()
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(default);
        require(
            cpu < libc::CPU_SETSIZE as usize && libc::CPU_ISSET(cpu, &available),
            "ATLAS_CPU is outside allowed affinity",
        )?;
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        require(
            libc::sched_setaffinity(0, std::mem::size_of_val(&set), &set) == 0,
            "Cannot set one CPU affinity",
        )?;
        let mut lim: libc::rlimit = std::mem::zeroed();
        require(
            libc::getrlimit(libc::RLIMIT_AS, &mut lim) == 0,
            "Cannot read address-space limit",
        )?;
        lim.rlim_cur = resources::legacy_address_space(lim.rlim_max);
        require(
            libc::setrlimit(libc::RLIMIT_AS, &lim) == 0,
            "Cannot set address-space limit",
        )?;
        libc::nice(10);
        Ok(cpu)
    }
}
pub fn configure_standalone() -> Result<usize> {
    resources::activate_scope(resources::StartupScope::Standalone);
    headroom()?;
    // SAFETY: initialized cpu_set_t/rlimit pointers, checked syscalls. All spawned threads inherit affinity.
    unsafe {
        let mut available: libc::cpu_set_t = std::mem::zeroed();
        require(
            libc::sched_getaffinity(0, std::mem::size_of_val(&available), &mut available) == 0,
            "Cannot read affinity",
        )?;
        let default = (0..libc::CPU_SETSIZE as usize)
            .find(|&i| libc::CPU_ISSET(i, &available))
            .ok_or("No allowed CPU")?;
        let cpu = std::env::var("ATLAS_CPU")
            .ok()
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(default);
        require(
            cpu < libc::CPU_SETSIZE as usize && libc::CPU_ISSET(cpu, &available),
            "ATLAS_CPU is outside allowed affinity",
        )?;
        let limits = resources::policy();
        let allowed = (0..libc::CPU_SETSIZE as usize)
            .filter(|&i| libc::CPU_ISSET(i, &available))
            .collect::<Vec<_>>();
        let mut lim: libc::rlimit = std::mem::zeroed();
        require(
            libc::getrlimit(libc::RLIMIT_AS, &mut lim) == 0,
            "Cannot read address-space limit",
        )?;
        resources::StartupScope::Standalone.admit(
            limits,
            allowed.len().min(resources::cpu_quota()?),
            0,
            resources::available_bytes()?,
            lim.rlim_max,
        )?;
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        for other in allowed
            .into_iter()
            .filter(|&i| i != cpu)
            .take(limits.cpu_count - 1)
        {
            libc::CPU_SET(other, &mut set);
        }
        require(
            libc::sched_setaffinity(0, std::mem::size_of_val(&set), &set) == 0,
            "Cannot set configured CPU affinity",
        )?;
        lim.rlim_cur = limits.address_space_bytes;
        require(
            libc::setrlimit(libc::RLIMIT_AS, &lim) == 0,
            "Cannot set address-space limit",
        )?;
        libc::nice(10);
        Ok(cpu)
    }
}
pub fn peak_rss_mib() -> f64 {
    unsafe {
        let mut r: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut r);
        r.ru_maxrss as f64 / 1024.
    }
}
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    disk_guard_for(path.parent().ok_or("No output parent")?, bytes.len() as u64)?;
    atomic_replace(
        path,
        |f| {
            f.write_all(bytes)?;
            f.sync_all()
        },
        |from, to| std::fs::rename(from, to),
    )
}

// Keep the old destination until the complete, synced replacement can be
// renamed on the same filesystem. Injectable operations keep failure tests local.
fn atomic_replace(
    path: &Path,
    prepare: impl FnOnce(&mut std::fs::File) -> std::io::Result<()>,
    publish: impl FnOnce(&Path, &Path) -> std::io::Result<()>,
) -> Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let parent = path.parent().ok_or("No output parent")?;
    let name = path.file_name().ok_or("No output filename")?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let mut selected = None;
    for _ in 0..16 {
        let mut temp_name = name.to_os_string();
        temp_name.push(format!(
            ".partial-{}-{stamp:x}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let temp = parent.join(temp_name);
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
        {
            Ok(file) => {
                selected = Some((temp, file));
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    let (temp, mut file) = selected.ok_or("Could not reserve an atomic-write temporary file")?;
    let result = prepare(&mut file).and_then(|()| publish(&temp, path));
    drop(file);
    match result {
        Ok(()) => Ok(()),
        Err(e) => {
            if let Err(cleanup) = std::fs::remove_file(&temp) {
                if cleanup.kind() != std::io::ErrorKind::NotFound {
                    return Err(format!(
                        "Atomic write failed: {e}; temporary cleanup failed: {cleanup}"
                    )
                    .into());
                }
            }
            Err(e.into())
        }
    }
}

#[cfg(test)]
mod atomic_tests {
    use super::*;
    use std::io::{self, Write};

    #[test]
    fn atomic_failures_clean_temporary_files_and_preserve_destination() {
        let root =
            std::env::temp_dir().join(format!("atlas-atomic-failures-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("calibration.json");
        std::fs::write(&path, b"original").unwrap();
        for phase in ["write", "sync", "rename"] {
            let result = atomic_replace(
                &path,
                |file| {
                    if phase == "write" {
                        return Err(io::Error::other("Injected write failure"));
                    }
                    file.write_all(b"new")?;
                    if phase != "rename" {
                        return Err(io::Error::other(format!("Injected {phase} failure")));
                    }
                    file.sync_all()
                },
                |_, _| Err(io::Error::other("Injected rename failure")),
            );
            assert!(result.is_err());
            assert_eq!(std::fs::read(&path).unwrap(), b"original");
            assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1);
        }
        let mut names = Vec::new();
        for body in [b"one".as_slice(), b"two".as_slice()] {
            atomic_replace(
                &path,
                |file| {
                    file.write_all(body)?;
                    file.sync_all()
                },
                |temp, dest| {
                    assert_eq!(temp.parent(), dest.parent());
                    names.push(temp.to_path_buf());
                    std::fs::rename(temp, dest)
                },
            )
            .unwrap();
            assert_eq!(std::fs::read(&path).unwrap(), body);
        }
        assert_ne!(names[0], names[1]);
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn failed_creation_does_not_remove_unowned_files_and_stale_names_do_not_block_retry() {
        let root =
            std::env::temp_dir().join(format!("atlas-atomic-ownership-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("calibration.json");
        let stale = path.with_extension(format!("partial-{}", std::process::id()));
        std::fs::write(&stale, b"not owned by this invocation").unwrap();
        let called = std::cell::Cell::new(false);
        assert!(atomic_replace(
            &root.join("missing/output"),
            |_| {
                called.set(true);
                Ok(())
            },
            |a, b| std::fs::rename(a, b)
        )
        .is_err());
        assert!(!called.get());
        atomic_replace(
            &path,
            |f| {
                f.write_all(b"complete")?;
                f.sync_all()
            },
            |a, b| std::fs::rename(a, b),
        )
        .unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"complete");
        assert_eq!(
            std::fs::read(&stale).unwrap(),
            b"not owned by this invocation"
        );
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 2);
        std::fs::remove_dir_all(root).unwrap();
    }
}

pub mod error;
pub use error::Error;

pub mod command;
pub mod parameter;
