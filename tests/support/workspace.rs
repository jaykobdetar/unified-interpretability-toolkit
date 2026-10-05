//! Integration fixtures in one test binary share the real workspace ledger.
use std::sync::{Mutex, MutexGuard};

pub fn guard() -> MutexGuard<'static, ()> {
    static ISOLATION: Mutex<()> = Mutex::new(());
    ISOLATION.lock().unwrap_or_else(|error| error.into_inner())
}
