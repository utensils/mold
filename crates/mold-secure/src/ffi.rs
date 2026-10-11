//! Caller-owned buffers, bounded handles, and one registry lock per operation.
#![allow(clippy::missing_safety_doc)] // The complete pointer/ownership contract is in mold_secure.h.
use crate::*;
use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
};
enum Handle {
    Identity(Identity),
    Session(Session),
}
#[derive(Default)]
struct Registry {
    next: u64,
    handles: BTreeMap<u64, Handle>,
}
static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
fn registry() -> std::sync::MutexGuard<'static, Registry> {
    REGISTRY
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}
fn insert(value: Handle) -> u64 {
    let mut r = registry();
    if r.handles.len() >= 4096 {
        return 0;
    }
    let Some(next) = r.next.checked_add(1) else {
        return 0;
    };
    r.next = next;
    r.handles.insert(next, value);
    next
}
fn code(error: Error) -> i32 {
    match error {
        Error::Input => -1,
        Error::State => -2,
        Error::Authentication => -3,
        Error::Limit => -4,
        Error::Random => -5,
    }
}
unsafe fn input<'a>(ptr: *const u8, len: usize, max: usize) -> Result<&'a [u8]> {
    if len > max || ptr.is_null() {
        return Err(Error::Input);
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}
unsafe fn output(ptr: *mut u8, cap: usize, bytes: &[u8]) -> i32 {
    if ptr.is_null() || cap < bytes.len() {
        return -4;
    }
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
    bytes.len() as i32
}
#[no_mangle]
pub extern "C" fn mold_secure_identity_new() -> u64 {
    Identity::generate()
        .map(|v| insert(Handle::Identity(v)))
        .unwrap_or(0)
}
#[no_mangle]
pub unsafe extern "C" fn mold_secure_identity_restore(ptr: *const u8, len: usize) -> u64 {
    unsafe { input(ptr, len, 64) }
        .and_then(Identity::from_bytes)
        .map(|v| insert(Handle::Identity(v)))
        .unwrap_or(0)
}
#[no_mangle]
pub unsafe extern "C" fn mold_secure_identity_export(handle: u64, ptr: *mut u8, cap: usize) -> i32 {
    let r = registry();
    let Some(Handle::Identity(v)) = r.handles.get(&handle) else {
        return -2;
    };
    unsafe { output(ptr, cap, &zeroize::Zeroizing::new(v.to_bytes())) }
}
#[no_mangle]
pub unsafe extern "C" fn mold_secure_identity_binding(
    handle: u64,
    ptr: *mut u8,
    cap: usize,
) -> i32 {
    let r = registry();
    let Some(Handle::Identity(v)) = r.handles.get(&handle) else {
        return -2;
    };
    unsafe { output(ptr, cap, &v.binding()) }
}
#[no_mangle]
pub unsafe extern "C" fn mold_secure_identity_sign(
    handle: u64,
    data: *const u8,
    len: usize,
    ptr: *mut u8,
    cap: usize,
) -> i32 {
    let Ok(data) = (unsafe { input(data, len, 65535) }) else {
        return -1;
    };
    let r = registry();
    let Some(Handle::Identity(v)) = r.handles.get(&handle) else {
        return -2;
    };
    unsafe { output(ptr, cap, &v.sign(data)) }
}
#[no_mangle]
pub unsafe extern "C" fn mold_secure_session_new(
    identity: u64,
    data: *const u8,
    len: usize,
) -> u64 {
    let Ok(data) = (unsafe { input(data, len, 16384) }) else {
        return 0;
    };
    let Ok(config) = serde_json::from_slice(data) else {
        return 0;
    };
    let session = {
        let r = registry();
        let Some(Handle::Identity(v)) = r.handles.get(&identity) else {
            return 0;
        };
        Session::new(v, config)
    };
    session.map(|v| insert(Handle::Session(v))).unwrap_or(0)
}
#[no_mangle]
pub unsafe extern "C" fn mold_secure_session_handshake_write(
    handle: u64,
    ptr: *mut u8,
    cap: usize,
) -> i32 {
    if ptr.is_null() || cap < 512 {
        return -4;
    }
    let mut r = registry();
    let Some(Handle::Session(v)) = r.handles.get_mut(&handle) else {
        return -2;
    };
    match v.handshake_write() {
        Ok(bytes) => unsafe { output(ptr, cap, &bytes) },
        Err(e) => code(e),
    }
}
#[no_mangle]
pub unsafe extern "C" fn mold_secure_session_handshake_read(
    handle: u64,
    data: *const u8,
    len: usize,
) -> i32 {
    let Ok(data) = (unsafe { input(data, len, 512) }) else {
        return -1;
    };
    let mut r = registry();
    let Some(Handle::Session(v)) = r.handles.get_mut(&handle) else {
        return -2;
    };
    v.handshake_read(data).map(|_| 0).unwrap_or_else(code)
}
#[no_mangle]
pub extern "C" fn mold_secure_session_ready(handle: u64) -> i32 {
    let r = registry();
    let Some(Handle::Session(v)) = r.handles.get(&handle) else {
        return -2;
    };
    i32::from(v.established())
}
#[no_mangle]
pub unsafe extern "C" fn mold_secure_session_seal(
    handle: u64,
    data: *const u8,
    len: usize,
    ptr: *mut u8,
    cap: usize,
) -> i32 {
    let Ok(data) = (unsafe { input(data, len, MAX_PLAINTEXT) }) else {
        return -1;
    };
    if ptr.is_null() || cap < len + 16 {
        return -4;
    }
    let mut r = registry();
    let Some(Handle::Session(v)) = r.handles.get_mut(&handle) else {
        return -2;
    };
    match v.seal(data) {
        Ok(bytes) => unsafe { output(ptr, cap, &bytes) },
        Err(e) => code(e),
    }
}
#[no_mangle]
pub unsafe extern "C" fn mold_secure_session_open(
    handle: u64,
    data: *const u8,
    len: usize,
    ptr: *mut u8,
    cap: usize,
) -> i32 {
    let Ok(data) = (unsafe { input(data, len, MAX_CIPHERTEXT) }) else {
        return -1;
    };
    if ptr.is_null() || cap < len.saturating_sub(16) {
        return -4;
    }
    let mut r = registry();
    let Some(Handle::Session(v)) = r.handles.get_mut(&handle) else {
        return -2;
    };
    match v.open(data) {
        Ok(bytes) => unsafe { output(ptr, cap, &bytes) },
        Err(e) => code(e),
    }
}
#[no_mangle]
pub extern "C" fn mold_secure_release(handle: u64) {
    registry().handles.remove(&handle);
}
#[no_mangle]
pub unsafe extern "C" fn mold_secure_verify_signature(
    public: *const u8,
    data: *const u8,
    len: usize,
    signature: *const u8,
) -> i32 {
    let (Ok(public), Ok(data), Ok(signature)) = (
        unsafe { input(public, 32, 32) },
        unsafe { input(data, len, 65535) },
        unsafe { input(signature, 64, 64) },
    ) else {
        return 0;
    };
    i32::from(verify_signature(public, data, signature))
}

#[no_mangle]
pub unsafe extern "C" fn mold_secure_session_handshake_hash(
    handle: u64,
    ptr: *mut u8,
    cap: usize,
) -> i32 {
    let r = registry();
    let Some(Handle::Session(v)) = r.handles.get(&handle) else {
        return -2;
    };
    match v.handshake_hash() {
        Ok(bytes) => unsafe { output(ptr, cap, &bytes) },
        Err(e) => code(e),
    }
}
