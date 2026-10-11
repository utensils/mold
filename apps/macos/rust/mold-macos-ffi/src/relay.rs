//! Cancellable managed outbound connector, independent of the engine runtime.
use super::{
    connection_addresses, guarded, relay_lifetime::EngineLifetime, str_arg, ALIVE, ENGINE_PORT,
};
use std::ffi::c_char;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

struct Connector {
    generation: u64,
    cancel: CancellationToken,
    thread: std::thread::JoinHandle<()>,
    origin: String,
    alive: Arc<AtomicBool>,
    ready: Arc<AtomicBool>,
}
static CONNECTOR: Mutex<Option<Connector>> = Mutex::new(None);
// Serialize publication with terminal withdrawal without holding the join mutex.
static PUBLICATION: Mutex<()> = Mutex::new(());
static ENGINE_LIFETIME: EngineLifetime = EngineLifetime::new();
fn engine_matches(generation: u64) -> bool {
    ENGINE_LIFETIME.matches(generation, ALIVE.load(Ordering::SeqCst))
}

fn validate_endpoint(endpoint: &str) -> anyhow::Result<()> {
    mold_relay::aws::validate_aws_endpoint(endpoint, false)?;
    // Publication applies an additional size limit. Validate with the same
    // authority before creating a thread, so a rejected start has no survivor.
    mold_server::ConnectionAddresses::default().set_secure_websocket_url(Some(endpoint))
}

/// LAN availability is independent of a cloud enrollment/renewal round trip.
/// The server still refuses to open this listener without authentication.
#[no_mangle]
pub extern "C" fn mold_local_network_set_enabled(enabled: bool) {
    guarded("local_network_access", (), || {
        connection_addresses().set_local_network_enabled(enabled && ALIVE.load(Ordering::SeqCst));
    });
}

/// Separate explicit direct-API listener; stopping the proxy leaves it alone.
#[no_mangle]
pub extern "C" fn mold_regular_network_set_enabled(enabled: bool) {
    guarded("regular_network_access", (), || {
        connection_addresses().set_regular_network_enabled(enabled && ALIVE.load(Ordering::SeqCst));
    });
}

/// Returns a JSON array of bound direct origins, or a negative readiness error.
/// Called off the UI thread; the caller supplies a bounded wait and output buffer.
/// # Safety
/// `out` must reference a writable buffer of `capacity` bytes.
#[no_mangle]
pub unsafe extern "C" fn mold_regular_network_ready(
    out: *mut u8,
    capacity: usize,
    timeout_ms: u32,
) -> i32 {
    guarded("regular_network_ready", -1, || {
        if out.is_null() || capacity == 0 || capacity > 65_536 || !ALIVE.load(Ordering::SeqCst) {
            return -1;
        }
        let generation = ENGINE_LIFETIME.snapshot();
        let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        else {
            return -1;
        };
        let result = runtime.block_on(connection_addresses().regular_network_ready(
            Duration::from_millis(u64::from(timeout_ms.clamp(1, 30_000))),
        ));
        runtime.shutdown_timeout(Duration::from_millis(100));
        let Ok(endpoints) = result else {
            return -2;
        };
        if !engine_matches(generation) {
            return -2;
        }
        let Ok(bytes) = serde_json::to_vec(&endpoints) else {
            return -1;
        };
        if bytes.len() > capacity {
            return -3;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len());
        }
        bytes.len() as i32
    })
}

fn managed_origin(origin: &str, host_id: &str) -> anyhow::Result<String> {
    mold_relay::aws::validate_host_namespace(host_id)?;
    // Reuse server's origin authority, including credential/path rejection.
    mold_server::ConnectionAddresses::default().set_managed_relay_origin(Some(origin))?;
    let url = url::Url::parse(origin)?;
    if !url
        .host_str()
        .is_some_and(|host| host.starts_with(&format!("{host_id}.")))
    {
        anyhow::bail!("managed origin does not match the enrolled host");
    }
    Ok(url.origin().ascii_serialization())
}

/// # Safety
/// Every non-null pointer must reference a valid NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn mold_relay_start(
    endpoint: *const c_char,
    host_id: *const c_char,
    local_port: u16,
    public_origin: *const c_char,
) -> i32 {
    guarded("relay_start", 1, || {
        let Some(endpoint) = (unsafe { str_arg(endpoint) }) else {
            return 1;
        };
        let Some(host_id) = (unsafe { str_arg(host_id) }) else {
            return 1;
        };
        let Some(origin) = (unsafe { str_arg(public_origin) }) else {
            return 1;
        };
        let generation = ENGINE_LIFETIME.snapshot();
        if local_port == 0
            || !ALIVE.load(Ordering::SeqCst)
            || ENGINE_PORT.load(Ordering::SeqCst) != local_port
        {
            return 1;
        }
        let Ok(origin) = managed_origin(origin, host_id) else {
            return 1;
        };
        if validate_endpoint(endpoint).is_err() {
            return 1;
        }
        let Some(root) = mold_core::Config::mold_dir() else {
            return 1;
        };
        let Ok(identity) = mold_relay::read_identity(&root.join("secure/host.identity")) else {
            return 1;
        };
        if mold_secure::namespace(&identity.signing_public()) != host_id {
            return 1;
        }
        let identity = Arc::new(identity);
        let Some(policy) = connection_addresses().watch_secure_policy() else {
            return 1;
        };
        let Ok(mut slot) = CONNECTOR.lock() else {
            return 1;
        };
        if !engine_matches(generation) || ENGINE_PORT.load(Ordering::SeqCst) != local_port {
            return 1;
        }
        if slot
            .as_ref()
            .is_some_and(|connector| !connector.thread.is_finished())
        {
            return 2;
        }
        if let Some(previous) = slot.take() {
            let _ = previous.thread.join();
        }
        let _ = connection_addresses().set_managed_relay_origin(None);
        let cancel = CancellationToken::new();
        let thread_cancel = cancel.clone();
        let alive = Arc::new(AtomicBool::new(true));
        let thread_alive = alive.clone();
        let ready = Arc::new(AtomicBool::new(false));
        let thread_ready = ready.clone();
        let published_endpoint = endpoint.to_owned();
        let endpoint = endpoint.to_owned();
        let Ok(thread) = std::thread::Builder::new()
            .name("mold-managed-relay".into())
            .spawn(move || {
                guarded("relay_runtime", (), || {
                    let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    else {
                        return;
                    };
                    runtime.block_on(async {
                        let target = ([127, 0, 0, 1], local_port).into();
                        let (status, mut receiver) = tokio::sync::watch::channel(false);
                        let connector = mold_relay::secure::connect_with_status(
                            &endpoint,
                            mold_relay::secure::HostConfig {
                                identity,
                                target,
                                policy,
                            },
                            thread_cancel.clone(),
                            false,
                            status,
                        );
                        tokio::pin!(connector);
                        let mut watch = tokio::time::interval(Duration::from_millis(250));
                        loop {
                            tokio::select! {
                                _ = thread_cancel.cancelled() => break,
                                _ = &mut connector => break,
                                changed = receiver.changed() => {
                                    if changed.is_err() { break; }
                                    thread_ready.store(*receiver.borrow(), Ordering::SeqCst);
                                },
                                _ = watch.tick() => if !engine_matches(generation) { break }
                            }
                        }
                    });
                    runtime.shutdown_timeout(Duration::from_secs(1));
                });
                let _publication = PUBLICATION.lock().unwrap_or_else(|p| p.into_inner());
                thread_alive.store(false, Ordering::SeqCst);
                thread_ready.store(false, Ordering::SeqCst);
                let _ = connection_addresses().set_managed_relay_origin(None);
            })
        else {
            return 1;
        };
        *slot = Some(Connector {
            generation,
            cancel,
            thread,
            origin,
            alive,
            ready,
        });
        let _publication = PUBLICATION.lock().unwrap_or_else(|p| p.into_inner());
        // Engine exit invalidates without waiting for CONNECTOR. A start that
        // raced exit must cancel before it can republish addresses or LAN access.
        if !engine_matches(generation) || ENGINE_PORT.load(Ordering::SeqCst) != local_port {
            if let Some(connector) = slot.as_ref() {
                connector.cancel.cancel();
            }
            return 1;
        }
        if connection_addresses()
            .set_secure_websocket_url(Some(&published_endpoint))
            .is_err()
        {
            if let Some(connector) = slot.as_ref() {
                connector.cancel.cancel();
            }
            return 1;
        }
        connection_addresses().set_local_network_enabled(true);
        0
    })
}

#[no_mangle]
pub extern "C" fn mold_relay_is_alive() -> bool {
    guarded("relay_is_alive", false, || {
        ALIVE.load(Ordering::SeqCst)
            && CONNECTOR.lock().is_ok_and(|slot| {
                slot.as_ref().is_some_and(|connector| {
                    engine_matches(connector.generation)
                        && connector.alive.load(Ordering::SeqCst)
                        && !connector.cancel.is_cancelled()
                        && !connector.thread.is_finished()
                })
            })
    })
}

#[no_mangle]
pub extern "C" fn mold_relay_is_ready() -> bool {
    mold_relay_is_alive()
        && CONNECTOR.lock().is_ok_and(|slot| {
            slot.as_ref()
                .is_some_and(|connector| connector.ready.load(Ordering::SeqCst))
        })
}

/// Copies the public self-certifying namespace, without exposing identity secrets.
/// # Safety
/// `output` must refer to a writable buffer of at least `capacity` bytes.
#[no_mangle]
pub unsafe extern "C" fn mold_secure_host_namespace(output: *mut u8, capacity: usize) -> i32 {
    guarded("secure_host_namespace", 1, || {
        if output.is_null() || capacity < 32 || !ALIVE.load(Ordering::SeqCst) {
            return 1;
        }
        let Some(root) = mold_core::Config::mold_dir() else {
            return 1;
        };
        let Ok(identity) = mold_relay::read_identity(&root.join("secure/host.identity")) else {
            return 1;
        };
        let namespace = mold_secure::namespace(&identity.signing_public());
        unsafe {
            std::ptr::copy_nonoverlapping(namespace.as_ptr(), output, 32);
        }
        0
    })
}

#[no_mangle]
pub extern "C" fn mold_relay_stop() -> i32 {
    guarded("relay_stop", 1, || {
        connection_addresses().set_local_network_enabled(false);
        let _ = connection_addresses().set_secure_websocket_url(None);
        let Ok(mut slot) = CONNECTOR.lock() else {
            return 1;
        };
        let _ = connection_addresses().set_managed_relay_origin(None);
        stop_connector(&mut slot, Duration::from_secs(2))
    })
}

fn stop_connector(slot: &mut Option<Connector>, budget: Duration) -> i32 {
    let Some(connector) = slot.as_ref() else {
        return 0;
    };
    connector.cancel.cancel();
    let deadline = Instant::now() + budget;
    while !connector.thread.is_finished() {
        if Instant::now() >= deadline {
            return 1;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let Some(connector) = slot.take() else {
        return 0;
    };
    if connector.thread.join().is_err() {
        return 1;
    }
    0
}

pub(super) fn cancel_for_engine_exit() {
    // Monotonic withdrawal cannot be missed when ALIVE becomes true again on
    // a rapid restart, even if another thread currently owns CONNECTOR.
    ENGINE_LIFETIME.invalidate();
    connection_addresses().set_regular_network_enabled(false);
    // Engine exit must never wait behind native stop's join budget.
    if let Ok(slot) = CONNECTOR.try_lock() {
        if let Some(connector) = slot.as_ref() {
            connector.cancel.cancel();
        }
    }
    let _publication = PUBLICATION.lock().unwrap_or_else(|p| p.into_inner());
    connection_addresses().set_local_network_enabled(false);
    let _ = connection_addresses().set_secure_websocket_url(None);
    let _ = connection_addresses().set_managed_relay_origin(None);
}

/// # Safety
/// A non-null pointer must reference a valid NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn mold_relay_set_public_origin(origin: *const c_char) -> i32 {
    guarded("relay_set_public_origin", 1, || {
        if origin.is_null() {
            let _publication = PUBLICATION.lock().unwrap_or_else(|p| p.into_inner());
            return i32::from(
                connection_addresses()
                    .set_managed_relay_origin(None)
                    .is_err(),
            );
        }
        let Some(origin) = (unsafe { str_arg(origin) }) else {
            return 1;
        };
        let Ok(url) = url::Url::parse(origin) else {
            return 1;
        };
        let Ok(slot) = CONNECTOR.lock() else { return 1 };
        let Some(connector) = slot.as_ref() else {
            return 1;
        };
        let _publication = PUBLICATION.lock().unwrap_or_else(|p| p.into_inner());
        if !engine_matches(connector.generation)
            || !connector.alive.load(Ordering::SeqCst)
            || connector.cancel.is_cancelled()
            || connector.thread.is_finished()
            || url.origin().ascii_serialization() != connector.origin
        {
            return 1;
        }
        i32::from(
            connection_addresses()
                .set_managed_relay_origin(Some(origin))
                .is_err(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn endpoint_is_publishable_before_any_connector_starts() {
        assert!(validate_endpoint("wss://relay.example/prod").is_ok());
        assert!(validate_endpoint(&format!("wss://relay.example/{}", "a".repeat(2048))).is_err());
    }
    #[test]
    fn bounded_stop_keeps_ownership_until_finished_and_allows_restart() {
        let cancel = CancellationToken::new();
        let (release, wait) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            let _ = wait.recv();
        });
        let mut slot = Some(Connector {
            generation: 0,
            cancel: cancel.clone(),
            thread,
            origin: "https://example.com".into(),
            alive: Arc::new(AtomicBool::new(true)),
            ready: Arc::new(AtomicBool::new(false)),
        });
        assert_eq!(stop_connector(&mut slot, Duration::from_millis(1)), 1);
        assert!(cancel.is_cancelled());
        assert!(
            slot.is_some(),
            "timed-out thread retains exclusive connector ownership"
        );
        release.send(()).unwrap();
        assert_eq!(stop_connector(&mut slot, Duration::from_secs(1)), 0);
        assert!(slot.is_none());
        slot = Some(Connector {
            generation: 0,
            cancel: CancellationToken::new(),
            thread: std::thread::spawn(|| {}),
            origin: "https://example.com".into(),
            alive: Arc::new(AtomicBool::new(true)),
            ready: Arc::new(AtomicBool::new(false)),
        });
        assert_eq!(stop_connector(&mut slot, Duration::from_secs(1)), 0);
    }
    #[test]
    fn origin_is_bound_to_enrolled_namespace() {
        let host = "0123456789abcdef0123456789abcdef";
        assert_eq!(
            managed_origin(&format!("https://{host}.example.com/"), host).unwrap(),
            format!("https://{host}.example.com")
        );
        assert!(managed_origin("https://other.example.com", host).is_err());
        assert!(managed_origin(&format!("https://{host}.example.com/path"), host).is_err());
        assert!(managed_origin(&format!("http://{host}.example.com"), host).is_err());
    }
}
