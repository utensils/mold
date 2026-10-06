//! Cancellable managed outbound connector, independent of the engine runtime.
use super::{connection_addresses, guarded, str_arg, ALIVE, ENGINE_PORT};
use std::ffi::c_char;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

struct Connector {
    cancel: CancellationToken,
    thread: std::thread::JoinHandle<()>,
    origin: String,
    alive: Arc<AtomicBool>,
}
static CONNECTOR: Mutex<Option<Connector>> = Mutex::new(None);
// Serialize publication with terminal withdrawal without holding the join mutex.
static PUBLICATION: Mutex<()> = Mutex::new(());

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
    token: *const c_char,
    host_id: *const c_char,
    local_port: u16,
    public_origin: *const c_char,
) -> i32 {
    guarded("relay_start", 1, || {
        let Some(endpoint) = (unsafe { str_arg(endpoint) }) else {
            return 1;
        };
        let Some(token) = (unsafe { str_arg(token) }) else {
            return 1;
        };
        let Some(host_id) = (unsafe { str_arg(host_id) }) else {
            return 1;
        };
        let Some(origin) = (unsafe { str_arg(public_origin) }) else {
            return 1;
        };
        if local_port == 0
            || !ALIVE.load(Ordering::SeqCst)
            || ENGINE_PORT.load(Ordering::SeqCst) != local_port
        {
            return 1;
        }
        let Ok(origin) = managed_origin(origin, host_id) else {
            return 1;
        };
        if mold_relay::aws::validate_aws_endpoint(endpoint, false).is_err()
            || token.len() < 32
            || token.len() > 4096
            || !token.bytes().all(|b| b.is_ascii_graphic())
        {
            return 1;
        }
        let Ok(mut slot) = CONNECTOR.lock() else {
            return 1;
        };
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
        let endpoint = endpoint.to_owned();
        let token = token.to_owned();
        let host_id = host_id.to_owned();
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
                        let connector = mold_relay::aws::connect_managed(
                            &endpoint,
                            ([127, 0, 0, 1], local_port).into(),
                            token,
                            Some(host_id),
                            false,
                            thread_cancel.clone(),
                            mold_relay::RelayOptions::default(),
                        );
                        tokio::pin!(connector);
                        let mut watch = tokio::time::interval(Duration::from_millis(250));
                        loop {
                            tokio::select! {
                                _ = thread_cancel.cancelled() => break,
                                _ = &mut connector => break,
                                _ = watch.tick() => if !ALIVE.load(Ordering::SeqCst) { break }
                            }
                        }
                    });
                    runtime.shutdown_timeout(Duration::from_secs(1));
                });
                let _publication = PUBLICATION.lock().unwrap_or_else(|p| p.into_inner());
                thread_alive.store(false, Ordering::SeqCst);
                let _ = connection_addresses().set_managed_relay_origin(None);
            })
        else {
            return 1;
        };
        *slot = Some(Connector {
            cancel,
            thread,
            origin,
            alive,
        });
        0
    })
}

#[no_mangle]
pub extern "C" fn mold_relay_is_alive() -> bool {
    guarded("relay_is_alive", false, || {
        ALIVE.load(Ordering::SeqCst)
            && CONNECTOR.lock().is_ok_and(|slot| {
                slot.as_ref().is_some_and(|connector| {
                    connector.alive.load(Ordering::SeqCst)
                        && !connector.cancel.is_cancelled()
                        && !connector.thread.is_finished()
                })
            })
    })
}

#[no_mangle]
pub extern "C" fn mold_relay_stop() -> i32 {
    guarded("relay_stop", 1, || {
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
    // Engine exit must never wait behind native stop's join budget.
    if let Ok(slot) = CONNECTOR.try_lock() {
        if let Some(connector) = slot.as_ref() {
            connector.cancel.cancel();
        }
    }
    let _publication = PUBLICATION.lock().unwrap_or_else(|p| p.into_inner());
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
        if !ALIVE.load(Ordering::SeqCst)
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
    fn bounded_stop_keeps_ownership_until_finished_and_allows_restart() {
        let cancel = CancellationToken::new();
        let (release, wait) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            let _ = wait.recv();
        });
        let mut slot = Some(Connector {
            cancel: cancel.clone(),
            thread,
            origin: "https://example.com".into(),
            alive: Arc::new(AtomicBool::new(true)),
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
            cancel: CancellationToken::new(),
            thread: std::thread::spawn(|| {}),
            origin: "https://example.com".into(),
            alive: Arc::new(AtomicBool::new(true)),
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
