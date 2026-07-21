//! Bonjour discovery through Apple's DNS-SD API. This uses the sanctioned
//! mDNSResponder path on iOS and therefore needs no multicast entitlement.

use std::time::Duration;

use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveredHost {
    pub name: String,
    pub host: String,
    pub port: u16,
}

#[tauri::command]
pub async fn discover_mold_hosts(timeout_ms: Option<u32>) -> Result<Vec<DiscoveredHost>, String> {
    let timeout = Duration::from_millis(u64::from(timeout_ms.unwrap_or(2500).clamp(500, 10_000)));
    tauri::async_runtime::spawn_blocking(move || scan(timeout))
        .await
        .map_err(|error| format!("discovery task failed: {error}"))?
}

#[cfg(not(target_vendor = "apple"))]
fn scan(_timeout: Duration) -> Result<Vec<DiscoveredHost>, String> {
    Err("Bonjour discovery is available in the Apple build".into())
}

#[cfg(target_vendor = "apple")]
fn scan(timeout: Duration) -> Result<Vec<DiscoveredHost>, String> {
    apple::scan(timeout)
}

#[cfg(target_vendor = "apple")]
mod apple {
    use std::ffi::{c_char, c_void, CStr, CString};
    use std::time::{Duration, Instant};

    use super::DiscoveredHost;

    type ServiceRef = *mut c_void;
    type Flags = u32;
    type Error = i32;
    const ADD: Flags = 0x2;
    const SERVICE: &CStr = c"_mold._tcp";

    type BrowseReply = unsafe extern "C" fn(
        ServiceRef,
        Flags,
        u32,
        Error,
        *const c_char,
        *const c_char,
        *const c_char,
        *mut c_void,
    );
    type ResolveReply = unsafe extern "C" fn(
        ServiceRef,
        Flags,
        u32,
        Error,
        *const c_char,
        *const c_char,
        u16,
        u16,
        *const u8,
        *mut c_void,
    );

    unsafe extern "C" {
        fn DNSServiceBrowse(
            service: *mut ServiceRef,
            flags: Flags,
            interface: u32,
            regtype: *const c_char,
            domain: *const c_char,
            callback: BrowseReply,
            context: *mut c_void,
        ) -> Error;
        fn DNSServiceResolve(
            service: *mut ServiceRef,
            flags: Flags,
            interface: u32,
            name: *const c_char,
            regtype: *const c_char,
            domain: *const c_char,
            callback: ResolveReply,
            context: *mut c_void,
        ) -> Error;
        fn DNSServiceRefSockFD(service: ServiceRef) -> i32;
        fn DNSServiceProcessResult(service: ServiceRef) -> Error;
        fn DNSServiceRefDeallocate(service: ServiceRef);
    }

    struct Hit {
        name: CString,
        regtype: CString,
        domain: CString,
        interface: u32,
    }

    #[derive(Default)]
    struct BrowseState {
        hits: Vec<Hit>,
    }

    struct ResolveState {
        service: ServiceRef,
        done: bool,
        host: Option<DiscoveredHost>,
    }

    unsafe extern "C" fn browsed(
        _service: ServiceRef,
        flags: Flags,
        interface: u32,
        error: Error,
        name: *const c_char,
        regtype: *const c_char,
        domain: *const c_char,
        context: *mut c_void,
    ) {
        if error != 0 || flags & ADD == 0 {
            return;
        }
        let state = unsafe { &mut *(context as *mut BrowseState) };
        let copy = |value: *const c_char| unsafe { CStr::from_ptr(value) }.to_owned();
        state.hits.push(Hit {
            name: copy(name),
            regtype: copy(regtype),
            domain: copy(domain),
            interface,
        });
    }

    unsafe extern "C" fn resolved(
        _service: ServiceRef,
        _flags: Flags,
        _interface: u32,
        error: Error,
        fullname: *const c_char,
        hosttarget: *const c_char,
        port_be: u16,
        _txt_len: u16,
        _txt: *const u8,
        context: *mut c_void,
    ) {
        let state = unsafe { &mut *(context as *mut ResolveState) };
        state.done = true;
        if error != 0 {
            return;
        }
        let host = unsafe { CStr::from_ptr(hosttarget) }
            .to_string_lossy()
            .trim_end_matches('.')
            .to_string();
        let full = unsafe { CStr::from_ptr(fullname) }.to_string_lossy();
        let name = full.split("._mold").next().unwrap_or(&host).to_string();
        state.host = Some(DiscoveredHost {
            name,
            host,
            port: u16::from_be(port_be),
        });
    }

    pub fn scan(timeout: Duration) -> Result<Vec<DiscoveredHost>, String> {
        let deadline = Instant::now() + timeout;
        let mut browse_state = Box::new(BrowseState::default());
        let mut browse: ServiceRef = std::ptr::null_mut();
        let error = unsafe {
            DNSServiceBrowse(
                &mut browse,
                0,
                0,
                SERVICE.as_ptr(),
                std::ptr::null(),
                browsed,
                (&mut *browse_state) as *mut BrowseState as *mut c_void,
            )
        };
        if error != 0 {
            return Err(format!("DNSServiceBrowse failed: {error}"));
        }

        let mut resolves: Vec<Box<ResolveState>> = Vec::new();
        while Instant::now() < deadline {
            for hit in browse_state.hits.drain(..) {
                let mut state = Box::new(ResolveState {
                    service: std::ptr::null_mut(),
                    done: false,
                    host: None,
                });
                let error = unsafe {
                    DNSServiceResolve(
                        &mut state.service,
                        0,
                        hit.interface,
                        hit.name.as_ptr(),
                        hit.regtype.as_ptr(),
                        hit.domain.as_ptr(),
                        resolved,
                        (&mut *state) as *mut ResolveState as *mut c_void,
                    )
                };
                if error == 0 {
                    resolves.push(state);
                }
            }

            let mut services = vec![browse];
            services.extend(
                resolves
                    .iter()
                    .filter(|item| !item.done)
                    .map(|item| item.service),
            );
            let mut fds: Vec<libc::pollfd> = services
                .iter()
                .map(|service| libc::pollfd {
                    fd: unsafe { DNSServiceRefSockFD(*service) },
                    events: libc::POLLIN,
                    revents: 0,
                })
                .collect();
            let wait = deadline
                .saturating_duration_since(Instant::now())
                .as_millis()
                .min(200) as i32;
            if unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, wait) } < 0 {
                break;
            }
            for (index, fd) in fds.iter().enumerate() {
                if fd.revents & libc::POLLIN != 0 {
                    unsafe { DNSServiceProcessResult(services[index]) };
                }
            }
        }

        unsafe { DNSServiceRefDeallocate(browse) };
        let mut found = Vec::new();
        for state in resolves {
            unsafe { DNSServiceRefDeallocate(state.service) };
            if let Some(host) = state.host {
                if !found
                    .iter()
                    .any(|item: &DiscoveredHost| item.host == host.host && item.port == host.port)
                {
                    found.push(host);
                }
            }
        }
        found.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(found)
    }
}
