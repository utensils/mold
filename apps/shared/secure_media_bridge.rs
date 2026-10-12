//! Shared desktop/mobile IPC for encrypted remote media. No persistent second store.
use mold_secure_client::{MediaBridge, PairedHost, SecureClient};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
use tauri::Manager;
struct Entry {
    owner: String,
    namespace: String,
    _bridge: MediaBridge,
    _permit: tokio::sync::OwnedSemaphorePermit,
}
#[derive(Default)]
struct Inner {
    entries: BTreeMap<String, Entry>,
    epochs: BTreeMap<String, u64>,
    namespace_epochs: BTreeMap<(String, String), u64>,
}
pub struct SecureForgetHook(pub fn(&tauri::WebviewWindow, &str) -> Result<(), String>);

pub struct MediaBridges {
    inner: Mutex<Inner>,
    slots: Arc<tokio::sync::Semaphore>,
}
impl Default for MediaBridges {
    fn default() -> Self {
        Self {
            inner: Default::default(),
            slots: Arc::new(tokio::sync::Semaphore::new(128)),
        }
    }
}
impl MediaBridges {
    pub fn close_window(&self, owner: &str) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        *inner.epochs.entry(owner.to_string()).or_default() += 1;
        inner.entries.retain(|_, entry| entry.owner != owner);
        inner
            .namespace_epochs
            .retain(|(entry_owner, _), _| entry_owner != owner);
        drop(inner);
        mold_secure_client::forget_ephemeral(owner, None);
        mold_secure_client::forget_regular_ephemeral(owner, None);
    }
}
fn caller_origin(window: &tauri::WebviewWindow) -> Result<String, String> {
    let url = window
        .url()
        .map_err(|_| "Cannot identify this app window".to_string())?;
    let host = url.host_str().unwrap_or_default();
    let local = host == "localhost"
        || host == "tauri.localhost"
        || host
            .trim_matches(['[', ']'])
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback());
    if !local || !matches!(url.scheme(), "tauri" | "http" | "https") {
        return Err("Secure media is available only to the trusted app window".into());
    }
    Ok(if url.scheme() == "tauri" {
        format!("tauri://{host}")
    } else {
        url.origin().ascii_serialization()
    })
}
#[derive(Serialize)]
pub struct MediaHandle {
    pub id: String,
    pub url: String,
}
#[tauri::command]
pub async fn secure_media_open(
    window: tauri::WebviewWindow,
    state: tauri::State<'_, MediaBridges>,
    host: PairedHost,
    path: String,
) -> Result<MediaHandle, String> {
    let origin = caller_origin(&window)?;
    let owner = window.label().to_string();
    let permit = state
        .slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| "Too many active media previews".to_string())?;
    let epoch = {
        let mut inner = state.inner.lock().unwrap_or_else(|e| e.into_inner());
        if window.app_handle().get_webview_window(&owner).is_none() {
            return Err("The media window closed".into());
        }
        let epoch = *inner.epochs.entry(owner.clone()).or_default();
        if !mold_secure_client::has_ephemeral_authority(&owner, &host) {
            return Err("This host pairing was removed or replaced".into());
        }
        let namespace_epoch = *inner
            .namespace_epochs
            .entry((owner.clone(), host.namespace.clone()))
            .or_default();
        (epoch, namespace_epoch)
    };
    let client = SecureClient::new(host.clone()).map_err(|e| e.to_string())?;
    let bridge = MediaBridge::start(client, host.origin.clone(), origin, vec![path.clone()])
        .await
        .map_err(|e| e.to_string())?;
    let url = bridge
        .url(&path)
        .ok_or_else(|| "Invalid media path".to_string())?
        .to_string();
    let id = uuid::Uuid::new_v4().simple().to_string();
    let mut inner = state.inner.lock().unwrap_or_else(|e| e.into_inner());
    if inner.epochs.get(&owner) != Some(&epoch.0)
        || inner
            .namespace_epochs
            .get(&(owner.clone(), host.namespace.clone()))
            != Some(&epoch.1)
        || !mold_secure_client::has_ephemeral_authority(&owner, &host)
        || window.app_handle().get_webview_window(&owner).is_none()
    {
        return Err("The media window closed".into());
    }
    inner.entries.insert(
        id.clone(),
        Entry {
            owner,
            namespace: host.namespace.clone(),
            _bridge: bridge,
            _permit: permit,
        },
    );
    Ok(MediaHandle { id, url })
}
#[tauri::command]
pub fn secure_media_close(
    window: tauri::WebviewWindow,
    state: tauri::State<'_, MediaBridges>,
    id: String,
) -> Result<(), String> {
    caller_origin(&window)?;
    let mut inner = state.inner.lock().unwrap_or_else(|e| e.into_inner());
    let entries = &mut inner.entries;
    if entries.get(&id).is_some_and(|e| e.owner != window.label()) {
        return Err("This media preview belongs to another window".into());
    }
    entries.remove(&id);
    Ok(())
}
#[tauri::command]
pub fn secure_host_register(window: tauri::WebviewWindow, host: PairedHost) -> Result<(), String> {
    caller_origin(&window)?;
    mold_secure_client::register_ephemeral(window.label(), host).map_err(|e| e.to_string())
}
#[tauri::command]
pub fn secure_host_forget(
    window: tauri::WebviewWindow,
    state: tauri::State<'_, MediaBridges>,
    namespace: String,
) -> Result<(), String> {
    caller_origin(&window)?;
    if let Some(hook) = window.try_state::<SecureForgetHook>() {
        (hook.0)(&window, &namespace)?;
    }
    let mut inner = state.inner.lock().unwrap_or_else(|e| e.into_inner());
    *inner
        .namespace_epochs
        .entry((window.label().to_string(), namespace.clone()))
        .or_default() += 1;
    inner
        .entries
        .retain(|_, entry| entry.owner != window.label() || entry.namespace != namespace);
    drop(inner);
    mold_secure_client::forget_ephemeral_namespace(window.label(), &namespace);
    Ok(())
}

#[tauri::command]
pub fn regular_host_register(
    window: tauri::WebviewWindow,
    host: mold_secure_client::RegularHost,
) -> Result<(), String> {
    caller_origin(&window)?;
    mold_secure_client::register_regular_ephemeral(window.label(), host).map_err(|e| e.to_string())
}
#[tauri::command]
pub fn regular_host_forget(window: tauri::WebviewWindow, origin: String) -> Result<(), String> {
    caller_origin(&window)?;
    mold_secure_client::forget_regular_ephemeral(window.label(), Some(&origin));
    Ok(())
}
