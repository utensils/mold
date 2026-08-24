use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

#[derive(Clone, Default)]
pub struct MobileBackgroundTaskState {
    tasks: Arc<Mutex<HashMap<String, usize>>>,
}

static NEXT_BACKGROUND_TASK_TOKEN: AtomicU64 = AtomicU64::new(1);

/// Ask iOS for its finite background-execution window while the phone is
/// preparing or admitting remote work. This is deliberately an assertion,
/// not a background mode: once the server accepts a generation, the server
/// owns it and foreground recovery reconnects to the durable job.
#[tauri::command]
pub fn begin_mobile_background_task(
    window: tauri::WebviewWindow,
    state: tauri::State<'_, MobileBackgroundTaskState>,
    name: String,
) -> Result<String, String> {
    let token = format!(
        "mobile-background-{}",
        NEXT_BACKGROUND_TASK_TOKEN.fetch_add(1, Ordering::Relaxed)
    );
    platform_begin_background_task(window, state.tasks.clone(), token, name)
}

/// Release a background assertion after the remote host has accepted the work
/// (or after preparation failed/cancelled). Releasing an already-expired token
/// is intentionally a no-op.
#[tauri::command]
pub fn end_mobile_background_task(
    window: tauri::WebviewWindow,
    state: tauri::State<'_, MobileBackgroundTaskState>,
    token: String,
) -> Result<(), String> {
    platform_end_background_task(window, state.tasks.clone(), token)
}

#[cfg(target_os = "ios")]
fn platform_begin_background_task(
    window: tauri::WebviewWindow,
    tasks: Arc<Mutex<HashMap<String, usize>>>,
    token: String,
    name: String,
) -> Result<String, String> {
    use block2::RcBlock;
    use objc2::MainThreadMarker;
    use objc2_foundation::NSString;
    use objc2_ui_kit::{UIApplication, UIBackgroundTaskInvalid};

    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    let callback_tasks = tasks.clone();
    let callback_token = token.clone();
    window
        .with_webview(move |_webview| {
            let result = (|| {
                let mtm = MainThreadMarker::new().ok_or_else(|| {
                    "iOS background task must start on the main thread".to_string()
                })?;
                let application = UIApplication::sharedApplication(mtm);
                let expiry_tasks = callback_tasks.clone();
                let expiry_token = callback_token.clone();
                let expiration_handler = RcBlock::new(move || {
                    let identifier = expiry_tasks
                        .lock()
                        .ok()
                        .and_then(|mut active| active.remove(&expiry_token));
                    if let (Some(identifier), Some(mtm)) = (identifier, MainThreadMarker::new()) {
                        UIApplication::sharedApplication(mtm).endBackgroundTask(identifier);
                    }
                });
                let task_name = NSString::from_str(&name);
                let identifier = application.beginBackgroundTaskWithName_expirationHandler(
                    Some(&task_name),
                    Some(&expiration_handler),
                );
                // SAFETY: UIKit exports this process-wide constant.
                if identifier == unsafe { UIBackgroundTaskInvalid } {
                    return Err("iOS did not grant background execution time".to_string());
                }
                callback_tasks
                    .lock()
                    .map_err(|_| "mobile background task state is unavailable".to_string())?
                    .insert(callback_token.clone(), identifier);
                Ok(callback_token.clone())
            })();
            let _ = sender.send(result);
        })
        .map_err(|error| format!("failed to start iOS background task: {error}"))?;

    receiver
        .recv()
        .map_err(|_| "iOS background task callback did not run".to_string())?
}

#[cfg(target_os = "ios")]
fn platform_end_background_task(
    window: tauri::WebviewWindow,
    tasks: Arc<Mutex<HashMap<String, usize>>>,
    token: String,
) -> Result<(), String> {
    use objc2::MainThreadMarker;
    use objc2_ui_kit::UIApplication;

    window
        .with_webview(move |_webview| {
            let identifier = tasks
                .lock()
                .ok()
                .and_then(|mut active| active.remove(&token));
            if let (Some(identifier), Some(mtm)) = (identifier, MainThreadMarker::new()) {
                UIApplication::sharedApplication(mtm).endBackgroundTask(identifier);
            }
        })
        .map_err(|error| format!("failed to end iOS background task: {error}"))
}

#[cfg(not(target_os = "ios"))]
fn platform_begin_background_task(
    _window: tauri::WebviewWindow,
    _tasks: Arc<Mutex<HashMap<String, usize>>>,
    token: String,
    _name: String,
) -> Result<String, String> {
    Ok(token)
}

#[cfg(not(target_os = "ios"))]
fn platform_end_background_task(
    _window: tauri::WebviewWindow,
    tasks: Arc<Mutex<HashMap<String, usize>>>,
    token: String,
) -> Result<(), String> {
    tasks
        .lock()
        .map_err(|_| "mobile background task state is unavailable".to_string())?
        .remove(&token);
    Ok(())
}
