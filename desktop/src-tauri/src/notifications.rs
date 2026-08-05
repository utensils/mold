use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NotificationAction {
    Gallery { filename: String },
}

static PENDING_ACTION: std::sync::Mutex<Option<NotificationAction>> = std::sync::Mutex::new(None);

/// Send through Apple's current UserNotifications framework so Notification
/// Center associates the alert with the running signed Mold bundle and uses
/// its CFBundleIconFile. The previous NSUserNotification/private-identity-image
/// path is ignored by current macOS releases and produced a generic icon.
#[tauri::command]
pub async fn send_native_notification(
    app: tauri::AppHandle,
    title: String,
    body: Option<String>,
    action: Option<NotificationAction>,
) -> Result<bool, String> {
    #[cfg(target_os = "macos")]
    {
        let _ = app;
        // UNUserNotificationCenter aborts the process with an uncatchable
        // NSInternalInconsistencyException ("bundleProxyForCurrentProcess is
        // nil") when the binary runs outside an .app bundle — i.e. `tauri dev`.
        if macos_bundle_identifier().is_none() {
            tracing::warn!(
                "skipping native notification: not running from an app bundle (dev mode)"
            );
            return Ok(false);
        }
        tauri::async_runtime::spawn_blocking(move || {
            send_macos_notification(&title, body.as_deref(), action.as_ref())
        })
        .await
        .map_err(|error| error.to_string())??;
        Ok(true)
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (app, title, body, action);
        Ok(false)
    }
}

#[tauri::command]
pub fn take_notification_action() -> Option<NotificationAction> {
    PENDING_ACTION.lock().ok()?.take()
}

/// Some(id) only when the process runs from a real .app bundle; None for bare
/// binaries (dev builds, test harnesses), where UserNotifications cannot work.
#[cfg(target_os = "macos")]
fn macos_bundle_identifier() -> Option<String> {
    use objc2_foundation::NSBundle;
    NSBundle::mainBundle()
        .bundleIdentifier()
        .map(|id| id.to_string())
}

#[cfg(target_os = "macos")]
fn wait_for_callback<T>(
    receiver: std::sync::mpsc::Receiver<T>,
    operation: &str,
) -> Result<T, String> {
    receiver
        .recv_timeout(std::time::Duration::from_secs(5))
        .map_err(|_| format!("Timed out while {operation}."))
}

#[cfg(target_os = "macos")]
fn send_macos_notification(
    title: &str,
    body: Option<&str>,
    action: Option<&NotificationAction>,
) -> Result<(), String> {
    use base64::Engine;
    use block2::RcBlock;
    use objc2_foundation::{NSError, NSString};
    use objc2_user_notifications::{
        UNAuthorizationOptions, UNMutableNotificationContent, UNNotificationRequest,
        UNUserNotificationCenter,
    };

    let center = UNUserNotificationCenter::currentNotificationCenter();

    // Requesting an already-decided permission is cheap and asynchronous; it
    // also covers a fresh install before the JS fallback has asked permission.
    let (auth_tx, auth_rx) = std::sync::mpsc::sync_channel(1);
    let auth = RcBlock::new(move |granted: objc2::runtime::Bool, error: *mut NSError| {
        let _ = auth_tx.send((granted.as_bool(), error.is_null()));
    });
    center.requestAuthorizationWithOptions_completionHandler(
        UNAuthorizationOptions::Alert | UNAuthorizationOptions::Sound,
        &auth,
    );
    let (granted, no_error) = wait_for_callback(auth_rx, "requesting notification permission")?;
    if !no_error {
        return Err("macOS couldn't resolve notification permission.".into());
    }
    if !granted {
        return Err("Notifications are not permitted for Mold.".into());
    }

    let content = UNMutableNotificationContent::new();
    content.setTitle(&NSString::from_str(title));
    content.setBody(&NSString::from_str(body.unwrap_or("")));
    let id = match action {
        Some(NotificationAction::Gallery { filename }) => format!(
            "mold-gallery:{}:{}",
            uuid::Uuid::new_v4(),
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(filename.as_bytes())
        ),
        None => uuid::Uuid::new_v4().to_string(),
    };
    let identifier = NSString::from_str(&id);
    let request =
        UNNotificationRequest::requestWithIdentifier_content_trigger(&identifier, &content, None);

    let (delivery_tx, delivery_rx) = std::sync::mpsc::sync_channel(1);
    let completion = RcBlock::new(move |error: *mut NSError| {
        let _ = delivery_tx.send(error.is_null());
    });
    center.addNotificationRequest_withCompletionHandler(&request, Some(&completion));
    if !wait_for_callback(delivery_rx, "delivering the notification")? {
        return Err("macOS rejected the notification request.".into());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn action_from_identifier(identifier: &str) -> Option<NotificationAction> {
    use base64::Engine;

    let mut parts = identifier.splitn(3, ':');
    if parts.next()? != "mold-gallery" || parts.next()?.is_empty() {
        return None;
    }
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(parts.next()?)
        .ok()?;
    let filename = String::from_utf8(bytes).ok()?;
    (!filename.is_empty()).then_some(NotificationAction::Gallery { filename })
}

#[cfg(target_os = "macos")]
mod delegate {
    use super::{action_from_identifier, NotificationAction, PENDING_ACTION};
    use block2::DynBlock;
    use objc2::runtime::ProtocolObject;
    use objc2::{define_class, msg_send, MainThreadMarker, MainThreadOnly};
    use objc2_foundation::{NSObject, NSObjectProtocol};
    use objc2_user_notifications::{
        UNNotificationResponse, UNUserNotificationCenter, UNUserNotificationCenterDelegate,
    };
    use tauri::{Emitter, Manager};

    static APP: std::sync::OnceLock<tauri::AppHandle> = std::sync::OnceLock::new();

    define_class!(
        #[unsafe(super(NSObject))]
        #[name = "MoldNotificationDelegate"]
        #[thread_kind = MainThreadOnly]
        struct MoldNotificationDelegate;

        unsafe impl NSObjectProtocol for MoldNotificationDelegate {}

        unsafe impl UNUserNotificationCenterDelegate for MoldNotificationDelegate {
            #[unsafe(method(userNotificationCenter:didReceiveNotificationResponse:withCompletionHandler:))]
            fn did_receive(
                &self,
                _center: &UNUserNotificationCenter,
                response: &UNNotificationResponse,
                completion: &DynBlock<dyn Fn()>,
            ) {
                let identifier = response.notification().request().identifier().to_string();
                if let Some(action) = action_from_identifier(&identifier) {
                    activate(action);
                }
                completion.call(());
            }
        }
    );

    impl MoldNotificationDelegate {
        fn new(mtm: MainThreadMarker) -> objc2::rc::Retained<Self> {
            let this = Self::alloc(mtm).set_ivars(());
            unsafe { msg_send![super(this), init] }
        }
    }

    fn activate(action: NotificationAction) {
        if let Ok(mut pending) = PENDING_ACTION.lock() {
            *pending = Some(action.clone());
        }
        let Some(app) = APP.get() else { return };
        if let Some(window) = app.get_webview_window("main") {
            let _ = window.unminimize();
            let _ = window.show();
            let _ = window.set_focus();
        }
        let _ = app.emit("notification-action", action);
    }

    pub fn install(app: &tauri::AppHandle) {
        let _ = APP.set(app.clone());
        let Some(mtm) = MainThreadMarker::new() else {
            tracing::warn!("notification delegate must be installed on the main thread");
            return;
        };
        let delegate = MoldNotificationDelegate::new(mtm);
        let center = UNUserNotificationCenter::currentNotificationCenter();
        center.setDelegate(Some(ProtocolObject::from_ref(&*delegate)));
        // UNUserNotificationCenter.delegate is weak. The delegate is app-wide
        // and intentionally lives until process exit.
        let _ = objc2::rc::Retained::into_raw(delegate);
    }
}

#[cfg(target_os = "macos")]
pub fn install_notification_delegate(app: &tauri::AppHandle) {
    delegate::install(app);
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::{
        action_from_identifier, macos_bundle_identifier, wait_for_callback, NotificationAction,
    };
    use base64::Engine;

    #[test]
    fn callback_wait_returns_the_delivered_value() {
        let (sender, receiver) = std::sync::mpsc::channel();
        sender.send(true).unwrap();
        assert!(wait_for_callback(receiver, "testing").unwrap());
    }

    /// The test binary runs bare from target/, exactly like `tauri dev` — the
    /// environment where UNUserNotificationCenter throws an uncatchable
    /// NSInternalInconsistencyException. The guard must detect it.
    #[test]
    fn bare_binary_has_no_bundle_identity() {
        assert!(macos_bundle_identifier().is_none());
    }

    #[test]
    fn gallery_action_round_trips_unicode_filename_from_request_identifier() {
        let filename = "mold video — final.mp4";
        let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(filename.as_bytes());
        assert_eq!(
            action_from_identifier(&format!("mold-gallery:request-id:{encoded}")),
            Some(NotificationAction::Gallery {
                filename: filename.into()
            })
        );
        assert_eq!(action_from_identifier("ordinary-request-id"), None);
    }
}
