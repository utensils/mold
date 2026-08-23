use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NotificationAction {
    Gallery { filename: Option<String> },
    Create,
    Models,
    Updates,
}

static PENDING_ACTION: std::sync::Mutex<Option<NotificationAction>> = std::sync::Mutex::new(None);

fn activate(app: &tauri::AppHandle, action: NotificationAction) {
    use tauri::{Emitter, Manager};

    if let Ok(mut pending) = PENDING_ACTION.lock() {
        *pending = Some(action.clone());
    }
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
    let _ = app.emit("notification-action", action);
}

/// Send through the platform path that preserves a click action. macOS uses
/// UserNotifications so Notification Center associates alerts with the signed
/// Mold bundle; Linux retains the XDG notification handle until activation.
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

    #[cfg(target_os = "linux")]
    {
        let Some(action) = action else {
            return Ok(false);
        };
        tauri::async_runtime::spawn_blocking(move || {
            let mut notification = notify_rust::Notification::new();
            notification.summary(&title).appname("Mold");
            if let Some(body) = body.as_deref() {
                notification.body(body);
            }
            notification.action("default", "Open Mold");
            let handle = notification.show().map_err(|error| error.to_string())?;
            std::thread::spawn(move || {
                handle.wait_for_action(move |response| {
                    if response == "default" {
                        activate(&app, action);
                    }
                });
            });
            Ok::<_, String>(())
        })
        .await
        .map_err(|error| error.to_string())??;
        Ok(true)
    }

    #[cfg(target_os = "windows")]
    {
        let Some(action) = action else {
            return Ok(false);
        };
        tauri::async_runtime::spawn_blocking(move || {
            windows_toast(app, &title, body.as_deref(), action)
        })
        .await
        .map_err(|error| error.to_string())??;
        Ok(true)
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        let _ = (app, title, body, action);
        Ok(false)
    }
}

/// Windows toasts are addressed by AppUserModelID, and only an *installed* app
/// owns one: the NSIS installer's Start Menu shortcut registers the bundle
/// identifier. A `tauri dev` run has no shortcut, so it falls back to
/// PowerShell's well-known AUMID — the toast and its click routing are real
/// either way, the alert is just attributed to PowerShell in dev.
#[cfg(target_os = "windows")]
fn windows_toast(
    app: tauri::AppHandle,
    title: &str,
    body: Option<&str>,
    action: NotificationAction,
) -> Result<(), String> {
    use tauri_winrt_notification::Toast;

    // Routing is the whole reason this bypasses the plugin's own toast: a
    // click has to reach the same `activate` the macOS delegate and the Linux
    // XDG handle reach, or the alert names a print the app never opens.
    let build = |app_id: &str| {
        let mut toast = Toast::new(app_id).title(title);
        if let Some(body) = body {
            toast = toast.text1(body);
        }
        let routed = app.clone();
        let action = action.clone();
        toast.on_activated(move |_| {
            activate(&routed, action.clone());
            Ok(())
        })
    };

    let identifier = tauri::Manager::config(&app).identifier.clone();
    if build(&identifier).show().is_ok() {
        return Ok(());
    }
    // An unregistered AUMID is the expected dev-mode failure, not a broken
    // notification system — retry under PowerShell's before giving up, so the
    // JS fallback is reserved for a genuinely unavailable toast surface.
    build(Toast::POWERSHELL_APP_ID)
        .show()
        .map_err(|error| error.to_string())
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

/// What `delegate::install` may do given the process it finds itself in.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq)]
enum DelegateInstall {
    Run,
    SkipUnbundled,
    SkipOffMainThread,
}

/// Bundle identity is checked before the thread, because touching
/// `UNUserNotificationCenter` at all from an unbundled process aborts it — and
/// the abort happens on the main thread, where the delegate would be installed.
#[cfg(target_os = "macos")]
fn delegate_install_decision(has_bundle_identity: bool, on_main_thread: bool) -> DelegateInstall {
    match (has_bundle_identity, on_main_thread) {
        (false, _) => DelegateInstall::SkipUnbundled,
        (true, false) => DelegateInstall::SkipOffMainThread,
        (true, true) => DelegateInstall::Run,
    }
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
        Some(NotificationAction::Gallery {
            filename: Some(filename),
        }) => format!(
            "mold-gallery:{}:{}",
            uuid::Uuid::new_v4(),
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(filename.as_bytes())
        ),
        Some(NotificationAction::Gallery { filename: None }) => {
            format!("mold-library:{}", uuid::Uuid::new_v4())
        }
        Some(NotificationAction::Create) => format!("mold-create:{}", uuid::Uuid::new_v4()),
        Some(NotificationAction::Models) => format!("mold-models:{}", uuid::Uuid::new_v4()),
        Some(NotificationAction::Updates) => format!("mold-updates:{}", uuid::Uuid::new_v4()),
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

    for (prefix, action) in [
        (
            "mold-library",
            NotificationAction::Gallery { filename: None },
        ),
        ("mold-create", NotificationAction::Create),
        ("mold-models", NotificationAction::Models),
        ("mold-updates", NotificationAction::Updates),
    ] {
        if identifier
            .strip_prefix(prefix)
            .is_some_and(|suffix| suffix.starts_with(':') && suffix.len() > 1)
        {
            return Some(action);
        }
    }

    let mut parts = identifier.splitn(3, ':');
    if parts.next()? != "mold-gallery" || parts.next()?.is_empty() {
        return None;
    }
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(parts.next()?)
        .ok()?;
    let filename = String::from_utf8(bytes).ok()?;
    (!filename.is_empty()).then_some(NotificationAction::Gallery {
        filename: Some(filename),
    })
}

#[cfg(target_os = "macos")]
mod delegate {
    use super::{
        action_from_identifier, delegate_install_decision, macos_bundle_identifier,
        DelegateInstall, NotificationAction,
    };
    use block2::DynBlock;
    use objc2::runtime::ProtocolObject;
    use objc2::{define_class, msg_send, MainThreadMarker, MainThreadOnly};
    use objc2_foundation::{NSObject, NSObjectProtocol};
    use objc2_user_notifications::{
        UNNotificationResponse, UNUserNotificationCenter, UNUserNotificationCenterDelegate,
    };

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
        let Some(app) = APP.get() else { return };
        super::activate(app, action);
    }

    pub fn install(app: &tauri::AppHandle) {
        let mtm = MainThreadMarker::new();
        match delegate_install_decision(macos_bundle_identifier().is_some(), mtm.is_some()) {
            DelegateInstall::SkipUnbundled => {
                tracing::warn!(
                    "skipping notification delegate: not running from an app bundle (dev mode)"
                );
                return;
            }
            DelegateInstall::SkipOffMainThread => {
                tracing::warn!("notification delegate must be installed on the main thread");
                return;
            }
            DelegateInstall::Run => {}
        }
        let Some(mtm) = mtm else { return };
        let _ = APP.set(app.clone());
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
        action_from_identifier, delegate_install_decision, macos_bundle_identifier,
        wait_for_callback, DelegateInstall, NotificationAction,
    };
    use base64::Engine;

    /// Regression: installing the delegate touches
    /// `UNUserNotificationCenter.currentNotificationCenter`, which aborts the
    /// process outside an .app bundle. Being on the main thread is exactly the
    /// case `tauri dev` hits, so the bundle check must be decided first.
    #[test]
    fn delegate_install_is_skipped_without_bundle_identity() {
        assert_eq!(
            delegate_install_decision(false, true),
            DelegateInstall::SkipUnbundled
        );
        assert_eq!(
            delegate_install_decision(false, false),
            DelegateInstall::SkipUnbundled
        );
        assert_eq!(
            delegate_install_decision(true, false),
            DelegateInstall::SkipOffMainThread
        );
        assert_eq!(delegate_install_decision(true, true), DelegateInstall::Run);
    }

    /// The bare test binary stands in for `tauri dev`: the real environment must
    /// resolve to a skip, not merely the synthetic `false` above.
    #[test]
    fn bare_binary_never_installs_the_delegate() {
        assert_eq!(
            delegate_install_decision(macos_bundle_identifier().is_some(), true),
            DelegateInstall::SkipUnbundled
        );
    }

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
                filename: Some(filename.into())
            })
        );
        assert_eq!(action_from_identifier("ordinary-request-id"), None);
    }

    #[test]
    fn workspace_actions_decode_only_from_mold_identifiers() {
        assert_eq!(
            action_from_identifier("mold-library:request-id"),
            Some(NotificationAction::Gallery { filename: None })
        );
        assert_eq!(
            action_from_identifier("mold-create:request-id"),
            Some(NotificationAction::Create)
        );
        assert_eq!(
            action_from_identifier("mold-models:request-id"),
            Some(NotificationAction::Models)
        );
        assert_eq!(
            action_from_identifier("mold-updates:request-id"),
            Some(NotificationAction::Updates)
        );
        assert_eq!(action_from_identifier("mold-create"), None);
        assert_eq!(action_from_identifier("mold-models:"), None);
    }
}
