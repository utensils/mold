/// Send through Apple's current UserNotifications framework so Notification
/// Center associates the alert with the running signed Mold bundle and uses
/// its CFBundleIconFile. The previous NSUserNotification/private-identity-image
/// path is ignored by current macOS releases and produced a generic icon.
#[tauri::command]
pub async fn send_native_notification(
    app: tauri::AppHandle,
    title: String,
    body: Option<String>,
) -> Result<bool, String> {
    #[cfg(target_os = "macos")]
    {
        let _ = app;
        tauri::async_runtime::spawn_blocking(move || {
            send_macos_notification(&title, body.as_deref())
        })
        .await
        .map_err(|error| error.to_string())??;
        Ok(true)
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (app, title, body);
        Ok(false)
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
fn send_macos_notification(title: &str, body: Option<&str>) -> Result<(), String> {
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
    let identifier = NSString::from_str(&uuid::Uuid::new_v4().to_string());
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

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::wait_for_callback;

    #[test]
    fn callback_wait_returns_the_delivered_value() {
        let (sender, receiver) = std::sync::mpsc::channel();
        sender.send(true).unwrap();
        assert!(wait_for_callback(receiver, "testing").unwrap());
    }
}
