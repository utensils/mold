use base64::Engine;

fn decode_image(data_b64: &str) -> Result<Vec<u8>, String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data_b64)
        .map_err(|_| "the selected print contains invalid image data".to_string())?;
    let png = bytes.starts_with(b"\x89PNG\r\n\x1a\n");
    let jpeg = bytes.starts_with(&[0xff, 0xd8, 0xff]);
    if png || jpeg {
        Ok(bytes)
    } else {
        Err("the selected print is not a PNG or JPEG image".to_string())
    }
}

#[cfg(any(target_os = "ios", test))]
fn require_platform_image<T>(image: Option<T>, action: &str) -> Result<T, String> {
    image.ok_or_else(|| format!("could not decode the image to {action}"))
}

async fn wait_for_image_action(
    receiver: std::sync::mpsc::Receiver<Result<(), String>>,
    action: &'static str,
) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .map_err(|_| format!("timed out while trying to {action} the image"))?
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
pub async fn copy_image_to_clipboard(
    window: tauri::WebviewWindow,
    data_b64: String,
) -> Result<(), String> {
    let bytes = decode_image(&data_b64)?;
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    window
        .with_webview(move |_webview| {
            #[cfg(target_os = "ios")]
            {
                use objc2_foundation::NSData;
                use objc2_ui_kit::{UIImage, UIPasteboard};

                let result = (|| {
                    // SAFETY: NSData copies this valid Rust slice before the
                    // closure returns; UIKit work runs on Tauri's main thread.
                    let data =
                        unsafe { NSData::dataWithBytes_length(bytes.as_ptr().cast(), bytes.len()) };
                    let image = require_platform_image(UIImage::imageWithData(&data), "copy")?;
                    // SAFETY: UIPasteboard is main-thread confined here and
                    // retains/copies the UIImage supplied by the live WebView.
                    unsafe { UIPasteboard::generalPasteboard().setImage(Some(&image)) }
                    Ok(())
                })();
                let _ = sender.send(result);
            }

            #[cfg(not(target_os = "ios"))]
            {
                let _ = (_webview, bytes);
                let _ = sender.send(Ok(()));
            }
        })
        .map_err(|error| format!("could not copy image: {error}"))?;
    wait_for_image_action(receiver, "copy").await
}

#[tauri::command]
pub async fn save_image_to_photos(
    window: tauri::WebviewWindow,
    data_b64: String,
) -> Result<(), String> {
    let bytes = decode_image(&data_b64)?;
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    window
        .with_webview(move |_webview| {
            #[cfg(target_os = "ios")]
            {
                use core::ffi::c_void;
                use objc2_foundation::NSData;
                use objc2_ui_kit::UIImage;

                unsafe extern "C" {
                    fn UIImageWriteToSavedPhotosAlbum(
                        image: *mut c_void,
                        completion_target: *mut c_void,
                        completion_selector: *mut c_void,
                        context_info: *mut c_void,
                    );
                }

                let result = (|| {
                    // SAFETY: NSData copies the bytes and UIImage validates
                    // them. Tauri invokes this callback on the UIKit main
                    // thread.
                    let data =
                        unsafe { NSData::dataWithBytes_length(bytes.as_ptr().cast(), bytes.len()) };
                    let image = require_platform_image(UIImage::imageWithData(&data), "save")?;
                    unsafe {
                        UIImageWriteToSavedPhotosAlbum(
                            std::ptr::from_ref::<UIImage>(&image).cast_mut().cast(),
                            core::ptr::null_mut(),
                            core::ptr::null_mut(),
                            core::ptr::null_mut(),
                        )
                    };
                    Ok(())
                })();
                let _ = sender.send(result);
            }

            #[cfg(not(target_os = "ios"))]
            {
                let _ = (_webview, bytes);
                let _ = sender.send(Ok(()));
            }
        })
        .map_err(|error| format!("could not save image to Photos: {error}"))?;
    wait_for_image_action(receiver, "save").await
}

#[cfg(test)]
mod tests {
    use base64::Engine;

    #[test]
    fn accepts_png_and_jpeg_but_rejects_other_media() {
        let encode = |bytes: &[u8]| base64::engine::general_purpose::STANDARD.encode(bytes);
        assert!(super::decode_image(&encode(b"\x89PNG\r\n\x1a\nrest")).is_ok());
        assert!(super::decode_image(&encode(&[0xff, 0xd8, 0xff, 0xe0])).is_ok());
        assert!(super::decode_image(&encode(b"not an image")).is_err());
        assert!(super::decode_image("not base64!").is_err());
    }

    #[test]
    fn failed_platform_decode_is_reported_to_the_caller() {
        assert_eq!(
            super::require_platform_image::<()>(None, "copy").unwrap_err(),
            "could not decode the image to copy"
        );
        assert!(super::require_platform_image(Some(()), "save").is_ok());
    }
}
