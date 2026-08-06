use base64::Engine;

fn validate_video_url(url: &str) -> Result<(), String> {
    if url.starts_with("https://") || url.starts_with("http://") {
        Ok(())
    } else {
        Err("the selected video does not have a valid host URL".to_string())
    }
}

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

#[cfg(target_os = "ios")]
fn save_downloaded_video(staged: &std::path::Path) -> Result<(), String> {
    use block2::{DynBlock, RcBlock};
    use objc2_foundation::{NSString, NSURL};
    use objc2_photos::{PHAssetChangeRequest, PHPhotoLibrary};
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    let file_url = NSURL::fileURLWithPath(&NSString::from_str(&staged.to_string_lossy()));
    let created = Arc::new(AtomicBool::new(false));
    let created_in_block = Arc::clone(&created);
    let changes = RcBlock::new(move || {
        // SAFETY: PhotoKit reads the retained file URL while executing its
        // change transaction and copies the asset before the call returns.
        let request =
            unsafe { PHAssetChangeRequest::creationRequestForAssetFromVideoAtFileURL(&file_url) };
        created_in_block.store(request.is_some(), Ordering::Release);
    });
    let change_ptr = std::ptr::from_ref::<DynBlock<dyn Fn()>>(&changes).cast_mut();
    // SAFETY: The heap-copied block remains alive for this synchronous PhotoKit
    // transaction, and the staged file is retained until it completes.
    let save_result =
        unsafe { PHPhotoLibrary::sharedPhotoLibrary().performChangesAndWait_error(change_ptr) };
    save_result.map_err(|error| error.localizedDescription().to_string())?;
    if created.load(Ordering::Acquire) {
        Ok(())
    } else {
        Err("Photos could not create a video asset from this file".to_string())
    }
}

#[cfg(target_os = "ios")]
async fn download_and_save_video(url: String) -> Result<(), String> {
    use std::io::Write;

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_nanos();
    let staged =
        std::env::temp_dir().join(format!("mold-video-{}-{unique}.mp4", std::process::id()));
    let download_result = async {
        // reqwest deliberately leaves provider selection to the app so the
        // iOS shell can use the smaller, well-supported ring backend.
        let _ = rustls::crypto::ring::default_provider().install_default();
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()
            .map_err(|error| format!("could not prepare the video download: {error}"))?;
        let mut response = client
            .get(&url)
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(|error| format!("could not download the video: {error}"))?;
        let mut file = std::fs::File::create(&staged)
            .map_err(|error| format!("could not stage the video: {error}"))?;
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| format!("could not download the video: {error}"))?
        {
            file.write_all(&chunk)
                .map_err(|error| format!("could not stage the video: {error}"))?;
        }
        file.sync_all()
            .map_err(|error| format!("could not finish staging the video: {error}"))
    }
    .await;
    if let Err(error) = download_result {
        let _ = std::fs::remove_file(&staged);
        return Err(error);
    }

    tauri::async_runtime::spawn_blocking(move || {
        let result = save_downloaded_video(&staged);
        let _ = std::fs::remove_file(&staged);
        result
    })
    .await
    .map_err(|error| error.to_string())?
}

#[tauri::command]
pub async fn save_video_to_photos(url: String) -> Result<(), String> {
    validate_video_url(&url)?;
    #[cfg(target_os = "ios")]
    return download_and_save_video(url).await;

    #[cfg(not(target_os = "ios"))]
    Ok(())
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

    #[test]
    fn video_downloads_accept_only_http_urls() {
        assert!(super::validate_video_url("https://studio.example/media.mp4").is_ok());
        assert!(super::validate_video_url("http://studio.local:7680/media.mp4").is_ok());
        assert!(super::validate_video_url("file:///private/video.mp4").is_err());
        assert!(super::validate_video_url("blob:generated-video").is_err());
        assert!(super::validate_video_url("javascript:alert(1)").is_err());
    }
}
