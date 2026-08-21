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

fn sanitize_animation_filename(filename: &str) -> Result<String, String> {
    let filename = std::path::Path::new(filename)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| "the exported animation does not have a valid filename".to_string())?;
    let extension = std::path::Path::new(filename)
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !matches!(extension.as_str(), "gif" | "png" | "webp") {
        return Err("the exported animation format does not match its filename".to_string());
    }
    Ok(filename.to_string())
}

#[cfg(any(target_os = "ios", test))]
fn validate_animation(bytes: &[u8], filename: &str) -> Result<String, String> {
    let filename = sanitize_animation_filename(filename)?;
    let extension = std::path::Path::new(&filename)
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let valid = match extension.as_str() {
        "gif" => bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a"),
        // APNG uses the ordinary PNG signature. WebP remains available on
        // hosts that advertise it even though the current default is GIF/APNG.
        "png" => bytes.starts_with(b"\x89PNG\r\n\x1a\n"),
        "webp" => bytes.starts_with(b"RIFF") && bytes.get(8..12).is_some_and(|tag| tag == b"WEBP"),
        _ => false,
    };
    if !valid {
        return Err("the exported animation format does not match its filename".to_string());
    }
    Ok(filename)
}

#[cfg(target_os = "ios")]
#[derive(Debug)]
struct CachedAnimationExport {
    reuse_key: String,
    path: std::path::PathBuf,
}

#[cfg(target_os = "ios")]
fn animation_export_cache() -> &'static std::sync::Mutex<Option<CachedAnimationExport>> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<Option<CachedAnimationExport>>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(None))
}

#[cfg(target_os = "ios")]
fn take_cached_animation(reuse_key: &str) -> Option<std::path::PathBuf> {
    let cached = animation_export_cache().lock().ok()?.take()?;
    if cached.reuse_key == reuse_key && cached.path.is_file() {
        Some(cached.path)
    } else {
        let _ = std::fs::remove_file(cached.path);
        None
    }
}

#[cfg(target_os = "ios")]
fn cache_animation(reuse_key: String, path: std::path::PathBuf) {
    if let Ok(mut cache) = animation_export_cache().lock()
        && let Some(replaced) = cache.replace(CachedAnimationExport { reuse_key, path })
    {
        let _ = std::fs::remove_file(replaced.path);
    }
}

#[cfg(any(target_os = "ios", test))]
fn cleanup_animation_exports_in(directory: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let is_export = entry
            .file_name()
            .to_str()
            .is_some_and(|name| name.starts_with("mold-export-"));
        let is_file = entry.file_type().is_ok_and(|kind| kind.is_file());
        if is_export && is_file {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

#[cfg(target_os = "ios")]
pub(crate) fn cleanup_stale_animation_exports() {
    cleanup_animation_exports_in(&std::env::temp_dir());
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
                let _ = sender.send(Err(
                    "copying images is not implemented on Android yet".to_string(),
                ));
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
                let _ = sender.send(Err(
                    "saving images is not implemented on Android yet".to_string(),
                ));
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
    Err("saving videos is not implemented on Android yet".to_string())
}

#[tauri::command]
pub async fn share_exported_animation(
    window: tauri::WebviewWindow,
    url: String,
    api_key: Option<String>,
    request: serde_json::Value,
    filename: String,
    reuse_key: String,
) -> Result<String, String> {
    validate_video_url(&url)?;
    let filename = sanitize_animation_filename(&filename)?;

    #[cfg(target_os = "ios")]
    {
        use block2::{DynBlock, RcBlock};
        use objc2::{MainThreadOnly, runtime::Bool};
        use objc2_foundation::{NSArray, NSError, NSURL};
        use objc2_ui_kit::{UIActivityType, UIActivityViewController, UIViewController};
        use std::io::Write;

        let staged = if let Some(cached) = take_cached_animation(&reuse_key) {
            cached
        } else {
            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|error| error.to_string())?
                .as_nanos();
            let staged = std::env::temp_dir().join(format!("mold-export-{unique}-{filename}"));
            let download_result = async {
                let _ = rustls::crypto::ring::default_provider().install_default();
                let client = reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(600))
                    .build()
                    .map_err(|error| format!("could not prepare the animation export: {error}"))?;
                let body = serde_json::to_vec(&request)
                    .map_err(|error| format!("could not encode the animation options: {error}"))?;
                let mut request = client
                    .post(&url)
                    .header(reqwest::header::CONTENT_TYPE, "application/json")
                    .body(body);
                if let Some(api_key) = api_key.filter(|key| !key.is_empty()) {
                    request = request.header("x-api-key", api_key);
                }
                let mut response = request
                    .send()
                    .await
                    .and_then(reqwest::Response::error_for_status)
                    .map_err(|error| format!("could not export the animation: {error}"))?;
                let mut file = std::fs::File::create(&staged)
                    .map_err(|error| format!("could not stage the animation export: {error}"))?;
                let mut header = Vec::with_capacity(12);
                let mut written = 0_u64;
                const MAX_EXPORT_BYTES: u64 = 2 * 1024 * 1024 * 1024;
                while let Some(chunk) = response
                    .chunk()
                    .await
                    .map_err(|error| format!("could not download the animation export: {error}"))?
                {
                    written = written.saturating_add(chunk.len() as u64);
                    if written > MAX_EXPORT_BYTES {
                        return Err(
                            "the animation export exceeds the 2 GB iPhone limit".to_string()
                        );
                    }
                    if header.len() < 12 {
                        let remaining = 12 - header.len();
                        header.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
                    }
                    file.write_all(&chunk).map_err(|error| {
                        format!("could not stage the animation export: {error}")
                    })?;
                }
                file.sync_all()
                    .map_err(|error| format!("could not finish the animation export: {error}"))?;
                validate_animation(&header, &filename)?;
                Ok::<(), String>(())
            }
            .await;
            if let Err(error) = download_result {
                let _ = std::fs::remove_file(&staged);
                return Err(error);
            }
            staged
        };
        let staged_for_sheet = staged.clone();
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let setup_sender = sender.clone();
        let schedule = window.with_webview(move |webview| {
            // SAFETY: Tauri supplies the live UIViewController and invokes
            // this callback on UIKit's main thread. The activity controller
            // retains the file URL and copies its completion block.
            let result = (|| {
                // SAFETY: All Objective-C objects below are live for the
                // duration documented above, and UIKit retains what it
                // needs after presentation.
                unsafe {
                    let controller = webview
                        .view_controller()
                        .cast::<UIViewController>()
                        .as_ref()
                        .ok_or_else(|| "could not find the iOS view controller".to_string())?;
                    let main_thread = objc2::MainThreadMarker::new().ok_or_else(|| {
                        "the iOS share sheet must open on the main thread".to_string()
                    })?;
                    if controller.presentedViewController().is_some() {
                        return Err("another iOS sheet is already open".to_string());
                    }
                    let path =
                        objc2_foundation::NSString::from_str(&staged_for_sheet.to_string_lossy());
                    let file_url = NSURL::fileURLWithPath(&path);
                    let typed_items = NSArray::arrayWithObject(&*file_url);
                    let items: &NSArray = typed_items.cast_unchecked();
                    let activity =
                        UIActivityViewController::initWithActivityItems_applicationActivities(
                            UIActivityViewController::alloc(main_thread),
                            items,
                            None,
                        );
                    if let (Some(popover), Some(view)) =
                        (activity.popoverPresentationController(), controller.view())
                    {
                        popover.setSourceView(Some(&view));
                    }
                    let completion = RcBlock::new(
                        move |_activity: *mut UIActivityType,
                              completed: Bool,
                              _returned_items: *mut NSArray,
                              error: *mut NSError| {
                            let outcome = if let Some(error) = error.as_ref() {
                                Err(error.localizedDescription().to_string())
                            } else if completed.as_bool() {
                                Ok("shared".to_string())
                            } else {
                                Ok("cancelled".to_string())
                            };
                            let _ = sender.send(outcome);
                        },
                    );
                    let completion_ptr = std::ptr::from_ref::<
                        DynBlock<dyn Fn(*mut UIActivityType, Bool, *mut NSArray, *mut NSError)>,
                    >(&completion)
                    .cast_mut();
                    activity.setCompletionWithItemsHandler(completion_ptr);
                    controller.presentViewController_animated_completion(&activity, true, None);
                }
                Ok(())
            })();
            if let Err(error) = result {
                let _ = setup_sender.send(Err(error));
            }
        });
        if let Err(error) = schedule {
            cache_animation(reuse_key, staged);
            return Err(format!("could not open the iOS share sheet: {error}"));
        }
        let received = tauri::async_runtime::spawn_blocking(move || {
            receiver
                .recv()
                .map_err(|_| "the iOS share sheet closed unexpectedly".to_string())?
        })
        .await;
        let result = match received {
            Ok(result) => result,
            Err(error) => Err(error.to_string()),
        };
        if result.as_deref() == Ok("shared") {
            let _ = std::fs::remove_file(staged);
        } else {
            cache_animation(reuse_key, staged);
        }
        result
    }

    #[cfg(not(target_os = "ios"))]
    {
        let _ = (window, api_key, request, filename, reuse_key);
        Err("sharing animations is not implemented on Android yet".to_string())
    }
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

    #[test]
    fn animation_shares_require_matching_safe_formats() {
        assert!(super::validate_animation(b"GIF89a rest", "clip.gif").is_ok());
        assert!(super::validate_animation(b"\x89PNG\r\n\x1a\nrest", "clip.png").is_ok());
        assert!(super::validate_animation(b"RIFFxxxxWEBPrest", "clip.webp").is_ok());
        assert!(super::validate_animation(b"GIF89a rest", "clip.png").is_err());
        assert!(super::validate_animation(b"GIF89a rest", "clip.mp4").is_err());
        assert!(super::validate_animation(b"GIF89a rest", "../clip.gif").is_ok());
    }

    #[test]
    fn startup_cleanup_removes_only_staged_animation_exports() {
        let directory =
            std::env::temp_dir().join(format!("mold-media-cleanup-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let staged = directory.join("mold-export-123-clip.gif");
        let unrelated = directory.join("keep-me.gif");
        std::fs::write(&staged, b"GIF89a").unwrap();
        std::fs::write(&unrelated, b"GIF89a").unwrap();

        super::cleanup_animation_exports_in(&directory);

        assert!(!staged.exists());
        assert!(unrelated.exists());
        std::fs::remove_dir_all(directory).unwrap();
    }
}
