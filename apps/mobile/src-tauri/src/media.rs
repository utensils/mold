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

/// A container the phone may hand to the native share sheet.
///
/// This is a share ALLOWLIST, not a media-type registry: every entry is
/// something `POST /api/gallery/export/:filename` actually returns — the
/// turntable animations the export sheet collects options for, and the
/// geometry transcodes of a stored GLB. A container the phone never exports
/// this way (mp4, say) stays out, so the list must not be reused as a general
/// media check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExportedMediaKind {
    /// GIF turntable or clip animation.
    Gif,
    /// APNG animation, which carries the ordinary PNG signature.
    Apng,
    /// WebP animation, on hosts that advertise it.
    Webp,
    /// Binary glTF — the only container mold STORES for a mesh.
    Glb,
    /// Wavefront OBJ geometry transcode.
    Obj,
    /// Stereolithography geometry transcode, binary or ASCII.
    Stl,
    /// Polygon File Format geometry transcode.
    Ply,
}

/// How many leading bytes the share path keeps before it decides a download
/// really is what its filename claims. Binary STL states its facet count at
/// offset 80 and has no signature at all, so the probe has to reach the whole
/// 84-byte header.
#[cfg(any(target_os = "ios", test))]
const EXPORT_HEADER_PROBE_BYTES: usize = 84;

/// The statements a Wavefront OBJ may legally open with. `v `/`vn `/`vt `/`f `
/// keep their separator so a bare `v` cannot pass for geometry.
#[cfg(any(target_os = "ios", test))]
const OBJ_LINE_PREFIXES: [&[u8]; 8] = [b"#", b"v ", b"vn ", b"vt ", b"f ", b"o ", b"g ", b"mtllib"];

impl ExportedMediaKind {
    /// The media type an Android share intent advertises for this container.
    /// iOS derives its own UTType from the staged file's extension, so only
    /// the Android path reads this.
    #[cfg_attr(not(target_os = "android"), allow(dead_code))]
    fn mime(self) -> &'static str {
        match self {
            Self::Gif => "image/gif",
            Self::Apng => "image/png",
            Self::Webp => "image/webp",
            Self::Glb => "model/gltf-binary",
            Self::Obj => "model/obj",
            Self::Stl => "model/stl",
            Self::Ply => "application/x-ply",
        }
    }

    /// Whether an export's bytes match the container its filename claims.
    ///
    /// `head` is the download's leading bytes (up to
    /// [`EXPORT_HEADER_PROBE_BYTES`]) and `total_bytes` its full length: a
    /// binary STL is well-formed only when the facet count in its header
    /// accounts for exactly the rest of the file.
    #[cfg(any(target_os = "ios", test))]
    fn matches(self, head: &[u8], total_bytes: u64) -> bool {
        match self {
            Self::Gif => head.starts_with(b"GIF87a") || head.starts_with(b"GIF89a"),
            // APNG uses the ordinary PNG signature. WebP remains available on
            // hosts that advertise it even though the current default is
            // GIF/APNG.
            Self::Apng => head.starts_with(b"\x89PNG\r\n\x1a\n"),
            Self::Webp => {
                head.starts_with(b"RIFF") && head.get(8..12).is_some_and(|tag| tag == b"WEBP")
            }
            Self::Glb => head.starts_with(b"glTF"),
            Self::Obj => first_content_line(head).is_some_and(|line| {
                OBJ_LINE_PREFIXES
                    .iter()
                    .any(|prefix| line.starts_with(prefix))
            }),
            Self::Stl => head.starts_with(b"solid") || binary_stl_covers(head, total_bytes),
            Self::Ply => head.starts_with(b"ply\n") || head.starts_with(b"ply\r\n"),
        }
    }
}

/// The first line of a text export that carries content, trimmed of ASCII
/// whitespace. A probe that stops mid-line still answers, because every
/// prefix that identifies an OBJ is far shorter than the probe.
#[cfg(any(target_os = "ios", test))]
fn first_content_line(head: &[u8]) -> Option<&[u8]> {
    head.split(|byte| *byte == b'\n')
        .map(|line| {
            let start = line
                .iter()
                .position(|byte| !byte.is_ascii_whitespace())
                .unwrap_or(line.len());
            let end = line
                .iter()
                .rposition(|byte| !byte.is_ascii_whitespace())
                .map_or(start, |index| index + 1);
            &line[start..end]
        })
        .find(|line| !line.is_empty())
}

/// Binary STL is an 80-byte comment, a little-endian facet count, then 50
/// bytes per facet — and no magic bytes of its own, so the length IS the
/// check.
#[cfg(any(target_os = "ios", test))]
fn binary_stl_covers(head: &[u8], total_bytes: u64) -> bool {
    head.get(80..84).is_some_and(|count| {
        let facets = u32::from_le_bytes([count[0], count[1], count[2], count[3]]);
        84 + 50 * u64::from(facets) == total_bytes
    })
}

/// The container a requested export claims, or `None` when it is not one the
/// phone shares natively.
fn exported_media_kind(filename: &str) -> Option<ExportedMediaKind> {
    let extension = std::path::Path::new(filename)
        .extension()
        .and_then(|extension| extension.to_str())?
        .to_ascii_lowercase();
    match extension.as_str() {
        "gif" => Some(ExportedMediaKind::Gif),
        "png" => Some(ExportedMediaKind::Apng),
        "webp" => Some(ExportedMediaKind::Webp),
        "glb" => Some(ExportedMediaKind::Glb),
        "obj" => Some(ExportedMediaKind::Obj),
        "stl" => Some(ExportedMediaKind::Stl),
        "ply" => Some(ExportedMediaKind::Ply),
        _ => None,
    }
}

/// The bare filename an export may be staged under, with the container it
/// claims. A path is never accepted as one.
fn sanitize_export_filename(filename: &str) -> Result<(String, ExportedMediaKind), String> {
    let filename = std::path::Path::new(filename)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| "the export does not have a valid filename".to_string())?;
    let kind = exported_media_kind(filename)
        .ok_or_else(|| "the exported format is not one this phone can share".to_string())?;
    Ok((filename.to_string(), kind))
}

#[cfg(target_os = "ios")]
#[derive(Debug)]
struct CachedMediaExport {
    reuse_key: String,
    path: std::path::PathBuf,
}

#[cfg(target_os = "ios")]
fn media_export_cache() -> &'static std::sync::Mutex<Option<CachedMediaExport>> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<Option<CachedMediaExport>>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(None))
}

#[cfg(target_os = "ios")]
fn take_cached_export(reuse_key: &str) -> Option<std::path::PathBuf> {
    let cached = media_export_cache().lock().ok()?.take()?;
    if cached.reuse_key == reuse_key && cached.path.is_file() {
        Some(cached.path)
    } else {
        let _ = std::fs::remove_file(cached.path);
        None
    }
}

#[cfg(target_os = "ios")]
fn cache_export(reuse_key: String, path: std::path::PathBuf) {
    if let Ok(mut cache) = media_export_cache().lock()
        && let Some(replaced) = cache.replace(CachedMediaExport { reuse_key, path })
    {
        let _ = std::fs::remove_file(replaced.path);
    }
}

#[cfg(any(target_os = "ios", test))]
fn cleanup_media_exports_in(directory: &std::path::Path) {
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
pub(crate) fn cleanup_stale_media_exports() {
    cleanup_media_exports_in(&std::env::temp_dir());
}

/// The on-device folder "Save to Mold folder" writes into. On iOS it is
/// `<Documents>/Mold`, which `UIFileSharingEnabled` and
/// `LSSupportsOpeningDocumentsInPlace` in the app's Info.plist expose as
/// Files ▸ On My iPhone ▸ Mold; on Android the plugin writes the same name
/// under the public Downloads collection.
#[cfg(any(target_os = "ios", test))]
const MOLD_FOLDER_NAME: &str = "Mold";

/// Where a "Save to Mold folder" export landed, as the phone's toast names it.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MoldFolderSave {
    /// The final filename, after collision-safe renaming.
    pub filename: String,
    /// The absolute path (iOS) or `content://` URI (Android) of the file.
    pub location: String,
    /// The place in the words the OS file browser uses:
    /// `Files ▸ Mold ▸ chair.stl` or `Downloads/Mold/chair.stl`.
    pub label: String,
}

impl MoldFolderSave {
    #[cfg(any(target_os = "ios", test))]
    fn for_ios_path(path: &std::path::Path) -> Self {
        let filename = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        Self {
            label: format!("Files ▸ {MOLD_FOLDER_NAME} ▸ {filename}"),
            location: path.to_string_lossy().into_owned(),
            filename,
        }
    }
}

/// The first free name for `filename` inside `folder`: the name itself, then
/// `name (2).ext`, `name (3).ext`, … — the numbering Finder and the Files app
/// use, so a second export of the same print never overwrites the first.
#[cfg(any(target_os = "ios", test))]
fn unique_destination(folder: &std::path::Path, filename: &str) -> std::path::PathBuf {
    let candidate = folder.join(filename);
    if !candidate.exists() {
        return candidate;
    }
    let path = std::path::Path::new(filename);
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(filename);
    let extension = path.extension().and_then(|extension| extension.to_str());
    (2_u32..)
        .map(|number| {
            folder.join(match extension {
                Some(extension) => format!("{stem} ({number}).{extension}"),
                None => format!("{stem} ({number})"),
            })
        })
        .find(|candidate| !candidate.exists())
        .expect("an unbounded counter finds a free name")
}

/// Move a validated staged download into the Mold folder under a
/// collision-safe name, creating the folder on first use. The staged file is
/// left in place on failure so the caller can keep it for a retry.
#[cfg(any(target_os = "ios", test))]
fn place_in_mold_folder(
    staged: &std::path::Path,
    folder: &std::path::Path,
    filename: &str,
) -> Result<std::path::PathBuf, String> {
    std::fs::create_dir_all(folder)
        .map_err(|error| format!("could not create the Mold folder: {error}"))?;
    let destination = unique_destination(folder, filename);
    if std::fs::rename(staged, &destination).is_err() {
        // A staging directory on another volume cannot be renamed across;
        // copy, then drop the original only once the copy is complete.
        std::fs::copy(staged, &destination)
            .map_err(|error| format!("could not write into the Mold folder: {error}"))?;
        let _ = std::fs::remove_file(staged);
    }
    Ok(destination)
}

/// Run one `POST /api/gallery/export/:filename` to a staged temp file and
/// check the bytes against the container the filename claims. Shared by the
/// share sheet and the Mold folder, so both doors validate identically.
#[cfg(target_os = "ios")]
async fn download_export(
    url: &str,
    api_key: Option<String>,
    request: &serde_json::Value,
    filename: &str,
    media_kind: ExportedMediaKind,
) -> Result<std::path::PathBuf, String> {
    use std::io::Write;

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
            .map_err(|error| format!("could not prepare the export: {error}"))?;
        let body = serde_json::to_vec(request)
            .map_err(|error| format!("could not encode the export options: {error}"))?;
        let mut request = client
            .post(url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body);
        if let Some(api_key) = api_key.filter(|key| !key.is_empty()) {
            request = request.header("x-api-key", api_key);
        }
        let mut response = request
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(|error| format!("could not export this print: {error}"))?;
        let mut file = std::fs::File::create(&staged)
            .map_err(|error| format!("could not stage the export: {error}"))?;
        let mut header = Vec::with_capacity(EXPORT_HEADER_PROBE_BYTES);
        let mut written = 0_u64;
        const MAX_EXPORT_BYTES: u64 = 2 * 1024 * 1024 * 1024;
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| format!("could not download the export: {error}"))?
        {
            written = written.saturating_add(chunk.len() as u64);
            if written > MAX_EXPORT_BYTES {
                return Err("the export exceeds the 2 GB iPhone limit".to_string());
            }
            if header.len() < EXPORT_HEADER_PROBE_BYTES {
                let remaining = EXPORT_HEADER_PROBE_BYTES - header.len();
                header.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
            }
            file.write_all(&chunk)
                .map_err(|error| format!("could not stage the export: {error}"))?;
        }
        file.sync_all()
            .map_err(|error| format!("could not finish the export: {error}"))?;
        if !media_kind.matches(&header, written) {
            return Err(
                "the exported file does not match the format its filename claims".to_string(),
            );
        }
        Ok::<(), String>(())
    }
    .await;
    if let Err(error) = download_result {
        let _ = std::fs::remove_file(&staged);
        return Err(error);
    }
    Ok(staged)
}

#[cfg(any(target_os = "ios", test))]
fn require_platform_image<T>(image: Option<T>, action: &str) -> Result<T, String> {
    image.ok_or_else(|| format!("could not decode the image to {action}"))
}

#[cfg(not(target_os = "android"))]
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
    platform_copy_image(window, bytes, data_b64).await
}

#[cfg(target_os = "android")]
async fn platform_copy_image(
    window: tauri::WebviewWindow,
    _bytes: Vec<u8>,
    data_b64: String,
) -> Result<(), String> {
    use tauri::Manager;
    use tauri_plugin_mold_mobile_native::MoldMobileNativeExt;

    window
        .app_handle()
        .mold_mobile_native()
        .copy_image_to_clipboard(data_b64)
        .await
        .map_err(|error| error.to_string())
}

#[cfg(not(target_os = "android"))]
async fn platform_copy_image(
    window: tauri::WebviewWindow,
    bytes: Vec<u8>,
    _data_b64: String,
) -> Result<(), String> {
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
                    "copying images is available in the mobile builds".to_string()
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
    platform_save_image(window, bytes, data_b64).await
}

#[cfg(target_os = "android")]
async fn platform_save_image(
    window: tauri::WebviewWindow,
    _bytes: Vec<u8>,
    data_b64: String,
) -> Result<(), String> {
    use tauri::Manager;
    use tauri_plugin_mold_mobile_native::MoldMobileNativeExt;

    window
        .app_handle()
        .mold_mobile_native()
        .save_image_to_photos(data_b64)
        .await
        .map_err(|error| error.to_string())
}

#[cfg(not(target_os = "android"))]
async fn platform_save_image(
    window: tauri::WebviewWindow,
    bytes: Vec<u8>,
    _data_b64: String,
) -> Result<(), String> {
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
                    "saving images is available in the mobile builds".to_string()
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
pub async fn save_video_to_photos(app: tauri::AppHandle, url: String) -> Result<(), String> {
    validate_video_url(&url)?;
    platform_save_video(app, url).await
}

#[cfg(target_os = "ios")]
async fn platform_save_video(_app: tauri::AppHandle, url: String) -> Result<(), String> {
    download_and_save_video(url).await
}

#[cfg(target_os = "android")]
async fn platform_save_video(app: tauri::AppHandle, url: String) -> Result<(), String> {
    use tauri_plugin_mold_mobile_native::MoldMobileNativeExt;

    app.mold_mobile_native()
        .save_video_to_photos(url)
        .await
        .map_err(|error| error.to_string())
}

#[cfg(not(any(target_os = "ios", target_os = "android")))]
async fn platform_save_video(_app: tauri::AppHandle, _url: String) -> Result<(), String> {
    Err("saving videos is available in the mobile builds".to_string())
}

/// Export a print through `POST /api/gallery/export/:filename` and hand the
/// result to the phone's own share sheet.
///
/// This covers every container the phone exports — a turntable or clip
/// animation and, since the mesh family shipped, the geometry transcodes of a
/// stored GLB. The command keeps its original `share_exported_animation` name
/// because the Android plugin's permission contract is generated from it; the
/// Rust behind it is container-agnostic.
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
    let (filename, media_kind) = sanitize_export_filename(&filename)?;

    #[cfg(target_os = "ios")]
    {
        use block2::{DynBlock, RcBlock};
        use objc2::{MainThreadOnly, runtime::Bool};
        use objc2_foundation::{NSArray, NSError, NSURL};
        use objc2_ui_kit::{UIActivityType, UIActivityViewController, UIViewController};

        let staged = if let Some(cached) = take_cached_export(&reuse_key) {
            cached
        } else {
            download_export(&url, api_key, &request, &filename, media_kind).await?
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
            cache_export(reuse_key, staged);
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
            cache_export(reuse_key, staged);
        }
        result
    }

    #[cfg(not(target_os = "ios"))]
    {
        #[cfg(target_os = "android")]
        {
            use tauri::Manager;
            use tauri_plugin_mold_mobile_native::{MoldMobileNativeExt, ShareExportRequest};

            let request_json = serde_json::to_string(&request)
                .map_err(|error| format!("could not encode the export options: {error}"))?;
            return window
                .app_handle()
                .mold_mobile_native()
                .share_exported_media(ShareExportRequest {
                    url,
                    api_key,
                    request_json,
                    filename,
                    // Android names the type its chooser advertises; the
                    // allowlist above is the single authority on it.
                    mime_type: media_kind.mime().to_string(),
                    reuse_key,
                })
                .await
                .map_err(|error| error.to_string());
        }

        #[cfg(not(target_os = "android"))]
        {
            let _ = (window, api_key, request, filename, media_kind, reuse_key);
            Err("sharing exports is available in the mobile builds".to_string())
        }
    }
}

/// Export a print through `POST /api/gallery/export/:filename` and write the
/// result into the on-device Mold folder instead of a share sheet.
///
/// The other half of the export pair: the same allowlist, the same staged
/// download and byte check as [`share_exported_animation`], and the same
/// `reuse_key`, so a file staged for a share the user backed out of is
/// saved without a second download. iOS writes `<Documents>/Mold/<name>`,
/// which the app's Info.plist exposes in the Files app; Android hands the
/// staged file to the plugin, which files it under `Downloads/Mold` through
/// MediaStore. Both answer with the final name and a label for the toast.
#[tauri::command]
pub async fn save_export_to_mold_folder(
    window: tauri::WebviewWindow,
    url: String,
    api_key: Option<String>,
    request: serde_json::Value,
    filename: String,
    reuse_key: String,
) -> Result<MoldFolderSave, String> {
    validate_video_url(&url)?;
    let (filename, media_kind) = sanitize_export_filename(&filename)?;

    #[cfg(target_os = "ios")]
    {
        use tauri::Manager;

        let staged = if let Some(cached) = take_cached_export(&reuse_key) {
            cached
        } else {
            download_export(&url, api_key, &request, &filename, media_kind).await?
        };
        let folder = window
            .app_handle()
            .path()
            .document_dir()
            .map_err(|error| format!("could not find this iPhone's Documents folder: {error}"))?
            .join(MOLD_FOLDER_NAME);
        let staged_for_move = staged.clone();
        let placed = tauri::async_runtime::spawn_blocking(move || {
            place_in_mold_folder(&staged_for_move, &folder, &filename)
        })
        .await
        .map_err(|error| error.to_string())?;
        match placed {
            Ok(path) => Ok(MoldFolderSave::for_ios_path(&path)),
            Err(error) => {
                // The download itself is fine; keep it for a retry or a share.
                cache_export(reuse_key, staged);
                Err(error)
            }
        }
    }

    #[cfg(not(target_os = "ios"))]
    {
        #[cfg(target_os = "android")]
        {
            use tauri::Manager;
            use tauri_plugin_mold_mobile_native::{MoldMobileNativeExt, ShareExportRequest};

            let request_json = serde_json::to_string(&request)
                .map_err(|error| format!("could not encode the export options: {error}"))?;
            return window
                .app_handle()
                .mold_mobile_native()
                .save_exported_media_to_mold_folder(ShareExportRequest {
                    url,
                    api_key,
                    request_json,
                    filename,
                    mime_type: media_kind.mime().to_string(),
                    reuse_key,
                })
                .await
                .map(|saved| MoldFolderSave {
                    filename: saved.filename,
                    location: saved.location,
                    label: saved.label,
                })
                .map_err(|error| error.to_string());
        }

        #[cfg(not(target_os = "android"))]
        {
            let _ = (window, api_key, request, filename, media_kind, reuse_key);
            Err("the Mold folder is available in the mobile builds".to_string())
        }
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

    /// The share path only ever sees a leading probe of the download, so this
    /// mirrors it: the whole slice is the head, and its length is the export's
    /// length.
    fn validate(bytes: &[u8], filename: &str) -> Result<String, String> {
        let (filename, kind) = super::sanitize_export_filename(filename)?;
        if kind.matches(bytes, bytes.len() as u64) {
            Ok(filename)
        } else {
            Err("format mismatch".to_string())
        }
    }

    /// A binary STL of `triangles` facets: an 80-byte comment header, the
    /// little-endian facet count, then 50 bytes each.
    fn binary_stl(triangles: u32) -> Vec<u8> {
        let mut bytes = vec![0_u8; 80];
        bytes.extend_from_slice(&triangles.to_le_bytes());
        bytes.extend(std::iter::repeat_n(0_u8, 50 * triangles as usize));
        bytes
    }

    #[test]
    fn animation_shares_require_matching_safe_formats() {
        assert!(validate(b"GIF89a rest", "clip.gif").is_ok());
        assert!(validate(b"\x89PNG\r\n\x1a\nrest", "clip.png").is_ok());
        assert!(validate(b"RIFFxxxxWEBPrest", "clip.webp").is_ok());
        assert!(validate(b"GIF89a rest", "clip.png").is_err());
        assert!(validate(b"GIF89a rest", "clip.mp4").is_err());
        assert!(validate(b"GIF89a rest", "../clip.gif").is_ok());
    }

    /// The geometry transcodes `POST /api/gallery/export/:filename` returns
    /// for a stored GLB reach the native share sheet exactly like a turntable
    /// does.
    #[test]
    fn mesh_shares_accept_every_advertised_geometry_container() {
        assert!(validate(b"glTF\x02\x00\x00\x00rest", "armchair.glb").is_ok());
        assert!(validate(b"# Exported by mold\nv 0 0 0\n", "armchair.obj").is_ok());
        assert!(validate(b"\n\nmtllib armchair.mtl\n", "armchair.obj").is_ok());
        assert!(validate(b"vn 0 1 0\n", "armchair.obj").is_ok());
        assert!(validate(b"solid armchair\nfacet normal 0 0 1\n", "armchair.stl").is_ok());
        assert!(validate(&binary_stl(3), "armchair.stl").is_ok());
        assert!(validate(b"ply\nformat binary_little_endian 1.0\n", "armchair.ply").is_ok());
        assert!(validate(b"ply\r\nformat ascii 1.0\r\n", "armchair.ply").is_ok());
        assert!(validate(b"glTF\x02\x00\x00\x00rest", "../meshes/armchair.glb").is_ok());
    }

    /// The allowlist stays tight, and the bytes have to agree with the name:
    /// neither a container the phone never exports nor a mislabelled download
    /// reaches the share sheet.
    #[test]
    fn mesh_shares_reject_unknown_containers_and_mismatched_bytes() {
        assert!(validate(b"# Exported by mold\n", "armchair.txt").is_err());
        assert!(validate(b"glTF\x02\x00\x00\x00", "armchair.mp4").is_err());
        assert!(validate(b"glTF\x02\x00\x00\x00", "armchair").is_err());
        assert!(validate(b"GIF89a rest", "armchair.stl").is_err());
        assert!(validate(b"GIF89a rest", "armchair.ply").is_err());
        assert!(validate(b"GIF89a rest", "armchair.glb").is_err());
        assert!(validate(b"GIF89a rest", "armchair.obj").is_err());
        assert!(validate(b"\x89PNG\r\n\x1a\nrest", "armchair.glb").is_err());
        // A truncated binary STL claims more facets than it carries.
        let mut short = binary_stl(3);
        short.truncate(short.len() - 1);
        assert!(validate(&short, "armchair.stl").is_err());
    }

    #[test]
    fn every_shareable_container_names_its_own_media_type() {
        let mime = |filename: &str| super::sanitize_export_filename(filename).unwrap().1.mime();
        assert_eq!(mime("clip.gif"), "image/gif");
        assert_eq!(mime("clip.png"), "image/png");
        assert_eq!(mime("clip.webp"), "image/webp");
        assert_eq!(mime("armchair.glb"), "model/gltf-binary");
        assert_eq!(mime("armchair.obj"), "model/obj");
        assert_eq!(mime("armchair.stl"), "model/stl");
        assert_eq!(mime("armchair.ply"), "application/x-ply");
    }

    /// Binary STL states its facet count at offset 80 and carries no
    /// signature, so a probe shorter than its 84-byte header could not tell a
    /// real export from an arbitrary download.
    #[test]
    fn the_header_probe_reaches_the_binary_stl_facet_count() {
        assert!(super::EXPORT_HEADER_PROBE_BYTES >= 84);
    }

    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "mold-media-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    /// The Mold folder never overwrites: a second export with the same name
    /// gets ` (2)`, a third ` (3)`, and the extension stays where a Files app
    /// can read the type off it.
    #[test]
    fn a_saved_export_never_overwrites_an_earlier_one() {
        let folder = scratch_dir("collision");
        assert_eq!(
            super::unique_destination(&folder, "chair.stl"),
            folder.join("chair.stl")
        );
        std::fs::write(folder.join("chair.stl"), b"solid chair").unwrap();
        assert_eq!(
            super::unique_destination(&folder, "chair.stl"),
            folder.join("chair (2).stl")
        );
        std::fs::write(folder.join("chair (2).stl"), b"solid chair").unwrap();
        assert_eq!(
            super::unique_destination(&folder, "chair.stl"),
            folder.join("chair (3).stl")
        );
        // A dotted stem keeps every dot but the extension's.
        std::fs::write(folder.join("armchair.v2.glb"), b"glTF").unwrap();
        assert_eq!(
            super::unique_destination(&folder, "armchair.v2.glb"),
            folder.join("armchair.v2 (2).glb")
        );
        std::fs::remove_dir_all(folder).unwrap();
    }

    /// Placing an export creates the Mold folder on first use, moves the
    /// staged download in under a collision-safe name, and answers with the
    /// final path so the toast can name it.
    #[test]
    fn placing_an_export_creates_the_mold_folder_and_moves_the_staged_file() {
        let root = scratch_dir("place");
        let staged = root.join("mold-export-1-chair.obj");
        std::fs::write(&staged, b"# Exported by mold\nv 0 0 0\n").unwrap();
        let folder = root.join("Documents").join(super::MOLD_FOLDER_NAME);

        let placed = super::place_in_mold_folder(&staged, &folder, "chair.obj").unwrap();

        assert_eq!(placed, folder.join("chair.obj"));
        assert_eq!(
            std::fs::read(&placed).unwrap(),
            b"# Exported by mold\nv 0 0 0\n"
        );
        assert!(!staged.exists(), "the staged download is moved, not copied");

        let again = root.join("mold-export-2-chair.obj");
        std::fs::write(&again, b"v 1 1 1\n").unwrap();
        let placed_again = super::place_in_mold_folder(&again, &folder, "chair.obj").unwrap();
        assert_eq!(placed_again, folder.join("chair (2).obj"));
        assert_eq!(
            std::fs::read(&placed).unwrap(),
            b"# Exported by mold\nv 0 0 0\n"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    /// The toast names where the file went in the words the Files app uses.
    #[test]
    fn the_ios_save_label_names_the_files_app_path() {
        let folder = scratch_dir("label");
        let save = super::MoldFolderSave::for_ios_path(&folder.join("chair (2).stl"));
        assert_eq!(save.filename, "chair (2).stl");
        assert_eq!(save.label, "Files ▸ Mold ▸ chair (2).stl");
        assert_eq!(
            save.location,
            folder.join("chair (2).stl").to_string_lossy()
        );
        std::fs::remove_dir_all(folder).unwrap();
    }

    /// The Mold folder takes exactly what the share sheet takes: the same
    /// allowlist gates both doors, so a container the phone never exports
    /// cannot be written there either.
    #[test]
    fn the_mold_folder_accepts_only_shareable_containers() {
        for filename in [
            "chair.glb",
            "chair.obj",
            "chair.stl",
            "chair.ply",
            "turn.gif",
            "turn.png",
            "turn.webp",
        ] {
            assert!(
                super::sanitize_export_filename(filename).is_ok(),
                "{filename}"
            );
        }
        assert!(super::sanitize_export_filename("chair.mp4").is_err());
        assert!(super::sanitize_export_filename("chair.txt").is_err());
        assert!(super::sanitize_export_filename("../chair.stl").is_ok());
        assert_eq!(
            super::sanitize_export_filename("../chair.stl").unwrap().0,
            "chair.stl"
        );
    }

    #[test]
    fn startup_cleanup_removes_only_staged_media_exports() {
        let directory =
            std::env::temp_dir().join(format!("mold-media-cleanup-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let staged = directory.join("mold-export-123-clip.gif");
        let unrelated = directory.join("keep-me.gif");
        std::fs::write(&staged, b"GIF89a").unwrap();
        std::fs::write(&unrelated, b"GIF89a").unwrap();

        super::cleanup_media_exports_in(&directory);

        assert!(!staged.exists());
        assert!(unrelated.exists());
        std::fs::remove_dir_all(directory).unwrap();
    }
}
