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

#[tauri::command]
pub fn copy_image_to_clipboard(
    window: tauri::WebviewWindow,
    data_b64: String,
) -> Result<(), String> {
    let bytes = decode_image(&data_b64)?;
    window
        .with_webview(move |_webview| {
            #[cfg(target_os = "ios")]
            {
                use objc2_foundation::NSData;
                use objc2_ui_kit::{UIImage, UIPasteboard};

                // SAFETY: NSData copies this valid Rust slice before the
                // closure returns; UIKit work runs on Tauri's main thread.
                let data =
                    unsafe { NSData::dataWithBytes_length(bytes.as_ptr().cast(), bytes.len()) };
                if let Some(image) = UIImage::imageWithData(&data) {
                    // SAFETY: UIPasteboard is main-thread confined here and
                    // retains/copies the UIImage supplied by the live WebView.
                    unsafe { UIPasteboard::generalPasteboard().setImage(Some(&image)) };
                }
            }

            #[cfg(not(target_os = "ios"))]
            let _ = (_webview, bytes);
        })
        .map_err(|error| format!("could not copy image: {error}"))
}

#[tauri::command]
pub fn save_image_to_photos(window: tauri::WebviewWindow, data_b64: String) -> Result<(), String> {
    let bytes = decode_image(&data_b64)?;
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

                // SAFETY: NSData copies the bytes and UIImage validates them.
                // Tauri invokes the callback on the UIKit main thread.
                let data =
                    unsafe { NSData::dataWithBytes_length(bytes.as_ptr().cast(), bytes.len()) };
                if let Some(image) = UIImage::imageWithData(&data) {
                    unsafe {
                        UIImageWriteToSavedPhotosAlbum(
                            std::ptr::from_ref::<UIImage>(&image).cast_mut().cast(),
                            core::ptr::null_mut(),
                            core::ptr::null_mut(),
                            core::ptr::null_mut(),
                        )
                    };
                }
            }

            #[cfg(not(target_os = "ios"))]
            let _ = (_webview, bytes);
        })
        .map_err(|error| format!("could not save image to Photos: {error}"))
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
}
