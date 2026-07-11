use tauri::{image::Image, AppHandle};
use tauri_plugin_clipboard_manager::ClipboardExt;

fn decode_clipboard_image(bytes: &[u8]) -> Result<Image<'static>, String> {
    let decoded = image::load_from_memory(bytes)
        .map_err(|error| format!("Could not decode image for clipboard: {error}"))?
        .into_rgba8();
    let (width, height) = decoded.dimensions();
    Ok(Image::new_owned(decoded.into_raw(), width, height))
}

#[tauri::command]
pub async fn clipboard_write_image(app: AppHandle, bytes: Vec<u8>) -> Result<(), String> {
    let image = tauri::async_runtime::spawn_blocking(move || decode_clipboard_image(&bytes))
        .await
        .map_err(|error| format!("Clipboard image decoder stopped unexpectedly: {error}"))??;
    app.clipboard()
        .write_image(&image)
        .map_err(|error| format!("Could not write image to clipboard: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
    use std::io::Cursor;

    fn encoded(format: ImageFormat) -> Vec<u8> {
        let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 3, Rgba([10, 20, 30, 255])));
        let mut bytes = Cursor::new(Vec::new());
        image.write_to(&mut bytes, format).unwrap();
        bytes.into_inner()
    }

    #[test]
    fn decodes_png_and_jpeg_into_clipboard_rgba() {
        for format in [ImageFormat::Png, ImageFormat::Jpeg] {
            let image = decode_clipboard_image(&encoded(format)).unwrap();
            assert_eq!((image.width(), image.height()), (2, 3));
            assert_eq!(image.rgba().len(), 2 * 3 * 4);
        }
    }

    #[test]
    fn rejects_invalid_encoded_image_bytes() {
        assert!(decode_clipboard_image(b"not an image").is_err());
    }
}
