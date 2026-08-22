#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PickedIdentityPhoto {
    pub cancelled: bool,
    pub filename: Option<String>,
    pub mime_type: Option<String>,
    pub size_bytes: Option<u64>,
    pub data_b64: Option<String>,
}

#[tauri::command]
pub async fn pick_identity_photo(
    app: tauri::AppHandle,
    source: String,
) -> Result<PickedIdentityPhoto, String> {
    #[cfg(target_os = "android")]
    {
        use tauri_plugin_mold_mobile_native::MoldMobileNativeExt;
        let response = app
            .mold_mobile_native()
            .pick_identity_photo(source)
            .await
            .map_err(|error| error.to_string())?;
        return Ok(PickedIdentityPhoto {
            cancelled: response.cancelled,
            filename: response.filename,
            mime_type: response.mime_type,
            size_bytes: response.size_bytes,
            data_b64: response.data_b64,
        });
    }

    #[cfg(not(target_os = "android"))]
    {
        let _ = (app, source);
        Err("the native identity photo picker is available only on Android".to_owned())
    }
}
