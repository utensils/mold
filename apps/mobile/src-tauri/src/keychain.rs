const SERVICE: &str = "com.utensils.mold.remote-api-key";

#[tauri::command]
pub fn keychain_set_api_key(host_id: String, api_key: String) -> Result<(), String> {
    security_framework::passwords::set_generic_password(SERVICE, &host_id, api_key.as_bytes())
        .map_err(|error| format!("could not save API key: {error}"))
}

#[tauri::command]
pub fn keychain_get_api_key(host_id: String) -> Result<Option<String>, String> {
    match security_framework::passwords::get_generic_password(SERVICE, &host_id) {
        Ok(bytes) => String::from_utf8(bytes)
            .map(Some)
            .map_err(|_| "saved API key is not valid UTF-8".to_string()),
        Err(_) => Ok(None),
    }
}

#[tauri::command]
pub fn keychain_delete_api_key(host_id: String) -> Result<(), String> {
    // Deletion is idempotent for the UI: a missing item already satisfies the
    // requested end state.
    let _ = security_framework::passwords::delete_generic_password(SERVICE, &host_id);
    Ok(())
}
