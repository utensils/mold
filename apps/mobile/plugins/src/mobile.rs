use serde::de::DeserializeOwned;
use tauri::{
    plugin::{PluginApi, PluginHandle},
    AppHandle, Runtime,
};

use crate::models::*;

pub fn init<R: Runtime, C: DeserializeOwned>(
    _app: &AppHandle<R>,
    api: PluginApi<R, C>,
) -> crate::Result<MoldMobileNative<R>> {
    let handle =
        api.register_android_plugin("com.utensils.mold.mobile_native", "MoldMobileNativePlugin")?;
    Ok(MoldMobileNative(handle))
}

/// Access to the mold-mobile-native APIs.
pub struct MoldMobileNative<R: Runtime>(PluginHandle<R>);

impl<R: Runtime> MoldMobileNative<R> {
    pub fn set_api_key(&self, host_id: String, api_key: String) -> crate::Result<()> {
        self.0
            .run_mobile_plugin("setApiKey", SetApiKeyRequest { host_id, api_key })
            .map_err(Into::into)
    }

    pub fn get_api_key(&self, host_id: String) -> crate::Result<Option<String>> {
        self.0
            .run_mobile_plugin::<GetApiKeyResponse>("getApiKey", HostKeyRequest { host_id })
            .map(|response| response.api_key)
            .map_err(Into::into)
    }

    pub fn delete_api_key(&self, host_id: String) -> crate::Result<()> {
        self.0
            .run_mobile_plugin("deleteApiKey", HostKeyRequest { host_id })
            .map_err(Into::into)
    }

    pub async fn discover_mold_hosts(&self, timeout_ms: u32) -> crate::Result<Vec<DiscoveredHost>> {
        self.0
            .run_mobile_plugin_async::<DiscoveryResponse>(
                "discoverMoldHosts",
                DiscoveryRequest { timeout_ms },
            )
            .await
            .map(|response| response.hosts)
            .map_err(Into::into)
    }

    pub async fn copy_image_to_clipboard(&self, data_b64: String) -> crate::Result<()> {
        self.0
            .run_mobile_plugin_async("copyImageToClipboard", ImageDataRequest { data_b64 })
            .await
            .map_err(Into::into)
    }

    pub async fn save_image_to_photos(&self, data_b64: String) -> crate::Result<()> {
        self.0
            .run_mobile_plugin_async("saveImageToPhotos", ImageDataRequest { data_b64 })
            .await
            .map_err(Into::into)
    }

    pub async fn save_video_to_photos(&self, url: String) -> crate::Result<()> {
        self.0
            .run_mobile_plugin_async("saveVideoToPhotos", VideoUrlRequest { url })
            .await
            .map_err(Into::into)
    }

    pub async fn share_exported_animation(
        &self,
        request: ShareAnimationRequest,
    ) -> crate::Result<String> {
        self.0
            .run_mobile_plugin_async::<ShareAnimationResponse>("shareExportedAnimation", request)
            .await
            .map(|response| response.outcome)
            .map_err(Into::into)
    }

    pub fn set_mobile_appearance(&self, appearance: String) -> crate::Result<()> {
        self.0
            .run_mobile_plugin("setMobileAppearance", AppearanceRequest { appearance })
            .map_err(Into::into)
    }
}
