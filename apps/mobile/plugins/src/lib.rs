use tauri::{
    plugin::{Builder, TauriPlugin},
    Manager, Runtime,
};

pub use models::*;

mod mobile;

mod error;
mod models;

pub use error::{Error, Result};

use mobile::MoldMobileNative;

/// Extensions to [`tauri::App`], [`tauri::AppHandle`] and [`tauri::Window`] to access the mold-mobile-native APIs.
pub trait MoldMobileNativeExt<R: Runtime> {
    fn mold_mobile_native(&self) -> &MoldMobileNative<R>;
}

impl<R: Runtime, T: Manager<R>> crate::MoldMobileNativeExt<R> for T {
    fn mold_mobile_native(&self) -> &MoldMobileNative<R> {
        self.state::<MoldMobileNative<R>>().inner()
    }
}

/// Initializes the plugin.
pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("mold-mobile-native")
        .setup(|app, api| {
            let mold_mobile_native = mobile::init(app, api)?;
            app.manage(mold_mobile_native);
            Ok(())
        })
        .build()
}
