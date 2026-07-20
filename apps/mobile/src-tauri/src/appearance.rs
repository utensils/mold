use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MobileAppearance {
    System,
    Light,
    Dark,
}

#[cfg(any(test, target_os = "ios"))]
impl MobileAppearance {
    const fn native_style(self) -> NativeUserInterfaceStyle {
        match self {
            Self::System => NativeUserInterfaceStyle::Unspecified,
            Self::Light => NativeUserInterfaceStyle::Light,
            Self::Dark => NativeUserInterfaceStyle::Dark,
        }
    }
}

#[cfg(any(test, target_os = "ios"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NativeUserInterfaceStyle {
    Unspecified,
    Light,
    Dark,
}

/// Keep UIKit's appearance in step with Mold's WebView appearance. UIKit's
/// default status-bar style follows this trait, so the glyphs remain readable
/// over both light and dark safe-area chrome.
#[tauri::command]
pub fn set_mobile_appearance(
    window: tauri::WebviewWindow,
    appearance: MobileAppearance,
) -> Result<(), String> {
    window
        .with_webview(move |webview| {
            #[cfg(target_os = "ios")]
            {
                use objc2_ui_kit::{UIUserInterfaceStyle, UIViewController};

                let style = match appearance.native_style() {
                    NativeUserInterfaceStyle::Unspecified => UIUserInterfaceStyle::Unspecified,
                    NativeUserInterfaceStyle::Light => UIUserInterfaceStyle::Light,
                    NativeUserInterfaceStyle::Dark => UIUserInterfaceStyle::Dark,
                };

                // SAFETY: Tauri documents `view_controller()` as the live
                // UIViewController for this WebView. `with_webview` executes
                // this callback on the main thread, as UIKit requires.
                let controller = unsafe {
                    webview
                        .view_controller()
                        .cast::<UIViewController>()
                        .as_ref()
                };
                if let Some(controller) = controller {
                    controller.setOverrideUserInterfaceStyle(style);
                    controller.setNeedsStatusBarAppearanceUpdate();
                }
            }

            #[cfg(not(target_os = "ios"))]
            let _ = (webview, appearance);
        })
        .map_err(|error| format!("failed to update mobile appearance: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_each_web_appearance_to_the_matching_native_trait() {
        assert_eq!(
            MobileAppearance::System.native_style(),
            NativeUserInterfaceStyle::Unspecified
        );
        assert_eq!(
            MobileAppearance::Light.native_style(),
            NativeUserInterfaceStyle::Light
        );
        assert_eq!(
            MobileAppearance::Dark.native_style(),
            NativeUserInterfaceStyle::Dark
        );
    }
}
