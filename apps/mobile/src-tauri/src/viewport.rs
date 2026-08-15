/// Native pickers and the software keyboard can leave WKWebView using their
/// temporary presentation frame. Reapplying the controller's bounds forces
/// WebKit to publish the real viewport before the web shell lays itself out.
#[tauri::command]
pub fn restore_mobile_viewport(window: tauri::WebviewWindow) -> Result<(), String> {
    window
        .with_webview(|webview| {
            #[cfg(target_os = "ios")]
            {
                use objc2_ui_kit::{UIView, UIViewAutoresizing, UIViewController};

                // SAFETY: Tauri supplies live UIKit objects and runs this
                // callback on the main thread. Both pointers are borrowed only
                // for the duration of the callback.
                unsafe {
                    let view = webview.inner().cast::<UIView>().as_ref();
                    let controller = webview
                        .view_controller()
                        .cast::<UIViewController>()
                        .as_ref();
                    if let (Some(view), Some(root)) =
                        (view, controller.and_then(UIViewController::view))
                    {
                        root.layoutIfNeeded();
                        view.setAutoresizingMask(
                            UIViewAutoresizing::FlexibleWidth | UIViewAutoresizing::FlexibleHeight,
                        );
                        view.setFrame(root.bounds());
                        view.setNeedsLayout();
                        view.layoutIfNeeded();
                    }
                }
            }

            #[cfg(not(target_os = "ios"))]
            let _ = webview;
        })
        .map_err(|error| format!("failed to restore mobile viewport: {error}"))
}
