/// Native pickers and the software keyboard can leave WKWebView using their
/// temporary presentation frame or retaining the scroll offset WebKit used to
/// reveal an editor above the keyboard. Restore both: fixing only the frame
/// leaves the whole shell shifted upward with an equal dead region below it.
#[tauri::command]
pub fn restore_mobile_viewport(window: tauri::WebviewWindow) -> Result<(), String> {
    window
        .with_webview(|webview| {
            #[cfg(target_os = "ios")]
            {
                use objc2_ui_kit::{UIScrollView, UIView, UIViewAutoresizing, UIViewController};

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
                        let bounds = root.bounds();
                        view.setAutoresizingMask(
                            UIViewAutoresizing::FlexibleWidth | UIViewAutoresizing::FlexibleHeight,
                        );
                        view.setFrame(bounds);

                        // WKWebView owns a UIScrollView. Keyboard avoidance can
                        // leave that view scrolled even though Mold's document
                        // itself is intentionally non-scrolling. Ask through
                        // Objective-C so this stays scoped to the iOS-only path.
                        let scroll_view: *const UIScrollView = objc2::msg_send![view, scrollView];
                        if let Some(scroll_view) = scroll_view.as_ref() {
                            scroll_view.setContentOffset(bounds.origin);
                        }

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
