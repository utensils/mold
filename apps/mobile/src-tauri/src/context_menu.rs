//! Extend WKWebView's already-presented image menu through UIKit's public
//! `UIContextMenuInteraction` API.
//!
//! WebKit does not route image-element menus through the public WKUIDelegate
//! context-menu callback. The frontend therefore tells the shell when its
//! gallery image received `contextmenu`; the shell updates the visible UIKit
//! menu without replacing WebKit's Share, Save, Copy, or system actions.

#[cfg(target_os = "ios")]
use std::{ptr::NonNull, time::Duration};

#[cfg(target_os = "ios")]
use block2::RcBlock;
#[cfg(target_os = "ios")]
use dispatch2::{DispatchQueue, DispatchTime};
#[cfg(target_os = "ios")]
use objc2::{rc::Retained, runtime::AnyObject, sel};
#[cfg(target_os = "ios")]
use objc2_foundation::{NSArray, NSString};
#[cfg(target_os = "ios")]
use objc2_ui_kit::{UIAction, UIImage, UIMenu};

#[cfg(target_os = "ios")]
const SELECT_GALLERY_PRINT_SCRIPT: &str =
    "window.dispatchEvent(new CustomEvent('mold:native-gallery-select'))";
#[cfg(target_os = "ios")]
const UPSCALE_GALLERY_PRINT_SCRIPT: &str =
    "window.dispatchEvent(new CustomEvent('mold:native-gallery-upscale'))";

#[cfg(target_os = "ios")]
fn evaluate(webview: NonNull<AnyObject>, script: &str) {
    let script = NSString::from_str(script);
    // SAFETY: The pointer is the app-owned WKWebView and every caller runs on
    // the main queue while the window remains alive.
    unsafe {
        let _: () = objc2::msg_send![
            webview.as_ref(),
            evaluateJavaScript: &*script,
            completionHandler: core::ptr::null::<core::ffi::c_void>()
        ];
    }
}

#[cfg(target_os = "ios")]
fn append_gallery_actions(
    menu: NonNull<UIMenu>,
    webview: NonNull<AnyObject>,
    upscale_label: Option<&str>,
) -> NonNull<UIMenu> {
    // SAFETY: UIKit owns the visible menu for the duration of this block.
    let menu = unsafe { menu.as_ref() };
    let children = menu.children();
    let select_title = NSString::from_str("Select");
    let has_select =
        (0..children.count()).any(|index| children.objectAtIndex(index).title() == select_title);

    let select_handler = RcBlock::new(move |_action: NonNull<UIAction>| {
        evaluate(webview, SELECT_GALLERY_PRINT_SCRIPT);
    });
    let symbol = UIImage::systemImageNamed(&NSString::from_str("checkmark.circle"));
    // SAFETY: UIKit copies the handler block into the UIAction.
    let action = unsafe {
        UIAction::actionWithTitle_image_identifier_handler(
            &select_title,
            symbol.as_deref(),
            None,
            RcBlock::as_ptr(&select_handler),
            objc2::MainThreadMarker::new_unchecked(),
        )
    };
    let mut children = if has_select {
        children
    } else {
        children.arrayByAddingObject(&action.into_super())
    };
    if let Some(label) = upscale_label {
        let title = NSString::from_str(label);
        let has_upscale =
            (0..children.count()).any(|index| children.objectAtIndex(index).title() == title);
        if !has_upscale {
            let upscale_handler = RcBlock::new(move |_action: NonNull<UIAction>| {
                evaluate(webview, UPSCALE_GALLERY_PRINT_SCRIPT);
            });
            let symbol = UIImage::systemImageNamed(&NSString::from_str("arrow.up.right.square"));
            // SAFETY: UIKit copies the handler block into the UIAction.
            let action = unsafe {
                UIAction::actionWithTitle_image_identifier_handler(
                    &title,
                    symbol.as_deref(),
                    None,
                    RcBlock::as_ptr(&upscale_handler),
                    objc2::MainThreadMarker::new_unchecked(),
                )
            };
            children = children.arrayByAddingObject(&action.into_super());
        }
    }
    let updated = menu.menuByReplacingChildren(&children);
    // UIKit expects an autoreleased menu from the update block.
    NonNull::new(Retained::autorelease_return(updated)).expect("UIKit returned a null menu")
}

#[cfg(target_os = "ios")]
fn update_context_menu_interactions(
    view: NonNull<AnyObject>,
    webview: NonNull<AnyObject>,
    upscale_label: Option<&str>,
) {
    // SAFETY: `view` belongs to the live WKWebView hierarchy. `interactions`
    // and `subviews` are public UIView properties, and this function only runs
    // on UIKit's main queue.
    unsafe {
        let interactions: Retained<NSArray<AnyObject>> =
            objc2::msg_send![view.as_ref(), interactions];
        for index in 0..interactions.count() {
            let interaction = interactions.objectAtIndex(index);
            let supports_update: bool = objc2::msg_send![
                &*interaction,
                respondsToSelector: sel!(updateVisibleMenuWithBlock:)
            ];
            if supports_update {
                let upscale_label = upscale_label.map(str::to_owned);
                let update = RcBlock::new(move |menu: NonNull<UIMenu>| -> NonNull<UIMenu> {
                    append_gallery_actions(menu, webview, upscale_label.as_deref())
                });
                let _: () = objc2::msg_send![
                    &*interaction,
                    updateVisibleMenuWithBlock: &*update
                ];
            }
        }

        let subviews: Retained<NSArray<AnyObject>> = objc2::msg_send![view.as_ref(), subviews];
        for index in 0..subviews.count() {
            update_context_menu_interactions(
                NonNull::from(&*subviews.objectAtIndex(index)),
                webview,
                upscale_label,
            );
        }
    }
}

#[cfg(target_os = "ios")]
fn update_now_and_after_presentation(webview: NonNull<AnyObject>, upscale_label: Option<String>) {
    update_context_menu_interactions(webview, webview, upscale_label.as_deref());

    // The DOM `contextmenu` notification can race UIKit's presentation by a
    // run-loop turn. Retry briefly; duplicate insertion is prevented above.
    for delay in [80, 240] {
        let address = webview.as_ptr() as usize;
        let upscale_label = upscale_label.clone();
        let when = DispatchTime::try_from(Duration::from_millis(delay))
            .expect("context-menu retry delay must fit dispatch time");
        DispatchQueue::main()
            .after(when, move || {
                // SAFETY: The WKWebView is app-owned for the window lifetime,
                // and this closure executes on the main queue.
                let webview = unsafe { NonNull::new_unchecked(address as *mut AnyObject) };
                update_context_menu_interactions(webview, webview, upscale_label.as_deref());
            })
            .expect("main dispatch queue must accept context-menu update");
    }
}

#[tauri::command]
pub fn extend_gallery_context_menu(
    window: tauri::WebviewWindow,
    upscale_label: Option<String>,
) -> Result<(), String> {
    #[cfg(target_os = "ios")]
    window
        .with_webview(|webview| {
            let webview = NonNull::new(webview.inner().cast::<AnyObject>())
                .expect("Tauri supplied a null WKWebView");
            update_now_and_after_presentation(webview, upscale_label);
        })
        .map_err(|error| format!("failed to extend iOS gallery context menu: {error}"))?;

    #[cfg(not(target_os = "ios"))]
    let _ = (window, upscale_label);

    Ok(())
}
