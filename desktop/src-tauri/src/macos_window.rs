use objc2::msg_send;
use objc2_app_kit::{NSColor, NSWindow, NSWindowButton};
use objc2_quartz_core::CALayer;
use tauri::Window;

const INACTIVE_FILL_WHITE: f64 = 0.827;
const INACTIVE_BORDER_WHITE: f64 = 0.725;
const TRAFFIC_LIGHT_RADIUS: f64 = 7.0;

#[derive(Debug, PartialEq)]
struct TrafficLightStyle {
    fill_white: f64,
    border_white: f64,
    radius: f64,
}

fn traffic_light_style(focused: bool) -> Option<TrafficLightStyle> {
    (!focused).then_some(TrafficLightStyle {
        fill_white: INACTIVE_FILL_WHITE,
        border_white: INACTIVE_BORDER_WHITE,
        radius: TRAFFIC_LIGHT_RADIUS,
    })
}

fn set_layer_shape(layer: &CALayer, border_width: f64, corner_radius: f64) {
    // SAFETY: These are public CALayer selectors whose arguments are CGFloat;
    // Mold supports only 64-bit macOS targets, where CGFloat is f64.
    unsafe {
        let _: () = msg_send![layer, setBorderWidth: border_width];
        let _: () = msg_send![layer, setCornerRadius: corner_radius];
    }
}

/// macOS 26 can leave the standard controls present and accessible but make
/// their inactive overlay-titlebar appearance fully transparent. A backing
/// layer on the native buttons restores AppKit's expected gray inactive state
/// without replacing their actions, hit testing, accessibility, or active art.
pub fn set_traffic_light_focus(window: &Window, focused: bool) -> tauri::Result<()> {
    // SAFETY: On macOS Tauri documents ns_window() as a pointer to this live
    // NSWindow. The callback borrows it only for the current window event.
    let ns_window = unsafe { &*(window.ns_window()? as *mut NSWindow) };
    let style = traffic_light_style(focused);
    let fill = style
        .as_ref()
        .map(|style| NSColor::colorWithWhite_alpha(style.fill_white, 1.0));
    let border = style
        .as_ref()
        .map(|style| NSColor::colorWithWhite_alpha(style.border_white, 1.0));

    for kind in [
        NSWindowButton::CloseButton,
        NSWindowButton::MiniaturizeButton,
        NSWindowButton::ZoomButton,
    ] {
        let Some(button) = ns_window.standardWindowButton(kind) else {
            continue;
        };
        button.setWantsLayer(true);
        let Some(layer) = button.layer() else {
            continue;
        };

        if let Some(style) = &style {
            let fill_color = fill.as_ref().expect("inactive fill").CGColor();
            let border_color = border.as_ref().expect("inactive border").CGColor();
            layer.setBackgroundColor(Some(&fill_color));
            layer.setBorderColor(Some(&border_color));
            set_layer_shape(&layer, 0.5, style.radius);
        } else {
            layer.setBackgroundColor(None);
            layer.setBorderColor(None);
            set_layer_shape(&layer, 0.0, 0.0);
        }
        layer.setNeedsDisplay();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inactive_windows_receive_gray_native_button_backing() {
        let style = traffic_light_style(false).expect("inactive style");
        assert_eq!(style.fill_white, INACTIVE_FILL_WHITE);
        assert_eq!(style.border_white, INACTIVE_BORDER_WHITE);
        assert_eq!(style.radius, TRAFFIC_LIGHT_RADIUS);
        assert_eq!(traffic_light_style(true), None);
    }
}
