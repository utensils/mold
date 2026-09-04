//! Native desktop menu bar. Custom items emit a `menu`
//! event with their id; the frontend maps ids onto the same actions the
//! keyboard shortcuts use. Text-editing works because Edit keeps the
//! predefined clipboard items.

use tauri::menu::{AboutMetadata, Menu, MenuItemBuilder, PredefinedMenuItem, SubmenuBuilder};
use tauri::{AppHandle, Emitter, Runtime};

fn accelerator(key: &str) -> String {
    let modifier = if cfg!(target_os = "macos") {
        "Cmd"
    } else {
        "Ctrl"
    };
    format!("{modifier}+{key}")
}

/// File and Generate items, in the binding lexicon (docs/design/README.md §2):
/// plain words, never the engine word. Ids and accelerators are the contract
/// the frontend maps onto its own shortcut actions, so only the labels move.
const FILE_MENU_ITEMS: [(&str, &str, Option<&str>); 2] = [
    ("new-generation", "New Image", Some("N")),
    ("new-sequence", "New Clip", None),
];

const GENERATE_MENU_ITEMS: [(&str, &str, &str); 4] = [
    ("generate", "Generate", "Return"),
    ("expand-prompt", "Write More For Me", "E"),
    ("randomize-seed", "Surprise Me", "R"),
    ("cancel-job", "Stop", "."),
];

const NAVIGATION_MENU_ITEMS: [(&str, &str, &str); 5] = [
    ("nav:/create", "New image", "1"),
    ("nav:/queue", "Queue", "2"),
    ("nav:/library", "My images", "3"),
    ("nav:/models", "Styles", "4"),
    ("nav:/machines", "Machines", "5"),
];

/// macOS gets the predefined Window items; every other platform builds the same
/// three entries as ordinary commands, handled by the platform-independent
/// `on_menu_event` arms below.
#[cfg(not(target_os = "macos"))]
const WINDOW_MENU_ITEMS: [(&str, &str); 3] = [
    ("window:minimize", "Minimize"),
    ("window:toggle-maximize", "Maximize"),
    ("window:close", "Close Window"),
];

fn app_name(is_development: bool) -> &'static str {
    if is_development {
        "mold-dev"
    } else {
        "Mold"
    }
}

fn about_metadata(app_name: &str) -> AboutMetadata<'static> {
    AboutMetadata {
        name: Some(app_name.into()),
        authors: Some(vec!["James Brink".into(), "Jeffrey Dilley".into()]),
        comments: Some("Local AI image and video generation.".into()),
        copyright: Some("Copyright © 2026 James Brink and Jeffrey Dilley".into()),
        license: Some("MIT License".into()),
        website: Some("https://github.com/utensils/mold".into()),
        website_label: Some("Mold on GitHub".into()),
        credits: Some("Core contributors:\nJames Brink\nJeffrey Dilley".into()),
        ..Default::default()
    }
}

#[cfg(target_os = "macos")]
pub fn set_process_name(is_development: bool) {
    if is_development {
        use objc2_foundation::{NSProcessInfo, NSString};

        NSProcessInfo::processInfo().setProcessName(&NSString::from_str(app_name(true)));
    }
}

#[cfg(not(target_os = "macos"))]
pub fn set_process_name(_is_development: bool) {}

pub fn build<R: Runtime>(app: &AppHandle<R>, is_development: bool) -> tauri::Result<Menu<R>> {
    let app_name = app_name(is_development);
    let about_label = format!("About {app_name}");
    let about = about_metadata(app_name);

    let app_menu = SubmenuBuilder::new(app, app_name)
        .item(&PredefinedMenuItem::about(
            app,
            Some(&about_label),
            Some(about),
        )?)
        .separator()
        .item(&MenuItemBuilder::with_id("check-for-updates", "Check for Updates…").build(app)?)
        .separator()
        .item(
            &MenuItemBuilder::with_id("settings", "Settings…")
                .accelerator(accelerator(","))
                .build(app)?,
        )
        .separator();
    // Services and the hide/show family are macOS application-menu concepts
    // with no counterpart on Windows or Linux, where they render as dead
    // entries.
    #[cfg(target_os = "macos")]
    let app_menu = app_menu
        .services()
        .separator()
        .hide()
        .hide_others()
        .show_all()
        .separator();
    let app_menu = app_menu.quit().build()?;

    let mut file = SubmenuBuilder::new(app, "File");
    for (id, label, key) in FILE_MENU_ITEMS {
        let mut item = MenuItemBuilder::with_id(id, label);
        if let Some(key) = key {
            item = item.accelerator(accelerator(key));
        }
        file = file.item(&item.build(app)?);
    }
    let file = file
        .separator()
        .item(&PredefinedMenuItem::close_window(app, None)?)
        .build()?;

    let edit = SubmenuBuilder::new(app, "Edit")
        .undo()
        .redo()
        .separator()
        .cut()
        .copy()
        .paste()
        .select_all()
        .build()?;

    let mut generate = SubmenuBuilder::new(app, "Generate");
    for (id, label, key) in GENERATE_MENU_ITEMS {
        // Stop is destructive, so it sits below a separator.
        if id == "cancel-job" {
            generate = generate.separator();
        }
        generate = generate.item(
            &MenuItemBuilder::with_id(id, label)
                .accelerator(accelerator(key))
                .build(app)?,
        );
    }
    let generate = generate.build()?;

    let mut view = SubmenuBuilder::new(app, "View");
    for (id, label, key) in NAVIGATION_MENU_ITEMS {
        view = view.item(
            &MenuItemBuilder::with_id(id, label)
                .accelerator(accelerator(key))
                .build(app)?,
        );
    }
    let view = view
        .separator()
        .item(
            &MenuItemBuilder::with_id("toggle-sidebar", "Toggle Sidebar")
                .accelerator(accelerator("\\"))
                .build(app)?,
        )
        .separator()
        .item(
            &MenuItemBuilder::with_id("actual-size", "Actual Size")
                .accelerator(accelerator("0"))
                .build(app)?,
        )
        .item(
            &MenuItemBuilder::with_id("zoom-in", "Zoom In")
                .accelerator(accelerator("="))
                .build(app)?,
        )
        .item(
            &MenuItemBuilder::with_id("zoom-out", "Zoom Out")
                .accelerator(accelerator("-"))
                .build(app)?,
        )
        .separator()
        // The webview's own context menu (the usual devtools entry point) is
        // suppressed app-wide; this is the sanctioned way in for debugging.
        .item(
            &MenuItemBuilder::with_id("devtools", "Developer Tools")
                .accelerator(if cfg!(target_os = "macos") {
                    "Cmd+Alt+I"
                } else {
                    "Ctrl+Shift+I"
                })
                .build(app)?,
        )
        .separator()
        .fullscreen()
        .build()?;

    #[cfg(not(target_os = "macos"))]
    let window = {
        let mut window = SubmenuBuilder::new(app, "Window");
        for (id, label) in WINDOW_MENU_ITEMS {
            let mut item = MenuItemBuilder::with_id(id, label);
            if id == "window:close" {
                item = item.accelerator(accelerator("W"));
            }
            window = window.item(&item.build(app)?);
        }
        window.build()?
    };

    // `bring_all_to_front` is a macOS-only predefined item, so this arm is
    // macOS-only too — Windows took it by being "not Linux" and got a Window
    // menu whose last entry does nothing.
    #[cfg(target_os = "macos")]
    let window = SubmenuBuilder::new(app, "Window")
        .minimize()
        .maximize()
        .separator()
        .item(&PredefinedMenuItem::bring_all_to_front(app, None)?)
        .build()?;

    let help = SubmenuBuilder::new(app, "Help")
        .item(&MenuItemBuilder::with_id("help:docs", "Mold Documentation").build(app)?)
        .item(&MenuItemBuilder::with_id("help:api", "API Reference").build(app)?)
        .item(&MenuItemBuilder::with_id("help:logs", "Open Logs").build(app)?)
        .build()?;

    let menu = Menu::with_items(
        app,
        &[&app_menu, &file, &edit, &generate, &view, &window, &help],
    )?;

    app.on_menu_event(move |app, event| {
        let id = event.id().0.clone();
        match id.as_str() {
            "devtools" => {
                use tauri::Manager;
                if let Some(window) = app.get_webview_window("main") {
                    if window.is_devtools_open() {
                        window.close_devtools();
                    } else {
                        window.open_devtools();
                    }
                }
            }
            "window:minimize" => {
                use tauri::Manager;
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.minimize();
                }
            }
            "window:toggle-maximize" => {
                use tauri::Manager;
                if let Some(window) = app.get_webview_window("main") {
                    if window.is_maximized().unwrap_or(false) {
                        let _ = window.unmaximize();
                    } else {
                        let _ = window.maximize();
                    }
                }
            }
            "window:close" => {
                use tauri::Manager;
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.close();
                }
            }
            "help:docs" => {
                use tauri_plugin_opener::OpenerExt;
                let _ = app
                    .opener()
                    .open_url("https://utensils.io/mold/", None::<&str>);
            }
            "help:logs" => {
                use tauri_plugin_opener::OpenerExt;
                let dir = mold_core::Config::load_or_default().resolved_log_dir();
                let _ = app.opener().open_path(dir.to_string_lossy(), None::<&str>);
            }
            _ => {
                let _ = app.emit("menu", id);
            }
        }
    });

    Ok(menu)
}

#[cfg(test)]
mod tests {
    #[cfg(not(target_os = "macos"))]
    use super::WINDOW_MENU_ITEMS;
    use super::{
        about_metadata, accelerator, app_name, FILE_MENU_ITEMS, GENERATE_MENU_ITEMS,
        NAVIGATION_MENU_ITEMS,
    };

    #[test]
    fn about_metadata_credits_both_core_contributors() {
        let about = about_metadata("Mold");

        assert_eq!(
            about.authors,
            Some(vec!["James Brink".into(), "Jeffrey Dilley".into()])
        );
        assert_eq!(
            about.credits.as_deref(),
            Some("Core contributors:\nJames Brink\nJeffrey Dilley")
        );
    }

    #[test]
    fn uses_the_platform_primary_modifier() {
        let expected = if cfg!(target_os = "macos") {
            "Cmd+K"
        } else {
            "Ctrl+K"
        };
        assert_eq!(accelerator("K"), expected);
    }

    #[test]
    fn development_menu_identity_is_distinct_from_release() {
        assert_eq!(app_name(true), "mold-dev");
        assert_eq!(app_name(false), "Mold");
    }

    #[test]
    fn navigation_menu_matches_the_sidebar_in_the_lexicon() {
        // The same five destinations, words, and ⌘ digits as the sidebar
        // (desktop/src/lib/shortcuts.ts NAV_ROUTES); Settings stays on ⌘,.
        assert_eq!(
            NAVIGATION_MENU_ITEMS,
            [
                ("nav:/create", "New image", "1"),
                ("nav:/queue", "Queue", "2"),
                ("nav:/library", "My images", "3"),
                ("nav:/models", "Styles", "4"),
                ("nav:/machines", "Machines", "5"),
            ]
        );
    }

    #[test]
    fn file_and_generate_menus_speak_the_binding_lexicon() {
        // docs/design/README.md §2: New image / Short clip / Write more for me
        // / Surprise me / Stop. The ids stay put — the frontend maps them.
        assert_eq!(
            FILE_MENU_ITEMS,
            [
                ("new-generation", "New Image", Some("N")),
                ("new-sequence", "New Clip", None),
            ]
        );
        assert_eq!(
            GENERATE_MENU_ITEMS,
            [
                ("generate", "Generate", "Return"),
                ("expand-prompt", "Write More For Me", "E"),
                ("randomize-seed", "Surprise Me", "R"),
                ("cancel-job", "Stop", "."),
            ]
        );
    }

    /// Windows and Linux build the Window menu from plain command items whose
    /// handlers are platform-independent. macOS keeps the predefined items,
    /// including `bring_all_to_front`, which exists only there.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn non_mac_window_menu_uses_supported_command_items() {
        assert_eq!(
            WINDOW_MENU_ITEMS,
            [
                ("window:minimize", "Minimize"),
                ("window:toggle-maximize", "Maximize"),
                ("window:close", "Close Window"),
            ]
        );
    }
}
