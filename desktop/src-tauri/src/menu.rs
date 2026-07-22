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
        credits: Some(
            "Core contributors and equal project owners:\nJames Brink\nJeffrey Dilley".into(),
        ),
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
        .separator()
        .services()
        .separator()
        .hide()
        .hide_others()
        .show_all()
        .separator()
        .quit()
        .build()?;

    let file = SubmenuBuilder::new(app, "File")
        .item(
            &MenuItemBuilder::with_id("new-generation", "New Generation")
                .accelerator(accelerator("N"))
                .build(app)?,
        )
        .item(&MenuItemBuilder::with_id("new-chain", "New Chain").build(app)?)
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

    let generate = SubmenuBuilder::new(app, "Generate")
        .item(
            &MenuItemBuilder::with_id("generate", "Generate")
                .accelerator(accelerator("Return"))
                .build(app)?,
        )
        .item(
            &MenuItemBuilder::with_id("expand-prompt", "Expand Prompt")
                .accelerator(accelerator("E"))
                .build(app)?,
        )
        .item(
            &MenuItemBuilder::with_id("randomize-seed", "Randomize Seed")
                .accelerator(accelerator("R"))
                .build(app)?,
        )
        .separator()
        .item(
            &MenuItemBuilder::with_id("cancel-job", "Cancel Job")
                .accelerator(accelerator("."))
                .build(app)?,
        )
        .build()?;

    let view = SubmenuBuilder::new(app, "View")
        .item(
            &MenuItemBuilder::with_id("nav:/generate", "Generate")
                .accelerator(accelerator("1"))
                .build(app)?,
        )
        .item(
            &MenuItemBuilder::with_id("nav:/gallery", "Gallery")
                .accelerator(accelerator("2"))
                .build(app)?,
        )
        .item(
            &MenuItemBuilder::with_id("nav:/chains", "Chains")
                .accelerator(accelerator("3"))
                .build(app)?,
        )
        .item(
            &MenuItemBuilder::with_id("nav:/models", "Models")
                .accelerator(accelerator("4"))
                .build(app)?,
        )
        .item(
            &MenuItemBuilder::with_id("nav:/history", "History")
                .accelerator(accelerator("5"))
                .build(app)?,
        )
        .item(
            &MenuItemBuilder::with_id("nav:/jobs", "Jobs")
                .accelerator(accelerator("6"))
                .build(app)?,
        )
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
    use super::{about_metadata, accelerator, app_name};

    #[test]
    fn about_metadata_credits_both_core_contributors() {
        let about = about_metadata("Mold");

        assert_eq!(
            about.authors,
            Some(vec!["James Brink".into(), "Jeffrey Dilley".into()])
        );
        assert!(about.credits.as_deref().unwrap().contains("James Brink"));
        assert!(about.credits.as_deref().unwrap().contains("Jeffrey Dilley"));
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
}
