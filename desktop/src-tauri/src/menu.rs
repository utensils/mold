//! Native macOS menu bar (design spec §7). Custom items emit a `menu`
//! event with their id; the frontend maps ids onto the same actions the
//! keyboard shortcuts use. Text-editing works because Edit keeps the
//! predefined clipboard items.

use tauri::menu::{AboutMetadata, Menu, MenuItemBuilder, PredefinedMenuItem, SubmenuBuilder};
use tauri::{AppHandle, Emitter, Runtime};

pub fn build<R: Runtime>(app: &AppHandle<R>) -> tauri::Result<Menu<R>> {
    let about = AboutMetadata {
        name: Some("Mold".into()),
        comments: Some("Local AI image and video generation.".into()),
        ..Default::default()
    };

    let app_menu = SubmenuBuilder::new(app, "Mold")
        .item(&PredefinedMenuItem::about(
            app,
            Some("About Mold"),
            Some(about),
        )?)
        .separator()
        .item(
            &MenuItemBuilder::with_id("settings", "Settings…")
                .accelerator("Cmd+,")
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
                .accelerator("Cmd+N")
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
                .accelerator("Cmd+Return")
                .build(app)?,
        )
        .item(
            &MenuItemBuilder::with_id("expand-prompt", "Expand Prompt")
                .accelerator("Cmd+E")
                .build(app)?,
        )
        .item(
            &MenuItemBuilder::with_id("randomize-seed", "Randomize Seed")
                .accelerator("Cmd+R")
                .build(app)?,
        )
        .separator()
        .item(
            &MenuItemBuilder::with_id("cancel-job", "Cancel Job")
                .accelerator("Cmd+.")
                .build(app)?,
        )
        .build()?;

    let view = SubmenuBuilder::new(app, "View")
        .item(
            &MenuItemBuilder::with_id("nav:/generate", "Generate")
                .accelerator("Cmd+1")
                .build(app)?,
        )
        .item(
            &MenuItemBuilder::with_id("nav:/gallery", "Gallery")
                .accelerator("Cmd+2")
                .build(app)?,
        )
        .item(
            &MenuItemBuilder::with_id("nav:/chains", "Chains")
                .accelerator("Cmd+3")
                .build(app)?,
        )
        .item(
            &MenuItemBuilder::with_id("nav:/models", "Models")
                .accelerator("Cmd+4")
                .build(app)?,
        )
        .item(
            &MenuItemBuilder::with_id("nav:/history", "History")
                .accelerator("Cmd+5")
                .build(app)?,
        )
        .separator()
        .item(
            &MenuItemBuilder::with_id("toggle-sidebar", "Toggle Sidebar")
                .accelerator("Cmd+\\")
                .build(app)?,
        )
        .separator()
        .item(
            &MenuItemBuilder::with_id("actual-size", "Actual Size")
                .accelerator("Cmd+0")
                .build(app)?,
        )
        .item(
            &MenuItemBuilder::with_id("zoom-in", "Zoom In")
                .accelerator("Cmd+=")
                .build(app)?,
        )
        .item(
            &MenuItemBuilder::with_id("zoom-out", "Zoom Out")
                .accelerator("Cmd+-")
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
