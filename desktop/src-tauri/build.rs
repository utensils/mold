fn main() {
    windows_common_controls_manifest_for_tests();
    tauri_build::build()
}

/// Give integration-test binaries the Common Controls v6 dependency that
/// `tauri_build` embeds in the app binary.
///
/// `tao` and `muda` import `TaskDialogIndirect`, `SetWindowSubclass`,
/// `DefSubclassProc`, and `RemoveWindowSubclass` — all version 6 exports.
/// `System32\comctl32.dll` is still 5.82 and exports none of them; version 6
/// is resolved side-by-side, and only for a binary whose manifest asks for it.
/// `tauri_build` writes that manifest for the app, but a `tests/` binary links
/// the same library with no manifest at all, so the loader binds it to 5.82 and
/// the process dies before `main` with STATUS_ENTRYPOINT_NOT_FOUND (0xc0000139)
/// — no output, no backtrace, and a cargo error that looks like a crash in the
/// test rather than a link-time omission.
///
/// `/MANIFESTDEPENDENCY` states the dependency without a manifest file, and
/// `-tests` scopes it to test targets so the app binary keeps the exact
/// manifest `tauri_build` generates rather than gaining a second source of it.
fn windows_common_controls_manifest_for_tests() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc") {
        return;
    }
    println!("cargo::rustc-link-arg-tests=/MANIFEST:EMBED");
    println!(
        "cargo::rustc-link-arg-tests=/MANIFESTDEPENDENCY:type='win32' \
         name='Microsoft.Windows.Common-Controls' version='6.0.0.0' \
         processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'"
    );
}
