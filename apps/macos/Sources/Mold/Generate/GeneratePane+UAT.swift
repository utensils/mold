import Foundation

// The UAT-only seed for a source picture. Split from `GeneratePane.swift`
// purely for size.
extension GeneratePane {
    /// `MOLD_NATIVE_SOURCE_IMAGE=<path>` preloads a source picture once, the
    /// way `MOLD_NATIVE_DESTINATION` seeds where the window opens -- so a UAT
    /// run can photograph the Refine group's mask row (and the sheet it
    /// opens onto) without driving an `NSOpenPanel`. The `sourceImage == nil`
    /// guard is what makes this a ONE-TIME seed: once set, the field is no
    /// longer nil and a later re-run of this task is a no-op.
    ///
    /// Not `private`: the main file's `.task` calls it.
    func seedSourceImageIfRequested() {
        guard controller.draft.media.sourceImage == nil,
              let path = ProcessInfo.processInfo.environment["MOLD_NATIVE_SOURCE_IMAGE"],
              let data = try? Data(contentsOf: URL(fileURLWithPath: path))
        else { return }
        controller.draft.media.sourceImage = data.base64EncodedString()
        controller.draft.media.sourceImageName = URL(fileURLWithPath: path).lastPathComponent
    }
}

/// `MOLD_NATIVE_LIBRARY_PICKER=1` opens `SourceImageWell`'s "From Library…"
/// sheet at launch, mirroring `SettingsUAT`'s `MOLD_NATIVE_SETTINGS_TAB` --
/// a deterministic capture instead of a menu press.
enum GenerateUAT {
    static let envVar = "MOLD_NATIVE_LIBRARY_PICKER"

    static func wantsLibraryPicker(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        environment[envVar] != nil
    }
}
