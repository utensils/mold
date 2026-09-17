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
        guard controller.draft.sourceImage == nil,
              let path = ProcessInfo.processInfo.environment["MOLD_NATIVE_SOURCE_IMAGE"],
              let data = try? Data(contentsOf: URL(fileURLWithPath: path))
        else { return }
        controller.draft.sourceImage = data.base64EncodedString()
        controller.draft.sourceImageName = URL(fileURLWithPath: path).lastPathComponent
    }
}
