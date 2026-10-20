import AppKit
import Observation

nonisolated enum StartupHomeAccessPolicy {
    static func accepts(expected: URL, selected: URL) -> Bool {
        var directory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: selected.path, isDirectory: &directory),
              directory.boolValue else { return false }
        // NSOpenPanel supplies a directory hint (trailing slash). Compare the
        // canonical filesystem paths, not URL serialization hints.
        return expected.standardizedFileURL.resolvingSymlinksInPath().path
            == selected.standardizedFileURL.resolvingSymlinksInPath().path
    }
}

/// Repairs access to the already-resolved folder; never selects another Mold home.
@MainActor @Observable
final class StartupHomeAccess {
    private(set) var isChoosing = false
    private(set) var notice: String?
    private var panel: NSOpenPanel?
    private var generation = 0
    private var scopedURL: URL?

    func choose(existingHome: URL) {
        guard !isChoosing else { return }
        isChoosing = true
        generation += 1
        let mine = generation
        let chooser = NSOpenPanel()
        panel = chooser
        chooser.title = "Allow access to Mold folder"
        chooser.message = "Select your existing Mold folder. Your library location and credentials will stay the same."
        chooser.prompt = "Allow Access"
        chooser.directoryURL = existingHome
        chooser.canChooseDirectories = true
        chooser.canChooseFiles = false
        chooser.canCreateDirectories = false
        chooser.allowsMultipleSelection = false
        chooser.begin { [weak self] response in
            guard let self, generation == mine else { return }
            panel = nil
            guard response == .OK, let selected = chooser.url else {
                isChoosing = false
                return
            }
            let scoped = selected.startAccessingSecurityScopedResource()
            Task { [weak self] in
                let accepted = await Task.detached {
                    StartupHomeAccessPolicy.accepts(expected: existingHome, selected: selected)
                }.value
                guard let self, generation == mine else {
                    if scoped { selected.stopAccessingSecurityScopedResource() }
                    return
                }
                isChoosing = false
                guard accepted else {
                    if scoped { selected.stopAccessingSecurityScopedResource() }
                    notice = "Choose the existing Mold folder: \(existingHome.path). No library location was changed."
                    return
                }
                if scoped {
                    if scopedURL == nil { scopedURL = selected }
                    else { selected.stopAccessingSecurityScopedResource() }
                }
                notice = "Access allowed. Mold is finishing preparation. If it still waits, quit and reopen Mold."
            }
        }
    }

    func cancel() {
        generation += 1
        panel?.cancel(nil)
        panel = nil
        isChoosing = false
        scopedURL?.stopAccessingSecurityScopedResource()
        scopedURL = nil
    }
}
