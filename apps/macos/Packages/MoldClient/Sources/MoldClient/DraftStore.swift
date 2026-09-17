import Foundation

/// The one file the Generate draft survives a quit in.
///
/// Application Support, NOT `UserDefaults`: preferences are read whole on
/// every launch and written back whole, and a prompt somebody pasted three
/// pages of is not a preference. The directory is
/// `SecretStore.applicationSupport`'s, so a UAT run under `MOLD_NATIVE_FRESH`
/// gets the throwaway twin and can exercise a first launch without ever
/// reading -- or overwriting -- a real draft.
///
/// `nonisolated`: every call here touches the disk and belongs off the main
/// actor, and the app's default MainActor isolation would otherwise pull it
/// back on.
public nonisolated struct DraftStore: Sendable {
    let url: URL

    public init(directory: URL = SecretStore.applicationSupport()) {
        url = directory.appending(path: "generate-draft.json")
    }

    /// The descriptor, or `nil`.
    ///
    /// THREE outcomes and the middle one matters: no file is a first launch;
    /// a file at another VERSION is discarded whole, because a half-restored
    /// draft is worse than an empty one; and a file that does not parse is
    /// PARKED beside itself rather than clobbered, so a bug here never
    /// silently eats the thing it was meant to protect (`SecretStore+File`'s
    /// own rule).
    public func load() -> DraftDescriptor? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        guard let descriptor = try? MoldJSON.localDecoder.decode(
            DraftDescriptor.self, from: data) else {
            park()
            return nil
        }
        guard descriptor.version == DraftDescriptor.currentVersion else {
            park()
            return nil
        }
        return descriptor
    }

    /// Writes it, atomically. A failure is not worth a sentence: the draft is
    /// on screen, nothing has been lost yet, and the next change tries again.
    public func save(_ descriptor: DraftDescriptor) {
        guard let data = try? MoldJSON.localEncoder.encode(descriptor) else { return }
        let directory = url.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let temporary = directory.appending(path: "generate-draft.\(UUID().uuidString).tmp")
        guard (try? data.write(to: temporary)) != nil else {
            try? FileManager.default.removeItem(at: temporary)
            return
        }
        // `rename(2)`, not `moveItem`: atomic, replaces, and leaves no window
        // in which the file on disk is half a draft.
        guard rename(temporary.path(percentEncoded: false),
                     url.path(percentEncoded: false)) == 0 else {
            try? FileManager.default.removeItem(at: temporary)
            return
        }
    }

    public func clear() {
        try? FileManager.default.removeItem(at: url)
    }

    /// Moves an unreadable document aside, exactly once -- the FIRST failure
    /// is the one worth keeping.
    private func park() {
        let parked = url.appendingPathExtension("corrupt")
        guard !FileManager.default.fileExists(atPath: parked.path(percentEncoded: false)) else {
            try? FileManager.default.removeItem(at: url)
            return
        }
        try? FileManager.default.moveItem(at: url, to: parked)
    }
}
