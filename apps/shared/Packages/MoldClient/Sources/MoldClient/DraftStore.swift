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
    let writes = DraftWriteGate()

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

    /// Reserve before scheduling a debounced write, so a detached older write
    /// cannot overwrite a later edit or the synchronous quit flush.
    public func reserveWrite() -> UInt64 { writes.reserve() }

    /// Inputs commit before the descriptor that names them. Old/scalar-only
    /// callers keep their existing behavior; Mac supplies the input snapshot.
    @discardableResult
    public func save(_ descriptor: DraftDescriptor, inputs: DraftInputSnapshot? = nil,
                     revision: UInt64? = nil) -> Bool {
        let revision = revision ?? reserveWrite()
        return writes.writer.withLock {
            guard writes.isCurrent(revision) else { return false }
            do {
                var saved = descriptor
                let previous = load()?.localInputsSHA256
                if let inputs { saved.localInputsSHA256 = try saveInputs(inputs) }
                let data = try MoldJSON.localEncoder.encode(saved)
                try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                let committed = try writes.commit(revision) { try writePrivate(data, to: url) }
                if committed {
                    pruneInputs(keeping: Set([saved.localInputsSHA256, previous].compactMap { $0 }))
                }
                return committed
            } catch { return false }
        }
    }

    public func clear() {
        let revision = reserveWrite()
        writes.writer.withLock {
            _ = try? writes.commit(revision) {
                try? FileManager.default.removeItem(at: url)
                try? FileManager.default.removeItem(at: inputsDirectory)
                writes.inputs = nil
            }
        }
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
