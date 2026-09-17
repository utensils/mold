import Foundation

// The file half of `SecretStore`: one read, one atomic write, and the rule
// that an unparseable document is parked rather than overwritten. Split from
// the accessors purely for size. Every method here runs under the store's own
// lock -- `Loaded` is passed `inout` rather than read off `self` so that is
// visible at the call site.
extension SecretStore {
    /// The document, read once per process and then held. A file that is not
    /// there is an empty store; a file that does not parse is an empty store
    /// that remembers it owes the original a rename (`secrets.rs:121-148`).
    func loaded(_ slot: inout Loaded?) throws -> Loaded {
        if let slot { return slot }
        let fresh: Loaded
        if let raw = try? Data(contentsOf: url) {
            if let map = try? MoldJSON.localDecoder.decode([String: String].self, from: raw) {
                fresh = Loaded(map: map, corrupt: false)
            } else {
                fresh = Loaded(map: [:], corrupt: true)
            }
        } else {
            fresh = Loaded(map: [:], corrupt: false)
        }
        slot = fresh
        return fresh
    }

    /// Write the whole document, owner-only, atomically.
    ///
    /// `0600` is set on the temporary file BEFORE the rename, not after it
    /// (`secrets.rs:159-167`): a chmod afterwards leaves a window in which the
    /// real path is world-readable, and `replaceItemAt` is not usable here
    /// because it carries the ORIGINAL file's attributes onto the new one --
    /// which would keep a 0644 left behind by an older build forever.
    func save(_ state: inout Loaded) throws {
        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        if state.corrupt {
            // Exactly once, and never over an existing parked copy: the first
            // failure is the one worth keeping.
            let parked = url.appendingPathExtension("corrupt")
            if !FileManager.default.fileExists(atPath: parked.path(percentEncoded: false)) {
                try? FileManager.default.moveItem(at: url, to: parked)
            }
            state.corrupt = false
        }
        let encoder = MoldJSON.localEncoder
        let temporary = directory.appending(path: "secrets.json.\(UUID().uuidString).tmp")
        try encoder.encode(state.map).write(to: temporary)
        do {
            try FileManager.default.setAttributes([.posixPermissions: 0o600],
                                                  ofItemAtPath: temporary.path(percentEncoded: false))
            // `rename(2)`, not `moveItem`: it is atomic, it replaces an
            // existing file, and it does not touch the mode just set.
            guard rename(temporary.path(percentEncoded: false), url.path(percentEncoded: false)) == 0
            else {
                throw SecretStoreError.couldNotReplace(path: url.path(percentEncoded: false),
                                                       code: errno)
            }
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }
    }
}
