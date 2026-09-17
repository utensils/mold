import Foundation

// The file half of `SecretStore`: one read, one atomic write, and the rule
// that an unparseable document is parked rather than overwritten. Split from
// the accessors purely for size. Every method here runs under the store's own
// lock -- `Loaded` is passed `inout` rather than read off `self` so that is
// visible at the call site.
extension SecretStore {
    /// The document, read once per process and then held.
    ///
    /// Three outcomes, and the middle one is the whole point. A file that is
    /// NOT THERE is an empty store -- a first run. A file that does not PARSE
    /// is an empty store that remembers it owes the original a rename
    /// (`secrets.rs:121-148`). A file that EXISTS but will not read -- the
    /// wrong owner after a `sudo` launch, an ACL, a transient I/O error -- is
    /// a refusal: treating it as empty is how the next `set` renamed a
    /// one-entry document over every other machine's key (review E2). Parking
    /// it would not help, because whatever stopped the read will stop the
    /// rename; the store refuses, and `HostStore.report` says so.
    ///
    /// The refusal is never cached, so the moment the file reads again it
    /// reads.
    func loaded(_ slot: inout Loaded?) throws -> Loaded {
        if let slot { return slot }
        let path = url.path(percentEncoded: false)
        var raw: Data?
        do {
            raw = try Data(contentsOf: url)
        } catch {
            guard !FileManager.default.fileExists(atPath: path) else {
                throw SecretStoreError.unreadable(path: path, reason: error.localizedDescription)
            }
            raw = nil
        }
        let fresh: Loaded
        if let raw {
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
    /// The temporary file is owner-only from its FIRST BYTE and every failure
    /// path removes it. `replaceItemAt` is not usable here because it carries
    /// the ORIGINAL file's attributes onto the new one -- which would keep a
    /// 0644 left behind by an older build forever.
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
        let temporary = directory.appending(path: "secrets.json.\(UUID().uuidString).tmp")
        do {
            try Self.writeOwnerOnly(MoldJSON.localEncoder.encode(state.map), to: temporary)
            // `rename(2)`, not `moveItem`: it is atomic, it replaces an
            // existing file, and it does not touch the mode it was made with.
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

    /// One file, created with mode `0600` and flushed to the platter.
    ///
    /// NOT `Data.write(to:)` plus a chmod: that creates at the process umask,
    /// so a complete copy of every credential exists world-readable until the
    /// next statement runs -- and if the write itself throws (ENOSPC, EDQUOT,
    /// EIO) it is neither chmodded nor removed, and nothing ever cleans up a
    /// UUID-named file (review E3). `createFile` applies the attributes AT
    /// creation, so there is no window at all.
    ///
    /// The `fsync` is for the caller after it: `rename(2)` makes the metadata
    /// durable but not the bytes, and `LegacyKeychain` deletes the only other
    /// copy of a key the moment the write returns.
    static func writeOwnerOnly(_ data: Data, to url: URL) throws {
        let path = url.path(percentEncoded: false)
        guard FileManager.default.createFile(atPath: path, contents: data,
                                             attributes: [.posixPermissions: 0o600])
        else { throw SecretStoreError.couldNotReplace(path: path, code: errno) }
        let descriptor = open(path, O_RDONLY)
        guard descriptor >= 0 else { return }
        fsync(descriptor)
        close(descriptor)
    }
}
