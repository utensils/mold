import Foundation

public extension MoldHome {
    /// Why a bootstrap pointer that EXISTS must not be silently ignored.
    ///
    /// `read_saved_mold_dir` (`crates/mold-core/src/config.rs:1339-1362`) says
    /// it plainly: an empty or non-absolute pointer is an ERROR there, because
    /// "every other malformed/unreadable state must fail closed at process
    /// startup". `resolve` answers `~/.mold` for the same file, which is the
    /// right answer for a Mac that never had a pointer -- but the in-process
    /// engine FORCES that answer into `MOLD_HOME`, so mold's own guard can
    /// never fire. A pointer truncated by a crash mid-write or a restored
    /// backup then builds a brand-new empty home: no models, no prints, no
    /// queue, indistinguishable from total data loss while the real library
    /// sits untouched on the other drive (review 05-M5).
    ///
    /// `nil` when there is nothing to refuse: no pointer file at all is a
    /// first run, and an explicit `MOLD_HOME` never consults one.
    static func pointerRefusal(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        home: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> String? {
        if let explicit = environment["MOLD_HOME"], !explicit.isEmpty { return nil }
        let pointer = pointerPath(environment: environment, home: home)
        let path = pointer.path(percentEncoded: false)
        guard FileManager.default.fileExists(atPath: path) else { return nil }
        guard let raw = try? String(contentsOf: pointer, encoding: .utf8) else {
            return "Mold's saved home pointer at \(path) can't be read, so this Mac's "
                + "library can't be located. Fix it, or set MOLD_HOME."
        }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed.hasPrefix("/") else {
            return "Mold's saved home pointer at \(path) doesn't name an absolute path, so "
                + "this Mac's library can't be located. Fix it, or set MOLD_HOME. Starting "
                + "anyway would build a new, empty home in ~/.mold."
        }
        return nil
    }
}
