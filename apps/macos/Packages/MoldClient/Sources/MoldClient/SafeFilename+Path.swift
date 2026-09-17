import Foundation

// A checked name, as a path. Split from the rule itself for size: one file
// says what a safe name IS, this one says where it may land.
public extension SafeFilename {

    /// The file this name names inside `directory`, or `nil` when it would not
    /// stay there.
    ///
    /// Compared by PARENT, so the answer is "this is one component inside that
    /// directory" and not "the string starts with the right prefix" -- which
    /// `/tmp/cache-evil` satisfies for `/tmp/cache`.
    ///
    /// The parent is resolved through symlinks, because `standardizedFileURL`
    /// removes `.` and `..` and stops there: any other unsandboxed process on
    /// this Mac can plant a link at the cache directory and `write(to:)`
    /// follows it. The LEAF is deliberately not resolved -- it is refused
    /// outright by `isFreshDestination` at the write, because a link there is
    /// a file this app did not create and has no business writing through.
    public static func url(_ name: String, in directory: URL) -> URL? {
        guard isSafe(name) else { return nil }
        let base = directory.resolvingSymlinksInPath().standardizedFileURL
        let file = base.appending(path: name).standardizedFileURL
        guard directoryPath(file.deletingLastPathComponent()) == directoryPath(base),
              file.lastPathComponent == name
        else { return nil }
        return file
    }

    /// Whether this app may write to `file` -- true when nothing is there, or
    /// when what is there is an ordinary file it can replace.
    ///
    /// A symbolic link is refused: `Data.write(to:)` FOLLOWS one, so a link
    /// planted at a cache path is server-controlled bytes landing wherever it
    /// points. This is the leaf half of `url(_:in:)`'s containment, and the
    /// two are always used together.
    public static func isFreshDestination(_ file: URL) -> Bool {
        let values = try? file.resourceValues(forKeys: [.isSymbolicLinkKey, .isRegularFileKey])
        guard let values else { return true }  // nothing there at all
        if values.isSymbolicLink == true { return false }
        return values.isRegularFile ?? true
    }

    /// A directory's path without the trailing separator `deletingLastPathComponent`
    /// leaves behind, so two of them can be compared at all.
    private static func directoryPath(_ url: URL) -> String {
        var path = url.path(percentEncoded: false)
        while path.count > 1, path.hasSuffix("/") { path.removeLast() }
        return path
    }
}
