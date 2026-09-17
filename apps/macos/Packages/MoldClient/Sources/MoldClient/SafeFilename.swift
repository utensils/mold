import Foundation

/// A filename a machine sent, checked before this app makes a path out of it.
///
/// `GalleryPrint.filename` is whatever the host on the other end put in its
/// JSON, and this app is deliberately NOT sandboxed, writes that name into its
/// media cache, and `removeItem`s at the same path before copying a save into
/// place. `URL.appending(path:)` keeps `/` and `..` verbatim -- they are legal
/// path characters -- and POSIX resolves them at the moment of the write, so a
/// listing answering `"../../../../Library/LaunchAgents/evil.plist"` is
/// arbitrary file creation, and on the multi-print save path an arbitrary
/// delete.
///
/// An honest mold cannot produce such a name: `clean_gallery_filename` and
/// `render_gallery_thumbnail`'s `clean_name != filename` check refuse them at
/// the source (`crates/mold-server/src/routes.rs:10679-10683`), which is
/// exactly why this client never noticed. But a machine is added here by bare
/// address and `HostAddress` fills in `http://`, so "a machine I added once" is
/// a weaker boundary than "code I shipped". The rule is therefore written down
/// once, here, and applied twice: at the decode, so an unsafe name never
/// reaches the app's model, and again at every site that turns a name into a
/// path, so one missed decode is still not a write outside the cache.
public enum SafeFilename {
    /// APFS and HFS+ both stop at 255 UTF-8 bytes for one component.
    public static let maxBytes = 255

    /// Why a name was refused. Carried so a log line can say which rule it
    /// broke rather than "bad filename".
    public enum Reason: String, Hashable, Sendable {
        /// Empty, or nothing but whitespace.
        case empty
        /// Longer than one path component may be.
        case tooLong
        /// Carries `/`, `\` or `:` -- a separator on POSIX, on Windows, or in
        /// the Finder's own rendering of a name.
        case separator
        /// Carries NUL or another control character.
        case controlCharacter
        /// Is `.` or `..`.
        case relative
        /// Starts with a dot, which hides the file and is never a print's name.
        case hidden
        /// Percent-decodes into something that is not a single component --
        /// `..%2Fevil` arrives from JSON as those nine literal characters.
        case encodedSeparator
        /// Anything `lastPathComponent` disagrees with: an absolute path, a
        /// trailing separator, a name that is not one component after all.
        case notOneComponent
    }

    public struct Rejected: Error, Hashable, Sendable {
        public let name: String
        public let reason: Reason

        public init(name: String, reason: Reason) {
            self.name = name
            self.reason = reason
        }
    }

    public static func isSafe(_ name: String) -> Bool {
        (try? validated(name)) != nil
    }

    /// The name back, or a refusal naming the rule it broke.
    @discardableResult
    public static func validated(_ name: String) throws -> String {
        func refuse(_ reason: Reason) -> Rejected { Rejected(name: name, reason: reason) }

        guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw refuse(.empty)
        }
        guard name.utf8.count <= maxBytes else { throw refuse(.tooLong) }
        guard !name.unicodeScalars.contains(where: isControl) else {
            throw refuse(.controlCharacter)
        }
        guard !name.contains("/"), !name.contains("\\"), !name.contains(":") else {
            throw refuse(.separator)
        }
        guard name != ".", name != ".." else { throw refuse(.relative) }
        guard !name.hasPrefix(".") else { throw refuse(.hidden) }
        // Belt and braces over the three literal separators above: whatever
        // Foundation itself thinks this name's last component is, it has to be
        // the whole name.
        guard (name as NSString).lastPathComponent == name else {
            throw refuse(.notOneComponent)
        }
        if let decoded = name.removingPercentEncoding, decoded != name,
           decoded != (decoded as NSString).lastPathComponent || decoded.hasPrefix(".") {
            throw refuse(.encodedSeparator)
        }
        return name
    }

    /// The file this name names inside `directory`, or `nil` when it would not
    /// stay there.
    ///
    /// Standardised on both sides and compared by PARENT, so the answer is
    /// "this is one component inside that directory" and not "the string
    /// starts with the right prefix" -- which `/tmp/cache-evil` satisfies for
    /// `/tmp/cache`.
    public static func url(_ name: String, in directory: URL) -> URL? {
        guard isSafe(name) else { return nil }
        let base = directory.standardizedFileURL
        let file = base.appending(path: name).standardizedFileURL
        guard directoryPath(file.deletingLastPathComponent()) == directoryPath(base),
              file.lastPathComponent == name
        else { return nil }
        return file
    }

    /// A directory's path without the trailing separator `deletingLastPathComponent`
    /// leaves behind, so two of them can be compared at all.
    private static func directoryPath(_ url: URL) -> String {
        var path = url.path(percentEncoded: false)
        while path.count > 1, path.hasSuffix("/") { path.removeLast() }
        return path
    }

    /// A server-supplied string made safe to use as a path component, by
    /// replacing everything that could make it more than one.
    ///
    /// For values that are NOT names of anything -- a `media_version`, which
    /// this app only uses because it needs a directory to put a print in. A
    /// filename is validated and refused; a version is folded, because
    /// refusing one would refuse the print.
    /// `limit` is there because a folded value is usually part of a longer
    /// component -- a cache key is a machine's UUID AND this -- and a name
    /// that is one byte too long is a directory nobody can create and a
    /// materialization that silently answers nothing.
    public static func folded(_ value: String, fallback: String,
                              limit: Int = maxBytes) -> String {
        var folded = String(value.unicodeScalars.map { scalar -> Character in
            isControl(scalar) || "/\\:".unicodeScalars.contains(scalar)
                ? "-" : Character(scalar)
        })
        if folded.hasPrefix(".") { folded = "-" + folded.dropFirst() }
        while folded.utf8.count > Swift.min(limit, maxBytes) { folded.removeLast() }
        return isSafe(folded) ? folded : fallback
    }

    private static func isControl(_ scalar: Unicode.Scalar) -> Bool {
        scalar.value < 0x20 || scalar.value == 0x7F
    }
}
