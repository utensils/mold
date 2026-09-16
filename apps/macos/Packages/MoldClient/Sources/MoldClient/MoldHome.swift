import Foundation

/// Where mold keeps its models, its database and its gallery on this Mac.
///
/// The Swift twin of `Config::mold_dir` in `crates/mold-core/src/config.rs`,
/// and it has to be, because the engine this app starts in-process is the same
/// engine `mold serve` starts: if the two disagree about the home, one Mac has
/// two libraries and the app appears to have lost every model and every print.
/// The order is mold's own -- `MOLD_HOME`, then the bootstrap pointer the
/// Desktop app writes when someone moves their home to another drive, then
/// `~/.mold`.
public struct MoldHome: Equatable, Sendable {
    /// Which of the three answers this was, because they are not equally
    /// trustworthy: a chosen home that is missing is a problem to report,
    /// while a default one that is missing is simply a first run.
    public enum Source: Equatable, Sendable {
        case environment
        case saved
        case fallback
    }

    public let url: URL
    public let source: Source

    /// Set when the home someone deliberately chose is not currently there --
    /// an external drive, usually. mold's own `ensure_saved_mold_dir_available`
    /// refuses to start in that state for the same reason: silently creating a
    /// fresh empty home in its place is indistinguishable from data loss.
    public var unavailableReason: String? {
        guard source == .saved,
              !FileManager.default.fileExists(atPath: url.path(percentEncoded: false))
        else { return nil }
        return "Mold's home at \(url.path(percentEncoded: false)) isn't available. "
            + "Reconnect its drive, or set MOLD_HOME to somewhere else."
    }

    public static func resolve(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        home: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> MoldHome {
        if let explicit = environment["MOLD_HOME"], !explicit.isEmpty {
            return MoldHome(url: URL(filePath: explicit), source: .environment)
        }
        if let saved = savedPath(environment: environment, home: home) {
            return MoldHome(url: saved, source: .saved)
        }
        return MoldHome(url: home.appending(path: ".mold"), source: .fallback)
    }

    /// `dirs::config_dir()` is `~/Library/Application Support` on macOS, not
    /// `~/.config` -- the same call returns different places per platform and
    /// the Rust side is the one that writes this file.
    public static func pointerPath(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        home: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> URL {
        if let override = environment["MOLD_HOME_POINTER_PATH"], !override.isEmpty {
            return URL(filePath: override)
        }
        return home.appending(path: "Library/Application Support/mold/home")
    }

    private static func savedPath(environment: [String: String], home: URL) -> URL? {
        let pointer = pointerPath(environment: environment, home: home)
        guard let raw = try? String(contentsOf: pointer, encoding: .utf8) else { return nil }
        let path = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        // A relative or empty pointer is a damaged file, and guessing what it
        // meant is how you end up writing a gallery into the wrong directory.
        guard path.hasPrefix("/") else { return nil }
        return URL(filePath: path)
    }
}
