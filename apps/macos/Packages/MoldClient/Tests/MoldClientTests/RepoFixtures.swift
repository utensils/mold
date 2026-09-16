import Foundation

/// Locating things relative to the test file rather than to a bundle, so the
/// tests work under `swift test`, under `xcodebuild`, and from Xcode without
/// three different resource configurations.
enum RepoFixtures {
    static var testDirectory: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    }

    static func fixture(_ name: String) throws -> Data {
        try Data(contentsOf: testDirectory.appending(path: "Fixtures/\(name)"))
    }

    /// Walks up until it finds the mold checkout, so a cross-language contract
    /// test can read the Rust it is pinned against.
    static var repoRoot: URL? {
        var dir = testDirectory
        for _ in 0..<10 {
            if FileManager.default.fileExists(
                atPath: dir.appending(path: "crates/mold-core/src/manifest.rs").path
            ) { return dir }
            let parent = dir.deletingLastPathComponent()
            if parent == dir { break }
            dir = parent
        }
        return nil
    }
}
