import Foundation
import MoldClient
import Testing

@testable import Mold

/// The server renders exactly two thumbnail renditions and answers any other
/// `?size=` with a 422 (`thumbnails.rs` `SIZES`). Quick Look on a mesh asked
/// for a 1024 px poster, the `try?` around it swallowed the refusal, and
/// Space did nothing on a mesh tile (UAT 2026-09-17 #5). The list lives once
/// in `MediaURL` and is pinned here to the Rust it mirrors.
struct ThumbnailSizesTests {
    /// **Fails today**: `MediaURL` has no such list.
    @Test func theRenditionsAreTheServersOwn() throws {
        let rust = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // Tests/MoldTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // apps/macos
            .deletingLastPathComponent() // apps
            .deletingLastPathComponent() // <repo>
            .appending(path: "crates/mold-server/src/thumbnails.rs")
        let source = try String(contentsOf: rust, encoding: .utf8)
        let declared = source
            .split(separator: "\n")
            .first { $0.contains("pub const SIZES") }
            .flatMap { $0.split(separator: "[").last }
            .map { $0.split(separator: "]").first ?? "" }
            .map { $0.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) } }
        #expect(declared == MediaURL.thumbnailSizes)
        #expect(MediaURL.largestThumbnail == 512)
    }
}
