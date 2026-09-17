import Foundation
import MoldClient
import Testing

@testable import Mold

/// The media cache, as a thing that writes server-supplied strings to disk.
///
/// **Fails today**: `url(for:fetch:)` builds its path by appending the host's
/// `filename` and `media_version` to the cache root with no validation at all
/// (`PrintMaterializer.swift:49,63-65,85-90`), in an app that is deliberately
/// not sandboxed.
@MainActor
struct PrintMaterializerTests {
    private func root() -> URL {
        FileManager.default.temporaryDirectory
            .appending(path: "mold-materializer-\(UUID().uuidString)")
    }

    private func entry(_ filename: String, mediaVersion: String? = "v1") -> LibraryEntry {
        let version = mediaVersion.map { "\"\($0)\"" } ?? "null"
        let json = """
        {"filename": "\(filename)", "metadata": {}, "timestamp": 1000,
         "media_version": \(version)}
        """
        let print = try! MoldJSON.decoder.decode(GalleryPrint.self, from: Data(json.utf8))
        return LibraryEntry(host: MoldHost(name: "plato", baseURL: URL(string: "http://p")!),
                            print: print)
    }

    @Test func theFileLandsInsideTheKeyedDirectoryUnderItsOwnName() async throws {
        let root = root()
        defer { try? FileManager.default.removeItem(at: root) }
        let materializer = PrintMaterializer(root: root)

        let url = try #require(await materializer.url(for: entry("robot.png")) {
            Data("bytes".utf8)
        })

        #expect(url.lastPathComponent == "robot.png")
        #expect(url.deletingLastPathComponent().deletingLastPathComponent()
            .standardizedFileURL == root.standardizedFileURL)
        #expect(FileManager.default.fileExists(atPath: url.path))
    }

    /// `media_version` is the machine's string too, and it is the only thing
    /// this app ever makes a DIRECTORY name out of. A print with an odd one is
    /// still a print, so it is folded rather than refused -- but it cannot
    /// name a directory anywhere else.
    @Test func aTraversingMediaVersionCannotEscapeTheCacheRoot() async throws {
        let root = root()
        defer { try? FileManager.default.removeItem(at: root) }
        let materializer = PrintMaterializer(root: root)

        let url = try #require(await materializer.url(
            for: entry("robot.png", mediaVersion: "../../../../escaped")
        ) { Data("bytes".utf8) })

        #expect(url.deletingLastPathComponent().deletingLastPathComponent()
            .standardizedFileURL == root.standardizedFileURL)
        #expect(!FileManager.default.fileExists(
            atPath: root.deletingLastPathComponent().appending(path: "escaped").path))
    }

    /// mold's own versions carry a colon, which the Finder renders as a slash.
    @Test func aColonInAVersionNeverReachesTheFileSystem() async throws {
        let root = root()
        defer { try? FileManager.default.removeItem(at: root) }
        let materializer = PrintMaterializer(root: root)

        let url = try #require(await materializer.url(
            for: entry("robot.png", mediaVersion: "sha256:abc")
        ) { Data("bytes".utf8) })

        #expect(!url.deletingLastPathComponent().lastPathComponent.contains(":"))
        #expect(url.deletingLastPathComponent().lastPathComponent.hasSuffix("sha256-abc"))
    }
}
