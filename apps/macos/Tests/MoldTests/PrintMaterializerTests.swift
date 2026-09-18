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
    /// What a preview panel would be holding, without opening one.
    @MainActor private final class Held {
        var urls: [URL] = []
    }

    private func root() -> URL {
        FileManager.default.temporaryDirectory
            .appending(path: "mold-materializer-\(UUID().uuidString)")
    }

    private func entry(_ filename: String, mediaVersion: String? = "v1",
                       host: UUID = UUID()) -> LibraryEntry {
        let version = mediaVersion.map { "\"\($0)\"" } ?? "null"
        let json = """
        {"filename": "\(filename)", "metadata": {}, "timestamp": 1000,
         "media_version": \(version)}
        """
        let print = try! MoldJSON.decoder.decode(GalleryPrint.self, from: Data(json.utf8))
        return LibraryEntry(
            host: MoldHost(id: host, name: "workstation", baseURL: URL(string: "http://p")!),
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

    // MARK: - The budget

    /// **Fails today**: `url(for:fetch:)` runs `enforceBudget()` between the
    /// write and the return (`PrintMaterializer.swift:74`), and
    /// `CacheBudget.evictions` correctly refuses to keep a file bigger than
    /// the whole cap -- so the clip is downloaded, written, deleted, and a URL
    /// to the deleted file is handed back. Every consumer then fails with no
    /// message.
    @Test func aPrintTooBigForTheCacheIsStillHandedOverAndSaidOutLoud() async throws {
        let root = root()
        defer { try? FileManager.default.removeItem(at: root) }
        AppStorageSuite.defaults.set(0, forKey: PrintMaterializer.capKey)
        defer { AppStorageSuite.defaults.removeObject(forKey: PrintMaterializer.capKey) }
        let materializer = PrintMaterializer(root: root)

        let url = try #require(await materializer.url(for: entry("clip.mp4")) {
            Data(repeating: 7, count: 4_096)
        })

        #expect(FileManager.default.fileExists(atPath: url.path))
        let note = try #require(materializer.note)
        #expect(note.contains("clip.mp4"))
    }

    /// The panel reads its item's URL lazily, from its own queues.
    @Test func aFileQuickLookIsShowingIsNotEvictedUnderIt() async throws {
        let root = root()
        defer { try? FileManager.default.removeItem(at: root) }
        let held = Held()
        let materializer = PrintMaterializer(root: root, inUse: { held.urls })
        let open = try #require(await materializer.url(for: entry("open.png")) {
            Data(repeating: 1, count: 2_048)
        })
        held.urls = [open]

        AppStorageSuite.defaults.set(0, forKey: PrintMaterializer.capKey)
        defer { AppStorageSuite.defaults.removeObject(forKey: PrintMaterializer.capKey) }
        materializer.enforceBudget()

        #expect(FileManager.default.fileExists(atPath: open.path))
    }

    /// An ordinary print, with nothing holding it, still goes when the cache
    /// is over its cap -- the sparing is about what is in use, not a reprieve.
    @Test func aPrintNothingIsHoldingIsStillEvicted() async throws {
        let root = root()
        defer { try? FileManager.default.removeItem(at: root) }
        let materializer = PrintMaterializer(root: root)
        let old = try #require(await materializer.url(for: entry("old.png")) {
            Data(repeating: 1, count: 2_048)
        })

        AppStorageSuite.defaults.set(0, forKey: PrintMaterializer.capKey)
        defer { AppStorageSuite.defaults.removeObject(forKey: PrintMaterializer.capKey) }
        materializer.enforceBudget()

        #expect(!FileManager.default.fileExists(atPath: old.path))
    }

    /// **Fails today**: a spared entry is simply skipped, and nothing else is
    /// evicted to make up for it -- so once anything is held the cache sits
    /// over its cap as a stable state rather than a transient one.
    @Test func whatIsSparedComesOffTheBudgetRatherThanOutOfTheReckoning() async throws {
        let root = root()
        defer { try? FileManager.default.removeItem(at: root) }
        let held = Held()
        let materializer = PrintMaterializer(root: root, inUse: { held.urls })
        // Two of these do not fit in one megabyte; one does.
        let open = try #require(await materializer.url(for: entry("open.png")) {
            Data(repeating: 1, count: 700_000)
        })
        let other = try #require(await materializer.url(for: entry("other.png")) {
            Data(repeating: 1, count: 700_000)
        })
        held.urls = [open]

        // Room for one of the two. The held one is unavoidable, so the cap
        // that is left over is nothing -- and the other has to go.
        AppStorageSuite.defaults.set(1, forKey: PrintMaterializer.capKey)
        defer { AppStorageSuite.defaults.removeObject(forKey: PrintMaterializer.capKey) }
        materializer.enforceBudget()

        #expect(FileManager.default.fileExists(atPath: open.path))
        #expect(!FileManager.default.fileExists(atPath: other.path))
    }

    /// **Fails today**: `QuickLook.items` is only ever replaced, so every
    /// folder of the last preview stays pinned against eviction for the rest
    /// of the process -- panel shut or not.
    @Test func quickLookLettingGoUnpinsWhatItWasShowing() async throws {
        let root = root()
        defer { try? FileManager.default.removeItem(at: root) }
        let materializer = PrintMaterializer(root: root,
                                             inUse: { QuickLook.shared.heldURLs })
        let file = try #require(await materializer.url(for: entry("open.png")) {
            Data(repeating: 1, count: 2_048)
        })
        QuickLook.shared.release()

        AppStorageSuite.defaults.set(0, forKey: PrintMaterializer.capKey)
        defer { AppStorageSuite.defaults.removeObject(forKey: PrintMaterializer.capKey) }
        materializer.enforceBudget()

        #expect(QuickLook.shared.heldURLs.isEmpty)
        #expect(!FileManager.default.fileExists(atPath: file.path))
    }

    /// **Fails today**: `contents` takes `files.first` and reports THAT file's
    /// size as the whole folder's (`PrintMaterializer+Budget.swift:24`). The
    /// folder key excludes the filename -- it is the host and the folded
    /// `media_version`, or the TIMESTAMP where a host sends none -- so a batch
    /// published in one second shares a folder and the cache under-reports by
    /// the size of the batch. "Using" in Settings lies by the same factor and
    /// the cap is never reached.
    @Test func afolderHoldingSeveralPrintsIsMeasuredWhole() async throws {
        let root = root()
        defer { try? FileManager.default.removeItem(at: root) }
        let materializer = PrintMaterializer(root: root)
        // Same host, same version -- one folder, the collision the materializer
        // documents and the flight key already guards against.
        let workstation = UUID()
        for name in ["a.png", "b.png", "c.png"] {
            _ = await materializer.url(for: entry(name, mediaVersion: "shared", host: workstation)) {
                Data(repeating: 1, count: 1_000)
            }
        }

        #expect(materializer.contents.count == 1)
        #expect(materializer.usedBytes == 3_000)
    }

    /// A folder with nothing measurable in it -- what a failed write leaves
    /// behind -- was invisible to accounting AND to eviction, so it was never
    /// cleaned up.
    @Test func anEmptyFolderIsAccountedForAndSweptAway() throws {
        let root = root()
        defer { try? FileManager.default.removeItem(at: root) }
        let materializer = PrintMaterializer(root: root)
        let stray = root.appending(path: "stray-folder")
        try FileManager.default.createDirectory(at: stray, withIntermediateDirectories: true)

        #expect(materializer.contents.map(\.name) == ["stray-folder"])
        #expect(materializer.usedBytes == 0)

        materializer.enforceBudget()
        #expect(!FileManager.default.fileExists(atPath: stray.path))
    }

    /// **Fails today**: `contents` prefers `contentAccessDate`
    /// (`PrintMaterializer+Budget.swift:28`) while `touch` writes only the
    /// MODIFICATION date, so on APFS -- which does keep access dates -- the
    /// hand-maintained recency signal was never the one read.
    @Test func touchingAFileIsWhatDecidesHowRecentlyItWasUsed() async throws {
        let root = root()
        defer { try? FileManager.default.removeItem(at: root) }
        let materializer = PrintMaterializer(root: root)
        let older = try #require(await materializer.url(for: entry("older.png")) {
            Data(repeating: 1, count: 8)
        })
        _ = try #require(await materializer.url(for: entry("newer.png")) {
            Data(repeating: 1, count: 8)
        })

        // Deliberately BACKDATED: the file was just written, so only `touch`'s
        // own attribute can make it look like the older of the two.
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: 1_000)],
            ofItemAtPath: older.path)

        let oldestName = materializer.contents.min { $0.lastUsed < $1.lastUsed }?.name
        #expect(oldestName == older.deletingLastPathComponent().lastPathComponent)
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
