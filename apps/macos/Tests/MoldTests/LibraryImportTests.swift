import Foundation
import MoldClient
import Testing

@testable import Mold

/// Importing files from this Mac into a machine's library.
@MainActor
struct LibraryImportTests {
    private func machine(_ name: String = "plato") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    /// Three real files in a temporary folder, with one name in the middle
    /// that does not exist -- the unreadable pick, without needing a mounted
    /// volume to go away mid-import.
    private func batch() throws -> (urls: [URL], folder: URL) {
        let folder = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "mold-import-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var urls: [URL] = []
        for name in ["one.png", "two.png", "three.png"] {
            let url = folder.appending(path: name)
            try Data("not really a picture, and it does not have to be".utf8).write(to: url)
            urls.append(url)
        }
        urls.insert(folder.appending(path: "gone.png"), at: 1)
        return (urls, folder)
    }

    /// **Fails today**: the read failure `return`s out of the loop, so one
    /// unreadable file in the middle of a ten-file pick imports the ones
    /// before it and NEVER ATTEMPTS the rest. The silence it replaced was
    /// "nine imported, no message"; this is one imported and a toast that
    /// looks like the whole import is accounted for.
    @Test func oneUnreadableFileDoesNotAbandonTheBatch() async throws {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let library = LibraryStore(hosts: hosts)
        let actions = LibraryActions(hosts: hosts, library: library)
        let (urls, folder) = try batch()
        defer { try? FileManager.default.removeItem(at: folder) }

        await actions.send(urls, to: plato)

        // Every file that could be read was sent, in order, and the one that
        // could not is the only thing reported -- AFTER the batch, because
        // every successful import clears that machine's failures, so a report
        // made mid-loop is wiped by the next file that works.
        #expect(fake.importedNames == ["one.png", "two.png", "three.png"])
        #expect(hosts.failures.count == 1)
        #expect(hosts.failures.first?.verb.contains("gone.png") == true)
    }

    /// Several unreadable files are one line, not one line per file -- and it
    /// says how many, because naming only the first would under-report what
    /// did not arrive.
    @Test func severalUnreadableFilesAreOneLineThatSaysHowMany() async throws {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let library = LibraryStore(hosts: hosts)
        let actions = LibraryActions(hosts: hosts, library: library)
        let (urls, folder) = try batch()
        defer { try? FileManager.default.removeItem(at: folder) }

        await actions.send(urls + [folder.appending(path: "also-gone.png")], to: plato)

        #expect(fake.importedNames == ["one.png", "two.png", "three.png"])
        #expect(hosts.failures.count == 1)
        #expect(hosts.failures.first?.verb == "import 2 of those files")
    }

    /// A failure to UPLOAD is about the MACHINE, not about that one file, so
    /// it stops -- the distinction the read's `continue` rests on.
    @Test func aRefusedUploadStopsTheBatch() async throws {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        fake.refuses = ["importPrint"]
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let library = LibraryStore(hosts: hosts)
        let actions = LibraryActions(hosts: hosts, library: library)
        let (urls, folder) = try batch()
        defer { try? FileManager.default.removeItem(at: folder) }

        await actions.send(urls, to: plato)

        #expect(fake.callCount("importPrint") == 1)
        #expect(hosts.failures.count == 1)
    }
}
