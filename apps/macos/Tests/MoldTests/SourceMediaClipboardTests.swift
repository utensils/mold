import AppKit
import Testing
@testable import Mold
@testable import MoldClient

@MainActor
struct SourceMediaClipboardTests {
    @Test func fileSurvivesViewLifetimeUntilThePasteboardChanges() async throws {
        let pasteboard = NSPasteboard(name: .init("mold-source-test-\(UUID())"))
        let clipboard = SourceMediaClipboard()
        var file: SourceMediaFile? = try SourceMediaFile(extension: "wav")
        let url = try #require(file?.url)
        clipboard.copy(try #require(file), pasteboard: pasteboard, isCurrent: { true })
        file = nil
        #expect(FileManager.default.fileExists(atPath: url.path))
        #expect(clipboard.file != nil)
        pasteboard.clearContents()
        try await Task.sleep(for: .milliseconds(500))
        #expect(clipboard.file == nil)
        #expect(!FileManager.default.fileExists(atPath: url.path))
        pasteboard.releaseGlobally()
    }
    @Test func normalTerminationUnlinksTheClipboardOriginal() throws {
        let center = NotificationCenter()
        let pasteboard = NSPasteboard(name: .init("mold-source-test-\(UUID())"))
        let clipboard = SourceMediaClipboard(center: center)
        var file: SourceMediaFile? = try SourceMediaFile(extension: "wav")
        let url = try #require(file?.url)
        clipboard.copy(try #require(file), pasteboard: pasteboard, isCurrent: { true })
        file = nil
        center.post(name: NSApplication.willTerminateNotification, object: nil)
        #expect(clipboard.file == nil)
        #expect(!FileManager.default.fileExists(atPath: url.path))
        pasteboard.releaseGlobally()
    }
    @Test func revokedAuthorityReleasesPrivateClipboardOriginal() async throws {
        let pasteboard = NSPasteboard(name: .init("mold-source-test-\(UUID())"))
        let clipboard = SourceMediaClipboard()
        var current = true
        var file: SourceMediaFile? = try SourceMediaFile(extension: "wav")
        let url = try #require(file?.url)
        clipboard.copy(try #require(file), pasteboard: pasteboard, isCurrent: { current })
        file = nil; current = false
        try await Task.sleep(for: .milliseconds(500))
        #expect(clipboard.file == nil)
        #expect(!FileManager.default.fileExists(atPath: url.path))
        pasteboard.releaseGlobally()
    }
}
