import AppKit
import MoldClient
import SwiftUI
import Testing
@testable import Mold

/// Isolated native windows backed only by fakes. Screenshots are inspectable
/// UAT evidence; this never launches or operates the installed app.
@MainActor struct LibrarySurfaceRenderingTests {
    @Test func scopedCollectionsAndSyncControlsRender() async throws {
        let local = MoldHost(name: "This Mac", baseURL: URL(string: "http://fixture-local")!)
        let remote = MoldHost(name: "Workstation", baseURL: URL(string: "http://fixture-remote")!)
        let hosts = HostStore(hosts: [local, remote]) { FakeBackend(host: $0) }
        hosts.reachability[local.id] = .up(FakeFixtures.serverStatus())
        hosts.reachability[remote.id] = .up(FakeFixtures.serverStatus())
        let session = LibrarySyncSession(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        let library = LibraryStore(hosts: hosts, syncSession: session)
        library.collectionInventoryAvailable = [local.id, remote.id]
        let shelf = try #require(CollectionShelf.merge([local.id: [Collection(id: "hidden", name: "Hidden drafts", slug: "hidden-drafts", hidden: true)] ]).first)
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(#"{"filename":"fixture.png","metadata":{},"timestamp":1000,"collections":["hidden"]}"#.utf8))
        library.items = [LibraryEntry(host: local, print: print)]
        let navigation = LibraryNavigation(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        navigation.query.tokens = [.machine(id: remote.id, name: remote.name)]
        try await capture(CollectionRow(shelf: shelf, renaming: .constant(nil))
            .environment(library).environment(navigation).padding(), name: "mac-library-absent-shelf", height: 100)
        navigation.query.tokens = [.machine(id: local.id, name: local.name)]
        try await capture(CollectionRow(shelf: shelf, renaming: .constant(nil))
            .environment(library).environment(navigation).padding(), name: "mac-library-hidden-shelf", height: 100)

        // An empty local fake completes without any media copy or network request.
        let destination = try #require(MoldEngine.localHost(port: 7680, apiKey: "fixture"))
        let backend = FakeBackend(host: destination)
        let emptyHosts = HostStore(hosts: [destination]) { _ in backend }
        emptyHosts.reachability[destination.id] = .up(FakeFixtures.serverStatus())
        let emptyLibrary = LibraryStore(hosts: emptyHosts, syncSession: session)
        session.start(in: emptyLibrary)
        await settle { session.nextRun != nil }
        defer { session.stop(in: emptyLibrary) }
        emptyLibrary.localSaveReport = "Sync complete. All prints are already saved on This Mac."
        try await capture(LibraryActivityStatus().environment(emptyLibrary), name: "mac-library-sync-complete-countdown", height: 180)
        try await capture(LibrarySyncReportSheet().environment(emptyLibrary), name: "mac-library-sync-clean-details", height: 180)
        emptyLibrary.localSaveFailures = ["Legacy fixture.mov: Original input is unavailable on the source machine."]
        emptyLibrary.localSaveIssueKeys = ["fixture": "fixture-unchanged-issue"]
        try await capture(LibrarySyncReportSheet().environment(emptyLibrary), name: "mac-library-sync-issue-acknowledgment", height: 500)
        #expect(!emptyLibrary.localSaveAlertPresented, "A successful sync does not demand acknowledgment")
        #expect(session.isEnabled)
        session.stop(in: emptyLibrary)
        #expect(!session.isEnabled)
    }

    private func capture<V: View>(_ content: V, name: String, height: CGFloat) async throws {
        let view = NSHostingView(rootView: content.background(Color(nsColor: .windowBackgroundColor)))
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 700, height: height),
            styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = view
        window.orderFront(nil)
        defer { window.close() }
        try await Task.sleep(for: .milliseconds(300))
        view.layoutSubtreeIfNeeded()
        view.needsDisplay = true
        view.displayIfNeeded()
        let bitmap = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
        view.cacheDisplay(in: view.bounds, to: bitmap)
        let data = try #require(bitmap.representation(using: .png, properties: [:]))
        try data.write(to: URL(fileURLWithPath: "/tmp/\(name).png"))
        #expect(bitmap.pixelsWide > 0 && bitmap.pixelsHigh > 0)
    }
}
