import AppKit
import MoldClient
import SwiftUI
import Testing
@testable import Mold

@MainActor
struct QueueScrollStabilityTests {
    @Test func missingInputsDoNotMoveTheBottomViewport() async throws {
        let host = MoldHost(name: "fixture", baseURL: URL(string: "http://fixture")!)
        let backend = FakeBackend(host: host)
        backend.queueThumbnailPending = true
        let hosts = HostStore(hosts: [host]) { _ in backend }
        hosts.reachability[host.id] = .up(FakeFixtures.serverStatus())
        let content = List {
            ForEach(0..<80) { index in
                QueueRow(entry: FakeFixtures.queueEntry("scroll-\(index)", state: "queued"),
                         actions: QueueRowActions(), sourceHost: host, act: { _ in })
            }
        }.listStyle(.inset).environment(hosts)
        let view = NSHostingView(rootView: content)
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 650, height: 450),
                              styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = view
        window.orderFront(nil)
        defer { backend.queueThumbnailPending = false; window.close() }
        await settle { table(in: view)?.numberOfRows == 80 }
        let list = try #require(table(in: view))
        list.scrollRowToVisible(79)
        try await Task.sleep(for: .milliseconds(150))
        view.layoutSubtreeIfNeeded()
        let clip = try #require(list.enclosingScrollView?.contentView)
        let before = clip.bounds.origin.y
        let height = list.rect(ofRow: 79).height
        #expect(before > 0)
        backend.queueThumbnailPending = false
        await settle { backend.queueThumbnailReturned }
        try await Task.sleep(for: .milliseconds(200))
        view.layoutSubtreeIfNeeded()
        #expect(abs(list.rect(ofRow: 79).height - height) < 1)
        #expect(abs(clip.bounds.origin.y - before) < 1,
                "Resolving absent source previews must not change the bottom scroll offset")
    }

    private func table(in view: NSView) -> NSTableView? {
        if let table = view as? NSTableView { return table }
        return view.subviews.lazy.compactMap { table(in: $0) }.first
    }
}
