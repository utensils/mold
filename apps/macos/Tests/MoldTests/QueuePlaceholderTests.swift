import AppKit
import MoldClient
import MoldStyle
import SwiftUI
import Testing
@testable import Mold

@MainActor struct QueuePlaceholderTests {
    @Test(arguments: [false, true])
    func sourceFreeQueueRowKeepsAVisiblePreviewSlot(dark: Bool) async throws {
        let host = MoldHost(name: "fixture", baseURL: URL(string: "http://fixture")!)
        let backend = FakeBackend(host: host)
        let hosts = HostStore(hosts: [host]) { _ in backend }
        hosts.reachability[host.id] = .up(FakeFixtures.serverStatus())
        let entry = try MoldJSON.decoder.decode(QueueEntry.self, from: Data(#"{"id":"empty","state":"queued","model":"fixture"}"#.utf8))
        let view = NSHostingView(rootView: QueueSourceThumbnail(entry: entry, host: host)
            .environment(hosts).environment(QueueStore(hosts: hosts)))
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 80, height: 90),
            styleMask: [.borderless], backing: .buffered, defer: false)
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        window.isReleasedWhenClosed = false
        window.contentView = view
        window.orderFront(nil)
        defer { window.close() }
        try await Task.sleep(for: .milliseconds(150))
        view.layoutSubtreeIfNeeded()
        // Compare against transparent content: the old empty slot was hidden
        // with opacity(0), even though its layout space was reserved.
        let bitmap = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
        view.cacheDisplay(in: view.bounds, to: bitmap)
        var visible = 0
        for x in 0..<bitmap.pixelsWide {
            for y in 0..<bitmap.pixelsHigh {
                if (bitmap.colorAt(x: x, y: y)?.alphaComponent ?? 0) > 0.02 { visible += 1 }
            }
        }
        #expect(visible > 200, "A source-free job needs a quiet visible placeholder")
        try #require(bitmap.representation(using: .png, properties: [:]))
            .write(to: URL(fileURLWithPath: "/tmp/mold-queue-placeholder-\(dark ? "dark" : "light").png"))
    }

    @Test(arguments: [false, true])
    func placeholderStatesRenderInBothAppearances(dark: Bool) throws {
        let content = HStack(spacing: 24) {
            ForEach(MediaPreviewPlaceholder.State.allCases, id: \.self) { state in
                VStack(spacing: 8) {
                    MediaPreviewPlaceholder(state: state).frame(width: 48, height: 48)
                    Text(state.label).font(.caption)
                }
            }
        }
        .padding(24)
        .background(.background)
        .environment(\.colorScheme, dark ? .dark : .light)
        let renderer = ImageRenderer(content: content)
        let bitmap = NSBitmapImageRep(cgImage: try #require(renderer.cgImage))
        try #require(bitmap.representation(using: .png, properties: [:]))
            .write(to: URL(fileURLWithPath: "/tmp/mold-queue-states-\(dark ? "dark" : "light").png"))
    }

}
