import AppKit
import MoldClient
import SwiftUI
import Testing

@testable import Mold

@MainActor
struct QueueSourceLayoutTests {
    @Test(arguments: [400.0, 800.0])
    func heldRecoveryControlsRenderAtNarrowAndWideWidths(width: Double) async throws {
        let host = MoldHost(name: "fixture", baseURL: URL(string: "http://fixture")!)
        let hosts = HostStore(hosts: [host])
        let downloads = DownloadStore(hosts: hosts, licenses: LicenseStore(hosts: hosts))
        let destination = TransferStore.TransferDestination(id: host.id, name: "Another machine", queueDepth: 3)
        let row = QueueHoldRow(entry: FakeFixtures.queueEntry("held-layout", state: "held", model: "MiniMax H3 Turbo"),
            hold: .missingModel("h3", sentence: "This model is not installed."),
            pullThenRetry: { _ in }, tryAgain: {}, moveToDestinations: [destination], moveTo: { _ in },
            cancel: {}, inspect: {}, actions: QueueRowActions(retry: true, cancel: true))
            .environment(downloads).padding(12).frame(width: width)
        let view = NSHostingView(rootView: row.background(Color(nsColor: .windowBackgroundColor)))
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: width, height: 200),
                              styleMask: [.borderless], backing: .buffered, defer: false)
        window.appearance = NSAppearance(named: .aqua)
        window.isReleasedWhenClosed = false
        window.contentView = view
        window.orderFront(nil)
        defer { window.close() }
        await settle { view.fittingSize.height >= 60 }
        window.setContentSize(CGSize(width: width, height: view.fittingSize.height))
        view.layoutSubtreeIfNeeded()
        #expect(abs(view.bounds.width - width) < 0.01)
        #expect(view.bounds.height >= 60)
        if width == 400 {
            let controls = NSHostingView(rootView: VStack(alignment: .leading, spacing: 8) {
                Button("Download and Retry") {}
                Menu("Move to") { Button("Another machine") {} }
                Button("Failure Details") {}
            }.buttonStyle(.bordered).controlSize(.small))
            let title = NSHostingView(rootView: Text("MiniMax H3 Turbo"))
            #expect(view.bounds.height >= controls.fittingSize.height + title.fittingSize.height + 24,
                    "Three complete recovery controls and the title must fit inside the padded row")
        }
        let bitmap = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
        view.cacheDisplay(in: view.bounds, to: bitmap)
        let bytes = try #require(bitmap.representation(using: .png, properties: [:]))
        try bytes.write(to: URL(fileURLWithPath: "/tmp/mold-mac-held-controls-\(Int(width)).png"))
    }

    @Test func delayedThumbnailCannotOutgrowInitialRowMeasurement() async throws {
        let host = MoldHost(name: "fixture", baseURL: URL(string: "http://fixture")!)
        let backend = FakeBackend(host: host)
        backend.queueThumbnailPending = true
        backend.queueThumbnailBytes = Data(base64Encoded:
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=")
        let hosts = HostStore(hosts: [host]) { _ in backend }
        hosts.reachability[host.id] = .up(FakeFixtures.serverStatus())
        let row = QueueRow(entry: FakeFixtures.queueEntry("source", state: "queued", model: "MiniMax H3 FL2VA"),
            actions: QueueRowActions(), caption: "First caption line\nSecond caption line", sourceHost: host, act: { _ in })
            .environment(hosts).frame(width: 600)
        let view = NSHostingView(rootView: row)
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 600, height: 200),
                              styleMask: [.borderless], backing: .buffered, defer: false)
        window.appearance = NSAppearance(named: .darkAqua)
        window.isReleasedWhenClosed = false
        window.contentView = view
        defer { backend.queueThumbnailPending = false; window.close() }
        view.layoutSubtreeIfNeeded()
        let sourceStack = NSHostingView(rootView: VStack(alignment: .leading, spacing: 3) {
            Color.clear.frame(width: 48, height: 48)
            Text("Source").font(.caption)
        }.padding(.vertical, 3))
        let required = sourceStack.fittingSize.height
        let initial = view.fittingSize.height
        backend.queueThumbnailPending = false
        await settle { backend.queueThumbnailReturned }
        // Rendering settles independently of the backend returning its bytes.
        await settle { view.fittingSize.height >= required }
        view.layoutSubtreeIfNeeded()
        let loaded = view.fittingSize.height
        #expect(loaded >= required) // Measure the rendered image, caption and row padding.
        #expect(initial >= loaded, "A native List may cache the initial row height; loading must not grow it")
    }
    @Test(arguments: [true, false])
    func nativeListContainsLoadedRowsAndCompactsMissingSources(hasSource: Bool) async throws {
        let host = MoldHost(name: "fixture", baseURL: URL(string: "http://fixture")!)
        let backend = FakeBackend(host: host)
        backend.queueThumbnailPending = true
        if hasSource {
            backend.queueThumbnailBytes = Data(base64Encoded:
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=")
        }
        let hosts = HostStore(hosts: [host]) { _ in backend }
        hosts.reachability[host.id] = .up(FakeFixtures.serverStatus())
        let content = List(selection: Binding<String?>.constant(nil)) {
            Section(host.name) {
                ForEach(0..<2) { index in
                    QueueRow(entry: FakeFixtures.queueEntry("job-\(index)", state: "queued", model: "MiniMax H3 FL2VA"),
                        actions: QueueRowActions(), caption: "First caption line\nSecond caption line", sourceHost: host,
                        isReorderable: true, inspect: {}, act: { _ in })
                        .tag("job-\(index)")
                }
                .onMove { _, _ in }
            }
        }.listStyle(.inset).environment(hosts).preferredColorScheme(.dark)
        let view = NSHostingView(rootView: content)
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 600, height: 300),
                              styleMask: [.borderless], backing: .buffered, defer: false)
        window.appearance = NSAppearance(named: .darkAqua)
        window.isReleasedWhenClosed = false
        window.contentView = view
        window.orderFront(nil)
        defer { backend.queueThumbnailPending = false; window.close() }
        await settle { self.table(in: view)?.numberOfRows == 3 }
        let table = try #require(self.table(in: view))
        let initial = table.rect(ofRow: 1).height
        #expect(!hasSourcePixels(in: view), "The pending placeholder must not satisfy the loaded-image criterion")
        backend.queueThumbnailPending = false
        await settle { backend.queueThumbnailReturned }
        let source = NSHostingView(rootView: VStack(alignment: .leading, spacing: 3) {
            Color.clear.frame(width: 48, height: 48)
            Text("Source").font(.caption)
        }.padding(.vertical, 3)).fittingSize.height
        await settle {
            let height = table.rect(ofRow: 1).height
            return hasSource ? (height >= source && self.hasSourcePixels(in: view)) : height < initial
        }
        let first = table.rect(ofRow: 1)
        let second = table.rect(ofRow: 2)
        #expect(first.maxY <= second.minY)
        if hasSource {
            #expect(initial >= source)
            #expect(hasSourcePixels(in: view), "Wait for the actual loaded white image, not its transparent pending placeholder")
            #expect(first.height >= source, "Image and its caption must fit inside the native table row")
            #expect(first.height == initial)
        } else {
            #expect(first.height < initial, "A missing source must return to the compact text-only row")
            let caption = NSHostingView(rootView: VStack(alignment: .leading, spacing: 2) {
                Text("MiniMax H3 FL2VA")
                Text("First caption line\nSecond caption line").font(.caption)
            }.padding(.vertical, 3)).fittingSize.height
            #expect(first.height >= caption, "Both caption lines must fit even without a source")
        }
    }

    /// The fixture is white. A dark native row cannot contain a tall white run
    /// until its transparent loading reservation becomes the actual image.
    func hasSourcePixels(in view: NSView) -> Bool {
        view.layoutSubtreeIfNeeded()
        guard let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return false }
        view.cacheDisplay(in: view.bounds, to: bitmap)
        for x in 0..<min(bitmap.pixelsWide, 120) {
            var run = 0
            for y in 0..<bitmap.pixelsHigh {
                let color = bitmap.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB)
                if let color, color.redComponent > 0.95, color.greenComponent > 0.95, color.blueComponent > 0.95 {
                    run += 1
                    // Image is 48px, with a 6px corner radius; body/caption glyphs
                    // cannot occupy this many contiguous vertical pixels.
                    if run >= 48 - 6 { return true }
                } else { run = 0 }
            }
        }
        return false
    }

    private func table(in view: NSView) -> NSTableView? {
        if let table = view as? NSTableView { return table }
        for child in view.subviews { if let table = self.table(in: child) { return table } }
        return nil
    }

}
