import AppKit
import MoldClient
import SwiftUI
import Testing

@testable import Mold

@MainActor
struct QueueBatchLayoutTests {
    @Test(arguments: [false, true])
    func batchChildrenRemainSeparateNativeListRows(expanded: Bool) async throws {
        let host = MoldHost(name: "fixture", baseURL: URL(string: "http://fixture")!)
        let backend = FakeBackend(host: host)
        backend.queueThumbnailBytes = Data(base64Encoded:
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=")
        let hosts = HostStore(hosts: [host]) { _ in backend }
        hosts.reachability[host.id] = .up(FakeFixtures.serverStatus())
        let queue = QueueStore(hosts: hosts)
        let downloads = DownloadStore(hosts: hosts, licenses: LicenseStore(hosts: hosts))
        let transfers = TransferStore(hosts: hosts, queue: queue)
        let entries = (0..<2).map { (index: Int) in
            FakeFixtures.queueEntry("child-\(index)", batchId: "batch", batchIndex: index, model: "MiniMax H3 FL2VA")
        }
        let group = try #require(QueueGroup.build(entries, children: [:]).first)
        var selection: String?
        let content = List(selection: Binding(get: { selection }, set: { selection = $0 })) {
            Section(host.name) {
                ForEach([group]) { group in
                    QueueBatchRow(group: group, sourceHost: host, actions: QueueRowActions(),
                        childActions: { _ in QueueRowActions() }, rowAct: { _, _ in }, groupAct: { _ in },
                        expanded: expanded)
                        .tag(group.id)
                }.onMove { _, _ in }
            }
        }.listStyle(.inset).environment(hosts).environment(queue)
            .environment(downloads).environment(transfers).preferredColorScheme(.dark)
        let view = NSHostingView(rootView: content)
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 600, height: 450),
                              styleMask: [.borderless], backing: .buffered, defer: false)
        window.appearance = NSAppearance(named: .darkAqua)
        window.isReleasedWhenClosed = false
        window.contentView = view
        window.orderFront(nil)
        defer { window.close() }
        await settle { self.table(in: view)?.numberOfRows == (expanded ? 4 : 2) }
        let table = try #require(self.table(in: view))
        #expect(table.numberOfRows == (expanded ? 4 : 2), "Each expanded child needs its own selectable native row")
        await settle { QueueSourceLayoutTests().hasSourcePixels(in: view) }
        #expect(QueueSourceLayoutTests().hasSourcePixels(in: view))
        let source = NSHostingView(rootView: VStack(alignment: .leading, spacing: 3) {
            Color.clear.frame(width: 48, height: 48)
            Text("Source").font(.caption)
        }.padding(.vertical, 3)).fittingSize.height
        for index in 1..<table.numberOfRows {
            #expect(table.rect(ofRow: index).height >= source)
            if index > 1 { #expect(table.rect(ofRow: index - 1).maxY <= table.rect(ofRow: index).minY) }
        }
        if expanded {
            table.selectRowIndexes(IndexSet(integer: 2), byExtendingSelection: false)
            await settle { selection == "child-0" }
            #expect(selection == "child-0", "Selection must identify the child, not only its parent batch")
        }
    }

    private func table(in view: NSView) -> NSTableView? {
        if let table = view as? NSTableView { return table }
        for child in view.subviews { if let table = self.table(in: child) { return table } }
        return nil
    }
}
