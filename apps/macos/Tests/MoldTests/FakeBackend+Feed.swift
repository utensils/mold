import Foundation
import MoldClient
@testable import Mold

/// A scriptable change log behind `galleryChanges(since:limit:)`.
///
/// Off by default, so a machine answers 404 exactly as an older host does and
/// every existing test keeps the listing path. With `serving` on, each request
/// first marks every name whose planted row (`prints` + `trashedRows`) differs
/// from what the log last saw -- so a test scripts the machine by editing the
/// same arrays it always has -- then pages the dirty set from the cursor.
/// Cursors are `"<epoch>:<seq>"`; another epoch answers `reset: true`.
struct FakeFeedLog {
    var serving = false
    var instanceID = "fake-instance"
    var epoch = 1
    var seq = 0
    var dirty: [String: Int] = [:]
    var seen: [String: GalleryPrint] = [:]
    /// The `since` of every page asked, in order.
    var requested: [String?] = []
}

extension FakeBackend {
    /// Turns the log on and advertises it, so the store follows the feed.
    @MainActor
    func serveFeed(on hosts: HostStore) {
        feed.serving = true
        let json = #"{"gallery":{"changes_feed":{"protocol_version":1}}}"#
        hosts.capabilities[host.id] = try! MoldJSON.decoder.decode(Capabilities.self, from: Data(json.utf8))
    }

    func galleryChanges(since cursor: String?, limit: Int) async throws -> GalleryChangesPage {
        try record("galleryChanges")
        await pause("galleryChanges")
        guard feed.serving else { throw MoldClientError.http(status: 404, code: nil, message: nil) }
        feed.requested.append(cursor)
        var current: [String: GalleryPrint] = [:]
        for row in prints + trashedRows { current[row.filename] = row }
        for name in Set(current.keys).union(feed.seen.keys) where current[name] != feed.seen[name] {
            feed.seq += 1
            feed.dirty[name] = feed.seq
        }
        feed.seen = current
        var after = 0
        var reset = false
        if let cursor {
            let parts = cursor.split(separator: ":").compactMap { Int($0) }
            if parts.count == 2, parts[0] == feed.epoch, parts[1] <= feed.seq { after = parts[1] } else { reset = true }
        }
        let pending = feed.dirty.filter { $0.value > after }.sorted { $0.value < $1.value }
        let page = pending.prefix(max(limit, 1))
        let more = pending.count > page.count
        let changes = page.map { name, _ in
            current[name].map { GalleryChange(filename: name, image: $0) }
                ?? GalleryChange(filename: name, removed: true)
        }
        return GalleryChangesPage(instanceId: feed.instanceID,
                                  cursor: "\(feed.epoch):\(more ? page.last!.value : feed.seq)",
                                  more: more, reset: reset, changes: changes)
    }
}
