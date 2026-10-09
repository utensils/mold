import Foundation
import MoldClient

@MainActor extension LibraryStore {
    func refreshUnreadMedia() {
        let previous = unreadMedia
        unreadMedia.retainHosts(Set(hosts.hosts.map(\.id)))
        let query = LibraryScope.all.resolve(LibraryQuery(), shelves: shelves, hiddenCollectionIDs: hiddenCollectionIDs)
        unreadMedia.observe(entries: items, visible: query.apply(to: items),
                            loadedHosts: Set(perHost.keys), presentHosts: Set(hosts.hosts.map(\.id)))
        if unreadMedia != previous { unreadMedia.save(to: readDefaults) }
        if unreadMedia.count != previous.count { unreadCountChanged?(unreadCount) }
    }
}
