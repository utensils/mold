import Foundation
import MoldClient

/// Discover (DESIGN.md §5.5): the machine's catalog proxy, searched as you
/// type (debounced), filtered by family and sorted by what the machine
/// offers, one page at a time.
@Observable
final class CatalogStore {
    struct State {
        var query = CatalogQuery(includeNSFW: false)
        var listing: CatalogListing?
        var entries: [CatalogEntry] = []
        var isSearching = false
    }

    private(set) var byHost: [MoldHost.ID: State] = [:]
    @ObservationIgnored let hosts: HostStore
    @ObservationIgnored private var searches: [MoldHost.ID: Task<Void, Never>] = [:]

    init(hosts: HostStore) {
        self.hosts = hosts
    }

    func state(on id: MoldHost.ID) -> State { byHost[id] ?? State() }
    func hasMore(on id: MoldHost.ID) -> Bool { state(on: id).entries.count < (state(on: id).listing?.total ?? 0) }

    /// The machine's first sort option, once, before the first search.
    func start(on id: MoldHost.ID) {
        if byHost[id]?.query.sort == nil, let first = hosts.capabilities[id]?.catalogSortOptions.first {
            byHost[id, default: State()].query.sort = first
        }
        if byHost[id]?.listing == nil { search(on: id, debounce: false) }
    }

    func setText(_ text: String, on id: MoldHost.ID) { change(id) { $0.text = text.isEmpty ? nil : text } }
    func setFamily(_ family: String?, on id: MoldHost.ID) { change(id) { $0.family = family } }
    func setSort(_ sort: String?, on id: MoldHost.ID) { change(id) { $0.sort = sort } }

    private func change(_ id: MoldHost.ID, _ edit: (inout CatalogQuery) -> Void) {
        var state = byHost[id] ?? State()
        let before = state.query
        edit(&state.query)
        state.query.page = nil
        byHost[id] = state
        if state.query != before { search(on: id, debounce: true) }
    }

    func search(on id: MoldHost.ID, debounce: Bool) {
        searches[id]?.cancel()
        guard let host = hosts.host(id) else { return }
        let client = hosts.backend(for: host)
        let query = state(on: id).query
        byHost[id, default: State()].isSearching = true
        searches[id] = Task { [weak self] in
            if debounce { try? await Task.sleep(for: .milliseconds(350)) }
            guard !Task.isCancelled, let self else { return }
            do {
                let listing = try await client.searchCatalog(query)
                guard self.byHost[id]?.query == query else { return }
                self.byHost[id]?.listing = listing
                self.byHost[id]?.entries = listing.entries
                self.hosts.clearFailures(for: id, doing: String(localized: "search the catalog"))
            } catch is CancellationError {
                return
            } catch {
                self.hosts.report(host, doing: String(localized: "search the catalog"), error)
            }
            self.byHost[id]?.isSearching = false
        }
    }

    /// The next page, appended.
    func more(on id: MoldHost.ID) async {
        guard let host = hosts.host(id), var state = byHost[id], let listing = state.listing,
              state.entries.count < listing.total, !state.isSearching else { return }
        var next = state.query
        next.page = listing.page + 1
        do {
            let page = try await hosts.backend(for: host).searchCatalog(next)
            state.entries += page.entries.filter { entry in !state.entries.contains { $0.id == entry.id } }
            state.listing = page
            state.query = next
            byHost[id] = state
        } catch {
            hosts.report(host, doing: String(localized: "search the catalog"), error)
        }
    }
}
