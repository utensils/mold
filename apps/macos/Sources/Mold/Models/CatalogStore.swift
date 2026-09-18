import Foundation
import MoldClient

/// What each machine's catalog browser is showing right now.
///
/// Per host, and never merged: a search against workstation says nothing about
/// hal9000's own catalog reach. The upstream proxy already caches a search
/// for five minutes in-process (`mold-catalog`'s live layer), so this store
/// re-asks freely on every filter change rather than building a second cache
/// of its own -- it only remembers the CURRENT query and its results, not
/// every query asked this launch.
@MainActor
@Observable
final class CatalogStore {
    // Not `private`: `CatalogStore+Credentials` (Settings ▸ Accounts, S7)
    // reads and writes both from a second file, and `private` does not cross
    // a file boundary even within one type.
    struct HostState {
        var query = CatalogQuery(includeNSFW: false)
        /// The most recent page answered, kept for its `total` and
        /// `providerErrors` -- refreshed to whichever page last answered, so
        /// a provider that failed on page 1 and recovered by page 2 is
        /// reported accurately.
        var listing: CatalogListing?
        /// Every entry seen for the CURRENT query, across every page fetched
        /// so far -- reset wholesale on a fresh search, appended to by
        /// `more(on:)`.
        var entries: [CatalogEntry] = []
        var isSearching = false
        var searchTask: Task<Void, Never>?
        var credentials: CatalogCredentialStatus?
    }

    let hosts: HostStore
    var byHost: [MoldHost.ID: HostState] = [:]

    init(hosts: HostStore) {
        self.hosts = hosts
    }

    func query(on host: MoldHost.ID) -> CatalogQuery { byHost[host]?.query ?? CatalogQuery(includeNSFW: false) }
    func entries(on host: MoldHost.ID) -> [CatalogEntry] { byHost[host]?.entries ?? [] }
    func total(on host: MoldHost.ID) -> Int { byHost[host]?.listing?.total ?? 0 }
    func isSearching(on host: MoldHost.ID) -> Bool { byHost[host]?.isSearching ?? false }
    func credentials(on host: MoldHost.ID) -> CatalogCredentialStatus? { byHost[host]?.credentials }

    /// `false` until this host's FIRST search answers -- the pane's Discover
    /// subtitle says nothing rather than a fabricated "0 results" before
    /// then (design S6b).
    func hasAnswered(on host: MoldHost.ID) -> Bool { byHost[host]?.listing != nil }

    /// The query's sort starts unset, which draws the Sort `Picker` with
    /// nothing selected. Seeded from the machine's own first advertised
    /// option the moment this host is adopted, and left alone once a
    /// person (or an earlier adoption) has already chosen one (design S6b).
    func adopt(_ host: MoldHost.ID, sortOptions: [String]) {
        guard byHost[host]?.query.sort == nil, let first = sortOptions.first else { return }
        byHost[host, default: HostState()].query.sort = first
    }

    /// One provider being down while the other answered is a PARTIAL
    /// SUCCESS, not a failure -- shown as a note above the rows it did get,
    /// never through `hosts.failures` (design S6).
    func providerErrors(on host: MoldHost.ID) -> [CatalogProviderError] { byHost[host]?.listing?.providerErrors ?? [] }

    func hasMore(on host: MoldHost.ID) -> Bool { entries(on: host).count < total(on: host) }

    func setText(_ text: String, on host: MoldHost.ID) {
        mutateQuery(on: host) { $0.text = text.isEmpty ? nil : text }
    }

    func setFamily(_ family: String?, on host: MoldHost.ID) {
        mutateQuery(on: host) { $0.family = family }
    }

    func setSort(_ sort: String?, on host: MoldHost.ID) {
        mutateQuery(on: host) { $0.sort = sort }
    }

    private func mutateQuery(on host: MoldHost.ID, _ change: (inout CatalogQuery) -> Void) {
        var state = byHost[host] ?? HostState()
        change(&state.query)
        state.query.page = nil
        byHost[host] = state
        search(on: host)
    }

    /// Debounced 350ms, the placement precedent
    /// (`GenerateController.refreshPlacement`), and cancellation-safe: a
    /// filter changed again before this fired, and that is the app changing
    /// its mind, not a failed request.
    func search(on host: MoldHost.ID) {
        byHost[host, default: HostState()].searchTask?.cancel()
        guard let client = hosts.backend(for: host) else { return }
        let query = byHost[host]?.query ?? CatalogQuery(includeNSFW: false)
        byHost[host, default: HostState()].isSearching = true
        byHost[host]?.searchTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(350))
            guard !Task.isCancelled else { return }
            guard let self else { return }
            do {
                let listing = try await client.searchCatalog(query)
                // A later filter change may have moved this host's query on
                // while this was in flight -- an answer to a stale question
                // must not overwrite what is now asked for.
                guard self.byHost[host]?.query == query else { return }
                self.byHost[host]?.listing = listing
                self.byHost[host]?.entries = listing.entries
                self.byHost[host]?.isSearching = false
                self.hosts.succeeded(on: host, doing: "search the catalog")
            } catch is CancellationError {
                // Superseded by a later filter change, not a failed request.
            } catch {
                self.byHost[host]?.isSearching = false
                self.hosts.report(error, on: host, doing: "search the catalog")
            }
        }
    }

    /// The next page, appended to what this query already holds.
    func more(on host: MoldHost.ID) async {
        guard let client = hosts.backend(for: host), var state = byHost[host], let listing = state.listing
        else { return }
        guard state.entries.count < listing.total else { return }
        var next = state.query
        next.page = listing.page + 1
        do {
            let page = try await client.searchCatalog(next)
            state.entries += page.entries
            state.listing = page
            state.query = next
            byHost[host] = state
            hosts.succeeded(on: host, doing: "search the catalog")
        } catch {
            hosts.report(error, on: host, doing: "search the catalog")
        }
    }
}
