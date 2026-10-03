import Foundation
import MoldClient

/// A sheet owns one machine's history. Server search reaches older rows too.
@Observable final class PromptHistoryStore {
    enum State { case loading, ready, offline, unavailable, failed }
    private(set) var state: State = .loading
    private(set) var entries: [HistoryEntry] = []
    private(set) var clearing = false
    private(set) var message: String?
    let host: MoldHost
    @ObservationIgnored private let hosts: HostStore
    @ObservationIgnored private var generation = 0
    @ObservationIgnored private var requestedQuery = ""
    @ObservationIgnored private var loadedQuery: String?

    init(host: MoldHost, hosts: HostStore) { self.host = host; self.hosts = hosts }

    func load(query: String) async {
        requestedQuery = query
        guard !clearing else { return }
        generation += 1
        let token = generation
        if loadedQuery != query { entries = [] }
        guard let current = hosts.host(host.id), hosts.isUp(current) else { state = .offline; return }
        state = .loading; message = nil
        do {
            let listing = try await hosts.backend(for: current).history(limit: 50, query: query)
            guard token == generation, !Task.isCancelled else { return }
            var seen: Set<String> = []
            entries = listing.entries.filter { seen.insert($0.id).inserted }
            loadedQuery = query; state = .ready
        } catch is CancellationError {
            return
        } catch {
            guard token == generation, !Task.isCancelled else { return }
            if case let MoldClientError.http(status, code, _) = error, status == 503, code == "HISTORY_UNAVAILABLE" {
                state = .unavailable
            } else { state = .failed; message = error.localizedDescription }
        }
    }

    func clear(query: String) async {
        guard !clearing, let current = hosts.host(host.id), hosts.isUp(current) else { return }
        requestedQuery = query
        generation += 1; clearing = true
        do {
            try await hosts.backend(for: current).clearHistory(keeping: nil)
            entries = []; loadedQuery = nil
            clearing = false
            await load(query: requestedQuery)
        } catch {
            clearing = false; state = .failed; message = error.localizedDescription
        }
    }

    static func recall(_ prompt: String, into draft: inout RenderDraft) { draft.prompt = prompt }
}
