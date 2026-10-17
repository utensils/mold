import Foundation
import MoldClient

// Re-reading ONE machine's prints.
//
// `refresh()` asks every machine, which is right when the pane is opened or
// somebody presses Refresh. It is not right after a single machine published
// a single print: desktop re-reads only the owning host
// (`gallery.refreshHost(authority.sourceKey)`), and on a three-machine fleet
// the fleet-wide read is three listings for one finished clip.
@MainActor
extension LibraryStore {
    func refresh(on host: MoldHost.ID) async {
        guard let machine = hosts.host(host) else { return }
        let client = hosts.backend(for: machine)
        do {
            try await replica(for: machine).pull(pullRequest(machine, .live), using: client)
            if hosts.host(host) == machine { hosts.succeeded(on: host, doing: "list its prints") }
        } catch {
            if hosts.host(host) == machine { hosts.report(error, on: host, doing: "list its prints") }
        }
        settleTimelines()
    }
}
