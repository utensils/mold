import Foundation
import MoldClient

@MainActor
extension HostStore {
    func applyConnection(_ updated: MoldHost, expectedURL: URL? = nil) {
        guard var current = host(updated.id), current.apiKey == updated.apiKey,
              expectedURL.map({ $0 == current.baseURL }) ?? true else { return }
        current.baseURL = updated.baseURL
        current.connectionEndpoints = updated.connectionEndpoints
        current.connectionInstanceID = updated.connectionInstanceID
        hosts = hosts.map { $0.id == updated.id ? current : $0 }
        HostPersistence.save(hosts.filter { $0.id != MoldEngine.localHostID })
    }

    func recordConnectionFailure(_ error: any Error, on id: MoldHost.ID) {
        guard host(id) != nil else { return }
        reachability[id] = .down(error.reasonSentence)
        reconcileEventStreams()
    }
}
