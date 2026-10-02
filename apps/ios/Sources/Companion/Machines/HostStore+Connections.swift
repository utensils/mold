import Foundation
import MoldClient

extension HostStore {
    func applyConnection(_ updated: MoldHost, expectedURL: URL? = nil) {
        guard var current = host(updated.id), current.apiKey == updated.apiKey,
              expectedURL.map({ $0 == current.baseURL }) ?? true else { return }
        current.connectionOriginalURL = updated.connectionOriginalURL ?? current.connectionOriginalURL ?? current.baseURL
        current.baseURL = updated.baseURL
        current.connectionEndpoints = updated.connectionEndpoints
        current.connectionInstanceID = updated.connectionInstanceID
        setHosts(hosts.map { $0.id == updated.id ? current : $0 })
        do { try persist() } catch { report(updated.id, name: updated.name, doing: "remember its connection routes", error) }
    }
}
