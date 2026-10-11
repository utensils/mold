import Foundation
import MoldClient

extension HostStore {
    @discardableResult
    func pair(_ payload: MobilePairingPayload, replacing id: MoldHost.ID? = nil,
              claim: (MobilePairingPayload) async throws -> PairingClaim = { payload in
                  try await HTTPBackend.claimPairing(payload, clientName: "Mold Studio on Mac", clientKind: "macos")
              }) async throws -> MoldHost {
        let expected = id.flatMap(host)
        let answer = try await claim(payload)
        try Task.checkCancellation()
        if let id, host(id) != expected { throw CancellationError() }
        guard let secure = answer.secure, let url = answer.resolvedBaseURL ?? URL(string: payload.baseURL),
              let key = answer.apiKey, !key.isEmpty else { throw SecureConnectionError.pairingRequired }
        let previous = id.flatMap(host) ?? hosts.first { $0.secure?.namespace == secure.namespace || $0.connectionInstanceID == answer.instanceId }
        var paired = MoldHost(id: previous?.id ?? UUID(), name: previous?.name ?? answer.hostname ?? payload.name, baseURL: url,
                              apiKey: key, secure: secure)
        paired.connectionEndpoints = ConnectionRoutes.sanitized(answer.endpoints ?? payload.endpoints ?? [])
        paired.connectionInstanceID = answer.instanceId
        paired.connectionOriginalURL = URL(string: payload.baseURL)
        try HostPersistence.setAPIKey(key, for: paired.id)
        if previous != nil { hosts = hosts.map { $0.id == paired.id ? paired : $0 }; forget(paired.id) }
        else { hosts.append(paired) }
        HostPersistence.save(hosts.filter { $0.id != MoldEngine.localHostID })
        await refresh(paired)
        return host(paired.id) ?? paired
    }
}
