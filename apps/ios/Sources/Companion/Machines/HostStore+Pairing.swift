import Foundation
import MoldClient
import UIKit

/// Redeems a scanned, pasted or opened pairing code
/// (`https://utensils.io/mold/pair#...`, or the older `mold://pair?...`).
typealias PairingClaimer = (MobilePairingPayload, _ clientName: String, _ clientKind: String) async throws -> PairingClaim

extension HostStore {
    /// Claims the code, then files the machine: a machine already in the
    /// list -- same fleet identity, or same address -- gets the new key
    /// instead of a duplicate row; a new one is added. The key goes straight
    /// to the Keychain and nowhere else.
    @discardableResult
    func pair(_ payload: MobilePairingPayload, claim: PairingClaimer,
              client: PairingClient = .current) async throws -> MoldHost {
        let answer = try await claim(payload, client.name, client.kind)
        if let known = existing(for: payload) {
            try update(known.id, name: known.name,
                       address: (answer.resolvedBaseURL?.absoluteString ?? payload.baseURL), apiKey: answer.apiKey ?? "")
            return rememberRoutes(answer, payload: payload, host: host(known.id) ?? known)
        }
        let name = answer.hostname.flatMap { $0.isEmpty ? nil : $0 } ?? payload.name
        let added = try add(name: name, address: (answer.resolvedBaseURL?.absoluteString ?? payload.baseURL), apiKey: answer.apiKey,
                       makeDefault: hosts.isEmpty)
        return rememberRoutes(answer, payload: payload, host: added)
    }

    private func rememberRoutes(_ answer: PairingClaim, payload: MobilePairingPayload, host: MoldHost) -> MoldHost {
        var updated = host
        updated.connectionOriginalURL = URL(string: payload.baseURL)
        updated.connectionEndpoints = ConnectionRoutes.sanitized(answer.endpoints ?? payload.endpoints ?? [])
        updated.connectionInstanceID = answer.instanceId
        applyConnection(updated)
        return updated
    }

    private func existing(for payload: MobilePairingPayload) -> MoldHost? {
        hosts.first { host in
            if host.connectionInstanceID == payload.instanceId { return true }
            if case let .up(status) = reachability(of: host), status.instanceId == payload.instanceId {
                return true
            }
            guard let url = URL(string: payload.baseURL) else { return false }
            return HostAddress.sameOrigin(host.baseURL, url)
        }
    }
}

/// How this device introduces itself to the machine it pairs with: listed
/// under the Mac's Paired Phones, and revocable there.
struct PairingClient {
    let name: String
    /// `iphone` or `ipad`: the server files anything else as `mobile`.
    let kind: String

    @MainActor static var current: PairingClient {
        let isPad = UIDevice.current.userInterfaceIdiom == .pad
        return PairingClient(name: isPad ? "Mold Studio on iPad" : "Mold Studio on iPhone",
                             kind: isPad ? "ipad" : "iphone")
    }
}

/// Why pairing failed, in one sentence -- the scanner and an opened link say
/// it the same way.
enum PairingFailure {
    static func sentence(_ error: any Error, name: String) -> String {
        switch error {
        case let error as HostEditError: error.errorDescription ?? ""
        case let error as PairingClaimError: error.errorDescription ?? ""
        default: "\(name) couldn't pair: \(error.failureSentence)"
        }
    }
}
