import Foundation
import MoldClient
import UIKit

/// Redeems a scanned or pasted pairing code (`mold://pair?...`).
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
                       address: payload.baseURL, apiKey: answer.apiKey ?? "")
            return host(known.id) ?? known
        }
        let name = answer.hostname.flatMap { $0.isEmpty ? nil : $0 } ?? payload.name
        return try add(name: name, address: payload.baseURL, apiKey: answer.apiKey,
                       makeDefault: hosts.isEmpty)
    }

    private func existing(for payload: MobilePairingPayload) -> MoldHost? {
        hosts.first { host in
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
