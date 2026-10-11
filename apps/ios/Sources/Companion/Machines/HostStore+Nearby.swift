import Foundation
import MoldClient

extension HostStore {
    /// Bonjour is only a hint. Even an exact TXT identity cannot authorize
    /// sending a paired key to a new address; its anonymous proof must pass.
    func recoverNearby(_ found: NearbyBrowser.Machine, address: String) async {
        guard let identity = found.instanceID, !identity.isEmpty,
              let original = hosts.first(where: { $0.connectionInstanceID == identity }),
              original.secure != nil,
              let endpoint = ConnectionRoutes.sanitized([ConnectionEndpoint(url: address, kind: .lan)]).first,
              let url = URL(string: endpoint.url), original.baseURL != url || !isUp(original) else { return }
        var candidate = original
        candidate.connectionOriginalURL = original.connectionOriginalURL ?? original.baseURL
        candidate.connectionEndpoints = [endpoint]
        do {
            guard var verified = try await backend(for: candidate).resolvedConnection(),
                  verified.baseURL == url, verified.id == original.id,
                  verified.apiKey == original.apiKey, verified.secure == original.secure, verified.connectionInstanceID == identity,
                  host(original.id) == original, !Task.isCancelled else { return }
            verified.connectionOriginalURL = candidate.connectionOriginalURL
            verified.connectionEndpoints = ConnectionRoutes.sanitized([endpoint] + (original.connectionEndpoints ?? []))
            stopWatching(original.id)
            applyConnection(verified, expectedURL: original.baseURL)
            await refresh(verified)
        } catch {
            // An unverified announcement cannot change the saved machine or
            // turn a working relay red. The ordinary health read owns status.
        }
    }

    func recoverNearby(_ machines: [NearbyBrowser.Machine], browser: NearbyBrowser) async {
        await withTaskGroup(of: Void.self) { group in
            for machine in machines {
                guard let identity = machine.instanceID, !identity.isEmpty,
                      let known = hosts.first(where: { $0.connectionInstanceID == identity }),
                      known.secure != nil else { continue }
                if isUp(known), known.connectionEndpoints?.contains(where: {
                    $0.kind == .lan && $0.url == known.baseURL.absoluteString
                }) == true { continue }
                group.addTask { await self.resolveNearby(machine, browser: browser) }
            }
        }
    }

    private func resolveNearby(_ machine: NearbyBrowser.Machine, browser: NearbyBrowser) async {
        guard let address = try? await browser.address(of: machine), !Task.isCancelled else { return }
        await recoverNearby(machine, address: address)
    }

    /// A host can return without the phone changing networks. A failed SSE
    /// watcher has already stopped, so foreground health recovery owns this.
    func refreshUnavailable() async {
        await withTaskGroup(of: Void.self) { group in
            for host in hosts {
                guard case .down = reachability(of: host) else { continue }
                group.addTask { await self.refresh(host) }
            }
        }
    }
}
