import Foundation
import MoldClient

/// What to do about one machine `PeerSection` found nearby.
///
/// Pure and testable with no `HostStore`: "already have" is asked twice, by
/// origin (`known`) and by fleet identity (`knownInstance`), because a
/// machine found again at a second address is not a new machine -- and a
/// peer whose `url` does not even normalize is never offered at all.
enum PeerAction: Equatable {
    /// This machine, or one we already talk to.
    case skip
    /// No key needed: add it without a sheet.
    case add(name: String, url: URL)
    /// Wants a key: open the editor prefilled, rather than fail silently.
    case edit(name: String, address: String)

    static func resolve(
        _ peer: DiscoveryPeer,
        known: (URL) -> Bool,
        knownInstance: (String) -> Bool
    ) -> PeerAction {
        guard !peer.isThisMachine else { return .skip }
        guard let url = HostAddress.normalize(peer.url) else { return .skip }
        guard !known(url) else { return .skip }
        if let id = peer.instanceId, knownInstance(id) { return .skip }

        guard peer.authRequired else { return .add(name: peer.name, url: url) }
        return .edit(name: peer.name, address: HostAddress.displayString(for: url))
    }
}
