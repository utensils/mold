import MoldClient

extension CompanionStores {
    /// The single place a concrete backend is built. `make lint` fails if one
    /// is constructed anywhere else, so "what is this app talking to" is a
    /// decision in one file -- and every store test can hand in a fake.
    static let http: (MoldHost) -> any MoldBackend = { HTTPBackend(host: $0) }

    /// Redeeming a pairing code talks to a machine that is not in the list
    /// yet, with no key -- still a concrete backend, so it is built here too.
    static let claim: PairingClaimer = { payload, name, kind in
        try await HTTPBackend.claimPairing(payload, clientName: name, clientKind: kind)
    }
}
