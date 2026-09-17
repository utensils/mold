import MoldClient
import SwiftUI

/// Mobile pairing, directly under Address (design decision 12) -- for a
/// machine this app holds an OPERATOR key for. `auth_required` is the only
/// honest gate (`PairedClients.canPair`'s own doc): `pairing_available` is
/// `true` even on a keyless host, so it answers a different question.
struct PairingSection: View {
    @Environment(HostStore.self) private var hosts
    @Environment(PairingStore.self) private var pairing
    let host: MoldHost
    @State private var showingSheet = false
    @State private var pendingRevoke: Destruction?

    /// What the section draws. `nil` means not asked yet, so nothing is
    /// drawn -- the same "no view rather than a placeholder" rule `memory(_:)`
    /// (`MachinesPane+Sections.swift`) already follows.
    enum SectionState: Equatable {
        case absent
        case needsOperator
        case databaseOff
        case clients([PairedClient])
    }

    /// Pure, so the four branches are tested with no view -- `AccountsSettings`'s
    /// own idiom. `authority == .paired` wins even over a stale answer: it is
    /// the more recent fact, caught from the very call that would have
    /// refreshed `answer`.
    static func resolve(_ answer: PairedClients?, authority: PairingStore.Authority?) -> SectionState? {
        if authority == .paired { return .needsOperator }
        guard let answer else { return nil }
        guard answer.authRequired else { return .absent }
        guard answer.pairingAvailable else { return .databaseOff }
        return .clients(answer.clients)
    }

    private var state: SectionState? { Self.resolve(pairing.byHost[host.id], authority: pairing.authority[host.id]) }

    var body: some View {
        content
            // Runs across every state, including `nil` -- that IS the state
            // that needs the fetch. A `PeerSection`-style task inside only
            // the drawn branch would never run for a machine not asked yet.
            .task(id: host.id) { await load() }
            .sheet(isPresented: $showingSheet) { PairingSheet(host: host) }
            .destructionDialog($pendingRevoke)
    }

    @ViewBuilder private var content: some View {
        switch state {
        case nil, .absent:
            EmptyView()
        case .needsOperator:
            Section("Pairing") {
                Text("This app's key for this machine can't manage paired access -- only an operator key can.")
                    .foregroundStyle(.secondary)
            }
        case .databaseOff:
            Section("Pairing") {
                Text("This machine's metadata database is off, so paired access is unavailable.")
                    .foregroundStyle(.secondary)
            }
        case let .clients(clients):
            Section("Pairing") {
                ForEach(clients) { client in row(client) }
                Button("Pair a Phone…") { showingSheet = true }
            }
        }
    }

    private func load() async {
        guard !pairing.isSeeded else { return }
        if let fixture = Self.fixtureIfRequested() {
            pairing.seed(from: fixture)
        } else {
            await pairing.refresh(on: host.id)
        }
    }

    private func row(_ client: PairedClient) -> some View {
        LabeledContent {
            Button("Revoke", role: .destructive) { confirmRevoke(client) }
                .buttonStyle(.link)
        } label: {
            VStack(alignment: .leading, spacing: 2) {
                Text(client.name)
                Text(lastSeen(client)).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private func lastSeen(_ client: PairedClient) -> String {
        guard let ms = client.lastUsedAtMs else { return "Never used" }
        let date = Date(timeIntervalSince1970: TimeInterval(ms) / 1_000)
        return date.formatted(.relative(presentation: .named))
    }

    private func confirmRevoke(_ client: PairedClient) {
        pendingRevoke = Destruction(
            title: "Revoke “\(client.name)”?",
            message: "This device will need to be paired again to reach this machine.",
            verb: "Revoke"
        ) { Task { await pairing.revoke(client, on: host.id) } }
    }
}
