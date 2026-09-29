import MoldClient
import SwiftUI

/// Add a Machine (DESIGN.md §5.4): scan a pairing code, pick one found on this
/// network, or type an address -- each ending at the same confirm step.
struct AddMachineSheet: View {
    @Environment(AppRouter.self) private var router
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Group {
                if let prefill = router.addPrefill {
                    AddressForm(initialName: prefill.name, initialAddress: prefill.address) { dismiss() }
                } else {
                    choices
                }
            }
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
        .onDisappear { router.addPrefill = nil }
    }

    private var choices: some View {
        List {
            Section {
                NavigationLink {
                    PairingScanView { dismiss() }
                } label: {
                    ChoiceRow(title: String(localized: "Scan a Pairing Code"), symbol: "qrcode.viewfinder",
                              compactTitle: String(localized: "Scan Code"),
                              detail: String(localized: "On your Mac, open Machines ▸ your machine ▸ Pair a Phone…"))
                }
                NavigationLink {
                    NearbyPicker { dismiss() }
                } label: {
                    ChoiceRow(title: String(localized: "Nearby"), symbol: "wifi",
                              compactTitle: String(localized: "Nearby"),
                              detail: String(localized: "Machines running mold on this network."))
                }
                NavigationLink {
                    AddressForm(initialName: "", initialAddress: "") { dismiss() }
                } label: {
                    ChoiceRow(title: String(localized: "Enter an Address"), symbol: "keyboard",
                              compactTitle: String(localized: "Enter Address"),
                              detail: String(localized: "A name, an IP address, or a Tailscale address."))
                }
            }
        }
        .navigationTitle("Add a Machine")
        .navigationBarTitleDisplayMode(.inline)
    }
}

/// One of the three ways in: a symbol, a title, and a line on what it is for.
struct ChoiceRow: View {
    @Environment(\.dynamicTypeSize) private var size
    let title: String
    let symbol: String
    let compactTitle: String
    let detail: String

    var body: some View {
        Label {
            VStack(alignment: .leading, spacing: 2) {
                Text(size.isAccessibilitySize ? compactTitle : title).font(.headline)
                if !size.isAccessibilitySize {
                    Text(detail).font(.subheadline).foregroundStyle(.secondaryText)
                }
            }
        } icon: {
            Image(systemName: symbol).foregroundStyle(.tint)
        }
        .padding(.vertical, 6)
        .accessibilityLabel(title)
        .accessibilityHint(detail)
    }
}

/// Machines found on this network that are not in the list yet.
struct NearbyPicker: View {
    @Environment(HostStore.self) private var hosts
    @Environment(NearbyBrowser.self) private var nearby
    let done: () -> Void

    var body: some View {
        let fresh = nearby.machines.filter { !hosts.knows($0) }
        List {
            if let problem = nearby.problem {
                Text(problem).foregroundStyle(.secondaryText)
            } else if fresh.isEmpty {
                Label("Looking on this network…", systemImage: "wifi")
                    .foregroundStyle(.secondaryText)
            }
            ForEach(fresh) { machine in NearbyRow(machine: machine, done: done) }
        }
        .navigationTitle("Nearby")
        .onAppear { nearby.start() }
    }
}

/// A found machine: its name, and Add -- which resolves where it answers and
/// opens the address form there, so the check and a key are one step away.
struct NearbyRow: View {
    @Environment(NearbyBrowser.self) private var nearby
    @Environment(AppRouter.self) private var router
    let machine: NearbyBrowser.Machine
    var done: (() -> Void)?
    @State private var resolving = false
    @State private var problem: String?

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(machine.name)
                if let problem { Text(problem).font(.footnote).foregroundStyle(.secondaryText) }
            }
            Spacer(minLength: 8)
            if resolving { ProgressView() } else {
                Button("Add") { Task { await add() } }
                    .buttonStyle(.bordered)
                    .accessibilityLabel("Add \(machine.name)")
            }
        }
        .padding(.horizontal, done == nil ? 16 : 0)
    }

    private func add() async {
        resolving = true
        defer { resolving = false }
        do {
            let address = try await nearby.address(of: machine)
            router.addMachine(.init(name: machine.name, address: address))
        } catch {
            problem = String(localized: "\(machine.name) stopped answering before it could be added.")
        }
    }
}
