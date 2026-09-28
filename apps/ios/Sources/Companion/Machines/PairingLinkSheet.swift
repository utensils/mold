import MoldClient
import SwiftUI

/// A pairing code that arrived as a link -- the Camera app reading the Mac's
/// QR, or a tapped `https://utensils.io/mold/pair#…` -- shown for the person
/// to confirm. A link anyone can send never pairs the phone by itself: this
/// sheet names the machine and its address, and only Pair claims it.
struct PairingLinkSheet: View {
    @Environment(HostStore.self) private var hosts
    @Environment(CompanionStores.self) private var stores
    @Environment(\.dismiss) private var dismiss
    let link: AppRouter.PairingLink

    @State private var phase: Phase = .asking

    enum Phase: Equatable {
        case asking
        case pairing
        case failed(String)
    }

    var body: some View {
        NavigationStack {
            Form {
                if let payload = link.payload {
                    details(payload)
                } else {
                    Section {
                        Label(link.failure ?? MobilePairingPayload.ParseError.notPairingCode.errorDescription ?? "",
                              systemImage: "exclamationmark.triangle")
                    } footer: {
                        Text("Make a new code on your Mac: Machines ▸ your machine ▸ Pair a Phone…")
                            .foregroundStyle(.secondaryText)
                    }
                }
            }
            .navigationTitle(link.payload == nil ? "Can't Pair" : "Pair This Phone?")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(link.payload == nil ? "Done" : "Cancel") { dismiss() }
                }
                if let payload = link.payload, !payload.isExpired() {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Pair") { pair(payload) }
                            .disabled(phase == .pairing)
                    }
                }
            }
        }
        .interactiveDismissDisabled(phase == .pairing)
    }

    @ViewBuilder private func details(_ payload: MobilePairingPayload) -> some View {
        Section {
            row("Machine", payload.name)
            row("Address", payload.baseURL, mono: true)
        } footer: {
            Text("Pair only with a machine you trust: what you make on this phone is sent to it.")
                .foregroundStyle(.secondaryText)
        }
        if payload.isExpired() {
            Section {
                Label("This code has expired. Make a new one on your Mac: Machines ▸ \(payload.name) ▸ Pair a Phone…",
                      systemImage: "clock.badge.exclamationmark")
            }
        }
        switch phase {
        case .asking: EmptyView()
        case .pairing:
            Label { Text("Pairing with \(payload.name)…") } icon: { ProgressView() }
        case let .failed(reason):
            Section {
                Label(reason, systemImage: "exclamationmark.triangle").foregroundStyle(.red)
            }
        }
    }

    /// Always stacked: an address is long, and a row never truncates it.
    private func row(_ label: LocalizedStringKey, _ value: String, mono: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.subheadline).foregroundStyle(.secondaryText)
            Text(value).font(mono ? .body.monospaced() : .body).textSelection(.enabled)
        }
        .accessibilityElement(children: .combine)
    }

    private func pair(_ payload: MobilePairingPayload) {
        phase = .pairing
        Task {
            do {
                try await hosts.pair(payload, claim: stores.claimPairing)
                dismiss()
            } catch {
                phase = .failed(PairingFailure.sentence(error, name: payload.name))
            }
        }
    }
}
