import Foundation
import MoldClient
import SwiftUI

/// How a machine's state reads, in one place -- the Mac's `HostStatus.swift`
/// words, so the same machine says the same thing on both.
extension HostStore.Reachability {
    /// Status semantics, not brand colour, and never the only signal: every
    /// dot sits beside `summary`.
    var tint: Color {
        switch self {
        case .unknown, .checking: .gray
        case .up: .green
        case .needsKey: .orange
        case .down: .red
        }
    }

    /// The short form, for a card or a row.
    var summary: String? {
        switch self {
        case .unknown: nil
        case .checking: String(localized: "Checking…")
        case let .up(status):
            status.busy
                ? String(localized: "Busy · \(status.versionLabel)")
                : String(localized: "Ready · \(status.versionLabel)")
        case .needsKey: String(localized: "Needs an API key")
        case let .down(reason): reason
        }
    }

    /// The long form, for the Add sheet's live check: the only answer on
    /// screen, so it says what answered.
    var sentence: String? {
        switch self {
        case .unknown: nil
        case .checking: String(localized: "Checking…")
        case let .up(status):
            [status.hostname, "mold \(status.versionLabel)", status.hardware]
                .compactMap(\.self).joined(separator: " · ")
        case .needsKey: String(localized: "This machine is there but wants an API key.")
        case let .down(reason): reason
        }
    }

    var isUp: Bool {
        if case .up = self { return true }
        return false
    }
}

extension HostStore {
    /// Whether a machine found on the network is already in the list: by the
    /// fleet identity it announced when it has one (the same box at a second
    /// address is still the same box), else by name.
    func knows(_ found: NearbyBrowser.Machine) -> Bool {
        if let instance = found.instanceID {
            let known = reachability.values.compactMap { state -> String? in
                if case let .up(status) = state { return status.instanceId }
                return nil
            }
            if known.contains(instance) { return true }
        }
        return hosts.contains { $0.name == found.name }
    }

    /// What a card says about a machine that is not answering: since when,
    /// when it has answered before, and why, when the check said.
    func silence(of host: MoldHost) -> String? {
        guard case let .down(reason) = reachability(of: host) else { return nil }
        guard let last = lastAnswered[host.id] else { return reason }
        return String(localized: "Not answering since \(last.formatted(date: .omitted, time: .shortened)) — it may be asleep or off the network.")
    }
}

/// The dot every machine list draws, labelled for VoiceOver.
struct StatusDot: View {
    let reachability: HostStore.Reachability
    @ScaledMetric(relativeTo: .body) private var size = 9

    var body: some View {
        Circle()
            .fill(reachability.tint)
            .frame(width: size, height: size)
            .accessibilityLabel(reachability.summary ?? String(localized: "Not checked yet"))
    }
}
