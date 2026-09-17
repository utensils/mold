import MoldClient
import SwiftUI

/// How a machine's state reads, in one place.
///
/// The sidebar, the Settings list and the host editor all print this. They
/// used to each carry their own switch, which is how the same machine could
/// say "Needs an API key" in one list and show a bare red dot in another.
extension HostStore.Reachability {
    /// `.green`/`.red` here are status semantics, not brand color -- the same
    /// meaning the system uses in its own connection indicators.
    var tint: Color {
        switch self {
        case .unknown, .checking: .secondary
        case .up: .green
        // Amber, not red: the machine is fine, the credential is missing.
        case .needsKey: .orange
        case .down: .red
        }
    }

    var isChecking: Bool {
        if case .checking = self { return true }
        return false
    }

    /// The short form, for a row in a list.
    var summary: String? {
        switch self {
        case .unknown: nil
        case .checking: "Checking…"
        case let .up(status): status.busy ? "Busy · \(status.version)" : "Ready · \(status.version)"
        case .needsKey: "Needs an API key"
        case let .down(reason): reason
        }
    }

    /// The long form, for the editor -- it is the only answer on screen, so it
    /// says what answered rather than just that something did.
    var sentence: String? {
        switch self {
        case .unknown: nil
        case .checking: "Checking…"
        case let .up(status):
            [status.hostname,
             "mold \(status.version)",
             status.hardware]
                .compactMap(\.self).joined(separator: " · ")
        case .needsKey:
            "This machine is there but wants an API key."
        case let .down(reason): reason
        }
    }
}

extension ServerStatus {
    /// What the machine renders with, collapsed the way a person would say it:
    /// four identical cards are "4× NVIDIA L40S", not four lines.
    var hardware: String? {
        guard let gpus, let first = gpus.first else { return nil }
        guard gpus.count > 1 else { return first.name }
        let names = Set(gpus.map(\.name))
        return names.count == 1 ? "\(gpus.count)× \(first.name)" : "\(gpus.count) GPUs"
    }
}

/// The dot every machine list uses.
struct HostStatusDot: View {
    let reachability: HostStore.Reachability

    var body: some View {
        Image(systemName: "circle.fill")
            .font(.system(size: 7))
            .foregroundStyle(reachability.tint)
            .accessibilityLabel(reachability.summary ?? "Not checked yet")
    }
}
