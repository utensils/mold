import MoldClient
import SwiftUI

/// One machine in Settings, with what it last said.
struct HostSettingsRow: View {
    let host: MoldHost
    let reachability: HostStore.Reachability

    var body: some View {
        HStack(spacing: 8) {
            HostStatusDot(reachability: reachability)
            VStack(alignment: .leading, spacing: 1) {
                Text(host.name)
                Text(verbatim: HostAddress.displayString(for: host.baseURL))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 8)
            if host.apiKey != nil {
                Image(systemName: "key.fill")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .help("Using a stored API key")
            }
            if reachability.isChecking {
                ProgressView().controlSize(.small)
            } else if let summary = reachability.summary {
                Text(summary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .help(summary)
            }
        }
        .padding(.vertical, 2)
    }
}

/// The answer from the machine the host editor is pointed at.
///
/// Shows nothing at all until there is something to say: an empty sheet
/// reporting "Unreachable" about an address nobody has typed reads as a fault.
struct HostProbeSummary: View {
    let reachability: HostStore.Reachability
    /// `nil` when there is no address to check, which hides the button.
    let recheck: (() -> Void)?

    var body: some View {
        HStack(spacing: 6) {
            if reachability.isChecking {
                ProgressView().controlSize(.small)
                Text("Checking…").font(.caption).foregroundStyle(.secondary)
            } else if let sentence = reachability.sentence {
                HostStatusDot(reachability: reachability)
                Text(sentence)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                if let recheck {
                    Button("Check again", systemImage: "arrow.clockwise", action: recheck)
                        .labelStyle(.iconOnly)
                        .buttonStyle(.borderless)
                        .controlSize(.small)
                }
            }
        }
    }
}
