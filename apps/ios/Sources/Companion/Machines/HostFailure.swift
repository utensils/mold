import Foundation
import MoldClient
import SwiftUI

/// One thing that went wrong, said once, in words: the machine, what did not
/// happen, the machine's own reason, the way forward. Shown in an inline
/// banner a person dismisses -- never a toast, never a modal (DESIGN.md §2).
struct HostFailure: Identifiable, Equatable {
    static let reachVerb = "answer"

    let id = UUID()
    let host: MoldHost.ID?
    let doing: String
    let text: String
}

extension HostStore {
    /// Records a failure. The same machine failing the same way replaces its
    /// earlier line rather than stacking a second.
    func report(_ host: MoldHost.ID?, name: String?, doing: String, _ error: Error) {
        let who = name ?? "A machine"
        let text = "\(who) couldn't \(doing): \(error.reason) \(error.advice ?? "")"
            .trimmingCharacters(in: .whitespaces)
        let kept = failures.filter { !($0.host == host && $0.doing == doing) }
        setFailures([HostFailure(host: host, doing: doing, text: text)] + kept.prefix(4))
    }

    func report(_ host: MoldHost, doing: String, _ error: Error) {
        report(host.id, name: host.name, doing: doing, error)
    }

    func dismiss(_ failure: HostFailure) {
        setFailures(failures.filter { $0.id != failure.id })
    }

    /// It worked this time, so its old line is no longer true.
    func clearFailures(for host: MoldHost.ID, doing: String? = nil) {
        setFailures(failures.filter { $0.host != host || (doing != nil && $0.doing != doing) })
    }
}

/// The inline banner: newest failure first, each with its own ✕.
struct FailureBanner: View {
    @Environment(HostStore.self) private var hosts

    var body: some View {
        if let failure = hosts.failures.first {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                Label {
                    Text(failure.text).fixedSize(horizontal: false, vertical: true)
                } icon: {
                    Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                }
                Spacer(minLength: 0)
                Button { withAnimation { hosts.dismiss(failure) } } label: {
                    Label("Dismiss", systemImage: "xmark").labelStyle(.iconOnly)
                }
                .buttonStyle(.borderless)
                .frame(minWidth: 44, minHeight: 44)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(.regularMaterial, in: .rect(cornerRadius: 16))
            .padding(.horizontal, 16)
            .accessibilityElement(children: .combine)
            .accessibilityAction(named: "Dismiss") { hosts.dismiss(failure) }
        }
    }
}
