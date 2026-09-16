import MoldClient
import SwiftUI

// Reading what the person typed. Split from the sheet itself because this
// half is the part with rules -- what an address resolves to, whose machine
// that already is, and what to say when it is neither.
extension HostEditor {
    // MARK: - What the address resolves to

    var resolved: URL? { HostAddress.normalize(address) }

    var duplicate: MoldHost? {
        resolved.flatMap { hosts.host(at: $0, excluding: existing?.id) }
    }

    var suggestedName: String {
        resolved.map(HostAddress.suggestedName) ?? "plato"
    }

    /// The one line under the address: what went wrong, whose address this
    /// already is, or -- when we changed what was typed -- what we will use.
    ///
    /// A plain `Text?` rather than a `@ViewBuilder`, because every branch is a
    /// styled `Text` and a builder would widen it to `_ConditionalContent`.
    var hint: Text? {
        if let problem = addressProblem {
            return Text(problem).foregroundStyle(.red)
        }
        if let duplicate {
            return Text("\(duplicate.name) already uses this address.").foregroundStyle(.red)
        }
        let typed = address.trimmingCharacters(in: .whitespacesAndNewlines)
        if let resolved, resolved.absoluteString != typed {
            return Text(verbatim: resolved.absoluteString).foregroundStyle(.secondary)
        }
        return nil
    }

    var addressProblem: String? {
        guard !address.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        do {
            _ = try HostAddress.resolve(address)
            return nil
        } catch {
            return error.message
        }
    }
}
