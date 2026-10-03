import Foundation

public extension RetainedSourceMedia {
    /// Restore only unfilled roles. Sessions stay on their archive's host;
    /// cross-host and multi-output batches relay bounded bytes once per member.
    static func hydrated(
        _ admission: BatchAdmission, filename: String, members: [Member],
        sameHost: Bool, origin: any MoldBackend, target: any MoldBackend
    ) async throws -> BatchAdmission {
        guard let first = admission.requests.first else { return admission }
        let wanted = Self.members(members, forHydrating: first)
        guard !wanted.isEmpty else { return admission }
        try Task.checkCancellation()
        if sameHost, admission.requests.count == 1 {
            do {
                var sending = admission
                sending.retainedMediaSession = try await target.retainedMediaReuseSession(
                    for: filename, members: wanted.map(\.memberId), target: first).sessionHandle
                try Task.checkCancellation()
                return sending
            } catch let error as MoldClientError {
                guard case let .http(_, code, _) = error,
                      let refusal = code.flatMap(Refusal.init(rawValue:)),
                      refusal.isWorthOneMoreAttempt else { throw error }
                try Task.checkCancellation()
                do {
                    var sending = admission
                    sending.retainedMediaSession = try await target.retainedMediaReuseSession(
                        for: filename, members: wanted.map(\.memberId), target: first).sessionHandle
                    try Task.checkCancellation()
                    return sending
                } catch {
                    try Task.checkCancellation()
                }
            }
        }
        if let refusal = relayRefusal(wanted, copies: admission.requests.count) { throw refusal }
        var fetched: [(member: Member, bytes: Data)] = []
        for member in wanted {
            try Task.checkCancellation()
            fetched.append((member, try await origin.retainedSourceMediaBytes(
                for: filename, member: member.memberId)))
        }
        try Task.checkCancellation()
        return BatchAdmission(clientBatchId: admission.clientBatchId,
            requests: try admission.requests.map { try relayed(fetched, into: $0) })
    }
}
