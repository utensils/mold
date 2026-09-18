import Foundation
import MoldClient

/// How a render gets the print's own conditioning media back.
///
/// Two routes, and which one applies is not a preference -- it is where the
/// bytes live:
///
/// - **Same machine**: a one-use SESSION. The host already holds the bytes, so
///   nothing moves: it mints a handle bound to the sha256 of this exact
///   request, this credential, this instance and this archive, and hydrates
///   the admission itself.
/// - **Another machine**: a RELAY. A session cannot travel -- the host binds
///   it to its own instance id -- so the bytes are downloaded from the print's
///   origin and inlined into the request. Paths, pin ids and store identities
///   never cross; only the file's contents do.
///
/// A free type rather than more `GenerateController`, which is already over
/// the type-size budget.
@MainActor
struct RetainedMediaHydration: Sendable {
    let authority: ReuseStore.Authority
    let hosts: HostStore

    /// What to do with an admission once it is built.
    enum Outcome: Sendable {
        /// Send this handle in `x-mold-retained-media-session`.
        case session(String)
        /// Send these requests instead: the bytes are already in them.
        case requests([GenerateRequest])
        /// Nothing retained is wanted -- every role the print kept is one the
        /// person has already filled in themselves, or one this build cannot
        /// place.
        case nothingToDo
    }

    /// Resolves the media for one admission.
    ///
    /// `requests` is the whole batch: a SESSION binds exactly one child
    /// (`validate_reuse_batch_cardinality`), so a batch of four takes the
    /// relay even on the machine that made the print -- four siblings all
    /// needing the same picture is an ordinary thing to ask for, and refusing
    /// it when the bytes are right there would be a rule with no reason
    /// behind it.
    func hydrate(_ requests: [GenerateRequest], on target: MoldHost.ID,
                 backend: any MoldBackend) async throws -> Outcome {
        guard let first = requests.first else { return .nothingToDo }
        let wanted = RetainedSourceMedia.members(authority.members, forHydrating: first)
        guard !wanted.isEmpty else { return .nothingToDo }
        if target == authority.origin, requests.count == 1 {
            let session = try await backend.retainedMediaReuseSession(
                for: authority.filename, members: wanted.map(\.memberId), target: first)
            return .session(session.sessionHandle)
        }
        return .requests(try await relay(wanted, into: requests))
    }

    /// Downloads each member ONCE from the print's origin and inlines it into
    /// every sibling -- four copies of one render condition on the same
    /// picture, and fetching it four times would be the same bytes four times.
    private func relay(
        _ members: [RetainedSourceMedia.Member], into requests: [GenerateRequest]
    ) async throws -> [GenerateRequest] {
        // Asked from the sizes the INVENTORY already reported, so nothing is
        // downloaded for a relay that could never be sent.
        if let refusal = RetainedSourceMedia.relayRefusal(
            members, copies: requests.count) { throw refusal }
        guard let origin = hosts.backend(for: authority.origin) else {
            throw MoldClientError.unreachable(
                "The machine that made this print isn't connected.")
        }
        var fetched: [(member: RetainedSourceMedia.Member, bytes: Data)] = []
        for member in members {
            fetched.append((member, try await origin.retainedSourceMediaBytes(
                for: authority.filename, member: member.memberId)))
        }
        return try requests.map { try RetainedSourceMedia.relayed(fetched, into: $0) }
    }
}

/// The one line the submit path calls, so nothing about retained media has to
/// live inside `GenerateController`.
@MainActor
enum RetainedMedia {
    /// The admission to actually send. The client batch id is CARRIED, never
    /// re-minted: it is the idempotency fence, and a relay that changed it
    /// would make a lost response unrecoverable.
    static func hydrated(
        _ admission: BatchAdmission, with hydration: RetainedMediaHydration?,
        on host: MoldHost, backend: any MoldBackend
    ) async throws -> BatchAdmission {
        guard let hydration else { return admission }
        switch try await hydration.hydrate(
            admission.requests, on: host.id, backend: backend) {
        case .nothingToDo:
            return admission
        case let .session(handle):
            var sending = admission
            sending.retainedMediaSession = handle
            return sending
        case let .requests(hydrated):
            return BatchAdmission(clientBatchId: admission.clientBatchId,
                                  requests: hydrated)
        }
    }
}
