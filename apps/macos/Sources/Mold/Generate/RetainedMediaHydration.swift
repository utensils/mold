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
        guard target == authority.origin, requests.count == 1 else {
            return .requests(try await relay(wanted, into: requests))
        }
        do {
            return .session(try await mint(wanted, for: first, on: backend))
        } catch let error as MoldClientError {
            guard case let .http(_, code, _) = error,
                  let refusal = code.flatMap(RetainedSourceMedia.Refusal.init(rawValue:)),
                  refusal.isWorthOneMoreAttempt
            else { throw error }
            // The handle expired, or the print was re-published between the
            // probe and now. Both describe the HANDLE, not the archive, so
            // one more mint is the whole repair.
            if let handle = try? await mint(wanted, for: first, on: backend) {
                return .session(handle)
            }
            // Still not: the bytes are on this very machine, so carry them
            // rather than refuse a render it can obviously make.
            return .requests(try await relay(wanted, into: requests))
        }
    }

    private func mint(
        _ members: [RetainedSourceMedia.Member], for request: GenerateRequest,
        on backend: any MoldBackend
    ) async throws -> String {
        try await backend.retainedMediaReuseSession(
            for: authority.filename, members: members.map(\.memberId), target: request
        ).sessionHandle
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
    /// A machine's retained-media refusal, in this app's words. Its own type
    /// so `Error.sentence` reads it and the pane shows the sentence rather
    /// than the host's API prose.
    struct Refused: LocalizedError {
        let sentence: String
        var errorDescription: String? { sentence }
    }

    /// The admission to actually send. The client batch id is CARRIED, never
    /// re-minted: it is the idempotency fence, and a relay that changed it
    /// would make a lost response unrecoverable.
    static func hydrated(
        _ admission: BatchAdmission, with hydration: RetainedMediaHydration?,
        on host: MoldHost, backend: any MoldBackend
    ) async throws -> BatchAdmission {
        guard let hydration else { return admission }
        let outcome: RetainedMediaHydration.Outcome
        do {
            outcome = try await hydration.hydrate(
                admission.requests, on: host.id, backend: backend)
        } catch let error as MoldClientError {
            // A machine's retained-media code becomes THIS app's sentence,
            // with the way forward in it. Anything else is an ordinary
            // transport failure and already reads as one.
            guard case let .http(_, code, _) = error,
                  let sentence = RetainedSourceMedia.refusalSentence(for: code)
            else { throw error }
            throw Refused(sentence: sentence)
        }
        switch outcome {
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
