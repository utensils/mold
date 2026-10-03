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
        guard let first = requests.first,
              !RetainedSourceMedia.members(authority.members, forHydrating: first).isEmpty
        else { return .nothingToDo }
        guard let origin = hosts.backend(for: authority.origin) else {
            throw MoldClientError.unreachable("The machine that made this print isn't connected.")
        }
        let hydrated = try await RetainedSourceMedia.hydrated(
            BatchAdmission(requests: requests), filename: authority.filename,
            members: authority.members, sameHost: target == authority.origin,
            origin: origin, target: backend)
        if let handle = hydrated.retainedMediaSession { return .session(handle) }
        return hydrated.requests == requests ? .nothingToDo : .requests(hydrated.requests)
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
