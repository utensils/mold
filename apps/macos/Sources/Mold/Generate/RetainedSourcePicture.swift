import Foundation
import MoldClient

/// Getting a reused print's own picture into the source WELL.
///
/// It started as the long-clip route's workaround: `POST /api/chain-jobs`
/// redeems no reuse session -- only `/api/generate`, `/api/generate/stream`
/// and `/api/generation-batches` do (`routes.rs:3080`, `:3475`, `:4662`) --
/// but the chain WIRE carries the bytes per stage (`source_image_b64`), so
/// fetching the picture into the draft's own well needs no server support.
/// It is now EVERY route's: an invisible authority the host applied at
/// submit left the well empty, no Strength control, and no way to see what
/// the render would start from (UAT 2026-09-17 #3). Once the picture is an
/// ordinary attachment the request carries the bytes itself,
/// `members(_:forHydrating:)` asks the host for nothing it already has, and
/// the session is left for what a well cannot hold -- a mask, an identity
/// photo, audio.
///
/// It ANSWERS rather than writing through an `inout` draft, because the
/// draft lives on an actor-isolated store and cannot be passed `inout`
/// across an `await`; the caller applies it in one step when the bytes are
/// in hand.
@MainActor
enum RetainedSourcePicture {

    /// The one role a well can hold (and the one a chain body can carry).
    static let carriedRole = "source_image"

    /// The member to fetch, if this authority has one the well can take and
    /// the well is empty -- a picture somebody attached themselves is never
    /// overwritten.
    static func member(
        of authority: ReuseStore.Authority, forHydrating outgoing: GenerateRequest?
    ) -> RetainedSourceMedia.Member? {
        guard let outgoing, outgoing.sourceImage == nil else { return nil }
        return RetainedSourceMedia.members(authority.members, forHydrating: outgoing)
            .first { $0.role == carriedRole }
    }

    /// The picture, or the sentence to show instead.
    enum Fetched: Sendable {
        case picture(String)
        case refused(String)
    }

    static func fetch(
        _ member: RetainedSourceMedia.Member, of authority: ReuseStore.Authority,
        hosts: HostStore
    ) async -> Fetched {
        if let refusal = RetainedSourceMedia.relayRefusal([member], copies: 1) {
            return .refused(refusal.errorDescription ?? "")
        }
        guard let origin = hosts.backend(for: authority.origin) else {
            return .refused("The machine that made this print isn\u{2019}t connected, so "
                + "its source picture couldn\u{2019}t be fetched. Attach one and press "
                + "Develop again.")
        }
        do {
            return .picture(try await origin.retainedSourceMediaBytes(
                for: authority.filename, member: member.memberId).base64EncodedString())
        } catch {
            return .refused(RetainedSourceMedia.refusalSentence(for: code(of: error))
                ?? "This print\u{2019}s source picture couldn\u{2019}t be fetched from its "
                + "machine. Attach one and press Develop again.")
        }
    }

    /// Puts it in the well. The well is the point: the picture stops being
    /// invisible authority the host applies and becomes an ordinary
    /// attachment a person can see, change and remove.
    static func place(_ picture: String, named name: String, in draft: inout RenderDraft) {
        draft.media.sourceImage = picture
        draft.media.sourceImageName = name
        // The picked picture too, so a later re-fit does not compound: here it
        // IS the original, never a fitted copy of one.
        draft.media.sourceImageOriginal = picture
        draft.media.sourceImageOriginalName = name
    }

    /// The first request the draft would build, for asking whether a retained
    /// role would be hydrated at all. Built through the request builder rather
    /// than by reading the wells, so the answer cannot disagree with what
    /// actually ships (an exclusive well parks its media, and a probe that
    /// looked at `media.sourceImage` would not know).
    static func outgoing(_ controller: GenerateController, on host: MoldHost,
                         hosts: HostStore) -> GenerateRequest? {
        guard let model = controller.modelName else { return nil }
        return RenderRequest.batch(
            controller.draft, model: model, copies: 1, randomBase: 0,
            maxIdentityPhotos: hosts.capabilities(of: host)?.maxIdentityPhotos ?? 0
        ).first
    }

    private static func code(of error: any Error) -> String? {
        guard case let .http(_, code, _)? = error as? MoldClientError else { return nil }
        return code
    }
}
