import Foundation
import MoldClient

/// Getting a reused print's own picture onto a LONG clip.
///
/// `POST /api/chain-jobs` redeems no reuse session -- only `/api/generate`,
/// `/api/generate/stream` and `/api/generation-batches` do (`routes.rs:3080`,
/// `:3475`, `:4662`). But the chain WIRE carries the bytes per stage
/// (`source_image_b64`), so the relay needs no server support at all: fetch
/// the retained picture and put it in the draft's own source well, and the
/// render goes out as an ordinary long clip that happens to start from it.
///
/// Doing it HERE rather than inside the submit is what leaves
/// `ChainSubmission.take`'s synchronous ordering alone: nothing about which
/// press takes the canvas moves, because the authority is taken before the
/// await and a second press therefore finds none and goes straight down the
/// ordinary path.
@MainActor
enum ChainRetainedSource {

    /// The one role a chain body can carry. Anything else the print retained
    /// cannot ride this route whatever we do.
    static let carriedRole = "source_image"

    /// The member to fetch, if this authority has one a chain can use.
    static func member(
        of authority: ReuseStore.Authority, forHydrating outgoing: GenerateRequest?
    ) -> RetainedSourceMedia.Member? {
        guard let outgoing, outgoing.sourceImage == nil else { return nil }
        return RetainedSourceMedia.members(authority.members, forHydrating: outgoing)
            .first { $0.role == carriedRole }
    }

    /// The picture, or the sentence to show instead.
    ///
    /// It ANSWERS rather than writing through an `inout` draft, because the
    /// draft lives on an actor-isolated store and cannot be passed `inout`
    /// across an `await`; the caller applies it in one step when the bytes
    /// are in hand.
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

    /// Puts it in the well. The well is the point: on this route the picture
    /// stops being invisible authority the host applies and becomes an
    /// ordinary attachment a person can see, change and remove.
    static func place(_ picture: String, named name: String, in draft: inout RenderDraft) {
        draft.media.sourceImage = picture
        draft.media.sourceImageName = name
        // The picked picture too, so a later re-fit does not compound: here it
        // IS the original, never a fitted copy of one.
        draft.media.sourceImageOriginal = picture
        draft.media.sourceImageOriginalName = name
    }

    private static func code(of error: any Error) -> String? {
        guard case let .http(_, code, _)? = error as? MoldClientError else { return nil }
        return code
    }
}
