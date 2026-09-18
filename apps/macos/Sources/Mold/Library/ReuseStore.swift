import Foundation
import MoldClient

/// What a Use These Settings is holding: which print's retained source media
/// the next render may be hydrated from, and anything the person should be
/// told about it.
///
/// Its own store rather than more state on `GenerateController`, which is
/// already over the type-size budget -- and because this outlives the click:
/// the probe runs while the recipe is already back in the draft, and the
/// answer arrives after the pane has drawn.
///
/// The RULE the whole thing rests on (`CLAUDE.md`, "Durable gallery source
/// media"): every client ALWAYS asks, because `OutputMetadata` under-reports
/// what a host kept -- inline `source_video`, `audio_file` and `mask_image`
/// bytes leave no marker at all. The metadata decides only whether an
/// unavailable answer is worth a sentence.
@MainActor
@Observable
final class ReuseStore {
    let hosts: HostStore

    /// The print the draft on screen came from, once a machine has answered
    /// for it. `nil` means nothing to hydrate from -- a fresh draft, a reuse
    /// still in flight, or an edit that invalidated one.
    private(set) var authority: Authority?
    /// The draft the reuse landed in. The authority is good only while the
    /// draft still IS that one -- see `pending(for:)`.
    private var restored: RenderDraft?
    /// What to say about a print whose media cannot be restored. Shown once,
    /// beside the prompt, and cleared by the next reuse or the next submit.
    var notice: String?
    /// Monotonic fence. A second Use These Settings while the first probe is
    /// in the air must win: the older answer is for a print nobody is looking
    /// at any more, and installing it would hydrate the new render from the
    /// old print's archive.
    private var version = 0

    init(hosts: HostStore) { self.hosts = hosts }

    /// One print's retained media, on the ONE machine that holds it.
    struct Authority: Equatable, Sendable {
        let filename: String
        /// The machine whose archive owns the bytes -- never the machine the
        /// next render is going to, which may be another one entirely.
        let origin: MoldHost.ID
        let members: [RetainedSourceMedia.Member]
    }

    /// Opens a reuse. Clears whatever the last one left and returns the fence
    /// the probe must still be current against.
    @discardableResult
    func begin() -> Int {
        version += 1
        authority = nil
        restored = nil
        notice = nil
        return version
    }

    /// Whether the answer in hand is still about the print on screen.
    func isCurrent(_ fence: Int) -> Bool { fence == version }
    /// The fence a step started AFTER the probe must still be current against.
    var currentFence: Int { version }

    /// Asks EVERY known copy of the print, in order, and keeps the first
    /// machine that can actually hand the media over.
    ///
    /// The Library lists a print once per machine that holds it, and mirroring
    /// an output does not copy the producing host's private archive -- so a
    /// copy answering `unavailable_legacy` says nothing about the machine that
    /// made it. A concrete archive or auth failure is preferred over a
    /// mirror's blank, and one unreachable copy must not hide a reachable
    /// archive (port of `useReuseStillPrint.ts:45-84`).
    func probe(_ copies: [PrintID], fence: Int, disclosing metadata: OutputMetadata) async {
        var unavailable: (PrintID, RetainedSourceMedia.Availability)?
        for copy in copies {
            guard isCurrent(fence) else { return }
            guard let client = hosts.backend(for: copy.host) else { continue }
            guard let inventory = try? await client.retainedSourceMedia(for: copy.filename)
            else { continue }
            guard isCurrent(fence) else { return }
            if inventory.availability == .available {
                authority = Authority(filename: copy.filename, origin: copy.host,
                                      members: inventory.members)
                return
            }
            // Replace a held answer only with one that has something to SAY.
            // Studio's condition is the same two clauses, but studio has no
            // `.unknown` member -- this build added one deliberately, and
            // without this a newer machine answering a fifth state would
            // overwrite a concrete `legacy` and leave the person told nothing
            // at all (`disclosure(.unknown)` is nil).
            guard RetainedSourceMedia.disclosure(inventory.availability) != nil
            else { continue }
            if unavailable == nil || unavailable?.1 == .unavailableLegacy {
                unavailable = (copy, inventory.availability)
            }
        }
        guard isCurrent(fence), let unavailable else { return }
        // Only a print whose OWN metadata says conditioning bytes shipped is
        // worth a sentence. A text-to-image print's archive entry has no pins
        // either, and the host cannot tell the two apart.
        guard RetainedSourceMedia.disclosable(metadata) else { return }
        // Never over the top of one already said: the reuse may have had
        // something more immediate to report -- a model this machine no
        // longer has -- and one line is one line.
        guard notice == nil else { return }
        notice = RetainedSourceMedia.disclosure(unavailable.1)
    }

    /// A long clip is rendered as a CHAIN JOB, and `POST /api/chain-jobs` is
    /// not one of the three doors that redeem a reuse session -- only
    /// `/api/generate`, `/api/generate/stream` and `/api/generation-batches`
    /// do (`routes.rs:3080`, `:3475`, `:4662`). So a clip past the
    /// checkpoint's own clip size cannot be hydrated from the print's
    /// archive, and this SAYS so rather than rendering it without the picture
    /// it was supposed to start from.
    /// `outgoing` is the request the draft would actually build, so the
    /// warning is given ONLY when something would have been hydrated. Gated
    /// on the authority alone, it told a person who had attached the picture
    /// themselves to attach a picture.
    func warnIfTheRouteCannotCarryMedia(chained: Bool, outgoing: GenerateRequest?) {
        guard chained, let authority, let outgoing else { return }
        guard !RetainedSourceMedia.members(authority.members, forHydrating: outgoing)
            .isEmpty
        else { return }
        notice = "This clip is long enough that the machine renders it in "
            + "pieces, and that route can\u{2019}t restore the print\u{2019}s own source "
            + "picture. Attach one before developing."
    }

    /// Forgets the print entirely -- a new draft is not that print any more.
    func clear() {
        version += 1
        authority = nil
        restored = nil
        notice = nil
    }
}

// Putting the authority down again, which is the whole of this type's other
// half. An authority that is never released conditions renders nobody asked
// for and, when the print goes away, refuses every render after it.
extension ReuseStore {

    /// Records the draft the recipe landed in, AFTER the model was adopted --
    /// adoption clamps, parks and echoes the pipeline, so a snapshot taken
    /// before it would differ from the draft the pane actually shows and the
    /// authority would be dropped before anyone touched anything.
    func arm(_ draft: RenderDraft) { restored = draft }

    /// The authority, if it still describes the draft on screen.
    ///
    /// ANY edit puts it down. The draft is the whole recipe -- the prompt,
    /// the model, the canvas, the wells, the sampler -- so this is the
    /// cheapest honest form of desktop's rule that a new handoff supersedes
    /// the prior print's authority (`composer.ts:29-59`), and it is stricter
    /// than desktop needs to be because this app shows no restored picture in
    /// the well: nothing else would tell a person that the render they are
    /// now composing is still conditioned on somebody else's print.
    func pending(for draft: RenderDraft) -> Authority? {
        guard let authority, restored == draft else { return nil }
        return authority
    }

    /// The authority, CONSUMED. A handle is good for one admission and a
    /// relay's bytes are carried by the request that took them, so the submit
    /// that takes this is the last one to have it -- which is also what makes
    /// a print the machine can no longer honour refuse exactly one render
    /// instead of every one after it.
    /// An authority that is HELD is put down either way -- taken when the
    /// draft is still the one it came with, dropped when it is not, because a
    /// draft that has moved on is not that print any more and an authority
    /// nobody can see must not sit waiting for the edit to be undone.
    ///
    /// When nothing is held this disturbs NOTHING: a press must not bump the
    /// fence under a probe still in the air, nor wipe a sentence nobody has
    /// read yet.
    func take(for draft: RenderDraft) -> Authority? {
        guard authority != nil else { return nil }
        let taken = pending(for: draft)
        clear()
        return taken
    }

    /// What the pane says while a print's media is waiting to ride along.
    ///
    /// The available path used to be completely silent -- the person was told
    /// when the picture would NOT come back and never when it would, which is
    /// the disclosure exactly inverted.
    ///
    /// Only about what the host will still apply: a picture already placed in
    /// the well (`ReuseStore+Picture`) is said by the well, and a line about
    /// it here would announce the same thing twice.
    func attachmentSentence(for draft: RenderDraft) -> String? {
        guard let authority = pending(for: draft) else { return nil }
        let remaining = authority.members.filter {
            !($0.role == RetainedSourcePicture.carriedRole && draft.media.sourceImage != nil)
        }
        guard !remaining.isEmpty else { return nil }
        let machine = hosts.host(authority.origin)?.name ?? "its machine"
        let what = remaining.count == 1
            ? "the source media" : "\(remaining.count) source files"
        return "Using \(what) from \(authority.filename) on \(machine)."
    }
}

// A HANDLE IS NEVER HELD ACROSS AN EDIT, because one is never held at all.
// The host binds it to the sha256 of `target_request`, so the mint happens
// INSIDE the submit, against the exact request going out, and the handle is
// consumed by the very next call. There is no window in which the draft can
// move under a minted session, and nothing persists one -- `BatchAdmission`
// excludes it from its coding keys.
//
// An edit to a hydrated ROLE is handled structurally rather than by watching
// for it: `RetainedSourceMedia.members(_:forHydrating:)` reads the outgoing
// request at mint time and asks only for the roles it has no bytes for.
// Someone who reattached their own picture keeps it, the retained one is not
// requested, and the host's `RETAINED_MEDIA_REUSE_TARGET_CONFLICT` can never
// fire for something this client chose to send.
