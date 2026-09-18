import Foundation
import MoldClient

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
