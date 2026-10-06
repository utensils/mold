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
    /// Visible typed references survive ordinary authoring edits. Hidden roles
    /// retain whole-draft fencing; selection changes are explicitly invalidated.
    func pending(for draft: RenderDraft) -> Authority? {
        guard let authority, let restored,
              RetainedReferenceGuard.canReuseDraft(draft, original: restored),
              authority.instance == nil || authority.instance == hosts.instanceID(of: authority.origin) else { return nil }
        return authority
    }

    /// Descriptor-only rows need the retained archive that grants their bytes.
    func availableFields(for draft: RenderDraft, request: GenerateRequest) -> Set<RetainedSourceMedia.Field> {
        guard let authority = pending(for: draft) else { return [] }
        return Set(RetainedSourceMedia.members(authority.members, forHydrating: request)
            .compactMap { RetainedSourceMedia.fieldForRole[$0.role] })
    }

    func referenceRefusal(for draft: RenderDraft) -> String? {
        if restoring { return notice ?? "Verifying retained conditioning on its original machine…" }
        if let authority = pending(for: draft), draft.media.sourceImage == nil,
           authority.members.filter({ RetainedSourceMedia.fieldForRole[$0.role] == .sourceImage }).count > 1 {
            return "This archive has multiple source pictures for one input. Attach the picture to use before generating."
        }
        let references = draft.media.generationReferences
        guard references.contains(where: { $0.media.authority == "descriptor" }) else { return nil }
        if RetainedReferenceGuard.canHydrate(references: references,
            original: restored?.media.generationReferences ?? [],
            members: pending(for: draft)?.members ?? []) { return nil }
        return "The retained references are unavailable. Replace or remove them before generating."
    }

    /// Visible references keep their archive across submissions; each press
    /// mints a fresh session. Legacy hidden conditioning is consumed once so
    /// it cannot silently condition a later, unrelated draft.
    func take(for draft: RenderDraft) -> Authority? {
        guard authority != nil else { return nil }
        let taken = pending(for: draft)
        // These references are visible attachments, not a one-use session.
        // Each press mints its own session against the exact outgoing request.
        if taken != nil, !draft.media.generationReferences.isEmpty { return taken }
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
            !(RetainedSourceMedia.fieldForRole[$0.role] == .sourceImage && draft.media.sourceImage != nil)
        }
        guard !remaining.isEmpty else { return nil }
        let machine = hosts.host(authority.origin)?.name ?? "its machine"
        let what = remaining.count == 1
            ? "the source media" : "\(remaining.count) source files"
        if authority.members.contains(where: { $0.role.hasPrefix("stage_source:") }) {
            let source = authority.members.contains { $0.role == "stage_source:0" } ? " and source picture" : ""
            return "Reusing the first stage’s settings\(source). Other stage inputs remain in the retained archive on \(machine)."
        }
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
