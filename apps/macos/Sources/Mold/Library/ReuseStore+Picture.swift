import Foundation
import MoldClient

// The print's own picture goes in the well, on every route.
extension ReuseStore {
    func sourceForSubmission(in draft: RenderDraft, outgoing: GenerateRequest?,
                             live: () -> RenderDraft, fence: Int, mediaRevisions: () -> [RetainedSourceMedia.Field: Int] = { [:] }) async -> RenderDraft? {
        guard isCurrent(fence) else { return nil }
        let placed = await placePicture(in: draft, outgoing: outgoing, live: live, mediaRevisions: mediaRevisions)
        guard isCurrent(fence) else { return nil }
        attachingSource = false
        return placed
    }

    /// The draft with the print's picture placed in its source well, or nil
    /// when there is nothing to place: no authority, no `source_image`
    /// member, or a well already holding a picture of the person's own.
    ///
    /// Runs AFTER the probe and re-arms on the placed draft, so the authority
    /// survives the one edit this store made itself. A fetch that fails is
    /// said once, the way the long-clip route already says it, and the well
    /// stays empty rather than pretending.
    ///
    /// `live` is the draft as it is NOW: a person who edited the prompt while
    /// the picture downloaded keeps that edit, and the print's picture is
    /// not placed over it -- the authority is theirs to have moved off.
    func placePicture(in draft: RenderDraft, outgoing: GenerateRequest?,
                      live: () -> RenderDraft, mediaRevisions: () -> [RetainedSourceMedia.Field: Int] = { [:] }) async -> RenderDraft? {
        guard let authority, let route = hosts.host(authority.origin), let original = restored else { return nil }
        let revisions = initialMediaRevisions ?? mediaRevisions()
        let inventoryFields = Set(authority.members.compactMap(RetainedSourceMedia.draftField))
            .intersection(RetainedSourceMedia.materializableFields)
        var fields = RetainedSourceMedia.vacantDraftFields(in: draft.media)
        fields = fields.filter { revisions[$0, default: 0] == mediaRevisions()[$0, default: 0] }
        var members = authority.members.filter { RetainedSourceMedia.draftField(for: $0).map(fields.contains) == true }
        if !members.contains(where: { RetainedSourceMedia.draftField(for: $0) == .sourceImage }) { members.removeAll { RetainedSourceMedia.draftField(for: $0) == .maskImage } }
        guard !members.isEmpty else { retireMaterializedFields(inventoryFields); return nil }
        let fence = currentFence
        attachingSource = true
        defer { if isCurrent(fence) { attachingSource = false } }
        let backend = hosts.backend(for: route)
        do {
            if let refusal = RetainedSourceMedia.relayRefusal(members, copies: 1) { throw refusal }
            var downloaded: [(member: RetainedSourceMedia.Member, bytes: Data)] = []
            var bodyBytes = 0
            for member in members {
                let bytes = try await backend.retainedSourceMediaBytes(for: authority.filename, member: member.memberId)
                guard isCurrent(fence), !Task.isCancelled else { return nil }
                guard (authority.route == nil || authority.route == hosts.host(authority.origin)), authority.instance == nil || authority.instance == hosts.instanceID(of: authority.origin), hosts.host(authority.origin) == route else {
                    restorationFailed = true
                    notice = "The source machine changed. Reselect the print before generating."
                    return nil
                }
                bodyBytes += (bytes.count + 2) / 3 * 4
                guard bodyBytes <= RequestBodyLimit.bytes else {
                    throw RetainedSourceMedia.RelayFailure.tooLarge(bytes: bodyBytes, copies: 1)
                }
                downloaded.append((member, bytes))
            }
            var superseded = inventoryFields.filter { revisions[$0, default: 0] != mediaRevisions()[$0, default: 0] }
            if superseded.contains(.sourceImage) { superseded.insert(.maskImage) }
            downloaded.removeAll { RetainedSourceMedia.draftField(for: $0.member).map(superseded.contains) == true }
            let liveDraft = live()
            var placed = try RetainedSourceMedia.materializedDraft(downloaded, into: liveDraft)
            if let capabilities = liveDraft.media.adoptedReferenceCapabilities {
                // Keep the already-adopted older-server reference policy when
                // no additive reference block was advertised.
                let references = placed.media.editImages
                let weight = placed.media.referenceWeight
                placed.media.reconcile(for: capabilities, model: selectionModel)
                if capabilities.referenceImages == nil && liveDraft.media.sourceMode != .single {
                    placed.media.sourceMode = liveDraft.media.sourceMode
                    placed.media.parked.referenceWeight = nil
                    placed.media.editImages = references
                    placed.media.referenceWeight = weight
                    placed.media.parked.editImages = []
                }
                BoundaryFramePolicy.apply(to: &placed, capabilities: capabilities)
            }
            if liveDraft.media.generationReferences != original.media.generationReferences { retireMaterializedFields([.references]) }
            retireMaterializedFields(inventoryFields)
            arm(placed)
            return placed
        } catch {
            if isCurrent(fence) { restorationFailed = true }
            if isCurrent(fence), notice == nil { notice = "The original input files couldn't be restored. Reconnect their machine or attach them again." }
            return nil
        }
    }
}
