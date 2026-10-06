import Foundation
import MoldClient

/// A reuse probe belongs to one draft, not the next print opened while its
/// network request is in flight. No session or private archive path persists.
@Observable
final class RetainedReuse {
    struct Authority {
        let filename: String
        let origin: MoldHost.ID
        let members: [RetainedSourceMedia.Member]
    }
    private(set) var authority: Authority?
    private(set) var probing = false
    var notice: String?
    private var version = 0
    private var originalReferences: [GenerationReference] = []

    func begin(_ draft: RenderDraft) -> Int {
        originalReferences = draft.media.generationReferences
        version += 1
        authority = nil
        notice = nil
        probing = true
        return version
    }

    func isCurrent(_ fence: Int, draft: RenderDraft) -> Bool {
        version == fence
    }

    func probe(_ entry: LibraryEntry, fence: Int, controller: GenerateController) async {
        defer { if version == fence { probing = false } }
        var unavailable: RetainedSourceMedia.Availability?
        for copy in entry.everyCopy {
            guard isCurrent(fence, draft: controller.draft), !Task.isCancelled else { return }
            guard let backend = controller.hosts.backend(for: copy.hostID),
                  let inventory = try? await backend.retainedSourceMedia(for: copy.print.filename)
            else { continue }
            guard isCurrent(fence, draft: controller.draft), !Task.isCancelled else { return }
            guard inventory.availability == .available else {
                if RetainedSourceMedia.disclosure(inventory.availability) != nil,
                   unavailable == nil || unavailable == .unavailableLegacy {
                    unavailable = inventory.availability
                }
                continue
            }
            authority = Authority(filename: copy.print.filename, origin: copy.hostID, members: inventory.members)
            // Show the source in its ordinary well, so it can be replaced,
            // fitted or removed. Other retained roles hydrate at submission.
            let outgoing = controller.modelName.flatMap {
                RenderRequest.batch(controller.draft, model: $0, copies: 1, randomBase: 0,
                    maxIdentityPhotos: controller.hosts.capabilities[copy.hostID]?.maxIdentityPhotos ?? 0).first
            }
            if let outgoing, controller.draft.media.sourceImage == nil,
               let member = RetainedSourceMedia.members(inventory.members, forHydrating: outgoing)
                .first(where: { RetainedSourceMedia.fieldForRole[$0.role] == .sourceImage }) {
                do {
                    if let refusal = RetainedSourceMedia.relayRefusal([member], copies: 1) { throw refusal }
                    let bytes = try await backend.retainedSourceMediaBytes(for: copy.print.filename, member: member.memberId)
                    guard isCurrent(fence, draft: controller.draft), !Task.isCancelled else { return }
                    var pairedMask: String?
                    if controller.draft.media.maskImage == nil,
                       controller.draft.media.parked.maskImage == nil,
                       let mask = inventory.members.first(where: { $0.role == "mask_image" }) {
                        if let refusal = RetainedSourceMedia.relayRefusal([member, mask], copies: 1) { throw refusal }
                        let maskBytes = try await backend.retainedSourceMediaBytes(
                            for: copy.print.filename, member: mask.memberId)
                        guard isCurrent(fence, draft: controller.draft), !Task.isCancelled else { return }
                        pairedMask = maskBytes.base64EncodedString()
                    }
                    let encoded = bytes.base64EncodedString()
                    // A user-selected image wins even if it arrived while
                    // the retained bytes were in flight.
                    if controller.draft.media.sourceImage == nil {
                        controller.draft.media.sourceImage = encoded
                        controller.draft.media.sourceImageName = member.displayName
                        controller.draft.media.sourceImageOriginal = encoded
                        controller.draft.media.sourceImageOriginalName = member.displayName
                        if controller.draft.media.maskImage == nil {
                            controller.draft.media.maskImage = controller.draft.media.parked.maskImage ?? pairedMask
                            controller.draft.media.parked.maskImage = nil
                        }
                    }
                } catch {
                    guard isCurrent(fence, draft: controller.draft), !Task.isCancelled else { return }
                    // An invisible source must not be applied after a failed preview.
                    authority = nil
                    notice = "The source media couldn't be restored. Attach it again before generating."
                    return
                }
            }
            // A source in the well is now ordinary authored media. Removing
            // it must never revive a hidden archive attachment.
            let pairedSource = inventory.members.contains { RetainedSourceMedia.fieldForRole[$0.role] == .sourceImage }
            let remaining = inventory.members.filter {
                RetainedSourceMedia.fieldForRole[$0.role] != .sourceImage && !(pairedSource && $0.role == "mask_image")
            }
            authority = remaining.isEmpty ? nil : Authority(filename: copy.print.filename,
                origin: copy.hostID, members: remaining)
            if inventory.members.contains(where: { $0.role.hasPrefix("stage_source:") }) {
                let source = inventory.members.contains { $0.role == "stage_source:0" } ? " and source picture" : ""
                notice = "Reusing the first stage’s settings\(source). Other stage inputs remain in the retained archive."
            } else if !remaining.isEmpty {
                let files = remaining.count == 1 ? "a retained source file" : "\(remaining.count) retained source files"
                notice = "Using \(files) from \(copy.hostName)."
            }
            return
        }
        if isCurrent(fence, draft: controller.draft), RetainedSourceMedia.disclosable(entry.print.metadata),
           let unavailable { notice = RetainedSourceMedia.disclosure(unavailable) }
    }

    /// Each press hydrates a fresh request. A failed admission or another
    /// press keeps these disclosed files until explicit dismissal/selection.
    func snapshot() -> Authority? { authority }

    func canHydrateReferences(_ references: [GenerationReference]) -> Bool {
        guard !probing, let authority else { return false }
        return RetainedReferenceGuard.canHydrate(references: references, original: originalReferences,
            members: authority.members)
    }

    func clear() {
        originalReferences = []
        version += 1
        authority = nil
        probing = false
        notice = nil
    }
}
