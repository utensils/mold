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
    private var restoreFailed = false
    var canRetry: Bool { restoreFailed && !probing && selectedEntry != nil }
    private var expectedFields: Set<RetainedSourceMedia.Field> = []
    private var originalSourceRevision: Int?
    private var selectedEntry: LibraryEntry?
    var canDiscard: Bool { authority != nil || !expectedFields.isEmpty }
    private var originalReferences: [GenerationReference] = []

    func begin(_ draft: RenderDraft, metadata: OutputMetadata? = nil, sourceRevision: Int? = nil) -> Int {
        expectedFields = Self.expectedFields(in: metadata)
        originalSourceRevision = sourceRevision
        originalReferences = draft.media.generationReferences
        version += 1
        authority = nil
        restoreFailed = false
        notice = nil
        probing = true
        return version
    }

    func isCurrent(_ fence: Int, draft: RenderDraft) -> Bool {
        version == fence
    }

    func probe(_ entry: LibraryEntry, fence: Int, controller: GenerateController) async {
        defer { if version == fence { probing = false } }
        guard version == fence else { return }
        selectedEntry = entry
        var unavailable: RetainedSourceMedia.Availability?
        var candidate: (copy: LibraryEntry, inventory: RetainedSourceMedia.Inventory)?
        var coverage = -1
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
            let availableFields = Set(inventory.members.compactMap { member -> RetainedSourceMedia.Field? in
                let field = RetainedSourceMedia.fieldForRole[member.role]
                return field == .identityImages ? .identityImage : field
            })
            let covered = expectedFields.intersection(availableFields).count
            if covered > coverage {
                candidate = (copy, inventory)
                coverage = covered
            }
            if expectedFields.isSubset(of: availableFields) { break }
        }
        guard let (copy, inventory) = candidate,
              let backend = controller.hosts.backend(for: copy.hostID) else {
            if isCurrent(fence, draft: controller.draft), !expectedFields.isEmpty {
                restoreFailed = true
                notice = unavailable.flatMap(RetainedSourceMedia.disclosure)
                    ?? "The original media couldn't be restored. Reconnect its machine or attach it again."
            }
            return
        }
        guard isCurrent(fence, draft: controller.draft), !Task.isCancelled else { return }
        expectedFields.formUnion(inventory.members.compactMap { member -> RetainedSourceMedia.Field? in
            guard !RetainedSourceMedia.notReusableRoles.contains(member.role) else { return nil }
            let field = RetainedSourceMedia.fieldForRole[member.role]
            return field == .identityImages ? .identityImage : field
        })
        authority = Authority(filename: copy.print.filename, origin: copy.hostID, members: inventory.members)
        if let refusal = sourcePictureRefusal(in: controller.draft) {
            notice = refusal
            return
        }
        // Show the source in its ordinary well, so it can be replaced,
        // fitted or removed. Other retained roles hydrate at submission.
        let outgoing = controller.modelName.flatMap {
            RenderRequest.batch(controller.draft, model: $0, copies: 1, randomBase: 0,
                maxIdentityPhotos: controller.hosts.capabilities[copy.hostID]?.maxIdentityPhotos ?? 0).first
        }
        let sourceUnchanged = originalSourceRevision == nil || originalSourceRevision == controller.sourceMediaRevision
        if !sourceUnchanged { expectedFields.subtract([.sourceImage, .maskImage]) }
        if let outgoing, sourceUnchanged, controller.draft.media.sourceImage == nil,
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
                if controller.draft.media.sourceImage == nil,
                   originalSourceRevision == nil || originalSourceRevision == controller.sourceMediaRevision {
                    controller.draft.media.sourceImage = encoded
                    controller.draft.media.sourceImageName = member.displayName
                    controller.draft.media.sourceImageOriginal = encoded
                    controller.draft.media.sourceImageOriginalName = member.displayName
                    if controller.draft.media.maskImage == nil {
                        controller.draft.media.maskImage = controller.draft.media.parked.maskImage ?? pairedMask
                        controller.draft.media.parked.maskImage = nil
                    }
                }
                // These wells now carry ordinary authored media; removing
                // either is an explicit choice, never an archive revival.
                expectedFields.subtract([.sourceImage, .maskImage])
            } catch {
                guard isCurrent(fence, draft: controller.draft), !Task.isCancelled else { return }
                // An invisible source must not be applied after a failed preview.
                authority = nil
                restoreFailed = true
                notice = "The source media couldn't be restored. Attach it again before generating."
                return
            }
        }
        // A source in the well is now ordinary authored media. Removing
        // it must never revive a hidden archive attachment.
        if controller.draft.media.sourceImage != nil {
            expectedFields.subtract([.sourceImage, .maskImage])
        }
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
        let restoredRequest = controller.modelName.flatMap {
            RenderRequest.batch(controller.draft, model: $0, copies: 1, randomBase: 0,
                maxIdentityPhotos: controller.hosts.capabilities[copy.hostID]?.maxIdentityPhotos ?? 0).first
        }
        if hasMissingFields(for: restoredRequest) {
            restoreFailed = true
            notice = "Some original media couldn't be restored. Attach it again or retry."
        }
    }

    func sourcePictureRefusal(in draft: RenderDraft) -> String? {
        guard draft.media.sourceImage == nil, let authority,
              authority.members.filter({ RetainedSourceMedia.fieldForRole[$0.role] == .sourceImage }).count > 1 else { return nil }
        return "This archive has multiple source pictures for one input. Attach the picture to use before generating."
    }

    /// Each press hydrates a fresh request. A failed admission or another
    /// press keeps these disclosed files until explicit dismissal/selection.
    func snapshot() -> Authority? { authority }

    func canHydrateReferences(_ references: [GenerationReference]) -> Bool {
        guard !probing, let authority else { return false }
        return RetainedReferenceGuard.canHydrate(references: references, original: originalReferences,
            members: authority.members)
    }

    func restorationRefusal(for request: GenerateRequest?) -> String? {
        guard !probing, !expectedFields.isEmpty else { return nil }
        return hasMissingFields(for: request) ? "Restore or reattach this print’s source media before generating." : nil
    }

    private func hasMissingFields(for request: GenerateRequest?) -> Bool {
        let retained = Set((authority?.members ?? []).compactMap { member -> RetainedSourceMedia.Field? in
            let field = RetainedSourceMedia.fieldForRole[member.role]
            return field == .identityImages ? .identityImage : field
        })
        return expectedFields.contains { field in
            !retained.contains(field) && (request?.isVacant(field) ?? true)
        }
    }

    func retry(controller: GenerateController) {
        guard let entry = selectedEntry, !probing else { return }
        // Retry the selected archive with the original authoring fences.
        // A retry never makes edited descriptors or removed wells "original".
        version += 1
        let fence = version
        authority = nil
        restoreFailed = false
        notice = nil
        probing = true
        Task { await probe(entry, fence: fence, controller: controller) }
    }

    private static func expectedFields(in metadata: OutputMetadata?) -> Set<RetainedSourceMedia.Field> {
        guard let metadata else { return [] }
        var fields: Set<RetainedSourceMedia.Field> = []
        if metadata.sourceImageSha256 != nil { fields.insert(.sourceImage) }
        if metadata.controlModel != nil || metadata.controlScale != nil { fields.insert(.controlImage) }
        if !metadata.identityDigests.isEmpty { fields.insert(.identityImage) }
        if !metadata.editImageDigests.isEmpty { fields.insert(.editImages) }
        if !(metadata.references ?? []).isEmpty { fields.insert(.references) }
        if !(metadata.keyframes ?? []).isEmpty { fields.insert(.keyframes) }
        if metadata.audioFilePath != nil { fields.insert(.audioFile) }
        if metadata.sourceVideoPath != nil { fields.insert(.sourceVideo) }
        if metadata.extendVideoPath != nil || metadata.extendOverlapFrames != nil { fields.insert(.extendVideo) }
        return fields
    }

    func clear() {
        expectedFields = []
        restoreFailed = false
        selectedEntry = nil
        originalSourceRevision = nil
        originalReferences = []
        version += 1
        authority = nil
        probing = false
        notice = nil
    }
}
