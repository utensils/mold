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
        let route: MoldHost
        let instance: String?
    }
    private(set) var authority: Authority?
    private(set) var probing = false
    var notice: String?
    private var version = 0
    var restoreFailed = false
    var canRetry: Bool { restoreFailed && !probing && selectedEntry != nil }
    var expectedFields: Set<RetainedSourceMedia.Field> = []
    private var originalSourceRevision: Int?
    private var selectedEntry: LibraryEntry?
    var canDiscard: Bool { authority != nil || !expectedFields.isEmpty }
    var originalRevisions: [RetainedSourceMedia.Field: Int] = [:]
    var settledFields: Set<RetainedSourceMedia.Field> = []
    weak var hosts: HostStore?
    private var originalReferences: [GenerationReference] = []

    func begin(_ draft: RenderDraft, metadata: OutputMetadata? = nil, sourceRevision: Int? = nil, mediaRevisions: [RetainedSourceMedia.Field: Int] = [:]) -> Int {
        originalRevisions = mediaRevisions
        settledFields = []
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
        hosts = controller.hosts
        selectedEntry = entry
        var unavailable: RetainedSourceMedia.Availability?
        var candidate: (copy: LibraryEntry, inventory: RetainedSourceMedia.Inventory, route: MoldHost, instance: String?)?
        var coverage = -1
        for copy in entry.everyCopy {
            guard isCurrent(fence, draft: controller.draft), !Task.isCancelled else { return }
            guard let route = controller.hosts.host(copy.hostID) else { continue }
            let instance = controller.hosts.instanceID(of: copy.hostID)
            let backend = controller.hosts.backend(for: route)
            guard let inventory = try? await backend.retainedSourceMedia(for: copy.print.filename),
                  controller.hosts.host(copy.hostID) == route,
                  controller.hosts.instanceID(of: copy.hostID) == instance else { continue }
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
                candidate = (copy, inventory, route, instance)
                coverage = covered
            }
            if expectedFields.isSubset(of: availableFields) { break }
        }
        guard let (copy, inventory, route, instance) = candidate else {
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
        let captured = Authority(filename: copy.print.filename, origin: copy.hostID,
            members: inventory.members, route: route, instance: instance)
        authority = captured
        if let refusal = sourcePictureRefusal(in: controller.draft) {
            notice = refusal
            return
        }
        guard await materialize(captured, fence: fence, controller: controller) else {
            if isCurrent(fence, draft: controller.draft) { authority = nil }
            return
        }
        guard isCurrent(fence, draft: controller.draft) else { return }
        let remaining = inventory.members.filter { member in
            guard let field = RetainedSourceMedia.draftField(for: member) else { return true }
            return !settledFields.contains(field)
        }
        authority = remaining.isEmpty ? nil : Authority(filename: copy.print.filename,
            origin: copy.hostID, members: remaining, route: route, instance: instance)
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
    func snapshot() -> Authority? {
        guard let authority, let hosts, originIsCurrent(authority, hosts: hosts) else { return nil }
        return authority
    }

    func canHydrateReferences(_ references: [GenerationReference]) -> Bool {
        guard !probing, let authority = snapshot() else { return false }
        return RetainedReferenceGuard.canHydrate(references: references, original: originalReferences,
            members: authority.members)
    }

    func restorationRefusal(for request: GenerateRequest?) -> String? {
        guard !probing, !expectedFields.isEmpty else { return nil }
        return hasMissingFields(for: request) ? "Restore or reattach this print’s source media before generating." : nil
    }

    private func hasMissingFields(for request: GenerateRequest?) -> Bool {
        let retained = Set((snapshot()?.members ?? []).compactMap { member -> RetainedSourceMedia.Field? in
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
        originalRevisions = [:]
        settledFields = []
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
