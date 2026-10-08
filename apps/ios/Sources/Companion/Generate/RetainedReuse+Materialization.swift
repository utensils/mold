import Foundation
import MoldClient

extension RetainedReuse {
    func originIsCurrent(_ authority: Authority, hosts: HostStore) -> Bool {
        hosts.host(authority.origin) == authority.route
            && hosts.instanceID(of: authority.origin) == authority.instance
    }

    /// Visible media becomes ordinary authoring. Its archive role is retired
    /// even after a later removal; Retry cannot silently bring it back.
    func materialize(_ captured: Authority, fence: Int, controller: GenerateController) async -> Bool {
        let inventoryFields = Set(captured.members.compactMap(RetainedSourceMedia.draftField))
            .intersection(RetainedSourceMedia.materializableFields)
        func unchanged(_ field: RetainedSourceMedia.Field) -> Bool {
            originalRevisions[field, default: 0] == controller.mediaRevisions[field, default: 0]
        }
        let changed = inventoryFields.filter { !unchanged($0) }
        settledFields.formUnion(changed)
        expectedFields.subtract(changed)
        var wanted = inventoryFields.subtracting(settledFields)
            .intersection(RetainedSourceMedia.vacantDraftFields(in: controller.draft.media))
        if !wanted.contains(.sourceImage) {
            wanted.remove(.maskImage)
            settledFields.insert(.maskImage)
            expectedFields.remove(.maskImage)
        }
        let members = captured.members.filter { RetainedSourceMedia.draftField(for: $0).map(wanted.contains) == true }
        guard !members.isEmpty else { return true }
        do {
            if let refusal = RetainedSourceMedia.relayRefusal(members, copies: 1) { throw refusal }
            let backend = controller.hosts.backend(for: captured.route)
            var downloaded: [(member: RetainedSourceMedia.Member, bytes: Data)] = []
            var bodyBytes = 0
            for member in members {
                let bytes = try await backend.retainedSourceMediaBytes(for: captured.filename, member: member.memberId)
                guard isCurrent(fence, draft: controller.draft), !Task.isCancelled else { return false }
                guard originIsCurrent(captured, hosts: controller.hosts) else { throw URLError(.cancelled) }
                bodyBytes += (bytes.count + 2) / 3 * 4
                guard bodyBytes <= RequestBodyLimit.bytes else {
                    throw RetainedSourceMedia.RelayFailure.tooLarge(bytes: bodyBytes, copies: 1)
                }
                downloaded.append((member, bytes))
            }
            // Check revisions again after all awaits, including an attachment
            // added and then removed while another role was downloading.
            let superseded = wanted.filter { !unchanged($0) }
            settledFields.formUnion(superseded)
            expectedFields.subtract(superseded)
            wanted.subtract(superseded)
            if !wanted.contains(.sourceImage) {
            wanted.remove(.maskImage)
            settledFields.insert(.maskImage)
            expectedFields.remove(.maskImage)
        }
            downloaded.removeAll { !wanted.contains(RetainedSourceMedia.draftField(for: $0.member)!) }
            var placed = try RetainedSourceMedia.materializedDraft(downloaded, into: controller.draft)
            if let recipe = controller.recipe {
                placed.media.reconcile(for: recipe.capabilities, family: controller.model?.family, model: controller.modelName)
                BoundaryFramePolicy.apply(to: &placed, capabilities: recipe.capabilities)
            }
            controller.draft = placed
            settledFields.formUnion(wanted)
            expectedFields.subtract(wanted)
            return true
        } catch {
            guard isCurrent(fence, draft: controller.draft), !Task.isCancelled else { return false }
            restoreFailed = true
            notice = "The original input files couldn't be restored. Reconnect their machine or attach them again."
            return false
        }
    }
}
