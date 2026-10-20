import Foundation

/// Display-only descriptors; they never become fake request media authorities.
public struct VideoOriginalReference: Identifiable, Hashable, Sendable {
    public let id: Int
    public let descriptor: ReferenceProvenance
    public let member: RetainedSourceMedia.Member?

    public static func ordered(metadata: OutputMetadata, inventory: RetainedSourceMedia.Inventory) -> [Self] {
        let references: [ReferenceProvenance]
        let stage: Int?
        if let top = metadata.references, !top.isEmpty { references = top; stage = nil }
        else if let stages = metadata.chain?.stages, let last = stages.last {
            references = last.generationReferences ?? []; stage = stages.count - 1
        } else { return [] }
        return references.enumerated().map { index, descriptor in
            let member: RetainedSourceMedia.Member? = if inventory.availability == .available, let digest = descriptor.sha256 {
                inventory.members.first { candidate in
                    let correctRole = stage.map { candidate.role == "stage_reference:\($0):\(index)" }
                        ?? (candidate.role == "references" || candidate.role.hasPrefix("stage_reference:"))
                    return correctRole && candidate.contentSha256 == digest
                }
            } else { nil }
            return .init(id: index, descriptor: descriptor, member: member)
        }
    }
}
