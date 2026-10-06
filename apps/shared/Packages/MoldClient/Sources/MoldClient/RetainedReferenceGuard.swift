import Foundation

/// Retained pins preserve the original reference order. Media edits put that authority down.
public nonisolated enum RetainedReferenceGuard {
    /// Visible typed references stay attached while ordinary authoring controls change.
    /// Hidden legacy roles keep the conservative whole-draft fence.
    public static func canReuseDraft(_ draft: RenderDraft, original: RenderDraft) -> Bool {
        if original.media.generationReferences.isEmpty { return draft == original }
        return draft.media == original.media && draft.pipeline == original.pipeline
    }
    public static func canHydrate(references: [GenerationReference], original: [GenerationReference],
                                  members: [RetainedSourceMedia.Member]) -> Bool {
        !references.isEmpty && references == original
            && references.allSatisfy { $0.media.authority == "descriptor" }
            && members.filter { $0.role == "references" }.count == references.count
    }
}
