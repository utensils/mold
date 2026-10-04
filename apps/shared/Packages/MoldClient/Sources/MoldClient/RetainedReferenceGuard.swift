import Foundation

/// Retained pins preserve the original reference order. Media edits put that authority down.
public nonisolated enum RetainedReferenceGuard {
    public static func canHydrate(references: [GenerationReference], original: [GenerationReference],
                                  members: [RetainedSourceMedia.Member]) -> Bool {
        !references.isEmpty && references == original
            && references.allSatisfy { $0.media.authority == "descriptor" }
            && members.filter { $0.role == "references" }.count == references.count
    }
}
