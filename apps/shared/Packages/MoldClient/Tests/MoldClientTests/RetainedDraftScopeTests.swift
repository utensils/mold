import Foundation
import Testing
@testable import MoldClient

struct RetainedDraftScopeTests {
    private var original: RenderDraft {
        var draft = RenderDraft()
        draft.media.generationReferences = [GenerationReference(kind: "image",
            media: .init(authority: "descriptor"), mimeType: "image/png",
            provenance: .init(sha256: String(repeating: "a", count: 64)), width: 32, height: 32)]
        return draft
    }

    @Test func ordinaryControlsKeepUnchangedReferences() {
        let draft = original
        var edited = draft
        edited.prompt = "new prompt"; edited.width = 704; edited.height = 1280
        edited.seed = 9; edited.title = "new title"; edited.batchSize = 2
        #expect(RetainedReferenceGuard.canReuseDraft(edited, original: draft))
    }

    @Test func mediaAndPipelineChangesInvalidateAuthority() {
        let draft = original
        var edited = draft
        edited.media.generationReferences.removeAll()
        #expect(!RetainedReferenceGuard.canReuseDraft(edited, original: draft))
        edited = draft; edited.pipeline = "another-recipe"
        #expect(!RetainedReferenceGuard.canReuseDraft(edited, original: draft))
        edited = draft; edited.media.maskImage = "replacement"
        #expect(!RetainedReferenceGuard.canReuseDraft(edited, original: draft))
    }

    @Test func hiddenLegacyRolesKeepWholeDraftFence() {
        let draft = RenderDraft()
        var edited = draft; edited.prompt = "new prompt"
        #expect(!RetainedReferenceGuard.canReuseDraft(edited, original: draft))
    }
}
