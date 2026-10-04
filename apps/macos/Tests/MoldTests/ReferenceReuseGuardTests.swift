import Foundation
import MoldClient
import MoldClientTesting
import Testing
@testable import Mold

@MainActor
struct ReferenceReuseGuardTests {
    @Test func descriptorsCannotSubmitWithoutRetainedAuthority() {
        let backend = FakeBackend()
        let store = ReuseStore(hosts: HostStore(hosts: []) { _ in backend })
        var draft = RenderDraft()
        draft.media.generationReferences = [GenerationReference(
            kind: "image", media: .init(authority: "descriptor"), mimeType: "image/png",
            provenance: .init(name: "retained.png"), width: 32, height: 32)]
        #expect(store.referenceRefusal(for: draft) != nil)
        draft.media.generationReferences[0].media = .init(authority: "inline", data: "AAAA")
        #expect(store.referenceRefusal(for: draft) == nil)
    }
}
