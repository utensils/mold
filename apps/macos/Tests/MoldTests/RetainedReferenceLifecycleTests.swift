import Foundation
import MoldClient
import Testing
@testable import Mold

@MainActor
struct RetainedReferenceLifecycleTests {
    private func fixture() async -> (ReuseStore, RenderDraft) {
        let host = MoldHost(name: "Origin", baseURL: URL(string: "http://fixture")!)
        let backend = FakeBackend(host: host)
        let members = (0..<4).map { RetainedSourceMedia.Member(memberId: "r\($0)", role: "references", displayName: "Image", sizeBytes: 4) }
        backend.retainedInventories["clip.mp4"] = .init(availability: .available, members: members)
        let store = ReuseStore(hosts: HostStore(hosts: [host]) { _ in backend })
        var draft = RenderDraft()
        draft.media.generationReferences = (0..<4).map { GenerationReference(kind: "image",
            media: .init(authority: "descriptor"), mimeType: "image/png",
            provenance: .init(sha256: String(repeating: String($0), count: 64)), width: 32, height: 32) }
        let metadata = try! MoldJSON.decoder.decode(OutputMetadata.self, from: Data(#"{"prompt":"fixture","model":"fixture","seed":1,"steps":4,"guidance":0,"width":32,"height":32,"version":"fixture"}"#.utf8))
        let fence = store.begin()
        store.arm(draft)
        await store.probe([PrintID(host: host.id, filename: "clip.mp4")], fence: fence, disclosing: metadata)
        return (store, draft)
    }

    @Test func editsAndRepeatedPressesKeepVisibleReferences() async {
        let (store, original) = await fixture()
        var draft = original
        draft.prompt = "edited"; draft.seed = 42; draft.width = 704; draft.height = 1280
        #expect(store.referenceRefusal(for: draft) == nil)
        #expect(store.take(for: draft) != nil)
        #expect(store.take(for: draft) != nil)
        #expect(store.referenceRefusal(for: draft) == nil)
    }

    @Test func partialReplacementNeverHydratesTheWrongOrderedReference() async {
        let (store, original) = await fixture()
        var draft = original
        draft.media.generationReferences.removeFirst()
        #expect(store.referenceRefusal(for: draft) != nil)
        draft = original
        draft.media.generationReferences[0].media = .init(authority: "inline", data: "replacement")
        #expect(store.referenceRefusal(for: draft) != nil)
        store.clear()
        #expect(store.referenceRefusal(for: original) != nil)
    }

    @Test func resetDoesNotRestoreHiddenArchiveAttachments() async {
        let (store, _) = await fixture()
        store.clear()
        #expect(store.pending(for: RenderDraft()) == nil)
    }
}
