import Foundation
import MoldClient
import Testing
@testable import Mold

@MainActor
struct DraftInputRecoveryTests {
    private func directory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    @Test func snapshotRecoveryRequiresExplicitCurrentInputConfirmation() throws {
        let directory = try directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = DraftStore(directory: directory)
        let descriptor = DraftDescriptor(RenderDraft(), model: "model", family: nil, recipeID: nil)
        #expect(store.save(descriptor, inputs: DraftInputSnapshot(DraftMedia())))
        let original = try #require(store.load())
        let snapshot = directory.appending(path: "generate-draft-inputs/\(try #require(original.localInputsSHA256)).json")
        try FileManager.default.removeItem(at: snapshot)
        let persistence = DraftPersistence(store: store, delay: .zero)
        #expect(persistence.restore() != nil)
        #expect(persistence.recoveryRefusal != nil)
        var replacement = DraftMedia(); replacement.sourceImage = "AQID"
        persistence.flush(descriptor, inputs: DraftInputSnapshot(replacement))
        #expect(store.load()?.localInputsSHA256 == original.localInputsSHA256)
        #expect(persistence.recoveryRefusal != nil)
        persistence.discardUnavailableInputs()
        persistence.flush(descriptor, inputs: DraftInputSnapshot(replacement))
        #expect(persistence.recoveryRefusal == nil)
        #expect(try store.loadInputs(for: try #require(store.load())) == DraftInputSnapshot(replacement))
    }

    @Test func failedInputWriteDisplaysSaveNoticeAndKeepsThePriorDraft() throws {
        let directory = try directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = DraftStore(directory: directory)
        let descriptor = DraftDescriptor(RenderDraft(), model: "original", family: nil, recipeID: nil)
        #expect(store.save(descriptor))
        try Data().write(to: directory.appending(path: "generate-draft-inputs"))
        let persistence = DraftPersistence(store: store)
        var new = descriptor; new.model = "new"
        persistence.flush(new, inputs: DraftInputSnapshot(DraftMedia()))
        #expect(persistence.saveNotice?.contains("could not be saved") == true)
        #expect(store.load()?.model == "original")
    }

    @Test func savedLocatorNeverAuthorizesChangedDescriptorSlots() throws {
        let metadata = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data(#"{"model":"fixture","references":[{"kind":"image","mime_type":"image/png","sha256":"abc","width":32,"height":32}]}"#.utf8))
        let saved = SavedReuse(origin: UUID(), instance: "instance", filename: "clip.mp4", model: "fixture", recipe: nil, metadata: metadata, archive: "archive", output: "output")
        var media = RenderDraft(reusing: metadata).media
        let original = DraftInputSnapshot(media, retainedReuseFingerprint: saved.fingerprint)
        #expect(saved.acceptsSnapshot(original, model: "fixture", recipe: nil))
        #expect(!saved.acceptsSnapshot(original, model: "different", recipe: nil))
        #expect(!saved.acceptsSnapshot(original, model: "fixture", recipe: "other"))
        media.generationReferences[0].provenance?.sha256 = "changed"
        let changed = DraftInputSnapshot(media, retainedReuseFingerprint: saved.fingerprint)
        #expect(!saved.acceptsSnapshot(changed, model: "fixture", recipe: nil))
        let stale = DraftInputSnapshot(RenderDraft(reusing: metadata).media, retainedReuseFingerprint: "old")
        #expect(!saved.acceptsSnapshot(stale, model: "fixture", recipe: nil))
    }
}
