import Foundation
import MoldClient
import Testing
@testable import Mold

@MainActor
struct SavedReuseTests {
    @Test func locatorStoresNoBytesOrSessionAndRestartsFailClosed() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = SavedReuseFile(directory: directory)
        let metadata = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data(#"{"prompt":"fixture","model":"fixture","seed":1,"steps":4,"guidance":0,"width":32,"height":32,"version":"fixture","references":[{"kind":"image","mime_type":"image/png","sha256":"abc","width":32,"height":32}]}"#.utf8))
        let saved = SavedReuse(origin: UUID(), instance: "instance", filename: "clip.mp4", model: "fixture", recipe: nil, metadata: metadata, archive: "archive", output: "output")
        file.save(saved)
        #expect(file.load() == saved)
        let text = try String(contentsOf: file.url, encoding: .utf8)
        #expect(!text.contains("session_handle"))
        #expect(!text.contains("member_id"))
        #expect(!text.contains("\"data\""))
        let hosts = HostStore(hosts: [])
        let controller = GenerateController(hosts: hosts, defaults: ConfigStore(hosts: hosts))
        let store = ReuseStore(hosts: hosts, savedFile: file)
        store.restoreSaved(into: controller)
        #expect(controller.draft.media.generationReferences.count == 1)
        #expect(store.referenceRefusal(for: controller.draft) != nil)
        #expect(store.authority == nil)
        store.clear()
        #expect(file.load() == nil)
    }
    private func fixture() throws -> (ReuseStore, GenerateController, FakeBackend, MoldHost) {
        let host = MoldHost(name: "Origin", baseURL: URL(string: "http://fixture")!)
        let backend = FakeBackend(host: host)
        let hosts = HostStore(hosts: [host]) { _ in backend }
        hosts.instanceIDs[host.id] = "original"
        let metadata = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data(#"{"prompt":"fixture","model":"fixture","seed":1,"steps":4,"guidance":0,"width":32,"height":32,"version":"fixture","references":[{"kind":"image","mime_type":"image/png","sha256":"abc","width":32,"height":32}]}"#.utf8))
        backend.retainedTransferOffers["clip.mp4"] = .init(archiveIdentitySha256: "archive", members: [], outputSha256: "output", outputSizeBytes: 4, metadata: metadata)
        backend.retainedInventories["clip.mp4"] = .init(availability: .available, members: [.init(memberId: "r0", role: "references", displayName: "image", sizeBytes: 4)])
        let controller = GenerateController(hosts: hosts, defaults: ConfigStore(hosts: hosts))
        controller.modelName = "fixture"
        let store = ReuseStore(hosts: hosts)
        store.savedRecipe = .init(origin: host.id, instance: "original", filename: "clip.mp4", model: "fixture", recipe: nil, metadata: metadata, archive: "archive", output: "output")
        store.selectionModel = "fixture"
        store.restoring = true
        controller.draft = RenderDraft(reusing: metadata)
        return (store, controller, backend, host)
    }

    @Test func verifiedOriginRestoresVisibleReferences() async throws {
        let (store, controller, _, _) = try fixture()
        await store.recoverSaved(controller)
        #expect(!store.restoring)
        #expect(store.referenceRefusal(for: controller.draft) == nil)
    }

    @Test func replacingServerDuringInventoryCannotUnblockSavedReferences() async throws {
        let (store, controller, backend, host) = try fixture()
        var finish: CheckedContinuation<RetainedSourceMedia.Inventory, Error>?
        let inventory = try #require(backend.retainedInventories["clip.mp4"])
        backend.retainedInventoryResponder = { _ in try await withCheckedThrowingContinuation { finish = $0 } }
        let recovering = Task { await store.recoverSaved(controller) }
        await settle { finish != nil }
        store.hosts.instanceIDs[host.id] = "replacement"
        try #require(finish).resume(returning: inventory)
        await recovering.value
        #expect(store.restoring)
        #expect(store.referenceRefusal(for: controller.draft) != nil)
    }

    @Test func editedMediaBeforeRememberIsPersistedAsInvalidated() async throws {
        let (store, controller, _, host) = try fixture()
        store.restoring = false
        let metadata = try #require(store.savedRecipe).metadata
        store.savedRecipe = nil
        store.arm(controller.draft)
        await store.probe([.init(host: host.id, filename: "clip.mp4")], fence: store.currentFence, disclosing: metadata)
        controller.draft.media.generationReferences.removeAll()
        await store.remember(metadata, model: "fixture", recipe: nil, draft: controller.draft, fence: store.currentFence)
        #expect(store.savedRecipe?.invalidated == true)
    }

    @Test func replacementOriginAndPartialEditStayBlockedAcrossRetry() async throws {
        let (store, controller, _, host) = try fixture()
        controller.draft.media.generationReferences.removeAll()
        store.selectionChanged(model: "fixture", recipe: nil, draft: controller.draft)
        #expect(store.savedRecipe?.invalidated == true)
        await store.recoverSaved(controller)
        #expect(store.restoring)
        store.hosts.instanceIDs[host.id] = "replacement"
        await store.recoverSaved(controller)
        #expect(store.restoring)
    }

    @Test func ordinaryProbeCannotLabelOldInventoryWithNewInstance() async throws {
        let (store, controller, backend, host) = try fixture()
        var finish: CheckedContinuation<RetainedSourceMedia.Inventory, Error>?
        let inventory = try #require(backend.retainedInventories["clip.mp4"])
        backend.retainedInventoryResponder = { _ in try await withCheckedThrowingContinuation { finish = $0 } }
        store.restoring = false
        store.arm(controller.draft)
        let metadata = try #require(store.savedRecipe).metadata
        let probing = Task { await store.probe([.init(host: host.id, filename: "clip.mp4")],
            fence: store.currentFence, disclosing: metadata) }
        await settle { finish != nil }
        store.hosts.instanceIDs[host.id] = "replacement"
        try #require(finish).resume(returning: inventory)
        await probing.value
        #expect(store.authority == nil)
    }

    @Test func initialLegacyAdoptionDoesNotInvalidateMediaBaseline() throws {
        let (store, controller, _, _) = try fixture()
        let metadata = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data(#"{"prompt":"fixture","model":"fixture","seed":1,"steps":4,"guidance":0,"width":32,"height":32,"version":"fixture"}"#.utf8))
        let old = try #require(store.savedRecipe)
        store.savedRecipe = .init(origin: old.origin, instance: old.instance, filename: old.filename,
            model: old.model, recipe: old.recipe, metadata: metadata, archive: old.archive, output: old.output)
        controller.draft = RenderDraft(reusing: metadata)
        controller.draft.width = 1024
        controller.draft.prompt = "ordinary recalled draft"
        store.selectionChanged(model: "fixture", recipe: nil, draft: controller.draft)
        #expect(store.savedRecipe?.invalidated == false)
    }

    @Test func realH3AdoptionPreservesTheSavedConditioningBaseline() async throws {
        let (store, controller, backend, host) = try fixture()
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let document = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appending(path: "docs/generated/generation-profiles-v1.json"))) as! [String: Any]
        let profiles = document["profiles"] as! [[String: Any]]
        let entry = try #require(profiles.first { String(describing: $0["models"] ?? "").contains("minimax-h3-ref2va:") })
        let body: [String: Any] = ["name": "fixture", "description": "Fixture", "family": "minimax-h3", "generation_profile": entry["profile"]!]
        let model = try MoldJSON.decoder.decode(Model.self, from: JSONSerialization.data(withJSONObject: body))
        controller.adopt(model: model, on: host.id, keepingDraft: true)
        store.adoptRestoredBaseline(controller, model: model)
        store.selectionChanged(model: controller.modelName, recipe: controller.recipeID, draft: controller.draft)
        #expect(store.savedRecipe?.invalidated == false)
        #expect(controller.draft.media.adoptedReferenceCapabilities != nil)
        await store.recoverSaved(controller)
        #expect(!store.restoring)
        #expect(store.referenceRefusal(for: controller.draft) == nil)
        #expect(backend.retainedInventoryRequests == ["clip.mp4"])
    }

}
