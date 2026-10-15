import Foundation
import MoldClient
import Testing
@testable import Mold

@MainActor struct LocalReusePersistenceTests {
    @Test func materializedSourceSurvivesPromptEditsAndOfflineRestartWithoutWarning() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = SavedReuseFile(directory: directory)
        let host = MoldHost(name: "Origin", baseURL: URL(string: "http://fixture")!)
        let backend = FakeBackend(host: host)
        let hosts = HostStore(hosts: [host]) { _ in backend }
        hosts.instanceIDs[host.id] = "original"
        let metadata = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data(#"{"model":"fixture","source_image_name":"source.png"}"#.utf8))
        backend.retainedInventories["image.png"] = .init(availability: .available, members: [
            .init(memberId: "source", role: "source_image", displayName: "source.png", sizeBytes: 3)
        ])
        backend.retainedMemberBytes["source"] = Data([1, 2, 3])
        let controller = GenerateController(hosts: hosts, defaults: ConfigStore(hosts: hosts))
        controller.modelName = "fixture"
        controller.draft = RenderDraft(reusing: metadata)
        let reuse = ReuseStore(hosts: hosts, savedFile: file)
        reuse.arm(controller.draft)
        await reuse.probe([.init(host: host.id, filename: "image.png")], fence: reuse.currentFence, disclosing: metadata)
        await reuse.remember(metadata, model: "fixture", recipe: nil, draft: controller.draft, fence: reuse.currentFence)
        controller.draft = try #require(await reuse.placePicture(in: controller.draft, outgoing: nil, live: { controller.draft }))
        #expect(controller.draft.media.sourceImage == "AQID")
        controller.draft.prompt = "an ordinary prompt edit"
        reuse.selectionChanged(model: "fixture", recipe: nil, draft: controller.draft)
        let fingerprint = reuse.persistenceFingerprint(for: controller.draft, model: "fixture", recipe: nil)
        #expect(fingerprint == nil, "Fully local inputs must not depend on a stale archive locator")
        let inputs = DraftInputSnapshot(controller.draft.media, retainedReuseFingerprint: fingerprint)
        // A restart has its durable local bytes, but no reachable origin.
        let offline = HostStore(hosts: [])
        let restartedController = GenerateController(hosts: offline, defaults: ConfigStore(hosts: offline))
        try inputs.apply(to: &restartedController.draft.media)
        let restarted = ReuseStore(hosts: offline, savedFile: file)
        restarted.restoreSaved(into: restartedController, preserving: inputs, model: "fixture")
        #expect(restartedController.draft.media.sourceImage == "AQID")
        #expect(restarted.notice == nil)
        #expect(!restarted.restoring)
        #expect(restarted.referenceRefusal(for: restartedController.draft) == nil)
        #expect(restarted.authority == nil)
    }

    @Test(arguments: ["missing", "stale", "descriptor", "legacy", "version"])
    func localCompletionCannotBypassUnresolvedOrMissingSnapshots(scenario: String) throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = SavedReuseFile(directory: directory)
        let metadata = try MoldJSON.decoder.decode(OutputMetadata.self, from: Data(#"{"model":"fixture","references":[{"kind":"image","mime_type":"image/png","sha256":"abc","width":32,"height":32}]}"#.utf8))
        var saved = SavedReuse(origin: UUID(), instance: "original", filename: "image.png", model: "fixture", recipe: nil, metadata: metadata)
        if scenario != "legacy" { saved.localInputsComplete = true }
        if scenario == "version" { saved.version = 2 }
        file.save(saved)
        var draft = RenderDraft()
        draft.media.sourceImage = "AQID"
        if scenario == "descriptor" { draft.media.generationReferences = RenderDraft(reusing: metadata).media.generationReferences }
        let inputs = DraftInputSnapshot(draft.media, retainedReuseFingerprint: scenario == "stale" ? "old" : nil)
        let hosts = HostStore(hosts: [])
        let controller = GenerateController(hosts: hosts, defaults: ConfigStore(hosts: hosts))
        controller.draft = draft
        let reuse = ReuseStore(hosts: hosts, savedFile: file)
        reuse.restoreSaved(into: controller, preserving: scenario == "missing" ? nil : inputs, model: "fixture")
        #expect(reuse.restoring)
        #expect(reuse.referenceRefusal(for: controller.draft) != nil)
        #expect(reuse.authority == nil)
    }

}
