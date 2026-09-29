import Foundation
import MoldClient
import MoldClientTesting
import Testing

@testable import MoldCompanion

/// Generate against a fake machine: the right batch to the right machine,
/// a second press queued (never Stop), Stop reaching the machine, and a model
/// switch adopting its recipe.
@MainActor
struct GenerateControllerTests {
    @Test func accessibilityComposerYieldsToActiveCanvas() {
        #expect(GenerateView.composerHeightFraction(run: .idle, accessibility: true) == 0.9)
        #expect(GenerateView.composerHeightFraction(run: .submitting, accessibility: true) == 0.55)
        #expect(GenerateView.composerHeightFraction(run: .failed("Offline"), accessibility: true) == 0.55)
        #expect(GenerateView.composerHeightFraction(run: .idle, accessibility: false) == 0.55)
    }

    private func model(_ name: String = "flux-dev:q4") throws -> Model {
        let doc = try JSONSerialization.jsonObject(with: Data(contentsOf: profilesURL())) as! [String: Any]
        let rows = doc["profiles"] as! [[String: Any]]
        let row = rows.first { ($0["models"] as! [[String: Any]]).contains { $0["model"] as? String == name } }!
        let profile = try JSONSerialization.data(withJSONObject: row["profile"]!)
        let json = #"{"name":"\#(name)","family":"flux","description":"FLUX.1 Dev — best quality","downloaded":true,"generation_profile":"#
            + String(data: profile, encoding: .utf8)! + "}"
        return try MoldJSON.decoder.decode(Model.self, from: Data(json.utf8))
    }

    private func profilesURL() -> URL {
        var url = URL(fileURLWithPath: #filePath)
        while url.pathComponents.count > 1, !FileManager.default.fileExists(atPath: url.appending(path: "docs/generated").path) {
            url.deleteLastPathComponent()
        }
        return url.appending(path: "docs/generated/generation-profiles-v1.json")
    }

    nonisolated private static func batch(_ id: String, state: String) throws -> BatchStatus {
        let json = #"{"id":"\#(id)","client_batch_id":"c-\#(id)","children":[{"index":1,"job_id":"j-\#(id)","state":"\#(state)""#
            + (state == "complete" ? #","result":{"filename":"a.png"}"# : "") + "}]}"
        return try MoldJSON.decoder.decode(BatchStatus.self, from: Data(json.utf8))
    }

    private func setUp() async throws -> (GenerateController, FakeBackend) {
        let fake = FakeBackend()
        fake.stub("status()", returning: try MoldJSON.decoder.decode(ServerStatus.self, from: Data(
            #"{"version":"0.32.0","busy":false,"uptime_secs":1}"#.utf8)))
        fake.stub("capabilities()", returning: try MoldJSON.decoder.decode(Capabilities.self, from: Data("{}".utf8)))
        fake.stub("models()", returning: [try model()])
        fake.stub("jobPreview(jobId:)") { _ in JobProgress?.none }
        let file = HostListFile(url: FileManager.default.temporaryDirectory.appending(path: "h-\(UUID()).json"))
        let hosts = HostStore(list: file, credentials: HostStoreTests.MemoryCredentials(), makeBackend: { _ in fake })
        try hosts.add(name: "workstation", address: "10.0.0.4", apiKey: nil, makeDefault: true)
        await hosts.refreshAll()
        let ledger = PendingLedger(url: FileManager.default.temporaryDirectory.appending(path: "p-\(UUID()).json"))
        let drafts = DraftStore(directory: FileManager.default.temporaryDirectory.appending(path: "d-\(UUID())"))
        let generate = GenerateController(hosts: hosts, ledger: ledger, drafts: drafts)
        generate.settleChoice()
        generate.draft.prompt = "a lighthouse at dusk"
        return (generate, fake)
    }

    @Test func offlineInventoryIsNotAnInstructionToInstall() async throws {
        let (generate, _) = try await setUp()
        generate.hosts.setReachability(.down("Offline"), for: generate.hosts.hosts[0].id)
        #expect(generate.blocker == "No machine is answering. Check Machines to reconnect.")
    }

    @Test func choosingAModelAdoptsItsRecipeAndAutoFindsTheMachine() async throws {
        let (generate, _) = try await setUp()
        #expect(generate.modelName == "flux-dev:q4")
        #expect(generate.recipe != nil)
        #expect(generate.draft.steps == generate.recipe?.defaults.steps)
        #expect(generate.target?.name == "workstation")
        #expect(generate.blocker == nil)
    }

    @Test func aPressAdmitsOneRequestPerCopyToThatMachine() async throws {
        let (generate, fake) = try await setUp()
        generate.draft.batchSize = 1
        fake.stub("submit(_:)") { _ in try Self.batch("b1", state: "running") }
        fake.stubStream("batchEvents(id:)") { _ -> AsyncThrowingStream<BatchStatus, Error> in
            AsyncThrowingStream { $0.yield(try! Self.batch("b1", state: "complete")); $0.finish() }
        }
        generate.generate()
        try await waitUntil { if case .finished = generate.run { true } else { false } }
        let admission = try #require(fake.calls.first { $0.route == "submit(_:)" }?.arguments.first as? BatchAdmission)
        #expect(admission.requests.count == 1)
        #expect(admission.requests.first?.prompt == "a lighthouse at dusk")
        #expect(generate.ledger.batches.isEmpty, "a settled batch leaves the ledger")
    }

    @Test func aSecondPressQueuesRatherThanReplacing() async throws {
        let (generate, fake) = try await setUp()
        fake.stub("submit(_:)") { _ in try Self.batch("b\(fake.count("submit(_:)"))", state: "running") }
        fake.stubStream("batchEvents(id:)") { _ -> AsyncThrowingStream<BatchStatus, Error> in AsyncThrowingStream { _ in } }
        generate.generate()
        try await waitUntil { generate.activeBatch != nil }
        generate.generate()
        try await waitUntil { generate.queued.count == 1 }
        #expect(generate.run.isBusy)
    }

    @Test func stopCancelsOnTheMachine() async throws {
        let (generate, fake) = try await setUp()
        fake.stub("submit(_:)") { _ in try Self.batch("b1", state: "running") }
        fake.stubStream("batchEvents(id:)") { _ -> AsyncThrowingStream<BatchStatus, Error> in AsyncThrowingStream { _ in } }
        fake.stub("cancelBatch(id:)") { _ in () }
        generate.generate()
        try await waitUntil { generate.activeBatch != nil }
        generate.stop()
        try await waitUntil { fake.count("cancelBatch(id:)") == 1 }
        #expect(generate.run == .idle)
    }

    @Test func nothingSubmitsWhileTheControlsSayNo() async throws {
        let (generate, fake) = try await setUp()
        generate.draft.prompt = ""
        if generate.recipe?.capabilities.promptRequirement == .required {
            #expect(generate.blocker == "Describe what you want first.", "the shared refusal, word for word as on the Mac")
            generate.generate()
            #expect(fake.count("submit(_:)") == 0)
        }
    }

    @Test func theSettledBatchIsTheOneThatFinishedNotTheNextInLine() async throws {
        let (generate, fake) = try await setUp()
        fake.stub("submit(_:)") { _ in try Self.batch("b\(fake.count("submit(_:)"))", state: "running") }
        fake.stub("batchStatus(id:)") { args in try Self.batch(args.first as! String, state: "running") }
        fake.stubStream("batchEvents(id:)") { _ -> AsyncThrowingStream<BatchStatus, Error> in AsyncThrowingStream { _ in } }
        var settled: [String] = []
        generate.settled = { batch, _ in settled.append(batch.id) }
        generate.generate()
        try await waitUntil { generate.activeBatch != nil }
        generate.generate()
        try await waitUntil { generate.queued.count == 1 }
        let first = try #require(generate.activeBatch)
        generate.settle(try Self.batch(first.id, state: "complete"), active: first)
        #expect(settled == [first.id])
        #expect(generate.activeBatch?.id != first.id, "the next batch took the canvas after the callback")
    }

    @Test func aRenderThatFinishedWhileAwaySettlesOnReturnWithoutResubmitting() async throws {
        let (generate, fake) = try await setUp()
        fake.stub("submit(_:)") { _ in try Self.batch("b1", state: "running") }
        fake.stubStream("batchEvents(id:)") { _ -> AsyncThrowingStream<BatchStatus, Error> in AsyncThrowingStream { _ in } }
        generate.generate()
        try await waitUntil { generate.activeBatch != nil }
        generate.suspendFollowing()
        #expect(generate.activeBatch != nil, "going to the background keeps the batch")
        fake.stub("batchStatus(id:)") { _ in try Self.batch("b1", state: "complete") }
        generate.resumeFollowing()
        try await waitUntil { if case .finished = generate.run { true } else { false } }
        #expect(fake.count("submit(_:)") == 1)
        #expect(generate.ledger.batches.isEmpty)
    }

    @Test(arguments: ["ltx-2.5-22b-distilled:bf16", "hunyuan3d-2.1:fp16"])
    func coldLaunchRestoresTheDraftKindAfterModelsArrive(_ name: String) async throws {
        let (original, _) = try await setUp()
        let savedModel = try model(name)
        let host = try #require(original.hosts.hosts.first)
        let expectedRecipe = try #require(savedModel.generationProfile?.recipes.first)
        original.hosts.setModels([savedModel], for: host.id)
        original.setKind(expectedRecipe.makes)
        original.choose(savedModel)
        original.draft.prompt = expectedRecipe.capabilities.promptRequirement == .ignored ? "" : "Keep this draft without generating"
        original.draft.steps = expectedRecipe.steps.clamp(expectedRecipe.defaults.steps + 1)
        let authored = original.draft
        original.saveDraft()

        // The composition root restores the draft before any machine answers.
        original.hosts.setModels(nil, for: host.id)
        original.hosts.setReachability(nil, for: host.id)
        let restored = GenerateController(hosts: original.hosts, ledger: original.ledger, drafts: original.drafts)
        restored.settleChoice()
        original.hosts.setModels([savedModel], for: host.id)
        original.hosts.setReachability(.up(try MoldJSON.decoder.decode(ServerStatus.self, from: Data(
            #"{"version":"0.32.0","busy":false,"uptime_secs":1}"#.utf8))), for: host.id)
        restored.settleChoice()

        #expect(restored.kind == expectedRecipe.makes)
        #expect(restored.modelName == name)
        #expect(restored.recipe?.makes == expectedRecipe.makes)
        #expect(restored.draft.prompt == authored.prompt)
        #expect(restored.draft.steps == authored.steps, "restoration must not reset authored options")
        #expect(restored.draft.offersAudioControl == authored.offersAudioControl)
        #expect(restored.draft.enableAudio == authored.enableAudio)
    }

    @Test func restoredClipWaitsForItsMachineWithoutAdoptingAnotherMachinesDefaults() async throws {
        let (original, _) = try await setUp()
        let clip = try model("ltx-2.5-22b-distilled:bf16")
        let slow = MoldHost(name: "Slow clip machine", baseURL: URL(string: "http://10.0.0.5:7680")!)
        original.hosts.setHosts(original.hosts.hosts + [slow])
        let up = try #require(original.hosts.reachability[original.hosts.hosts[0].id])
        original.hosts.setReachability(up, for: slow.id)
        original.hosts.setModels([clip], for: slow.id)
        original.setKind(.clip)
        original.choose(clip)
        original.draft.prompt = "A saved clip"
        original.saveDraft()
        original.hosts.setReachability(.checking, for: slow.id)
        original.hosts.setModels(nil, for: slow.id)

        let restored = GenerateController(hosts: original.hosts, ledger: original.ledger, drafts: original.drafts)
        restored.settleChoice()
        #expect(restored.modelName == clip.name, "the fast picture machine must not replace the saved clip")
        original.hosts.setModels([clip], for: slow.id)
        original.hosts.setReachability(up, for: slow.id)
        restored.settleChoice()
        #expect(restored.kind == .clip)
        #expect(restored.modelName == clip.name)
        #expect(restored.draft.prompt == "A saved clip")
    }

    @Test func explicitStillChoiceCancelsPendingClipRestoration() async throws {
        let (generate, _) = try await setUp()
        generate.restoringChoice = true
        generate.modelName = "ltx-2.5-22b-distilled:bf16"
        generate.setKind(.picture)
        #expect(!generate.restoringChoice)
        #expect(generate.modelName == "flux-dev:q4")
        #expect(generate.recipe?.makes == .picture)
    }

    private func waitUntil(_ condition: () -> Bool) async throws {
        for _ in 0..<200 where !condition() { try await Task.sleep(for: .milliseconds(20)) }
        #expect(condition())
    }
}
