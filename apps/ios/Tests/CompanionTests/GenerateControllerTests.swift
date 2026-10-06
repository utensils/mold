import CoreGraphics
import Foundation
import MoldClient
import MoldClientTesting
import SwiftUI
import Testing
import UIKit

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
        let generate = GenerateController(hosts: hosts, ledger: ledger, drafts: drafts, initialMachine: .auto)
        generate.settleChoice()
        generate.draft.prompt = "a lighthouse at dusk"
        return (generate, fake)
    }

    @Test func pendingLibraryReuseIsConsumedOnFirstMountAndChangesWithoutReplay() async throws {
        let (generate, fake) = try await setUp()
        fake.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .unavailableLegacy, members: []))
        func entry(_ name: String) throws -> LibraryEntry {
            let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(
                #"{"filename":"\#(name).png","metadata":{"model":"flux-dev:q4","prompt":"\#(name)"},"timestamp":1790000000,"format":"png"}"#.utf8))
            return LibraryEntry(host: generate.hosts.hosts[0], print: print)
        }
        let router = AppRouter()
        router.reuse(try entry("first"))
        let scene = try #require(UIApplication.shared.connectedScenes.first as? UIWindowScene)
        let previous = scene.keyWindow
        let window = UIWindow(windowScene: scene)
        let library = LibraryStore(hosts: generate.hosts)
        var appearances = 0
        func mount() {
            window.rootViewController = UIHostingController(rootView: NavigationStack {
                GenerateView().onAppear { appearances += 1 }
            }.environment(generate).environment(generate.hosts).environment(router).environment(library))
            window.makeKeyAndVisible()
        }
        defer {
            window.isHidden = true
            window.rootViewController = nil
            previous?.makeKey()
        }
        mount()
        try await waitUntil { router.pendingReuse == nil }
        #expect(generate.draft.prompt == "first")
        try await waitUntil { !generate.retainedReuse.probing }
        #expect(fake.count("retainedSourceMedia(for:)") == 1)

        router.reuse(try entry("second"))
        try await waitUntil { router.pendingReuse == nil }
        #expect(generate.draft.prompt == "second")
        try await waitUntil { !generate.retainedReuse.probing }
        #expect(fake.count("retainedSourceMedia(for:)") == 2)

        generate.draft.prompt = "edited after reuse"
        window.rootViewController = UIViewController()
        mount()
        // A rendered remount must keep the edit, rather than replay a consumed handoff.
        try await waitUntil { appearances == 2 }
        #expect(router.pendingReuse == nil)
        #expect(generate.draft.prompt == "edited after reuse")
        #expect(fake.count("retainedSourceMedia(for:)") == 2)
    }

    @Test func reuseAlwaysProbesAndShowsRetainedSourceInWell() async throws {
        let (generate, fake) = try await setUp()
        let member = RetainedSourceMedia.Member(memberId: "m", role: "source_image", displayName: "original.png", sizeBytes: 3)
        fake.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: [member]))
        fake.stub("retainedSourceMediaBytes(for:member:)", returning: Data([1, 2, 3]))
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(
            #"{"filename":"a.png","metadata":{"model":"flux-dev:q4","prompt":"reuse"},"timestamp":1790000000,"format":"png"}"#.utf8))
        generate.reuse(LibraryEntry(host: generate.hosts.hosts[0], print: print))
        #expect(generate.retainedReuse.probing)
        try await waitUntil { !generate.retainedReuse.probing }
        #expect(fake.count("retainedSourceMedia(for:)") == 1)
        #expect(generate.draft.media.sourceImage == "AQID")
        #expect(generate.draft.media.sourceImageOriginal == "AQID")
        #expect(generate.draft.media.sourceImageName == "original.png")
    }

    @Test func conflictingChainAndOrdinarySourceNeverChooseOneSilently() async throws {
        let (generate, fake) = try await setUp()
        let members = ["source_image", "stage_source:0"].enumerated().map { index, role in
            RetainedSourceMedia.Member(memberId: "m\(index)", role: role, displayName: role, sizeBytes: 3)
        }
        fake.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: members))
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(
            #"{"filename":"chain.mp4","metadata":{"model":"flux-dev:q4","prompt":"reuse"},"timestamp":1790000000,"format":"mp4"}"#.utf8))
        generate.reuse(LibraryEntry(host: generate.hosts.hosts[0], print: print))
        try await waitUntil { !generate.retainedReuse.probing }
        #expect(generate.draft.media.sourceImage == nil)
        #expect(fake.count("retainedSourceMediaBytes(for:member:)") == 0)
        #expect(generate.blocker?.contains("multiple source pictures") == true)
        generate.draft.media.sourceImage = "chosen"
        #expect(generate.retainedReuse.sourcePictureRefusal(in: generate.draft) == nil)
    }

    @Test func retainedSourcesSurviveRepeatedSubmissionsAndPromptEditsUntilRemoved() async throws {
        let (generate, fake) = try await setUp()
        let source = RetainedSourceMedia.Member(memberId: "source", role: "source_image", displayName: "Original.png", sizeBytes: 3)
        let mask = RetainedSourceMedia.Member(memberId: "audio", role: "audio_file", displayName: "Audio.wav", sizeBytes: 3)
        fake.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: [source, mask]))
        fake.stub("retainedSourceMediaBytes(for:member:)", returning: Data([1, 2, 3]))
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(
            #"{"filename":"a.png","metadata":{"model":"flux-dev:q4","prompt":"reuse"},"timestamp":1790000000,"format":"png"}"#.utf8))
        generate.reuse(LibraryEntry(host: generate.hosts.hosts[0], print: print))
        try await waitUntil { !generate.retainedReuse.probing }
        #expect(generate.retainedReuse.snapshot()?.members.map(\.role) == ["audio_file"])
        generate.draft.prompt = "another composition"
        generate.draft.seed = 123
        #expect(generate.retainedReuse.snapshot()?.members.map(\.role) == ["audio_file"])
        #expect(generate.retainedReuse.snapshot()?.members.map(\.role) == ["audio_file"])
        generate.draft.media.sourceImage = nil
        #expect(generate.retainedReuse.snapshot()?.members.contains { $0.role == "source_image" } == false)
        generate.retainedReuse.clear()
        #expect(generate.retainedReuse.snapshot() == nil)
        #expect(generate.retainedReuse.notice == nil)
    }

    @Test func retainedMaskFitsWithItsSourceForChangedAspectAndPadRepaint() async throws {
        let (generate, fake) = try await setUp()
        let context = try #require(CGContext(data: nil, width: 128, height: 64, bitsPerComponent: 8,
            bytesPerRow: 0, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue))
        context.setFillColor(gray: 0, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: 128, height: 64))
        context.setFillColor(gray: 1, alpha: 1)
        context.fill(CGRect(x: 64, y: 0, width: 8, height: 64))
        let image = try #require(context.makeImage())
        let png = try #require(SourceFitRender.encodePNG(image))
        let members = ["source_image", "mask_image"].map {
            RetainedSourceMedia.Member(memberId: $0, role: $0, displayName: $0 + ".png", sizeBytes: png.count)
        }
        fake.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: members))
        fake.stub("retainedSourceMediaBytes(for:member:)", returning: png)
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(
            #"{"filename":"a.png","metadata":{"model":"flux-dev:q4","prompt":"reuse"},"timestamp":1790000000,"format":"png"}"#.utf8))
        generate.reuse(LibraryEntry(host: generate.hosts.hosts[0], print: print))
        try await waitUntil { !generate.retainedReuse.probing }
        #expect(generate.draft.media.maskImage == png.base64EncodedString())
        #expect(generate.retainedReuse.snapshot() == nil, "Restored source and mask are ordinary draft media, never hidden hydration")
        for policy in [SourceFit.default, .padRepaint] {
            var draft = generate.draft
            draft.width = 64; draft.height = 64
            draft.media.sourceFit = policy
            let prepared = try await draft.fittingSource(recipe: generate.recipe)
            let transform = SourceFitTransform.resolve(source: (128, 64), target: (64, 64), policy: policy)
            let expected = try #require(await SourceFitRender.mask(existing: png, transform: transform, sourceSpace: true))
            #expect(prepared.media.maskImage == expected.base64EncodedString())
            #expect(prepared.media.maskImage != png.base64EncodedString())
            let encodedSource = try #require(prepared.media.sourceImage)
            let fitted = try #require(Data(base64Encoded: encodedSource))
            #expect(PictureImport.pixelSize(of: fitted)?.width == 64)
            #expect(PictureImport.pixelSize(of: fitted)?.height == 64)
        }
    }

    @Test func retainedPairHonorsExplicitSourceAndMaskOverrides() async throws {
        for hasSource in [false, true] {
            let (generate, fake) = try await setUp()
            generate.draft.media.maskImage = "user-mask"
            if hasSource { generate.draft.media.sourceImage = "user-source" }
            let members = ["source_image", "mask_image"].map {
                RetainedSourceMedia.Member(memberId: $0, role: $0, displayName: $0 + ".png", sizeBytes: 3)
            }
            fake.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: members))
            fake.stub("retainedSourceMediaBytes(for:member:)", returning: Data([1, 2, 3]))
            let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(
                #"{"filename":"a.png","metadata":{"model":"flux-dev:q4","prompt":"reuse"},"timestamp":1790000000,"format":"png"}"#.utf8))
            generate.reuse(LibraryEntry(host: generate.hosts.hosts[0], print: print))
            try await waitUntil { !generate.retainedReuse.probing }
            #expect(generate.draft.media.maskImage == "user-mask")
            #expect(generate.draft.media.sourceImage == (hasSource ? "user-source" : "AQID"))
            #expect(generate.retainedReuse.snapshot() == nil)
            #expect(fake.count("retainedSourceMediaBytes(for:member:)") == (hasSource ? 0 : 1))
        }
    }

    @Test func explicitModelChoiceInvalidatesPendingRetainedProbe() async throws {
        let (generate, _) = try await setUp()
        let fence = generate.retainedReuse.begin(generate.draft)
        generate.choose(try model())
        #expect(!generate.retainedReuse.isCurrent(fence, draft: generate.draft))
        #expect(!generate.retainedReuse.probing)
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

    @Test func resetOptionsRestoresRandomSeedAndCenteredCrop() async throws {
        let (generate, _) = try await setUp()
        generate.draft.seed = 42
        generate.draft.locksSeed = true
        generate.draft.media.sourceFit = .padFit
        generate.resetOptions()
        #expect(!generate.draft.locksSeed)
        #expect(generate.draft.seed == nil)
        #expect(generate.draft.media.sourceFit == .default)
    }

    @Test func sourcePixelsAreFittedBeforeBatchAdmission() async throws {
        let (generate, fake) = try await setUp()
        let context = try #require(CGContext(data: nil, width: 128, height: 64,
            bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        let image = try #require(context.makeImage())
        let data = try #require(SourceFitRender.encodePNG(image))
        generate.draft.width = 64
        generate.draft.height = 64
        generate.draft.canvasIntent = .manual
        generate.draft.media.sourceImage = data.base64EncodedString()
        generate.draft.media.sourceImageOriginal = data.base64EncodedString()
        fake.stub("submit(_:)") { _ in try Self.batch("b1", state: "running") }
        fake.stubStream("batchEvents(id:)") { _ -> AsyncThrowingStream<BatchStatus, Error> in AsyncThrowingStream { _ in } }
        generate.generate()
        try await waitUntil { fake.count("submit(_:)") == 1 }
        let admission = try #require(fake.calls.first { $0.route == "submit(_:)" }?.arguments.first as? BatchAdmission)
        let sent = try #require(admission.requests.first?.sourceImage.flatMap { Data(base64Encoded: $0) })
        let pixels = try #require(PictureImport.pixelSize(of: sent))
        #expect(pixels.width == 64 && pixels.height == 64)
        #expect(admission.requests.first?.sourceFit == .default)
        #expect(generate.draft.media.sourceImageOriginal == data.base64EncodedString())
        generate.stop()
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
        let restored = GenerateController(hosts: original.hosts, ledger: original.ledger, drafts: original.drafts, initialMachine: .auto)
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

        let restored = GenerateController(hosts: original.hosts, ledger: original.ledger, drafts: original.drafts, initialMachine: .auto)
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
