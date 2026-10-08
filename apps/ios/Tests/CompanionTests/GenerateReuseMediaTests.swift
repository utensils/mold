import Foundation
import MoldClient
import MoldClientTesting
import Testing
@testable import MoldCompanion

@MainActor extension GenerateControllerTests {
    @Test(arguments: ["minimax-h3-fl2va:comfy-pruned-int8", "wan22-i2v-a14b:fp8", "ltx-2.5-22b-distilled:bf16"])
    func reuseMaterializesOriginalFramesInTheirVisibleWells(name: String) async throws {
        let (generate, fake) = try await setUp()
        fake.stub("models()", returning: [try model(name)])
        await generate.hosts.refreshAll()
        let h3 = name.hasPrefix("minimax")
        let frames = h3 ? 141 : 97
        let original = h3 ? [KeyframeCondition(frame: frames - 1, image: "last", name: "last.png")]
            : [KeyframeCondition(frame: 0, image: "first", name: "first.png"), .init(frame: frames - 1, image: "last", name: "last.png")]
        var members = original.enumerated().map { RetainedSourceMedia.Member(memberId: "frame-\($0.offset)", role: "keyframes", displayName: "frame", sizeBytes: 80) }
        if h3 { members.insert(.init(memberId: "first", role: "source_image", displayName: "first.png", sizeBytes: 3), at: 0) }
        fake.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: members))
        fake.stub("retainedSourceMediaBytes(for:member:)") { args in
            let member = args[1] as! String
            if member == "first" { return Data([1, 2, 3]) }
            return try MoldJSON.encoder.encode(original[Int(member.split(separator: "-").last!)!])
        }
        generate.reuse(try reuseEntry(generate, name: name, extra: "\"frames\":\(frames)"))
        try await waitUntil { !generate.retainedReuse.probing }
        #expect(generate.draft.media.keyframes.count == original.count)
        #expect(generate.draft.media.keyframes.map(\.image) == original.map(\.image))
        if h3 { #expect(generate.draft.media.sourceImage == "AQID") }
        #expect(generate.retainedReuse.snapshot()?.members.contains { $0.role == "keyframes" } != true)
        generate.draft.media.keyframes = []
        generate.retainedReuse.retry(controller: generate)
        try await waitUntil { !generate.retainedReuse.probing }
        #expect(generate.draft.media.keyframes.isEmpty, "Removing restored frames cannot revive hidden conditioning")
    }

    @Test func lateFrameBytesCannotReviveAnAttachmentAddedThenRemoved() async throws {
        let (generate, fake) = try await setUp()
        let gate = ReuseGate()
        let member = RetainedSourceMedia.Member(memberId: "last", role: "keyframes", displayName: "last.png", sizeBytes: 80)
        fake.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: [member]))
        fake.stub("retainedSourceMediaBytes(for:member:)") { _ in
            await gate.wait()
            return try MoldJSON.encoder.encode(KeyframeCondition(frame: 96, image: "old"))
        }
        generate.reuse(try reuseEntry(generate, name: "flux-dev:q4"))
        try await waitUntil { fake.count("retainedSourceMediaBytes(for:member:)") == 1 }
        generate.draft.media.keyframes = [.init(frame: 96, image: "replacement")]
        generate.draft.media.keyframes = []
        await gate.release()
        try await waitUntil { !generate.retainedReuse.probing }
        #expect(generate.draft.media.keyframes.isEmpty)
        #expect(generate.retainedReuse.snapshot()?.members.contains { $0.role == "keyframes" } != true)
    }

    @Test(arguments: ["flux-dev:q4", "qwen-image-2.1:q8", "qwen-image-edit-2511:q4", "flux2-klein:bf16", "sdxl-base:fp16", "ltx-2.5-22b-distilled:bf16", "minimax-h3-ref2va:official-bf16", "hunyuan3d-2mv:fp16"])
    func reuseClearsEveryPreviousAttachmentIncludingParkedMedia(name: String) async throws {
        let (generate, fake) = try await setUp()
        fake.stub("models()", returning: [try model(name)])
        await generate.hosts.refreshAll()
        generate.draft.media = staleMedia()
        fake.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .unavailableLegacy, members: []))
        generate.reuse(try reuseEntry(generate, name: name))
        try await waitUntil { !generate.retainedReuse.probing }
        let media = generate.draft.media
        #expect(media.sourceImage == nil && media.sourceImageOriginal == nil && media.maskImage == nil)
        #expect(media.editImages.isEmpty && media.generationReferences.isEmpty)
        #expect(media.identity?.photos.isEmpty != false && media.control?.image == nil)
        #expect(media.keyframes.isEmpty && media.boundaryKeyframes.isEmpty)
        #expect(media.audioFile == nil && media.sourceVideo == nil && media.extendVideo == nil)
        #expect(media.parked == ParkedConditioning())
    }

    @Test(arguments: ["image", "video", "audio", "named_image"])
    func reuseReplacesTypedReferencesWithSelectedPrintDescriptors(kind: String) async throws {
        let (generate, fake) = try await setUp()
        generate.draft.media = staleMedia()
        let role = kind == "named_image" ? #", "image_role":"front""# : ""
        let metadata = #""references":[{"kind":"\#(kind == "named_image" ? "image" : kind)","mime_type":"image/png","name":"selected.png","sha256":"selected","width":64,"height":64\#(role)}]"#
        let member = RetainedSourceMedia.Member(memberId: "selected", role: "references", displayName: "selected.png", sizeBytes: 3)
        fake.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: [member]))
        generate.reuse(try reuseEntry(generate, name: "missing-model", extra: metadata))
        try await waitUntil { !generate.retainedReuse.probing }
        let ref = try #require(generate.draft.media.generationReferences.first)
        #expect(ref.kind == kind)
        #expect(ref.name == "selected.png" && ref.media.authority == "descriptor")
        #expect(ref.media.data == nil && ref.media.handle == nil)
        #expect(generate.draft.media.parked.generationReferences.isEmpty)
        #expect(generate.retainedReuse.canHydrateReferences(generate.draft.media.generationReferences))
    }

    @Test(arguments: [RetainedSourceMedia.Availability.unavailableLegacy, .unavailableMissingOrCorrupt, .available])
    func missingRetainedReferencesNeverReusePreviousDraftBytes(availability: RetainedSourceMedia.Availability) async throws {
        let (generate, fake) = try await setUp()
        generate.draft.media = staleMedia()
        fake.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: availability, members: []))
        generate.reuse(try reuseEntry(generate, name: "missing-model", extra: #""references":[{"kind":"image","mime_type":"image/png","name":"missing.png"}]"#))
        try await waitUntil { !generate.retainedReuse.probing }
        #expect(generate.blocker == "Reattach this print’s reference media before generating.")
        #expect(generate.draft.media.sourceImage == nil && generate.draft.media.editImages.isEmpty)
        #expect(generate.draft.media.generationReferences.first?.media.authority == "descriptor")
    }

    @Test func offlineArchiveBlocksOptionalSourceUntilExplicitlyDiscarded() async throws {
        let (generate, fake) = try await setUp()
        fake.stub("retainedSourceMedia(for:)") { _ in throw URLError(.notConnectedToInternet) }
        generate.reuse(try reuseEntry(generate, name: "flux-dev:q4", extra: #""source_image_sha256":"original""#))
        try await waitUntil { !generate.retainedReuse.probing }
        #expect(generate.retainedReuse.notice != nil)
        #expect(generate.blocker == "Restore or reattach this print’s source media before generating.")
        generate.retainedReuse.clear()
        #expect(generate.blocker == nil)
    }

    @Test func lateSourceNeverRevivesAnAttachmentAddedThenRemoved() async throws {
        let (generate, fake) = try await setUp()
        let gate = ReuseGate()
        let member = RetainedSourceMedia.Member(memberId: "source", role: "source_image", displayName: "original.png", sizeBytes: 3)
        fake.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: [member]))
        fake.stub("retainedSourceMediaBytes(for:member:)") { _ in await gate.wait(); return Data([1, 2, 3]) }
        generate.reuse(try reuseEntry(generate, name: "flux-dev:q4"))
        try await waitUntil { fake.count("retainedSourceMediaBytes(for:member:)") == 1 }
        generate.draft.media.sourceImage = "replacement"
        generate.draft.media.sourceImage = nil
        await gate.release()
        try await waitUntil { !generate.retainedReuse.probing }
        #expect(generate.draft.media.sourceImage == nil)
        #expect(generate.retainedReuse.snapshot()?.members.contains { $0.role == "source_image" } != true)
    }

    @Test func switchingReusedPrintsFencesEarlierSourceBytes() async throws {
        let (generate, fake) = try await setUp()
        let gate = ReuseGate()
        let member = RetainedSourceMedia.Member(memberId: "source", role: "source_image", displayName: "original.png", sizeBytes: 3)
        fake.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: [member]))
        fake.stub("retainedSourceMediaBytes(for:member:)") { _ in await gate.wait(); return Data([1, 2, 3]) }
        generate.reuse(try reuseEntry(generate, name: "flux-dev:q4"))
        try await waitUntil { fake.count("retainedSourceMediaBytes(for:member:)") == 1 }
        fake.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .unavailableLegacy, members: []))
        generate.reuse(try reuseEntry(generate, name: "flux-dev:q4"))
        try await waitUntil { !generate.retainedReuse.probing }
        await gate.release()
        await Task.yield()
        #expect(generate.draft.media.sourceImage == nil)
        #expect(generate.retainedReuse.snapshot() == nil)
    }

    @Test(arguments: [
        "source_image|source_image_sha256|source", "control_image|control_model|controlnet-canny-sd15:fp16", "edit_images|edit_image_sha256s|[\"edit\"]",
        "identity_image|id_image_sha256|face", "identity_images|id_image_sha256|face",
        "identity_image|id_image_sha256s|[\"face\"]", "identity_images|id_image_sha256s|[\"face\"]",
        "audio_file|audio_file_path|audio.wav", "source_video|source_video_path|video.mp4",
        "extend_video|extend_video_path|clip.mp4", "keyframes|keyframes|[{\"frame\":0,\"sha256\":\"frame\"}]"
    ])
    func everyExpectedRetainedRoleRequiresArchiveOrReplacement(scenario: String) async throws {
        let pieces = scenario.split(separator: "|", maxSplits: 2).map(String.init)
        let value = pieces[2].hasPrefix("[") ? pieces[2] : "\"" + pieces[2] + "\""
        let extra = "\"" + pieces[1] + "\":" + value
        let (generate, fake) = try await setUp()
        fake.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .unavailableMissingOrCorrupt, members: []))
        generate.reuse(try reuseEntry(generate, name: "flux-dev:q4", extra: extra))
        try await waitUntil { !generate.retainedReuse.probing }
        #expect(generate.retainedReuse.restorationRefusal(for: nil) != nil)
        let member = RetainedSourceMedia.Member(memberId: "retained", role: pieces[0], displayName: "selected", sizeBytes: 3)
        fake.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: [member]))
        fake.stub("retainedSourceMediaBytes(for:member:)", returning: pieces[0] == "keyframes" ? try! MoldJSON.encoder.encode(KeyframeCondition(frame: 0, image: "AQID", name: "frame.png")) : Data([1, 2, 3]))
        generate.retainedReuse.retry(controller: generate)
        try await waitUntil { !generate.retainedReuse.probing }
        var request = GenerateRequest(prompt: "selected", model: "flux", width: 512, height: 512, steps: 4, guidance: 1)
        request.sourceImage = generate.draft.media.sourceImage
        #expect(generate.retainedReuse.restorationRefusal(for: request) == nil)
    }

    @Test func failedSourceDownloadWithoutMetadataMarkerStillBlocksAndRetries() async throws {
        let (generate, fake) = try await setUp()
        let member = RetainedSourceMedia.Member(memberId: "source", role: "source_image", displayName: "original.png", sizeBytes: 3)
        fake.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: [member]))
        fake.stub("retainedSourceMediaBytes(for:member:)") { _ in throw URLError(.timedOut) }
        generate.reuse(try reuseEntry(generate, name: "flux-dev:q4"))
        try await waitUntil { !generate.retainedReuse.probing }
        #expect(generate.blocker == "Restore or reattach this print’s source media before generating.")
        fake.stub("retainedSourceMediaBytes(for:member:)", returning: Data([1, 2, 3]))
        generate.retainedReuse.retry(controller: generate)
        try await waitUntil { !generate.retainedReuse.probing }
        #expect(generate.draft.media.sourceImage == "AQID")
        #expect(generate.blocker == nil)
        generate.draft.media.sourceImage = nil
        #expect(generate.blocker == nil, "Removing restored ordinary source is explicit")
    }

    @Test func attachingSourceDoesNotDiscardUnrelatedMissingAudio() async throws {
        let (generate, fake) = try await setUp()
        fake.stub("retainedSourceMedia(for:)") { _ in throw URLError(.notConnectedToInternet) }
        generate.reuse(try reuseEntry(generate, name: "flux-dev:q4", extra: #""audio_file_path":"original.wav""#))
        try await waitUntil { !generate.retainedReuse.probing }
        let png = Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=")!
        fake.stub("media(_:trashed:)", returning: png)
        await generate.useAsSource(try reuseEntry(generate, name: "flux-dev:q4"))
        #expect(generate.draft.media.sourceImage != nil)
        #expect(generate.blocker == "Restore or reattach this print’s source media before generating.")
    }

    @Test func retryForMissingRoleDoesNotReviveRemovedSourceOrRebaselineReferences() async throws {
        let (generate, fake) = try await setUp()
        let members = ["source_image", "audio_file"].map {
            RetainedSourceMedia.Member(memberId: $0, role: $0, displayName: $0, sizeBytes: 3)
        }
        fake.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: members))
        fake.stub("retainedSourceMediaBytes(for:member:)", returning: Data([1, 2, 3]))
        generate.reuse(try reuseEntry(generate, name: "flux-dev:q4"))
        try await waitUntil { !generate.retainedReuse.probing }
        generate.draft.media.sourceImage = nil
        generate.retainedReuse.retry(controller: generate)
        try await waitUntil { !generate.retainedReuse.probing }
        #expect(generate.draft.media.sourceImage == nil)

        let reference = RetainedSourceMedia.Member(memberId: "reference", role: "references", displayName: "original", sizeBytes: 3)
        fake.stub("retainedSourceMedia(for:)", returning: RetainedSourceMedia.Inventory(availability: .available, members: [reference]))
        generate.reuse(try reuseEntry(generate, name: "missing-model", extra: #""references":[{"kind":"image","mime_type":"image/png","name":"original.png"}]"#))
        try await waitUntil { !generate.retainedReuse.probing }
        generate.draft.media.generationReferences[0].provenance = .init(name: "different.png")
        generate.retainedReuse.retry(controller: generate)
        try await waitUntil { !generate.retainedReuse.probing }
        #expect(!generate.retainedReuse.canHydrateReferences(generate.draft.media.generationReferences))
    }

    @Test(arguments: [false, true])
    func partialArchiveSurvivesUnavailableCopyAndPrefersCompleteCopy(completeCopy: Bool) async throws {
        let (generate, fake) = try await setUp()
        var entry = try reuseEntry(generate, name: "flux-dev:q4", extra: #""source_image_sha256":"source","id_image_sha256":"face""#)
        let copyPrint = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data(#"{"filename":"copy.png","timestamp":1,"metadata":{}}"#.utf8))
        entry.copies = [LibraryEntry(host: generate.hosts.hosts[0], print: copyPrint)]
        let face = RetainedSourceMedia.Member(memberId: "face", role: "identity_image", displayName: "face", sizeBytes: 3)
        let source = RetainedSourceMedia.Member(memberId: "source", role: "source_image", displayName: "source", sizeBytes: 3)
        fake.stub("retainedSourceMedia(for:)") { args in
            if args[0] as? String == "selected.png" {
                return RetainedSourceMedia.Inventory(availability: .available, members: [face])
            }
            return RetainedSourceMedia.Inventory(availability: completeCopy ? .available : .unavailableLegacy,
                members: completeCopy ? [face, source] : [])
        }
        fake.stub("retainedSourceMediaBytes(for:member:)", returning: Data([1, 2, 3]))
        generate.reuse(entry)
        generate.draft.media.sourceImage = "authored replacement"
        try await waitUntil { !generate.retainedReuse.probing }
        #expect(generate.retainedReuse.snapshot() == nil)
        #expect(generate.draft.media.identity?.photos.first?.encoded == "AQID" || generate.draft.media.parked.identity?.photos.first?.encoded == "AQID")
        #expect(generate.draft.media.sourceImage == "authored replacement")
    }

    private func reuseEntry(_ generate: GenerateController, name: String, extra: String = "") throws -> LibraryEntry {
        let fields = extra.isEmpty ? "" : "," + extra
        let print = try MoldJSON.decoder.decode(GalleryPrint.self, from: Data("{\"filename\":\"selected.png\",\"timestamp\":1,\"metadata\":{\"model\":\"\(name)\",\"prompt\":\"Selected\"\(fields)}}".utf8))
        return LibraryEntry(host: generate.hosts.hosts[0], print: print)
    }

    private func staleMedia() -> DraftMedia {
        var media = DraftMedia()
        media.sourceImage = "old-source"; media.sourceImageOriginal = "old-original"; media.maskImage = "old-mask"
        media.editImages = ["old-edit"]
        media.generationReferences = [.init(kind: "image", media: .init(authority: "upload", handle: "old-lease"), mimeType: "image/png")]
        media.identity = .init(photos: [.init(encoded: "old-face", name: "old")])
        media.control = .init(image: "old-control", model: "old-adapter")
        media.keyframes = [.init(frame: 0, image: "old-frame")]
        media.boundaryKeyframes = ["h3-endpoints": media.keyframes]
        media.audioFile = "old-audio"; media.sourceVideo = "old-video"; media.extendVideo = "old-extend"
        media.parked.sourceImage = "parked-source"; media.parked.maskImage = "parked-mask"
        media.parked.editImages = ["parked-edit"]; media.parked.generationReferences = media.generationReferences
        media.parked.identity = media.identity; media.parked.control = media.control; media.parked.keyframes = media.keyframes
        media.parked.audioFile = "parked-audio"; media.parked.sourceVideo = "parked-video"; media.parked.extendVideo = "parked-extend"
        return media
    }
}

private actor ReuseGate {
    private var continuation: CheckedContinuation<Void, Never>?
    private var released = false
    func wait() async {
        if released { return }
        await withCheckedContinuation { continuation = $0 }
    }
    func release() { released = true; continuation?.resume(); continuation = nil }
}
