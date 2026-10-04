import Foundation
import Testing
@testable import MoldClient

@Test func typedReferenceWireAndPlacementPreserveOrder() throws {
    var request = GenerateRequest(prompt: "private", model: "ref", width: 768, height: 768, steps: 5, guidance: 1)
    request.references = [GenerationReference(kind: "image", media: .init(authority: "inline", data: "YQ=="), mimeType: "image/png", provenance: .init(name: "private.png", sha256: String(repeating: "a", count: 64)), width: 64, height: 64)]
    let json = try JSONSerialization.jsonObject(with: MoldJSON.encoder.encode(request)) as! [String: Any]
    #expect((json["references"] as? [[String: Any]])?.first?["kind"] as? String == "image")
    let redacted = request.redactedForPlacement().references!.first!
    #expect(redacted.media.authority == "descriptor")
    #expect(redacted.media.data == nil)
    #expect(redacted.provenance?.name == nil)
    #expect(ExpandTask.forRequest(family: "minimax-h3", request: request) == .referenceToAudioVideo)
}

@Test func importedAudioHasExactDecodedSamples() async throws {
    let samples = 96_000
    let rate = 32_000
    var bytes = Data()
    func text(_ value: String) { bytes.append(contentsOf: value.utf8) }
    func u16(_ value: UInt16) { var little = value.littleEndian; withUnsafeBytes(of: &little) { bytes.append(contentsOf: $0) } }
    func u32(_ value: UInt32) { var little = value.littleEndian; withUnsafeBytes(of: &little) { bytes.append(contentsOf: $0) } }
    text("RIFF"); u32(UInt32(36 + samples * 2)); text("WAVEfmt "); u32(16)
    u16(1); u16(1); u32(UInt32(rate)); u32(UInt32(rate * 2)); u16(2); u16(16)
    text("data"); u32(UInt32(samples * 2)); bytes.append(Data(count: samples * 2))
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
    try bytes.write(to: url)
    defer { try? FileManager.default.removeItem(at: url) }
    let ref = try await GenerationReferenceImporter.load(url: url)
    #expect(ref.kind == "audio")
    #expect(ref.sampleCount == samples)
    #expect(ref.sampleRate == rate)
    #expect(ref.channels == 1)
    #expect(ref.durationMs == 3000)
}

@Test func typedReferenceParkingAndAuthoringAreDistinctFromSubmission() throws {
    let json = Data("""
        {"generation_references":{"mode":"adjustable","required":true,"kinds":["image","video","audio"],"max_count":12,"max_images":9,"max_videos":3,"max_audios":3,"min_duration_ms":2000,"max_duration_ms":15000,"max_video_duration_ms":15000,"max_audio_duration_ms":15000,"max_inline_bytes":33554432,"requires_visual":true}}
        """.utf8)
    let capable = try MoldJSON.decoder.decode(RecipeCapabilities.self, from: json)
    let plain = try MoldJSON.decoder.decode(RecipeCapabilities.self, from: Data("{}".utf8))
    var media = DraftMedia()
    var audio = GenerationReference(kind: "audio", media: .init(authority: "inline", data: "YQ=="), mimeType: "audio/wav")
    audio.durationMs = 3000; audio.sampleCount = 96000; audio.sampleRate = 32000; audio.channels = 1
    media.appendGenerationReference(audio)
    #expect(media.generationReferenceError(capabilities: capable, allowIncomplete: true) == nil)
    #expect(media.generationReferenceError(capabilities: capable) != nil)
    media.reconcile(for: plain)
    #expect(media.generationReferences.isEmpty)
    #expect(media.parked.generationReferences == [audio])
    media.reconcile(for: capable)
    #expect(media.generationReferences == [audio])
}

@Test func importedMp4UsesVideoProbeEvenWhenImageIOSuppliesAThumbnail() async throws {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("Fixtures/reference-video.mp4")
    let ref = try await GenerationReferenceImporter.load(url: url)
    #expect(ref.kind == "video")
    #expect(ref.mimeType == "video/mp4")
    #expect(ref.frameCount == 72)
    #expect(ref.durationMs == 3000)
    #expect(ref.fps == 24)
    #expect(ref.hasAudio == true)
    #expect(ref.audioSampleCount == 96_000)
    #expect(ref.audioSampleRate == 32_000)
    #expect(ref.audioDurationMs == 3000)
    #expect(ref.audioChannels == 1)
}

@Test func importedHevcMp4RefusesUnsupportedServerCodec() async throws {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("Fixtures/reference-hevc.mp4")
    await #expect(throws: ReferenceImportError.self) {
        try await GenerationReferenceImporter.load(url: url)
    }
}
