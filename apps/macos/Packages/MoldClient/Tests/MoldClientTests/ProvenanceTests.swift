import Foundation
import Testing

@testable import MoldClient

/// What a print records about how it was made, read from prints a real host
/// actually holds.
///
/// `Fixtures/provenance-hal9000.json` is fourteen verbatim rows from
/// hal9000 (`GET /api/gallery`, 0.29.0, 2026-09-17), chosen so that between
/// them they carry every field reuse restores. Its own header names the
/// capture.
///
/// **Fails today**: `OutputMetadata` decodes eighteen fields, so the title,
/// the adapters, the identity knobs, the scheduler, the retained-media
/// markers and the rest are all dropped on the floor.
enum Provenance {
    static func prints() throws -> [GalleryPrint] {
        struct Capture: Decodable { let prints: [GalleryPrint] }
        return try MoldJSON.decoder.decode(
            Capture.self, from: RepoFixtures.fixture("provenance-hal9000.json")
        ).prints
    }

    static func metadata(_ filename: String) throws -> OutputMetadata {
        try #require(try prints().first { $0.filename == filename }).metadata
    }
}

@Test func everyCapturedPrintStillDecodes() throws {
    // The floor matters: an empty array would satisfy every test below.
    #expect(try Provenance.prints().count == 14)
}

@Test func restoresTheFilingAPrintWasMadeUnder() throws {
    let filed = try Provenance.metadata("mold-hunyuan3d-mini-turbo-fp16-1788387693156~nsfw.glb")
    #expect(filed.title?.isEmpty == false)
    #expect(filed.tags?.isEmpty == false)
    #expect(filed.collection?.isEmpty == false)
}

@Test func restoresTheAdapterStackAndItsLegacySingularTwin() throws {
    let withLoras = try Provenance.metadata("mold-ltx-2-19b-distilled-fp8-1786851614872.mp4")
    let loras = try #require(withLoras.loras)
    #expect(loras.count >= 1)
    #expect(loras.allSatisfy { !$0.path.isEmpty })
    // The same print carries the legacy singular pair, which is all an older
    // print has.
    #expect(withLoras.lora != nil)
    #expect(withLoras.loraScale != nil)
}

@Test func restoresTheIdentityKnobsAndTheirDigests() throws {
    let face = try Provenance.metadata("mold-jibmix-flux-fp8-1788383514251~nsfw.png")
    #expect(face.idWeight != nil)
    #expect(face.idStartStep != nil)
    #expect(face.idImageName != nil)
    #expect(face.idImageSha256 != nil)
}

@Test func restoresTheSamplerAndTheCropPolicy() throws {
    #expect(try Provenance.metadata("mold-realistic-vision-v5-fp16-1784141403169.png")
        .scheduler != nil)
    #expect(try Provenance.metadata("mold-ltx-2.5-22b-distilled-q8-1789532686738.mp4")
        .sourceFit != nil)
    #expect(try Provenance.metadata("mold-ltx-2.5-22b-distilled-q8-1789532686738.mp4")
        .strength != nil)
}

@Test func restoresTheResolvedMeshControls() throws {
    let mesh = try #require(
        try Provenance.metadata("mold-hunyuan3d-mini-turbo-fp16-1788503140147.glb").mesh)
    #expect(mesh.octreeResolution != nil)
}

@Test func restoresTheClipControlsAndTheChosenPipeline() throws {
    let clip = try Provenance.metadata("mold-ltx-2.5-22b-distilled-q8-1789532686738.mp4")
    #expect(clip.enableAudio != nil)
    #expect(clip.pipeline != nil)
    #expect(clip.pipelineRequested != nil)
    #expect(clip.durationPredictionRequested == true)
}

@Test func restoresTheUpscalerAndTheRewriteProvenance() throws {
    #expect(try Provenance.metadata("mold-real-esrgan-x4plus-fp16-1788488679298-upscaled.png")
        .upscaleModel != nil)
    let rewritten = try Provenance.metadata("mold-jibmix-flux-fp8-1788207348881~nsfw.png")
    #expect(rewritten.originalPrompt?.isEmpty == false)
    #expect(rewritten.promptTransform != nil)
}

@Test func readsTheMarkersThatSayConditioningBytesShipped() throws {
    #expect(try Provenance.metadata("mold-ltx-2-19b-distilled-fp8-1786851614872.mp4")
        .sourceImageSha256 != nil)
    #expect(try Provenance.metadata("mold-real-esrgan-x4plus-fp16-1788488679298-upscaled.png")
        .editImageDigests.isEmpty == false)
    // The whole point of the odd spelling: this key is `edit_image_sha256s`
    // on the wire and `editImageSha256S` after conversion, because `256` is a
    // word boundary to Foundation's `capitalized`. Spelled the obvious way,
    // it decodes as nil on every print and nothing says so.
    #expect(try Provenance.metadata("mold-jibmix-flux-fp8-1788383514251~nsfw.png")
        .identityDigests.count == 1)
    #expect(try Provenance.metadata("mold-wan22-i2v-a14b-q5-1787718371258~nsfw.mp4")
        .keyframes?.isEmpty == false)
    #expect(try Provenance.metadata(
        "mold-minimax-h3-ref2va-comfy-pruned-int8-1788197310484~nsfw.mp4")
        .references?.isEmpty == false)
    #expect(try Provenance.metadata(
        "mold-ltx-2-19b-distilled-fp8-1787860285996~uat-ltx2-extend.mp4")
        .extendOverlapFrames != nil)
}

@Test func readsAChainPrintsOwnProvenance() throws {
    let authored = try Provenance.metadata(
        "mold-chain-8c352a5a9ac5d5c23549e66d96f07c97f26331a7798de3fce244cdc4da754073-take-1.mp4")
    #expect(authored.chainJobId != nil)
    #expect(try #require(authored.chain).stageCount == 3)
    // An auto-chained one-shot carries the same block and no durable id.
    let ephemeral = try Provenance.metadata(
        "mold-chain-3cb05c3e9d8468260cb56042d6c579e243ae76ca7d6475ccaa1777c6ff49f581-take-1.mp4")
    #expect(ephemeral.chain != nil)
    #expect(ephemeral.chainJobId == nil)
}

@Test func aSequencePrintOffersItsFirstStageAndNeverTheJoinedWall() throws {
    let sequence = try Provenance.metadata(
        "mold-chain-8c352a5a9ac5d5c23549e66d96f07c97f26331a7798de3fce244cdc4da754073-take-1.mp4")
    let joined = try #require(sequence.prompt)
    #expect(joined.contains("\n"))
    #expect(sequence.firstStagePrompt == "a slow cinematic dolly through a neon-lit "
        + "rain-soaked alley at night, volumetric fog")
    #expect(!sequence.firstStagePrompt.contains("\n"))
}

@Test func anOrdinaryPrintsPromptIsItsPrompt() throws {
    let plain = try Provenance.metadata("mold-qwen-image-q8-1789529980561.png")
    #expect(plain.firstStagePrompt == plain.prompt)
}

@Test func anUnknownEnumValueNeverCostsTheWholePrint() throws {
    // A host newer than this build sends a pipeline, a scheduler and a format
    // this binary has never heard of. Every one of them is a plain `String`
    // here for exactly this reason: dropping the print would lose its prompt,
    // its seed and its size over a word.
    let row = """
    {"filename":"mold-x-1.png","timestamp":1,"metadata":{"prompt":"p","model":"m",
     "seed":1,"steps":4,"guidance":1.0,"width":8,"height":8,"version":"99.0",
     "scheduler":"solver-from-the-future","pipeline":"nine-stage",
     "output_format":"jxl","spatial_upscale":"x4","temporal_upscale":"x8"}}
    """
    let print = try MoldJSON.decoder.decode(
        GalleryPrint.self, from: Data(row.utf8))
    #expect(print.metadata.scheduler == "solver-from-the-future")
    #expect(print.metadata.pipeline == "nine-stage")
    #expect(print.metadata.outputFormat == "jxl")
    #expect(print.metadata.spatialUpscale == "x4")
}
