import Foundation
import Testing

@testable import MoldClient

/// `GenerateRequest.encode(to:)` is a hand-written field list over a struct
/// whose `CodingKeys` is declared separately. Adding a stored property
/// compiles, gets a key for free, is happily set by `RenderDraft+Request` --
/// and silently never reaches the wire. `GenerateRequestTests` asserts about
/// twenty named keys and nothing about the SET being complete, which the
/// client review called the highest-value missing test in the package.
///
/// This is that test, and it is deliberately reflective rather than a second
/// hand-written list: a list would have to be remembered too.
@Suite struct GenerateRequestExhaustiveTests {
    /// Every stored property populated. A new one added to the struct with no
    /// value here fails `everyStoredPropertyIsPopulated` below, which is the
    /// nudge to decide whether it belongs on the wire.
    private func populated() -> GenerateRequest {
        var request = GenerateRequest(
            prompt: "a tin robot", model: "flux-dev:q4", width: 1024, height: 1024,
            steps: 20, guidance: 3.5, batchSize: 1, negativePrompt: "blurry",
            seed: 42, saveToGallery: false)
        request.frames = 97
        request.fps = 24
        request.pipeline = "auto"
        request.enableAudio = true
        request.videoOnly = true
        request.sourceImage = "SRC"
        request.sourceImageName = "s.png"
        request.strength = 0.6
        request.editImages = ["REF"]
        request.referenceWeight = 0.8
        request.maskImage = "MASK"
        request.loras = [LoraChoice(path: "/x.safetensors", scale: 0.8, name: "X")]
        request.idImage = "FACE"
        request.idImageName = "face.png"
        request.idImages = ["FACE"]
        request.idImageNames = ["face.png"]
        request.idWeight = 1.0
        request.idStartStep = 2
        request.controlImage = "CTRL"
        request.controlModel = "controlnet-canny-sd15:fp16"
        request.controlScale = 1.0
        request.keyframes = [KeyframeCondition(frame: 1, image: "K", name: "k.png")]
        request.extendVideo = "VID"
        request.extendOverlapFrames = 9
        request.audioFile = "AUD"
        request.sourceVideo = "SRCVID"
        request.scheduler = "uni-pc"
        request.cfgPlus = true
        request.sampleShift = 5.0
        request.distillStrengthHigh = 1.0
        request.distillStrengthLow = 0.8
        request.guidanceOverrides = Ltx2GuidanceOverrides(
            stgScale: 1.0, stgBlocks: [3], rescaleScale: 0.7,
            modalityScale: 3.0, skipStep: 0)
        request.sourceFit = .default
        request.outputFormat = "png"
        request.upscaleModel = "real-esrgan-x4plus:fp16"
        request.title = "Robots"
        request.tags = ["metal"]
        request.collection = .named("Robots")
        request.originalPrompt = "a robot"
        request.promptTransform = PromptTransformProvenance(
            operation: .expand, rootPrompt: "a robot", sourcePrompt: "a robot",
            sourceKind: .direct, task: .textToImage, dimensions: [])
        request.batchId = "b1"
        request.batchIndex = 1
        request.batchCount = 4
        return request
    }

    /// The fixture above has to keep up with the struct, or the exhaustiveness
    /// check below would silently stop checking the new field.
    @Test func everyStoredPropertyIsPopulated() {
        let unset = Mirror(reflecting: populated()).children.compactMap { child -> String? in
            guard let label = child.label else { return nil }
            // `Optional.none` reflects as a `.optional` display style with no
            // children; anything else is a value.
            let mirror = Mirror(reflecting: child.value)
            guard mirror.displayStyle == .optional, mirror.children.isEmpty else { return nil }
            return label
        }
        #expect(unset.isEmpty, "add these to `populated()`: \(unset.sorted())")
    }

    /// **Fails today** the moment a stored property is added without a line in
    /// `encode(to:)`: the encoded key set is compared against the struct's own
    /// stored properties, not against a list somebody has to remember.
    @Test func everyStoredPropertyReachesTheWire() throws {
        let data = try MoldJSON.encoder.encode(populated())
        let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let encoded = Set(json.keys)

        let expected = Set(Mirror(reflecting: populated()).children.compactMap {
            $0.label.map(Self.snakeCased)
        })
        #expect(expected.subtracting(encoded).isEmpty,
                "never reach the wire: \(expected.subtracting(encoded).sorted())")
        // And nothing is invented: every key on the wire is a property.
        #expect(encoded.subtracting(expected).isEmpty,
                "encoded but not stored: \(encoded.subtracting(expected).sorted())")
    }

    /// `JSONEncoder.keyEncodingStrategy = .convertToSnakeCase`'s own rule, for
    /// the plain lowerCamelCase names this struct uses.
    private static func snakeCased(_ label: String) -> String {
        var out = ""
        for character in label {
            if character.isUppercase {
                out.append("_")
                out.append(Character(character.lowercased()))
            } else {
                out.append(character)
            }
        }
        return out
    }
}
