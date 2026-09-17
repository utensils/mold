import Foundation

// The wire key list, in an extension rather than nested in the struct's own
// declaration purely for size -- it is still `GenerateRequest.CodingKeys` and
// still visible from `encode(to:)` below, which is the whole reason it is
// spelled out at all rather than synthesized (a synthesized one is private).
extension GenerateRequest {
    /// Declared explicitly, rather than left to the compiler, ONLY because a
    /// synthesized `CodingKeys` is `private` and therefore invisible from
    /// `GenerateRequest+Encoding.swift`'s `encode(to:)` -- every other type
    /// in this package leaves `CodingKeys` to `MoldJSON`'s snake_case
    /// conversion (see its own doc comment) and this is the sole exception,
    /// forced by splitting the encoder out for size. Case names still match
    /// the properties one for one, so there is nothing here for a typo to
    /// hide behind.
    enum CodingKeys: String, CodingKey {
        case prompt, model, width, height, steps, guidance, batchSize, negativePrompt, seed,
             saveToGallery, frames, fps, pipeline, enableAudio, videoOnly, sourceImage,
             sourceImageName, strength, editImages, referenceWeight, maskImage, loras, idImage,
             idImageName, idImages, idImageNames, idWeight, idStartStep, controlImage,
             controlModel, controlScale, keyframes, extendVideo, extendOverlapFrames, audioFile,
             sourceVideo, scheduler, cfgPlus, sampleShift, distillStrengthHigh,
             distillStrengthLow, guidanceOverrides, sourceFit,
             outputFormat, upscaleModel, title, tags, collection,
             originalPrompt, promptTransform, batchId, batchIndex, batchCount
    }
}

// The hand-written wire encoding for `GenerateRequest`. Split out of
// `GenerateRequest.swift` purely for size -- the struct is closing in on
// eighty fields across M4's three slices and the two halves were already
// close to the 150-line lint on their own.
//
// `CodingKeys` is synthesized from the struct's stored properties (there is
// no manual `CodingKeys` declaration anywhere), which is what lets this file
// reference `.prompt`, `.maskImage`, and so on without redeclaring the key
// list -- see `MoldJSON`'s doc comment for why the wire and local decoders
// are deliberately different instances.
public extension GenerateRequest {
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(prompt, forKey: .prompt)
        try container.encode(model, forKey: .model)
        try container.encode(width, forKey: .width)
        try container.encode(height, forKey: .height)
        try container.encode(steps, forKey: .steps)
        try container.encode(guidance, forKey: .guidance)
        try container.encode(batchSize, forKey: .batchSize)
        try container.encodeIfPresent(negativePrompt, forKey: .negativePrompt)
        try container.encodeIfPresent(seed, forKey: .seed)
        try container.encodeIfPresent(saveToGallery, forKey: .saveToGallery)
        try container.encodeIfPresent(frames, forKey: .frames)
        try container.encodeIfPresent(fps, forKey: .fps)
        try container.encodeIfPresent(pipeline, forKey: .pipeline)
        try container.encodeIfPresent(enableAudio, forKey: .enableAudio)
        try container.encodeIfPresent(videoOnly, forKey: .videoOnly)
        try container.encodeIfPresent(sourceImage, forKey: .sourceImage)
        try container.encodeIfPresent(sourceImageName, forKey: .sourceImageName)
        try container.encodeIfPresent(strength, forKey: .strength)
        try container.encodeIfPresent(editImages, forKey: .editImages)
        try container.encodeIfPresent(referenceWeight, forKey: .referenceWeight)
        try container.encodeIfPresent(maskImage, forKey: .maskImage)
        try encodeLoras(into: &container)
        try container.encodeIfPresent(idImage, forKey: .idImage)
        try container.encodeIfPresent(idImageName, forKey: .idImageName)
        try container.encodeIfPresent(idImages, forKey: .idImages)
        try container.encodeIfPresent(idImageNames, forKey: .idImageNames)
        try container.encodeIfPresent(idWeight, forKey: .idWeight)
        try container.encodeIfPresent(idStartStep, forKey: .idStartStep)
        try container.encodeIfPresent(controlImage, forKey: .controlImage)
        try container.encodeIfPresent(controlModel, forKey: .controlModel)
        try container.encodeIfPresent(controlScale, forKey: .controlScale)
        try container.encodeIfPresent(keyframes, forKey: .keyframes)
        try container.encodeIfPresent(extendVideo, forKey: .extendVideo)
        try container.encodeIfPresent(extendOverlapFrames, forKey: .extendOverlapFrames)
        try container.encodeIfPresent(audioFile, forKey: .audioFile)
        try container.encodeIfPresent(sourceVideo, forKey: .sourceVideo)
        try container.encodeIfPresent(scheduler, forKey: .scheduler)
        try container.encodeIfPresent(cfgPlus, forKey: .cfgPlus)
        try container.encodeIfPresent(sampleShift, forKey: .sampleShift)
        try container.encodeIfPresent(distillStrengthHigh, forKey: .distillStrengthHigh)
        try container.encodeIfPresent(distillStrengthLow, forKey: .distillStrengthLow)
        try container.encodeIfPresent(guidanceOverrides, forKey: .guidanceOverrides)
        try container.encodeIfPresent(sourceFit, forKey: .sourceFit)
        try container.encodeIfPresent(outputFormat, forKey: .outputFormat)
        try container.encodeIfPresent(upscaleModel, forKey: .upscaleModel)
        try container.encodeIfPresent(title, forKey: .title)
        try container.encodeIfPresent(tags, forKey: .tags)
        try container.encodeIfPresent(collection, forKey: .collection)
        try container.encodeIfPresent(originalPrompt, forKey: .originalPrompt)
        // `wireSafe`, never the block itself: a strict Rust enum has no
        // `unknown` variant, and one re-encoded `"unknown"` refuses the whole
        // body (`PromptTransform+Wire.swift`).
        try container.encodeIfPresent(promptTransform?.wireSafe, forKey: .promptTransform)
        try container.encodeIfPresent(batchId, forKey: .batchId)
        try container.encodeIfPresent(batchIndex, forKey: .batchIndex)
        try container.encodeIfPresent(batchCount, forKey: .batchCount)
    }

    /// `loras[].path`/`loras[].scale` only -- `LoraChoice.name` is display
    /// only and must never reach the wire (`Lora.swift`'s own doc comment).
    /// A plain `container.encodeIfPresent(loras, forKey:)` would encode
    /// `LoraChoice`'s own `Codable` conformance instead, which includes
    /// `name`.
    private func encodeLoras(into container: inout KeyedEncodingContainer<CodingKeys>) throws {
        guard let loras, !loras.isEmpty else { return }
        try container.encode(loras.map { LoraWireEntry(path: $0.path, scale: $0.scale) }, forKey: .loras)
    }
}

/// The wire shape of one adapter: `types.rs:2648-2667`'s `LoraWeight` minus
/// `expert`, which this app never sets.
private struct LoraWireEntry: Encodable {
    let path: String
    let scale: Double
}
