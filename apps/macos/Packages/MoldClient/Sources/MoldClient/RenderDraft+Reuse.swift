import Foundation

// Use These Settings: a finished print's provenance, back in the draft.
//
// Port of `applyMetadataToForm` (`desktop/src/lib/generateForm.ts:1455-1609`),
// in that order. Two rules run through the whole thing:
//
// 1. A FRESH draft, never a mutation of what is on screen. Metadata carries
//    no bytes, so every byte-bearing well starts empty and only the retained
//    source-media probe may fill one in -- pairing a staged picture with
//    somebody else's print is how a restore renders a thing nobody asked for.
// 2. Nothing here consults a recipe. The caller adopts the print's model
//    straight after (`GenerateController.adopt(keepingDraft: true)`), and
//    THAT is what clamps, coerces the output format, and parks a control the
//    target recipe does not advertise. Deciding it here would mean deciding
//    it twice, with the second answer winning.
public extension RenderDraft {
    init(reusing metadata: OutputMetadata) {
        self.init()

        // A sequence's `prompt` is every stage newline-joined, so this takes
        // the FIRST stage. There is no door back to the sequence on any
        // surface -- a stitched print is provenance.
        prompt = metadata.firstStagePrompt
        originalPrompt = metadata.originalPrompt
        promptTransform = metadata.promptTransform
        // A recorded "" is an explicit empty; absence is a print made before
        // the field. The draft carries no recipe default to tell them apart
        // against, so both restore as empty and `adopting` clears it again on
        // a recipe with no negative branch.
        negativePrompt = metadata.negativePrompt ?? ""
        title = metadata.title ?? ""
        tags = metadata.tags ?? []
        collectionName = metadata.collection

        // The generation canvas beats the delivered one: an upscaled print's
        // `width` describes the file, not the render to repeat.
        width = metadata.generationWidth ?? metadata.width ?? width
        height = metadata.generationHeight ?? metadata.height ?? height
        // The recorded size IS the answer, so the canvas is manual. Left
        // following a source, it would be re-derived the moment the retained
        // source re-attaches and the print would come back another shape.
        canvasIntent = .manual
        steps = metadata.steps ?? steps
        guidance = metadata.guidance ?? guidance
        if let strength = metadata.strength { self.strength = strength }
        // Restored AND locked, as web does (`useGenerateForm.ts`: a recorded
        // seed makes `seedMode` static): Use These Settings means these
        // settings, the seed among them, and the lock is one click to undo.
        // Left unlocked, a reuse silently rendered a different picture from
        // the one it was named after (UAT 2026-09-17 #3).
        seed = metadata.seed
        locksSeed = metadata.seed != nil

        restoreSampler(from: metadata)
        restoreConditioning(from: metadata)
        restoreOutput(from: metadata)
    }

    private mutating func restoreSampler(from metadata: OutputMetadata) {
        advanced.scheduler = metadata.scheduler
        advanced.cfgPlus = metadata.cfgPlus ?? false
        advanced.sampleShift = metadata.sampleShift
        advanced.distillStrengthHigh = metadata.distillStrengthHigh
        advanced.distillStrengthLow = metadata.distillStrengthLow
        guard let overrides = metadata.guidanceOverrides else { return }
        advanced.stgScale = overrides.stgScale
        advanced.rescaleScale = overrides.rescaleScale
        advanced.modalityScale = overrides.modalityScale
        advanced.skipStep = overrides.skipStep
        // The draft holds the block list as the TEXT someone types, so a
        // half-typed entry is not rewritten under the cursor.
        advanced.stgBlocks = (overrides.stgBlocks ?? []).map(String.init)
            .joined(separator: ", ")
    }

    private mutating func restoreConditioning(from metadata: OutputMetadata) {
        if let fit = metadata.sourceFit { media.sourceFit = fit }
        media.loras = Self.adapters(of: metadata)
        if metadata.controlModel != nil || metadata.controlScale != nil {
            // The picture is bytes and metadata carries none; the adapter and
            // its strength are settings and restore exactly. The request
            // builder ships the pair or neither, so a half-filled control is
            // a staged state and never a 422.
            media.control = ControlConditioning(
                model: metadata.controlModel,
                scale: metadata.controlScale ?? Control.defaultScale)
        }
        // The knobs restore; the face does not exist in metadata, which
        // records a digest and never a person. An empty set ships NOTHING at
        // all (`applyIdentity` returns early), so the knobs cannot reach the
        // wire without the photograph they describe -- the retained
        // source-media probe is what puts one back.
        guard metadata.carriedAFace else { return }
        media.identity = IdentityConditioning(
            photos: [],
            weight: metadata.idWeight ?? Identity.weightDefault,
            startStep: metadata.idStartStep ?? Identity.startStepDefault)
    }

    private mutating func restoreOutput(from metadata: OutputMetadata) {
        frames = metadata.frames
        fps = metadata.fps.map { Int($0.rounded()) }
        enableAudio = metadata.enableAudio ?? false
        videoOnly = metadata.videoOnly ?? false
        // `pipeline` records what RAN; only `pipeline_requested` says the
        // author named it. Restoring the former on a print that named nothing
        // pins a choice nobody made (`generateForm.ts` `pipelineForSettingsReuse`).
        pipeline = metadata.pipelineRequested == true ? metadata.pipeline : nil
        outputFormat = metadata.outputFormat
        upscaleModel = metadata.upscaleModel
    }

    /// The adapter stack, with the legacy singular pair read as a stack of
    /// one -- that is all a print made before `loras` existed carries.
    private static func adapters(of metadata: OutputMetadata) -> [LoraChoice] {
        let recorded = metadata.loras ?? metadata.lora.map {
            [MetadataLora(path: $0, scale: metadata.loraScale ?? Lora.defaultScale)]
        } ?? []
        return recorded.prefix(Lora.defaultMaxStack).map {
            LoraChoice(path: $0.path, scale: $0.scale, name: adapterName(of: $0.path))
        }
    }

    /// Port of `loraNameFromPath` (`generateForm.ts:1422-1426`), minus the
    /// camera-motion labels this app has no table for.
    private static func adapterName(of path: String) -> String {
        let base = path.split(separator: "/").last.map(String.init) ?? path
        guard base.lowercased().hasSuffix(".safetensors") else { return base }
        return String(base.dropLast(".safetensors".count))
    }
}
