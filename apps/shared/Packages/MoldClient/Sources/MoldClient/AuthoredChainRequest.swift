import Foundation

/// An immutable completed take on the same machine, never a media path.
public struct ChainContinuation: Codable, Hashable, Sendable {
    public var jobId: String
    public var finalizeTake: Int
    public init(jobId: String, finalizeTake: Int) { self.jobId = jobId; self.finalizeTake = finalizeTake }
}

public struct ChainLora: Codable, Hashable, Sendable {
    public var path: String
    public var scale: Double
    public var name: String?
    public init(path: String, scale: Double, name: String? = nil) {
        self.path = path; self.scale = scale; self.name = name
    }
}

/// Stage-local authored inputs. Source payloads are omitted by ordinary job detail.
public struct ChainStageInput: Codable, Hashable, Sendable {
    public var prompt: String
    public var frames: Int
    public var sourceImage: String?
    public var negativePrompt: String?
    public var seedOffset: UInt64?
    public var transition: String
    public var fadeFrames: Int?
    public var loras: [ChainLora]?
    public var generationReferences: [GenerationReference]?
    public var keyframes: [KeyframeCondition]?
    public var authoredMediaSha256: String?
    public init(prompt: String, frames: Int, transition: String = "smooth") {
        self.prompt = prompt; self.frames = frames; self.transition = transition
    }
}

/// Authored work has durable history and publishes a recoverable Library job id.
/// This is deliberately separate from AutoChainRequest's hidden implementation detail.
public struct AuthoredChainRequest: Codable, Hashable, Sendable {
    public var continueFrom: ChainContinuation?
    public var model: String
    public var stages: [ChainStageInput]
    public var motionTailFrames: Int
    public var width: Int
    public var height: Int
    public var fps: Int
    public var seed: UInt64?
    public var steps: Int
    public var guidance: Double
    public var strength: Double
    public var outputFormat: String = "mp4"
    public var outputMode: String = "sequence"
    public var enableAudio: Bool?
    public var sourceFit: SourceFit?
    public var title: String?
    public var tags: [String]?
    public var collection: CollectionRef?
    public var originalPrompt: String?
    public var promptTransform: PromptTransformProvenance?
    public var batchId: String?
    public var batchIndex: Int?
    public var batchCount: Int?

    public init(_ request: GenerateRequest, stages: [ChainStageInput], limits: ChainLimits,
                continueFrom: ChainContinuation? = nil, motionTailFrames: Int? = nil, repeatReferences: Bool = false) throws {
        let prepared = try request.authoredChainInputs()
        try Self.requireRepresentable(prepared)
        self.continueFrom = continueFrom
        model = prepared.model; self.stages = stages
        try Self.applyComposerInputs(prepared, to: &self.stages, repeatReferences: repeatReferences)
        self.motionTailFrames = motionTailFrames ?? limits.authoring?.motionTailFrames ?? 17
        width = prepared.width; height = prepared.height; fps = prepared.fps ?? limits.fps ?? 24
        seed = prepared.seed; steps = prepared.steps; guidance = prepared.guidance
        strength = prepared.strength ?? 1; enableAudio = prepared.enableAudio
        sourceFit = prepared.sourceFit
        title = prepared.title; tags = prepared.tags; collection = prepared.collection
        originalPrompt = prepared.originalPrompt; promptTransform = prepared.promptTransform?.wireSafe
        batchId = prepared.batchId; batchIndex = prepared.batchIndex; batchCount = prepared.batchCount
        try validateSubmission(limits: limits)
    }

    public func validateSubmission(limits: ChainLimits) throws {
        try validateMediaBudget()
        if sourceFit != nil, limits.authoring?.sourceFitSupported != true {
            throw SequenceDraftError.invalid("Update this machine before generating joined clips with source crop settings.")
        }
        guard limits.supportsSequence else {
            throw SequenceDraftError.invalid(limits.sequenceUnsupportedReason ?? "This model does not support joined clips.")
        }
        guard model == limits.model, width > 0, height > 0, fps > 0,
              !stages.isEmpty, stages.count <= limits.maxStages else {
            throw SequenceDraftError.invalid("Choose a supported model and keep this part within its clip limit.")
        }
        if continueFrom != nil, limits.linkedParts?.supported != true {
            throw SequenceDraftError.invalid("Update this machine before continuing a video.")
        }
        guard continueFrom == nil || (continueFrom!.finalizeTake > 0 && !continueFrom!.jobId.isEmpty) else {
            throw SequenceDraftError.invalid("The original completed video is unavailable.")
        }
        if continueFrom != nil, stages.first?.transition != "smooth" {
            throw SequenceDraftError.invalid("The first clip of a continued part must use Smooth.")
        }
        guard stages.allSatisfy({ !$0.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && $0.frames > 0 && $0.frames <= limits.framesPerClipCap }) else {
            throw SequenceDraftError.invalid("Each clip needs a prompt and a supported length.")
        }
        guard motionTailFrames >= 0, stages.enumerated().allSatisfy({ index, stage in
            let carries = (index > 0 || continueFrom != nil) && stage.transition == "smooth"
            return !carries || stage.frames > motionTailFrames
        }) else {
            throw SequenceDraftError.invalid("Each joined clip must include new frames after its incoming overlap.")
        }
        if let authoring = limits.authoring {
            guard stages.allSatisfy({ $0.frames >= authoring.minFrames
                && ($0.frames - authoring.frameOffset) % max(1, authoring.frameStep) == 0 }) else {
                throw SequenceDraftError.invalid("Choose a clip length supported by this model.")
            }
        }
        if let transitions = limits.transitionModes,
           !stages.allSatisfy({ transitions.contains($0.transition) }) {
            throw SequenceDraftError.invalid("This machine does not support the selected transition.")
        }
        if enableAudio == true, limits.supportsAudio == false {
            throw SequenceDraftError.invalid("This model does not generate audio.")
        }
        if continueFrom != nil, motionTailFrames > 0,
           stages.first?.sourceImage != nil || stages.first?.keyframes?.contains(where: { $0.frame == 0 }) == true {
            throw SequenceDraftError.invalid("The previous video's final frame supplies the next clip's opening.")
        }
    }

    private func validateMediaBudget() throws {
        var decodedBytes = 0
        var members = continueFrom == nil ? 0 : motionTailFrames
        for stage in stages {
            var payloads: [String] = []
            if let image = stage.sourceImage { payloads.append(image); members += 1 }
            for keyframe in stage.keyframes ?? [] { payloads.append(keyframe.image); members += 1 }
            for reference in stage.generationReferences ?? [] {
                guard reference.media.authority == "inline", reference.media.data?.isEmpty == false,
                      reference.media.handle == nil, reference.media.path == nil else {
                    throw SequenceDraftError.invalid("Joined clips require original inline reference media. Reattach the reference before generating.")
                }
                members += 1
                if let data = reference.media.data { payloads.append(data) }
            }
            guard members <= 64 else {
                throw SequenceDraftError.invalid("This part contains too many attachments. Use fewer clips or references.")
            }
            for payload in payloads {
                let count = payload.utf8.count
                let padding = payload.hasSuffix("==") ? 2 : (payload.hasSuffix("=") ? 1 : 0)
                let bytes = max(0, count / 4 * 3 - padding)
                guard bytes <= 32 * 1_024 * 1_024 - decodedBytes else {
                    throw SequenceDraftError.invalid("Attachments across this part exceed 32 MiB. Use fewer clips or smaller references.")
                }
                decodedBytes += bytes
            }
        }
    }

    static func applyComposerInputs(_ request: GenerateRequest, to stages: inout [ChainStageInput],
                                    repeatReferences: Bool) throws {
        guard !stages.isEmpty else { return }
        if stages[0].sourceImage == nil { stages[0].sourceImage = request.sourceImage }
        for keyframe in request.keyframes ?? [] {
            let closing = request.frames.map { keyframe.frame == $0 - 1 && keyframe.frame > 0 } ?? false
            let index = closing ? stages.count - 1 : 0
            var mapped = keyframe
            if closing { mapped.frame = stages[index].frames - 1 }
            guard mapped.frame >= 0, mapped.frame < stages[index].frames else {
                throw SequenceDraftError.invalid("A keyframe lies outside its clip. Adjust its position before adding clips.")
            }
            if let existing = stages[index].keyframes?.first(where: { $0.frame == mapped.frame }) {
                guard existing == mapped else {
                    throw SequenceDraftError.invalid("Remove the duplicate global and clip endpoint images.")
                }
            } else { stages[index].keyframes = (stages[index].keyframes ?? []) + [mapped] }
        }
        for index in stages.indices {
            if stages[index].generationReferences == nil, index == 0 || repeatReferences {
                stages[index].generationReferences = request.references
            }
            if stages[index].negativePrompt == nil { stages[index].negativePrompt = request.negativePrompt }
            if stages[index].loras == nil {
                stages[index].loras = request.loras?.map { .init(path: $0.path, scale: $0.scale, name: $0.name) }
            }
        }
    }

    static func requireRepresentable(_ request: GenerateRequest) throws {
        var unsupported: [String] = []
        if request.pendingReferenceInputs?.isEmpty == false { unsupported.append("unprepared reference media") }
        if request.batchSize != 1 { unsupported.append("multiple copies") }
        if request.audioFile != nil { unsupported.append("conditioning audio") }
        if request.sourceVideo != nil { unsupported.append("source video") }
        if request.extendVideo != nil || request.extendOverlapFrames != nil { unsupported.append("video extension") }
        if request.pipeline != nil { unsupported.append("pipeline override") }
        if request.guidanceOverrides != nil { unsupported.append("guidance overrides") }
        if request.maskImage != nil { unsupported.append("mask") }
        if request.editImages?.isEmpty == false { unsupported.append("edit images") }
        if request.idImage != nil || request.idImages?.isEmpty == false { unsupported.append("identity photographs") }
        if request.controlImage != nil || request.controlModel != nil { unsupported.append("ControlNet") }
        if request.scheduler != nil { unsupported.append("scheduler override") }
        if request.cfgPlus == true { unsupported.append("CFG++") }
        if request.sampleShift != nil { unsupported.append("sample shift") }
        if request.distillStrengthHigh != nil || request.distillStrengthLow != nil { unsupported.append("distillation overrides") }
        if request.transparentBackground == true { unsupported.append("transparency") }
        if request.upscaleModel != nil || request.spatialUpscale != nil || request.temporalUpscale != nil { unsupported.append("upscaling") }
        if request.videoOnly == true { unsupported.append("video-only override") }
        if request.referenceWeight != nil { unsupported.append("reference-weight override") }
        if request.saveToGallery == false { unsupported.append("saving outside the Library") }
        guard unsupported.isEmpty else {
            throw SequenceDraftError.invalid("Joined clips cannot preserve \(unsupported.joined(separator: ", ")). Remove these options before generating.")
        }
    }
}

public enum SequenceDraftError: Error, LocalizedError, Equatable, Sendable {
    case invalid(String)
    public var errorDescription: String? { if case let .invalid(message) = self { message } else { nil } }
}
