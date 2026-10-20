import Foundation

/// A captured source authority. An identically named print on another machine is
/// never sufficient: the app must re-read this job through its captured backend.
public struct ChainContinuationSource: Hashable, Sendable {
    public let hostId: UUID
    public let instanceId: String?
    public let jobId: String
    public let take: Int
    public let script: ChainScript
    public let limits: ChainLimits
    public var model: String { script.chain.model }
    public var settings: ChainScriptSettings { script.chain }

    public init(hostId: UUID, instanceId: String?, detail: ChainJobDetail, limits: ChainLimits) throws {
        guard let instanceId, !instanceId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw SequenceDraftError.invalid("Reconnect to the original machine before continuing this video.")
        }
        guard limits.linkedParts?.supported == true else {
            throw SequenceDraftError.invalid("Update this machine before continuing a video.")
        }
        guard detail.state == .completed, let latest = detail.latestFinalize,
              let take = latest.take, take > 0, latest.galleryFilename != nil,
              let script = detail.script, script.schema == "mold.chain.v1",
              script.chain.model == detail.model, detail.model == limits.model else {
            throw SequenceDraftError.invalid("The original completed video is no longer available to continue.")
        }
        self.hostId = hostId; self.instanceId = instanceId; self.jobId = detail.id
        self.take = take; self.script = script; self.limits = limits
    }

    public func request(stages: [ChainStageInput], seed: UInt64? = nil,
                        composer: GenerateRequest? = nil) throws -> AuthoredChainRequest {
        let settings = script.chain
        var base = GenerateRequest(prompt: stages.first?.prompt ?? "", model: settings.model,
            width: settings.width, height: settings.height, steps: settings.steps,
            guidance: settings.guidance, seed: seed)
        base.sourceFit = settings.sourceFit
        base.fps = settings.fps; base.strength = settings.strength; base.enableAudio = settings.enableAudio
        if let originalComposer = composer {
            let composer = try originalComposer.authoredChainInputs()
            try AuthoredChainRequest.requireRepresentable(composer)
            base.seed = seed ?? composer.seed
            base.steps = composer.steps; base.guidance = composer.guidance; base.strength = composer.strength
            base.sourceFit = composer.sourceFit
            base.sourceImage = composer.sourceImage; base.keyframes = composer.keyframes
            base.references = composer.references; base.frames = composer.frames
            base.negativePrompt = composer.negativePrompt; base.loras = composer.loras
            base.title = composer.title; base.tags = composer.tags; base.collection = composer.collection
            base.originalPrompt = composer.originalPrompt; base.promptTransform = composer.promptTransform
            base.batchId = composer.batchId; base.batchIndex = composer.batchIndex; base.batchCount = composer.batchCount
        }
        var request = try AuthoredChainRequest(base, stages: stages, limits: limits,
            continueFrom: .init(jobId: jobId, finalizeTake: take), motionTailFrames: settings.motionTailFrames, repeatReferences: true)
        request.motionTailFrames = settings.motionTailFrames
        try request.validateSubmission(limits: limits)
        return request
    }

    public func requireOwner(hostId: UUID, instanceId: String?) throws {
        guard let instanceId, !instanceId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              self.hostId == hostId, self.instanceId == instanceId else {
            throw SequenceDraftError.invalid("Continue this video on its original machine.")
        }
    }
}
