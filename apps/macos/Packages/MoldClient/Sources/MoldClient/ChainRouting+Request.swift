import Foundation

// Routing a BUILT request, and the body an automatic chain submits.
public extension ChainRouting {
    /// Where this request runs: one denoise, an ephemeral chain, or a refusal
    /// by name. Port of `decideGenerateRequestRouting`
    /// (`chainRouting.ts:311-369`).
    ///
    /// `advertisedMaxFrames` is the host's own single-request ceiling --
    /// `TemporalProfile.durationCappedMaxFrames(fps:)`, the figure the door
    /// that actually refuses applies. Reading the family table where one half
    /// of this function trusted the row and the other did not is the bug
    /// studio's comment at `:349-353` records.
    static func decide(
        request: GenerateRequest, family: String?,
        sourceImage: SourceImageCapability? = nil,
        tierDefault: Int? = nil, advertisedMaxFrames: Int? = nil,
        motionTail: Int = defaultMotionTail
    ) -> Decision {
        let decision = decide(
            frames: request.frames, family: family, model: request.model,
            motionTail: motionTail, sourceImage: sourceImage,
            tierDefault: tierDefault, advertisedMaxFrames: advertisedMaxFrames)
        guard case let .chain(clipFrames, _, _) = decision else { return decision }

        let unsupported = AutoChainField.unsupported(in: request)
        guard !unsupported.isEmpty else { return decision }

        // The options win where one denoise can still hold the whole clip:
        // preserving what was asked for beats splitting it.
        let frames = request.frames ?? 0
        if let cap = advertisedMaxFrames, frames <= cap {
            return .single(preserved: unsupported)
        }
        let fps = Swift.max(1, request.fps ?? 24)
        let determiner = unsupported.count == 1 ? "that option" : "those options"
        return .reject("\(frames) frames exceeds the \(advertisedMaxFrames ?? clipFrames)-frame "
            + "single-shot limit at \(fps) fps, and automatic chaining can’t preserve "
            + "\(AutoChainField.list(unsupported)). Reduce Frames, remove \(determiner), "
            + "or script the clips yourself with mold run --script.")
    }
}

/// The body `POST /api/chain-jobs` takes for an automatic split.
///
/// Port of `buildAutoChainRequest` (`desktop/src/lib/chainRouting.ts:30-62`).
/// `ephemeral` is the whole distinction between this and an authored
/// sequence: absent means authored, and an ephemeral job is hidden from
/// sequence history, emits no `chain_job_queued`, and publishes ONE print
/// whose `chain_job_id` is `None`.
public struct AutoChainRequest: Encodable, Hashable, Sendable {
    public let outputMode = "one-shot"
    public let model: String
    public let prompt: String
    public let totalFrames: Int
    public let clipFrames: Int
    public let motionTailFrames: Int
    public let width: Int
    public let height: Int
    public let fps: Int?
    public let seed: UInt64?
    public let steps: Int
    public let guidance: Double
    public let strength: Double?
    public let outputFormat: String?
    public let sourceImage: String?
    public let enableAudio: Bool?
    public let originalPrompt: String?
    /// A stitched long video is still ONE print. Dropping these would silently
    /// unfile -- and untitle -- every video long enough to auto-chain, which
    /// is exactly the render most worth naming.
    public let title: String?
    public let tags: [String]?
    public let collection: CollectionRef?
    public let ephemeral = true

    public init(_ request: GenerateRequest, clipFrames: Int, motionTail: Int) {
        model = request.model
        prompt = request.prompt
        totalFrames = request.frames ?? 0
        self.clipFrames = clipFrames
        motionTailFrames = motionTail
        width = request.width
        height = request.height
        fps = request.fps
        seed = request.seed
        steps = request.steps
        guidance = request.guidance
        strength = request.strength
        outputFormat = request.outputFormat
        sourceImage = request.sourceImage
        enableAudio = request.enableAudio
        originalPrompt = request.originalPrompt
        title = request.title
        tags = request.tags
        collection = request.collection
    }
}
