import Foundation

/// Local authoring inputs, never a reusable upload lease or retained-source
/// permission. Non-inline references retain only an unresolved visible slot;
/// SavedReuse still owns origin revalidation before they can be submitted.
public struct DraftInputSnapshot: Codable, Hashable, Sendable {
    public let version: Int
    public let active: ParkedConditioning
    public let parked: ParkedConditioning
    public let sourcePixels: SourcePixels?
    public let boundaryKeyframes: [String: [KeyframeCondition]]
    public let boundaryWire: String?
    public let lastExclusiveWrite: ExclusiveWell?
    public let sourceFit: SourceFit
    public let retainedReuseFingerprint: String?

    public init(_ media: DraftMedia, retainedReuseFingerprint: String? = nil) {
        version = 1
        var active = ParkedConditioning()
        active.sourceImage = media.sourceImage
        active.sourceImageName = media.sourceImageName
        active.sourceImageOriginal = media.sourceImageOriginal
        active.sourceImageOriginalName = media.sourceImageOriginalName
        active.editImages = media.editImages
        active.generationReferences = media.generationReferences
        active.referenceWeight = media.referenceWeight
        active.maskImage = media.maskImage
        active.identity = media.identity
        active.control = media.control
        active.loras = media.loras
        active.keyframes = media.keyframes
        active.extendVideo = media.extendVideo
        active.extendVideoName = media.extendVideoName
        active.extendOverlapFrames = media.extendOverlapFrames
        active.audioFile = media.audioFile
        active.audioFileName = media.audioFileName
        active.sourceVideo = media.sourceVideo
        active.sourceVideoName = media.sourceVideoName
        self.active = Self.withoutScopedAuthority(active)
        parked = Self.withoutScopedAuthority(media.parked)
        sourcePixels = media.sourceImagePixels
        boundaryKeyframes = media.boundaryKeyframes
        boundaryWire = media.adoptedReferenceCapabilities.flatMap { BoundaryFramePolicy.resolve(capabilities: $0) }
        lastExclusiveWrite = media.lastExclusiveWrite
        sourceFit = media.sourceFit
        self.retainedReuseFingerprint = retainedReuseFingerprint
    }

    /// Restore authoring data before adopting the live model's capabilities.
    /// Endpoint keyframes return through their own protocol namespace, so an
    /// H3 closing frame cannot be mistaken for an interpolation keyframe.
    public func apply(to media: inout DraftMedia) throws {
        guard version == 1 else { throw CocoaError(.fileReadCorruptFile) }
        let active = Self.withoutScopedAuthority(active)
        media.sourceImage = active.sourceImage
        media.sourceImageName = active.sourceImageName
        media.sourceImageOriginal = active.sourceImageOriginal
        media.sourceImageOriginalName = active.sourceImageOriginalName
        media.editImages = active.editImages
        media.generationReferences = active.generationReferences
        media.referenceWeight = active.referenceWeight
        media.maskImage = active.maskImage
        media.identity = active.identity
        media.control = active.control
        media.loras = active.loras
        media.keyframes = active.keyframes
        media.extendVideo = active.extendVideo
        media.extendVideoName = active.extendVideoName
        media.extendOverlapFrames = active.extendOverlapFrames
        media.audioFile = active.audioFile
        media.audioFileName = active.audioFileName
        media.sourceVideo = active.sourceVideo
        media.sourceVideoName = active.sourceVideoName
        media.parked = Self.withoutScopedAuthority(parked)
        media.sourceImagePixels = sourcePixels
        media.boundaryKeyframes = boundaryKeyframes
        if let boundaryWire {
            guard ["h3-endpoints", "wan-pair"].contains(boundaryWire) else { throw CocoaError(.fileReadCorruptFile) }
            media.boundaryKeyframes[boundaryWire] = active.keyframes
            media.keyframes = media.boundaryKeyframes.removeValue(forKey: "interpolation") ?? []
        }
        media.lastExclusiveWrite = lastExclusiveWrite
        media.sourceFit = sourceFit
    }

    private static func withoutScopedAuthority(_ inputs: ParkedConditioning) -> ParkedConditioning {
        var result = inputs
        result.generationReferences = inputs.generationReferences.map { reference in
            var copy = reference
            copy.media = reference.media.authority == "inline" && reference.media.data != nil
                ? .init(authority: "inline", data: reference.media.data)
                : .init(authority: "descriptor")
            return copy
        }
        return result
    }
}
