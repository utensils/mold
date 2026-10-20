import Foundation

public extension VideoJobRequest {
    /// Display-only planning never transfers pictures, audio, videos, upload
    /// handles, or retained-media descriptors. Installed model/task identity
    /// owns grid/carry limits; full validation still checks required inputs.
    var planningProjection: Self {
        var result = self
        result.validationRevision = nil
        result.retainedMediaSession = nil
        result.generation.pendingReferenceInputs = nil
        result.generation.sourceImage = nil; result.generation.sourceImageName = nil
        result.generation.editImages = nil; result.generation.references = nil
        result.generation.referenceWeight = nil; result.generation.maskImage = nil
        result.generation.idImage = nil; result.generation.idImageName = nil
        result.generation.idImages = nil; result.generation.idImageNames = nil
        result.generation.idWeight = nil; result.generation.idStartStep = nil
        result.generation.controlImage = nil; result.generation.controlModel = nil
        result.generation.controlScale = nil; result.generation.keyframes = nil
        result.generation.extendVideo = nil; result.generation.extendOverlapFrames = nil
        result.generation.audioFile = nil; result.generation.sourceVideo = nil
        return result
    }
}
