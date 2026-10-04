import Foundation

extension ReferenceUploadMetadata {
    func canonical(original: GenerationReference, index expected: Int) throws -> GenerationReference {
        let mismatch = ReferenceUploadPolicy.refusal("REFERENCE_UPLOAD_RESPONSE_MISMATCH", "The host returned mismatched reference metadata.")
        guard index == expected, kind == original.kind,
              sha256.lowercased() == original.provenance?.sha256?.lowercased(),
              name == original.provenance?.name else { throw mismatch }
        var canonical = GenerationReference(kind: kind, media: .init(authority: "descriptor"), mimeType: mimeType,
                                            provenance: original.provenance)
        canonical.provenance?.sha256 = sha256.lowercased()
        func positive(_ n: Int?) -> Bool { (n ?? 0) > 0 }
        func audioChannelsValid(_ n: Int?) -> Bool { n == 1 || n == 2 }
        switch kind {
        case "image":
            guard mimeType == original.mimeType, positive(width), positive(height) else { throw mismatch }
            canonical.width = width; canonical.height = height
        case "audio":
            guard ["audio/wav", "audio/x-wav", "audio/wave"].contains(original.mimeType.lowercased()),
                  mimeType == "audio/wav", positive(durationMs), positive(sampleRate), positive(sampleCount),
                  audioChannelsValid(channels) else { throw mismatch }
            canonical.durationMs = durationMs; canonical.sampleRate = sampleRate
            canonical.sampleCount = sampleCount; canonical.channels = channels
        case "video":
            guard original.mimeType == "video/mp4", mimeType == "video/mp4", positive(width), positive(height),
                  positive(frameCount), positive(durationMs), let fps, fps.isFinite, fps > 0 else { throw mismatch }
            let hasAudio = hasAudio ?? false
            if hasAudio {
                guard positive(audioDurationMs), positive(audioSampleCount), positive(audioSampleRate),
                      audioChannelsValid(audioChannels) else { throw mismatch }
            } else {
                guard audioDurationMs == nil, audioSampleCount == nil,
                      audioSampleRate == nil, audioChannels == nil else { throw mismatch }
            }
            canonical.width = width; canonical.height = height; canonical.frameCount = frameCount
            canonical.durationMs = durationMs; canonical.fps = fps; canonical.hasAudio = hasAudio
            canonical.audioDurationMs = audioDurationMs; canonical.audioSampleCount = audioSampleCount
            canonical.audioSampleRate = audioSampleRate; canonical.audioChannels = audioChannels
        default: throw mismatch
        }
        return canonical
    }
}
