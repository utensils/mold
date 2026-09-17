import Foundation

public extension ExpandTask {
    /// What kind of render a prompt is being written for, derived from the
    /// request it will ride on.
    ///
    /// Port of `studio/lib/expandTask.ts:55-121`, itself a browser-safe mirror
    /// of `mold_core::ExpandTask::for_generation`. `/api/expand` resolves a
    /// task internally from the family alone and never echoes it back
    /// (`routes.rs:4036-4045`), so this app used to hard-code `.textToImage`
    /// into every accepted offer -- an LTX-2 clip, an img2img render and a
    /// keyframe interpolation all recorded that their prompt was written for
    /// a still (findings 01#13, 02#12). Derived here, sent as
    /// `ExpandRequest.task`, and recorded as provenance, so the server and
    /// the print agree.
    ///
    /// The one arm not ported is MiniMax-H3's ordered `references`: this app
    /// has no H3 reference strip, so there is nothing to read.
    static func forRequest(family: String?, request: GenerateRequest) -> ExpandTask {
        let normalized = (family ?? "").trimmingCharacters(in: .whitespaces).lowercased()
        let isH3 = h3Families.contains(normalized)
        guard isH3 || videoFamilies.contains(normalized) else { return .textToImage }
        if isH3 { return h3Task(request) }

        switch request.pipeline {
        case "t2a": return .textToAudio
        case "retake": return .retake
        case "keyframe": return .keyframeInterpolation
        case "a2-vid", "lip-dub": return .audioDrivenVideo
        case nil:
            // The engine's implicit-pipeline priority.
            if request.audioFile != nil { return .audioDrivenVideo }
            if (request.keyframes?.count ?? 0) > 1 { return .keyframeInterpolation }
        default:
            break
        }
        if request.sourceVideo != nil || request.extendVideo != nil { return .videoToVideo }
        if request.sourceImage != nil { return .imageToVideo }
        // A single-frame wan render with no conditioning is a still (#798):
        // prompt work is visual description, not shot direction. Deliberately
        // AFTER the source checks, so a source-conditioned one-frame request
        // keeps its source-preserving contract.
        if wanFamilies.contains(normalized), request.frames == 1 { return .textToImage }
        return .textToVideo
    }

    private static func h3Task(_ request: GenerateRequest) -> ExpandTask {
        let first = request.sourceImage != nil
        let last = !(request.keyframes ?? []).isEmpty
        if first, last { return .keyframeInterpolation }
        if first || last { return .imageToVideo }
        return .textToVideo
    }

    private static let h3Families: Set<String> = ["minimax-h3", "minimax_h3", "minimaxh3"]
    private static let wanFamilies: Set<String> = ["wan", "wan2.1", "wan2.2"]
    private static let videoFamilies: Set<String> =
        Set(["ltx2", "ltx-2", "ltx-video"]).union(wanFamilies)
}
