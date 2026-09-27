import Foundation

// Parking and restoring the S6b clip-media fields across a recipe switch --
// split from `DraftMedia+Park.swift` purely for size. Same rule as every
// other reconciliation here: a live value always wins over a parked one.
public extension DraftMedia {
    /// Parks/restores the keyframe list, same shape as `reconcileEditImages`
    /// minus the count cap -- there is no per-recipe keyframe limit.
    mutating func reconcileKeyframes(supported: Bool) {
        if supported {
            if keyframes.isEmpty, !parked.keyframes.isEmpty {
                keyframes = parked.keyframes
                parked.keyframes = []
            }
        } else if !keyframes.isEmpty {
            parked.keyframes = keyframes
            keyframes = []
        }
    }

    /// Parks/restores the extend continuation -- the video, its display
    /// name and its overlap travel together, like `reconcileSourceImage`'s
    /// pair.
    mutating func reconcileExtend(supported: Bool) {
        if supported {
            if extendVideo == nil, let restored = parked.extendVideo {
                extendVideo = restored
                extendVideoName = parked.extendVideoName
                extendOverlapFrames = parked.extendOverlapFrames
                parked.extendVideo = nil
                parked.extendVideoName = nil
                parked.extendOverlapFrames = nil
            }
        } else if extendVideo != nil {
            parked.extendVideo = extendVideo
            parked.extendVideoName = extendVideoName
            parked.extendOverlapFrames = extendOverlapFrames
            extendVideo = nil
            extendVideoName = nil
            extendOverlapFrames = nil
        }
    }

    /// Parks/restores the conditioning audio file and its display name.
    mutating func reconcileAudioFile(supported: Bool) {
        if supported {
            if audioFile == nil, let restored = parked.audioFile {
                audioFile = restored
                audioFileName = parked.audioFileName
                parked.audioFile = nil
                parked.audioFileName = nil
            }
        } else if audioFile != nil {
            parked.audioFile = audioFile
            parked.audioFileName = audioFileName
            audioFile = nil
            audioFileName = nil
        }
    }

    /// Parks/restores the reference source video and its display name.
    mutating func reconcileSourceVideo(supported: Bool) {
        if supported {
            if sourceVideo == nil, let restored = parked.sourceVideo {
                sourceVideo = restored
                sourceVideoName = parked.sourceVideoName
                parked.sourceVideo = nil
                parked.sourceVideoName = nil
            }
        } else if sourceVideo != nil {
            parked.sourceVideo = sourceVideo
            parked.sourceVideoName = sourceVideoName
            sourceVideo = nil
            sourceVideoName = nil
        }
    }
}
