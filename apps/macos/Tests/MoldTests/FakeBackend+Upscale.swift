import Foundation
import MoldClient
@testable import Mold

/// The state behind `FakeBackend.extras`.
///
/// Its own type, and its own file, for one reason: a class's stored
/// properties must live in its main declaration, and `FakeBackend.swift` is
/// the file every lane is editing. One box there, everything else here.
final class FakeExtras: @unchecked Sendable {
    /// Every durable clip upscale this fake machine holds, in the order
    /// `framewiseUpscales()` answers with. A test plants the LIST, never one
    /// job: recovery is a question about the whole listing, and MOVING a job
    /// -- rewriting this between two polls -- is how "it finished while we
    /// were asking" is said without a clock.
    var framewiseJobs: [VideoUpscaleJob] = []
    /// What `POST /api/video-upscale-jobs` answers. Unplanted throws, the
    /// same rule as every other route on this fake.
    var startedFramewiseAnswer: VideoUpscaleJob?
    var stillUpscaleAnswer: GalleryImageUpscale?
    var upscaledStills: [(filename: String, model: String)] = []
    var startedFramewise: [(filename: String, model: String)] = []
    var framewiseTransitions: [(id: String, to: FramewiseTransition)] = []
}

extension FakeBackend {
    func upscaleLibraryImage(
        filename: String, model: String, tileSize: Int?
    ) async throws -> GalleryImageUpscale {
        try record("upscaleLibraryImage")
        extras.upscaledStills.append((filename: filename, model: model))
        await pause("upscaleLibraryImage")
        guard let answer = extras.stillUpscaleAnswer else { throw notPlanted() }
        return answer
    }

    func startFramewiseUpscale(
        filename: String, model: String, tileSize: Int?
    ) async throws -> VideoUpscaleJob {
        try record("startFramewiseUpscale")
        extras.startedFramewise.append((filename: filename, model: model))
        await pause("startFramewiseUpscale")
        guard let answer = extras.startedFramewiseAnswer else { throw notPlanted() }
        return answer
    }

    func framewiseUpscales() async throws -> [VideoUpscaleJob] {
        try record("framewiseUpscales")
        await pause("framewiseUpscales")
        return extras.framewiseJobs
    }

    /// Answers the job's CURRENT state, so a test moves a job by rewriting
    /// `extras.framewiseJobs` rather than by waiting for anything.
    func framewiseUpscale(id: String) async throws -> VideoUpscaleJob {
        try record("framewiseUpscale")
        await pause("framewiseUpscale")
        guard let job = extras.framewiseJobs.first(where: { $0.id == id }) else {
            throw notPlanted()
        }
        return job
    }

    func transitionFramewiseUpscale(
        id: String, to transition: FramewiseTransition
    ) async throws -> VideoUpscaleJob {
        try record("transitionFramewiseUpscale")
        extras.framewiseTransitions.append((id: id, to: transition))
        await pause("transitionFramewiseUpscale")
        guard let job = extras.framewiseJobs.first(where: { $0.id == id }) else {
            throw notPlanted()
        }
        return job
    }
}
