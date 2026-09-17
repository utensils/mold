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
    nonisolated(unsafe) var framewiseJobs: [VideoUpscaleJob] = []
    /// What `POST /api/video-upscale-jobs` answers. Unplanted throws, the
    /// same rule as every other route on this fake.
    nonisolated(unsafe) var startedFramewiseAnswer: VideoUpscaleJob?
    nonisolated(unsafe) var stillUpscaleAnswer: GalleryImageUpscale?
    nonisolated(unsafe) var upscaledStills: [(filename: String, model: String)] = []
    nonisolated(unsafe) var startedFramewise: [(filename: String, model: String)] = []
    nonisolated(unsafe) var framewiseTransitions: [(id: String, to: FramewiseTransition)] = []

    /// Parks every `framewiseUpscale` answer until `releaseFramewise()`.
    ///
    /// Deliberately NOT `delays`, which sleeps: a sleep ends the instant its
    /// task is cancelled, so a poll that is replaced answers immediately and
    /// the one sequence worth testing -- an answer still in flight ACROSS the
    /// replacement -- cannot be produced with it. A response already on the
    /// wire does not vanish because the app changed its mind.
    ///
    /// Setting it ARMS it, so hold -> release -> hold again works; the
    /// release is a LATCH, so an ask that records its call before parking its
    /// continuation is not left waiting for a wake-up that already happened.
    nonisolated(unsafe) var framewiseHeldOpen = false {
        didSet { if framewiseHeldOpen { framewiseReleased = false } }
    }
    nonisolated(unsafe) var framewiseReleased = false
    nonisolated(unsafe) var framewiseWaiters: [CheckedContinuation<Void, Never>] = []

    /// What `/api/activity` answers. `nil` THROWS, the same rule as every
    /// other unplanted route on this fake -- a store that reaches for it
    /// unexpectedly fails the test rather than quietly getting an idle
    /// machine.
    nonisolated(unsafe) var activitySnapshot: ActiveWorkSnapshot?

    /// Every whole-queue gate call, as `paused` was ASKED for.
    nonisolated(unsafe) var gateCalls: [Bool] = []
    /// What the gate answers with, overriding the ask -- how a test plants a
    /// machine that refuses to move, and proves the store writes the
    /// MACHINE's answer rather than its own intent.
    nonisolated(unsafe) var gateAnswer: Bool?
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
        if extras.framewiseHeldOpen, !extras.framewiseReleased {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                extras.framewiseWaiters.append(continuation)
            }
        }
        guard let job = extras.framewiseJobs.first(where: { $0.id == id }) else {
            throw notPlanted()
        }
        return job
    }

    /// Lets every parked `framewiseUpscale` answer land.
    func releaseFramewise() {
        extras.framewiseReleased = true
        let waiting = extras.framewiseWaiters
        extras.framewiseWaiters = []
        for continuation in waiting { continuation.resume() }
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
