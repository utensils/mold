import Foundation
import Testing
@testable import MoldClient

/// The framewise clip upscale, read from a real machine's own answer.
///
/// `video-upscale-jobs-hal9000.json` is `GET /api/video-upscale-jobs` on
/// hal9000 (mold 0.29.0, b015496e, captured 2026-09-17, keyless). Three
/// durable rows: one completed and two failed.
///
/// **Fails today**: nothing in this app reads `/api/video-upscale-jobs` at all.
@MainActor
struct UpscaleWireTests {

    private func hostJobs() throws -> [VideoUpscaleJob] {
        try MoldJSON.decoder.decode(
            [VideoUpscaleJob].self,
            from: RepoFixtures.fixture("video-upscale-jobs-hal9000.json"))
    }

    @Test func aRealMachinesDurableUpscalesDecode() throws {
        let jobs = try hostJobs()
        #expect(jobs.count == 3)
        let completed = try #require(jobs.first { $0.state == .completed })
        #expect(completed.id.hasPrefix("vup-"))
        #expect(completed.model == "real-esrgan-x4plus:fp16")
        #expect(completed.completedFrames == 124)
        #expect(completed.totalFrames == 124)
        #expect(completed.outputFilename?.isEmpty == false)
        guard case let .library(filename) = try #require(completed.source) else {
            Issue.record("the captured row names a library source")
            return
        }
        #expect(filename.hasSuffix(".mp4"))
    }

    /// A failed row carries the host's own sentence, and `framewiseStatus`
    /// hands it back verbatim rather than saying "failed".
    @Test func aFailedUpscaleSpeaksTheMachinesOwnSentence() throws {
        let failed = try #require(try hostJobs().first { $0.state == .failed })
        let reason = try #require(failed.error)
        #expect(UpscalePlan.status(of: failed) == reason)
    }

    /// A state added after this build must not lose the whole listing --
    /// every other open enum on this wire degrades the same way.
    @Test func aStateThisBuildHasNeverHeardOfStillDecodes() throws {
        let json = #"""
        [{"id": "vup-1", "state": "reticulating", "model": "real-esrgan-x4plus:fp16",
          "completed_frames": 0, "total_frames": 0,
          "source": {"kind": "library", "filename": "a.mp4"}}]
        """#
        let jobs = try MoldJSON.decoder.decode([VideoUpscaleJob].self, from: Data(json.utf8))
        #expect(jobs.first?.state == .unknown)
    }

    /// Ported from `studio/lib/upscale.ts:14-26`, in that exact order of
    /// preference: a downloaded real-esrgan-x4plus, then anything downloaded,
    /// then an undownloaded real-esrgan-x4plus, then the first row, then the
    /// manifest name.
    @Test func theDefaultUpscalerPrefersADownloadedRealEsrgan() {
        let choices = [
            UpscalerChoice(name: "swinir:fp16", isDownloaded: true),
            UpscalerChoice(name: "real-esrgan-x4plus:fp16", isDownloaded: true),
        ]
        #expect(UpscalePlan.defaultUpscaler(choices) == "real-esrgan-x4plus:fp16")
        #expect(UpscalePlan.defaultUpscaler([choices[0]]) == "swinir:fp16")
        #expect(UpscalePlan.defaultUpscaler([
            UpscalerChoice(name: "swinir:fp16", isDownloaded: false),
            UpscalerChoice(name: "real-esrgan-x4plus", isDownloaded: false),
        ]) == "real-esrgan-x4plus")
        #expect(UpscalePlan.defaultUpscaler([
            UpscalerChoice(name: "swinir:fp16", isDownloaded: false),
        ]) == "swinir:fp16")
        #expect(UpscalePlan.defaultUpscaler([]) == "real-esrgan-x4plus:fp16")
    }

    /// `real-esrgan-x4plus-anime` is a DIFFERENT model: upstream's regex
    /// anchors the tag boundary at `:` or end of string (`upscale.ts:18`).
    @Test func aNamePrefixIsNotTheDefaultUpscaler() {
        let choices = [UpscalerChoice(name: "real-esrgan-x4plus-anime:fp16", isDownloaded: false),
                       UpscalerChoice(name: "real-esrgan-x4plus:fp16", isDownloaded: false)]
        #expect(UpscalePlan.defaultUpscaler(choices) == "real-esrgan-x4plus:fp16")
    }

    @Test func progressIsAbsentUntilTheHostHasCountedTheFrames() {
        #expect(UpscalePlan.progress(of: job(state: "running", done: 0, total: 0)) == nil)
        #expect(UpscalePlan.progress(of: job(state: "running", done: 31, total: 124)) == 0.25)
        // Clamped both ways: a host that counted a frame twice never reads 101%.
        #expect(UpscalePlan.progress(of: job(state: "running", done: 200, total: 124)) == 1)
    }

    @Test func theStatusSentenceCountsFromOne() {
        #expect(UpscalePlan.status(of: job(state: "queued")) == "Queued")
        #expect(UpscalePlan.status(of: job(state: "running", done: 0, total: 0))
            == "Preparing source video")
        #expect(UpscalePlan.status(of: job(state: "running", done: 0, total: 124))
            == "Upscaling frame 1 of 124")
        // The last frame's completion must not read "frame 125 of 124".
        #expect(UpscalePlan.status(of: job(state: "running", done: 124, total: 124))
            == "Upscaling frame 124 of 124")
        #expect(UpscalePlan.status(of: job(state: "finalizing")) == "Finalizing video")
        #expect(UpscalePlan.status(of: job(state: "paused")) == "Paused — ready to resume")
        #expect(UpscalePlan.status(of: job(state: "completed")) == "Complete")
        #expect(UpscalePlan.status(of: job(state: "cancelled")) == "Cancelled")
    }

    @Test func onlyUnsettledWorkIsPolled() {
        for state in ["queued", "running", "finalizing"] {
            #expect(UpscalePlan.shouldPoll(job(state: state)), "\(state) is still moving")
        }
        for state in ["paused", "completed", "failed", "cancelled", "reticulating"] {
            #expect(!UpscalePlan.shouldPoll(job(state: state)), "\(state) is not")
        }
        #expect(!UpscalePlan.shouldPoll(nil))
    }

    /// `recoverableFramewiseUpscale` (`videoUpscale.ts:82-94`): the same
    /// library filename, and NOT settled. Note what upstream does with a
    /// state it does not know -- it is not one of the three terminal ones, so
    /// it counts as recoverable, and this mirrors that deliberately.
    @Test func recoveryFindsThisPrintsOwnUnsettledJob() {
        let mine = job(state: "running", filename: "clip.mp4")
        let settled = job(state: "completed", filename: "clip.mp4")
        let someoneElses = job(state: "running", filename: "other.mp4")
        #expect(UpscalePlan.recoverable(in: [settled, someoneElses, mine],
                                        filename: "clip.mp4")?.state == .running)
        #expect(UpscalePlan.recoverable(in: [settled, someoneElses], filename: "clip.mp4") == nil)
        #expect(UpscalePlan.recoverable(in: [job(state: "reticulating", filename: "clip.mp4")],
                                        filename: "clip.mp4") != nil)
    }

    /// An upload-sourced job belongs to nobody's library print, so it can
    /// never be recovered onto one however its filename reads.
    @Test func anUploadSourcedJobIsNeverRecoveredOntoAPrint() {
        let json = #"""
        {"id": "vup-up", "state": "running", "model": "m", "completed_frames": 0,
         "total_frames": 9, "source": {"kind": "upload", "handle": "clip.mp4"}}
        """#
        let uploaded = try! MoldJSON.decoder.decode(VideoUpscaleJob.self, from: Data(json.utf8))
        #expect(UpscalePlan.recoverable(in: [uploaded], filename: "clip.mp4") == nil)
    }

    private func job(state: String, done: Int = 0, total: Int = 0,
                     filename: String = "clip.mp4") -> VideoUpscaleJob {
        let json = #"""
        {"id": "vup-\#(state)", "state": "\#(state)", "model": "real-esrgan-x4plus:fp16",
         "completed_frames": \#(done), "total_frames": \#(total),
         "error": "ffprobe is required for Framewise upscale",
         "source": {"kind": "library", "filename": "\#(filename)"}}
        """#
        return try! MoldJSON.decoder.decode(VideoUpscaleJob.self, from: Data(json.utf8))
    }
}
