import Foundation

/// One upscaler a machine could run, as the picker needs to see it.
public struct UpscalerChoice: Hashable, Sendable {
    public let name: String
    public let isDownloaded: Bool

    public init(name: String, isDownloaded: Bool) {
        self.name = name
        self.isDownloaded = isDownloaded
    }
}

/// Which upscaler to use, how far a clip job has got, and whether to keep
/// asking -- the port of `studio/lib/upscale.ts`, which web and desktop both
/// read. Pure, so the wording and the arithmetic are a test rather than
/// something you watch a progress bar for.
public enum UpscalePlan {

    /// `real-esrgan-x4plus` at any tag, DOWNLOADED, is the answer wherever
    /// the machine has it; then anything downloaded; then the same model
    /// undownloaded; then whatever is first; then the manifest name, which is
    /// what a host with no upscaler listing at all gets asked for
    /// (`upscale.ts:14-26`).
    public static func defaultUpscaler(_ choices: [UpscalerChoice]) -> String {
        choices.first { $0.isDownloaded && isRealEsrganX4Plus($0.name) }?.name
            ?? choices.first(where: \.isDownloaded)?.name
            ?? choices.first { isRealEsrganX4Plus($0.name) }?.name
            ?? choices.first?.name
            ?? "real-esrgan-x4plus:fp16"
    }

    /// `/^real-esrgan-x4plus(?::|$)/` (`upscale.ts:18`). The boundary matters:
    /// `real-esrgan-x4plus-anime` is a different model, not a tag of this one.
    private static func isRealEsrganX4Plus(_ name: String) -> Bool {
        name == "real-esrgan-x4plus" || name.hasPrefix("real-esrgan-x4plus:")
    }

    /// 0...1, or nil while the host has not counted the frames yet
    /// (`upscale.ts:28-31`). Clamped both ways.
    public static func progress(of job: VideoUpscaleJob) -> Double? {
        guard job.totalFrames > 0 else { return nil }
        return min(1, max(0, Double(job.completedFrames) / Double(job.totalFrames)))
    }

    /// The one sentence shown beside the job (`upscale.ts:33-52`). A failure
    /// is the HOST's sentence, never this app's summary of it.
    public static func status(of job: VideoUpscaleJob) -> String {
        switch job.state {
        case .queued: "Queued"
        case .running where job.totalFrames > 0:
            "Upscaling frame \(min(job.completedFrames + 1, job.totalFrames)) of \(job.totalFrames)"
        case .running: "Preparing source video"
        case .finalizing: "Finalizing video"
        case .paused: "Paused — ready to resume"
        case .completed: "Complete"
        case .failed: job.error.flatMap { $0.isEmpty ? nil : $0 } ?? "Framewise upscale failed"
        case .cancelled: "Cancelled"
        // A state this build has never heard of. It is not settled, so say
        // the true thing -- something is happening -- rather than inventing a
        // lifecycle for it.
        case .unknown: "Working"
        }
    }

    /// Whether this job is still moving and should be asked about again
    /// (`upscale.ts:54-56`). A paused job is NOT polled: it moves when
    /// somebody resumes it, and that reply is the next answer.
    public static func shouldPoll(_ job: VideoUpscaleJob?) -> Bool {
        switch job?.state {
        case .queued, .running, .finalizing: true
        default: false
        }
    }

    /// The unsettled job already upscaling this exact print, if the host has
    /// one (`videoUpscale.ts:82-94`). Opening the Library after a restart, or
    /// on a second Mac, must find the job rather than offering to start a
    /// duplicate.
    ///
    /// Mirrors upstream exactly, INCLUDING what it does with a state neither
    /// build knows: it is not one of the three terminal ones, so it counts as
    /// recoverable. `shouldPoll` then declines to chase it, which is the
    /// honest pair -- the row is shown, and this app does not pretend to know
    /// where it is going.
    public static func recoverable(
        in jobs: [VideoUpscaleJob], filename: String
    ) -> VideoUpscaleJob? {
        jobs.first { $0.libraryFilename == filename && !$0.state.isTerminal }
    }
}
