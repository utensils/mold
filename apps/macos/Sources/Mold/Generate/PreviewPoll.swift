import Foundation
import MoldClient

/// The denoise preview and the step counter, which the batch events stream
/// deliberately does not carry. Its own type because `GenerateController` is
/// over its budget and a poll that owns a cadence is a whole concern.
@MainActor
enum PreviewPoll {
    /// Step progress and the denoise preview, which the events stream
    /// deliberately does not carry.
    ///
    /// With several children, one settles while the others keep running --
    /// so this re-reads the current run's status every tick and follows
    /// whichever child is still live, falling back to the last one it had
    /// rather than going quiet the moment the first child finishes.
    static func follow(_ status: BatchStatus, backend: any MoldBackend,
                       of controller: GenerateController) -> Task<Void, Never> {
        var target = status.children.first?.jobId
        return Task { [weak controller] in
            while !Task.isCancelled {
                if case let .running(current, _) = controller?.run {
                    target = current.children.first { $0.state.isLive }?.jobId ?? target
                }
                if let jobId = target,
                   let progress = try? await backend.jobPreview(jobId: jobId),
                   case let .running(current, _) = controller?.run {
                    controller?.run = .running(current, progress)
                }
                try? await Task.sleep(for: .milliseconds(700))
            }
        }
    }
}
