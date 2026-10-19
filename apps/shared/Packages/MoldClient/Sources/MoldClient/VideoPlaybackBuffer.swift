import AVFoundation

/// AVFoundation may report successful preroll while a custom loader is still
/// empty. Resume only with contiguous, actually loaded media ahead of the head.
enum VideoPlaybackBuffer {
    static func canResume(at time: CMTime, duration: CMTime, ranges: [CMTimeRange], isEmpty: Bool) -> Bool {
        guard !isEmpty, time.isNumeric else { return false }
        let remaining = duration.isNumeric ? max(0, duration.seconds - time.seconds) : Double.infinity
        guard remaining > 0 else { return false }
        let required = min(0.25, remaining)
        return ranges.contains { range in
            range.start.isNumeric && range.duration.isNumeric && range.start <= time
                && range.end.seconds - time.seconds >= required - 0.001
        }
    }
}
