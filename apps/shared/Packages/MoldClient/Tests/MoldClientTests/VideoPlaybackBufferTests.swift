import AVFoundation
import Testing
@testable import MoldClient

struct VideoPlaybackBufferTests {
    private func time(_ value: Double) -> CMTime { CMTime(seconds: value, preferredTimescale: 60_000) }
    @Test func readinessRequiresRealContiguousHeadroom() {
        let ranges = [CMTimeRange(start: .zero, duration: time(0.54))]
        #expect(!VideoPlaybackBuffer.canResume(at: time(0.5), duration: time(3), ranges: ranges, isEmpty: true))
        #expect(!VideoPlaybackBuffer.canResume(at: time(0.5), duration: time(3), ranges: ranges, isEmpty: false))
        #expect(VideoPlaybackBuffer.canResume(at: .zero, duration: time(3), ranges: ranges, isEmpty: false))
        #expect(!VideoPlaybackBuffer.canResume(at: time(0.5), duration: time(3),
            ranges: [CMTimeRange(start: time(1), duration: time(1))], isEmpty: false))
    }
    @Test func remainingTailCanBeShorterThanTheHeadroomTarget() {
        #expect(VideoPlaybackBuffer.canResume(at: time(2.9), duration: time(3),
            ranges: [CMTimeRange(start: time(2.8), duration: time(0.2))], isEmpty: false))
        #expect(!VideoPlaybackBuffer.canResume(at: time(3), duration: time(3),
            ranges: [CMTimeRange(start: .zero, duration: time(3))], isEmpty: false))
    }
}
