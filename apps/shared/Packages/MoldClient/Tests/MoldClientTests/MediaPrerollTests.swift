import AVFoundation
import Foundation
import Synchronization
import Testing
@testable import MoldClient

private final class PrimingItem: AVPlayerItem, @unchecked Sendable {
    override var status: AVPlayerItem.Status { .readyToPlay }
    override var loadedTimeRanges: [NSValue] {
        [NSValue(timeRange: CMTimeRange(start: .zero, duration: CMTime(seconds: 0.512, preferredTimescale: 1000)))]
    }
}
private final class PrimingPlayer: AVPlayer, @unchecked Sendable {
    struct State {
        var starts = 0
        var cancellations = 0
        var automatic = false
        var completion: (@Sendable (Bool) -> Void)?
    }
    let priming = Mutex(State())
    override var status: AVPlayer.Status { .readyToPlay }
    override func preroll(atRate rate: Float, completionHandler: (@Sendable (Bool) -> Void)? = nil) {
        let automatic = priming.withLock { state in
            state.starts += 1; state.completion = completionHandler; return state.automatic
        }
        if automatic { completionHandler?(true) }
    }
    override func cancelPendingPrerolls() {
        let completion = priming.withLock { state in state.cancellations += 1; return state.completion }
        completion?(false)
    }
}

@MainActor struct MediaPrerollTests {
    private func fixture(automatic: Bool = false) -> (PlayableMedia, PrimingPlayer) {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appending(path: "Fixtures/endpoint-audio-overhang.mp4")
        let player = PrimingPlayer(playerItem: PrimingItem(url: url))
        player.priming.withLock { $0.automatic = automatic }
        let backend = HTTPBackend(host: MoldHost(name: "fixture", baseURL: URL(string: "http://127.0.0.1:1")!))
        return (PlayableMedia(backend: backend, url: url), player)
    }
    private func started(_ player: PrimingPlayer) async throws {
        for _ in 0..<100 {
            if player.priming.withLock({ $0.starts }) > 0 { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        Issue.record("Ready player never began native preroll")
    }
    @Test func readyHalfSecondBufferCanPrimeWithoutStartingPlayback() async throws {
        let (media, player) = fixture(automatic: true)
        try await media.prepare(player, timeout: .seconds(2))
        #expect(player.priming.withLock { $0.starts } == 1)
        #expect(player.rate == 0)
        #expect(player.currentTime() == .zero)
        #expect(CMTimeRangeGetEnd(player.currentItem!.loadedTimeRanges[0].timeRangeValue).seconds < 1)
    }
    @Test func cancellationReleasesPendingPrerollAndNeverStartsPlayback() async throws {
        let (media, player) = fixture()
        let work = Task { try await media.prepare(player, timeout: .seconds(2)) }
        try await started(player); work.cancel()
        await #expect(throws: CancellationError.self) { try await work.value }
        #expect(player.priming.withLock { $0.cancellations } == 1)
        #expect(player.rate == 0)
    }
    @Test func timeoutCancelsNativePreroll() async throws {
        let (media, player) = fixture()
        do { try await media.prepare(player, timeout: .milliseconds(100)); Issue.record("Unfinished preroll succeeded") }
        catch let error as URLError { #expect(error.code == .timedOut) }
        #expect(player.priming.withLock { $0.cancellations } == 1)
    }
    @Test func replacementAndLateCallbackCannotPrimeOrCancelTheSuccessor() async throws {
        let (media, player) = fixture()
        let work = Task { try await media.prepare(player, timeout: .seconds(2)) }
        try await started(player)
        player.replaceCurrentItem(with: AVPlayerItem(url: URL(fileURLWithPath: "/not-read-successor.mp4")))
        await #expect(throws: CancellationError.self) { try await work.value }
        player.priming.withLock { $0.completion }?(true)
        #expect(player.priming.withLock { $0.cancellations } == 0)
        #expect(player.rate == 0)
    }
}
