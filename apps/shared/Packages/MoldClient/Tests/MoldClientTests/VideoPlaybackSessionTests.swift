import AVFoundation
import Foundation
import Testing
@testable import MoldClient

@MainActor struct VideoPlaybackSessionTests {
    func fixture(_ name: String = "endpoint-bframes") -> AVPlayer {
        let file = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/\(name).mp4")
        return AVPlayer(playerItem: AVPlayerItem(url: file))
    }
    func settle(_ predicate: () -> Bool) async throws {
        for _ in 0..<300 { if predicate() { return }; try await Task.sleep(for: .milliseconds(10)) }
        #expect(predicate())
    }
    @Test func naturalEndHoldsExactFinalPresentationFrameAndPlayRestarts() async throws {
        let player = fixture()
        let session = VideoPlaybackSession()
        session.attach(player)
        session.handleEnd(repeating: false)
        try await settle { session.ended && abs(player.currentTime().seconds - 2) < 0.001 }
        #expect(player.rate == 0)
        session.play()
        try await settle { !session.ended && player.rate > 0 && player.currentTime().seconds < 1 }
        session.cancel()
    }
    @Test func audioOverhangIsNotTruncatedAndScrubbingFullEndHoldsVideoFrame() async throws {
        let player = fixture("endpoint-audio-overhang")
        let session = VideoPlaybackSession()
        session.attach(player)
        let duration = try await player.currentItem!.asset.load(.duration)
        #expect(duration.seconds > 2.9)
        #expect(!player.currentItem!.forwardPlaybackEndTime.isNumeric)
        session.seek(to: duration.seconds)
        try await settle { session.ended && abs(player.currentTime().seconds - 2) < 0.001 }
        session.cancel()
    }
    @Test func lateOldItemEndpointCannotSeekOrResumeItsSuccessor() async throws {
        let player = fixture()
        var blocked: CheckedContinuation<CMTime, Never>?
        var calls = 0
        let session = VideoPlaybackSession(resolveFrame: { asset in
            calls += 1
            if calls == 1 { return await withCheckedContinuation { blocked = $0 } }
            return try await VideoPlaybackEndpoint.resolve(in: asset)
        })
        session.attach(player)
        session.handleEnd(repeating: false)
        try await settle { blocked != nil }
        let successor = AVPlayerItem(url: URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/endpoint-single.mp4"))
        player.replaceCurrentItem(with: successor)
        session.attach(player)
        blocked?.resume(returning: CMTime(seconds: 2, preferredTimescale: 4))
        for _ in 0..<10 { await Task.yield() }
        #expect(player.currentItem === successor)
        #expect(!session.ended)
        #expect(player.rate == 0)
        #expect(player.currentTime().seconds < 0.25)
        session.cancel()
    }

    @Test func explicitNativeScrubSupersedesAPendingNaturalEndSeek() async throws {
        let player = fixture()
        var blocked: CheckedContinuation<CMTime, Never>?
        let session = VideoPlaybackSession(resolveFrame: { _ in
            await withCheckedContinuation { blocked = $0 }
        })
        session.attach(player)
        session.handleEnd(repeating: false)
        try await settle { blocked != nil }
        _ = await player.currentItem!.seek(to: CMTime(seconds: 0.5, preferredTimescale: 4), toleranceBefore: .zero, toleranceAfter: .zero)
        session.nativeTimeJumped()
        blocked?.resume(returning: CMTime(seconds: 2, preferredTimescale: 4))
        try await Task.sleep(for: .milliseconds(250))
        #expect(!session.ended)
        #expect(abs(player.currentTime().seconds - 0.5) < 0.001)
        session.cancel()
    }

    @Test func remintedItemKeepsPausedAndEndedIntentAndNewerPauseWins() async throws {
        let player = fixture()
        let session = VideoPlaybackSession()
        session.attach(player)
        session.play()
        let playing = session.captureIntent()
        session.pause()
        let replacement = fixture()
        session.restore(playing, on: replacement, isActive: true)
        #expect(!session.intendsToPlay)
        #expect(replacement.rate == 0)
        session.handleEnd(repeating: false)
        try await settle { session.ended && abs(replacement.currentTime().seconds - 2) < 0.001 }
        let ended = session.captureIntent()
        let final = fixture()
        session.restore(ended, on: final, isActive: true)
        try await settle { session.ended && abs(final.currentTime().seconds - 2) < 0.001 }
        #expect(final.rate == 0)
        session.cancel()
    }

    @Test func internalSeekPauseDoesNotErasePlaybackIntentButInactivityDoes() async throws {
        let player = fixture()
        let session = VideoPlaybackSession()
        session.attach(player)
        session.handleEnd(repeating: false)
        try await settle { session.ended && abs(player.currentTime().seconds - 2) < 0.001 }
        session.play()
        session.nativeRateChanged(0)
        #expect(session.intendsToPlay, "The controller's own zero-seek pause is not a new user pause")
        let beforeBackground = session.captureIntent()
        session.suspend()
        let replacement = fixture()
        session.restore(beforeBackground, on: replacement, isActive: false)
        #expect(!session.intendsToPlay)
        #expect(replacement.rate == 0)
        session.cancel()
    }

    @Test func repeatStartsAtZeroAndReplacementFencesLateEndpointResolution() async throws {
        let player = fixture()
        let session = VideoPlaybackSession()
        session.attach(player)
        session.handleEnd(repeating: true)
        try await settle { player.rate > 0 && player.currentTime().seconds < 1 }
        #expect(!session.ended)
        let successor = AVPlayerItem(url: URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/endpoint-single.mp4"))
        player.replaceCurrentItem(with: successor)
        session.attach(player)
        session.cancel()
        try await Task.sleep(for: .milliseconds(30))
        #expect(player.currentItem === successor)
        #expect(player.rate == 0)
        #expect(!session.ended)
    }
}
