import AVFoundation
import Foundation
import Observation

/// Item-bound transport state. The displayed final frame and the logical ended state
/// are separate: holding a presentation sample must still make Play restart at zero.
@MainActor @Observable public final class VideoPlaybackSession {
    public private(set) var ended = false
    public private(set) var intendsToPlay = false
    @ObservationIgnored private var intentRevision: UInt64 = 0
    public struct Intent: Equatable, Sendable {
        let revision: UInt64
        public let wasPlaying: Bool
        public let wasEnded: Bool
    }
    public func captureIntent() -> Intent { .init(revision: intentRevision, wasPlaying: intendsToPlay, wasEnded: ended) }
    public func restore(_ captured: Intent, on player: AVPlayer, isActive: Bool) {
        let chosen = intentRevision == captured.revision ? captured : captureIntent()
        attach(player)
        ended = chosen.wasEnded
        guard isActive else { suspend(); return }
        if chosen.wasEnded { handleEnd(repeating: false) }
        else if chosen.wasPlaying { play() }
    }
    private func setIntent(_ playing: Bool, explicit: Bool = false) {
        if intendsToPlay != playing || explicit { intentRevision &+= 1 }
        intendsToPlay = playing
    }
    public func nativeRateChanged(_ rate: Float) {
        guard let player, let item, player.currentItem === item, player.rate == rate else { return }
        if rate > 0 {
            if ended { play() } else { setIntent(true) }
        } else if pendingDestination == nil && item.status != .failed { setIntent(false) }
    }
    @ObservationIgnored private var player: AVPlayer?
    @ObservationIgnored private var item: AVPlayerItem?
    @ObservationIgnored private var endpoint: CMTime?
    @ObservationIgnored private var epoch = UUID()
    @ObservationIgnored private var seekRevision = 0
    @ObservationIgnored private var pendingDestination: Destination?
    @ObservationIgnored private var frameTask: Task<(CMTime, CMTime)?, Never>?
    @ObservationIgnored private var seekTask: Task<Void, Never>?

    @ObservationIgnored private let resolveFrame: @MainActor (AVAsset) async throws -> CMTime
    public init(resolveFrame: @escaping @MainActor (AVAsset) async throws -> CMTime = { try await VideoPlaybackEndpoint.resolve(in: $0) }) {
        self.resolveFrame = resolveFrame
    }

    public func attach(_ player: AVPlayer) {
        if self.player === player, item === player.currentItem {
            if frameTask?.isCancelled == true, let item { prepareFrame(item.asset) }
            return
        }
        cancel()
        self.player = player; item = player.currentItem
        guard let item else { return }
        prepareFrame(item.asset)
    }

    private func prepareFrame(_ asset: AVAsset) {
        let resolveFrame = resolveFrame
        frameTask = Task {
            do {
                let duration = try await asset.load(.duration)
                let final = try await resolveFrame(asset)
                try Task.checkCancellation()
                return (final, duration)
            } catch { return nil }
        }
    }

    public func handleEnd(repeating: Bool) {
        setIntent(repeating)
        if repeating { transport(.zero, resume: true) }
        else { ended = true; transport(.end, resume: false) }
    }

    public func play() {
        setIntent(true, explicit: true)
        if ended { transport(.zero, resume: true) }
        else { player?.play() }
    }

    public func seek(to seconds: Double) {
        guard seconds.isFinite, seconds >= 0 else { return }
        transport(.position(seconds), resume: (player?.rate ?? 0) > 0)
    }

    /// AVKit's native transport can seek or play independently of the custom Mac bar.
    public func nativeTimeJumped() {
        guard let player, let item, player.currentItem === item else { return }
        let current = player.currentTime(), duration = item.duration
        guard current.isNumeric, duration.isNumeric, duration > .zero else { return }
        if ended, let endpoint, CMTimeCompare(current, endpoint) == 0 { return }
        if current >= duration {
            // Natural end and AVKit's time-jump notification may arrive together.
            // A queued duplicate must not replace an already requested replay.
            if let pendingDestination {
                switch pendingDestination { case .zero, .end: return; case .position: break }
            }
            seek(to: duration.seconds)
        } else {
            if case .zero? = pendingDestination, current == .zero { return }
            // A new native scrub outranks an endpoint lookup that has not finished yet.
            seekRevision += 1
            if seekTask != nil { seekTask?.cancel(); item.cancelPendingSeeks() }
            seekTask = nil; pendingDestination = nil; ended = false
        }
    }

    public func pause() {
        setIntent(false, explicit: true)
        seekRevision += 1
        seekTask?.cancel(); pendingDestination = nil; item?.cancelPendingSeeks()
        if let item, player?.currentItem === item { player?.pause() }
    }

    public func suspend() {
        pause(); frameTask?.cancel()
    }

    public func cancel() {
        setIntent(false, explicit: true)
        epoch = UUID(); seekRevision += 1
        frameTask?.cancel(); frameTask = nil
        seekTask?.cancel(); seekTask = nil
        item?.cancelPendingSeeks()
        if let item, player?.currentItem === item { player?.pause() }
        item = nil; player = nil; endpoint = nil; pendingDestination = nil; ended = false
    }

    private enum Destination { case zero, end, position(Double) }
    private func transport(_ destination: Destination, resume: Bool) {
        guard let player, let item, player.currentItem === item else { return }
        seekRevision += 1
        let revision = seekRevision, capturedEpoch = epoch
        seekTask?.cancel(); item.cancelPendingSeeks()
        pendingDestination = destination
        // Never let a native Play request run the held tail audio while the zero seek waits.
        player.pause()
        let frameTask = frameTask
        seekTask = Task { [weak self] in
            let facts = await frameTask?.value
            guard let self, !Task.isCancelled, self.epoch == capturedEpoch,
                  self.seekRevision == revision, player.currentItem === item else { return }
            defer {
                if self.epoch == capturedEpoch && self.seekRevision == revision {
                    self.pendingDestination = nil; self.seekTask = nil
                }
            }
            self.endpoint = facts?.0
            let target: CMTime
            switch destination {
            case .zero: self.ended = false; target = .zero
            case .end:
                guard let final = facts?.0 else { return }
                self.ended = true; target = final
            case let .position(seconds):
                let duration = item.duration.isNumeric ? item.duration : (facts?.1 ?? .invalid)
                if duration.isNumeric, duration > .zero, seconds >= duration.seconds {
                    guard let final = facts?.0 else { return }
                    self.ended = true; target = final
                } else {
                    self.ended = false
                    target = CMTime(seconds: seconds, preferredTimescale: 60000)
                }
            }
            let finished = await item.seek(to: target, toleranceBefore: .zero, toleranceAfter: .zero)
            guard finished, !Task.isCancelled, self.epoch == capturedEpoch,
                  self.seekRevision == revision, player.currentItem === item else { return }
            if resume && !self.ended { player.play() }
        }
    }
}
