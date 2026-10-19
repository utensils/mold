import AVFoundation
import Combine
import Foundation
import Observation

/// Item-bound transport state. The displayed final frame and the logical ended state
/// are separate: holding a presentation sample must still make Play restart at zero.
@MainActor @Observable public final class VideoPlaybackSession {
    public private(set) var ended = false
    public private(set) var intendsToPlay = false
    public private(set) var buffering = false
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
        } else if pendingDestination == nil && item.status != .failed {
            if !player.automaticallyWaitsToMinimizeStalling && intendsToPlay && !ended,
               !item.duration.isNumeric || player.currentTime() < item.duration {
                recoverStall()
            } else if recoveryTask == nil { setIntent(false) }
        }
    }
    @ObservationIgnored private var player: AVPlayer?
    @ObservationIgnored private var item: AVPlayerItem?
    @ObservationIgnored private var endpoint: CMTime?
    @ObservationIgnored private var epoch = UUID()
    @ObservationIgnored private var seekRevision = 0
    @ObservationIgnored private var pendingDestination: Destination?
    @ObservationIgnored private var frameTask: Task<(CMTime, CMTime)?, Never>?
    @ObservationIgnored private var observers: [AnyCancellable] = []
    @ObservationIgnored private var recoveryRevision = UUID()
    @ObservationIgnored private var recoveryTask: Task<Void, Never>?
    @ObservationIgnored private var seekTask: Task<Void, Never>?

    @ObservationIgnored private let resolveFrame: @MainActor (AVAsset) async throws -> CMTime
    public init(resolveFrame: @escaping @MainActor (AVAsset) async throws -> CMTime = { try await VideoPlaybackEndpoint.resolve(in: $0) }) {
        self.resolveFrame = resolveFrame
    }

    public func attach(_ player: AVPlayer) {
        if self.player === player, item === player.currentItem {
            return
        }
        cancel()
        self.player = player; item = player.currentItem
        guard let item else { return }
        observeStream(player, item: item)
    }

    private func prepareFrame(_ asset: AVAsset) {
        guard frameTask == nil || frameTask?.isCancelled == true else { return }
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
        stopRecovery()
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
        transport(.position(seconds), resume: intendsToPlay)
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
            if case let .position(seconds)? = pendingDestination, abs(current.seconds - seconds) < 0.001 { return }
            // A new native scrub outranks an endpoint lookup that has not finished yet.
            seekRevision += 1
            if seekTask != nil { seekTask?.cancel(); item.cancelPendingSeeks() }
            seekTask = nil; pendingDestination = nil; ended = false
        }
    }

    public func pause() {
        stopRecovery()
        setIntent(false, explicit: true)
        seekRevision += 1
        seekTask?.cancel(); pendingDestination = nil; item?.cancelPendingSeeks()
        if let item, player?.currentItem === item { player?.pause() }
    }

    public func suspend() {
        pause(); frameTask?.cancel()
    }

    public func cancel() {
        observers.removeAll()
        stopRecovery()
        setIntent(false, explicit: true)
        epoch = UUID(); seekRevision += 1
        frameTask?.cancel(); frameTask = nil
        seekTask?.cancel(); seekTask = nil
        item?.cancelPendingSeeks()
        if let item, player?.currentItem === item { player?.pause() }
        item = nil; player = nil; endpoint = nil; pendingDestination = nil; ended = false
    }

    /// With a resource loader AVFoundation requires automatic stall waiting off.
    /// A buffer underrun therefore pauses the player; it is not a user Pause.
    private func observeStream(_ player: AVPlayer, item: AVPlayerItem) {
        let center = NotificationCenter.default
        observers = [
            center.publisher(for: .AVPlayerItemPlaybackStalled, object: item).receive(on: DispatchQueue.main).sink { [weak self] _ in
                MainActor.assumeIsolated { self?.recoverStall() }
            },
            center.publisher(for: AVPlayer.rateDidChangeNotification, object: player).sink { [weak self] notification in
                // Capture the event's rate before hopping executors. An earlier
                // Play notification must not look like Pause after a later stall.
                let rate = player.rate
                let reason = notification.userInfo?[AVPlayer.rateDidChangeReasonKey] as? AVPlayer.RateDidChangeReason
                if Thread.isMainThread {
                    MainActor.assumeIsolated { self?.transportRateChanged(rate, reason: reason, player: player, item: item) }
                } else {
                    Task { @MainActor in self?.transportRateChanged(rate, reason: reason, player: player, item: item) }
                }
            },
        ]
    }

    private func transportRateChanged(_ rate: Float, reason: AVPlayer.RateDidChangeReason?, player: AVPlayer, item: AVPlayerItem) {
        guard self.player === player, self.item === item, player.currentItem === item,
              rate == 0, pendingDestination == nil else { return }
        if reason == .setRateCalled || reason == .audioSessionInterrupted || reason == .appBackgrounded {
            stopRecovery()
            setIntent(false, explicit: true)
        }
    }

    private func stopRecovery() {
        recoveryRevision = UUID()
        recoveryTask?.cancel(); recoveryTask = nil; buffering = false
    }

    private func recoverStall() {
        guard recoveryTask == nil, intendsToPlay, !ended, pendingDestination == nil,
              let player, let item, player.currentItem === item,
              !player.automaticallyWaitsToMinimizeStalling, item.status != .failed else { return }
        let capturedEpoch = epoch, recovery = UUID()
        recoveryRevision = recovery
        buffering = true
        recoveryTask = Task { [weak self] in
            defer {
                if let self, self.recoveryRevision == recovery {
                    self.recoveryTask = nil; self.buffering = false
                }
            }
            // A long outage may outlive a preroll timeout. Keep waiting while
            // this item's Play intent is current; Pause/replacement cancels it.
            while !Task.isCancelled {
                do {
                    try await Task.sleep(for: .milliseconds(200))
                    guard let self, self.epoch == capturedEpoch, self.recoveryRevision == recovery,
                          self.intendsToPlay, !self.ended, self.pendingDestination == nil,
                          player.currentItem === item, item.status != .failed else { return }
                    guard VideoPlaybackBuffer.canResume(at: player.currentTime(), duration: item.duration,
                        ranges: item.loadedTimeRanges.map(\.timeRangeValue), isEmpty: item.isPlaybackBufferEmpty) else { continue }
                    if player.rate == 0 {
                        try await MediaPreroll.prepare(player, item: item, timeout: .seconds(30))
                    }
                    try Task.checkCancellation()
                    guard self.epoch == capturedEpoch, self.recoveryRevision == recovery,
                          self.intendsToPlay, !self.ended, self.pendingDestination == nil,
                          player.currentItem === item else { return }
                    guard VideoPlaybackBuffer.canResume(at: player.currentTime(), duration: item.duration,
                        ranges: item.loadedTimeRanges.map(\.timeRangeValue), isEmpty: item.isPlaybackBufferEmpty) else { continue }
                    player.play()
                    return
                } catch is CancellationError { return }
                catch {
                    guard let self, self.epoch == capturedEpoch, self.recoveryRevision == recovery,
                          self.intendsToPlay, player.currentItem === item, item.status != .failed else { return }
                }
            }
        }
    }

    private enum Destination { case zero, end, position(Double) }
    private func transport(_ destination: Destination, resume: Bool) {
        guard let player, let item, player.currentItem === item else { return }
        seekRevision += 1
        let revision = seekRevision, capturedEpoch = epoch
        seekTask?.cancel(); item.cancelPendingSeeks()
        stopRecovery()
        pendingDestination = destination
        // Never let a native Play request run the held tail audio while the zero seek waits.
        player.pause()
        // Ordinary playback/replay never decodes the tail while streaming.
        if case .end = destination { prepareFrame(item.asset) }
        seekTask = Task { [weak self] in
            guard let self else { return }
            let duration: CMTime
            switch destination {
            case .position:
                duration = item.duration.isNumeric ? item.duration : ((try? await item.asset.load(.duration)) ?? .invalid)
            default: duration = .invalid
            }
            let needsEndpoint: Bool
            switch destination {
            case .end: needsEndpoint = true
            case let .position(seconds): needsEndpoint = duration.isNumeric && duration > .zero && seconds >= duration.seconds
            case .zero: needsEndpoint = false
            }
            var facts: (CMTime, CMTime)?
            if needsEndpoint {
                self.prepareFrame(item.asset)
                facts = await self.frameTask?.value
            }
            guard !Task.isCancelled, self.epoch == capturedEpoch,
                  self.seekRevision == revision, player.currentItem === item else { return }
            defer {
                if self.epoch == capturedEpoch && self.seekRevision == revision {
                    self.pendingDestination = nil; self.seekTask = nil
                }
            }
            if let facts { self.endpoint = facts.0 }
            let target: CMTime
            switch destination {
            case .zero: self.ended = false; target = .zero
            case .end:
                guard let final = facts?.0 else { return }
                self.ended = true; target = final
            case let .position(seconds):
                if duration.isNumeric, duration > .zero, seconds >= duration.seconds {
                    guard let final = facts?.0 else { return }
                    self.ended = true; self.setIntent(false); target = final
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
