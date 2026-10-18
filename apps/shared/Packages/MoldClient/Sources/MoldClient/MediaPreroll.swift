import AVFoundation
import Foundation

/// A bounded, item-owned native preroll. Cancelling an old preparation must never
/// cancel a successor item's preroll on the same reusable player.
@MainActor enum MediaPreroll {
    static func prepare(_ player: AVPlayer, item: AVPlayerItem, timeout: Duration) async throws {
        let deadline = ContinuousClock.now + timeout
        while player.status != .readyToPlay {
            try check(player, item: item, deadline: deadline)
            try await Task.sleep(for: .milliseconds(20))
        }
        try check(player, item: item, deadline: deadline)
        // AVPlayer.h permits preroll only at rate zero. An already playing item
        // has crossed this boundary through its native transport controls.
        guard player.rate == 0 else { return }
        let request = MediaPrerollRequest(player: player, item: item)
        let monitor = Task { @MainActor in
            while !Task.isCancelled && !request.finished {
                if player.currentItem !== item || item.error != nil || player.error != nil || ContinuousClock.now >= deadline {
                    request.cancel(); return
                }
                do { try await Task.sleep(for: .milliseconds(20)) } catch { return }
            }
        }
        defer { monitor.cancel() }
        try await withTaskCancellationHandler {
            try Task.checkCancellation()
            let primed = await request.start()
            try check(player, item: item, deadline: deadline)
            guard primed else { throw CancellationError() }
        } onCancel: {
            Task { @MainActor in request.cancel() }
        }
    }

    private static func check(_ player: AVPlayer, item: AVPlayerItem, deadline: ContinuousClock.Instant) throws {
        try Task.checkCancellation()
        guard player.currentItem === item else { throw CancellationError() }
        if let error = item.error ?? player.error { throw error }
        guard ContinuousClock.now < deadline else { throw URLError(.timedOut) }
    }
}

@MainActor private final class MediaPrerollRequest {
    private let player: AVPlayer
    private let item: AVPlayerItem
    private var continuation: CheckedContinuation<Bool, Never>?
    private var started = false
    private(set) var finished = false
    init(player: AVPlayer, item: AVPlayerItem) { self.player = player; self.item = item }
    func start() async -> Bool {
        await withCheckedContinuation { continuation in
            guard !finished else { continuation.resume(returning: false); return }
            self.continuation = continuation
            started = true
            player.preroll(atRate: 1) { [weak self] value in
                Task { @MainActor in self?.finish(value) }
            }
        }
    }
    func cancel() {
        guard !finished else { return }
        finish(false)
        if started, player.currentItem === item { player.cancelPendingPrerolls() }
    }
    private func finish(_ value: Bool) {
        guard !finished else { return }
        finished = true
        continuation?.resume(returning: value); continuation = nil
    }
}
