import Foundation
import Observation

/// Thumbnail headers supply source ratios when the server has no dimension fields.
/// Batch discoveries; keep cheap geometry after decoded pixels are evicted.
@MainActor @Observable public final class SourceMediaAspects {
    private struct Measurement: Equatable {
        let key: SourcePreviewKey
        let aspect: Double
    }
    @ObservationIgnored private var measured: [String: Measurement] = [:]
    @ObservationIgnored private var pending: [String: Measurement] = [:]
    @ObservationIgnored private var delivery: Task<Void, Never>?
    public private(set) var revision = 0
    public init() {}

    public func aspect(for id: String, key: SourcePreviewKey) -> Double {
        _ = revision
        guard let hit = measured[id], matches(hit.key, key, id: id) else { return 1 }
        return hit.aspect
    }
    public func record(id: String, key: SourcePreviewKey, aspect: Double) {
        guard aspect.isFinite, aspect > 0 else { return }
        let next = Measurement(key: key, aspect: aspect)
        if let hit = measured[id], hit.aspect == aspect, matches(hit.key, key, id: id) { return }
        pending[id] = next
        guard delivery == nil else { return }
        delivery = Task { [weak self] in
            do { try await Task.sleep(for: .milliseconds(50)) } catch { return }
            self?.flush()
        }
    }
    private func matches(_ first: SourcePreviewKey, _ second: SourcePreviewKey, id: String) -> Bool {
        if first == second { return true }
        // A verified reusable-byte identity has the same oriented geometry on
        // every authorized copy, even when the preferred host changes.
        guard let digest = first.member.contentSha256, digest.count == 64,
              digest.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
              second.member.contentSha256 == digest, id == "sha256/" + digest else { return false }
        return true
    }
    public func flush() {
        delivery?.cancel(); delivery = nil
        guard !pending.isEmpty else { return }
        measured.merge(pending) { _, new in new }
        pending.removeAll(keepingCapacity: true)
        revision += 1
    }
    deinit { delivery?.cancel() }
}
