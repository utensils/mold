import Foundation
import Observation

/// Automatic display-only planning. The native adapter's current predicate
/// captures route, installation, credential revision and scalar draft identity.
/// This never prepares media, uploads inputs, or creates work.
@MainActor @Observable
public final class VideoPlanController {
    public private(set) var plan: VideoJobPlan?
    /// What the editor draws: the last answer for this owner and model, kept
    /// while the next length is calculated so a slider drag never blanks the
    /// clip line or the Length bounds between answers. Submission checks read
    /// `plan`, which is only ever the answer for the current draft.
    public private(set) var shownPlan: VideoJobPlan?
    public private(set) var error: String?
    public private(set) var isCalculating = false
    @ObservationIgnored private let debounceNanoseconds: UInt64
    @ObservationIgnored private var pending: Task<Void, Never>?
    @ObservationIgnored private var token = UUID()
    /// What `shownPlan` answered for. Only the length may change under it.
    @ObservationIgnored private var shownScope: Scope?
    private struct Scope: Equatable {
        let host: UUID?, destination: String?, model: String, fps: Int?, source: VideoJobSource?
    }

    public init(debounceNanoseconds: UInt64 = 250_000_000) { self.debounceNanoseconds = debounceNanoseconds }

    @discardableResult
    public func update(_ request: VideoJobRequest, backend: any MoldVideoBackend,
                       destinationInstanceId: String? = nil, current: @escaping @MainActor () -> Bool) -> Task<Void, Never> {
        let backend: any MoldVideoBackend = if let http = backend as? HTTPBackend, let destinationInstanceId {
            http.pinningVideoDestination(destinationInstanceId)
        } else { backend }
        reset()
        let scope = Scope(host: (backend as? any MoldBackend)?.host.id, destination: destinationInstanceId,
                          model: request.generation.model, fps: request.generation.fps, source: request.source)
        if scope != shownScope { shownPlan = nil; shownScope = scope }
        isCalculating = true
        let mine = token, delay = debounceNanoseconds, scalars = request.planningProjection
        let work = Task { [weak self] in
            do {
                if delay > 0 { try await Task.sleep(nanoseconds: delay) }
                try Task.checkCancellation()
                guard let self, mine == self.token else { return }
                defer { if mine == self.token { self.isCalculating = false } }
                guard current() else { return }
                let plan = try await backend.planVideoJob(scalars)
                guard mine == self.token, !Task.isCancelled, current() else { return }
                guard plan.validationRevision.isEmpty else { throw MoldClientError.malformedResponse }
                self.plan = plan; self.shownPlan = plan
            } catch {
                guard let self, mine == self.token else { return }
                self.isCalculating = false
                if !Task.isCancelled, current() { self.error = UserFacingError.describe(error); self.shownPlan = nil }
            }
        }
        pending = work
        return work
    }

    public func clear() {
        reset(); shownPlan = nil; shownScope = nil
    }

    private func reset() {
        token = UUID(); pending?.cancel(); pending = nil
        plan = nil; error = nil; isCalculating = false
    }

    func waitForPendingPlan() async { await pending?.value }
}
