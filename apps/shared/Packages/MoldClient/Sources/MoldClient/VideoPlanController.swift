import Foundation
import Observation

/// Automatic display-only planning. The native adapter's current predicate
/// captures route, installation, credential revision and scalar draft identity.
/// This never prepares media, uploads inputs, or creates work.
@MainActor @Observable
public final class VideoPlanController {
    public private(set) var plan: VideoJobPlan?
    public private(set) var error: String?
    public private(set) var isCalculating = false
    @ObservationIgnored private let debounceNanoseconds: UInt64
    @ObservationIgnored private var pending: Task<Void, Never>?
    @ObservationIgnored private var token = UUID()

    public init(debounceNanoseconds: UInt64 = 250_000_000) { self.debounceNanoseconds = debounceNanoseconds }

    @discardableResult
    public func update(_ request: VideoJobRequest, backend: any MoldVideoBackend,
                       destinationInstanceId: String? = nil, current: @escaping @MainActor () -> Bool) -> Task<Void, Never> {
        let backend: any MoldVideoBackend = if let http = backend as? HTTPBackend, let destinationInstanceId {
            http.pinningVideoDestination(destinationInstanceId)
        } else { backend }
        clear()
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
                self.plan = plan
            } catch {
                guard let self, mine == self.token else { return }
                self.isCalculating = false
                if !Task.isCancelled, current() { self.error = UserFacingError.describe(error) }
            }
        }
        pending = work
        return work
    }

    public func clear() {
        token = UUID(); pending?.cancel(); pending = nil
        plan = nil; error = nil; isCalculating = false
    }

    func waitForPendingPlan() async { await pending?.value }
}
