import Foundation
import MoldClient

/// Asking a machine where a render would run and roughly how long it would
/// take.
///
/// Read-only -- it reserves nothing -- and debounced, because it fires on
/// every draft change and a slider produces a great many of those. Its own
/// type rather than three more fields on `GenerateController`: the controller
/// is over the type-size budget, and a probe that owns a task, a debounce and
/// two results is a whole concern.
///
/// The request it sends is REDACTED (`GenerateRequest.redactedForPlacement`).
@MainActor
@Observable
final class PlacementProbe {
    private(set) var placement: PlacementPreview?
    private(set) var error: String?

    @ObservationIgnored private var task: Task<Void, Never>?
    /// A constructor parameter rather than a constant, so a test pins the
    /// behaviour without sleeping through it.
    @ObservationIgnored private let debounce: Duration

    init(debounce: Duration = .milliseconds(350)) {
        self.debounce = debounce
    }

    func refresh(draft: RenderDraft, model: String?, on host: MoldHost, hosts: HostStore) {
        task?.cancel()
        guard let model else { return }
        let request = draft.placementRequest(
            model: model, maxIdentityPhotos: hosts.capabilities(of: host)?.maxIdentityPhotos ?? 0
        )
        let copies = draft.batchSize
        let client = hosts.backend(for: host)
        task = Task { [weak self, debounce] in
            try? await Task.sleep(for: debounce)
            guard !Task.isCancelled else { return }
            do {
                // Four one-output children preview as four copies of one
                // output, not as one four-output child.
                self?.placement = try await client.placementPreview(request, copies: copies)
                self?.error = nil
            } catch is CancellationError {
                // Superseded by a later control change, not a failed request.
            } catch {
                self?.placement = nil
                self?.error = error.sentence
            }
        }
    }
}
