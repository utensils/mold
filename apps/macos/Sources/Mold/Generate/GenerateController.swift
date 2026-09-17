import Foundation
import MoldClient

/// The Generate pane's state: what is being authored, and what the host says
/// about it.
@MainActor
@Observable
final class GenerateController {
    /// Not `private`: `GenerateController+Run` reports a machine's cancel
    /// failure through it.
    let hosts: HostStore
    var draft = RenderDraft()
    var hostID: MoldHost.ID?
    var modelName: String?
    /// The chosen model's family, e.g. `"flux"` -- what an expand or remix
    /// request resolves through the prompting registry. Kept alongside
    /// `modelName` rather than re-derived, because the controller does not
    /// otherwise hold the `Model` it was chosen from.
    var modelFamily: String?

    private(set) var placement: PlacementPreview?
    private(set) var placementError: String?
    private var placementTask: Task<Void, Never>?

    /// Where a prompt rewrite stands. `GenerateController+Expand` reads and
    /// writes this; it lives here because every other piece of the pane's
    /// state does.
    var expansion: Expansion = .idle
    /// What `revertExpansion()` puts back, and until when -- see
    /// `canRevertExpansion`.
    var lastAcceptedPrompt: LastAcceptedPrompt?

    /// Whether the prompt capsule has slid off the bottom edge so the
    /// picture can be looked at. Visual only -- nothing about the draft or
    /// the run changes with it.
    var promptTucked = false

    var run: RunState = .idle
    var runTask: Task<Void, Never>?
    var activeBatch: (id: String, clientBatchId: String, host: MoldHost.ID)?

    init(hosts: HostStore) {
        self.hosts = hosts
    }

    /// Adopts a model while KEEPING the draft that was just restored.
    ///
    /// Reuse has already filled in the size, steps and guidance the print was
    /// made with; treating this as a fresh model choice would immediately
    /// overwrite them with the recipe's defaults.
    func adopt(model: Model, on host: MoldHost.ID, keepingDraft: Bool) {
        modelName = model.name
        modelFamily = model.family
        hostID = host
        if let recipe = model.defaultRecipe {
            draft = draft.adopting(recipe, isNewModel: !keepingDraft)
        }
    }

    /// Adopts a model, reconciling the draft against its recipe.
    func select(model: Model, on host: MoldHost.ID) {
        let isNewModel = model.name != modelName
        modelName = model.name
        modelFamily = model.family
        hostID = host
        if let recipe = model.defaultRecipe {
            draft = draft.adopting(recipe, isNewModel: isNewModel)
        }
    }

    /// Asks the host where this would run and roughly how long it would take.
    ///
    /// Read-only -- it reserves nothing. Debounced, because it fires on every
    /// control change and a slider produces a great many of those.
    func refreshPlacement(on host: MoldHost) {
        placementTask?.cancel()
        guard let modelName else { return }
        let request = draft.placementRequest(model: modelName)
        let copies = draft.batchSize
        let client = hosts.backend(for: host)
        placementTask = Task {
            try? await Task.sleep(for: .milliseconds(350))
            guard !Task.isCancelled else { return }
            do {
                // Four one-output children preview as four copies of one
                // output, not as one four-output child.
                placement = try await client.placementPreview(request, copies: copies)
                placementError = nil
            } catch is CancellationError {
                // Superseded by a later control change, not a failed request.
            } catch {
                placement = nil
                placementError = error.sentence
            }
        }
    }
}
