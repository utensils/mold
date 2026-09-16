import Foundation
import MoldClient

/// The Generate pane's state: what is being authored, and what the host says
/// about it.
@MainActor
@Observable
final class GenerateController {
    var draft = RenderDraft()
    var hostID: MoldHost.ID?
    var modelName: String?

    private(set) var placement: PlacementPreview?
    private(set) var placementError: String?
    private var placementTask: Task<Void, Never>?

    var run: RunState = .idle
    var runTask: Task<Void, Never>?
    var activeBatch: (id: String, host: MoldHost.ID)?

    /// Adopts a model while KEEPING the draft that was just restored.
    ///
    /// Reuse has already filled in the size, steps and guidance the print was
    /// made with; treating this as a fresh model choice would immediately
    /// overwrite them with the recipe's defaults.
    func adopt(model: Model, on host: MoldHost.ID, keepingDraft: Bool) {
        modelName = model.name
        hostID = host
        if let recipe = model.defaultRecipe {
            draft = draft.adopting(recipe, isNewModel: !keepingDraft)
        }
    }

    /// Adopts a model, reconciling the draft against its recipe.
    func select(model: Model, on host: MoldHost.ID) {
        let isNewModel = model.name != modelName
        modelName = model.name
        hostID = host
        if let recipe = model.defaultRecipe {
            draft = draft.adopting(recipe, isNewModel: isNewModel)
        }
    }

    /// Asks the host where this would run and roughly how long it would take.
    ///
    /// Read-only -- it reserves nothing. Debounced, because it fires on every
    /// control change and a slider produces a great many of those.
    func refreshPlacement(using backend: @escaping () -> (any MoldBackend)?) {
        placementTask?.cancel()
        guard let modelName else { return }
        let request = draft.request(model: modelName)
        placementTask = Task {
            try? await Task.sleep(for: .milliseconds(350))
            guard !Task.isCancelled, let client = backend() else { return }
            do {
                placement = try await client.placementPreview(request, copies: 1)
                placementError = nil
            } catch {
                placement = nil
                placementError = (error as? LocalizedError)?.errorDescription
                    ?? error.localizedDescription
            }
        }
    }
}
