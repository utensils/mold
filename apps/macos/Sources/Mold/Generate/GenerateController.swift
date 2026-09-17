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
    /// What a machine has been told a model's controls should start at --
    /// read on adoption, after the recipe's own numbers, and never on a KEPT
    /// draft. See `applyStoredDefaults`.
    let defaults: ModelDefaultsStore
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

    /// Whether the mask editor sheet is up. Flipped by the Refine group's
    /// Mask row (`RefineGroup.swift`); `GeneratePane` owns the `.sheet` this
    /// drives.
    var showsMaskEditor = false

    var run: RunState = .idle
    var runTask: Task<Void, Never>?
    var activeBatch: (id: String, clientBatchId: String, host: MoldHost.ID)?

    init(hosts: HostStore, defaults: ModelDefaultsStore) {
        self.hosts = hosts
        self.defaults = defaults
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
        guard let recipe = model.defaultRecipe else { return }
        let isNewModel = !keepingDraft
        draft = draft.adopting(recipe, isNewModel: isNewModel)
        applyStoredDefaults(for: model, on: host, recipe: recipe, isNewModel: isNewModel)
    }

    /// Adopts a model, reconciling the draft against its recipe.
    func select(model: Model, on host: MoldHost.ID) {
        let isNewModel = model.name != modelName
        modelName = model.name
        modelFamily = model.family
        hostID = host
        guard let recipe = model.defaultRecipe else { return }
        draft = draft.adopting(recipe, isNewModel: isNewModel)
        applyStoredDefaults(for: model, on: host, recipe: recipe, isNewModel: isNewModel)
    }

    /// Puts a machine's stored per-model defaults on top of the recipe's own
    /// numbers -- but only on a NEW model; `applying` is already a no-op on a
    /// kept draft, and this skips the store read entirely in that case.
    ///
    /// If this host's listing has never been read, nothing is applied yet;
    /// a refresh is kicked off and, once it lands, applied retroactively --
    /// but only if this is STILL the selected model and host by then. A
    /// second model choice made while that refresh was in flight makes its
    /// answer moot, and re-applying it over whatever is now on screen would
    /// silently overwrite a choice made in between.
    private func applyStoredDefaults(
        for model: Model, on host: MoldHost.ID, recipe: GenerationRecipe, isNewModel: Bool
    ) {
        guard isNewModel else { return }
        guard defaults.hasLoaded(on: host) else {
            Task { [weak self] in
                await self?.defaults.refresh(on: host)
                guard let self, self.modelName == model.name, self.hostID == host else { return }
                self.draft = self.draft.applying(
                    self.defaults.defaults(for: model.name, on: host), recipe: recipe, isNewModel: true)
            }
            return
        }
        draft = draft.applying(defaults.defaults(for: model.name, on: host), recipe: recipe, isNewModel: true)
    }

    /// Asks the host where this would run and roughly how long it would take.
    ///
    /// Read-only -- it reserves nothing. Debounced, because it fires on every
    /// control change and a slider produces a great many of those.
    func refreshPlacement(on host: MoldHost) {
        placementTask?.cancel()
        guard let modelName else { return }
        let request = draft.placementRequest(
            model: modelName, maxIdentityPhotos: hosts.capabilities(of: host)?.maxIdentityPhotos ?? 0
        )
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
