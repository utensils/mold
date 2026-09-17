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
    /// draft. See `GenerateController+Defaults.applyStoredDefaults`.
    let defaults: ConfigStore
    var draft = RenderDraft()
    var hostID: MoldHost.ID?
    var modelName: String?
    /// Which of the chosen model's recipes is running -- `nil` means its
    /// default. Reset to `nil` on every model change; `selectRecipe` is the
    /// only place that sets it to something else.
    var recipeID: String?
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
    var activeBatch: ActiveBatch?
    /// Stop, and a second press, while an admission is still unanswered.
    let submissions = SubmissionFence()
    /// The beat between a batch settling and the next one taking the canvas.
    let handoff: ResultHandoff
    /// Batches this pane admitted while another was still on screen, in
    /// admission order. The canvas follows `activeBatch`; when it settles or
    /// is stopped, the head of this list is followed next (M8 decision 8).
    ///
    /// `internal(set)`, not `private(set)`: mutated from
    /// `GenerateController+Run` and `GenerateController+Queue`, different
    /// files -- the same reason `HostStore.failures` is `internal(set)`.
    internal(set) var queued: [ActiveBatch] = []

    init(hosts: HostStore, defaults: ConfigStore, handoff: ResultHandoff = ResultHandoff()) {
        self.hosts = hosts
        self.defaults = defaults
        self.handoff = handoff
    }

    /// Adopts a model while KEEPING the draft that was just restored.
    ///
    /// Reuse has already filled in the size, steps and guidance the print was
    /// made with; treating this as a fresh model choice would immediately
    /// overwrite them with the recipe's defaults. Also pins `machineChoice`
    /// to this host (M8 decision 2): the model was adopted THERE, so "Use
    /// These Settings" must not leave the run pointed at Auto.
    func adopt(model: Model, on host: MoldHost.ID, keepingDraft: Bool) {
        modelName = model.name
        modelFamily = model.family
        hostID = host
        recipeID = nil
        machineChoice = host
        guard let recipe = model.defaultRecipe else { return }
        let isNewModel = !keepingDraft
        draft = draft.adopting(recipe, isNewModel: isNewModel, family: model.family, model: model.name)
        applyStoredDefaults(for: model, on: host, recipe: recipe, isNewModel: isNewModel)
    }

    /// Adopts a model, reconciling the draft against its recipe.
    func select(model: Model, on host: MoldHost.ID) {
        let isNewModel = model.name != modelName
        modelName = model.name
        modelFamily = model.family
        hostID = host
        if isNewModel { recipeID = nil }
        guard let recipe = model.defaultRecipe else { return }
        draft = draft.adopting(recipe, isNewModel: isNewModel, family: model.family, model: model.name)
        applyStoredDefaults(for: model, on: host, recipe: recipe, isNewModel: isNewModel)
    }

    /// Switches recipe on the SAME model -- one of LTX-2's pipelines, most
    /// often. Re-adopts the draft against it exactly like a model change
    /// does: steps, guidance, size, format and every group re-read.
    func selectRecipe(_ recipe: GenerationRecipe) {
        recipeID = recipe.id
        draft = draft.adopting(recipe, isNewModel: false, family: modelFamily, model: modelName)
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
