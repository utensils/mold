import MoldClient
import MoldStyle
import SwiftUI

/// Authoring a render.
///
/// The picture takes the pane and the controls float over it, so what you are
/// making stays the largest thing on screen.
struct GeneratePane: View {
    /// Not `private`: `GeneratePane+Result`, an extension in another file,
    /// routes Show in Library through it.
    @Binding var destination: Destination
    /// Not `private`, same reason -- the result verbs fetch bytes.
    @Environment(HostStore.self) var hosts
    /// Not `private`: the toolbar's model picker, in an extension in another
    /// file, reads it too.
    @Environment(ModelStore.self) var models
    /// Not `private`, same reason.
    @Environment(GenerateController.self) var controller
    /// Which print this draft came from, and whether its own conditioning
    /// media can be brought back. Not `private`: `startRun` is here, but the
    /// notice is drawn by the body below.
    @Environment(ReuseStore.self) var reuse
    @Environment(DraftPersistence.self) var drafts
    /// Persisted, and deliberately not `private`: the toolbar button that
    /// flips it lives in an extension in another file. Its own key beside
    /// `libraryShowsInspector` -- ⌥⌘I is one shortcut whose STATE is per
    /// destination.
    @AppStorage("generateShowsInspector", store: AppStorageSuite.defaults)
    var showsInspector = true
    /// The draft this pane was holding when the app last quit (`+Models`
    /// asks it which model to adopt), and what each machine says it will
    /// chain (`+Chain`). Neither is `private`: both are read from extensions.
    @State var chainLimits = ChainLimitsStore()

    var body: some View {
        @Bindable var controller = controller

        ZStack(alignment: .bottom) {
            canvas
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            PromptTuck(tucked: $controller.promptTucked, steps: controller.run.steps) {
                PromptPanel(recipe: recipe, draft: $controller.draft, model: selectedModel,
                            host: host, destination: $destination,
                            submit: startRun, cancel: cancelRun, stopAll: { controller.stopAll() },
                            maxBatch: maxBatch, chainLimits: advertisedChainLimits)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .reuseNotice(reuse, draft: controller.draft)
        .persistingDraft(controller, in: drafts)
        // Once per machine, model and rate -- never from `body`.
        .task(id: chainLimitsKey) { refreshChainLimits() }
        .navigationTitle("Generate")
        .navigationSubtitle(subtitle)
        // Before the pane's own `.toolbar`, so the column's switch is the
        // LAST item in the row and the model and recipe capsules stop at the
        // divider. See `TrailingColumn`.
        .trailingColumn(isShowing: $showsInspector) {
            GenerateInspector(recipe: recipe, model: selectedModel, host: host,
                              draft: $controller.draft, destination: $destination)
        }
        .toolbar { toolbar }
        // Escape brings the capsule back, the way it dismisses any other
        // temporary state -- a tucked prompt with no keyboard way out is a
        // corner someone can get stuck in.
        .onExitCommand { controller.promptTucked = false }
        .focusedSceneValue(\.promptTuck, PromptTuckAction(isTucked: controller.promptTucked) {
            controller.promptTucked.toggle()
        })
        .focusedSceneValue(\.inspectorToggle, InspectorToggle(isShowing: showsInspector) {
            showsInspector.toggle()
        })
        .sheet(isPresented: $controller.showsMaskEditor) {
            MaskEditorSheet(draft: $controller.draft)
        }
        .task { await loadModels() }
        .task { await controller.recoverPending() }
        .task { seedSourceImageIfRequested() }
        .onChange(of: hosts.reachability) { _, _ in adoptFirstReadyModel() }
        .onChange(of: controller.draft) { _, _ in refreshPlacement() }
        .onChange(of: controller.modelName) { _, _ in refreshPlacement() }
    }

    // MARK: - Canvas

    @ViewBuilder private var canvas: some View {
        if selectedModel == nil {
            ContentUnavailableView {
                Label("Pick a model", systemImage: "cube")
            } description: {
                Text(models.isLoading
                     ? "Looking at what each machine has installed…"
                     : "Choose a model from the toolbar to start.")
            }
        } else {
            RunCanvas(state: controller.run, actions: resultActions,
                      togglePrompt: { controller.promptTucked.toggle() },
                      onResultShown: controller.handoff.acknowledge)
        }
    }

    // MARK: - Selection

    /// Not `private`, same reason. An explicit Machine choice wins; else the
    /// machine the model was adopted on; else Auto's own answer (decision 2).
    var host: MoldHost? {
        controller.machineChoice.flatMap(hosts.host)
            ?? hosts.hosts.first { $0.id == controller.hostID }
            ?? hosts.preferredHost
    }

    /// Not `private`, same reason.
    var selectedModel: Model? {
        guard let host, let name = controller.modelName else { return nil }
        return models.model(named: name, on: host.id)
    }

    /// Not `private`: the toolbar's recipe picker needs it too. An id the
    /// current model does not advertise falls back to its own default.
    var recipe: GenerationRecipe? {
        if let recipeID = controller.recipeID,
           let recipe = selectedModel?.generationProfile?.recipe(named: recipeID) {
            return recipe
        }
        return selectedModel?.defaultRecipe
    }

    /// What this machine will admit in one batch. Absent means an older host,
    /// which is one at a time.
    private var maxBatch: Int {
        host.flatMap { hosts.capabilities(of: $0)?.maxBatchOutputs } ?? 1
    }

    private var subtitle: String {
        guard let host else { return "No machine" }
        guard let model = selectedModel else { return host.name }
        return "\(model.headline) · \(host.name)"
    }
}
