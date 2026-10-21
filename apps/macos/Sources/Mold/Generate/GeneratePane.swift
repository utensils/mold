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
    @Environment(PromptHistoryStore.self) var promptHistory
    @Environment(ExpandStore.self) var expansions
    /// Holds a licence the press has to accept first; the sheet itself is
    /// the root's (`RootView`). Not `private`: `+Licence` writes it.
    @Environment(DownloadStore.self) var downloads
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
            PromptTuck(tucked: $controller.promptTucked, steps: controller.run.steps) { availableHeight in
                PromptPanel(recipe: recipe, draft: $controller.draft, model: selectedModel,
                            host: host, destination: $destination,
                            submit: startRun, cancel: cancelRun, stopAll: { controller.stopAll() },
                            maxBatch: maxBatch, chainLimits: advertisedChainLimits,
                            maxHeight: availableHeight)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .reuseNotice(reuse, draft: controller.draft)
        .persistingDraft(controller, in: drafts)
        // Once per machine, model and rate -- never from `body`.
        .task(id: chainLimitsKey) {
            controller.recoverSequences()
            refreshChainLimits()
            await controller.recoverDurationJobs()
        }
        .onChange(of: advertisedChainLimits) { _, limits in
            guard !controller.sequence.restoringSource, let limits, let temporal = recipe?.temporal else { return }
            controller.sequence.fitLengths(minimum: limits.authoring?.minFrames ?? temporal.frames.min,
                maximum: min(limits.framesPerClipCap, temporal.frames.max),
                step: limits.authoring?.frameStep ?? temporal.frames.step)
        }
        .task(id: durationSourceKey) { await controller.restoreDurationSource() }
        .task(id: durationPlanKey) { controller.planDuration(on: host, recipe: recipe) }
        .navigationTitle("Generate")
        .navigationSubtitle(subtitle)
        // Before the pane's own `.toolbar`, so the column's switch is the
        // LAST item in the row and the model and recipe capsules stop at the
        // divider. See `TrailingColumn`.
        .trailingColumn(isShowing: $showsInspector, resizable: true) {
            GenerateInspector(recipe: recipe, model: selectedModel, host: host,
                              draft: $controller.draft, destination: $destination)
                .disabled(controller.sequence.preparing)
        }
        .toolbar { toolbar }
        // Escape brings the capsule back, the way it dismisses any other
        // temporary state -- a tucked prompt with no keyboard way out is a
        // corner someone can get stuck in.
        .onExitCommand { controller.promptTucked = false }
        .focusedSceneValue(\.promptTuck, PromptTuckAction(isTucked: controller.promptTucked) {
            controller.promptTucked.toggle()
        })
        .focusedSceneValue(\.expandPrompt, expandPromptAction)
        .focusedSceneValue(\.inspectorToggle, InspectorToggle(isShowing: showsInspector) {
            showsInspector.toggle()
        })
        .sheet(isPresented: $controller.showsMaskEditor) {
            MaskEditorSheet(draft: $controller.draft)
        }
        .task { await loadModels() }
        .task { await controller.recoverPending(); controller.recoverSequences() }
        .task { seedSourceImageIfRequested() }
        .onChange(of: hosts.reachability) { _, _ in adoptFirstReadyModel() }
        .onChange(of: controller.draft) { _, _ in
            reuse.selectionChanged(model: controller.modelName, recipe: controller.recipeID, draft: controller.draft)
            refreshPlacement()
        }
        .onChange(of: controller.modelName) { _, _ in
            reuse.selectionChanged(model: controller.modelName, recipe: controller.recipeID, draft: controller.draft)
            refreshPlacement()
        }
        .onChange(of: controller.recipeID) { _, _ in
            reuse.selectionChanged(model: controller.modelName, recipe: controller.recipeID, draft: controller.draft)
            refreshPlacement()
        }
        .onChange(of: host?.id) { _, _ in refreshPlacement() }
        .task(id: RetainedRecoveryKey(model: controller.modelName, recipe: controller.recipeID,
            reachability: hosts.reachability, instances: hosts.instanceIDs)) {
            await reuse.recoverSaved(controller)
        }
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
        if controller.draft.media.generationReferences.contains(where: { $0.freshVideo != nil }) {
            return models.referenceNormalizationHost(model: controller.modelName, hardSelection: controller.machineChoice, adopted: controller.hostID)
        }
        return controller.machineChoice.flatMap(hosts.host)
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
            return recipe.resolvingReferenceCapabilities(family: selectedModel?.family, model: selectedModel?.name)
        }
        return selectedModel?.defaultRecipe?.resolvingReferenceCapabilities(
            family: selectedModel?.family, model: selectedModel?.name)
    }

    /// What this machine will admit in one batch. Absent means an older host,
    /// which is one at a time.
    private var maxBatch: Int {
        guard let host, let model = selectedModel else { return 1 }
        let caps = hosts.capabilities(of: host)
        return ReferenceUploadPolicy.batchLimit(
            requests: [RenderRequest.one(controller.draft, model: model.name)], apiKey: host.apiKey,
            capabilities: caps?.referenceUploads, batchLimit: caps?.maxBatchOutputs ?? 1)
    }

    private var subtitle: String {
        guard let host else { return "No machine" }
        guard let model = selectedModel else { return host.name }
        return "\(model.headline) · \(host.name)"
    }

    private var expandPromptAction: (() -> Void)? {
        guard let recipe, let host else { return nil }
        let offer = ExpansionOffer.resolve(recipe: recipe, capabilities: hosts.capabilities(of: host))
        let visibility = PromptWand.Visibility.resolve(
            offer: offer, promptMode: recipe.capabilities.promptRequirement, prompt: controller.draft.prompt)
        let isWorking: Bool
        if case .working = expansions.expansion { isWorking = true } else { isWorking = false }
        guard PromptWand.shortcutAvailable(offer: offer, visibility: visibility, isWorking: isWorking)
        else { return nil }
        return {
            Task { await expansions.expand(controller, on: host, backend: hosts.backend(for: host)) }
        }
    }
}

private struct RetainedRecoveryKey: Equatable {
    let model: String?
    let recipe: String?
    let reachability: [UUID: HostStore.Reachability]
    let instances: [UUID: String]
}

extension GeneratePane {
    struct DurationPlanKey: Hashable {
        let request: VideoJobRequest?
        let owner: VideoJobOwner?
        let recipe: GenerationRecipe?
        let capable: Bool
    }
    var durationPlanKey: DurationPlanKey {
        .init(request: try? controller.durationRequest(on: host, planning: true),
              owner: controller.durationOwner(host), recipe: recipe,
              capable: host.flatMap { hosts.capabilities[$0.id] }?.canCreateVideoJob == true)
    }
}

extension GeneratePane {
    struct DurationSourceKey: Hashable {
        let source: VideoJobSource?
        let sourceOwner: VideoLibrarySourceOwner?
        let owner: VideoJobOwner?
        let reachable: Bool
    }
    var durationSourceKey: DurationSourceKey {
        let sourceHost = controller.videoDuration?.sourceOwner.flatMap { hosts.host($0.hostId) }
        return .init(source: controller.videoDuration?.draft.source, sourceOwner: controller.videoDuration?.sourceOwner,
                     owner: controller.durationOwner(sourceHost), reachable: sourceHost.map(hosts.isUp) ?? false)
    }
}
