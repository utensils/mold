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
    /// Persisted, and deliberately not `private`: the toolbar button that
    /// flips it lives in an extension in another file. Its own key beside
    /// `libraryShowsInspector` -- ⌥⌘I is one shortcut whose STATE is per
    /// destination.
    @AppStorage("generateShowsInspector", store: AppStorageSuite.defaults)
    var showsInspector = true
    /// The draft this pane was holding when the app last quit (`+Models`
    /// asks it which model to adopt), and what each machine says it will
    /// chain (`+Chain`). Neither is `private`: both are read from extensions.
    @State var drafts = DraftPersistence()
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
        .trailingColumn(isShowing: showsInspector) {
            GenerateInspector(recipe: recipe, model: selectedModel, host: host,
                              draft: $controller.draft, destination: $destination)
        }
        .persistingDraft(controller, in: drafts)
        // Once per machine, model and rate -- never from `body`.
        .task(id: chainLimitsKey) { refreshChainLimits() }
        .navigationTitle("Generate")
        .navigationSubtitle(subtitle)
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
            RunCanvas(state: controller.run, host: host, actions: resultActions,
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

    /// A clip longer than the checkpoint renders in one pass goes out as an
    /// EPHEMERAL chain job instead of a batch. The routing is resolved here
    /// because it needs the recipe, which the controller does not hold.
    private func startRun() {
        guard let host else { return }
        let routing = recipe.flatMap {
            ClipRouting.resolve(recipe: $0, model: selectedModel, draft: controller.draft,
                                limits: advertisedChainLimits)
        }?.decision ?? .single()
        // The chain door redeems no reuse session, but the chain WIRE carries
        // the bytes per stage -- so the print's picture is fetched into the
        // draft's own well and the render goes out as an ordinary long clip
        // that starts from it. The authority is TAKEN before the await, so a
        // second press finds none and takes the ordinary synchronous path.
        if case .chain = routing, let authority = reuse.pending(for: controller.draft),
           let member = RetainedSourcePicture.member(
               of: authority, forHydrating: outgoingProbe(on: host)) {
            reuse.clear()
            Task { await attachThenRun(member, of: authority) }
            return
        }
        // Whatever a chain still cannot carry -- a mask, an identity photo --
        // is said, and only when something would actually have been hydrated.
        if case .chain = routing {
            reuse.warnIfTheRouteCannotCarryMedia(chained: true, outgoing: outgoingProbe(on: host))
        }
        // TAKEN, not read: a handle is good for one admission and a relay's
        // bytes ride the request that took them, so the submit that gets this
        // is the last one to have it. That is what stops a print conditioning
        // renders nobody asked for, and what stops a print the machine can no
        // longer honour refusing every render after the first.
        controller.submit(
            on: host, backend: hosts.backend(for: host), routing: routing,
            retained: reuse.take(for: controller.draft).map {
                RetainedMediaHydration(authority: $0, hosts: hosts)
            })
    }

    /// Puts the print's picture in the source well, then runs -- or says why
    /// it could not, and runs nothing. A press that quietly rendered a long
    /// clip without the picture it was supposed to start from is the thing
    /// this whole path exists to stop.
    private func attachThenRun(
        _ member: RetainedSourceMedia.Member, of authority: ReuseStore.Authority
    ) async {
        switch await RetainedSourcePicture.fetch(member, of: authority, hosts: hosts) {
        case let .refused(sentence):
            reuse.notice = sentence
        case let .picture(picture):
            RetainedSourcePicture.place(picture, named: authority.filename,
                                        in: &controller.draft)
            startRun()
        }
    }

    private func outgoingProbe(on host: MoldHost) -> GenerateRequest? {
        RetainedSourcePicture.outgoing(controller, on: host, hosts: hosts)
    }

    private func cancelRun() { controller.stop() }

    private func refreshPlacement() {
        host.map(controller.refreshPlacement(on:))
    }
}
