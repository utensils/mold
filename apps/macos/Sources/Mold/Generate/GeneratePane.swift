import MoldClient
import MoldStyle
import SwiftUI

/// Authoring a render.
///
/// The picture takes the pane and the controls float over it in a material
/// panel, so what you are making stays the largest thing on screen.
struct GeneratePane: View {
    @Environment(HostStore.self) private var hosts
    @Environment(ModelStore.self) private var models
    @Environment(GenerateController.self) private var controller

    var body: some View {
        @Bindable var controller = controller

        ZStack(alignment: .bottom) {
            canvas
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            PromptPanel(recipe: recipe, draft: $controller.draft, model: selectedModel,
                        submit: startRun, cancel: cancelRun, maxBatch: maxBatch)
                .padding(20)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .navigationTitle("Generate")
        .navigationSubtitle(subtitle)
        .toolbar { toolbar }
        .task { await loadModels() }
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
            RunCanvas(state: controller.run, host: host)
        }
    }

    // MARK: - Selection

    private var host: MoldHost? {
        hosts.hosts.first { $0.id == controller.hostID } ?? hosts.preferredHost
    }

    private var selectedModel: Model? {
        guard let host, let name = controller.modelName else { return nil }
        return models.model(named: name, on: host.id)
    }

    private var recipe: GenerationRecipe? { selectedModel?.defaultRecipe }

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

    // MARK: - Toolbar

    @ToolbarContentBuilder private var toolbar: some ToolbarContent {
        ToolbarItem {
            ModelPicker(
                host: host,
                families: host.map { models.families(on: $0.id) } ?? [],
                selected: selectedModel
            ) { model in
                if let host { controller.select(model: model, on: host.id) }
            }
        }
    }

    private func loadModels() async {
        // Reachability decides which machine we land on, so make sure it is
        // known before choosing one.
        await hosts.refreshAll()
        await models.refresh(hosts: hosts.hosts) { hosts.backend(for: $0) }
        // Nothing chosen yet: start on something the machine can actually run.
        if controller.modelName == nil, let host,
           let first = models.ready(on: host.id).first {
            controller.select(model: first, on: host.id)
        }
    }

    private func startRun() {
        guard let host else { return }
        controller.submit(on: host, backend: hosts.backend(for: host))
    }

    private func cancelRun() {
        guard let host else { return }
        controller.cancel(backend: hosts.backend(for: host))
    }

    private func refreshPlacement() {
        controller.refreshPlacement {
            guard let host else { return nil }
            return hosts.backend(for: host)
        }
    }
}
