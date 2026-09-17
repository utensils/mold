import MoldClient
import MoldStyle
import SwiftUI

/// Everything the capsule is too small for.
///
/// Every group EXISTS only when the recipe advertises it, and none of them is
/// ever a disabled control: a model that cannot do a thing has nothing to say
/// about it, and a greyed row invites a click that will never work. The
/// ordering is how often a thing is touched, the same rule `LibraryInspector`
/// follows.
struct GenerateInspector: View {
    let recipe: GenerationRecipe?
    let model: Model?
    let host: MoldHost?
    @Binding var draft: RenderDraft
    @Binding var destination: Destination

    @Environment(HostStore.self) private var hosts
    @Environment(ModelStore.self) private var models
    @Environment(LibraryStore.self) private var library
    @Environment(GenerateController.self) private var controller

    @AppStorage("createShowsAdapters", store: AppStorageSuite.defaults)
    private var showsAdapters = true
    @AppStorage("createShowsIdentity", store: AppStorageSuite.defaults)
    private var showsIdentity = true
    @AppStorage("createShowsRefine", store: AppStorageSuite.defaults)
    private var showsRefine = true
    @AppStorage("createShowsOutput", store: AppStorageSuite.defaults)
    private var showsOutput = true
    @AppStorage("createShowsFileUnder", store: AppStorageSuite.defaults)
    private var showsFileUnder = false
    @AppStorage("createShowsRecent", store: AppStorageSuite.defaults)
    private var showsRecent = false

    var body: some View {
        Group {
            if model == nil {
                ContentUnavailableView("Nothing to set", systemImage: "slider.horizontal.3")
            } else {
                ScrollView { content.padding(16) }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder private var content: some View {
        VStack(alignment: .leading, spacing: 14) {
            if let recipe, let stack = recipe.capabilities.loraStack, let model, let host {
                DisclosureGroup("Adapters", isExpanded: $showsAdapters) {
                    AdaptersGroup(modelName: model.name, host: host, maxCount: stack.maxCount, draft: $draft)
                        .padding(.top, 6)
                }
                .font(.callout)
            }
            if let recipe, IdentityGroup.isShown(recipe: recipe, host: capabilities) {
                DisclosureGroup("Identity", isExpanded: $showsIdentity) {
                    IdentityGroup(maxPhotos: capabilities?.maxIdentityPhotos ?? 0, draft: $draft)
                        .padding(.top, 6)
                }
                .font(.callout)
            }
            if RefineGroup.isShown(recipe: recipe, models: hostModels) {
                DisclosureGroup("Refine", isExpanded: $showsRefine) {
                    RefineGroup(recipe: recipe, models: hostModels, draft: $draft, destination: $destination)
                        .padding(.top, 6)
                }
                .font(.callout)
            }
            DisclosureGroup("Output", isExpanded: $showsOutput) {
                OutputGroup(output: recipe?.capabilities.output, models: hostModels, draft: $draft)
                    .padding(.top, 6)
            }
            .font(.callout)
            if FileUnderGroup.isShown(capabilities: capabilities) {
                DisclosureGroup("File under", isExpanded: $showsFileUnder) {
                    FileUnderGroup(shelves: library.shelves, draft: $draft)
                        .padding(.top, 6)
                }
                .font(.callout)
            }
            RecentGroup(host: host, draft: $draft, isExpanded: $showsRecent, isBusy: controller.run.isBusy)
        }
    }

    private var capabilities: Capabilities? {
        host.flatMap { hosts.capabilities(of: $0) }
    }

    /// Every model this host has, upscalers included -- `ModelStore.ready`
    /// filters those out, so `OutputGroup` reads the raw list and does its
    /// own `isUpscaler && isReady` filtering (`UpscaleRow.resolve`).
    private var hostModels: [Model] {
        guard let host else { return [] }
        return models.byHost[host.id] ?? []
    }
}
