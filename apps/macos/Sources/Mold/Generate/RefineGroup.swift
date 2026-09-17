import AppKit
import MoldClient
import MoldStyle
import SwiftUI

/// The mask row and the ControlNet rows: what refines a picture the model
/// already made, rather than what it starts from.
///
/// Every row is drawn from a pure gate the same way `OutputGroup` and
/// `FileUnderGroup` are -- `maskRow` and `ControlNetRow.resolve`
/// (`RefineGroup+ControlNet.swift`) so a test never needs a view to ask what
/// this group would show.
struct RefineGroup: View {
    let recipe: GenerationRecipe?
    /// This host's whole model list -- filtered to installed ControlNet
    /// adapters by `ControlNetRow.resolve`, the same idiom `UpscaleRow`
    /// already uses for upscalers.
    let models: [Model]
    @Binding var draft: RenderDraft
    @Binding var destination: Destination

    @Environment(GenerateController.self) private var controller
    @State private var maskPreview: NSImage?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            maskSection
            controlNetSection
        }
        .task(id: draft.media.maskImage) { await loadMaskPreview() }
    }

    @ViewBuilder private var maskSection: some View {
        switch Self.maskRow(capabilities: recipe?.capabilities, hasSource: draft.media.sourceImage != nil) {
        case .hidden:
            EmptyView()
        case .needsSource:
            LabeledSection("Mask") {
                Text("Add a source picture first.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        case .ready:
            LabeledSection("Mask") {
                HStack(spacing: 8) {
                    Button("Edit mask…") { controller.showsMaskEditor = true }
                    if let maskPreview {
                        Image(nsImage: maskPreview)
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                            .frame(width: 28, height: 28)
                            .clipShape(RoundedRectangle(cornerRadius: Chrome.tileRadius, style: .continuous))
                        Text("Mask painted").font(.caption).foregroundStyle(.secondary)
                        Spacer()
                        Button {
                            draft.media.maskImage = nil
                        } label: {
                            Image(systemName: "minus.circle")
                        }
                        .buttonStyle(.plain)
                        .help("Remove this mask")
                    }
                }
            }
        }
    }

    private func loadMaskPreview() async {
        guard let base64 = draft.media.maskImage, let data = Data(base64Encoded: base64) else {
            maskPreview = nil
            return
        }
        maskPreview = NSImage(data: data)
    }
}

extension RefineGroup {
    /// What the Mask row shows, resolved purely from the recipe's own
    /// capabilities and whether a source picture currently survives.
    enum MaskRowState: Equatable {
        case hidden
        case needsSource
        case ready
    }

    /// Both `acceptsMask` and `readsSourceImage` have to say yes -- a mask
    /// with no source is meaningless and refused outright
    /// (`validation.rs:3101-3107`), and `RefineGroup+ControlNet.swift`'s
    /// `isShown` reuses this same question to decide the whole group.
    static func maskCapable(_ capabilities: RecipeCapabilities) -> Bool {
        capabilities.acceptsMask && capabilities.readsSourceImage
    }

    static func maskRow(capabilities: RecipeCapabilities?, hasSource: Bool) -> MaskRowState {
        guard let capabilities, maskCapable(capabilities) else { return .hidden }
        return hasSource ? .ready : .needsSource
    }

    /// Whether the whole Refine `DisclosureGroup` is worth drawing at all --
    /// a mask-only recipe still shows the group even with no ControlNet
    /// adapter installed, and vice versa.
    static func isShown(recipe: GenerationRecipe?, models: [Model]) -> Bool {
        guard let recipe else { return false }
        if maskCapable(recipe.capabilities) { return true }
        return ControlNetRow.resolve(control: recipe.capabilities.controlNet, models: models) != .hidden
    }
}
