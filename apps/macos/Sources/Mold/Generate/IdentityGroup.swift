import AppKit
import MoldClient
import MoldStyle
import SwiftUI

/// Face-identity conditioning: up to the host's own photo limit, a blend
/// weight, and how early in the denoise it starts riding.
///
/// Positive-only, like every conditioning input `RenderDraft+Park` covers --
/// `isShown` needs BOTH the recipe (this checkpoint is qualified) and the
/// host (this build actually links the identity adapter), and a staged
/// photo that fails either question is held rather than dropped (decision 5,
/// M4 design).
struct IdentityGroup: View {
    let maxPhotos: Int
    @Binding var draft: RenderDraft

    @State private var targeted = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            WrappingHStack(horizontalSpacing: 6, verticalSpacing: 6) {
                ForEach(photos) { photo in
                    IdentityPhotoWell(photo: photo) { remove(photo) }
                }
                if photos.count < maxPhotos { addWell }
            }
            LabeledSection("Weight") {
                SliderControl(value: weightBinding, range: Identity.weightRange, step: Identity.weightStep) {
                    Text(weightBinding.wrappedValue, format: .number.precision(.fractionLength(2)))
                }
            }
            LabeledSection("Start step") {
                // `Identity.startStepRange` moves with `draft.steps`, and
                // `RenderDraft.steps`'s own `didSet` keeps a carried value
                // inside it as Steps changes -- this range is never stale.
                Stepper(value: startStepBinding, in: Identity.startStepRange(steps: draft.steps)) {
                    Text(startStepBinding.wrappedValue.formatted())
                }
            }
            if maxPhotos > 1 {
                Text("Several photographs of one person are averaged into one identity.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var photos: [IdentityPhoto] { draft.identity?.photos ?? [] }

    private var addWell: some View {
        RoundedRectangle(cornerRadius: Chrome.wellRadius, style: .continuous)
            .fill(targeted ? Chrome.wellFillTargeted : Chrome.wellFill)
            .frame(width: 52, height: 52)
            .overlay { Image(systemName: "person.crop.circle.badge.plus").foregroundStyle(.tertiary) }
            .onTapGesture { choose() }
            .dropDestination(for: URL.self) { urls, _ in
                for url in urls { append(url) }
                return true
            } isTargeted: { targeted = $0 }
            .help("Add a photograph of the face to preserve")
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .webP, .heic, .tiff]
        panel.allowsMultipleSelection = true
        guard panel.runModal() == .OK else { return }
        for url in panel.urls { append(url) }
    }

    private func append(_ url: URL) {
        guard photos.count < maxPhotos, let data = try? Data(contentsOf: url) else { return }
        var conditioning = draft.identity ?? IdentityConditioning(photos: [])
        conditioning.photos.append(IdentityPhoto(encoded: data.base64EncodedString(), name: url.lastPathComponent))
        draft.identity = conditioning
    }

    private func remove(_ photo: IdentityPhoto) {
        guard var conditioning = draft.identity else { return }
        conditioning.photos.removeAll { $0.id == photo.id }
        draft.identity = conditioning.photos.isEmpty ? nil : conditioning
    }

    private var weightBinding: Binding<Double> {
        Binding(
            get: { draft.identity?.weight ?? Identity.weightDefault },
            set: { newValue in
                var conditioning = draft.identity ?? IdentityConditioning(photos: [])
                conditioning.weight = newValue
                draft.identity = conditioning
            }
        )
    }

    private var startStepBinding: Binding<Int> {
        Binding(
            get: { draft.identity?.startStep ?? Identity.startStepDefault },
            set: { newValue in
                var conditioning = draft.identity ?? IdentityConditioning(photos: [])
                conditioning.startStep = newValue
                draft.identity = conditioning
            }
        )
    }
}

extension IdentityGroup {
    /// Whether this group is drawn at all -- both the recipe (this
    /// checkpoint is qualified) and the host (this build links the identity
    /// adapter) have to say yes.
    static func isShown(recipe: GenerationRecipe, host: Capabilities?) -> Bool {
        recipe.capabilities.supportsIdentity == true && host?.supportsIdentity == true
    }
}
