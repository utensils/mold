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
    /// The server reads a PNG signature and then JPEG markers and nothing
    /// else (`identity.rs:831-880`), while the panel offered HEIC -- the
    /// default format of every iPhone photograph. The refusal belongs beside
    /// the control, not in a 422 after the upload (finding 02#7).
    @State private var importFailure: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            WrappingHStack(horizontalSpacing: 6, verticalSpacing: 6) {
                ForEach(photos) { photo in
                    IdentityPhotoWell(photo: photo) { remove(photo) }
                }
                if photos.count < maxPhotos { addWell }
            }
            LabeledSection("Weight") {
                SliderControl(name: "Identity weight", value: weightBinding,
                              range: Identity.weightRange, step: Identity.weightStep) {
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
            if let importFailure {
                Text(importFailure).font(.caption).foregroundStyle(.secondary)
            }
            if maxPhotos > 1 {
                Text("Several photographs of one person are averaged into one identity.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var photos: [IdentityPhoto] { draft.media.identity?.photos ?? [] }

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
            .accessibilityElement()
            .accessibilityLabel("Add a photograph of the face to preserve")
            .accessibilityAddTraits(.isButton)
            .accessibilityAction { choose() }
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .webP, .heic, .tiff]
        panel.allowsMultipleSelection = true
        guard panel.runModal() == .OK else { return }
        for url in panel.urls { append(url) }
    }

    /// Reads, conforms to PNG/JPEG and encodes off the main actor. A HEIC or
    /// TIFF photograph is TRANSCODED rather than refused -- it is the likeliest
    /// picture of a face on this Mac, and re-encoding it is the whole fix.
    private func append(_ url: URL) {
        guard photos.count < maxPhotos else { return }
        Task {
            do {
                let picked = try await PictureImport.load(
                    url, accepting: PictureImport.identityReadable)
                guard photos.count < maxPhotos else { return }
                var conditioning = draft.media.identity ?? IdentityConditioning(photos: [])
                conditioning.photos.append(
                    IdentityPhoto(encoded: picked.encoded, name: picked.name))
                draft.media.identity = conditioning
                importFailure = nil
            } catch {
                importFailure = error.reasonSentence
            }
        }
    }

    private func remove(_ photo: IdentityPhoto) {
        guard var conditioning = draft.media.identity else { return }
        conditioning.photos.removeAll { $0.id == photo.id }
        draft.media.identity = conditioning.photos.isEmpty ? nil : conditioning
    }

    private var weightBinding: Binding<Double> {
        Binding(
            get: { draft.media.identity?.weight ?? Identity.weightDefault },
            set: { newValue in
                var conditioning = draft.media.identity ?? IdentityConditioning(photos: [])
                conditioning.weight = newValue
                draft.media.identity = conditioning
            }
        )
    }

    private var startStepBinding: Binding<Int> {
        Binding(
            get: { draft.media.identity?.startStep ?? Identity.startStepDefault },
            set: { newValue in
                var conditioning = draft.media.identity ?? IdentityConditioning(photos: [])
                conditioning.startStep = newValue
                draft.media.identity = conditioning
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
