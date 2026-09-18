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
///
/// Its wells are `PictureWell`s like every other picture well in the app. They
/// used to be an `NSOpenPanel` and nothing else -- no Library door, no Paste,
/// no menu -- so a photograph of a face already in the fleet could not be used
/// as one without saving it to this Mac first (the owner's ask, 2026-09-17).
struct IdentityGroup: View {
    let maxPhotos: Int
    @Binding var draft: RenderDraft

    /// Not `private`: `IdentityGroup+Import`, an extension in another file,
    /// owns where a picked photograph goes.
    @State var targeted = false
    /// The server reads a PNG signature and then JPEG markers and nothing
    /// else (`identity.rs:831-880`), while the panel offers HEIC -- the
    /// default format of every iPhone photograph -- and the Library holds
    /// WebP. Both are transcoded on the way in; what cannot be read at all
    /// says so beside the control (finding 02#7).
    @State var importFailure: String?
    /// The one import in flight, so a slower file can never clear a newer
    /// one's message.
    @State var importTask: Task<Void, Never>?

    @Environment(HostStore.self) var hosts
    @Environment(LibraryStore.self) var library

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            WrappingHStack(horizontalSpacing: 6, verticalSpacing: 6) {
                ForEach(photos) { photo in
                    photoWell(photo)
                }
                if photos.count < maxPhotos { addWell }
            }
            // The whole group takes a drop, not just its add well: the wells
            // wrap, and the empty space beside them is where a dragged
            // photograph naturally lands.
            .dropDestination(for: PictureDrop.self) { drops, _ in
                stage(drops)
                return true
            } isTargeted: { targeted = $0 }
            // The padding is unconditional: a group that only pads itself
            // WHILE targeted jumps its whole inspector row under the cursor.
            .padding(2)
            .background(targeted ? Chrome.wellFillTargeted : .clear,
                        in: RoundedRectangle(cornerRadius: Chrome.wellRadius, style: .continuous))
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

    var photos: [IdentityPhoto] { draft.media.identity?.photos ?? [] }

    func remove(_ photo: IdentityPhoto) {
        guard var conditioning = draft.media.identity else { return }
        conditioning.photos.removeAll { $0.id == photo.id }
        draft.media.identity = conditioning.photos.isEmpty ? nil : conditioning
    }

    /// The same square the reference strip stages one in.
    static let photoSize = ReferenceStrip.thumbnailSize

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
