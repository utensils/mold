import MoldClient
import SwiftUI

/// The same relation-based source/reference layout both native apps author.
struct PictureWells: View {
    @Environment(GenerateController.self) private var generate
    @ScaledMetric(relativeTo: .body) private var side: CGFloat = 72

    var body: some View {
        let caps = generate.recipe?.capabilities
        let references = caps?.referenceImages(family: generate.model?.family, model: generate.modelName)
        let mode = SourceImageMode(references: references)
        let readsSource = Self.showsSourceWell(capabilities: caps, mode: mode)
        VStack(alignment: .leading, spacing: 8) {
            if readsSource || references != nil {
                ScrollView(.horizontal) {
                    HStack(alignment: .top, spacing: 10) {
                        if readsSource {
                            Well(title: String(localized: "Start from"), image: generate.draft.media.sourceImage,
                                 side: side, accepting: PictureImport.engineReadable,
                                 set: { DraftPictureAttachment.useAsSource($0, in: &generate.draft, recipe: generate.recipe) },
                                 clear: clearSource)
                        }
                        if let references { referenceWells(references, mode: mode) }
                    }
                }
                if let note = generate.draft.media.exclusiveWells, note.parked != nil {
                    Text(ExclusiveWells.note).font(.caption).foregroundStyle(.secondaryText)
                }
            }
            if let references, let weight = references.weight, weight.mode.isVisible,
               generate.draft.media.requestConditioning.carriesReferences {
                LabeledSlider(title: "Reference weight", value: Binding(
                    get: { generate.draft.media.referenceWeight ?? weight.default },
                    set: { generate.draft.media.referenceWeight = weight.clamp($0) }),
                    range: weight.min ... weight.max, step: weight.step)
            }
        }
    }

    static func showsSourceWell(capabilities: RecipeCapabilities?, mode: SourceImageMode) -> Bool {
        return capabilities?.readsSourceImage == true && mode.showsSourceWell
            && capabilities?.mesh?.namedViews?.mode.isVisible != true
            && capabilities?.generationReferences?.mode.isVisible != true
            && capabilities.flatMap { BoundaryFramePolicy.resolve(capabilities: $0) } == nil
    }

    @ViewBuilder private func referenceWells(_ capability: ReferenceImagesCapability, mode: SourceImageMode) -> some View {
        ForEach(Array(generate.draft.media.editImages.enumerated()), id: \.offset) { index, image in
            VStack {
                Well(title: referenceTitle(index, capability), image: image, side: side,
                     accepting: capability.acceptingTypes, number: index + 1,
                     set: { DraftPictureAttachment.replaceReference($0, at: index, in: &generate.draft, recipe: generate.recipe) },
                     clear: { DraftPictureAttachment.removeReference(at: index, from: &generate.draft, recipe: generate.recipe) })
                Menu {
                    Button("Move earlier") { move(index, to: index - 1) }.disabled(index == 0)
                    Button("Move later") { move(index, to: index + 1) }
                        .disabled(index + 1 == generate.draft.media.editImages.count)
                } label: {
                    Text("Order").font(.caption).fixedSize(horizontal: false, vertical: true)
                        .frame(minWidth: 44, minHeight: 44).contentShape(Rectangle())
                }
                .accessibilityLabel("Order image \(index + 1)")
                if capability.canvas == .lastReference, index + 1 == generate.draft.media.editImages.count {
                    Text("Sets canvas shape").font(.caption).foregroundStyle(.secondaryText)
                        .frame(maxWidth: side).multilineTextAlignment(.center)
                }
            }
        }
        if capability.hasRoom(for: generate.draft.media.editImages.count) {
            Well(title: capability.primaryIsTarget && generate.draft.media.editImages.isEmpty ? "Picture to edit" : "Add image",
                 image: nil, side: side, accepting: capability.acceptingTypes,
                 set: { DraftPictureAttachment.addReference($0, to: &generate.draft, capability: capability, recipe: generate.recipe) },
                 clear: {})
        }
    }

    private func referenceTitle(_ index: Int, _ capability: ReferenceImagesCapability) -> String {
        capability.primaryIsTarget && index == 0 ? "image 1, target" : "image \(index + 1)"
    }
    private func move(_ index: Int, to destination: Int) {
        DraftPictureAttachment.moveReference(from: index, to: destination, in: &generate.draft, recipe: generate.recipe)
    }
    private func clearSource() {
        generate.draft.media.sourceImage = nil
        generate.draft.media.sourceImageName = nil
        generate.draft.media.sourceImageOriginal = nil
        generate.draft.media.sourceImageOriginalName = nil
        generate.draft.media.sourceImagePixels = nil
    }
}
