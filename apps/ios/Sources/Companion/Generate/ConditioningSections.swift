import MoldClient
import SwiftUI

/// Add-on looks (LoRAs): what the machine has for this model, each with a
/// weight, up to the recipe's own stack limit.
struct AdaptersSection: View {
    @Environment(GenerateController.self) private var generate
    @Environment(HostStore.self) private var hosts
    let recipe: GenerationRecipe
    @State private var available: [LoraInfo] = []

    var body: some View {
        if let stack = recipe.capabilities.loraStack, stack.mode.isVisible {
            @Bindable var generate = generate
            Section {
                ForEach($generate.draft.media.loras) { $lora in
                    VStack(alignment: .leading) {
                        LabeledSlider(title: lora.name, value: $lora.scale, range: Lora.scaleRange, step: 0.05)
                    }
                    .swipeActions {
                        Button("Remove", role: .destructive) {
                            generate.draft.media.loras.removeAll { $0.id == lora.id }
                        }
                    }
                }
                if generate.draft.media.loras.count < stack.maxCount {
                    Menu {
                        ForEach(available.filter { info in !generate.draft.media.loras.contains { $0.path == info.path } }) { info in
                            Button(info.name) {
                                generate.draft.media.loras.append(LoraChoice(path: info.path, name: info.name))
                            }
                        }
                    } label: {
                        Label("Add a Look", systemImage: "plus")
                    }
                    .disabled(available.isEmpty)
                }
                if available.isEmpty {
                    Text("This machine has no add-on looks for this model.").foregroundStyle(.secondaryText)
                }
            } header: {
                SectionHeader(String(localized: "Add-on looks"))
            }
            .task(id: "\(generate.modelName ?? "")|\(generate.target?.id.uuidString ?? "")") {
                guard let host = generate.target, let model = generate.modelName else { return }
                available = (try? await hosts.backend(for: host).loras(compatibleWith: model)) ?? []
            }
        }
    }
}

/// A face to keep: the identity photos, how strongly, and from which step.
struct IdentitySection: View {
    @Environment(GenerateController.self) private var generate
    @Environment(HostStore.self) private var hosts
    @ScaledMetric(relativeTo: .body) private var side: CGFloat = 72
    let recipe: GenerationRecipe

    var body: some View {
        if recipe.capabilities.supportsIdentity == true {
            let limit = max(1, generate.target.flatMap { hosts.capabilities[$0.id]?.maxIdentityPhotos } ?? 1)
            let photos = generate.draft.media.identity?.photos ?? []
            Section {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 10) {
                        ForEach(Array(photos.enumerated()), id: \.offset) { index, photo in
                            Well(title: String(localized: "Face \(index + 1)"), image: photo.encoded, side: side,
                                 accepting: PictureImport.identityReadable,
                                 set: { generate.draft.media.identity?.photos[index] = IdentityPhoto(encoded: $0.encoded, name: $0.name) },
                                 clear: { removePhoto(at: index) })
                        }
                        if photos.count < limit {
                            Well(title: String(localized: "Add a face"), image: nil, side: side,
                                 accepting: PictureImport.identityReadable,
                                 set: { addPhoto($0) }, clear: {})
                        }
                    }
                }
                if generate.draft.media.identity != nil {
                    LabeledSlider(title: String(localized: "Identity strength"),
                                  value: Binding(get: { generate.draft.media.identity?.weight ?? Identity.weightDefault },
                                                 set: { generate.draft.media.identity?.weight = $0 }),
                                  range: Identity.weightRange, step: Identity.weightStep)
                    Stepper(value: Binding(get: { generate.draft.media.identity?.startStep ?? 0 },
                                           set: { generate.draft.media.identity?.startStep = $0 }),
                            in: 0 ... max(0, generate.draft.steps - 1)) {
                        AdaptiveRow { Text("Identity start step") } value: {
                            Text("\(generate.draft.media.identity?.startStep ?? 0)").monospacedDigit()
                        }
                    }
                }
            } header: {
                SectionHeader(String(localized: "Keep a face"))
            }
        }
    }

    private func addPhoto(_ picked: ImportedPicture) {
        let photo = IdentityPhoto(encoded: picked.encoded, name: picked.name)
        if generate.draft.media.identity == nil {
            generate.draft.media.identity = IdentityConditioning(photos: [photo])
        } else {
            generate.draft.media.identity?.photos.append(photo)
        }
    }

    private func removePhoto(at index: Int) {
        generate.draft.media.identity?.photos.remove(at: index)
        if generate.draft.media.identity?.photos.isEmpty == true { generate.draft.media.identity = nil }
    }
}

/// Refine: a ControlNet picture with its strength, and a mask painted over
/// the source picture to say where to change it.
struct RefineSection: View {
    @Environment(GenerateController.self) private var generate
    @ScaledMetric(relativeTo: .body) private var side: CGFloat = 72
    let recipe: GenerationRecipe
    @State private var editingMask = false

    var body: some View {
        let control = recipe.capabilities.controlNet
        let mask = recipe.capabilities.acceptsMask && generate.draft.media.sourceImage != nil
        if control?.mode.isVisible == true || mask {
            Section {
                if control?.mode.isVisible == true {
                    Well(title: String(localized: "Guide picture"), image: generate.draft.media.control?.image,
                         side: side, accepting: PictureImport.engineReadable,
                         set: { picked in
                             var next = generate.draft.media.control ?? ControlConditioning()
                             next.image = picked.encoded
                             next.name = picked.name
                             generate.draft.media.control = next
                         },
                         clear: { generate.draft.media.control = nil })
                    if generate.draft.media.control != nil {
                        LabeledSlider(title: String(localized: "Guide strength"),
                                      value: Binding(get: { generate.draft.media.control?.scale ?? 1 },
                                                     set: { generate.draft.media.control?.scale = $0 }),
                                      range: 0 ... 2, step: 0.05)
                    }
                }
                if mask {
                    Button(generate.draft.media.maskImage == nil ? "Paint Where to Change…" : "Edit the Mask…") {
                        editingMask = true
                    }
                    if generate.draft.media.maskImage != nil {
                        Button("Clear the Mask", role: .destructive) { generate.draft.media.maskImage = nil }
                    }
                }
            } header: {
                SectionHeader(String(localized: "Refine"))
            }
            .fullScreenCover(isPresented: $editingMask) { MaskEditor() }
        }
    }
}
