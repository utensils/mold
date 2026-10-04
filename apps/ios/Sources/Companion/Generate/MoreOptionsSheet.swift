import MoldClient
import SwiftUI

/// More options (DESIGN.md §5.1): the Mac inspector's sections, each present
/// only when the recipe reads it. Shape, Steps and Batch repeat at the top so
/// they are still reachable when the chip row folds away at large text.
struct MoreOptionsSheet: View {
    @Environment(GenerateController.self) private var generate
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        @Bindable var generate = generate
        NavigationStack {
            Form {
                if let recipe = generate.recipe {
                    Section {
                        OptionsControls(recipe: recipe)
                    }
                    SamplerSection(recipe: recipe)
                    SourceFitSection(recipe: recipe)
                    if generate.draft.media.requestConditioning.carriesSource, generate.draft.media.sourceImage != nil, recipe.capabilities.supportsStrength != false {
                        Section {
                            LabeledSlider(title: String(localized: "How much to change it"),
                                          value: $generate.draft.strength, range: 0.05 ... 1, step: 0.05)
                        } header: { SectionHeader(String(localized: "Start from a photo")) }
                    }
                    if recipe.temporal != nil { ClipSection() }
                    OutputSection(recipe: recipe)
                    AdaptersSection(recipe: recipe)
                    IdentitySection(recipe: recipe)
                    RefineSection(recipe: recipe)
                    FileUnderSection()
                } else {
                    Text("Choose a model first.").foregroundStyle(.secondaryText)
                }
            }
            .navigationTitle("More Options")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Reset") { generate.resetOptions() }
                }
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
        .presentationDetents([.medium, .large])
    }
}

/// Stick to my words, Repeat this look, and what to avoid.
private struct SamplerSection: View {
    @Environment(GenerateController.self) private var generate
    let recipe: GenerationRecipe

    var body: some View {
        @Bindable var generate = generate
        Section {
            if recipe.guidance.mode != .fixed, recipe.guidance.mode.isVisible {
                LabeledSlider(title: String(localized: "Stick to my words"), value: $generate.draft.guidance,
                              range: recipe.guidance.min ... recipe.guidance.max, step: recipe.guidance.step)
            }
            Picker("Seed", selection: $generate.draft.locksSeed) {
                Text("Random").tag(false)
                Text("Fixed").tag(true)
            }
            .pickerStyle(.menu)
            if generate.draft.locksSeed {
                AdaptiveRow { Text("Seed") } value: {
                    TextField("Seed", value: $generate.draft.seed, format: .number.grouping(.never))
                        .keyboardType(.numberPad)
                        .multilineTextAlignment(.trailing)
                        .font(.body.monospacedDigit())
                }
            }
            if recipe.capabilities.negativePrompt?.isAvailable ?? true {
                TextField("Avoid (optional)", text: $generate.draft.negativePrompt, axis: .vertical)
                    .lineLimit(1 ... 4)
            }
        } header: {
            SectionHeader(String(localized: "Look"))
        }
    }
}

/// A clip's sound, where the recipe offers it.
private struct ClipSection: View {
    @Environment(GenerateController.self) private var generate

    var body: some View {
        if generate.draft.offersAudioControl {
            Section {
                Toggle("Sound", isOn: Binding(get: { generate.draft.enableAudio },
                                              set: { generate.draft.preferredAudio = $0 }))
            } header: { SectionHeader(String(localized: "Clip")) }
        }
    }
}

/// Format, a transparent background, and whether the print is kept.
private struct OutputSection: View {
    @Environment(GenerateController.self) private var generate
    let recipe: GenerationRecipe

    var body: some View {
        @Bindable var generate = generate
        let formats = recipe.capabilities.output?.formats ?? []
        Section {
            if formats.count > 1 {
                Picker("Format", selection: Binding(
                    get: { generate.draft.outputFormat ?? recipe.capabilities.output?.defaultFormat ?? formats[0] },
                    set: { generate.draft.outputFormat = $0 })) {
                    ForEach(formats, id: \.self) { Text($0.uppercased()).tag($0) }
                }
            }
            if let transparency = recipe.capabilities.transparency, transparency.mode.isVisible {
                Toggle("Transparent background", isOn: $generate.draft.transparentBackground)
            }
            Toggle("Save to Library", isOn: $generate.draft.savesToGallery)
        } header: {
            SectionHeader(String(localized: "Output"))
        }
    }
}

/// A title, tags and a collection, filed as the print is made.
private struct FileUnderSection: View {
    @Environment(GenerateController.self) private var generate
    @Environment(LibraryStore.self) private var library
    @State private var tag = ""

    var body: some View {
        @Bindable var generate = generate
        Section {
            TextField("Title (optional)", text: $generate.draft.title)
            HStack {
                TextField("Add a tag", text: $tag)
                    .textInputAutocapitalization(.never)
                    .onSubmit(addTag)
                Button("Add", action: addTag).disabled(tag.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            ForEach(generate.draft.tags, id: \.self) { name in
                Text("#\(name)").swipeActions {
                    Button("Remove", role: .destructive) { generate.draft.tags.removeAll { $0 == name } }
                }
            }
            Picker("Collection", selection: $generate.draft.collectionName) {
                Text("None").tag(String?.none)
                ForEach(library.shelves) { Text($0.name).tag(Optional($0.name)) }
            }
        } header: {
            SectionHeader(String(localized: "File under"))
        }
    }

    private func addTag() {
        var clean = tag.trimmingCharacters(in: .whitespacesAndNewlines)
        while clean.hasPrefix("#") { clean.removeFirst() }
        guard !clean.isEmpty, !generate.draft.tags.contains(clean) else { return }
        generate.draft.tags.append(clean)
        tag = ""
    }
}

/// A slider that says its value in mono, and moves by the recipe's own step.
struct LabeledSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            AdaptiveRow { Text(title) } value: {
                Text(value.formatted(.number.precision(.fractionLength(0...2)))).monospacedDigit()
            }
            Slider(value: $value, in: range, step: step > 0 ? step : 0.01) { Text(title) }
        }
    }
}
