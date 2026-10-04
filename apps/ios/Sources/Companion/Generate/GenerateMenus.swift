import MoldClient
import SwiftUI

/// An explicit, full-width model control. Long names wrap in the composer
/// instead of competing with Kind and Machine in a fixed-height toolbar.
struct ModelMenu: View {
    @Environment(GenerateController.self) private var generate
    @State private var choosing = false

    var body: some View {
        Button { choosing = true } label: {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Model").foregroundStyle(.secondaryText)
                    Spacer(minLength: 8)
                    // a11y: decorative -- the button's label names its action.
                    Image(systemName: "chevron.up.chevron.down").accessibilityHidden(true)
                }
                .font(.caption)
                if let headline = generate.model?.headline {
                    Text(headline).fixedSize(horizontal: false, vertical: true)
                } else if let identity = generate.modelName {
                    // An unavailable model supplies a technical identity,
                    // like the identifier beneath a ModelChooser headline.
                    Text(verbatim: identity).font(.caption.monospaced())
                        .fixedSize(horizontal: false, vertical: true)
                } else {
                    Text("Choose a Model").fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
            .contentShape(.rect)
        }
        .buttonStyle(.bordered)
        .accessibilityIdentifier("choose-model")
        .accessibilityHint("Choose the kind, model, recipe and machine")
        .sheet(isPresented: $choosing) { ModelChooser() }
    }
}

struct ModelChooser: View {
    @Environment(GenerateController.self) private var generate
    @Environment(\.dismiss) private var dismiss
    @State private var search = ""

    var body: some View {
        NavigationStack {
            List {
                if search.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Section {
                        KindMenu().labelStyle(.titleAndIcon)
                        MachineMenu()
                    }
                    Section {
                        NavigationLink { ModelsView().navigationTitle("Models") } label: {
                            Text("Get More Models…").fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
                ForEach(matchingFamilies, id: \.family) { group in
                    let models = group.models
                    if !models.isEmpty {
                        Section {
                            Text(group.family).font(.headline)
                                .foregroundStyle(.primary)
                                .accessibilityAddTraits(.isHeader)
                            ForEach(models) { model in
                                Button { generate.choose(model); dismiss() } label: {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Label(model.headline, systemImage: model.name == generate.modelName ? "checkmark.circle.fill" : "circle")
                                        Text(verbatim: model.name).font(.caption.monospaced()).foregroundStyle(.secondaryText)
                                    }
                                    .fixedSize(horizontal: false, vertical: true)
                                }
                                .accessibilityIdentifier("model-" + model.name)
                            }
                        }
                    }
                }
                if generate.families.isEmpty {
                    Section {
                        Text("No installed models for this kind on an online machine. Choose another kind, check Machines, or use Get More Models.")
                            .foregroundStyle(.secondaryText)
                    }
                }
                if !generate.families.isEmpty, matchingFamilies.isEmpty {
                    Section {
                        Text("No matching models").font(.headline)
                        Text("Try another name or clear the search.").foregroundStyle(.secondaryText)
                        Button("Clear Search") { search = "" }
                    }
                }
                if search.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, generate.recipes.count > 1 {
                    Section {
                        ForEach(generate.recipes) { recipe in
                            Button { generate.chooseRecipe(recipe.id); dismiss() } label: {
                                Label(recipe.label, systemImage: recipe.id == generate.recipe?.id ? "checkmark.circle.fill" : "circle")
                            }
                        }
                    } header: {
                        Text("Recipe").foregroundStyle(.secondaryText)
                    }
                }
            }
            .scrollEdgeEffectHidden(true)
            .searchable(text: $search, prompt: "Find a model")
            .navigationTitle("Choose a Model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
            .safeAreaInset(edge: .bottom) {
                Button("Close Model Search") { dismiss() }
                    .buttonStyle(.bordered)
                    .frame(maxWidth: .infinity)
                    .padding(8)
                    .background(Color(uiColor: .systemBackground))
                    .accessibilityIdentifier("model-chooser-close")
            }
        }
        .accessibilityIdentifier("model-chooser")
        .presentationDetents([.large])
        .presentationSizing(.page)
    }

    private var matchingFamilies: [(family: String, models: [Model])] {
        let query = search.trimmingCharacters(in: .whitespacesAndNewlines)
        return generate.families.compactMap { group in
            let models = group.models.filter {
                query.isEmpty || $0.name.localizedCaseInsensitiveContains(query)
                    || $0.headline.localizedCaseInsensitiveContains(query)
            }
            return models.isEmpty ? nil : (group.family, models)
        }
    }
}

/// The machine: a dot and its name, or Auto. Only machines that are up AND
/// hold the chosen model can be pinned; the rest say why they cannot.
struct MachineMenu: View {
    @Environment(GenerateController.self) private var generate
    @Environment(HostStore.self) private var hosts
    @Environment(\.dynamicTypeSize) private var size

    var body: some View {
        Menu {
            Picker("Machine", selection: Binding(get: { generate.machine }, set: { generate.machine = $0 })) {
                Label("Auto", systemImage: "sparkles").tag(MachineChoice.auto)
                ForEach(hosts.hosts) { host in
                    Label(host.name, systemImage: "server.rack").tag(MachineChoice.pinned(host.id))
                        .disabled(!hosts.isUp(host))
                }
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "desktopcomputer").accessibilityHidden(true)
                StatusDot(reachability: generate.target.map { hosts.reachability(of: $0) } ?? .unknown)
                Text(size >= .xxLarge && generate.machine == .auto ? String(localized: "Auto") : label)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(minHeight: 44)
            .contentShape(Rectangle())
        }
        .accessibilityLabel(String(localized: "Machine, \(label)"))
    }

    private var label: String {
        switch generate.machine {
        case .auto: generate.target.map { String(localized: "Auto · \($0.name)") } ?? String(localized: "Auto")
        case let .pinned(id): hosts.host(id)?.name ?? String(localized: "Auto")
        }
    }
}
