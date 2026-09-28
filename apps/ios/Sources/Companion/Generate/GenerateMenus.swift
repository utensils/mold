import MoldClient
import SwiftUI

/// The model, as the toolbar's title: its plain name, and its id in mono
/// underneath (dropped into the menu itself at accessibility sizes, where two
/// lines would crowd the bar). The menu lists what the fleet has installed
/// for this kind, by family.
struct ModelMenu: View {
    @Environment(GenerateController.self) private var generate
    @Environment(AppRouter.self) private var router
    @Environment(\.dynamicTypeSize) private var size

    var body: some View {
        Menu {
            ForEach(generate.families, id: \.family) { group in
                Section(group.family) {
                    ForEach(group.models) { model in
                        Button { generate.choose(model) } label: {
                            if model.name == generate.modelName {
                                Label(model.headline, systemImage: "checkmark")
                            } else {
                                Text(model.headline)
                            }
                            Text(verbatim: model.name)
                        }
                    }
                }
            }
            if generate.recipes.count > 1 {
                Section("Recipe") {
                    ForEach(generate.recipes) { recipe in
                        Button { generate.chooseRecipe(recipe.id) } label: {
                            if recipe.id == generate.recipe?.id {
                                Label(recipe.label, systemImage: "checkmark")
                            } else {
                                Text(recipe.label)
                            }
                        }
                    }
                }
            }
            Divider()
            Button { router.selection = .go(.machines) } label: {
                Label("Get More Models…", systemImage: "arrow.down.circle")
            }
        } label: {
            VStack(spacing: 0) {
                Text(generate.model?.headline ?? String(localized: "Choose a Model"))
                    .font(.headline)
                if let name = generate.modelName, !size.isAccessibilitySize {
                    Text(verbatim: name).font(.caption.monospaced()).foregroundStyle(.secondaryText)
                }
            }
            .foregroundStyle(.primary)
        }
        .accessibilityLabel(String(localized: "Model, \(generate.model?.headline ?? String(localized: "none chosen"))"))
    }
}

/// The machine: a dot and its name, or Auto. Only machines that are up AND
/// hold the chosen model can be pinned; the rest say why they cannot.
struct MachineMenu: View {
    @Environment(GenerateController.self) private var generate
    @Environment(HostStore.self) private var hosts

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
                StatusDot(reachability: generate.target.map { hosts.reachability(of: $0) } ?? .unknown)
                Text(label).lineLimit(1)
            }
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
