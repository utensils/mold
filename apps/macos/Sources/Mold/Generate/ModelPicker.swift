import MoldClient
import MoldStyle
import SwiftUI

/// Choosing a model.
///
/// A popover rather than a `Menu`: a machine can have 170 models, which is an
/// unusable system menu, and a menu item flattens to one line — so the
/// manifest's plain-English trade-off, the one thing that tells you what a
/// model is FOR, would be thrown away.
struct ModelPicker: View {
    let host: MoldHost?
    let families: [(family: String, models: [Model])]
    let selected: Model?
    let choose: (Model) -> Void

    @State private var isPresented = false
    @State private var query = ""

    var body: some View {
        Button { isPresented = true } label: {
            Label(selected?.headline ?? "Model", systemImage: "cube")
        }
        .labelStyle(.titleAndIcon)
        .fixedSize()
        .help(selected?.description ?? "Choose a model")
        .popover(isPresented: $isPresented, arrowEdge: .bottom) {
            content
        }
    }

    private var content: some View {
        VStack(spacing: 0) {
            TextField("Search models", text: $query)
                .textFieldStyle(.roundedBorder)
                .padding(10)
            Divider()
            if matches.isEmpty {
                ContentUnavailableView(
                    host == nil ? "No machine" : "Nothing installed",
                    systemImage: "cube",
                    description: Text("Install a model from the Models tab.")
                )
                .frame(height: 200)
            } else {
                list
            }
        }
        .frame(width: 420, height: 440)
    }

    private var list: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 0, pinnedViews: .sectionHeaders) {
                ForEach(matches, id: \.family) { group in
                    Section {
                        ForEach(group.models) { model in
                            row(model)
                        }
                    } header: {
                        Text(group.family)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 4)
                            .background(.bar)
                    }
                }
            }
        }
    }

    private func row(_ model: Model) -> some View {
        Button {
            choose(model)
            isPresented = false
        } label: {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                VStack(alignment: .leading, spacing: 1) {
                    Text(model.headline)
                    if let tradeOff = model.tradeOff {
                        Text(tradeOff)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }
                }
                Spacer(minLength: 8)
                if model.name == selected?.name {
                    Image(systemName: "checkmark").foregroundStyle(.tint)
                }
            }
            .contentShape(Rectangle())
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
        }
        .buttonStyle(.plain)
    }

    private var matches: [(family: String, models: [Model])] {
        guard !query.isEmpty else { return families }
        let needle = query.lowercased()
        return families.compactMap { group in
            let hits = group.models.filter {
                $0.description.lowercased().contains(needle)
                    || $0.name.lowercased().contains(needle)
            }
            return hits.isEmpty ? nil : (family: group.family, models: hits)
        }
    }
}
