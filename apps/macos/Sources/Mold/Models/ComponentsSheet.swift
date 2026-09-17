import MoldClient
import SwiftUI

/// One row per manifest file `GET /api/models/:model/components` reports --
/// kind, name, present or missing, and the path in a selectable, copyable
/// `Text` (the Library inspector's provenance rows set this precedent).
///
/// Never draws `options` (design fact 4, M5): a widely-shared kind carries
/// dozens of unrelated candidate paths -- measured 103 on plato's
/// `transformer` slot -- and that list is a `models.<name>.<component>_path`
/// override's candidates, not provenance for this row.
struct ComponentsSheet: View {
    let model: Model
    let host: MoldHost
    @Environment(ModelStore.self) private var models
    @Environment(DownloadStore.self) private var downloads
    @Environment(\.dismiss) private var dismiss
    @State private var response: ModelComponentsResponse?
    @State private var isLoading = true

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(model.headline).font(.title2.weight(.semibold))
            content
            Spacer(minLength: 0)
            HStack {
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 460, height: 380)
        .task { await load() }
    }

    @ViewBuilder private var content: some View {
        if isLoading {
            ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
        } else if let response {
            VStack(alignment: .leading, spacing: 4) {
                Text(Self.countSentence(response.components.count))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                ScrollView {
                    VStack(alignment: .leading, spacing: 0) {
                        ForEach(response.components) { component in
                            row(component)
                            if component.id != response.components.last?.id { Divider() }
                        }
                    }
                }
            }
        } else {
            // A failure already left its sentence in the failure banner
            // (`hosts.failures`, wired at the pane); this is just what an
            // empty sheet says while that banner is up.
            Text("Couldn't read \(model.headline)'s components.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func row(_ component: ModelComponentStatus) -> some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text(component.name).font(.callout)
                Text(component.kind).font(.caption).foregroundStyle(.secondary)
                if let path = component.path {
                    Text(path)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .help(path)
                }
            }
            Spacer()
            if component.present {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
            } else {
                Label("Missing", systemImage: "exclamationmark.circle").foregroundStyle(.orange)
                Button("Repair") {
                    Task { await downloads.install(Self.repairTarget(for: component, model: model), on: host) }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding(.vertical, 6)
    }

    private func load() async {
        response = await models.components(of: model, on: host.id)
        isLoading = false
    }

    /// "Six files" -- a model being several files is the thing this sheet
    /// exists to show (design S5).
    static func countSentence(_ count: Int) -> String {
        "\(count) \(count == 1 ? "file" : "files")"
    }

    /// What Repair fetches -- the machine's own `repairModel` when the route
    /// named one, else this model's own name, never the wrong one of the
    /// two (design S5 test: repair installs the component the machine
    /// named, not the model whose sheet is open).
    static func repairTarget(for component: ModelComponentStatus, model: Model) -> String {
        component.repairModel ?? model.name
    }
}
