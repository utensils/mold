import MoldClient
import MoldStyle
import SwiftUI

/// The adapter stack: what a checkpoint's weights are blended with.
///
/// Every row is `draft.media.loras`, in the order it was added -- the request
/// carries them the same way (`RenderDraft+Request.swift`). What can be
/// ADDED comes from `LoraStore`, asked once per (machine, model) pair and
/// never matched by family here: the server already did that
/// (`catalog_api.rs:1098-1115`).
struct AdaptersGroup: View {
    let modelName: String
    let host: MoldHost
    let maxCount: Int
    @Binding var draft: RenderDraft

    @Environment(LoraStore.self) private var adapters

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(draft.media.loras) { row($0) }
            addSection
        }
        .task(id: "\(host.id)-\(modelName)") {
            await adapters.refresh(model: modelName, on: host.id)
        }
    }

    private func row(_ choice: LoraChoice) -> some View {
        let info = installed?.first { $0.path == choice.path }
        return VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(choice.name)
                    .lineLimit(1)
                    .truncationMode(.tail)
                Spacer()
                Button {
                    draft.media.loras.removeAll { $0.path == choice.path }
                } label: {
                    Image(systemName: "minus.circle")
                }
                .buttonStyle(.plain)
                .help("Remove this adapter")
            }
            SliderControl(name: "\(choice.name) strength", value: scaleBinding(for: choice),
                          range: Lora.scaleRange, step: 0.05) {
                Text(choice.scale, format: .number.precision(.fractionLength(2)))
            }
        }
        .help(helpText(for: choice, info: info))
        .contextMenu {
            // The row's own vocabulary first -- it is not an action ON the
            // row -- then the shared list (`GenerateMenus.adapterRow`).
            ForEach(info?.trainedWords ?? [], id: \.self) { word in
                Button("Insert \"\(word)\"") { insert(word) }
            }
            if !(info?.trainedWords ?? []).isEmpty { Divider() }
            adapterMenu(choice)
        }
    }

    @ViewBuilder private var addSection: some View {
        switch Rows.resolve(installed: installed, chosen: draft.media.loras, maxCount: maxCount) {
        case .none:
            EmptyView()
        case .empty:
            Text("No adapters installed for this model.")
                .font(.caption)
                .foregroundStyle(.secondary)
        case let .rows(available, canAdd):
            if canAdd {
                Menu("Add adapter…") {
                    ForEach(available) { info in
                        Button(info.name) { add(info) }
                    }
                }
                .menuStyle(.button)
                .fixedSize()
            }
        }
    }

    private var installed: [LoraInfo]? { adapters.rows(for: modelName, on: host.id) }

    private func add(_ info: LoraInfo) {
        draft.media.loras.append(LoraChoice(path: info.path, name: info.name))
    }

    private func insert(_ word: String) {
        draft.prompt += (draft.prompt.isEmpty ? "" : " ") + word
    }

    private func helpText(for choice: LoraChoice, info: LoraInfo?) -> String {
        guard let author = info?.author else { return choice.name }
        return "\(choice.name) — \(author)"
    }

    func scaleBinding(for choice: LoraChoice) -> Binding<Double> {
        Binding(
            get: { draft.media.loras.first { $0.path == choice.path }?.scale ?? Lora.defaultScale },
            set: { newValue in
                guard let index = draft.media.loras.firstIndex(where: { $0.path == choice.path }) else { return }
                draft.media.loras[index].scale = newValue
            }
        )
    }
}

extension AdaptersGroup {
    /// What the group draws, resolved purely from the store's own answer and
    /// the draft's own stack -- no view needed to test it.
    enum Resolution: Equatable {
        /// The machine hasn't answered for this (host, model) pair yet.
        case none
        /// A real answer: this family takes no adapters.
        case empty
        case rows(available: [LoraInfo], canAdd: Bool)
    }

    enum Rows {
        static func resolve(installed: [LoraInfo]?, chosen: [LoraChoice], maxCount: Int) -> Resolution {
            guard let installed else { return .none }
            guard !installed.isEmpty else { return .empty }
            let chosenPaths = Set(chosen.map(\.path))
            let available = installed.filter { !chosenPaths.contains($0.path) }
            let canAdd = chosen.count < maxCount && !available.isEmpty
            return .rows(available: available, canAdd: canAdd)
        }
    }
}
