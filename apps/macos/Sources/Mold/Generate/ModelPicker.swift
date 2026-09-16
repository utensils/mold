import MoldClient
import SwiftUI

/// Choosing a model.
///
/// The manifest already describes every model as "Title — plain-English
/// trade-off", so each row says what the thing is FOR rather than printing a
/// quantization tag and leaving the reader to infer it.
struct ModelPicker: View {
    let host: MoldHost?
    let families: [(family: String, models: [Model])]
    let selected: Model?
    let choose: (Model) -> Void

    var body: some View {
        Menu {
            if families.isEmpty {
                Text(host == nil ? "No machine" : "No models installed")
            }
            ForEach(families, id: \.family) { group in
                Section(group.family) {
                    ForEach(group.models) { model in
                        Button { choose(model) } label: {
                            row(model)
                        }
                    }
                }
            }
        } label: {
            Label(selected?.headline ?? "Model", systemImage: "cube")
        }
        // A toolbar menu shows icon-only by default, which would hide the one
        // thing this control is for: which model is chosen.
        .labelStyle(.titleAndIcon)
        .fixedSize()
        .help(selected?.description ?? "Choose a model")
    }

    private func row(_ model: Model) -> some View {
        VStack(alignment: .leading) {
            Text(model.headline)
            if let tradeOff = model.tradeOff {
                Text(tradeOff)
            }
        }
    }
}
