import MoldClient
import MoldStyle
import SwiftUI

/// One variant of a model: what it costs you and what you get for it.
struct ModelVariantRow: View {
    let model: Model
    let groupTitle: String
    let install: (Model) -> Void
    let progress: DownloadStore.Progress?

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            if model.tag != nil { tag }
            VStack(alignment: .leading, spacing: 2) {
                if let detail { Text(detail) }
                if model.isLoaded == true {
                    Text("Loaded and ready")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer(minLength: 12)
            if let size = model.sizeGb {
                Text(size.formatted(.number.precision(.fractionLength(1))) + " GB")
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            ModelStateLabel(model: model, progress: progress, install: install)
                .frame(minWidth: 132, alignment: .trailing)
        }
        .padding(.vertical, 3)
        .help(model.name)
    }

    /// What this row adds over its heading. A model whose description is
    /// already the heading has nothing more to say here.
    private var detail: String? {
        if let tradeOff = model.tradeOff { return tradeOff }
        return model.headline == groupTitle ? nil : model.headline
    }

    /// The quantization, as a label rather than as the row's identity.
    private var tag: some View {
        Text(model.tag?.uppercased() ?? "")
            .font(.caption.weight(.medium))
            .monospaced()
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Chrome.wellFill, in: RoundedRectangle(cornerRadius: Chrome.tileRadius))
            .frame(minWidth: 52, alignment: .leading)
    }

}
