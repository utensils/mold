import MoldClient
import SwiftUI

/// Manifest models install by their exact runnable id, even when their
/// upstream repository contains several checkpoints or requires a licence.
struct CuratedModelRow: View {
    @Environment(ModelStore.self) private var models
    @Environment(\.dynamicTypeSize) private var size
    let model: Model
    let host: MoldHost

    var body: some View {
        let stacked = RowAxis.for(size) == .vertical
        let layout = stacked ? AnyLayout(VStackLayout(alignment: .leading, spacing: 8))
            : AnyLayout(HStackLayout(alignment: .center, spacing: 12))
        layout {
            VStack(alignment: .leading, spacing: 4) {
                Text(model.headline).font(.headline)
                if model.hfRepo != nil { ModelSourceLabel(source: "hf") }
                if let tradeOff = model.tradeOff {
                    Text(tradeOff).font(.callout).foregroundStyle(.secondaryText)
                }
                Text(verbatim: model.name).font(.caption.monospaced()).foregroundStyle(.secondaryText)
                if let reason = model.runtimeUnavailableReason, model.runtimeAvailable == false {
                    Text(reason).font(.callout).foregroundStyle(.secondaryText)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            action.frame(maxWidth: stacked ? .infinity : nil)
        }
        .padding(.vertical, 4)
        .accessibilityIdentifier("curated-model-" + model.name)
    }

    @ViewBuilder private var action: some View {
        if let (job, row) = models.progress(for: model.name, on: host.id) {
            VStack(alignment: .leading, spacing: 4) {
                if let fraction = row.fraction { ProgressView(value: fraction) }
                Button("Cancel Download", role: .destructive) { Task { await models.cancel(job: job, on: host.id) } }
                    .buttonStyle(.bordered)
            }
        } else if model.isReady {
            Text("Installed").foregroundStyle(.secondaryText)
        } else if model.runtimeAvailable != false {
            Button(model.repairBytes == nil ? "Get" : "Repair") {
                Task { await models.install(model.name, on: host.id) }
            }
            .buttonStyle(.bordered)
            .accessibilityLabel(String(localized: "Get \(model.headline)"))
        }
    }
}

/// The same authored provider marks as the shared web SourceGlyph. Keep
/// the source in words alongside the mark, including with VoiceOver.
struct ModelSourceLabel: View {
    let source: String

    var body: some View {
        HStack(spacing: 5) {
            if source == "huggingface" || source == "hf" {
                Image("HuggingFace").resizable().scaledToFit().frame(width: 18, height: 18).accessibilityHidden(true)
                Text("Hugging Face")
            } else if source == "civitai" {
                Image("Civitai").resizable().scaledToFit().frame(width: 18, height: 18).accessibilityHidden(true)
                Text("Civitai")
            } else {
                Text(source)
            }
        }
        .font(.caption)
        .foregroundStyle(.secondaryText)
        .accessibilityElement(children: .combine)
    }
}
