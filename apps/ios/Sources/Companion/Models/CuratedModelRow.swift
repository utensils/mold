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
        let stacked = RowAxis.model(size, downloading: models.progress(for: model.name, on: host.id) != nil) == .vertical
        let layout = stacked ? AnyLayout(VStackLayout(alignment: .leading, spacing: 8))
            : AnyLayout(HStackLayout(alignment: .center, spacing: 12))
        layout {
            VStack(alignment: .leading, spacing: 4) {
                Text(model.headline).font(.headline)
                if model.hfRepo != nil { ModelSourceLabel(source: "hf") }
                if let summary = model.summary {
                    Text(summary).font(.callout).foregroundStyle(.secondaryText)
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
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("curated-model-" + model.name)
    }

    @ViewBuilder private var action: some View {
        if let (job, row) = models.progress(for: model.name, on: host.id) {
            ModelDownloadControls(row: row) {
                Task { await models.cancel(job: job, on: host.id) }
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

/// Keep progress below the full-width model identity. The visual Cancel label
/// is short, while VoiceOver names the operation and the hit area stays 44pt.
struct ModelDownloadControls: View {
    @Environment(\.dynamicTypeSize) private var size
    let row: DownloadProgress
    let cancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let fraction = row.fraction { ProgressView(value: fraction) }
            let layout = RowAxis.for(size) == .vertical
                ? AnyLayout(VStackLayout(alignment: .leading, spacing: 6))
                : AnyLayout(HStackLayout(alignment: .center, spacing: 8))
            layout {
                Text(verbatim: row.sentence(bytesPerSecond: nil))
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondaryText)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Button(role: .destructive, action: cancel) {
                    Text("Cancel").frame(minWidth: 44, minHeight: 44)
                }
                    .buttonStyle(.bordered)
                    .accessibilityLabel("Cancel Download")
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
