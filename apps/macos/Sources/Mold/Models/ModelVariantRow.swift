import MoldClient
import MoldStyle
import SwiftUI

/// One variant of a model: what it costs you and what you get for it.
struct ModelVariantRow: View {
    let model: Model
    let groupTitle: String

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
            ModelStateLabel(model: model)
                .frame(minWidth: 108, alignment: .trailing)
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

/// Whether a model can run right now, and what it would take if not.
struct ModelStateLabel: View {
    let model: Model

    var body: some View {
        if let remaining = model.repairBytes {
            // Partly installed is its own state. Calling it "not installed"
            // would hide that most of the bytes are already here.
            Label(
                "\(remaining.formatted(.byteCount(style: .file))) to finish",
                systemImage: "exclamationmark.arrow.trianglehead.2.clockwise.rotate.90"
            )
            .font(.caption)
            .foregroundStyle(.secondary)
        } else if model.downloaded == true {
            Label("Installed", systemImage: "checkmark.circle.fill")
                .font(.caption)
                .foregroundStyle(.secondary)
        } else {
            Text("Not installed")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
    }
}

/// The heading for a model's variants.
struct ModelGroupHeader: View {
    let title: String
    let repo: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            HStack(alignment: .firstTextBaseline) {
                Text(title).font(.headline)
                Spacer()
                if let repo {
                    Text(repo)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                        .truncationMode(.head)
                }
            }
        }
        .padding(.top, 6)
    }
}

/// A model that has exactly one, untagged variant: its name IS the row.
struct ModelSoloRow: View {
    let model: Model
    let title: String

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.headline)
                if let tradeOff = model.tradeOff {
                    Text(tradeOff).font(.caption).foregroundStyle(.secondary)
                }
            }
            Spacer(minLength: 12)
            if let size = model.sizeGb {
                Text(size.formatted(.number.precision(.fractionLength(1))) + " GB")
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            ModelStateLabel(model: model).frame(minWidth: 108, alignment: .trailing)
        }
        .padding(.vertical, 5)
        .help(model.name)
    }
}
