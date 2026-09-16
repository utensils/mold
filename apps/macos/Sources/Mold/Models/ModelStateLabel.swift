import MoldClient
import SwiftUI

/// Whether a model can run right now, and what it would take if not.
struct ModelStateLabel: View {
    let model: Model
    let progress: DownloadStore.Progress?
    let install: (Model) -> Void

    var body: some View {
        if let progress {
            downloading(progress)
        } else if let remaining = model.repairBytes {
            // Partly installed is its own state. Calling it "not installed"
            // would hide that most of the bytes are already here.
            Button {
                install(model)
            } label: {
                Text("\(remaining.formatted(.byteCount(style: .file))) to finish")
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .help("Fetch the rest of this model")
        } else if model.downloaded == true {
            Label("Installed", systemImage: "checkmark.circle.fill")
                .font(.caption)
                .foregroundStyle(.secondary)
        } else {
            Button("Install") { install(model) }
                .buttonStyle(.bordered)
                .controlSize(.small)
        }
    }

    /// Bytes rather than a bare percentage: on a 30 GB checkpoint, "18% "
    /// tells you much less than how much is left to come.
    private func downloading(_ progress: DownloadStore.Progress) -> some View {
        VStack(alignment: .trailing, spacing: 2) {
            ProgressView(value: progress.fraction ?? 0)
                .progressViewStyle(.linear)
                .frame(width: 120)
            if let done = progress.bytesDone, let total = progress.bytesTotal {
                Text("\(done.formatted(.byteCount(style: .file))) of \(total.formatted(.byteCount(style: .file)))")
                    .font(.caption2)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            } else {
                Text("Starting…").font(.caption2).foregroundStyle(.secondary)
            }
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
    let install: (Model) -> Void
    let progress: DownloadStore.Progress?

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
            ModelStateLabel(model: model, progress: progress, install: install)
                .frame(minWidth: 132, alignment: .trailing)
        }
        .padding(.vertical, 5)
        .help(model.name)
    }
}
