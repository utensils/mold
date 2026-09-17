import MoldClient
import SwiftUI

/// Whether a model can run right now, and what it would take if not -- the
/// State column's cell. Never a disabled control: every state offers
/// something to press, or says plainly that there is nothing left to do.
struct ModelStateCell: View {
    let model: Model
    let progress: DownloadStore.Progress?
    let install: (Model) -> Void
    /// `nil` off a row that is not mid-download -- there is nothing to
    /// cancel, so no button is drawn rather than a disabled one.
    let cancel: (() -> Void)?

    var body: some View {
        if let progress {
            downloading(progress)
        } else {
            switch model.installState {
            case .loaded:
                Label("Loaded", systemImage: "bolt.fill")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            case .installed:
                Label("Installed", systemImage: "checkmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            case let .needsRepair(remaining):
                // Partly installed is its own state. Calling it "not
                // installed" would hide that most of the bytes are already
                // here (design fact 6, M5: `downloaded && remaining > 0`).
                Button {
                    install(model)
                } label: {
                    Text("\(FileBytes.text(Int64(remaining))) to finish")
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .help("Fetch the rest of this model")
            case .available:
                Button("Install") { install(model) }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }
        }
    }

    /// Bytes rather than a bare percentage: on a 30 GB checkpoint, "18%"
    /// tells you much less than how much is left to come.
    private func downloading(_ progress: DownloadStore.Progress) -> some View {
        HStack(spacing: 6) {
            VStack(alignment: .trailing, spacing: 2) {
                ProgressView(value: progress.fraction ?? 0)
                    .progressViewStyle(.linear)
                    .frame(width: 100)
                if let done = progress.bytesDone, let total = progress.bytesTotal {
                    Text(FileBytes.progress(done: done, total: total))
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                } else {
                    Text("Starting…").font(.caption2).foregroundStyle(.secondary)
                }
            }
            if let cancel {
                Button(action: cancel) {
                    Image(systemName: "xmark.circle.fill")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .help("Cancel this download")
            }
        }
    }
}
