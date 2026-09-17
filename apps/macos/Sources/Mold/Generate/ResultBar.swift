import AppKit
import MoldClient
import SwiftUI

/// What you can do with the picture that just came out.
///
/// The print is already in the machine's gallery by the time it appears here,
/// so these act on the stored file rather than on anything held in memory --
/// which is also why "Show in Library" can simply go and find it.
struct ResultBar: View {
    let result: BatchResult
    let host: MoldHost?
    /// The SAME closures the contextual menu performs -- one definition of
    /// each verb, two surfaces rendering it.
    let actions: ResultActions

    var body: some View {
        HStack(spacing: 8) {
            Button { actions.save(result) } label: {
                Label("Save a Copy", systemImage: "square.and.arrow.down")
            }
            .disabled(host == nil)

            Button { actions.copy(result) } label: {
                Label("Copy", systemImage: "doc.on.doc")
            }
            .disabled(host == nil)

            Button(action: actions.showInLibrary) {
                Label("Show in Library", systemImage: "photo.on.rectangle.angled")
            }

            if let seed = result.seed {
                Spacer(minLength: 12)
                Text("seed \(String(seed))")
                    .font(.caption)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .resultContextMenu(result, actions: actions)
    }
}
