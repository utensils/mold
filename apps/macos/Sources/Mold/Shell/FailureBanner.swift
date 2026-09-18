import MoldStyle
import SwiftUI

/// One dismissable line per outstanding failure, drawn above whatever a pane
/// shows.
///
/// Never modal, and never blocking: a machine failing at one thing has no
/// bearing on whether the other machines' rows still work, so nothing here
/// should stop you from looking at them.
struct FailureBanner: View {
    let failures: [HostFailure]
    let dismiss: (HostFailure) -> Void

    var body: some View {
        ForEach(failures) { failure in
            // On the FIRST line's baseline, and on the toolbar's own trailing
            // inset: a two-line sentence used to centre the row, which left
            // the dismiss button floating a half-line below the toolbar
            // control it lines up under (the owner's screenshot, 2026-09-17).
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                Text(failure.sentence)
                    .font(.callout)
                    .lineLimit(2)
                Spacer()
                Button {
                    dismiss(failure)
                } label: {
                    Image(systemName: "xmark.circle.fill")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .accessibilityLabel("Dismiss")
                .help("Dismiss this message")
            }
            .padding(.horizontal, Chrome.toolbarEdgeInset)
            .padding(.vertical, 6)
            .background(.orange.opacity(0.12))
        }
    }
}

extension View {
    /// One line at each pane's call site: Library, Queue and Models each show
    /// what a machine could not do, above their own content.
    func failureBanner(_ hosts: HostStore) -> some View {
        VStack(spacing: 0) {
            FailureBanner(failures: hosts.failures, dismiss: hosts.dismiss)
            // The pane takes what is left, so the banner is pinned UNDER the
            // toolbar. Without it the stack shrank to its content and the
            // whole thing floated down the middle of the pane -- which is
            // what "at an odd offset below the toolbar" was.
            self.frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}
