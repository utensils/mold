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
            HStack(spacing: 8) {
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
            }
            .padding(.horizontal, 12)
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
            self
        }
    }
}
