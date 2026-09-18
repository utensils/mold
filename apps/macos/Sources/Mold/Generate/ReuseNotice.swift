import SwiftUI

/// One dismissable line about a print whose own source media could not be
/// brought back, drawn above the pane.
///
/// Shaped like `FailureBanner` on purpose -- it is the same kind of thing: a
/// sentence about something that did not work, which never blocks and never
/// takes the keyboard. It is deliberately NOT a failure on a machine: nothing
/// refused anything, the archive simply does not hold what this print would
/// need, and filing it under a machine's failures would put it in a list a
/// person clears by reconnecting.
struct ReuseNotice: View {
    let sentence: String
    let dismiss: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "photo.badge.exclamationmark")
                .foregroundStyle(.secondary)
            Text(sentence)
                .font(.callout)
                .lineLimit(2)
            Spacer()
            Button(action: dismiss) {
                Image(systemName: "xmark.circle.fill")
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .accessibilityLabel("Dismiss")
            .help("Dismiss this message")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.secondary.opacity(0.12))
    }
}

extension View {
    /// The Generate pane's one call site.
    func reuseNotice(_ reuse: ReuseStore) -> some View {
        VStack(spacing: 0) {
            if let sentence = reuse.notice {
                ReuseNotice(sentence: sentence) { reuse.notice = nil }
            }
            self
        }
    }
}
