import MoldClient
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
    /// The attachment line says what WILL ride along; the notice line says
    /// what could not.
    var glyph = "photo.badge.exclamationmark"
    let dismiss: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: glyph)
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
    ///
    /// TWO lines, and the order matters: what WILL happen first, then what
    /// could not. The available path used to say nothing at all -- a person
    /// was told when a print's picture would not come back and never when it
    /// would, which is the disclosure exactly inverted, and it is what let a
    /// render nobody associated with that print be silently conditioned on
    /// it. The ✕ is the only way to put it down by hand.
    func reuseNotice(_ reuse: ReuseStore, draft: RenderDraft) -> some View {
        VStack(spacing: 0) {
            if let attachment = reuse.attachmentSentence(for: draft) {
                ReuseNotice(sentence: attachment, glyph: "photo.on.rectangle.angled",
                            dismiss: { reuse.clear() })
            }
            if let sentence = reuse.notice {
                ReuseNotice(sentence: sentence) { reuse.notice = nil }
            }
            self
        }
    }
}
