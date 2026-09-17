import MoldClient
import SwiftUI

/// One line when the media cache could not keep a print.
///
/// Its own note rather than a `HostFailure`: no machine failed. The clip
/// arrived, it was handed over, and it is this Mac's cache that has no room to
/// keep it -- which the person can do something about, and which they
/// otherwise only find out by opening the same print again and watching it
/// download again.
struct MediaCacheNote: View {
    let sentence: String
    let dismiss: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "externaldrive.badge.exclamationmark")
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
        .background(.quaternary)
    }
}

extension View {
    /// The same shape `failureBanner` takes, above the pane's own content.
    @ViewBuilder func mediaCacheNote(_ materializer: PrintMaterializer) -> some View {
        VStack(spacing: 0) {
            if let sentence = materializer.note {
                MediaCacheNote(sentence: sentence) { materializer.note = nil }
            }
            self
        }
    }
}
