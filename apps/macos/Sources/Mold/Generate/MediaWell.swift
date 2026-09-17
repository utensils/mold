import AppKit
import MoldClient
import MoldStyle
import SwiftUI
import UniformTypeIdentifiers

/// A generic file well for the clip's non-picture inputs -- a continuation
/// clip, conditioning audio, a source video, a keyframe still.
///
/// NOT a refactor of `SourceImageWell`/`ControlPictureWell`: those draw a
/// picture preview, and everything this well takes (a clip, audio, or a
/// keyframe still shown only by its filename) draws a FILENAME instead. The
/// structural audit already refuted merging the two picture wells for the
/// same reason -- see `ControlPictureWell`'s own doc comment.
struct MediaWell: View {
    /// What one well is holding: the base64 bytes and the name to show for
    /// them. Both travel together so a caller can never end up with one set
    /// and not the other.
    struct Attachment: Equatable {
        var base64: String
        var name: String
    }

    let systemImage: String
    let allowedTypes: [UTType]
    let placeholder: String
    @Binding var attachment: Attachment?

    @State private var targeted = false

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: systemImage)
                .foregroundStyle(.secondary)
                .frame(width: 16)
            Text(attachment?.name ?? placeholder)
                .font(.callout)
                .foregroundStyle(attachment == nil ? .tertiary : .primary)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer(minLength: 0)
            if attachment != nil {
                Button {
                    attachment = nil
                } label: {
                    Image(systemName: "xmark.circle.fill")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .help("Remove")
            }
        }
        .padding(.horizontal, 8)
        .frame(height: Chrome.fieldHeight)
        .background(
            RoundedRectangle(cornerRadius: Chrome.fieldRadius, style: .continuous)
                .fill(targeted ? Chrome.wellFillTargeted : Chrome.wellFill)
        )
        .onTapGesture { choose() }
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first else { return false }
            load(url)
            return true
        } isTargeted: { targeted = $0 }
        .help(attachment?.name ?? "Drop a file, or click to choose one")
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = allowedTypes
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        load(url)
    }

    private func load(_ url: URL) {
        guard let data = try? Data(contentsOf: url) else { return }
        // mold takes every byte field as base64 on the wire, so the encode
        // happens here rather than at request time.
        attachment = Attachment(base64: data.base64EncodedString(), name: url.lastPathComponent)
    }
}
