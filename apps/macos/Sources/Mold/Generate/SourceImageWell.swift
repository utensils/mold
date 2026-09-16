import AppKit
import MoldClient
import MoldStyle
import SwiftUI
import UniformTypeIdentifiers

/// The still a render is conditioned on.
///
/// Only shown when the recipe says it reads one. A well on a text-only model
/// would collect bytes the host is going to refuse.
struct SourceImageWell: View {
    @Binding var draft: RenderDraft

    @State private var targeted = false
    @State private var preview: NSImage?

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: Chrome.wellRadius, style: .continuous)
                .fill(targeted ? Chrome.wellFillTargeted : Chrome.wellFill)
            if let preview {
                Image(nsImage: preview)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .clipShape(RoundedRectangle(cornerRadius: Chrome.wellRadius,
                                                style: .continuous))
            } else {
                Image(systemName: "photo")
                    .foregroundStyle(.tertiary)
            }
        }
        .frame(width: 64, height: 64)
        .overlay(alignment: .topTrailing) { clearButton }
        .onTapGesture { choose() }
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first else { return false }
            load(url)
            return true
        } isTargeted: { targeted = $0 }
        .help(draft.sourceImageName ?? "Drop a picture, or click to choose one")
        .accessibilityLabel("Source picture")
    }

    @ViewBuilder private var clearButton: some View {
        if draft.sourceImage != nil {
            Button {
                draft.sourceImage = nil
                draft.sourceImageName = nil
                preview = nil
            } label: {
                Image(systemName: "xmark.circle.fill")
            }
            .buttonStyle(.plain)
            .foregroundStyle(.white, Chrome.badgeBackdrop)
            .padding(3)
            .help("Remove this picture")
        }
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .webP, .heic, .tiff]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        load(url)
    }

    private func load(_ url: URL) {
        guard let data = try? Data(contentsOf: url) else { return }
        // mold takes every byte field as base64 on the wire, so the encode
        // happens here rather than at request time -- the draft holds exactly
        // what will be sent.
        draft.sourceImage = data.base64EncodedString()
        draft.sourceImageName = url.lastPathComponent
        preview = NSImage(data: data)
    }
}
