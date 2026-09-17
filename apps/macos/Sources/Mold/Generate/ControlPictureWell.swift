import AppKit
import MoldClient
import MoldStyle
import SwiftUI
import UniformTypeIdentifiers

/// The still that steers a ControlNet render, bound to `draft.media.control`.
///
/// A second instance of `SourceImageWell`'s idiom rather than a genericized
/// one: the two wells bind to different halves of the draft (`sourceImage`/
/// `sourceImageName` directly, vs `control?.image`/`control?.name` behind an
/// optional whose OTHER half -- the chosen adapter -- can already be set),
/// and folding that asymmetry into one generic view would cost more lines
/// than it saves for a single second caller.
struct ControlPictureWell: View {
    @Binding var draft: RenderDraft

    @State private var targeted = false
    @State private var preview: NSImage?
    /// What a file the engine cannot read said, beside the control that
    /// collected it rather than in a 422 after the upload (finding 02#7).
    @State private var importFailure: String?
    /// The one import in flight, cancelled by the next pick so an older,
    /// slower file can never overwrite a newer one.
    @State private var importTask: Task<Void, Never>?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            well
            if let importFailure {
                Text(importFailure)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: 140, alignment: .leading)
            }
        }
    }

    private var well: some View {
        ZStack {
            RoundedRectangle(cornerRadius: Chrome.wellRadius, style: .continuous)
                .fill(targeted ? Chrome.wellFillTargeted : Chrome.wellFill)
            if let preview {
                Image(nsImage: preview)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .clipShape(RoundedRectangle(cornerRadius: Chrome.wellRadius, style: .continuous))
            } else {
                Image(systemName: "wand.and.rays")
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
        .help(draft.media.control?.name ?? "Drop a control picture, or click to choose one")
        .accessibilityLabel("Control picture")
    }

    @ViewBuilder private var clearButton: some View {
        if draft.media.control?.image != nil {
            Button {
                draft.media.control?.image = nil
                draft.media.control?.name = nil
                preview = nil
                if draft.media.control?.model == nil { draft.media.control = nil }
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

    /// Through the one door: read, conform and encode OFF the main actor --
    /// the panel offers HEIC, the format every iPhone photograph arrives in,
    /// and nothing behind mold reads it, so it is transcoded here rather
    /// than uploaded whole and refused (findings 02#7 and 02#10).
    private func load(_ url: URL) {
        importTask?.cancel()
        importTask = Task {
            do {
                let picked = try await PictureImport.load(
                    url, accepting: PictureImport.engineReadable)
                guard !Task.isCancelled else { return }
                var control = draft.media.control ?? ControlConditioning()
                control.image = picked.encoded
                control.name = picked.name
                draft.media.control = control
                preview = NSImage(data: picked.data)
                importFailure = nil
            } catch is CancellationError {
                return
            } catch {
                importFailure = error.reasonSentence
            }
        }
    }
}
