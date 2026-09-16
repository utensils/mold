import AppKit
import MoldClient
import MoldStyle
import SwiftUI

/// The ordered reference pictures a recipe conditions on.
///
/// Order matters: where `primaryIsTarget` is set, the first one is the picture
/// being edited and the rest are references for it — so the strip is numbered
/// rather than a bag.
struct ReferenceStrip: View {
    let capability: ReferenceImagesCapability
    @Binding var draft: RenderDraft

    @State private var targeted = false

    var body: some View {
        HStack(spacing: 6) {
            ForEach(Array(draft.editImages.enumerated()), id: \.offset) { index, encoded in
                well(index: index, encoded: encoded)
            }
            if draft.editImages.count < (capability.maxCount ?? 1) {
                addWell
            }
        }
    }

    private func well(index: Int, encoded: String) -> some View {
        ZStack {
            if let data = Data(base64Encoded: encoded), let image = NSImage(data: data) {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                Chrome.wellFill
            }
        }
        .frame(width: 52, height: 52)
        .clipShape(RoundedRectangle(cornerRadius: Chrome.wellRadius, style: .continuous))
        .overlay(alignment: .topLeading) { badge(index) }
        .overlay(alignment: .topTrailing) { remove(index) }
        .help(label(index))
    }

    @ViewBuilder private func badge(_ index: Int) -> some View {
        if capability.primaryIsTarget {
            Text(index == 0 ? "Target" : "\(index)")
                .font(.caption2)
                .foregroundStyle(.white)
                .padding(.horizontal, 4)
                .padding(.vertical, 1)
                .background(Chrome.badgeBackdrop, in: Capsule())
                .padding(3)
        }
    }

    private func remove(_ index: Int) -> some View {
        Button {
            draft.editImages.remove(at: index)
        } label: {
            Image(systemName: "xmark.circle.fill")
        }
        .buttonStyle(.plain)
        .foregroundStyle(.white, Chrome.badgeBackdrop)
        .padding(2)
        .help("Remove this reference")
    }

    private var addWell: some View {
        RoundedRectangle(cornerRadius: Chrome.wellRadius, style: .continuous)
            .fill(targeted ? Chrome.wellFillTargeted : Chrome.wellFill)
            .frame(width: 52, height: 52)
            .overlay { Image(systemName: "plus").foregroundStyle(.tertiary) }
            .onTapGesture { choose() }
            .dropDestination(for: URL.self) { urls, _ in
                for url in urls { append(url) }
                return true
            } isTargeted: { targeted = $0 }
            .help(capability.primaryIsTarget && draft.editImages.isEmpty
                  ? "Choose the picture to edit"
                  : "Add a reference picture")
    }

    private func label(_ index: Int) -> String {
        guard capability.primaryIsTarget else { return "Reference \(index + 1)" }
        return index == 0 ? "The picture being edited" : "Reference \(index)"
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .webP, .heic, .tiff]
        panel.allowsMultipleSelection = true
        guard panel.runModal() == .OK else { return }
        for url in panel.urls { append(url) }
    }

    private func append(_ url: URL) {
        guard draft.editImages.count < (capability.maxCount ?? 1),
              let data = try? Data(contentsOf: url)
        else { return }
        draft.editImages.append(data.base64EncodedString())
    }
}
