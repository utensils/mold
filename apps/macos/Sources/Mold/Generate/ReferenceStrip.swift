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

    @Environment(HostStore.self) private var hosts
    @Environment(LibraryStore.self) private var library
    @State private var targeted = false
    @State private var showsLibrary = false

    var body: some View {
        HStack(spacing: 6) {
            ForEach(Array(draft.media.editImages.enumerated()), id: \.offset) { index, encoded in
                well(index: index, encoded: encoded)
            }
            if draft.media.editImages.count < (capability.maxCount ?? 1) {
                addWell
            }
        }
    }

    private func well(index: Int, encoded: String) -> some View {
        ReferenceWell(encoded: encoded)
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
            draft.media.editImages.remove(at: index)
        } label: {
            Image(systemName: "xmark.circle.fill")
        }
        .buttonStyle(.plain)
        .foregroundStyle(.white, Chrome.badgeBackdrop)
        .padding(2)
        .help("Remove this reference")
    }

    /// The same "Choose File…" / "From Library…" menu the source well
    /// offers (M8 decision 5), reusing `PictureSource` for both.
    private var addWell: some View {
        Menu {
            Button("Choose File…", action: chooseFile)
            Button("From Library…") { showsLibrary = true }
        } label: {
            RoundedRectangle(cornerRadius: Chrome.wellRadius, style: .continuous)
                .fill(targeted ? Chrome.wellFillTargeted : Chrome.wellFill)
                .frame(width: 52, height: 52)
                .overlay { Image(systemName: "plus").foregroundStyle(.tertiary) }
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .dropDestination(for: PictureDrop.self) { drops, _ in
            for drop in drops { handle(drop) }
            return true
        } isTargeted: { targeted = $0 }
        .help(addWellLabel)
        .accessibilityLabel(addWellLabel)
        .sheet(isPresented: $showsLibrary) {
            LibraryPickerSheet { entry, data in append((data, entry.print.filename)) }
        }
    }

    /// The one sentence the add well's tooltip and its VoiceOver label share.
    private var addWellLabel: String {
        capability.primaryIsTarget && draft.media.editImages.isEmpty
            ? "Choose the picture to edit"
            : "Add a reference picture"
    }

    private func label(_ index: Int) -> String {
        guard capability.primaryIsTarget else { return "Reference \(index + 1)" }
        return index == 0 ? "The picture being edited" : "Reference \(index)"
    }

    private func handle(_ drop: PictureDrop) {
        Task {
            do {
                append(try await PictureSource.bytes(of: drop, hosts: hosts, library: library))
            } catch {
                if case let .print(id) = drop {
                    hosts.report(error, on: id.host, doing: "fetch that picture")
                }
            }
        }
    }

    private func chooseFile() {
        guard let url = PictureSource.chooseFile(), let data = try? Data(contentsOf: url) else { return }
        append((data, url.lastPathComponent))
    }

    private func append(_ picked: (data: Data, name: String)) {
        guard draft.media.editImages.count < (capability.maxCount ?? 1) else { return }
        draft.media.editImages.append(picked.data.base64EncodedString())
    }
}

/// One reference thumbnail, decoded once per encoded string rather than once
/// per keystroke -- `body` re-runs on every draft edit (a slider drag fires
/// many), and `Data(base64Encoded:)` plus `NSImage(data:)` were both inside
/// it. The pattern `RunCanvas` uses for its own preview and result images.
private struct ReferenceWell: View {
    let encoded: String

    @State private var image: NSImage?

    var body: some View {
        ZStack {
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                Chrome.wellFill
            }
        }
        .frame(width: 52, height: 52)
        .clipShape(RoundedRectangle(cornerRadius: Chrome.wellRadius, style: .continuous))
        .task(id: encoded) {
            image = Data(base64Encoded: encoded).flatMap(NSImage.init(data:))
        }
    }
}
