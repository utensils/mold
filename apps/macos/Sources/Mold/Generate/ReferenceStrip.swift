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

    @Environment(HostStore.self) var hosts
    @Environment(LibraryStore.self) var library
    /// Not `private`: `ReferenceStrip+Import`, an extension in another file,
    /// owns getting a picture in and the menus that ask for one.
    @State var targeted = false
    @State var showsLibrary = false
    /// What a file the engine cannot read said, beside the control that
    /// collected it rather than in a 422 after the upload (finding 02#7).
    @State var importFailure: String?
    /// The one import in flight. Cancelled by the next pick, and the reason a
    /// multi-file drop lands in DROP order.
    @State var importTask: Task<Void, Never>?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                ForEach(Array(draft.media.editImages.enumerated()), id: \.offset) { index, encoded in
                    well(index: index, encoded: encoded)
                }
                if capability.hasRoom(for: draft.media.editImages.count) {
                    addWell
                }
            }
            if let importFailure {
                Text(importFailure)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: 160, alignment: .leading)
            }
        }
        .contextMenu { stripMenu }
    }

    private func well(index: Int, encoded: String) -> some View {
        ReferenceWell(encoded: encoded)
            .overlay(alignment: .topLeading) { badge(index) }
            .overlay(alignment: .topTrailing) { remove(index) }
            .help(label(index))
            .contextMenu { itemMenu(index) }
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
            handle(drops)
            return true
        } isTargeted: { targeted = $0 }
        .help(addWellLabel)
        .accessibilityLabel(addWellLabel)
        .sheet(isPresented: $showsLibrary) {
            LibraryPickerSheet(pick: append)
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
}
