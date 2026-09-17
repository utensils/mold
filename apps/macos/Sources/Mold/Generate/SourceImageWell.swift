import AppKit
import MoldClient
import MoldStyle
import SwiftUI

/// The still a render is conditioned on.
///
/// Only shown when the recipe says it reads one. A well on a text-only model
/// would collect bytes the host is going to refuse. Click opens a menu
/// (M8 decision 5): a file, a Library print, or Remove; a drop takes either
/// a Finder file or a print dragged out of the Library, through the shared
/// `PictureDrop`/`PictureSource` the reference strip also uses.
struct SourceImageWell: View {
    @Binding var draft: RenderDraft

    @Environment(HostStore.self) var hosts
    @Environment(LibraryStore.self) var library
    @State private var targeted = false
    @State var preview: NSImage?
    @State private var showsLibrary = false
    /// What a file the engine cannot read said, beside the control that
    /// collected it rather than in a 422 after the upload (finding 02#7).
    @State var importFailure: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            menu
            if let importFailure {
                Text(importFailure)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: 140, alignment: .leading)
            }
        }
    }

    private var menu: some View {
        Menu {
            menuItems
        } label: {
            well
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .dropDestination(for: PictureDrop.self) { drops, _ in
            guard let drop = drops.first else { return false }
            handle(drop)
            return true
        } isTargeted: { targeted = $0 }
        .help("Drop a picture or a Library print, or click to choose one")
        // The preview follows the DRAFT, not this well's own load path: a
        // source can arrive from a parked restore, a Reuse, or the UAT seed,
        // and a well that only previews what it loaded itself showed the
        // placeholder glyph over a picture that was really there. Decoded off
        // the main actor -- a 50 MB still is a visible stall otherwise.
        .task(id: draft.media.sourceImage) {
            preview = await PicturePreview.decode(draft.media.sourceImage)
        }
        .task { seedLibraryPickerIfRequested() }
        .accessibilityLabel("Source picture")
        .sheet(isPresented: $showsLibrary) {
            LibraryPickerSheet(pick: apply)
        }
    }

    @ViewBuilder private var menuItems: some View {
        let items = SourceImageWell.menuItems(hasPicture: draft.media.sourceImage != nil)
        Button(items[0], action: chooseFile)
        Button(items[1]) { showsLibrary = true }
        if items.count > 2 {
            Divider()
            Button(items[2], role: .destructive, action: clear)
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
                Image(systemName: "photo.badge.plus")
                    .foregroundStyle(.tertiary)
            }
        }
        .frame(width: 64, height: 64)
        .overlay(alignment: .bottomTrailing) { menuBadge }
    }

    /// Reads as a menu without a full disclosure triangle taking up room in
    /// a 64pt well.
    private var menuBadge: some View {
        Image(systemName: "chevron.down")
            .font(.caption2)
            .foregroundStyle(.white)
            .padding(4)
            .background(Chrome.badgeBackdrop, in: Circle())
            .padding(3)
    }

    /// `MOLD_NATIVE_LIBRARY_PICKER=1` opens the sheet once at launch, the
    /// way `GeneratePane.seedSourceImageIfRequested` seeds a source picture
    /// -- so a UAT run can photograph it without driving a menu press.
    private func seedLibraryPickerIfRequested() {
        guard !showsLibrary, GenerateUAT.wantsLibraryPicker() else { return }
        showsLibrary = true
    }
}

extension SourceImageWell {
    /// What the menu offers -- pure, so the row itself is tested with no
    /// view needed.
    static func menuItems(hasPicture: Bool) -> [String] {
        hasPicture ? ["Choose File…", "From Library…", "Remove"] : ["Choose File…", "From Library…"]
    }
}
