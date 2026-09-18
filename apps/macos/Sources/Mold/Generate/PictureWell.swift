import AppKit
import MoldClient
import MoldStyle
import SwiftUI

/// THE picture chooser: every well in the app that takes a picture is this
/// view with different closures.
///
/// The owner's ask, twice over (2026-09-17): "Identity should use the same
/// intuitive selector that allows for library or local. This should be true of
/// any source image reference. We should use the same component/selector logic
/// across everything." There were four selectors -- the source well's whole
/// idiom, the strip's copy of half of it, the ControlNet well's tap-to-open
/// panel, and the identity group's `NSOpenPanel` and nothing else -- so a
/// photograph already in the fleet could be a source but not a face.
///
/// This owns the doors: the click menu with its chevron badge, the IDENTICAL
/// contextual menu, the drop target and its targeted fill, the Library sheet,
/// the open panel, Paste's availability, the single-flight import and the
/// failure sentence beside the well. What stays in each caller is the only
/// thing that really differs: WHERE a picked picture goes.
struct PictureWell: View {
    /// What this well offers, declared in `GenerateMenus` so the click menu,
    /// the contextual menu and a test all read one list.
    let rows: [GenerateMenus.Row]
    /// The picture it is holding, base64 exactly as it will be sent. `nil`
    /// draws the placeholder glyph.
    ///
    /// The preview follows THIS rather than the well's own load path: a
    /// picture can arrive from a parked restore, a Reuse or the UAT seed, and
    /// a well that only previews what it loaded itself drew the placeholder
    /// over a picture that was really there.
    var picture: String?
    /// The glyph for an empty well. Different per well ON PURPOSE: the source
    /// still and the strip's add well sit side by side.
    var placeholder = SourceImageWell.placeholderGlyph
    /// What the machine can read from this well, applied to every door.
    var accepting: Set<String> = PictureImport.engineReadable
    /// Whether one pick may bring several pictures -- true for the wells that
    /// feed a list, false where a pick replaces one slot.
    var allowsMultiple = false
    var size: CGFloat = Self.standardSize
    /// Whether a plain click opens the menu. False for a staged picture, whose
    /// square carries its own inline controls (a ✕, an order badge) that a
    /// `Menu` label would swallow -- it keeps the contextual menu and the drop.
    var opensOnClick = true
    /// The word under the well saying what it is for, or `nil` where the
    /// caller captions a whole group of wells itself.
    var caption: String?
    /// The VoiceOver name, and the tooltip when `help` says nothing more.
    let label: String
    var help: String?
    /// `MOLD_NATIVE_LIBRARY_PICKER=1` opens the sheet once at launch so a UAT
    /// run can photograph it without driving a menu press. One well answers
    /// yes -- the source still.
    var seedsLibraryPicker = false
    /// Where a picked picture goes. Called once per picture, in pick order.
    let pick: (ImportedPicture) -> Void
    /// The rows this well's owner performs itself -- a mask, a removal, a
    /// reorder. The chooser routes only the rows that open a door.
    var perform: (GenerateAction) -> Void = { _ in }

    @Environment(HostStore.self) var hosts
    @Environment(LibraryStore.self) var library
    /// Not `private`: `PictureWell+Shape` draws with these and
    /// `PictureWell+Import` owns the doors and the single-flight import behind
    /// them -- both extensions in files of their own.
    @State var targeted = false
    @State var preview: NSImage?
    @State var showsLibrary = false
    @State var importFailure: String?
    @State var importTask: Task<Void, Never>?

    /// The wells beside the prompt. A staged picture in a strip is smaller
    /// (`ReferenceStrip.thumbnailSize`); this is the size of a well you pick
    /// INTO.
    static let standardSize: CGFloat = 64

    var body: some View {
        VStack(spacing: WellCaption.spacing) {
            square
            if let caption { WellCaption.text(caption) }
            if let importFailure {
                Text(importFailure)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: 140, alignment: .leading)
            }
        }
    }

    @ViewBuilder private var square: some View {
        clickable
            // The click menu and the contextual menu are the SAME list, and
            // the drop is the third door onto the same pipeline.
            .rowActionMenu(rows, perform: route)
            .dropDestination(for: PictureDrop.self) { drops, _ in
                handle(allowsMultiple ? drops : Array(drops.prefix(1)))
                return true
            } isTargeted: { targeted = $0 }
            .help(help ?? label)
            .accessibilityLabel(label)
            // Decoded off the main actor -- a 50 MB still is a visible stall.
            .task(id: picture) { preview = await PicturePreview.decode(picture) }
            .task { seedLibraryPickerIfRequested() }
            .sheet(isPresented: $showsLibrary) {
                LibraryPickerSheet(accepting: accepting, pick: deliver)
            }
    }

    private func seedLibraryPickerIfRequested() {
        guard seedsLibraryPicker, !showsLibrary, GenerateUAT.wantsLibraryPicker() else { return }
        showsLibrary = true
    }
}
