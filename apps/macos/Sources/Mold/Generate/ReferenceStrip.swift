import AppKit
import MoldClient
import MoldStyle
import SwiftUI

/// The ordered reference pictures a recipe conditions on.
///
/// Order matters: where `primaryIsTarget` is set, the first one is the picture
/// being edited and the rest are references for it — so the strip is numbered
/// rather than a bag. Every square in it is a `PictureWell`: the add well
/// chooses, and a staged one is replaced from the same two doors.
struct ReferenceStrip: View {
    let capability: ReferenceImagesCapability
    @Binding var draft: RenderDraft
    /// What the strip is, under it. Passed in because only
    /// `ImageConditioningWells` knows whether this recipe has PARKED it.
    var caption: String?

    /// A staged reference is smaller than a well you pick into -- four of them
    /// sit beside the prompt.
    static let thumbnailSize: CGFloat = 52
    /// Deliberately NOT the source well's glyph: the two squares sit side by
    /// side on every `combines` recipe, and the owner could not tell them
    /// apart.
    static let addGlyph = "plus"

    var body: some View {
        VStack(alignment: .leading, spacing: WellCaption.spacing) {
            HStack(alignment: .top, spacing: 6) {
                ForEach(Array(draft.media.editImages.enumerated()), id: \.offset) { index, encoded in
                    well(index: index, encoded: encoded)
                }
                if capability.hasRoom(for: draft.media.editImages.count) {
                    addWell
                }
            }
            if let caption { WellCaption.text(caption) }
        }
        .rowActionMenu(stripMenu) { perform($0, at: nil) }
    }

    /// A staged reference keeps its inline controls -- the order badge and the
    /// ✕ -- so it does not open its menu on a plain click: a `Menu` label
    /// swallows the taps those need. Its contextual menu and its drop are the
    /// chooser's, like every other well.
    private func well(index: Int, encoded: String) -> some View {
        PictureWell(
            rows: itemMenu(index),
            picture: encoded,
            size: Self.thumbnailSize,
            opensOnClick: false,
            label: label(index),
            pick: { replace($0, at: index) },
            perform: { perform($0, at: index) })
            .overlay(alignment: .topLeading) { badge(index) }
            .overlay(alignment: .topTrailing) { remove(index) }
    }

    /// The same three doors the source well offers, on a `+` that says what it
    /// is rather than being a second anonymous square.
    private var addWell: some View {
        PictureWell(
            rows: GenerateMenus.referenceAdd(canPaste: PicturePaste.hasPicture),
            placeholder: Self.addGlyph,
            allowsMultiple: true,
            size: Self.thumbnailSize,
            label: addWellLabel,
            pick: append)
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
