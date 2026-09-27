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
    /// Read by the canvas rule on every add, replace, move and removal
    /// (`RenderDraft.followLastReference`).
    let recipe: GenerationRecipe
    @Binding var draft: RenderDraft
    /// Pictures the request carries before the strip (`Layout.ordinalBase`).
    var ordinalBase = 0
    /// The last tile sets the canvas shape (`Layout.setsCanvas`).
    var setsCanvas = false
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
            accepting: Self.accepting(capability),
            size: Self.thumbnailSize,
            opensOnClick: false,
            alphaBed: true,
            label: label(index),
            pick: { replace($0, at: index) },
            perform: { perform($0, at: index) })
            .overlay(alignment: .topLeading) { badge(index) }
            .overlay(alignment: .topTrailing) { remove(index) }
            .overlay(alignment: .bottomLeading) { canvasMark(index) }
    }

    /// The same three doors the source well offers, on a `+` that says what it
    /// is rather than being a second anonymous square.
    private var addWell: some View {
        PictureWell(
            rows: GenerateMenus.referenceAdd(canPaste: PicturePaste.hasPicture),
            placeholder: Self.addGlyph,
            accepting: Self.accepting(capability),
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

    /// `Image N`, the prompt's own name for it, then what it is for.
    private func label(_ index: Int) -> String {
        let name = "Image \(Self.ordinal(index: index, base: ordinalBase))"
        if isTarget(index) { return "\(name), the picture being edited" }
        if setsCanvas, index == draft.media.editImages.count - 1 {
            return "\(name), reference, sets the canvas shape"
        }
        return "\(name), reference"
    }

    func isTarget(_ index: Int) -> Bool { capability.primaryIsTarget && index == 0 }
}

extension ReferenceStrip {
    /// The 1-based position the prompt and the expander use ("image 2").
    static func ordinal(index: Int, base: Int) -> Int { base + index + 1 }

    /// What a reference well passes through untouched: the recipe's own
    /// `formats` (the legacy PNG-and-JPEG pair where it names none). Anything
    /// else is converted to PNG on the way in, which keeps alpha -- never
    /// refused after the upload, and an accepted file's bytes are never
    /// re-encoded or flattened.
    static func accepting(_ capability: ReferenceImagesCapability) -> Set<String> {
        Set(capability.acceptedFormats.compactMap(PictureImport.typeIdentifier(forFormat:)))
    }
}
