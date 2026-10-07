import MoldClient
import MoldStyle
import SwiftUI

// What one reference and the strip's own background offer, rendered from
// `GenerateMenus` so a contextual menu and the inline controls beside it can
// never drift apart. Split from the strip's shape purely for size.
extension ReferenceStrip {
    /// One reference's own menu. ORDER matters: on a `primaryIsTarget` recipe
    /// index 0 is the picture being edited, so Move Left / Move Right are real
    /// instructions (`GenerateMenus.referenceItem`).
    func itemMenu(_ index: Int) -> [GenerateMenus.Row] {
        GenerateMenus.referenceItem(index: index, count: draft.media.editImages.count)
    }

    /// The strip's background. Add and Paste live on the add well itself, one
    /// square away, so this is about the strip as a whole.
    var stripMenu: [GenerateMenus.Row] {
        GenerateMenus.referenceStrip(count: draft.media.editImages.count)
    }

    /// Only the rows the chooser does not own reach here -- the ORDER, and the
    /// removals.
    func perform(_ action: GenerateAction, at index: Int?) {
        switch action {
        case .moveLeft:
            guard let index, index > 0 else { return }
            DraftPictureAttachment.moveReference(from: index, to: index - 1, in: &draft, recipe: recipe)
        case .moveRight:
            guard let index, index < draft.media.editImages.count - 1 else { return }
            DraftPictureAttachment.moveReference(from: index, to: index + 1, in: &draft, recipe: recipe)
        case .removeReference:
            guard let index, draft.media.editImages.indices.contains(index) else { return }
            DraftPictureAttachment.removeReference(at: index, from: &draft, recipe: recipe)
        case .removeAllReferences:
            draft.media.editImages.removeAll()
        default:
            break
        }
        // Order is the canvas on a `last-reference` recipe: a move or a
        // removal that changes the last picture reshapes a default canvas.
        draft.followLastReference(recipe: recipe)
    }

    /// Every tile is numbered the way the prompt and the expander address it
    /// ("image 2"), because order is part of the request; a target-first
    /// recipe's first picture says Target beside its number.
    func badge(_ index: Int) -> some View {
        let ordinal = Self.ordinal(index: index, base: ordinalBase)
        return Text(isTarget(index) ? "\(ordinal) Target" : "\(ordinal)")
            .font(.caption2.monospacedDigit())
            .foregroundStyle(.white)
            .padding(.horizontal, 4)
            .padding(.vertical, 1)
            .background(Chrome.badgeBackdrop, in: Capsule())
            .padding(3)
            .accessibilityHidden(true)
    }

    /// The last tile of a `last-reference` strip: this picture's shape is
    /// the canvas unless a size is picked.
    // a11y: decoration -- the tile's own label says it sets the canvas.
    @ViewBuilder func canvasMark(_ index: Int) -> some View {
        if setsCanvas, index == draft.media.editImages.count - 1 {
            Image(systemName: "aspectratio")
                .font(.caption2)
                .foregroundStyle(.white)
                .padding(3)
                .background(Chrome.badgeBackdrop, in: Circle())
                .padding(3)
                .help("Use this reference picture’s shape for the new image")
                .accessibilityHidden(true)
        }
    }

    func remove(_ index: Int) -> some View {
        Button {
            guard draft.media.editImages.indices.contains(index) else { return }
            DraftPictureAttachment.removeReference(at: index, from: &draft, recipe: recipe)
            draft.followLastReference(recipe: recipe)
        } label: {
            Image(systemName: "xmark.circle.fill")
        }
        .buttonStyle(.plain)
        .foregroundStyle(.white, Chrome.badgeBackdrop)
        .padding(2)
        .help("Remove this reference")
    }
}
