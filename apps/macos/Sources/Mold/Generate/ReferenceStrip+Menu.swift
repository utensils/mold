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
            draft.media.editImages.swapAt(index, index - 1)
        case .moveRight:
            guard let index, index < draft.media.editImages.count - 1 else { return }
            draft.media.editImages.swapAt(index, index + 1)
        case .removeReference:
            guard let index, draft.media.editImages.indices.contains(index) else { return }
            draft.media.editImages.remove(at: index)
        case .removeAllReferences:
            draft.media.editImages.removeAll()
        default:
            break
        }
    }

    @ViewBuilder func badge(_ index: Int) -> some View {
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

    func remove(_ index: Int) -> some View {
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
}
