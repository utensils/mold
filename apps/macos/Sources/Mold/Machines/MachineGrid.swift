import CoreGraphics
import MoldClient

/// The shape of the fleet grid, and moving around it with the arrow keys.
///
/// The column count is computed rather than left to `GridItem(.adaptive:)`
/// because the keyboard needs the same answer the layout used: "down" means
/// one ROW down, and a focus model that cannot see the columns can only guess.
/// One function, read by both.
enum MachineGrid {
    /// Wide enough for "4× NVIDIA L40S" and a memory bar beside it without
    /// wrapping, which is the widest line a card draws.
    static let minimumCardWidth: CGFloat = 300
    static let spacing: CGFloat = 12

    /// At least one: below the minimum width the grid is a single column, not
    /// a zero-column layout that draws nothing.
    static func columnCount(for width: CGFloat) -> Int {
        let fits = (width + spacing) / (minimumCardWidth + spacing)
        return max(1, Int(fits.rounded(.down)))
    }

    /// An arrow key, as the grid means it. Its own type rather than SwiftUI's
    /// `MoveCommandDirection` so the movement can be asked without a view.
    enum Move { case up, down, left, right }

    /// Where the focus lands. Nothing focused yet means the first card --
    /// pressing an arrow key with nothing picked should pick something.
    ///
    /// Deliberately no wrapping: left from the first card of a row stays where
    /// it is, rather than jumping to the end of the row above. Down lands on
    /// the LAST card when the row below is a short one, which is where the eye
    /// goes anyway.
    static func move(_ direction: Move, from focused: MoldHost.ID?,
                     in cards: [MachineCard], columns: Int) -> MoldHost.ID? {
        guard let first = cards.first?.id else { return nil }
        guard let index = cards.firstIndex(where: { $0.id == focused }) else { return first }
        let columns = max(1, columns)
        let landing: Int = switch direction {
        case .left: max(index - 1, 0)
        case .right: min(index + 1, cards.count - 1)
        case .up: index < columns ? index : index - columns
        case .down: below(index, of: cards.count, columns: columns)
        }
        return cards[landing].id
    }

    /// The card under this one: the same column on the next row, or the last
    /// card when that row is short. A card already on the last row stays --
    /// sliding sideways is not what "down" means.
    private static func below(_ index: Int, of count: Int, columns: Int) -> Int {
        let straightDown = index + columns
        if straightDown < count { return straightDown }
        let isLastRow = index / columns == (count - 1) / columns
        return isLastRow ? index : count - 1
    }
}
