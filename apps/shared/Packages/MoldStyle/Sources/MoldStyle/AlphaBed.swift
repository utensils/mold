import SwiftUI

/// The checkerboard drawn behind a picture that carries alpha -- and behind
/// every reference thumbnail -- so a transparent area reads as transparent
/// rather than as the colour of whatever happens to be under it.
///
/// Port of the web kit's `.ms-alpha-bed` (`ui/kit.css`): 8-point squares in
/// two tones DERIVED from the bed mixed toward the text colour, 16% and 6%.
/// Drawn in `.primary` at those opacities, so light and dark appearance each
/// get a readable board with no palette of its own. Draw it on a box sized
/// to the MEDIA, never on a letterbox, so the bars around a fitted picture
/// stay plain.
public struct AlphaBed: View {
    /// One square's edge -- the kit's 16px tile holds four of them.
    public static let cell: CGFloat = 8
    public static let lightOpacity = 0.16
    public static let darkOpacity = 0.06

    public init() {}

    public var body: some View {
        Canvas { context, size in
            let bounds = CGRect(origin: .zero, size: size)
            var light = Path()
            var dark = Path()
            for square in Self.squares(in: bounds) {
                if square.isLight { light.addRect(square.rect) } else { dark.addRect(square.rect) }
            }
            context.fill(dark, with: .color(.primary.opacity(Self.darkOpacity)))
            context.fill(light, with: .color(.primary.opacity(Self.lightOpacity)))
        }
        .accessibilityHidden(true)
    }

    /// Every square covering `rect`, clipped to it, starting light at its
    /// top-left corner. Shared with the AppKit viewer, which paints the same
    /// board under an `NSImageView`.
    public static func squares(in rect: CGRect, cell: CGFloat = cell) -> [(rect: CGRect, isLight: Bool)] {
        guard rect.width > 0, rect.height > 0, cell > 0 else { return [] }
        let columns = Int((rect.width / cell).rounded(.up))
        let rows = Int((rect.height / cell).rounded(.up))
        var out: [(rect: CGRect, isLight: Bool)] = []
        out.reserveCapacity(columns * rows)
        for row in 0 ..< rows {
            for column in 0 ..< columns {
                let square = CGRect(x: rect.minX + CGFloat(column) * cell,
                                    y: rect.minY + CGFloat(row) * cell,
                                    width: cell, height: cell).intersection(rect)
                out.append((square, (row + column).isMultiple(of: 2)))
            }
        }
        return out
    }

    /// Where an aspect-fitted picture of `content` lands, centred in
    /// `bounds` -- `NSImageView`'s `.scaleProportionallyUpOrDown` with
    /// `.alignCenter` -- so the board under it is exactly the picture's.
    public static func fittedRect(content: CGSize, in bounds: CGRect) -> CGRect {
        guard content.width > 0, content.height > 0 else { return .zero }
        let scale = min(bounds.width / content.width, bounds.height / content.height)
        let size = CGSize(width: content.width * scale, height: content.height * scale)
        return CGRect(x: bounds.midX - size.width / 2, y: bounds.midY - size.height / 2,
                      width: size.width, height: size.height)
    }
}
