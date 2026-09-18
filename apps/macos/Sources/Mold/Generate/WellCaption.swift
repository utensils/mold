import SwiftUI

/// What a picture well is FOR, in a word or two, under the well itself.
///
/// The prompt bar drew two anonymous rounded squares side by side on every
/// SD1.5 and SDXL recipe -- the source still and the reference strip's empty
/// add well, which `combines` keeps live at the same time -- with nothing but
/// a glyph telling them apart. And a PARKED well was distinguished by opacity
/// alone, which is not an explanation: it says something is different, never
/// what or why.
///
/// Pure strings drawn in a fixed column, so a test MEASURES them rather than
/// trusting them: a caption that wraps grows the whole bar by a line.
enum WellCaption {
    static func source(parked: Bool) -> String { caption("Source", parked: parked) }

    /// An ABSENT cap is UNBOUNDED, the way studio reads it -- so it is the
    /// plural, like every cap above one.
    static func references(max: Int?, parked: Bool) -> String {
        caption(max == 1 ? "Reference" : "References", parked: parked)
    }

    static let control = "Control"
    static let identityAdd = "Add photo"

    /// The widest a caption may draw. Not a frame -- a caption takes its own
    /// natural width so a 52pt add well does not sit in a 116pt column -- but
    /// the BUDGET a test holds every caption to, because the wells sit beside
    /// the prompt field and a long one takes the field's room.
    static let width: CGFloat = 116
    /// The gap between a well and its caption.
    static let spacing: CGFloat = 3
    /// One caption line at `.caption2`. Measured against the real font by
    /// `PictureWellTests`, in both directions -- too small clips it, too large
    /// is a second line's worth of empty bar.
    static let lineHeight: CGFloat = 13

    /// What a captioned well occupies, so the bar's height is a value rather
    /// than whatever SwiftUI happened to lay out.
    static func height(under size: CGFloat) -> CGFloat { size + spacing + lineHeight }

    /// The caption itself. One declaration, so every well draws it at the same
    /// size in the same column.
    static func text(_ caption: String) -> some View {
        Text(caption)
            .font(.caption2)
            .foregroundStyle(.secondary)
            .lineLimit(1)
            // Its own natural width, never the parent's: a caption squeezed
            // by a 52pt well would wrap, which is the one thing the height
            // constant above cannot survive.
            .fixedSize()
            .frame(height: lineHeight)
    }

    /// A parked well says so in words. The wording is "not used" rather than
    /// "disabled": the picture is kept and comes straight back when the active
    /// well is emptied.
    private static func caption(_ name: String, parked: Bool) -> String {
        parked ? "\(name) (not used)" : name
    }
}
