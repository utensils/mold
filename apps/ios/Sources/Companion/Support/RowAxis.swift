import SwiftUI

/// How a label/value row lays out at a given text size (DESIGN.md §6 rule 3).
///
/// Every row in the app that puts a label beside a value -- a machine card's
/// "Video memory · 14.9 / 24 GB", an Info sheet's "Model · flux-dev:q4", the
/// composer's estimate beside Generate -- asks this ONE question, so the
/// point at which the app stops fitting things side by side is a single
/// decision, tested in `RowAxisTests`, rather than a threshold per screen.
enum RowAxis: Equatable {
    case horizontal
    case vertical

    /// Side by side through xxxLarge, stacked from AX1 up. The seven standard
    /// sizes all fit a label and a short value on a phone's width; the five
    /// accessibility sizes run 28-53 pt body text, where "Video memory" alone
    /// can take the whole row. One threshold, never a per-screen guess.
    static func `for`(_ size: DynamicTypeSize) -> RowAxis {
        size.isAccessibilitySize ? .vertical : .horizontal
    }
}

/// A label and its value, side by side until the text is too large for that,
/// then stacked -- never truncated.
struct AdaptiveRow<Label: View, Value: View>: View {
    @Environment(\.dynamicTypeSize) private var size
    @ViewBuilder var label: Label
    @ViewBuilder var value: Value

    var body: some View {
        let horizontal = RowAxis.for(size) == .horizontal
        let layout = horizontal
            ? AnyLayout(HStackLayout(alignment: .firstTextBaseline, spacing: 8))
            : AnyLayout(VStackLayout(alignment: .leading, spacing: 4))
        layout {
            label
            if horizontal { Spacer(minLength: 8) }
            value.foregroundStyle(.secondaryText)
        }
    }
}
