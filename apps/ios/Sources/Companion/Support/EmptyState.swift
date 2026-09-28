import SwiftUI

/// A destination with nothing in it yet: what it will hold and how to get
/// there (DESIGN.md §7).
///
/// Not `ContentUnavailableView`: with a custom label and actions it failed the
/// shell audit at every size (clipped title, partial Dynamic Type) and it
/// cannot scroll, so AX5 pushed its button off screen. This one wraps every
/// line and scrolls once the text outgrows the screen, centred until then --
/// and at accessibility sizes it pins the action above the tab bar, because
/// the one thing to do here must never sit four screens down, under the glass.
struct EmptyState<Actions: View>: View {
    @Environment(\.dynamicTypeSize) private var size
    let title: String
    let symbol: String
    let message: String
    @ViewBuilder var actions: Actions

    var body: some View {
        ScrollView {
            VStack(spacing: 12) {
                Image(systemName: symbol)
                    .font(.largeTitle)
                    .imageScale(.large)
                    .foregroundStyle(.secondaryText)
                    .accessibilityHidden(true)
                Text(title)
                    .font(.title2.bold())
                    .accessibilityAddTraits(.isHeader)
                Text(message)
                    .foregroundStyle(.secondaryText)
                if !pinsActions {
                    actions
                        .padding(.top, 8)
                }
            }
            .multilineTextAlignment(.center)
            .padding(.horizontal, 24)
            .padding(.vertical, 32)
            .frame(maxWidth: .infinity)
        }
        .defaultScrollAnchor(.center, for: .alignment)
        .scrollBounceBehavior(.basedOnSize)
        .safeAreaBar(edge: .bottom) {
            if pinsActions {
                actions
                    .frame(maxWidth: .infinity)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .accessibilityElement(children: .contain)
                    .accessibilityIdentifier("bottom-chrome")
            }
        }
    }

    private var pinsActions: Bool { RowAxis.for(size) == .vertical }
}

extension EmptyState where Actions == EmptyView {
    init(title: String, symbol: String, message: String) {
        self.init(title: title, symbol: symbol, message: message) { EmptyView() }
    }
}

extension View {
    /// The one filled button a screen may have. Its fill is `ProminentFill`,
    /// not the accent: in dark mode no single blue gives white text 4.5:1 on
    /// the fill AND tint text 4.5:1 on a grouped row, so the two are split
    /// (DESIGN.md §6; verified by the shell audit).
    func prominentAction() -> some View {
        buttonStyle(.borderedProminent)
            .tint(Color("ProminentFill"))
    }
}

extension ShapeStyle where Self == Color {
    /// Secondary text that passes 4.5:1 in both appearances. The system's
    /// `.secondary` is 60% of the label colour -- about 4.4:1 on white in light
    /// mode -- and failed the shell's contrast audit on every screen.
    static var secondaryText: Color { Color("SecondaryText") }
}
