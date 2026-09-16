import MoldStyle
import SwiftUI

/// Slides the prompt capsule off the bottom edge, leaving a lip behind.
///
/// An offset rather than an `if`: removing the panel would tear down the
/// prompt field with it, taking the caret, the undo stack and any in-progress
/// input-method composition. Moving the same view off the edge keeps all of it.
///
/// The lip is not decoration. A render started before the capsule tucked is
/// still running, and its step marks are the one thing worth keeping on screen
/// while you look at the picture.
struct PromptTuck<Panel: View>: View {
    @Binding var tucked: Bool
    let steps: (done: Int, total: Int)?
    @ViewBuilder let panel: Panel

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var height: CGFloat = 0

    var body: some View {
        ZStack(alignment: .bottom) {
            panel
                .padding(20)
                .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { height = $0 }
                // Measured height includes the padding, so the capsule clears
                // the window edge completely rather than leaving a sliver.
                .offset(y: tucked ? height - PromptLip.height : 0)
                .allowsHitTesting(!tucked)
                .accessibilityHidden(tucked)
            PromptLip(steps: steps) { tucked = false }
                .opacity(tucked ? 1 : 0)
                // Invisible while the capsule is up, so it never takes a click
                // meant for the capsule sitting on top of it.
                .allowsHitTesting(tucked)
        }
        .animation(reduceMotion ? nil : .snappy, value: tucked)
    }
}

/// The sliver left at the bottom edge once the capsule has tucked away.
///
/// Drawn as the capsule's own top edge -- same material, same radius, same
/// hairline -- so it reads as the capsule having slid down rather than as a
/// second, unrelated control.
struct PromptLip: View {
    static let height: CGFloat = 22

    let steps: (done: Int, total: Int)?
    let show: () -> Void

    var body: some View {
        Button(action: show) {
            VStack(spacing: 0) {
                if let steps {
                    StepSegments(done: steps.done, total: steps.total)
                        // Inset by the radius, as on the capsule, so the lip's
                        // corners clip nothing.
                        .padding(.horizontal, Chrome.panelRadius)
                }
                Spacer(minLength: 0)
                Image(systemName: "chevron.up")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Spacer(minLength: 0)
            }
        }
        .buttonStyle(.plain)
        // Two frames, not one: the height is fixed because `PromptTuck`'s
        // offset is computed against it. A flexible `maxHeight` would let the
        // lip shrink and leave a gap under the tucked capsule.
        .frame(maxWidth: PromptPanel.maxWidth)
        .frame(height: Self.height)
        .background(.regularMaterial)
        .clipShape(shape)
        .overlay { shape.strokeBorder(.separator, lineWidth: 1) }
        .help("Show the prompt")
        .accessibilityLabel("Show the prompt")
    }

    private var shape: UnevenRoundedRectangle {
        UnevenRoundedRectangle(
            topLeadingRadius: Chrome.panelRadius,
            topTrailingRadius: Chrome.panelRadius,
            style: .continuous
        )
    }
}
