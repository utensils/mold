import SwiftUI

/// A Generate-only field: compact beside its label when both fit, otherwise
/// stacked. The control always receives the remaining width, never a fixed cap.
struct InspectorField<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    init(_ title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        InspectorFieldLayout {
            Text(title).font(.caption).foregroundStyle(.secondary)
            content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct InspectorFieldLayout: Layout {
    static let minimumControlWidth: CGFloat = 160
    static let horizontalGap: CGFloat = 12
    static let verticalGap: CGFloat = 5

    private func dimensions(width: CGFloat, subviews: Subviews) -> (label: CGSize, control: CGSize, horizontal: Bool) {
        let labelWidth = min(110, subviews[0].sizeThatFits(.unspecified).width)
        let horizontal = width >= labelWidth + Self.horizontalGap + Self.minimumControlWidth
        let label = subviews[0].sizeThatFits(.init(width: horizontal ? labelWidth : width, height: nil))
        let control = subviews[1].sizeThatFits(.init(
            width: horizontal ? width - labelWidth - Self.horizontalGap : width, height: nil))
        return (label, control, horizontal)
    }

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        guard subviews.count == 2 else { return .zero }
        let width = max(0, proposal.width ?? 280)
        let sizes = dimensions(width: width, subviews: subviews)
        return CGSize(width: width, height: sizes.horizontal
            ? max(sizes.label.height, sizes.control.height)
            : sizes.label.height + Self.verticalGap + sizes.control.height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        guard subviews.count == 2 else { return }
        let sizes = dimensions(width: bounds.width, subviews: subviews)
        let labelWidth = min(110, subviews[0].sizeThatFits(.unspecified).width)
        let controlX = sizes.horizontal ? labelWidth + Self.horizontalGap : 0
        subviews[0].place(at: CGPoint(x: bounds.minX, y: bounds.minY + (sizes.horizontal ? (bounds.height - sizes.label.height) / 2 : 0)),
            anchor: .topLeading, proposal: .init(width: sizes.horizontal ? labelWidth : bounds.width, height: sizes.label.height))
        subviews[1].place(at: CGPoint(x: bounds.minX + controlX,
            y: bounds.minY + (sizes.horizontal ? (bounds.height - sizes.control.height) / 2 : sizes.label.height + Self.verticalGap)),
            anchor: .topLeading, proposal: .init(width: bounds.width - controlX, height: sizes.control.height))
    }
}
