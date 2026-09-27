import SwiftUI

/// The surfaces the app draws on. There are three, and adding a fourth should
/// need an argument -- every surface that exists is one more thing to keep
/// consistent across light and dark.
public enum PanelStyle: Sendable {
    /// Floats over content: the prompt panel over the canvas. Material, so the
    /// picture beneath tints it.
    case floating
    /// Sits inside content: an inspector block, a grouped row.
    case inset
}

private struct Panel: ViewModifier {
    let style: PanelStyle

    func body(content: Content) -> some View {
        let panel = content
            .background(fill)
            .clipShape(shape)
            .overlay { shape.strokeBorder(stroke, lineWidth: 1) }

        switch style {
        case .floating:
            panel.shadow(
                color: .black.opacity(Chrome.shadowOpacity),
                radius: Chrome.shadowRadius,
                y: Chrome.shadowY
            )
        case .inset:
            panel
        }
    }

    private var shape: RoundedRectangle {
        RoundedRectangle(
            cornerRadius: style == .floating ? Chrome.panelRadius : Chrome.cardRadius,
            style: .continuous
        )
    }

    private var fill: AnyShapeStyle {
        switch style {
        case .floating: AnyShapeStyle(.regularMaterial)
        case .inset: AnyShapeStyle(.quaternary)
        }
    }

    private var stroke: Color {
        style == .floating ? Chrome.hairline : .clear
    }
}

public extension View {
    func panel(_ style: PanelStyle) -> some View { modifier(Panel(style: style)) }
}
