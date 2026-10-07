import AppKit
import SwiftUI

/// A bottom-anchored editor grows when its top handle is dragged upward.
struct PromptResizeHandle: View {
    @Binding var preferredHeight: Double
    let available: CGFloat
    @State private var dragStart: CGFloat?

    private var height: CGFloat { PromptEditorHeight.resolve(preferredHeight, available: available) }

    var body: some View {
        HStack {
            Spacer()
            Capsule().fill(.secondary.opacity(0.5)).frame(width: 38, height: 4)
            Spacer()
        }
        .frame(height: 14)
        .contentShape(Rectangle())
        .onHover { NSCursor.resizeUpDown.set(); if !$0 { NSCursor.arrow.set() } }
        .gesture(DragGesture(minimumDistance: 1, coordinateSpace: .global)
            .onChanged { value in
                if dragStart == nil { dragStart = height }
                preferredHeight = PromptEditorHeight.dragged(
                    from: dragStart ?? height, translation: value.translation.height, available: available)
            }
            .onEnded { _ in dragStart = nil })
        .help("Drag upward to make the prompt taller. Control-click for size options.")
        .accessibilityElement()
        .accessibilityLabel("Prompt editor height")
        .accessibilityValue("\(Int(height)) points")
        .accessibilityAdjustableAction { direction in
            adjust(direction == .increment ? 40 : -40)
        }
        .contextMenu {
            Button("Make Prompt Taller") { adjust(40) }
            Button("Make Prompt Shorter") { adjust(-40) }
            Button("Reset Prompt Height") { preferredHeight = PromptEditorHeight.initial }
        }
    }

    private func adjust(_ delta: CGFloat) {
        preferredHeight = PromptEditorHeight.resolve(height + delta, available: available)
    }
}
