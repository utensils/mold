import MoldClient
import MoldStyle
import SwiftUI

/// The mask sheet's controls: brush/erase, size, Invert, Clear, Cancel, Done.
extension MaskEditorSheet {
    var toolbar: some View {
        HStack(spacing: 12) {
            Picker("Tool", selection: $erasing) {
                Text("Brush").tag(false)
                Text("Erase").tag(true)
            }
            .pickerStyle(.segmented)
            .help("Brush marks areas to change; Erase removes marks")
            .labelsHidden()
            .fixedSize()

            Stepper {
                Text("\(Int(brushSize))px")
                    .monospacedDigit()
                    .frame(minWidth: 40, alignment: .leading)
            } onIncrement: {
                stepBrush(1)
            } onDecrement: {
                stepBrush(-1)
            }
            .help("Change the brush size. Use [ to make it smaller or ] to make it larger.")

            Spacer()

            Button("Invert") { strokes.invert() }
                .help("Swap the areas to change with the areas to keep")
            Button("Clear") { strokes.clear() }
                .help("Remove every painted mark from the mask")
                .disabled(strokes.isEmpty)

            Spacer()

            // Sized to their words: at the sheet's width the row once
            // truncated Cancel to "Can…".
            Button("Cancel") { requestCancel() }
                .keyboardShortcut(.cancelAction)
                .fixedSize()
            Button("Done") { finish() }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
                .fixedSize()
        }
    }

    /// Window-scoped equivalents that need no focus: `[`/`]` step the
    /// brush and ⌘Z undoes the last stroke, whichever manager
    /// `MaskUndo.resolve` finds live. A visible control answers the same
    /// action either way, so these exist purely for the chord.
    var hiddenShortcuts: some View {
        Group {
            Button(action: { stepBrush(-1) }) { EmptyView() }
                .keyboardShortcut("[")
            Button(action: { stepBrush(1) }) { EmptyView() }
                .keyboardShortcut("]")
            Button(action: { performUndo() }) { EmptyView() }
                .keyboardShortcut("z")
        }
        .hidden()
    }
}
