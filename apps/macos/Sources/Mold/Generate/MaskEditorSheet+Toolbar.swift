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
            .help("Brush size (also \u{2018}[\u{2019} and \u{2018}]\u{2019})")

            Spacer()

            Button("Invert") { strokes.invert() }
            Button("Clear") { strokes.clear() }
                .disabled(strokes.isEmpty)

            Spacer()

            Button("Cancel") { requestCancel() }
                .keyboardShortcut(.cancelAction)
            Button("Done") { finish() }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
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
