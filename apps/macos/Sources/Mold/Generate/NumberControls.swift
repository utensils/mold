import MoldClient
import SwiftUI

/// A slider with its value beside it, on one baseline.
///
/// `name` is not decoration: the caption that names a control is a SIBLING
/// `Text` in `ControlLabel`, which VoiceOver does not read as this slider's
/// label -- so every one of them announced itself as "50 percent, slider"
/// (finding 02#14). The label goes on the control that has the value.
struct SliderControl<Label: View>: View {
    let name: String
    let value: Binding<Double>
    let range: ClosedRange<Double>
    let step: Double
    @ViewBuilder let label: Label

    var body: some View {
        HStack(spacing: 6) {
            Slider(value: value, in: range, step: step)
                .controlSize(.small)
                .frame(minWidth: 80, maxWidth: 130)
                .accessibilityLabel(name)
            label.monospacedDigit().frame(minWidth: 30, alignment: .trailing)
                .accessibilityHidden(true)
        }
    }
}

struct StepsControl: View {
    let control: IntegerControl
    @Binding var draft: RenderDraft

    var body: some View {
        SliderControl(
            name: "Steps",
            value: Binding(get: { Double(draft.steps) },
                           set: { draft.steps = Int($0.rounded()) }),
            range: Double(control.min)...Double(control.max),
            step: Double(max(control.step, 1))
        ) {
            Text(draft.steps.formatted())
        }
        .help("Choose how many refinement steps to run. More steps take longer and may change the result.")
    }
}

struct GuidanceControl: View {
    let control: FloatControl
    @Binding var draft: RenderDraft

    var body: some View {
        SliderControl(name: "Guidance", value: $draft.guidance,
                      range: control.min...control.max,
                      step: control.step) {
            Text(draft.guidance, format: .number.precision(.fractionLength(1)))
        }
        .help("Choose how strongly the result follows your prompt. Higher is not always better.")
    }
}

/// How many pictures one press makes.
struct BatchControl: View {
    let maximum: Int
    @Binding var draft: RenderDraft

    var body: some View {
        Menu {
            ForEach(counts, id: \.self) { count in
                Button(count == 1 ? "1 result" : "\(count) results") { draft.batchSize = count }
            }
        } label: {
            Text(draft.batchSize == 1 ? "1 result" : "\(draft.batchSize) results")
        }
        .menuStyle(.button)
        .buttonStyle(.accessoryBar)
        .fixedSize()
        .help("Choose how many different results to make with one click")
    }

    /// Powers of two up to what the host will admit in one batch.
    private var counts: [Int] {
        var counts = [1]
        while let last = counts.last, last * 2 <= min(maximum, 16) {
            counts.append(last * 2)
        }
        return counts
    }
}
