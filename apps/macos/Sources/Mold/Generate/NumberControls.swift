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
    }
}

/// Clip length in seconds, because nobody thinks in frames.
///
/// The slider moves in frames so it can only ever land on the grid the family
/// accepts, but the readout is the duration that means something to a person.
struct LengthControl: View {
    let temporal: TemporalProfile
    /// What admission will really accept at this rate, and the tier's own
    /// single-clip ceiling -- NOT the advertised `frames.max`, which for
    /// LTX-2 is the figure at 120 fps and for wan is a resource guard
    /// (`ClipLengthBounds`, findings 01#3 and 02#3).
    let bounds: ClipLengthBounds
    @Binding var draft: RenderDraft

    var body: some View {
        SliderControl(
            name: "Length",
            value: Binding(
                get: { Double(draft.frames ?? temporal.frames.default) },
                set: { draft.frames = min(temporal.snap(Int($0.rounded())), bounds.max) }
            ),
            range: Double(bounds.min)...Double(max(bounds.max, bounds.min + 1)),
            step: Double(max(temporal.frames.step, 1))
        ) {
            Text(seconds)
        }
        .accessibilityValue(seconds)
        .help("\(draft.frames ?? temporal.frames.default) frames at \(temporal.fps.value) fps")
    }

    private var seconds: String {
        let duration = temporal.duration(forFrames: draft.frames ?? temporal.frames.default)
        return duration.formatted(.number.precision(.fractionLength(1))) + "s"
    }
}

/// How many pictures one press makes.
struct BatchControl: View {
    let maximum: Int
    @Binding var draft: RenderDraft

    var body: some View {
        Menu {
            ForEach(counts, id: \.self) { count in
                Button(count == 1 ? "1 seed" : "\(count) seeds") { draft.batchSize = count }
            }
        } label: {
            Text(draft.batchSize == 1 ? "1 seed" : "\(draft.batchSize) seeds")
        }
        .menuStyle(.button)
        .buttonStyle(.accessoryBar)
        .fixedSize()
        .help("Render several seeds at once")
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
