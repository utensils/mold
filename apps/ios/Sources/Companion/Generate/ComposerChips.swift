import MoldClient
import SwiftUI

/// Shape: aspect and size from the recipe's own ladder (`CanvasShape`, shared
/// with the Mac), in one menu.
struct ShapeChip: View {
    @Environment(GenerateController.self) private var generate
    let resolution: ResolutionProfile
    let short: Bool

    var body: some View {
        switch CanvasShape.resolve(resolution, width: generate.draft.width, height: generate.draft.height) {
        case let .menus(shape):
            Menu {
                ForEach(shape.aspects) { group in
                    Menu {
                        ForEach(group.presets) { preset in
                            Button { set(preset) } label: {
                                if preset.width == generate.draft.width, preset.height == generate.draft.height {
                                    Label(preset.label, systemImage: "checkmark")
                                } else {
                                    Text(preset.label)
                                }
                            }
                        }
                    } label: {
                        Label {
                            Text(group.id == shape.aspect
                                ? String(localized: "\(group.label) · Selected") : group.label)
                        } icon: {
                            if group.id == shape.aspect {
                                Image(systemName: "checkmark")
                            } else if let preset = group.presets.first {
                                Image(uiImage: AspectRatioIcon.image(width: preset.width, height: preset.height))
                            }
                        }
                    }
                    .accessibilityIdentifier("aspect-\(group.id)")
                }
            } label: {
                chipLabel(shape.aspect, detail: short ? nil : "\(generate.draft.width)×\(generate.draft.height)")
            }
            .buttonStyle(.bordered)
            .accessibilityLabel(String(localized: "Shape, \(shape.aspect), \(generate.draft.width) by \(generate.draft.height)"))
        case let .fixed(size):
            chipLabel(size, detail: nil).accessibilityLabel(String(localized: "Size, \(size)"))
        case .fromSource:
            chipLabel(String(localized: "From the source"), detail: nil)
        case .hidden:
            EmptyView()
        }
    }

    private func set(_ preset: SizePreset) {
        generate.draft.width = preset.width
        generate.draft.height = preset.height
        generate.draft.canvasIntent = .manual
    }
}

/// A chip's face: a plain word, then a mono figure when there is room.
func chipLabel(_ title: String, detail: String?) -> some View {
    ChipLabel(title: title, detail: detail)
}

private struct ChipLabel: View {
    @Environment(\.dynamicTypeSize) private var size
    let title: String
    let detail: String?

    var body: some View {
        let layout = RowAxis.for(size) == .vertical
            ? AnyLayout(VStackLayout(alignment: .leading, spacing: 4))
            : AnyLayout(HStackLayout(spacing: 4))
        layout {
            Text(title).fixedSize(horizontal: false, vertical: true)
            if let detail {
                Text(verbatim: detail).font(.caption.monospacedDigit()).foregroundStyle(.secondaryText)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(minHeight: 44)
    }
}

/// Steps or Batch: a menu of values across the recipe's range, reading as
/// "Steps 28". The Mac's stepper, in a form a thumb can hit.
struct StepperChip: View {
    let title: String
    let value: Int
    let range: ClosedRange<Int>
    let set: (Int) -> Void

    var body: some View {
        Menu {
            Stepper(value: Binding(get: { value }, set: set), in: range) {
                Text("\(title): \(value)")
            }
            ForEach(Self.stops(in: range), id: \.self) { stop in
                Button { set(stop) } label: {
                    if stop == value { Label("\(stop)", systemImage: "checkmark") } else { Text("\(stop)") }
                }
            }
        } label: {
            chipLabel(title, detail: "\(value)")
        }
        .buttonStyle(.bordered)
        .accessibilityLabel("\(title), \(value)")
        .accessibilityAdjustableAction { direction in
            switch direction {
            case .increment: set(min(range.upperBound, value + 1))
            case .decrement: set(max(range.lowerBound, value - 1))
            @unknown default: break
            }
        }
        .sensoryFeedback(.selection, trigger: value)
    }

    /// A handful of useful values across the range, never hundreds.
    static func stops(in range: ClosedRange<Int>) -> [Int] {
        let span = range.upperBound - range.lowerBound
        guard span > 8 else { return Array(range) }
        let step = max(1, span / 6)
        return Array(Set(stride(from: range.lowerBound, through: range.upperBound, by: step)).union([range.upperBound])).sorted()
    }
}

/// A clip's length in seconds, snapped to the recipe's frame grid; the mono
/// figure says frames and rate.
struct LengthChip: View {
    @Environment(GenerateController.self) private var generate
    let temporal: TemporalProfile

    var body: some View {
        @Bindable var generate = generate
        let selection = temporal.lengthSelection(fps: generate.draft.fps, frames: generate.draft.frames,
            family: generate.model?.family, model: generate.modelName, sourceImage: generate.recipe?.capabilities.sourceImage)
        let fps = selection.fps
        let frames = selection.frames
        let seconds = Double(frames) / Double(fps)
        Menu {
            if generate.draft.supportsDurationPrediction {
                Toggle("Automatic from prompt", isOn: $generate.draft.prefersAutomaticDuration)
                Divider()
            }
            ForEach(selection.options, id: \.frames) { option in
                Button { generate.draft.prefersAutomaticDuration = false; generate.draft.frames = option.frames } label: {
                    let text = String(localized: "\(durationText(option.seconds)) s")
                    if !generate.draft.usesAutomaticDuration && option.frames == frames { Label(text, systemImage: "checkmark") } else { Text(text) }
                }
            }
        } label: {
            chipLabel(generate.draft.usesAutomaticDuration ? "Auto length" : String(localized: "\(durationText(seconds)) s"),
                      detail: generate.draft.usesAutomaticDuration ? "From prompt" : "\(frames) f · \(Int(fps)) fps")
        }
        .buttonStyle(.bordered)
        .accessibilityLabel(generate.draft.usesAutomaticDuration ? "Length, automatic from prompt" : String(localized: "Length, \(durationText(seconds)) seconds"))
    }


    private func durationText(_ seconds: Double) -> String {
        seconds.formatted(.number.precision(.fractionLength(0...(seconds < 0.1 ? 3 : 1))))
    }
}
