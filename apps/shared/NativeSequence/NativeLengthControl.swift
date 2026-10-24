import MoldClient
import SwiftUI

/// "Length": one slider on the model's own frame grid, the same on iOS and
/// macOS. Automatic from prompt hides it; Longer video (Duration machines
/// only) switches to the scenes editor, where each scene has its own slider.
///
/// `duration` is nil for an ordinary render, whose slider writes the draft's
/// frame count; with it, the slider writes the Duration request's exact length
/// and a length within one generation is still a one-clip video parent.
struct NativeLengthControl: View {
    let grid: VideoLengthGrid
    let temporal: TemporalProfile
    @Binding var draft: RenderDraft
    var duration: Binding<VideoDurationDraft>?
    let prompt: String
    /// The grid came from the machine (limits or a plan), not a fallback.
    var authoritative = true

    private var extending: Bool { duration?.wrappedValue.source != nil }
    private var title: String { extending ? "Add duration" : "Length" }
    private var automatic: Bool { draft.usesAutomaticDuration && !extending }
    private var longer: Bool { duration?.wrappedValue.isLonger == true }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if draft.supportsDurationPrediction, !extending {
                Toggle("Automatic from prompt", isOn: $draft.prefersAutomaticDuration)
#if os(macOS)
                    .toggleStyle(.checkbox)
#endif
                    .help("Let the model choose the clip length from your prompt.")
                    .accessibilityIdentifier("length-automatic")
            }
            if !automatic {
                if let duration, longer {
                    Text(VideoLengthGrid.total(milliseconds: duration.wrappedValue.scenes.map {
                        (try? VideoDurationDraft.milliseconds($0.seconds)) ?? 0
                    }))
                    .font(.headline)
                    .accessibilityIdentifier("video-length-total")
                } else {
                    slider
                }
                if let duration {
                    Toggle("Longer video", isOn: Binding(
                        get: { duration.wrappedValue.isLonger },
                        set: { duration.wrappedValue.setLonger($0, prompt: prompt, on: grid) }))
#if os(macOS)
                        .toggleStyle(.checkbox)
#endif
                        .help("Write the video as scenes, each up to \(VideoLengthGrid.seconds(tenths: grid.sceneMaxTenths)) seconds with its own prompt.")
                        .accessibilityIdentifier("length-longer")
                }
            }
        }
        .onAppear(perform: settle)
        .onChange(of: grid) { settle() }
        .onChange(of: draft.fps) {
            guard duration == nil else { return }
            draft.fps = temporal.fps.resolvedRate(draft.fps)
        }
    }

    private var frames: Int { grid.snap(draft.frames ?? temporal.frames.default) }

    @ViewBuilder private var slider: some View {
        if let duration {
            if duration.wrappedValue.exceedsOneGeneration(on: grid), let milliseconds = duration.wrappedValue.lengthMilliseconds {
                // Too long to become scenes (`VideoDurationDraft.fit`): shown
                // as it is and still submitted; the server warns it repeats.
                HStack {
                    Text(title)
                    Spacer(minLength: 8)
                    Text("\(VideoLengthGrid.longLabel(milliseconds: milliseconds)) (one prompt, repeats across clips)")
                        .monospacedDigit().foregroundStyle(.secondaryText).multilineTextAlignment(.trailing)
                }
                .accessibilityElement(children: .combine)
                .accessibilityIdentifier("video-length-long")
            } else {
                secondsSlider(duration)
            }
        } else { framesSlider }
    }

    /// A Duration is trimmed to its exact length, so its slider is seconds in
    /// tenths and its readout the requested value as it is ("14.0 s").
    private func secondsSlider(_ duration: Binding<VideoDurationDraft>) -> some View {
        let milliseconds = duration.wrappedValue.lengthMilliseconds ?? UInt64(grid.singleMinTenths) * 100
        let label = VideoLengthGrid.label(milliseconds: milliseconds)
        let value = Binding<Double>(
            get: { min(max(Double(milliseconds) / 100, Double(grid.singleMinTenths)), Double(grid.singleMaxTenths)) },
            set: { duration.wrappedValue.setLength(tenths: Int($0.rounded()), on: grid) })
        return VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                Spacer(minLength: 8)
                Text(label).monospacedDigit().foregroundStyle(.secondaryText)
            }
            Slider(value: value, in: Double(grid.singleMinTenths)...Double(max(grid.singleMaxTenths, grid.singleMinTenths + 1)),
                   step: 1) { Text(title) }
                .labelsHidden()
                .accessibilityLabel(title)
                .accessibilityValue(label)
                .accessibilityIdentifier("video-length")
                .help("Choose the video's length, up to one generation of \(VideoLengthGrid.seconds(tenths: grid.singleMaxTenths)) seconds.")
        }
    }

    /// An ordinary render's frame count, snapped to the model's own grid.
    private var framesSlider: some View {
        let values = grid.frames
        let index = Binding<Double>(
            get: { Double(values.firstIndex(of: frames) ?? 0) },
            set: { value in
                draft.prefersAutomaticDuration = false
                draft.frames = values[min(max(Int(value.rounded()), 0), values.count - 1)]
            })
        return VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                Spacer(minLength: 8)
                Text(grid.label(frames: frames)).monospacedDigit().foregroundStyle(.secondaryText)
            }
            Slider(value: index, in: 0...Double(max(values.count - 1, 1)), step: 1) { Text(title) }
                .labelsHidden()
                .accessibilityLabel(title)
                .disabled(values.count < 2)
                .accessibilityValue(grid.label(frames: frames))
                .accessibilityIdentifier("video-length")
                .help("Choose the length. It uses \(frames) frames at \(grid.fps) frames per second.")
        }
    }

    /// Keep what is sent equal to what is shown: an ordinary frame count is
    /// re-snapped onto the grid, and a saved Duration the sliders cannot show
    /// becomes equal scenes of the same length (`VideoDurationDraft.fit`).
    private func settle() {
        if let duration {
            guard authoritative else { return }
            var value = duration.wrappedValue
            if value.fit(to: grid, prompt: prompt) { duration.wrappedValue = value }
        } else if let current = draft.frames, grid.snap(current) != current {
            draft.frames = grid.snap(current)
        }
    }
}

/// One scene of a Longer video: its prompt and its own Length slider, capped at
/// the longest scene that always stays one clip.
struct NativeDurationScene: View {
    @FocusState private var editing: Bool
    @Binding var draft: VideoDurationDraft
    let id: VideoDurationDraft.Scene.ID
    let grid: VideoLengthGrid
    let number: Int
    var body: some View {
        if let index = draft.scenes.firstIndex(where: { $0.id == id }) {
            let tenths = draft.sceneTenths(id, on: grid)
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Scene \(number)").font(.headline)
                    Spacer(minLength: 8)
                    Button("Remove scene \(number)", systemImage: "trash", role: .destructive) {
                        draft.scenes.removeAll { $0.id == id }
                        draft.updateDurationFromScenes()
                    }
                    .labelStyle(.iconOnly).disabled(draft.scenes.count < 2)
                    .frame(minWidth: 44, minHeight: 44)
                }
                TextField("Describe this scene…", text: $draft.scenes[index].prompt, axis: .vertical)
                    .lineLimit(2 ... 5).accessibilityLabel("Scene \(number) prompt")
                    .focused($editing)
#if os(macOS)
                    .focusedValue(\.editingText, editing ? true : nil)
#endif
                HStack {
                    Text("Length")
                    Spacer(minLength: 8)
                    Text(sceneLabel(index)).monospacedDigit().foregroundStyle(.secondaryText)
                }
                Slider(value: Binding(get: { Double(tenths) }, set: { draft.setScene(id, tenths: Int($0.rounded()), on: grid) }),
                       in: Double(grid.sceneMinTenths)...Double(max(grid.sceneMaxTenths, grid.sceneMinTenths + 1)), step: 1) {
                    Text("Scene \(number) length")
                }
                .labelsHidden()
                .accessibilityLabel("Scene \(number) length")
                .accessibilityValue(sceneLabel(index))
                .accessibilityIdentifier("scene-length-\(number)")
            }
            .padding(.vertical, 8)
        }
    }

    /// The scene's requested length as it is, one decimal.
    private func sceneLabel(_ index: Int) -> String {
        VideoLengthGrid.label(milliseconds: (try? VideoDurationDraft.milliseconds(draft.scenes[index].seconds))
            ?? UInt64(draft.sceneTenths(id, on: grid)) * 100)
    }
}

#if os(macOS)
// iOS uses its audited asset-backed token; macOS keeps system secondary text.
extension ShapeStyle where Self == Color {
    static var secondaryText: Color { .secondary }
}
#endif
