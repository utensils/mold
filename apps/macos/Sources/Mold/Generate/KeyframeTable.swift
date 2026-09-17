import AppKit
import MoldClient
import MoldStyle
import SwiftUI
import UniformTypeIdentifiers

/// One row per keyframe: a frame number and a `MediaWell`.
///
/// Rows stay sorted by frame (`DraftMedia.addingKeyframe`). A frame at or
/// past the clip's own length is never reachable in the first place --
/// `snappedFrame` clamps it the same way `TemporalProfile.snap` clamps a
/// frame COUNT, rather than accepting an out-of-range value and failing
/// later at the wire (`validation.rs:1705-1711`).
struct KeyframeTable: View {
    let temporal: TemporalProfile?
    @Binding var draft: RenderDraft

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Array(draft.media.keyframes.enumerated()), id: \.offset) { index, keyframe in
                row(index: index, keyframe: keyframe)
            }
            addButton
        }
    }

    @ViewBuilder private func row(index: Int, keyframe: KeyframeCondition) -> some View {
        if let temporal {
            HStack(spacing: 6) {
                Stepper(value: frameBinding(index: index, temporal: temporal),
                        in: 0...ceiling, step: Swift.max(temporal.frames.step, 1)) {
                    Text(keyframe.frame.formatted())
                        .monospacedDigit()
                        .frame(minWidth: 30, alignment: .trailing)
                }
                MediaWell(
                    systemImage: "photo", allowedTypes: [.png, .jpeg],
                    placeholder: "No picture", attachment: attachmentBinding(index: index)
                )
                Button {
                    draft.media.keyframes.remove(at: index)
                } label: {
                    Image(systemName: "minus.circle")
                }
                .buttonStyle(.plain)
                .help("Remove this keyframe")
            }
        }
    }

    @ViewBuilder private var addButton: some View {
        if let temporal {
            Button {
                addKeyframe(temporal: temporal)
            } label: {
                Label("Add keyframe", systemImage: "plus")
            }
            .buttonStyle(.plain)
            .font(.caption)
        }
    }

    private func addKeyframe(temporal: TemporalProfile) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url, let data = try? Data(contentsOf: url) else { return }
        let frame = nextFrame(temporal: temporal)
        let keyframe = KeyframeCondition(frame: frame, image: data.base64EncodedString(), name: url.lastPathComponent)
        draft.media.addingKeyframe(keyframe)
    }

    private func nextFrame(temporal: TemporalProfile) -> Int {
        guard let last = draft.media.keyframes.map(\.frame).max() else { return 0 }
        return Self.snappedFrame(last + Swift.max(temporal.frames.step, 1), temporal: temporal, frames: clipFrames)
    }

    private func frameBinding(index: Int, temporal: TemporalProfile) -> Binding<Int> {
        Binding(
            get: { draft.media.keyframes.indices.contains(index) ? draft.media.keyframes[index].frame : 0 },
            set: { newValue in
                guard draft.media.keyframes.indices.contains(index) else { return }
                draft.media.keyframes[index].frame = Self.snappedFrame(newValue, temporal: temporal, frames: clipFrames)
                draft.media.keyframes.sort { $0.frame < $1.frame }
            }
        )
    }

    private func attachmentBinding(index: Int) -> Binding<MediaWell.Attachment?> {
        Binding(
            get: {
                guard draft.media.keyframes.indices.contains(index) else { return nil }
                let keyframe = draft.media.keyframes[index]
                return MediaWell.Attachment(base64: keyframe.image, name: keyframe.name ?? "")
            },
            set: { newValue in
                guard draft.media.keyframes.indices.contains(index), let newValue else { return }
                draft.media.keyframes[index].image = newValue.base64
                draft.media.keyframes[index].name = newValue.name
            }
        )
    }

    private var ceiling: Int { Swift.max(clipFrames - 1, 0) }
    private var clipFrames: Int { draft.frames ?? temporal?.frames.default ?? 1 }
}

extension KeyframeTable {
    /// A requested frame, clamped strictly below `frames` -- never a value
    /// the server would refuse (`validation.rs:1705-1711`).
    static func snappedFrame(_ requested: Int, temporal: TemporalProfile, frames: Int) -> Int {
        let ceiling = Swift.max(frames - 1, 0)
        return Swift.min(Swift.max(temporal.snap(requested), 0), ceiling)
    }
}
